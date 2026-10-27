"""StructChain: a structured graph-text model of evidence chains.

For the K graph-proposed candidates of a question, every subset S (2 <= |S| <= 4) is a hypothesis chain with energy
    F(S) = sum_{i in S} U_i  +  max_{T in Trees(S)} sum_{(i,j) in T} P_ij  +  b_{|S|}(q)
  U_i  = text unary  CE(q, p_i)                     + graph unary  g_u(h_i, z_i)
  P_ij = text pair   CE(q, mark(p_i) </s> mark(p_j)) + graph pair   g_p(h_i, h_j, kg_ij)
where mark() wraps the KG-aligned cross-passage entity mentions (shared / synonymous entities) in <e>..</e>, h are the
graph proposer's node states and kg_ij the KG structure between the two passages. The chain distribution
p(S|q) = softmax_S F(S) is normalised exactly by enumerating all 781 subsets (K=12) and the model is trained end to end
by maximum likelihood of the gold chain. Inference returns the exact MAP chain and exact marginals P(i in chain).
"""
import argparse, itertools, json, math, os, random, re, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "BAAI/bge-reranker-v2-m3"
NPF = 6  # kg pair features


def subsets_and_trees(K, smin=2, smax=4):
    subs, trees, tree_sub = [], [], []
    for k in range(smin, smax + 1):
        for S in itertools.combinations(range(K), k):
            sid = len(subs); subs.append(S)
            edges = list(itertools.combinations(S, 2))
            for T in itertools.combinations(edges, k - 1):          # spanning trees of S (Kirchhoff: k^(k-2))
                nodes = set(S); adj = {v: set() for v in S}
                for a, b in T:
                    adj[a].add(b); adj[b].add(a)
                seen, st = {S[0]}, [S[0]]
                while st:
                    v = st.pop()
                    for w in adj[v]:
                        if w not in seen:
                            seen.add(w); st.append(w)
                if seen == nodes:
                    trees.append(T); tree_sub.append(sid)
    return subs, trees, tree_sub


def mark(text, names, n=700):
    t = text[:n]
    for nm in names:
        t = re.sub(r"(?i)\b(" + re.escape(nm) + r")\b", r"<e>\1</e>", t, count=2)
    return t


class StructChain(nn.Module):
    def __init__(self, K=12, use_graph=True, use_mark=True, use_pair=True, init=BASE, hdim=256):
        super().__init__()
        self.cfg = dict(K=K, use_graph=use_graph, use_mark=use_mark, use_pair=use_pair)
        self.tok = AutoTokenizer.from_pretrained(init)
        self.tok.add_special_tokens({"additional_special_tokens": ["<e>", "</e>"]})
        self.ce = AutoModelForSequenceClassification.from_pretrained(init, num_labels=1)
        self.ce.resize_token_embeddings(len(self.tok))
        self.gu = nn.Sequential(nn.Linear(hdim + 1, 256), nn.GELU(), nn.Linear(256, 1))
        self.gp = nn.Sequential(nn.Linear(2 * hdim + NPF, 256), nn.GELU(), nn.Linear(256, 1))
        self.len_prior = nn.Sequential(nn.Linear(hdim + 1, 64), nn.GELU(), nn.Linear(64, 3))
        subs, trees, tree_sub = subsets_and_trees(K)
        self.subs = subs
        E = torch.full((len(trees), 3, 2), -1, dtype=torch.long)
        for t, T in enumerate(trees):
            for e, (a, b) in enumerate(T):
                E[t, e] = torch.tensor([a, b])
        self.register_buffer("tree_edges", E, persistent=False)
        self.register_buffer("tree_sub", torch.tensor(tree_sub), persistent=False)
        M = torch.zeros(len(subs), K)
        for s, S in enumerate(subs):
            M[s, list(S)] = 1
        self.register_buffer("memb", M, persistent=False)
        self.register_buffer("size_idx", torch.tensor([len(S) - 2 for S in subs]), persistent=False)

    def ce_scores(self, q, docs, maxlen, bs=40):
        out = []
        for i in range(0, len(docs), bs):
            enc = self.tok([q] * len(docs[i:i + bs]), docs[i:i + bs], truncation="only_second", max_length=maxlen,
                           padding=True, return_tensors="pt").to(self.memb.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(self.ce(**enc).logits.squeeze(-1).float())
        return torch.cat(out)

    def energies(self, row):
        K = len(row["cands"]); dev = self.memb.device
        assert K == self.cfg["K"]
        c = row["cands"]; q = row["question"]
        U = self.ce_scores(q, [f"{x['title']}: {x['text'][:900]}" for x in c], 256)
        pairs = list(itertools.combinations(range(K), 2))
        P = torch.zeros(K, K, device=dev)
        if self.cfg["use_pair"]:
            docs = []
            for i, j in pairs:
                nm = row["shared"].get(f"{i},{j}", []) if self.cfg["use_mark"] else []
                docs.append(f"{c[i]['title']}: {mark(c[i]['text'], nm)} </s> {c[j]['title']}: {mark(c[j]['text'], nm)}")
            pv = self.ce_scores(q, docs, 384)
            ii = torch.tensor([p[0] for p in pairs], device=dev); jj = torch.tensor([p[1] for p in pairs], device=dev)
            P = P.index_put((ii, jj), pv).index_put((jj, ii), pv)
        hq = torch.zeros(257, device=dev)
        if self.cfg["use_graph"]:
            H = torch.tensor([x["h"] for x in c], device=dev)
            Z = torch.tensor([[x["z"]] for x in c], device=dev)
            U = U + self.gu(torch.cat([H, Z], -1)).squeeze(-1)
            pf = torch.tensor([row["pairfeat"][f"{i},{j}"] for i, j in pairs], device=dev).float()
            pf = torch.log1p(pf)
            ii = torch.tensor([p[0] for p in pairs], device=dev); jj = torch.tensor([p[1] for p in pairs], device=dev)
            gpv = self.gp(torch.cat([H[ii] * H[jj], (H[ii] - H[jj]).abs(), pf], -1)).squeeze(-1)
            G = torch.zeros(K, K, device=dev).index_put((ii, jj), gpv).index_put((jj, ii), gpv)
            P = P + G
            hq = torch.cat([H.mean(0), Z.mean(0)])
        # subset energies: unary sum + best spanning tree of pair potentials + query-conditioned size prior
        E = self.tree_edges
        valid = E[..., 0] >= 0
        ev = P[E[..., 0].clamp(min=0), E[..., 1].clamp(min=0)] * valid
        tree_score = ev.sum(-1)
        best = torch.full((len(self.subs),), -1e9, device=dev).scatter_reduce(0, self.tree_sub, tree_score, "amax")
        lp = self.len_prior(hq)
        Fs = self.memb @ U + best + lp[self.size_idx]
        return Fs, U, P

    def loss(self, row):
        Fs, _, _ = self.energies(row)
        gold = tuple(sorted(row["gold"]))
        if gold not in self._sub_index():
            return None
        return -F.log_softmax(Fs, 0)[self._sub_index()[gold]]

    def _sub_index(self):
        if not hasattr(self, "_si"):
            self._si = {S: k for k, S in enumerate(self.subs)}
        return self._si

    @torch.no_grad()
    def infer(self, row):
        Fs, U, P = self.energies(row)
        p = torch.softmax(Fs, 0)
        marg = (p.unsqueeze(1) * self.memb).sum(0)
        top = torch.topk(p, min(50, len(p)))
        return {"map": list(self.subs[int(Fs.argmax())]), "marg": marg.tolist(),
                "top_sets": [[list(self.subs[int(k)]), float(v)] for v, k in zip(top.values, top.indices)],
                "unary": U.tolist()}


def train(a):
    random.seed(a.seed); torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(f"{ROOT}/ckpt/{a.data}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if 2 <= len(r["gold"]) <= 4 and len(r["cands"]) == 12]
    if a.limit:
        rows = rows[:a.limit]
    m = StructChain(use_graph=not a.no_graph, use_mark=not a.no_mark, use_pair=not a.no_pair).cuda()
    m.ce.gradient_checkpointing_enable()
    head = [p for n, p in m.named_parameters() if not n.startswith("ce.")]
    opt = torch.optim.AdamW([{"params": m.ce.parameters(), "lr": a.lr}, {"params": head, "lr": 1e-3}], weight_decay=0.01)
    acc = 4; steps = a.epochs * len(rows) // acc + 5
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, 1e-3], total_steps=steps, pct_start=0.1)
    print("rows", len(rows), "subsets", len(m.subs), flush=True)
    m.train(); k = 0; t0 = time.time(); run = 0.0
    for ep in range(a.epochs):
        random.shuffle(rows)
        for r in rows:
            l = m.loss(r)
            if l is None:
                continue
            (l / acc).backward(); k += 1; run = 0.98 * run + 0.02 * float(l)
            if k % acc == 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(); sched.step()
            if k % 200 == 0:
                print(ep, k, round(run, 4), f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    m.ce.save_pretrained(a.out); m.tok.save_pretrained(a.out)
    torch.save({"heads": {n: p for n, p in m.state_dict().items() if not n.startswith("ce.")}, "cfg": m.cfg},
               f"{a.out}/heads.pt")
    print("saved", flush=True)


def load(path):
    st = torch.load(f"{path}/heads.pt", weights_only=False)
    m = StructChain(init=path, **{k: v for k, v in st["cfg"].items() if k != "K"})
    m.load_state_dict(st["heads"], strict=False)
    return m.cuda().eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--data", default="chain_train_v6")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--no_graph", action="store_true"); ap.add_argument("--no_mark", action="store_true")
    ap.add_argument("--no_pair", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    train(ap.parse_args())
