"""ChainWalk: evidence retrieval as a graph-supported autoregressive process.
One cross-encoder T defines p(c_t | q, c_<t) for every hop; the grounded evidence graph defines the *support*
(action space) of that distribution: A_t = proposer candidates U graph neighbours of the evidence chosen so far.
Reachability therefore grows with the chain instead of being fixed by a first-stage top-K, and the same network
that verifies a link is the one that proposes where to go next. Training = order-marginal chain likelihood with
negatives drawn from the exact action space the walk sees at test time. Inference = beam search; passages are
packed by their posterior marginal under the beam.

python chainwalk.py build                         -> ckpt/walk_train.jsonl
python chainwalk.py train <init_dir> <out> <seed>
python chainwalk.py infer <ds> <model_dir> <name> [variant]   variant: full | nograph | depth1"""
import json, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unicor import Corpus
from index import embed_queries
import budget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = int(os.environ.get("NB", 16))       # graph neighbours admitted per expansion
BEAM, DEPTH, MAXLEN = 4, 3, 640


def ptxt(C, p, n):
    return f"{C.titles[p]}: {C.texts[p][:n]}"


def doc(C, prefix, c):
    if not prefix:
        return ptxt(C, c, 900)
    return " [next] ".join(ptxt(C, p, 450) for p in prefix) + " [cand] " + ptxt(C, c, 600)


def neighbours(C, p):
    """passages linked to p in the grounded evidence graph: subject links (both directions) and shared fact arguments"""
    out = set(C.subj_out[p])
    te = C.title_e[p]
    if te is not None:
        out.update(C.e_pass[te][:60])
    for f in C.p_facts[p]:
        for e in C.f_args[f]:
            for e2 in [e] + C.e_syn[e][:3]:
                out.update(C.e_pass[e2][:30])
    out.discard(p)
    return out


def action_space(C, base, prefix, sq, use_graph=True):
    A = [c for c in base if c not in prefix]
    if use_graph and prefix:
        seen = set(A) | set(prefix)
        nb = [c for c in set().union(*(neighbours(C, p) for p in prefix)) if c not in seen]
        nb.sort(key=lambda c: -sq[c])
        A += nb[:NB * len(prefix)]
    return A


def score(m, tok, q, docs, bs=32):
    out = []
    for i in range(0, len(docs), bs):
        enc = tok([q] * len(docs[i:i + bs]), docs[i:i + bs], truncation="only_second", max_length=MAXLEN,
                  padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(m(**enc).logits.squeeze(-1).float())
    return torch.cat(out)


# ---------------------------------------------------------------- training data
def build():
    out = open(f"{ROOT}/ckpt/walk_train.jsonl", "w")
    for ds in ["hotpotqa", "2wikimultihopqa", "musique"]:
        C = Corpus(f"{ROOT}/data/train/{ds}/index.pt", device="cpu")
        qs = [json.loads(l) for l in open(f"{ROOT}/data/train/{ds}/questions.jsonl", encoding="utf-8")]
        Q = embed_queries([q["question"] for q in qs]).cpu()
        P = C.P.float()
        for q, qe in zip(qs, Q):
            gp = [C.pid2i[p] for p in q["gold_pids"] if p in C.pid2i]
            if len(gp) < 2:
                continue
            sq = (P @ qe.float()).numpy()
            base = np.argsort(-sq)[:12].tolist()
            ordered = bool(q.get("ordered"))
            steps, prefix, rem = [], [], list(gp)
            while rem:
                pos = [rem[0]] if ordered else list(rem)
                A = action_space(C, base, prefix, sq)
                neg = [c for c in A if c not in gp]
                steps.append({"prefix": list(prefix), "pos": pos, "neg": neg[:48]})
                nxt = pos[0] if ordered else max(pos, key=lambda c: sq[c])   # teacher path: easiest remaining gold
                prefix.append(nxt); rem.remove(nxt)
            out.write(json.dumps({"ds": ds, "question": q["question"], "steps": steps}) + "\n")
        print("built", ds, flush=True)


# ---------------------------------------------------------------- training
def train(init, outdir, seed):
    random.seed(seed); torch.manual_seed(seed)
    rows = [json.loads(l) for l in open(f"{ROOT}/ckpt/walk_train.jsonl", encoding="utf-8")]
    Cs = {ds: Corpus(f"{ROOT}/data/train/{ds}/index.pt", device="cpu") for ds in ["hotpotqa", "2wikimultihopqa", "musique"]}
    G = []
    for r in rows:
        for st in r["steps"]:
            G.append((r["ds"], r["question"], st))
    print("groups", len(G), flush=True)
    tok = AutoTokenizer.from_pretrained(init)
    m = AutoModelForSequenceClassification.from_pretrained(init, num_labels=1).cuda()
    m.gradient_checkpointing_enable()
    lr, acc = 1e-5, 4
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=len(G) // acc + 5, pct_start=0.1)
    random.shuffle(G); m.train(); t0 = time.time()
    for k, (ds, q, st) in enumerate(G, 1):
        C = Cs[ds]
        neg = random.sample(st["neg"], min(14, len(st["neg"])))
        cands = st["pos"] + neg
        if len(cands) < 2:
            continue
        docs = [doc(C, st["prefix"], c) for c in cands]
        enc = tok([q] * len(docs), docs, truncation="only_second", max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = m(**enc).logits.squeeze(-1).float()
        npos = len(st["pos"])
        loss = (torch.logsumexp(s, 0) - torch.logsumexp(s[:npos], 0)) / acc   # order-marginal: any remaining gold hop
        loss.backward()
        if k % acc == 0:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(); sched.step()
        if k % 500 == 0:
            print(k, round(float(loss) * acc, 4), f"{time.time() - t0:.0f}s", flush=True)
    m.save_pretrained(outdir); tok.save_pretrained(outdir)
    print("saved", outdir, flush=True)


# ---------------------------------------------------------------- inference
@torch.no_grad()
def walk(m, tok, C, q, base, sq, use_graph, depth):
    beams = [((), 0.0)]
    finals, u = [], None
    for t in range(depth):
        nxt = []
        for prefix, lp in beams:
            A = action_space(C, base, list(prefix), sq, use_graph)
            s = score(m, tok, q, [doc(C, list(prefix), c) for c in A])
            if t == 0:
                u = dict(zip(A, s.tolist()))
            ls = F.log_softmax(s, 0).tolist()
            nxt += [(prefix + (c,), lp + l) for c, l in zip(A, ls)]
        nxt.sort(key=lambda x: -x[1])
        beams = nxt[:BEAM]
        finals += beams
    # posterior marginal of each passage under the (length-normalised) beam distribution
    w = np.array([lp / len(ch) for ch, lp in finals]); w = np.exp(w - w.max()); w /= w.sum()
    marg = {}
    for (ch, _), wi in zip(finals, w):
        for c in ch:
            marg[c] = marg.get(c, 0.0) + wi
    best = max(finals, key=lambda x: x[1] / len(x[0]))[0]
    rest = sorted(marg, key=lambda c: -marg[c])
    tail = sorted(u, key=lambda c: -u[c])
    return list(dict.fromkeys(list(best) + rest + tail))


def infer(ds, mdir, name, variant="full"):
    tok = AutoTokenizer.from_pretrained(mdir)
    m = AutoModelForSequenceClassification.from_pretrained(mdir, num_labels=1, torch_dtype=torch.bfloat16).cuda().eval()
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    rows = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand_v7.jsonl", encoding="utf-8")]
    n = int(os.environ.get("LIMIT", len(qs))); qs, rows = qs[:n], rows[:n]
    Q = embed_queries([q["question"] for q in qs]).cpu(); P = C.P.float()
    use_graph, depth = variant != "nograph", (1 if variant == "depth1" else DEPTH)
    orders, t0 = [], time.time()
    for q, r, qe in zip(qs, rows, Q):
        sq = (P @ qe.float()).numpy()
        base = [c["pid"] for c in r["cands"]]
        orders.append(walk(m, tok, C, q["question"], base, sq, use_graph, depth))
    lat = (time.time() - t0) / len(qs)
    for B in [300, 450, 600, 800]:
        items = []
        for q, od in zip(qs, orders):
            pas, _ = budget.pack(C, od, B)
            items.append({"qid": q["qid"], "rank": od, "pas": pas, "facts": [], "sel_p": od[:2]})
        rk = lambda k: 100 * np.mean([len(set(q["gold_titles"]) & {C.titles[p] for p in o["rank"][:k]}) / len(set(q["gold_titles"])) for q, o in zip(qs, items)])
        cc = 100 * np.mean([float(set(q["gold_titles"]) <= {C.titles[p] for p in o["pas"]}) for q, o in zip(qs, items)])
        reach = 100 * np.mean([float(set(q["gold_titles"]) <= {C.titles[p] for p in o["rank"]}) for q, o in zip(qs, items)])
        tag = f"{name}_SB{B}"
        json.dump({"method": tag, "R@2": rk(2), "R@5": rk(5), "ChainCtx": cc, "Reach": reach, "latency_s": lat, "items": items},
                  open(f"{ROOT}/results/{ds}/{tag}.json", "w"))
        print(f"{ds} {tag:28s} R@2 {rk(2):.1f} R@5 {rk(5):.1f} Chain {cc:.1f} Reach {reach:.1f} lat {1000*lat:.0f}ms", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "build":
        build()
    elif a[0] == "train":
        train(a[1], a[2], int(a[3]))
    else:
        infer(a[1], a[2], a[3], a[4] if len(a) > 4 else "full")
