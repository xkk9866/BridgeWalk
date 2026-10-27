"""BridgeWalk: evidence-chain retrieval as a latent walk on the passage-entity mention graph.

A chain is a path  q -> c_1 -e_1-> c_2 -e_2-> ... -> c_T  on the bipartite graph of passages and the entities they
mention.  One cross-encoder Phi reads (question, candidate, chosen prefix) and, from a *single* forward pass, emits
  * s(c)      - how well c continues this chain for this question          (sequence read-out)
  * pi(e | c) - which entity mentioned in c the chain leaves c through      (token read-out over mention spans)
  * h(c)      - whether the chain is complete once c is appended           (sequence read-out)
so  p(chain) = prod_t softmax_{A_t}(s)[c_t] * (1-h(c_t))^{[t<T]} * h(c_T),
where the support A_t is the proposer's candidates (the empty bridge) plus the passages that mention the bridges
the model itself chose in the prefix.  The graph is no longer a fixed neighbourhood: its entity nodes are the latent
transition variables of the chain model.  Depth is decided by h, not by a cap or by length normalisation.

python bridgewalk.py build
python bridgewalk.py train <init_dir> <out> <seed>
python bridgewalk.py infer <ds> <model_dir> <name> [variant]
   variant: full | nostop | qbridge | allment | oie | nograph     (env: C0=prop|dense, KB, RB)"""
import json, math, os, random, re, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unicor import Corpus
from index import embed_queries
import budget
torch.set_num_threads(int(os.environ.get("NTH", 4)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEAM, MAXLEN = 4, 640
DEPTH = int(os.environ.get("DEPTH", 4))
NEG = os.environ.get("NEG", "mention")  # training negatives: widest mention support, or the fixed OpenIE neighbourhood
TRAINF = os.environ.get("TRAINF", "bridge_train.jsonl")
KB = int(os.environ.get("KB", 2))      # bridges followed out of each chosen passage
RB = int(os.environ.get("RB", 8))      # passages admitted per followed bridge (ordered by question similarity)
HALT = os.environ.get("HALT", "all")   # halting (sufficiency) loss over all candidates of a group, or positives only
ART = {"the", "a", "an"}
WORD = re.compile(r"[^\W_]+")


# ---------------------------------------------------------------- mention graph
class MentionGraph:
    """Bipartite passage-entity graph. An entity is linked to a passage if it is an argument of a fact extracted from
    it, its subject (title) entity, or if its surface form occurs in the passage text (text-grounded mention)."""

    def __init__(self, C, path, text_edges=True):
        self.C = C; self._sc = {}
        self.eidx = {e: i for i, e in enumerate(C.ents)}
        if os.path.exists(path):
            tm = torch.load(path, weights_only=False)
        else:
            tm = [[] for _ in C.ents]
            for p in range(C.nP):
                for e in self.spans(C.titles[p] + ": " + C.texts[p]):
                    tm[e].append(p)
            torch.save(tm, path)
        oie = [set(C.e_pass[e]) for e in range(C.nE)]
        self.M = [sorted(oie[e] | set(tm[e])) if text_edges else sorted(oie[e]) for e in range(C.nE)]

    def spans(self, s):
        """entity id -> list of (start, end) character spans of its mentions in s (n-grams over article-free words)"""
        w = [(m.group().lower(), m.start(), m.end()) for m in WORD.finditer(s)]
        w = [x for x in w if x[0] not in ART]
        out = {}
        for n in range(1, 7):
            for i in range(len(w) - n + 1):
                e = self.eidx.get(" ".join(x[0] for x in w[i:i + n]))
                if e is not None:
                    out.setdefault(e, []).append((w[i][1], w[i + n - 1][2]))
        return out

    def cspans(self, c, s):
        k = (c, len(s))
        if k not in self._sc:
            self._sc[k] = self.spans(s)
        return self._sc[k]

    def passages(self, e, sq, exclude, cap=-1):
        """passages linked to entity e (or one of its synonyms), most question-similar first (top RB unless cap=None)"""
        P = set(self.M[e])
        for e2 in self.C.e_syn[e][:3]:
            P.update(self.M[e2])
        P = [p for p in P if p not in exclude]
        P.sort(key=lambda p: -sq[p])
        return P if cap is None else P[:RB]


# ---------------------------------------------------------------- the network
class Heads(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.stop = nn.Sequential(nn.Linear(H, H // 4), nn.Tanh(), nn.Linear(H // 4, 1))
        self.bridge = nn.Sequential(nn.Linear(2 * H, H // 4), nn.Tanh(), nn.Linear(H // 4, 1))


def neighbours(C, p):
    """fixed OpenIE neighbourhood of p (ChainWalk): subject links and passages sharing a fact argument"""
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


def ptxt(C, p, n):
    return f"{C.titles[p]}: {C.texts[p][:n]}"


def doc(C, prefix, c):
    """candidate first (its mention spans start at offset 0), then the prefix, most recent hop first"""
    s = ptxt(C, c, 900 if not prefix else 600)
    for p in reversed(prefix):
        s += " [prev] " + ptxt(C, p, 450)
    return s


def forward(m, hd, tok, G, q, docs, cands, bs=int(os.environ.get("IBS", 16))):
    """one pass per candidate -> continuation logits s, stop logits h, and per-candidate bridge logits {e: logit}"""
    S, Hs, Bs = [], [], []
    for i in range(0, len(docs), bs):
        D = docs[i:i + bs]
        enc = tok([q] * len(D), D, truncation="only_second", max_length=MAXLEN, padding=True,
                  return_tensors="pt", return_offsets_mapping=True)
        off = enc.pop("offset_mapping"); seq = [enc.sequence_ids(j) for j in range(len(D))]
        enc = enc.to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid = m.roberta(**enc).last_hidden_state
            s = m.classifier(hid).squeeze(-1).float()
            cls = hid[:, 0]
            h = hd.stop(cls).squeeze(-1).float()
        S.append(s); Hs.append(h)
        for j, c in enumerate(cands[i:i + bs]):
            sp = G.cspans(c, D[j].split(" [prev] ")[0])
            if not sp:
                Bs.append(({}, None)); continue
            ids = list(sp)
            in2 = torch.tensor([x == 1 for x in seq[j]], device="cuda")
            o = off[j].to("cuda", non_blocking=True); st, en = o[:, 0], o[:, 1]
            ab = torch.tensor([[a, b, k] for k, e in enumerate(ids) for a, b in sp[e]], device="cuda")
            hit = ((st[None] < ab[:, 1:2]) & (en[None] > ab[:, 0:1]) & in2[None]).float()       # span x token
            Mk = torch.zeros(len(ids), hit.shape[1], device="cuda").index_add_(0, ab[:, 2], hit).clamp_(max=1)
            keep = Mk.sum(1) > 0
            ids = [e for e, k in zip(ids, keep.tolist()) if k]; Mk = Mk[keep]
            if not ids:
                Bs.append(({}, None)); continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                span = (Mk @ hid[j].float()) / Mk.sum(1, keepdim=True)
                z = hd.bridge(torch.cat([span, cls[j].float().expand_as(span)], -1)).squeeze(-1).float()
            Bs.append((ids, z))
    return torch.cat(S), torch.cat(Hs), Bs


# ---------------------------------------------------------------- training data
def build():
    out = open(f"{ROOT}/ckpt/{TRAINF}", "w")
    for ds in ["hotpotqa", "2wikimultihopqa", "musique"]:
        C = Corpus(f"{ROOT}/data/train/{ds}/index.pt", device="cpu")
        G = MentionGraph(C, f"{ROOT}/data/train/{ds}/text_mentions.pt")
        qs = [json.loads(l) for l in open(f"{ROOT}/data/train/{ds}/questions.jsonl", encoding="utf-8")]
        Q = embed_queries([q["question"] for q in qs]).cpu(); P = C.P.float()
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
                pool = set(base)
                for p in prefix:                                   # every bridge of the prefix: the walk's widest support
                    if NEG == "fixnb":
                        nb = sorted(neighbours(C, p) - set(prefix), key=lambda c: -sq[c])
                        pool.update(nb[:KB * RB])
                        continue
                    for e in G.spans(ptxt(C, p, 600)):
                        pool.update(G.passages(e, sq, set(prefix))[:8])
                neg = [c for c in pool if c not in gp and c not in prefix]
                lab = {}
                for c in pos:
                    after = [r for r in rem if r != c]
                    br = [e for e in G.spans(ptxt(C, c, 900 if not prefix else 600)) if set(G.M[e]) & set(after)] if after else []
                    lab[str(c)] = {"stop": int(not after), "bridges": br}
                steps.append({"prefix": list(prefix), "pos": pos, "neg": random.sample(neg, min(48, len(neg))), "lab": lab})
                nxt = pos[0] if ordered else max(pos, key=lambda c: sq[c])
                prefix.append(nxt); rem.remove(nxt)
            out.write(json.dumps({"ds": ds, "question": q["question"], "steps": steps}) + "\n")
        print("built", ds, flush=True)


def build_onpolicy(mine_dir, outf, per_q=3):
    """States the walk actually visits: proper prefixes of the walks that a cross-fitted half-model (trained on the
    other half of the questions) completes on held-out training questions.  Each prefix that does not yet cover the
    gold set is a training state: the remaining gold passages are the positives (order-marginal), negatives come from the
    widest support of that prefix, and halting / bridge labels are defined exactly as for teacher states."""
    out = open(f"{ROOT}/ckpt/{outf}", "w"); n = 0
    rows = [json.loads(l) for f in sorted(os.listdir(mine_dir)) for l in open(f"{mine_dir}/{f}", encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["ds"], []).append(r)
    for ds, rs in by.items():
        C = Corpus(f"{ROOT}/data/train/{ds}/index.pt", device="cpu")
        G = MentionGraph(C, f"{ROOT}/data/train/{ds}/text_mentions.pt")
        Q = embed_queries([r["question"] for r in rs]).cpu(); P = C.P.float()
        for r, qe in zip(rs, Q):
            gold = set(r["gold"]); sq = (P @ qe.float()).numpy()
            base = np.argsort(-sq)[:12].tolist()
            pref = []
            for ch in r["chains"]:
                for t in range(1, len(ch)):
                    pr = tuple(ch[:t])
                    if pr not in pref and not gold <= set(pr) and not set(pr) <= gold:   # states off the gold path
                        pref.append(pr)
            steps = []
            for pr in pref[:per_q]:
                rem = [g for g in r["gold"] if g not in pr]
                pool = set(base)
                for p in pr:
                    for e in G.spans(ptxt(C, p, 600)):
                        pool.update(G.passages(e, sq, set(pr))[:8])
                neg = [c for c in pool if c not in gold and c not in pr]
                lab = {}
                for c in rem:
                    after = [x for x in rem if x != c]
                    br = [e for e in G.spans(ptxt(C, c, 600)) if set(G.M[e]) & set(after)] if after else []
                    lab[str(c)] = {"stop": int(not after), "bridges": br}
                steps.append({"prefix": list(pr), "pos": rem, "neg": random.sample(neg, min(48, len(neg))), "lab": lab})
            if steps:
                out.write(json.dumps({"ds": ds, "question": r["question"], "steps": steps, "src": "onpol"}) + chr(10)); n += len(steps)
        print("on-policy", ds, n, flush=True)


# ---------------------------------------------------------------- training: one chain likelihood
class LoRA(nn.Module):
    """frozen base projection + trainable low-rank update (merged into the base weight when saved)"""

    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base = base; self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(x.dtype) * self.scale


LORA_T = ("query", "key", "value", "dense")


def add_lora(m):
    for layer in m.roberta.encoder.layer:
        for parent in [layer.attention.self, layer.attention.output, layer.intermediate, layer.output]:
            for n in LORA_T:
                if isinstance(getattr(parent, n, None), nn.Linear):
                    setattr(parent, n, LoRA(getattr(parent, n)))


def merge_lora(m):
    for layer in m.roberta.encoder.layer:
        for parent in [layer.attention.self, layer.attention.output, layer.intermediate, layer.output]:
            for n in LORA_T:
                mod = getattr(parent, n, None)
                if isinstance(mod, LoRA):
                    with torch.no_grad():
                        mod.base.weight += ((mod.B @ mod.A) * mod.scale).to(mod.base.weight.dtype)
                    setattr(parent, n, mod.base)


def train(init, outdir, seed):
    """Memory-lean training (shared GPU): frozen bf16 encoder + LoRA (r=16) on every projection, trainable sequence
    read-out and heads, and gradient caching so that the group softmax is exact while only CH documents are resident."""
    random.seed(seed); torch.manual_seed(seed)
    rows = [json.loads(l) for f in TRAINF.split(",") for l in open(f"{ROOT}/ckpt/{f}", encoding="utf-8")]
    Cs = {ds: Corpus(f"{ROOT}/data/train/{ds}/index.pt", device="cpu") for ds in ["hotpotqa", "2wikimultihopqa", "musique"]}
    Gs = {ds: MentionGraph(C, f"{ROOT}/data/train/{ds}/text_mentions.pt") for ds, C in Cs.items()}
    fold = os.environ.get("FOLD")             # cross-fitting: train on the questions of one hash fold only
    if fold is not None:
        import hashlib
        rows = [r for r in rows if int(hashlib.md5(r["question"].encode()).hexdigest(), 16) % 2 == int(fold)]
    Gr = [(r["ds"], r["question"], st) for r in rows for st in r["steps"]]
    Gr = Gr[:int(os.environ.get("SMOKE", len(Gr)))]
    if os.environ.get("TEACH_SUB"):          # DAgger round: keep every on-policy state plus a sample of teacher states
        nt = sum(len(r["steps"]) for r in rows if r.get("src") != "onpol")
        random.seed(seed); keep = set(random.sample(range(nt), min(nt, int(os.environ["TEACH_SUB"]))))
        it = iter(range(nt))
        Gr = [g for g, r in zip(Gr, [r for r in rows for _ in r["steps"]]) if r.get("src") == "onpol" or next(it) in keep]
    print("groups", len(Gr), flush=True)
    tok = AutoTokenizer.from_pretrained(init)
    full = bool(os.environ.get("FULLFT"))       # full fine-tuning (fp32 master weights, 8-bit AdamW) instead of LoRA
    m = AutoModelForSequenceClassification.from_pretrained(init, num_labels=1, torch_dtype=torch.float32 if full else torch.bfloat16).cuda()
    for p in m.parameters():
        p.requires_grad_(full)
    if not full:
        add_lora(m)
    m.classifier.float()
    for p in m.classifier.parameters():
        p.requires_grad_(True)
    m.gradient_checkpointing_enable(); m.config.use_cache = False
    m.roberta.embeddings.word_embeddings.weight.requires_grad_(False)
    m.enable_input_require_grads()
    hd = Heads(m.config.hidden_size).cuda()
    params = [p for p in m.parameters() if p.requires_grad] + list(hd.parameters())
    print("trainable", sum(p.numel() for p in params), flush=True)
    lr, acc, CH = (1e-5 if full else 2e-4), 4, int(os.environ.get("CH", 4))
    if full:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=len(Gr) // acc + 5, pct_start=0.1)
    for mod in m.modules():                                                    # deterministic passes for the gradient cache
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    random.shuffle(Gr); m.train(); t0 = time.time(); run = np.zeros(3); nrun = 0
    for k, (ds, q, st) in enumerate(Gr, 1):
        C, G = Cs[ds], Gs[ds]
        neg = random.sample(st["neg"], min(14, len(st["neg"])))
        cands = st["pos"] + neg
        if len(cands) < 2:
            continue
        docs = [doc(C, st["prefix"], c) for c in cands]
        npos = len(st["pos"])
        with torch.no_grad():                                                   # pass 1: all continuation logits
            s0, _, _ = forward(m, hd, tok, G, q, docs, cands, bs=CH)
        sl = s0.detach().requires_grad_(True)
        l_walk = torch.logsumexp(sl, 0) - torch.logsumexp(sl[:npos], 0)         # order-marginal transition
        l_walk.backward(); gs = sl.grad
        l_stop_v, l_br_v, nb = 0.0, 0.0, 0
        nbr = sum(1 for c in st["pos"] if st["lab"][str(c)]["bridges"])
        for i0 in range(0, len(docs), CH):                                      # pass 2: exact gradients, chunked
            s, h, Bs = forward(m, hd, tok, G, q, docs[i0:i0 + CH], cands[i0:i0 + CH], bs=CH)
            loss = (s * gs[i0:i0 + CH]).sum()
            for j in range(len(s)):
                i = i0 + j
                # sufficiency: h(c) = P(evidence c_<t + c answers q). Every candidate the walk can face is a training
                # state (a wrong passage, or a correct but incomplete prefix, is labelled 0).  HALT=pos restores the
                # positives-only halting loss (ablation).
                if i < npos:
                    y_h = float(st["lab"][str(cands[i])]["stop"])
                elif HALT == "pos":
                    break
                else:
                    y_h = 0.0
                ls_ = F.binary_cross_entropy_with_logits(h[j], torch.tensor(y_h, device="cuda")) / (npos if HALT == "pos" else len(cands))
                loss = loss + ls_; l_stop_v += float(ls_)
                if i >= npos:
                    continue
                lab = st["lab"][str(cands[i])]
                ids, z = Bs[j]
                if z is not None and lab["bridges"]:
                    y = torch.tensor([e in set(lab["bridges"]) for e in ids], device="cuda")
                    if y.any():                                                 # marginal over the valid bridges
                        lb = (torch.logsumexp(z, 0) - torch.logsumexp(z[y], 0)) / nbr
                        loss = loss + lb; l_br_v += float(lb); nb += 1
            (loss / acc).backward()
        run += [float(l_walk), l_stop_v, l_br_v]; nrun += 1
        if k % acc == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(); sched.step()
        if k % 500 == 0 or k == len(Gr):
            print(k, "walk/stop/bridge", np.round(run / nrun, 4).tolist(), f"{time.time() - t0:.0f}s",
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.2f}G", flush=True); run[:] = 0; nrun = 0
    merge_lora(m)
    m.save_pretrained(outdir); tok.save_pretrained(outdir); torch.save(hd.state_dict(), f"{outdir}/heads.pt")
    print("saved", outdir, flush=True)


# ---------------------------------------------------------------- inference: beam over latent walks
@torch.no_grad()
def seq_readouts(m, hd, tok, q, docs, bs=int(os.environ.get("IBS", 16))):
    """transition and halting read-outs for many candidates (length-sorted batches; no span pooling needed)"""
    S, H = torch.empty(len(docs)), torch.empty(len(docs))
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    for i in range(0, len(docs), bs):
        idx = order[i:i + bs]
        enc = tok([q] * len(idx), [docs[j] for j in idx], truncation="only_second", max_length=MAXLEN, padding=True,
                  return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid = m.roberta(**enc).last_hidden_state
            s = m.classifier(hid).squeeze(-1).float()
            h = hd.stop(hid[:, 0]).squeeze(-1).float()
        S[idx] = s.cpu(); H[idx] = h.cpu()
    return S, H


def bridges_of(m, hd, tok, G, q, prefix, c, variant, qe):
    """bridge read-out of the passage a surviving state ends in (same network, same input as its transition score)"""
    C = G.C
    if variant == "allment":                      # control: no pointer, every mentioned entity is followed
        return tuple(G.cspans(c, ptxt(C, c, 900 if not prefix else 600)))
    if variant == "fixnb":
        return ()
    _, _, Bs = forward(m, hd, tok, G, q, [doc(C, list(prefix), c)], [c])
    ids, z = Bs[0]
    if not ids:
        return ()
    if variant == "qbridge":                      # control: follow the entities most similar to the question
        sc = (C.E[torch.tensor(ids)].float() @ qe.float()).tolist()
    else:
        sc = z.tolist()
    return tuple(ids[j] for j in np.argsort(sc)[::-1][:KB])


def support(G, base, prefix, brs, sq, variant):
    A = [c for c in base if c not in prefix]
    if variant == "nograph" or not prefix:
        return A
    seen = set(A) | set(prefix)
    if variant == "fixnb":                        # matched control: fixed OpenIE neighbourhood, same admitted count
        for p in prefix:
            nb = sorted(neighbours(G.C, p) - seen, key=lambda c: -sq[c])[:KB * RB]
            A += nb; seen.update(nb)
        return A
    if variant == "allment":                      # same support size as the pointer walk, filled by question similarity
        pool = set()
        for e in brs:
            pool.update(G.passages(e, sq, seen, cap=None))
        pool = sorted(pool - seen, key=lambda p: -sq[p])
        return A + pool[:KB * RB * len(prefix)]
    for e in brs:
        for c in G.passages(e, sq, seen):
            A.append(c); seen.add(c)
    return A


@torch.no_grad()
def walk(m, hd, tok, G, q, base, sq, variant, qe=None):
    """beam search over latent walks.  A state is (prefix, log p(prefix, not halted), bridges, per-hop bridges)."""
    beams = [((), 0.0, (), ())]
    finals, u = [], None
    depth = 3 if variant == "nostop" else DEPTH
    for t in range(depth):
        nxt = []
        for prefix, lp, brs, hb in beams:
            A = support(G, base, prefix, brs, sq, variant)
            if not A:
                continue
            s, h = seq_readouts(m, hd, tok, q, [doc(G.C, list(prefix), c) for c in A])
            if t == 0:
                u = dict(zip(A, s.tolist()))
            ls = F.log_softmax(s, 0); lh = F.logsigmoid(h); lc = F.logsigmoid(-h)
            for i, c in enumerate(A):
                ch = prefix + (c,)
                if variant == "nostop":           # ChainWalk-style: fixed depth, length-normalised
                    finals.append((ch, (lp + float(ls[i])) / len(ch), hb, 0.0))
                    nxt.append((ch, lp + float(ls[i]), brs, hb))
                else:
                    finals.append((ch, lp + float(ls[i] + lh[i]), hb, float(torch.sigmoid(h[i])), lp, float(ls[i])))
                    nxt.append((ch, lp + float(ls[i] + lc[i]), brs, hb))
        nxt.sort(key=lambda x: -x[1])
        beams = []
        if t + 1 < depth:                         # bridges are read only for the states that survive the beam
            for ch, lp, brs, hb in nxt[:BEAM]:
                top = bridges_of(m, hd, tok, G, q, ch[:-1], ch[-1], variant, qe)
                beams.append((ch, lp, brs + top, hb + (top,)))
    w = np.array([f[1] for f in finals]); w = np.exp(w - w.max()); w /= w.sum()
    marg = {}
    for f, wi in zip(finals, w):
        for c in f[0]:
            marg[c] = marg.get(c, 0.0) + wi
    bi = int(np.argmax(w)); best = finals[bi][0]
    rest = sorted(marg, key=lambda c: -marg[c])
    tail = sorted(u, key=lambda c: -u[c])
    if os.environ.get("DUMP"):                  # diagnostics: top completed chains with their factors
        top = np.argsort(-w)[:40]
        walk.dump = [{"ch": list(finals[j][0]), "w": float(w[j]), "h": finals[j][3], "lp": finals[j][4], "ls": finals[j][5]} for j in top]
    return list(dict.fromkeys(list(best) + rest + tail)), list(best), [list(x) for x in finals[bi][2]], float(w[bi])


C0 = os.environ.get("C0", "prop")       # first-hop candidates: graph proposer ("prop") or dense top-12 ("dense")


def infer(ds, mdir, name, variant="full"):
    tok = AutoTokenizer.from_pretrained(mdir)
    m = AutoModelForSequenceClassification.from_pretrained(mdir, num_labels=1, torch_dtype=torch.bfloat16).cuda().eval()
    hd = Heads(m.config.hidden_size).cuda().eval(); hd.load_state_dict(torch.load(f"{mdir}/heads.pt"))
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    G = MentionGraph(C, f"{ROOT}/data/{ds}/text_mentions.pt", text_edges=(variant != "oie"))
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    rows = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand_v7.jsonl", encoding="utf-8")]
    n = int(os.environ.get("LIMIT", len(qs))); qs, rows = qs[:n], rows[:n]
    Q = embed_queries([q["question"] for q in qs]).cpu(); P = C.P.float()
    out, t0 = [], time.time()
    for q, r, qe in zip(qs, rows, Q):
        sq = (P @ qe.float()).numpy()
        base = np.argsort(-sq)[:12].tolist() if C0 == "dense" else [c["pid"] for c in r["cands"]]
        t1 = time.time()
        od, best, hb, wb = walk(m, hd, tok, G, q["question"], base, sq, variant, qe)
        out.append({"qid": q["qid"], "rank": od, "chain": best, "chain_br": hb, "w_map": wb, "lat": time.time() - t1})
        if os.environ.get("DUMP"):
            out[-1]["finals"] = walk.dump
    lat = (time.time() - t0) / len(qs)
    for B in [300, 450, 600, 800]:
        items = []
        for o in out:
            pas, _ = budget.pack(C, o["rank"], B)
            items.append({**o, "pas": pas, "facts": [], "sel_p": o["rank"][:2], "len": len(o["chain"])})
        rk = lambda k: 100 * np.mean([len(set(q["gold_titles"]) & {C.titles[p] for p in o["rank"][:k]}) / len(set(q["gold_titles"])) for q, o in zip(qs, items)])
        cc = 100 * np.mean([float(set(q["gold_titles"]) <= {C.titles[p] for p in o["pas"]}) for q, o in zip(qs, items)])
        reach = 100 * np.mean([float(set(q["gold_titles"]) <= {C.titles[p] for p in o["rank"]}) for q, o in zip(qs, items)])
        tag = f"{name}_SB{B}"
        json.dump({"method": tag, "R@2": rk(2), "R@5": rk(5), "ChainCtx": cc, "Reach": reach, "latency_s": lat, "items": items},
                  open(f"{ROOT}/results/{ds}/{tag}.json", "w"))
        print(f"{ds} {tag:28s} R@2 {rk(2):.1f} R@5 {rk(5):.1f} Chain {cc:.1f} Reach {reach:.1f} "
              f"len {np.mean([len(o['chain']) for o in out]):.2f} lat {1000*lat:.0f}ms", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "build":
        build()
    elif a[0] == "build_onpolicy":
        build_onpolicy(a[1], a[2])
    elif a[0] == "train":
        train(a[1], a[2], int(a[3]))
    else:
        infer(a[1], a[2], a[3], a[4] if len(a) > 4 else "full")
