"""CoverWalk: the evidence chain, not the hop, is the unit of probability, learning and decoding.

BridgeWalk scores a chain as a product of per-hop softmaxes over supports A_t whose size depends on the bridges
chosen so far.  With a dense mention graph these supports are heterogeneous, and a locally normalised walk suffers
label bias: a hop drawn from a small support looks certain, a hop drawn from a large one looks doubtful, independently
of whether the chain is right.  The diagnostic signature is that latent bridges raise reachability but lower selection.

CoverWalk keeps the same network (one cross-encoder pass -> transition s, halting h, bridge pointer pi) and changes
the unit of the model to the whole walk:
  * energy        F(g) = sum_t s(c_t | c_<t) + sum_{t<T} log(1 - h(c_t)) + log h(c_T)           (no per-hop normaliser)
  * posterior     p(g | q) = exp F(g) / sum_{g' in Beam(q)} exp F(g')                            (global normalisation)
  * learning      -log p(gold chains | q) over the walk's own beam (search-aware), + bridge log-likelihood
  * decoding      the context X maximises the expected chain coverage  sum_g p(g|q) 1[g subset X]  under the exact
                  token budget B (budgeted maximum coverage, greedy with the Khuller-Moss-Naor guarantee)
so the training objective, the posterior and the decision rule all refer to the same event that is evaluated.

python coverwalk.py mine <train_ds> <model_dir> <shard> <nshard>      chains of the current model on training questions
python coverwalk.py train <model_dir> <out> <seed>                    globally normalised fine-tuning
python coverwalk.py infer <ds> <model_dir> <name> <score: local|global> walks -> results/<ds>/<name>.walks.json
python coverwalk.py pack <ds> <name> <pack: mbr|marg>                 contexts at B = 300/450/600/800 from saved walks"""
import json, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unicor import Corpus
from index import embed_queries
import budget
from bridgewalk import (MentionGraph, Heads, ptxt, doc, forward, seq_readouts, bridges_of, support, add_lora,
                        merge_lora, BEAM, DEPTH, ROOT)

VAR = os.environ.get("VAR", "full")   # expansion rule at test time: full | fixnb | allment | qbridge | oie
NCH = 24          # chains per training question (mined beam chains + gold chains)
NKEEP = 64        # completed walks kept per test question (posterior support)


# ---------------------------------------------------------------- beam over latent walks (local or global energy)
@torch.no_grad()
def walk(m, hd, tok, G, q, base, sq, score):
    """returns completed walks [(chain, F)] and the hop-1 scores.  score = local: per-hop log-softmax (BridgeWalk);
    global: raw transition logits, so walks are compared by their unnormalised energy"""
    beams, finals, u, vis = [((), 0.0, ())], [], None, set()
    for t in range(DEPTH):
        nxt = []
        for prefix, lp, brs in beams:
            A = support(G, base, prefix, brs, sq, VAR)
            if not A:
                continue
            s, h = seq_readouts(m, hd, tok, q, [doc(G.C, list(prefix), c) for c in A])
            if t == 0:
                u = dict(zip(A, s.tolist()))
            vis.update(A)
            ls = F.log_softmax(s, 0) if score == "local" else s
            lh, lc = F.logsigmoid(h), F.logsigmoid(-h)
            for i, c in enumerate(A):
                finals.append((prefix + (c,), lp + float(ls[i] + lh[i])))
                nxt.append((prefix + (c,), lp + float(ls[i] + lc[i]), brs))
        nxt.sort(key=lambda x: -x[1])
        beams = []
        if t + 1 < DEPTH:
            for ch, lp, brs in nxt[:BEAM]:
                beams.append((ch, lp, brs + bridges_of(m, hd, tok, G, q, ch[:-1], ch[-1], VAR, walk.qe)))
    finals.sort(key=lambda f: -f[1])
    walk.vis = vis
    return finals, u


walk.qe = None


def load(mdir):
    tok = AutoTokenizer.from_pretrained(mdir)
    m = AutoModelForSequenceClassification.from_pretrained(mdir, num_labels=1, torch_dtype=torch.bfloat16).cuda().eval()
    hd = Heads(m.config.hidden_size).cuda().eval(); hd.load_state_dict(torch.load(f"{mdir}/heads.pt"))
    return tok, m, hd


def teacher(q, gp):
    """gold chains: the given order (MuSiQue) or, for set-valued chains, every order (at most 4 passages)"""
    import itertools
    if q.get("ordered"):
        return [tuple(gp)]
    return [tuple(o) for o in itertools.permutations(gp)][:6]


# ---------------------------------------------------------------- mining the model's own walks on training questions
def mine(ds, mdir, shard, nshard):
    tok, m, hd = load(mdir)
    C = Corpus(f"{ROOT}/data/train/{ds}/index.pt", device="cpu")
    G = MentionGraph(C, f"{ROOT}/data/train/{ds}/text_mentions.pt")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/train/{ds}/questions.jsonl", encoding="utf-8")]
    qs = [q for q in qs if sum(p in C.pid2i for p in q["gold_pids"]) >= 2]
    hold = os.environ.get("HOLD")             # cross-fitted mining: only questions of hash fold HOLD (unseen by this model)
    if hold is not None:
        import hashlib
        qs = [q for q in qs if int(hashlib.md5(q["question"].encode()).hexdigest(), 16) % 2 == int(hold)]
    qs = qs[shard::nshard]
    Q = embed_queries([q["question"] for q in qs]).cpu(); P = C.P.float()
    mo = os.environ.get("MINEOUT", f"{ROOT}/ckpt/cw_mine"); os.makedirs(mo, exist_ok=True)
    out = open(f"{mo}/{ds}_{hold}_{shard}.jsonl", "w"); hit, t0 = 0, time.time()
    for k, (q, qe) in enumerate(zip(qs, Q), 1):
        gp = [C.pid2i[p] for p in q["gold_pids"] if p in C.pid2i]
        sq = (P @ qe.float()).numpy()
        base = np.argsort(-sq)[:12].tolist()
        finals, _ = walk(m, hd, tok, G, q["question"], base, sq, "local")
        hit += any(set(f[0]) == set(gp) for f in finals[:NCH])
        out.write(json.dumps({"ds": ds, "question": q["question"], "gold": gp, "ordered": bool(q.get("ordered")),
                              "chains": [list(f[0]) for f in finals[:NCH]]}) + "\n"); out.flush()
        if k % 50 == 0:
            print(ds, shard, k, len(qs), f"gold-in-beam {100 * hit / k:.1f}", f"{time.time() - t0:.0f}s", flush=True)
    print("done", ds, shard, f"gold-in-beam {100 * hit / max(1, len(qs)):.1f}", flush=True)


# ---------------------------------------------------------------- globally normalised, search-aware training
def train(mdir, outdir, seed):
    """Fine-tunes the BridgeWalk network so that exp F(g), normalised over the walks the beam actually produces,
    is the posterior of the gold chain.  Gradient caching keeps the chain-level softmax exact: pass 1 scores every
    (prefix, passage) node of the question's walks without gradients, the chain loss gives d loss / d (s, h) per node,
    pass 2 back-propagates those node gradients in chunks.  The bridge pointer keeps its marginal likelihood on the
    gold chain (the walk's joint likelihood factorises into the chain energy and the pointer)."""
    random.seed(seed); torch.manual_seed(seed)
    md = os.environ.get("MINE", f"{ROOT}/ckpt/cw_mine")
    rows = [json.loads(l) for f in sorted(os.listdir(md)) for l in open(f"{md}/{f}", encoding="utf-8")]
    Cs = {ds: Corpus(f"{ROOT}/data/train/{ds}/index.pt", device="cpu") for ds in {r["ds"] for r in rows}}
    Gs = {ds: MentionGraph(C, f"{ROOT}/data/train/{ds}/text_mentions.pt") for ds, C in Cs.items()}
    rows = rows[:int(os.environ.get("SMOKE", len(rows)))]
    print("questions", len(rows), flush=True)
    tok = AutoTokenizer.from_pretrained(mdir)
    m = AutoModelForSequenceClassification.from_pretrained(mdir, num_labels=1, torch_dtype=torch.bfloat16).cuda()
    for p in m.parameters():
        p.requires_grad_(False)
    add_lora(m); m.classifier.float()
    for p in m.classifier.parameters():
        p.requires_grad_(True)
    m.gradient_checkpointing_enable(); m.config.use_cache = False; m.enable_input_require_grads()
    hd = Heads(m.config.hidden_size).cuda(); hd.load_state_dict(torch.load(f"{mdir}/heads.pt"))
    params = [p for p in m.parameters() if p.requires_grad] + list(hd.parameters())
    print("trainable", sum(p.numel() for p in params), flush=True)
    lr, acc, CH = float(os.environ.get("LR", 1e-4)), 4, int(os.environ.get("CH", 16))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=len(rows) // acc + 5, pct_start=0.1)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    random.shuffle(rows); m.train(); t0 = time.time(); run = np.zeros(3); nrun = 0
    for k, r in enumerate(rows, 1):
        C, G, q = Cs[r["ds"]], Gs[r["ds"]], r["question"]
        gold = set(r["gold"])
        gch = teacher(r, r["gold"])
        chains = [tuple(c) for c in r["chains"]]
        chains = list(dict.fromkeys(gch + chains))[:NCH + len(gch)]
        pos = [i for i, g in enumerate(chains) if set(g) == gold]
        nodes = list(dict.fromkeys((g[:t], g[t]) for g in chains for t in range(len(g))))
        nid = {n: i for i, n in enumerate(nodes)}
        docs = [doc(C, list(p), c) for p, c in nodes]; cands = [c for _, c in nodes]
        with torch.no_grad():                                                   # pass 1: every node, no graph
            s0, h0, _ = forward(m, hd, tok, G, q, docs, cands, bs=CH)
        sl, hl = s0.detach().requires_grad_(True), h0.detach().requires_grad_(True)
        Fg = torch.stack([sum(sl[nid[(g[:t], g[t])]] for t in range(len(g)))
                          + sum(F.logsigmoid(-hl[nid[(g[:t], g[t])]]) for t in range(len(g) - 1))
                          + F.logsigmoid(hl[nid[(g[:-1], g[-1])]]) for g in chains])
        l_chain = torch.logsumexp(Fg, 0) - torch.logsumexp(Fg[pos], 0)
        l_chain.backward(); gs, gh = sl.grad, hl.grad
        # bridge targets on the gold chains' nodes: entities of c that lead to a gold passage not yet in the prefix
        blab = {}
        for g in gch:
            for t in range(len(g) - 1):
                c, rest = g[t], gold - set(g[:t + 1])
                br = [e for e in G.spans(ptxt(C, c, 900 if t == 0 else 600)) if set(G.M[e]) & rest]
                if br:
                    blab[nid[(g[:t], c)]] = set(br)
        l_br = 0.0
        for i0 in range(0, len(nodes), CH):                                     # pass 2: exact node gradients
            s, h, Bs = forward(m, hd, tok, G, q, docs[i0:i0 + CH], cands[i0:i0 + CH], bs=CH)
            loss = (s * gs[i0:i0 + CH]).sum() + (h * gh[i0:i0 + CH]).sum()
            for j, (ids, z) in enumerate(Bs):
                if i0 + j in blab and z is not None:
                    y = torch.tensor([e in blab[i0 + j] for e in ids], device="cuda")
                    if y.any():
                        lb = (torch.logsumexp(z, 0) - torch.logsumexp(z[y], 0)) / len(blab)
                        loss = loss + lb; l_br += float(lb)
            (loss / acc).backward()
        p_gold = float(torch.exp(-l_chain))
        run += [float(l_chain), l_br, p_gold]; nrun += 1
        if k % acc == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(); sched.step()
        if k % 200 == 0 or k == len(rows):
            print(k, "chain/bridge/p_gold", np.round(run / nrun, 4).tolist(), f"nodes {len(nodes)}", f"{time.time() - t0:.0f}s",
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.2f}G", flush=True); run[:] = 0; nrun = 0
    merge_lora(m)
    m.save_pretrained(outdir); tok.save_pretrained(outdir); torch.save(hd.state_dict(), f"{outdir}/heads.pt")
    print("saved", outdir, flush=True)


# ---------------------------------------------------------------- test-time walks
def infer(ds, mdir, name, score):
    tok, m, hd = load(mdir)
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    G = MentionGraph(C, f"{ROOT}/data/{ds}/text_mentions.pt", text_edges=(VAR != "oie"))
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    cf = f"{ROOT}/results/{ds}/chain_cand_v7.jsonl"
    rows = [json.loads(l) for l in open(cf, encoding="utf-8")] if os.path.exists(cf) else None
    sh, nsh = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    idx = list(range(len(qs)))[sh::nsh]
    Q = embed_queries([qs[i]["question"] for i in idx]).cpu(); P = C.P.float()
    out = []
    for i, qe in zip(idx, Q):
        q = qs[i]
        sq = (P @ qe.float()).numpy()
        base = [c["pid"] for c in rows[i]["cands"]] if rows else np.argsort(-sq)[:12].tolist()
        t1 = time.time(); walk.qe = qe
        finals, u = walk(m, hd, tok, G, q["question"], base, sq, score)
        out.append({"qid": q["qid"], "i": i, "finals": [[list(f[0]), f[1]] for f in finals[:NKEEP]],
                    "u": sorted(u, key=lambda c: -u[c]), "vis": sorted(walk.vis), "lat": time.time() - t1})
        if len(out) % 100 == 0:
            print(ds, name, sh, len(out), flush=True)
    json.dump(out, open(f"{ROOT}/results/{ds}/{name}.walks{sh}.json", "w"))
    print("done", ds, name, sh, f"lat {np.mean([o['lat'] for o in out]):.2f}s", flush=True)


# ---------------------------------------------------------------- decoding: expected chain coverage under the budget
def posterior(finals):
    w = np.array([f[1] for f in finals]); w = np.exp(w - w.max()); return w / w.sum()


def mbr_pack(C, chains, w, tail, B, cost):
    """argmax_X sum_g w(g) 1[g subset X]  s.t.  tokens(X) <= B.  Greedy budgeted maximum coverage: repeatedly add the
    passages of the walk with the largest gain in expected coverage per added token (Khuller et al., 1999: the better of
    this greedy and the best single walk is a (1 - 1/sqrt(e))-approximation).  When no further walk fits, the remaining
    room maximises the expected number of covered gold passages, i.e. passages in order of posterior marginal."""
    sets = [frozenset(g) for g in chains]
    by = {}
    for S, g in zip(sets, chains):
        by.setdefault(S, g)

    def value(X):
        return sum(wi for S, wi in zip(sets, w) if S <= X)

    def fits(pas):
        return budget.ntok(budget.render(C, pas, [])) <= B

    def greedy(start):
        X, order = set(start), list(start)
        while True:
            v0 = value(X)
            cand = [((value(X | S) - v0) / sum(cost(p) for p in S - X), S) for S in by if not S <= X]
            cand = sorted([c for c in cand if c[0] > 0], key=lambda c: -c[0])
            for _, S in cand:                                # best ratio that still fits the exact budget
                new = [p for p in by[S] if p not in X]
                if fits(order + new):
                    X |= S; order += new; break
            else:
                return order
    order = greedy(())
    singles = [list(g) for g in chains if fits(list(g))]
    if singles:
        bs = max(singles, key=lambda g: value(set(g)))
        if value(set(bs)) > value(set(order)):
            order = greedy(tuple(bs))
    mu = {}
    for g, wi in zip(chains, w):
        for p in g:
            mu[p] = mu.get(p, 0.0) + wi
    fill = sorted(mu, key=lambda p: -mu[p]) + list(tail)
    pas, _ = budget.pack(C, list(dict.fromkeys(order + fill)), B)
    pas = [p for p in order if p in pas] + [p for p in pas if p not in order]
    return pas, list(dict.fromkeys(order + fill))


def marg_pack(C, chains, w, tail, B):
    """BridgeWalk's packer: MAP walk, then passages by posterior marginal, then the hop-1 ranking"""
    mu = {}
    for g, wi in zip(chains, w):
        for p in g:
            mu[p] = mu.get(p, 0.0) + wi
    rank = list(dict.fromkeys(list(chains[0]) + sorted(mu, key=lambda p: -mu[p]) + list(tail)))
    return budget.pack(C, rank, B)[0], rank


def pack(ds, name, mode):
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    W = sorted([o for f in os.listdir(f"{ROOT}/results/{ds}") if f.startswith(f"{name}.walks") for o in json.load(open(f"{ROOT}/results/{ds}/{f}"))],
               key=lambda o: o["i"])
    assert len(W) == len(qs), (len(W), len(qs))
    _c = {}
    cost = lambda p: _c.setdefault(p, budget.ntok(budget.render(C, [p], [])))
    tag0 = name if mode == "mbr" else f"{name}_marg"
    for B in [300, 450, 600, 800]:
        items = []
        for o in W:
            chains = [f[0] for f in o["finals"]]; w = posterior(o["finals"])
            pas, rank = mbr_pack(C, chains, w, o["u"], B, cost) if mode == "mbr" else marg_pack(C, chains, w, o["u"], B)
            items.append({"qid": o["qid"], "rank": rank, "pas": pas, "facts": [], "chain": chains[0], "w_map": float(w[0]),
                          "len": len(chains[0]), "lat": o["lat"], "sel_p": rank[:2]})
        gt = [set(q["gold_titles"]) for q in qs]
        rk = lambda k: 100 * np.mean([len(g & {C.titles[p] for p in o["rank"][:k]}) / len(g) for g, o in zip(gt, items)])
        cc = 100 * np.mean([float(g <= {C.titles[p] for p in o["pas"]}) for g, o in zip(gt, items)])
        reach = 100 * np.mean([float(g <= {C.titles[p] for p in w_["vis"]})
                               for g, w_ in zip(gt, W)])
        tag = f"{tag0}_SB{B}"
        json.dump({"method": tag, "R@2": rk(2), "R@5": rk(5), "ChainCtx": cc, "Reach": reach,
                   "latency_s": float(np.mean([o["lat"] for o in W])), "items": items}, open(f"{ROOT}/results/{ds}/{tag}.json", "w"))
        print(f"{ds} {tag:28s} R@2 {rk(2):.1f} R@5 {rk(5):.1f} Chain {cc:.1f} Reach {reach:.1f} Sel {100 * cc / max(reach, 1e-9):.1f} "
              f"avgP {np.mean([len(o['pas']) for o in items]):.2f}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "mine":
        os.makedirs(f"{ROOT}/ckpt/cw_mine", exist_ok=True); mine(a[1], a[2], int(a[3]), int(a[4]))
    elif a[0] == "train":
        train(a[1], a[2], int(a[3]))
    elif a[0] == "infer":
        infer(a[1], a[2], a[3], a[4])
    else:
        pack(a[1], a[2], a[3])
