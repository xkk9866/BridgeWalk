"""Where does a gold next hop sit inside the passage list of the entity that leads to it?
For every gold transition a->b (b reachable from a through an entity e mentioned in a, b in M(e)), report the
rank of b in M(e) under several orderings, taking the best e (oracle pointer)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from bridgewalk import MentionGraph, ptxt
from index import embed_queries
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ds = sys.argv[1]
C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
G = MentionGraph(C, f"{ROOT}/data/{ds}/text_mentions.pt")
qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")][:int(os.environ.get("LIMIT", 1000))]
Q = embed_queries([q["question"] for q in qs]).cpu().float(); P = C.P.float()
subj = {}
for p, e in enumerate(C.title_e):
    if e is not None: subj.setdefault(e, set()).add(p)
res = {k: [] for k in ["q", "subj+q", "parent", "q+parent", "deg", "global"]}
nolink = 0; ntr = 0
for q, qe in zip(qs, Q):
    gold = [C.title2i[t] for t in q["gold_titles"] if t in C.title2i]
    sq = (P @ qe).numpy()
    for a in gold:
        for b in gold:
            if a == b: continue
            sa = G.spans(ptxt(C, a, 900))
            es = [e for e in sa if b in G.M[e] or any(b in G.M[e2] for e2 in C.e_syn[e][:3])]
            if not es: continue
            ntr += 1
            sp = (P @ P[a]).numpy()
            best = {k: 10**9 for k in res}
            for e in es:
                M = set(G.M[e])
                for e2 in C.e_syn[e][:3]: M |= set(G.M[e2])
                M.discard(a); M = list(M)
                S = subj.get(e, set())
                orders = {"q": sorted(M, key=lambda p: -sq[p]),
                          "subj+q": sorted(M, key=lambda p: (p not in S, -sq[p])),
                          "parent": sorted(M, key=lambda p: -sp[p]),
                          "q+parent": sorted(M, key=lambda p: -(sq[p] + sp[p]))}
                for k, o in orders.items(): best[k] = min(best[k], o.index(b))
                best["deg"] = min(best["deg"], len(M))
            best["global"] = int((sq > sq[b]).sum())
            for k in res: res[k].append(best[k])
print(ds, "transitions with a link", ntr)
for k in ["global", "q", "subj+q", "parent", "q+parent"]:
    r = np.array(res[k]); print(f"  {k:9s} R@1 {100*(r<1).mean():.1f} R@4 {100*(r<4).mean():.1f} R@8 {100*(r<8).mean():.1f} R@16 {100*(r<16).mean():.1f} R@32 {100*(r<32).mean():.1f}")
d = np.array(res["deg"]); print("  min |M(e)| median", np.median(d), "mean", d.mean().round(1), ">8:", (100*(d > 8).mean()).round(1))
