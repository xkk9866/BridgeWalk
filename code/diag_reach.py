"""Diagnosis of ChainWalk reachability: is the bottleneck the graph itself or the query-similarity cut on neighbours?
For each test question: (i) oracle graph-connectivity of the gold chain from C0 with uncapped neighbourhoods,
(ii) the same with the dense top-(NB*t) cut, (iii) neighbourhood sizes, (iv) rank of the gold continuation under
q-similarity among the neighbours, (v) number of distinct bridge entities per prefix."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from index import embed_queries
from chainwalk import neighbours
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def connected(C, base, gold, nbf):
    # BFS over gold orderings: grow prefix with any gold passage admissible from current prefix
    pre, rem = [], set(gold)
    adm = set(base) & rem
    if not adm: return False, 0
    while rem:
        cand = (set(base) | nbf(pre)) & rem
        if not cand: return False, len(pre)
        x = min(cand); pre.append(x); rem.discard(x)
    return True, len(pre)

for ds in sys.argv[1:]:
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    rows = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand_v7.jsonl", encoding="utf-8")]
    Q = embed_queries([q["question"] for q in qs]).cpu(); P = C.P.float()
    NBc = {}
    def nb(p):
        if p not in NBc: NBc[p] = neighbours(C, p)
        return NBc[p]
    st = {"C0": [], "uncapped": [], "cut16": [], "nbsize": [], "goldrank": [], "ents_per_p": [], "hops": []}
    for q, r, qe in zip(qs, rows, Q):
        gold = {C.title2i[t] for t in q["gold_titles"] if t in C.title2i}
        base = [c["pid"] for c in r["cands"]]
        sq = (P @ qe.float()).numpy()
        st["hops"].append(len(gold)); st["C0"].append(gold <= set(base))
        un = lambda pre: set().union(*(nb(p) for p in pre)) if pre else set()
        def cut(pre):
            if not pre: return set()
            s = [c for c in un(pre) if c not in base and c not in pre]; s.sort(key=lambda c: -sq[c]); return set(s[:16 * len(pre)])
        st["uncapped"].append(connected(C, base, gold, un)[0]); st["cut16"].append(connected(C, base, gold, cut)[0])
        for g in gold:
            for a in gold - {g}:
                if g in nb(a) and g not in base:
                    s = sorted([c for c in nb(a) if c not in base], key=lambda c: -sq[c])
                    st["goldrank"].append(s.index(g)); st["nbsize"].append(len(nb(a)))
        st["ents_per_p"] += [len({e for f in C.p_facts[p] for e in C.f_args[f]}) for p in base[:3]]
    h = np.array(st["hops"])
    print(ds, "C0 %.1f uncapped %.1f cut16 %.1f" % tuple(100 * np.mean(st[k]) for k in ["C0", "uncapped", "cut16"]))
    for k in sorted(set(h)):
        msk = h == k
        print(f"  {k}hop n={msk.sum()} C0 {100*np.mean(np.array(st['C0'])[msk]):.1f} unc {100*np.mean(np.array(st['uncapped'])[msk]):.1f} cut {100*np.mean(np.array(st['cut16'])[msk]):.1f}")
    gr = np.array(st["goldrank"]); print("  nb size median %.0f mean %.0f | gold rank median %.0f, <16 %.1f%%, <48 %.1f%% | ents/passage %.1f" %
        (np.median(st["nbsize"]), np.mean(st["nbsize"]), np.median(gr), 100*np.mean(gr < 16), 100*np.mean(gr < 48), np.mean(st["ents_per_p"])), flush=True)
