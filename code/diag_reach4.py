"""Ceiling of (a) text-mention edges (entity string occurs in passage text) and (b) prefix-conditioned dense
expansion (query = question + prefix passage, top-k), on top of the capped evidence graph."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from chainwalk import neighbours
from index import embed_queries, norm_ent
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ds = sys.argv[1]; K = 20
C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
rows = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand_v7.jsonl", encoding="utf-8")]
eidx = {e: i for i, e in enumerate(C.ents)}
tm = [set() for _ in C.ents]          # entity -> passages whose text contains it
for p, t in enumerate(C.texts):
    w = norm_ent(C.titles[p] + " " + t).split()
    for n in range(1, 7):
        for i in range(len(w) - n + 1):
            e = eidx.get(" ".join(w[i:i + n]))
            if e is not None: tm[e].add(p)
deg = np.array([len(s) for s in tm]); print("text-mention degree median", np.median(deg[deg > 0]), "mean", deg.mean())
pents = [set(C.p_ents[p]) | {e for f in C.p_facts[p] for e in C.f_args[f]} for p in range(C.nP)]
def nb_tm(p, cap=10**9):
    out = set(neighbours(C, p))
    for e in pents[p]:
        if len(tm[e]) <= cap: out |= tm[e]
    out.discard(p); return out
P = C.P.float()
def run(N):
    ok, hop = [], []
    for qi, (q, r) in enumerate(zip(qs, rows)):
        gold = set(q["gold_titles"]); base = [c["pid"] for c in r["cands"]]
        have = {C.titles[p] for p in base} & gold; pre = [p for p in base if C.titles[p] in gold]
        grew = True
        while grew and have != gold:
            grew = False
            for p in list(pre):
                for b in N(qi, p):
                    if C.titles[b] in gold and C.titles[b] not in have: have.add(C.titles[b]); pre.append(b); grew = True
        ok.append(have == gold); hop.append(len(gold))
    ok, hop = np.array(ok), np.array(hop)
    return f"{100*ok.mean():.1f} " + " ".join(f"{k}h {100*ok[hop==k].mean():.1f}" for k in sorted(set(hop)))
print(ds, "graph      ", run(lambda qi, p: neighbours(C, p)))
for cap in [50, 200]:
    cache = {}
    def N(qi, p, cap=cap):
        if p not in cache: cache[p] = nb_tm(p, cap)
        return cache[p]
    print(ds, f"+textment<= {cap}", run(N), "| |N| mean", np.mean([len(v) for v in cache.values()]).round(0), flush=True)
dcache = {}
def ND(qi, p):
    k = (qi, p)
    if k not in dcache:
        e = embed_queries([qs[qi]["question"] + " " + C.titles[p] + ": " + C.texts[p][:400]]).cpu().float()[0]
        dcache[k] = set((P @ e).topk(K).indices.tolist())
    return dcache[k] | neighbours(C, p)
print(ds, f"+dense(q+p) top{K}", run(ND), flush=True)
