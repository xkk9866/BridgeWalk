"""Graph ceiling under different passage-neighbourhood definitions (title level, oracle ordering)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from chainwalk import neighbours
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ds = sys.argv[1]
C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
rows = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand_v7.jsonl", encoding="utf-8")]
def nb_unc(p):
    out = set(C.subj_out[p]); te = C.title_e[p]
    if te is not None: out.update(C.e_pass[te])
    for f in C.p_facts[p]:
        for e in C.f_args[f]:
            for e2 in [e] + C.e_syn[e]: out.update(C.e_pass[e2])
    out.discard(p); return out
def nb_ment(p):
    out = nb_unc(p)
    for e in C.p_ents[p]:
        for e2 in [e] + C.e_syn[e]: out.update(C.e_pass[e2])
    out.discard(p); return out
defs = {"capped": lambda p: neighbours(C, p), "uncapped": nb_unc, "mentions": nb_ment}
for name, nb in defs.items():
    cache = {}
    def N(p):
        if p not in cache: cache[p] = nb(p)
        return cache[p]
    ok, hop, sz = [], [], []
    for q, r in zip(qs, rows):
        gold = set(q["gold_titles"]); base = [c["pid"] for c in r["cands"]]
        have = {C.titles[p] for p in base} & gold; pre = [p for p in base if C.titles[p] in gold]
        grew = True
        while grew and have != gold:
            grew = False
            for p in list(pre):
                for b in N(p):
                    if C.titles[b] in gold and C.titles[b] not in have: have.add(C.titles[b]); pre.append(b); grew = True
        ok.append(have == gold); hop.append(len(gold)); sz += [len(N(p)) for p in base[:2]]
    ok, hop = np.array(ok), np.array(hop)
    print(ds, f"{name:9s} ceiling {100*ok.mean():.1f}", " ".join(f"{k}h {100*ok[hop==k].mean():.1f}" for k in sorted(set(hop))), f"| |N| median {np.median(sz):.0f} mean {np.mean(sz):.0f} p90 {np.percentile(sz,90):.0f}", flush=True)
