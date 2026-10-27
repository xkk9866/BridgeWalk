"""Title-level reachability ceiling of the static evidence graph, and anatomy of the unreachable chains."""
import json, os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from chainwalk import neighbours
from index import norm_ent
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ds = sys.argv[1]
C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
rows = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand_v7.jsonl", encoding="utf-8")]
W = {r: json.load(open(f"{ROOT}/results/{ds}/walk_full_SB450.json"))["items"] for r in [0]}[0]
t2p = {}
for i, t in enumerate(C.titles): t2p.setdefault(t, []).append(i)
res, fails = [], []
kinds = {"title_in_text": 0, "title_ent_missing": 0, "other": 0}
for q, r, w in zip(qs, rows, W):
    gold = set(q["gold_titles"]); base = [c["pid"] for c in r["cands"]]
    have = {C.titles[p] for p in base} & gold
    pre = [p for p in base if C.titles[p] in gold]
    grew = True
    while grew and have != gold:
        grew = False
        for p in list(pre):
            for b in neighbours(C, p):
                if C.titles[b] in gold and C.titles[b] not in have:
                    have.add(C.titles[b]); pre.append(b); grew = True
    ok = have == gold; res.append((ok, len(gold), gold <= {C.titles[p] for p in w["rank"]}))
    if not ok and have:
        for miss in gold - have:
            txt = " ".join(C.texts[p] for p in pre).lower()
            if miss.lower() in txt: kinds["title_in_text"] += 1
            elif C.title_e[t2p[miss][0]] is None: kinds["title_ent_missing"] += 1
            else: kinds["other"] += 1
            fails.append((q["question"], [C.titles[p] for p in pre], miss, C.texts[t2p[miss][0]][:150]))
res = np.array(res, dtype=float)
print(ds, "graph-ceiling %.1f | walk reach %.1f | ceiling&!walk %.1f" % (100*res[:,0].mean(), 100*res[:,2].mean(), 100*np.mean(res[:,0]*(1-res[:,2]))))
for k in sorted(set(res[:,1])): m = res[:,1]==k; print(f"  {int(k)}hop ceil {100*res[m,0].mean():.1f} walk {100*res[m,2].mean():.1f}")
print("  missing-hop anatomy (prefix partially reached):", kinds)
random.seed(0)
for f in random.sample(fails, min(6, len(fails))): print("   Q:", f[0], "| have:", f[1][:3], "| miss:", f[2], "|", f[3])
