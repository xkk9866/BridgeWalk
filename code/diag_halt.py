"""Halting diagnosis from a DUMP run: is early stopping caused by (a) halting on correct-but-incomplete prefixes, or
(b) halting on wrong passages (the halting read-out never saw negatives in training)?"""
import json, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ds, name = sys.argv[1], sys.argv[2]
T = torch.load(f"{ROOT}/data/{ds}/index.pt", weights_only=False)["titles"]
qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
R = json.load(open(f"{ROOT}/results/{ds}/{name}_SB450.json"))["items"]
cat = {"gold-complete": [], "gold-incomplete": [], "wrong": []}
top_kind = {k: 0 for k in cat}
for q, o in zip(qs, R):
    g = set(q["gold_titles"])
    for j, f in enumerate(o["finals"]):
        t = [T[p] for p in f["ch"]]
        kind = "wrong" if not set(t) <= g else ("gold-complete" if set(t) == g else "gold-incomplete")
        cat[kind].append(f["h"])
        if j == 0:
            top_kind[kind] += 1
n = len(R)
print(ds, name, "MAP chain kind:", {k: round(100 * v / n, 1) for k, v in top_kind.items()})
for k, v in cat.items():
    v = np.array(v)
    print(f"  h on {k:16s} n={len(v):5d} mean {v.mean():.3f}  frac>0.5 {100*(v > 0.5).mean():.1f}")
