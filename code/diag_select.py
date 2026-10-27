"""Why is a reached chain not selected? MAP-chain correctness, length vs gold hops, and the rank depth at which the
gold chain completes."""
import json, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ds, m = sys.argv[1], sys.argv[2]
T = torch.load(f"{ROOT}/data/{ds}/index.pt", weights_only=False)["titles"]
qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
R = json.load(open(f"{ROOT}/results/{ds}/{m}_SB450.json"))["items"]
hops = np.array([len(set(q["gold_titles"])) for q in qs])
L = np.array([len(o["chain"]) for o in R])
exact, sub, sup, depth, ncov = [], [], [], [], []
for q, o in zip(qs, R):
    g = set(q["gold_titles"]); ch = {T[p] for p in o["chain"]}
    exact.append(ch == g); sub.append(ch <= g); sup.append(g <= ch)
    seen = set(); d = None
    for i, p in enumerate(o["rank"]):
        seen.add(T[p])
        if g <= seen: d = i + 1; break
    depth.append(d)
print(ds, m, "MAP exact %.1f  MAP subset-of-gold %.1f  MAP covers gold %.1f" % (100*np.mean(exact), 100*np.mean(sub), 100*np.mean(sup)))
for h in sorted(set(hops)):
    k = hops == h
    print(f"  {h}-hop n={k.sum()} MAP len mean {L[k].mean():.2f} | len dist", np.bincount(L[k], minlength=5)[1:].tolist(),
          "| MAP covers gold %.1f" % (100*np.mean(np.array(sup)[k])))
d = np.array([x if x else 999 for x in depth])
print("  completion depth: <=hops %.1f  <=4 %.1f  <=6 %.1f  <=8 %.1f  <=12 %.1f  reached %.1f" % tuple(100*np.mean(c) for c in [d <= hops, d <= 4, d <= 6, d <= 8, d <= 12, d < 999]))
print("  mean passages in context @450", np.mean([len(o["pas"]) for o in R]).round(2))
