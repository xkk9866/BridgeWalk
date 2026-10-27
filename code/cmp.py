"""Chain-in-context / reach / R@2 on the first N test questions for a list of result files.
python cmp.py <ds> <N> name1 name2 ..."""
import json, sys
ds, N, names = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
qs = [json.loads(l) for l in open(f"data/{ds}/test.jsonl", encoding="utf-8")][:N]
import torch
d = torch.load(f"data/{ds}/index.pt", weights_only=False); T = d["titles"]
for n in names:
    R = json.load(open(f"results/{ds}/{n}.json"))["items"][:N]
    g = [set(q["gold_titles"]) for q in qs]
    cc = sum(gi <= {T[p] for p in o["pas"]} for gi, o in zip(g, R)) / N
    r2 = sum(len(gi & {T[p] for p in o["rank"][:2]}) / len(gi) for gi, o in zip(g, R)) / N
    rc = sum(gi <= {T[p] for p in o["rank"]} for gi, o in zip(g, R)) / N
    print(f"{n:30s} R@2 {100*r2:.1f} Chain {100*cc:.1f} Reach {100*rc:.1f} avgP {sum(len(o['pas']) for o in R)/N:.2f}")
