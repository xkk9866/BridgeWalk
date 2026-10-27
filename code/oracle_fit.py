"""Upper bound of chain-in-context: does the gold chain alone (its passages, rendered as the reader sees them) fit B?"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
import budget
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for ds in sys.argv[1:]:
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    need = []
    for q in qs:
        g = []
        for t in dict.fromkeys(q["gold_titles"]):
            if t in C.title2i: g.append(C.title2i[t])
        need.append(budget.ntok(budget.render(C, g, [])))
    need = np.array(need)
    print(ds, "gold-chain tokens median %.0f" % np.median(need), " fits: " + "  ".join(f"B{B} {100*(need <= B).mean():.1f}" for B in (300, 450, 600, 800, 1000)), flush=True)
