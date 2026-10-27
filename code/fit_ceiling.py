"""How much of the reader budget does the gold chain itself need?  For every test question we render only its gold
passages, exactly as the reader sees them, and count reader tokens.  Prints the fraction of chains that fit in B."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
import budget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for ds in sys.argv[1:] or ["musique", "2wikimultihopqa", "hotpotqa", "mhrag"]:
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    print(ds, "fields", sorted(qs[0].keys()))
    need, plen = [], []
    for q in qs:
        gp = []
        for t in q["gold_titles"]:
            i = C.title2i.get(t)
            if i is not None and i not in gp:
                gp.append(i)
        need.append(budget.ntok(budget.render(C, gp, [])))
        plen += [budget.ntok(budget.render(C, [p], [])) for p in gp]
    need = np.array(need)
    print(ds, f"passage tokens mean {np.mean(plen):.0f} median {np.median(plen):.0f}; gold-chain tokens mean {need.mean():.0f}",
          " ".join(f"fit@{B} {100 * np.mean(need <= B):.1f}" for B in (300, 450, 600, 800)), flush=True)
