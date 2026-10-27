"""Re-pack existing rankings at standard (larger) context sizes: strict budgets 1000/2000 reader tokens and top-5.
Rankings shorter than needed are continued with the same dense ranking for every method (identical fill rule).
python repack.py <ds> <method> [<method> ...]     (reads results/<ds>/<method>_SB450.json)"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from index import embed_queries
import budget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ds, methods = sys.argv[1], sys.argv[2:]
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    cache = f"{ROOT}/results/{ds}/dense_top60.json"
    if os.path.exists(cache):
        fill = json.load(open(cache))
    else:
        Q = embed_queries([q["question"] for q in qs]).cpu().float()
        fill = (Q @ C.P.float().t()).topk(60, dim=1).indices.tolist()
        json.dump(fill, open(cache, "w"))
    for m in methods:
        R = json.load(open(f"{ROOT}/results/{ds}/{m}_SB450.json"))
        for tag, B in [("SB1000", 1000), ("SB2000", 2000), ("top5", None)]:
            items = []
            for it, fl in zip(R["items"], fill):
                rank = list(dict.fromkeys(list(it["rank"]) + fl))
                pas = rank[:5] if B is None else budget.pack(C, rank, B)[0]
                items.append({k: v for k, v in it.items() if k not in ("pas", "rank")} | {"rank": rank, "pas": pas, "facts": []})
            out = {k: v for k, v in R.items() if k != "items"} | {"method": f"{m}_{tag}", "items": items}
            json.dump(out, open(f"{ROOT}/results/{ds}/{m}_{tag}.json", "w"))
        print(ds, m, "repacked", flush=True)


if __name__ == "__main__":
    main()
