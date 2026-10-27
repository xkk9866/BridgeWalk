"""StructChain inference + budget packing that maximises the expected probability that the reader sees the whole chain:
      J(X) = sum_S p(S|q) * 1[S subset of X]   subject to  tokens(X) <= B   (reader tokenizer, strict)
Greedy cost-benefit over the model's top chains, then marginal-ordered filling; the final context is verified by budget.pack.
python struct_infer.py <ds> <model_dir> <name> [cand_suffix]"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from struct_chain import load
import budget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS = [300, 450, 600, 800]


def plan_order(C, pids, res, B):
    cost = {p: budget.ntok(f"[P1] {C.titles[p]}: {C.texts[p]}") + 1 for p in pids}
    sets = [([pids[k] for k in S], pr) for S, pr in res["top_sets"]]
    X, used, order = set(), 0, []
    while True:
        best, bval = None, 0.0
        for S, _ in sets:
            add = [p for p in S if p not in X]
            if not add:
                continue
            c = sum(cost[p] for p in add)
            if used + c > B:
                continue
            gain = sum(pr for S2, pr in sets if set(S2) <= X | set(add) and not set(S2) <= X)
            if gain / c > bval:
                best, bval = add, gain / c
        if best is None:
            break
        for p in best:
            X.add(p); order.append(p); used += cost[p]
    rest = sorted([k for k in range(len(pids)) if pids[k] not in X], key=lambda k: -res["marg"][k])
    return order + [pids[k] for k in rest]


def main():
    ds, mdir, name = sys.argv[1:4]
    suf = sys.argv[4] if len(sys.argv) > 4 else "_v6"
    m = load(mdir)
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    rows = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand{suf}.jsonl", encoding="utf-8")]
    res_all, t0 = [], time.time()
    for r in rows:
        res_all.append(m.infer(r))
    lat = (time.time() - t0) / len(rows)
    for B in BUDGETS:
        out = []
        for q, r, res in zip(qs, rows, res_all):
            pids = [c["pid"] for c in r["cands"]]
            rank = [pids[k] for k in res["map"]] + [pids[k] for k in np.argsort(-np.array(res["marg"])) if k not in res["map"]]
            order = plan_order(C, pids, res, B)
            pas, _ = budget.pack(C, order, B)
            out.append({"qid": q["qid"], "rank": rank, "pas": pas, "facts": [], "sel_p": [pids[k] for k in res["map"]],
                        "marg": res["marg"]})
        r2 = np.mean([len(set(q["gold_titles"]) & {C.titles[p] for p in o["rank"][:2]}) / len(q["gold_titles"]) for q, o in zip(qs, out)])
        r5 = np.mean([len(set(q["gold_titles"]) & {C.titles[p] for p in o["rank"][:5]}) / len(q["gold_titles"]) for q, o in zip(qs, out)])
        cc = np.mean([float(set(q["gold_titles"]) <= {C.titles[p] for p in o["pas"]}) for q, o in zip(qs, out)])
        tag = f"{name}_SB{B}"
        json.dump({"method": tag, "R@2": 100 * r2, "R@5": 100 * r5, "ChainCtx": 100 * cc, "latency_s": lat, "items": out},
                  open(f"{ROOT}/results/{ds}/{tag}.json", "w"))
        print(f"{ds} {tag:26s} R@2 {100*r2:.1f} R@5 {100*r5:.1f} ChainCtx {100*cc:.1f} avgP {np.mean([len(o['pas']) for o in out]):.2f} lat {1000*lat:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
