"""Retrieval statistics for the ChainWalk paper: R@2, R@5, chain-in-context, reachability (chain anywhere in the ranked
list), and selection = chain-in-context / reachability. Writes results/walk_ret.json"""
import json, os, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = ["dense", "mdrT", "hybrid", "hipporag2", "rerankZS", "point", "chainS_s0", "walk_depth1", "walk_nograph", "walk_full", "walks1_full", "bridge_full", "bridge_nostop", "bridge_qbridge", "bridge_allment", "bridge_oie", "bridge_nograph", "bridges1_full", "bridge_dense0", "bridge_K1", "bridge_K3", "bridge_R4", "bridge_R16"]
out = {}
for ds in ["musique", "2wikimultihopqa", "hotpotqa"]:
    T = torch.load(f"{ROOT}/data/{ds}/index.pt", weights_only=False)["titles"]
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    out[ds] = {}
    for m in M:
        for B in (300, 450, 600):
            f = f"{ROOT}/results/{ds}/{m}_SB{B}.json"
            if not os.path.exists(f):
                continue
            R = json.load(open(f))["items"]; g = [set(q["gold_titles"]) for q in qs]
            r = lambda k: 100 * sum(len(gi & {T[p] for p in o["rank"][:k]}) / len(gi) for gi, o in zip(g, R)) / len(R)
            cc = 100 * sum(gi <= {T[p] for p in o["pas"]} for gi, o in zip(g, R)) / len(R)
            reach = 100 * sum(gi <= {T[p] for p in o["rank"]} for gi, o in zip(g, R)) / len(R)
            out[ds][f"{m}_SB{B}"] = {"R@2": r(2), "R@5": r(5), "Chain": cc, "Reach": reach, "Sel": 100 * cc / max(reach, 1e-9)}
    print(ds, flush=True)
json.dump(out, open(f"{ROOT}/results/walk_ret.json", "w"), indent=1)
