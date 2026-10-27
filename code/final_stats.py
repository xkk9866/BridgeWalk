"""Final statistics under strict reader-token budgets (SB*). Writes results/final_summary.json"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ["musique", "2wikimultihopqa", "hotpotqa", "mhrag"]
BUD = [300, 450, 600, 800]
rng = np.random.default_rng(0)
REF = os.environ.get("REF", "bridge_full")
BASE = ["dense", "hipporag2", "rerankZS", "gnn", "point", "chainDP", "chainPNB6", "structF_s0", "structF_s1", "structNG_s0",
        "structNM_s0", "structNP_s0", "structF_s0nosyn", "chainS_s0", "chainS_s1", "chainS_s0_prop", "chainNS", "mdr", "chainKG", "mdrT", "hybrid", "joint_s0_qret", "joint_s1_qret", "joint_s0_mst", "joint_s0_greedy", "joint_s0_point", "joint_s0_mmr", "joint_s0_cover",
        "walk_full", "walk_nograph", "walk_depth1", "walks1_full",
        "bridge_full", "bridge_nostop", "bridge_qbridge", "bridge_allment", "bridge_oie", "bridge_nograph", "bridges1_full", "bridge_dense0", "bridge_K1", "bridge_K3", "bridge_R4", "bridge_R16"]


def load(ds, m, res="results"):
    f = f"{ROOT}/{res}/{ds}/{m}.qa.json"
    if not os.path.exists(f):
        return None
    Q = json.load(open(f)); R = json.load(open(f"{ROOT}/{res}/{ds}/{m}.json"))
    summ = dict(Q["summary"]); summ["R@2"], summ["R@5"] = R.get("R@2"), R.get("R@5")   # recall from the (re-packed) ranking file
    return {"sum": summ, "em": np.array(Q["em"]), "f1": np.array(Q["f1"]), "R@2": R.get("R@2"), "R@5": R.get("R@5")}


def paired(a, b, B=10000):
    d = a - b; idx = rng.integers(0, len(d), (B, len(d))); bs = d[idx].mean(1)
    return {"diff": round(100 * float(d.mean()), 2), "ci95": [round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)],
            "p": round(float(2 * min((bs <= 0).mean(), (bs >= 0).mean())), 4)}


def main():
    out = {"qwen": {}, "llama": {}, "tests": {}}
    for ds in DS:
        out["qwen"][ds] = {}; out["llama"][ds] = {}; out["tests"][ds] = {}
        for m in BASE:
            for B in BUD:
                name = f"{m}_SB{B}"
                r = load(ds, name)
                if r:
                    s = r["sum"]
                    out["qwen"][ds][name] = {k: s.get(k) for k in ["EM", "F1", "tokens", "ChainCtx", "CorrectButInsufficient", "R@2", "R@5", "by_type"]}
                r = load(ds, name, "results_llama")
                if r:
                    out["llama"][ds][name] = {k: r["sum"].get(k) for k in ["EM", "F1", "tokens"]}
        for B in BUD:
            ref = load(ds, f"{REF}_SB{B}")
            if not ref:
                continue
            for o in BASE:
                if o == REF:
                    continue
                r = load(ds, f"{o}_SB{B}")
                if r:
                    out["tests"][ds][f"B{B}|{o}"] = {"EM": paired(ref["em"], r["em"]), "F1": paired(ref["f1"], r["f1"])}
        for res in ["results_llama"]:
            ref = load(ds, f"{REF}_SB450", res)
            if ref:
                for o in BASE:
                    r = load(ds, f"{o}_SB450", res)
                    if r and o != REF:
                        out["tests"][ds][f"llama450|{o}"] = {"EM": paired(ref["em"], r["em"])}
    json.dump(out, open(f"{ROOT}/results/final_summary.json", "w"), indent=1)
    print("ok")


if __name__ == "__main__":
    main()
