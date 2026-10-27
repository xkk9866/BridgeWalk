"""Collect ChainCoR results + paired bootstrap tests. Writes results/chain_summary.json"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ["musique", "2wikimultihopqa", "hotpotqa"]
rng = np.random.default_rng(0)


def load(ds, m, res="results"):
    f = f"{ROOT}/{res}/{ds}/{m}.qa.json"
    if not os.path.exists(f):
        return None
    Q = json.load(open(f)); R = json.load(open(f"{ROOT}/{res}/{ds}/{m}.json"))
    return {"sum": Q["summary"], "em": np.array(Q["em"]), "f1": np.array(Q["f1"]), "R@2": R.get("R@2"), "R@5": R.get("R@5")}


def paired(a, b, B=10000):
    d = a - b; idx = rng.integers(0, len(d), (B, len(d))); bs = d[idx].mean(1)
    return {"diff": round(100 * float(d.mean()), 2), "ci95": [round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)],
            "p": float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))}


def main():
    out = {"qwen": {}, "llama": {}, "tests": {}}
    base = ["bm25", "dense", "kg_only", "fixed_fusion", "rrf", "hipporag", "hipporag2"]
    budget = [f"{m}_{s}B{B}" for B in (300, 450, 600, 800) for m, s in
              [("chainP_s0", ""), ("chainP_s0", "f"), ("gnn_s0", ""), ("hipporag2", "eq"), ("dense", "eq"), ("rerankZS", "")]]
    extra = [f"{m}_B{B}" for B in (450, 600) for m in ["chainP_s1", "chainP_s1_f".replace("_f", "_f"), "chainNB_s0", "point_s0", "gnn_s1", "chainDP_s0"]]
    extra = [e.replace("chainP_s1_f_B", "chainP_s1_fB") for e in extra]
    for ds in DS:
        out["qwen"][ds] = {}
        for m in base + budget + extra:
            r = load(ds, m)
            if r:
                s = r["sum"]
                out["qwen"][ds][m] = {k: (round(s[k], 2) if isinstance(s.get(k), float) else s.get(k)) for k in
                                      ["EM", "F1", "tokens", "ChainCtx", "CorrectButInsufficient", "Trace", "RelCov", "R@2", "R@5", "latency_ms", "avgP", "avgF", "by_type"]}
        out["llama"][ds] = {}
        for m in ["chainP_s0_B600", "chainP_s0_fB600", "gnn_s0_B600", "hipporag2_eqB600", "dense_eqB600", "rerankZS_B600"]:
            r = load(ds, m, "results_llama")
            if r:
                out["llama"][ds][m] = {k: round(r["sum"][k], 2) for k in ["EM", "F1", "tokens"]}
        out["tests"][ds] = {}
        for B in (300, 450, 600, 800):
            ref = load(ds, f"chainP_s0_B{B}")
            if not ref:
                continue
            for o in [f"gnn_s0_B{B}", f"hipporag2_eqB{B}", f"dense_eqB{B}", f"rerankZS_B{B}", f"point_s0_B{B}", f"chainNB_s0_B{B}", f"chainDP_s0_B{B}", "hipporag2", "dense"]:
                r = load(ds, o)
                if r:
                    out["tests"][ds][f"B{B}|{o}"] = {"EM": paired(ref["em"], r["em"]), "F1": paired(ref["f1"], r["f1"])}
    json.dump(out, open(f"{ROOT}/results/chain_summary.json", "w"), indent=1)
    print("ok")


if __name__ == "__main__":
    main()
