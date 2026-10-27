"""Aggregate revision results: seed mean/std, per-question paired bootstrap tests, Pareto data.
Writes results/revision_summary.json"""
import json, os, glob, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ["musique", "2wikimultihopqa", "hotpotqa"]
rng = np.random.default_rng(0)


def titles(ds):
    import torch
    return torch.load(f"{ROOT}/data/{ds}/index.pt", weights_only=False)["titles"]


def perq(ds, name, T, qs):
    """per-question retrieval metrics from a results json"""
    f = f"{ROOT}/results/{ds}/{name}.json"
    if not os.path.exists(f):
        return None
    R = json.load(open(f)); out = {"R2": [], "R5": [], "ChainSel": [], "ChainCtx": [], "P": [], "F": []}
    for q, o in zip(qs, R["items"]):
        gold = set(q["gold_titles"])
        out["R2"].append(len(gold & {T[p] for p in o["rank"][:2]}) / len(gold))
        out["R5"].append(len(gold & {T[p] for p in o["rank"][:5]}) / len(gold))
        out["ChainSel"].append(float(gold <= {T[p] for p in o.get("sel_p", o["pas"])}))
        out["ChainCtx"].append(float(gold <= {T[p] for p in o["pas"]}))
        out["P"].append(len(o["pas"])); out["F"].append(len(o["facts"]))
    qa = f"{ROOT}/results/{ds}/{name}.qa.json"
    if os.path.exists(qa):
        Q = json.load(open(qa)); out["EM"] = Q["em"]; out["F1"] = Q["f1"]; out["summary"] = Q["summary"]
    return {k: (np.array(v, dtype=float) if isinstance(v, list) else v) for k, v in out.items()}


def seedavg(ds, base, seeds, T, qs):
    runs = [perq(ds, base.format(s=s), T, qs) for s in seeds]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None, None
    keys = [k for k in runs[0] if isinstance(runs[0][k], np.ndarray) and all(k in r for r in runs)]
    stats = {k: (100 * float(np.mean([r[k].mean() for r in runs])), 100 * float(np.std([r[k].mean() for r in runs])))
             if k not in ("P", "F") else (float(np.mean([r[k].mean() for r in runs])), 0.0) for k in keys}
    if all("summary" in r for r in runs):
        for k in ["tokens", "Trace", "RelCov", "CorrectButInsufficient"]:
            v = [r["summary"].get(k) for r in runs if r["summary"].get(k) is not None]
            if v:
                stats[k] = (float(np.mean(v)), float(np.std(v)))
    stats["n_seeds"] = len(runs)
    mean_pq = {k: np.mean([r[k] for r in runs], 0) for k in keys}
    return stats, mean_pq


def paired(a, b, B=10000):
    d = a - b; n = len(d)
    idx = rng.integers(0, n, (B, n)); bs = d[idx].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"diff": 100 * float(d.mean()), "ci95": [100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))], "p": float(p)}


def main():
    S = {"seed": {}, "tests": {}, "stop": {}, "inject": {}, "np": {}, "baselines": {}, "nsample": {}}
    for ds in DS:
        T = titles(ds); qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
        five, three = range(5), range(3)
        groups = {"unicor3": ("unicor3_s{s}", five), "unicor3_fill": ("unicor3_s{s}_fill", five),
                  "u3nr": ("u3nr_s{s}", five), "gnn": ("gnn_s{s}", five), "gnn_fill": ("gnn_s{s}_fill", five),
                  "u3nodiff": ("u3nodiff_s{s}", three), "u3noctrl": ("u3noctrl_s{s}", three), "u3noprov": ("u3noprov_s{s}", three),
                  "u3tf": ("u3tf_s{s}", three), "u3noconf": ("u3noconf_s{s}", three), "u3soft": ("u3soft_s{s}", three),
                  "pointer": ("pointer_s{s}", three), "unicor2_v2": ("unicor2", [0]),
                  "u3n300": ("u3n300_s{s}", [0, 1]), "u3n1000": ("u3n1000_s{s}", [0, 1]),
                  "gnnn300": ("gnnn300_s{s}", [0, 1]), "gnnn1000": ("gnnn1000_s{s}", [0, 1])}
        pq = {}
        S["seed"][ds] = {}
        for g, (pat, seeds) in groups.items():
            st, m = seedavg(ds, pat, seeds, T, qs)
            if st:
                S["seed"][ds][g] = st; pq[g] = m
        for b in ["bm25", "dense", "kg_only", "fixed_fusion", "rrf", "hipporag", "hipporag2", "dense_ft"]:
            st, m = seedavg(ds, b, [0], T, qs)
            if st:
                S["baselines"].setdefault(ds, {})[b] = st; pq[b] = m
        # paired tests: full model vs each comparator, per-question, seed-averaged
        S["tests"][ds] = {}
        for ref, others, metrics in [("unicor3", ["u3nr", "u3nodiff", "u3noctrl", "u3noprov", "u3tf", "u3noconf", "u3soft", "pointer", "gnn"], ["R2", "R5", "ChainSel"]),
                                     ("unicor3_fill", ["gnn_fill", "hipporag2", "dense", "dense_ft"], ["EM", "F1", "ChainCtx"]),
                                     ("unicor3", ["u3nr", "u3nodiff", "u3noctrl", "u3tf", "u3soft", "pointer"], ["EM", "F1"])]:
            for o in others:
                if ref in pq and o in pq:
                    for k in metrics:
                        if k in pq[ref] and k in pq[o]:
                            S["tests"][ds][f"{ref}|{o}|{k}"] = paired(pq[ref][k], pq[o][k])
        for tag in ["suff0.3", "suff0.5", "suff0.7", "suff0.9", "fix2", "fix3", "fix4", "fix5", "thr0.1", "thr0.2", "thr0.3"]:
            name = "unicor3_s0" if tag == "suff0.5" else f"unicor3_s0_{tag}"
            st, _ = seedavg(ds, name, [0], T, qs)
            if st:
                S["stop"].setdefault(ds, {})[tag] = st
        for name in ["unicor3_s0_inj", "u3nr_s0_inj", "u3noconf_s0_inj", "unicor3_s0", "u3nr_s0", "u3noconf_s0"]:
            st, _ = seedavg(ds, name, [0], T, qs)
            if st:
                S["inject"].setdefault(ds, {})[name] = st
        for name in ["unicor3_s0_np20", "unicor3_s0", "unicor3_s0_np60"]:
            f = f"{ROOT}/results/{ds}/{name}.json"
            if os.path.exists(f):
                R = json.load(open(f)); S["np"].setdefault(ds, {})[name] = {k: R.get(k) for k in ["R@2", "R@5", "ChainSel", "lat_cand_ms", "lat_net_ms", "lat_qenc_ms"]}
        # by-type QA for main methods
        for name in ["unicor3_s0", "unicor3_s0_fill", "gnn_s0_fill", "hipporag2", "dense", "dense_ft"]:
            f = f"{ROOT}/results/{ds}/{name}.qa.json"
            if os.path.exists(f):
                S.setdefault("by_type", {}).setdefault(ds, {})[name] = json.load(open(f))["summary"].get("by_type")
        for d in ["results_llama", "results_fp16"]:
            for name in ["dense", "hipporag2", "dense_ft", "gnn_s0_fill", "unicor3_s0", "unicor3_s0_fill"]:
                f = f"{ROOT}/{d}/{ds}/{name}.qa.json"
                if os.path.exists(f):
                    S.setdefault(d, {}).setdefault(ds, {})[name] = json.load(open(f))["summary"]
    if os.path.exists(f"{ROOT}/results/graph_audit.json"):
        S["audit"] = json.load(open(f"{ROOT}/results/graph_audit.json"))
    json.dump(S, open(f"{ROOT}/results/revision_summary.json", "w"), indent=1)
    print("ok")


if __name__ == "__main__":
    main()
