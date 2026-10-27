"""(server) Summary + paired bootstrap tests of BridgeWalk vs every baseline, from <dir>/<ds>/<m>_SB<B>.qa.json.
usage: python _bw_stats.py [results_dir] [out_json]"""
import json, os, sys
import numpy as np
RD = sys.argv[1] if len(sys.argv) > 1 else "results"
OUTF = sys.argv[2] if len(sys.argv) > 2 else "/tmp/bw_stats.json"
DS = ["musique", "2wikimultihopqa", "hotpotqa", "mhrag"]
BASE = ["dense", "mdrT", "hybrid", "hipporag2", "hipporag2off", "gfmragoff", "qwen3rr", "rerankZS", "point", "gnn",
        "chainS_s0", "walk_full", "bridge_full"]
OURS = ["bw", "bwth", "bws1"]
rng = np.random.default_rng(0)
def load(ds, k):
    f = f"{RD}/{ds}/{k}.qa.json"
    return json.load(open(f)) if os.path.exists(f) else None
out = {}
for ds in DS:
    o = out.setdefault(ds, {})
    for m in BASE + OURS + ["bw_full2k", "bw_lora2k"]:
        for B in (300, 450, 600, 800, 1000):
            d = load(ds, f"{m}_SB{B}")
            if d:
                o[f"{m}_SB{B}"] = {k: v for k, v in d["summary"].items() if not isinstance(v, list)}
    for ours in OURS:
        for B in (300, 450, 600, 800):
            a = load(ds, f"{ours}_SB{B}")
            if not a:
                continue
            ea = np.asarray(a["em"], float)
            sc = 100.0 if ea.max() <= 1 else 1.0
            for m in BASE + ["bw"]:
                b = load(ds, f"{m}_SB{B}")
                if not b or m == ours or len(b["em"]) != len(ea):
                    continue
                d = ea - np.asarray(b["em"], float)
                idx = rng.integers(0, len(d), (10000, len(d)))
                bs = d[idx].mean(1)
                p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
                o[f"test|{ours}|{m}|{B}"] = {"diff": float(sc * d.mean()), "p": p}
json.dump(out, open(OUTF, "w"))
print("ok")
