"""Retrieval overview: R@2, R@5, chain coverage in the packed context, reachability, latency for every *_SB<B>.json."""
import json, glob, os, sys
R = os.path.expanduser("~/research/grag/results")
B = sys.argv[1] if len(sys.argv) > 1 else "450"
pat = sys.argv[2] if len(sys.argv) > 2 else ""
for ds in ["musique", "2wikimultihopqa", "hotpotqa", "mhrag"]:
    print("==", ds)
    for f in sorted(glob.glob(f"{R}/{ds}/*_SB{B}.json")):
        n = os.path.basename(f)[:-5]
        if pat and pat not in n:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cc = d.get("ChainCtx", d.get("Chain"))
        print(f"  {n:34s} R@2 {d.get('R@2', 0):5.1f} R@5 {d.get('R@5', 0):5.1f} Chain {cc if cc is None else round(cc, 1)} "
              f"Reach {d.get('Reach')} lat {d.get('latency_s', 0) * 1000:.0f}ms")
