"""Print EM/F1/tokens of every *.qa.json at the given budgets, per dataset (quick status overview)."""
import json, glob, os, sys
R = os.path.expanduser("~/research/grag/results")
pat = sys.argv[1] if len(sys.argv) > 1 else ""
for ds in ["musique", "2wikimultihopqa", "hotpotqa", "mhrag"]:
    rows = []
    for f in sorted(glob.glob(f"{R}/{ds}/*.qa.json")):
        n = os.path.basename(f)[:-8]
        if pat and pat not in n:
            continue
        try:
            s = json.load(open(f))["summary"]
        except Exception as e:
            continue
        rows.append((n, s.get("EM"), s.get("F1"), s.get("tokens"), os.path.getmtime(f)))
    print("==", ds)
    for n, em, f1, t, mt in sorted(rows, key=lambda r: r[0]):
        print(f"  {n:34s} EM {em:5.1f} F1 {f1:5.1f} tok {t or 0:6.0f}")
