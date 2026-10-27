"""(runs on the server) EM/F1 of every *.qa.json in results/<ds>/ -> stdout as JSON."""
import json, glob, os
out = {}
for ds in ["musique", "2wikimultihopqa", "hotpotqa", "mhrag"]:
    for f in sorted(glob.glob(f"results/{ds}/*.qa.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        k = os.path.basename(f)[:-8]
        if isinstance(d, dict):
            s = d.get("summary", {})
            r = {m: v for m, v in s.items() if not isinstance(v, (list, dict))} if isinstance(s, dict) else {"s": str(s)[:80]}
            r["mtime"] = int(os.path.getmtime(f))
        else:
            r = {"list": len(d)}
        out.setdefault(ds, {})[k] = r
print(json.dumps(out))
