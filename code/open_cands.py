"""Open-corpus first stage, shared by every method: dense (bge) top-12 candidates per question in the candidate-row
format read by chain_infer.py / chainwalk.py / bridgewalk.py, plus the dense ranking itself as a method ("dense").
python open_cands.py <open_ds>"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from index import embed_queries
import budget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ds = sys.argv[1]
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    t0 = time.time()
    Q = embed_queries([q["question"] for q in qs]).cuda().half()
    P = C.P.cuda().half()
    top = torch.cat([(Q[i:i + 64] @ P.T).topk(60, dim=1).indices.cpu() for i in range(0, len(qs), 64)]).tolist()
    lat = (time.time() - t0) / len(qs)
    with open(f"{ROOT}/results/{ds}/chain_cand_v7.jsonl", "w", encoding="utf-8") as f:
        for q, t in zip(qs, top):
            f.write(json.dumps({"question": q["question"], "cands": [{"pid": p, "title": C.titles[p], "text": C.texts[p]} for p in t[:12]]}, ensure_ascii=False) + "\n")
    json.dump(top, open(f"{ROOT}/results/{ds}/dense_top60.json", "w"))
    for B in (300, 450, 600, 800):
        items = [{"qid": q["qid"], "rank": t, "pas": budget.pack(C, t, B)[0], "facts": []} for q, t in zip(qs, top)]
        rk = lambda k: 100 * np.mean([len(set(q["gold_titles"]) & {C.titles[p] for p in t[:k]}) / len(set(q["gold_titles"])) for q, t in zip(qs, top)])
        json.dump({"method": f"dense_SB{B}", "R@2": rk(2), "R@5": rk(5), "latency_s": lat, "items": items}, open(f"{ROOT}/results/{ds}/dense_SB{B}.json", "w"))
    print(ds, "passages", C.nP, "dense R@2 %.1f R@5 %.1f" % (rk(2), rk(5)), flush=True)


if __name__ == "__main__":
    main()
