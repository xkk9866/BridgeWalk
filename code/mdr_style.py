"""MDR-style iterative dense chain retrieval (zero-shot reimplementation with the shared bge encoder).
Hop 1: dense top-b passages for q. Hop 2: for each hop-1 passage p, re-encode [q ; p] as a new query and retrieve.
Chains (p1, p2) are scored by s(q,p1) + s([q;p1],p2) (beam b=5, as in MDR); the ranking interleaves chain members.
python mdr_style.py <ds>  -> results/<ds>/mdr.json (rank only; packed later by budget.py)"""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from index import embed_queries, embed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QPREF = "Represent this sentence for searching relevant passages: "


def main():
    ds = sys.argv[1]; beam = 5
    C = Corpus(f"{ROOT}/data/{ds}/index.pt")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    Q = embed_queries([q["question"] for q in qs]).cuda()
    items = []
    for q, qe in zip(qs, Q):
        s1 = (C.P @ qe).float()
        v1, h1 = s1.topk(beam)
        q2 = embed([QPREF + q["question"] + " " + C.titles[p] + ": " + C.texts[p][:600] for p in h1.tolist()]).cuda()
        s2 = (q2 @ C.P.T).float()                      # beam x N
        s2[:, h1] = -1e4
        v2, h2 = s2.topk(beam, dim=1)
        chains = sorted(((float(v1[i] + v2[i, j]), int(h1[i]), int(h2[i, j])) for i in range(beam) for j in range(beam)), reverse=True)
        rank = []
        for _, a, b in chains:
            for p in (a, b):
                if p not in rank:
                    rank.append(p)
        rank += [p for p in s1.topk(30).indices.tolist() if p not in rank]
        items.append({"qid": q["qid"], "rank": rank[:30], "pas": rank[:5], "facts": []})
    json.dump({"method": "mdr", "R@2": 0, "R@5": 0, "latency_s": 0, "items": items}, open(f"{ROOT}/results/{ds}/mdr.json", "w"))
    print(ds, "mdr done", flush=True)


if __name__ == "__main__":
    main()
