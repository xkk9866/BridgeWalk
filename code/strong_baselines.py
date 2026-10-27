"""Strong baselines (rankings; packed later by budget.py):
  mdrT   : trained MDR — hop 1 with the fine-tuned bge, hop 2 with the trained hop-2 encoder, beam 5, chain-score ranking.
  hybrid : BM25 + fine-tuned dense fused by RRF, top-12 reranked by the supervised pointwise cross-encoder.
python strong_baselines.py <ds>"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unicor import Corpus
from chain_infer import score
from chain_train import ptxt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QPREF = "Represent this sentence for searching relevant passages: "


def enc(m, texts):
    return torch.tensor(m.encode(texts, batch_size=128, normalize_embeddings=True), device="cuda")


def main():
    ds = sys.argv[1]
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    docs = [f"{t}: {x}" for t, x in zip(C.titles, C.texts)]
    h1 = SentenceTransformer(f"{ROOT}/ckpt/bge_ft", device="cuda"); h2 = SentenceTransformer(f"{ROOT}/ckpt/mdr_hop2", device="cuda")
    P1, P2 = enc(h1, docs), enc(h2, docs)
    Q1 = enc(h1, [QPREF + q["question"] for q in qs])
    # ---- trained MDR ----
    items, t0 = [], time.time()
    for q, qe in zip(qs, Q1):
        s1 = P1 @ qe; v1, i1 = s1.topk(5)
        q2 = enc(h2, [QPREF + q["question"] + " " + docs[p][:600] for p in i1.tolist()])
        s2 = q2 @ P2.T; s2[:, i1] = -1e4
        v2, i2 = s2.topk(5, dim=1)
        chains = sorted(((float(v1[a] + v2[a, b]), int(i1[a]), int(i2[a, b])) for a in range(5) for b in range(5)), reverse=True)
        rank = []
        for _, a, b in chains:
            for p in (a, b):
                if p not in rank:
                    rank.append(p)
        rank += [p for p in s1.topk(30).indices.tolist() if p not in rank]
        items.append({"qid": q["qid"], "rank": rank[:30], "pas": rank[:5], "facts": []})
    json.dump({"method": "mdrT", "R@2": 0, "R@5": 0, "latency_s": (time.time() - t0) / len(qs), "items": items},
              open(f"{ROOT}/results/{ds}/mdrT.json", "w"))
    # ---- hybrid BM25 + fine-tuned dense (RRF) -> supervised pointwise CE ----
    bm = json.load(open(f"{ROOT}/results/{ds}/bm25.json"))["items"]
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/ckpt/chain_point_s0")
    ce = AutoModelForSequenceClassification.from_pretrained(f"{ROOT}/ckpt/chain_point_s0", num_labels=1, torch_dtype=torch.bfloat16).cuda().eval()
    items, t0 = [], time.time()
    for q, qe, b in zip(qs, Q1, bm):
        dr = (P1 @ qe).topk(50).indices.tolist()
        sc = {}
        for lst in (b["rank"], dr):
            for i, p in enumerate(lst):
                sc[p] = sc.get(p, 0) + 1 / (60 + i)
        cand = sorted(sc, key=lambda p: -sc[p])[:12]
        s = score(ce, tok, q["question"], [ptxt({"title": C.titles[p], "text": C.texts[p]}) for p in cand])
        rank = [cand[i] for i in np.argsort(-s)]
        items.append({"qid": q["qid"], "rank": rank, "pas": rank[:5], "facts": []})
    json.dump({"method": "hybrid", "R@2": 0, "R@5": 0, "latency_s": (time.time() - t0) / len(qs), "items": items},
              open(f"{ROOT}/results/{ds}/hybrid.json", "w"))
    print(ds, "strong baselines done", flush=True)


if __name__ == "__main__":
    main()
