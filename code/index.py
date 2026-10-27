"""Build the Grounded Evidence Graph (passages / facts / entities) and embeddings for one corpus.
python index.py <corpus_dir>   (e.g. data/musique, data/train/hotpotqa)
Outputs <corpus_dir>/index.pt
"""
import json, os, re, sys, string
import numpy as np, torch
from sentence_transformers import SentenceTransformer

EMB = os.environ.get("EMB", "BAAI/bge-base-en-v1.5")
QPREF = "Represent this sentence for searching relevant passages: "


def norm_ent(s):
    s = s.lower().strip()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return " ".join(s.split())


_model = None


def encoder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB, device=os.environ.get("EMB_DEV", "cuda"))
        _model.max_seq_length = 512
    return _model


def embed(texts, bs=256):
    return torch.tensor(encoder().encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False,
                                         convert_to_numpy=True), dtype=torch.float16)


def build(cdir):
    docs = [json.loads(l) for l in open(f"{cdir}/corpus.jsonl", encoding="utf-8")]
    trip = {}
    for l in open(f"{cdir}/triples.jsonl", encoding="utf-8"):
        r = json.loads(l); trip[r["pid"]] = r["triples"]
    pids = [d["pid"] for d in docs]
    pidx = {p: i for i, p in enumerate(pids)}
    ents, eidx = [], {}

    def eid(s):
        k = norm_ent(s)
        if not k:
            return None
        if k not in eidx:
            eidx[k] = len(ents); ents.append(k)
        return eidx[k]

    facts, fidx, f_args, prov, ment = [], {}, [], [], set()
    for i, d in enumerate(docs):
        te = eid(d["title"])
        if te is not None:
            ment.add((te, i))
        for s, r, o in trip.get(d["pid"], []):
            a, b = eid(s), eid(o)
            if a is None or b is None or a == b:
                continue
            key = (a, r.lower().strip(), b)
            if key not in fidx:
                fidx[key] = len(facts); facts.append(f"{s} | {r} | {o}"); f_args.append((a, b))
            prov.append((fidx[key], i)); ment.add((a, i)); ment.add((b, i))
    prov = sorted(set(prov))
    P = embed([f"{d['title']}: {d['text']}" for d in docs])
    F = embed(facts)
    E = embed(ents)
    # synonymy edges between entities (cos > 0.85, top-5 neighbours)
    syn = []
    Ec = E.half().cuda() if len(ents) > 200000 else E.float().cuda()      # large corpora: fp16, small chunks
    CHK = 512 if len(ents) > 200000 else 4096
    for s in range(0, len(ents), CHK):
        sim = Ec[s:s + CHK] @ Ec.T
        v, ix = sim.topk(min(6, len(ents)), dim=1)
        for r in range(v.shape[0]):
            for vv, jj in zip(v[r].tolist(), ix[r].tolist()):
                if jj != s + r and vv > 0.85:
                    syn.append((s + r, jj))
    out = dict(pids=pids, titles=[d["title"] for d in docs], texts=[d["text"] for d in docs], ents=ents, facts=facts,
               f_args=torch.tensor(f_args, dtype=torch.long).reshape(-1, 2),
               prov=torch.tensor(prov, dtype=torch.long).reshape(-1, 2),
               ment=torch.tensor(sorted(ment), dtype=torch.long).reshape(-1, 2),
               syn=torch.tensor(syn, dtype=torch.long).reshape(-1, 2), P=P, F=F, E=E)
    torch.save(out, f"{cdir}/index.pt")
    print(cdir, "P", len(pids), "F", len(facts), "E", len(ents), "prov", len(prov), "ment", len(ment), "syn", len(syn))


def embed_queries(qs):
    return embed([QPREF + q for q in qs])


if __name__ == "__main__":
    for c in sys.argv[1:]:
        build(c)
