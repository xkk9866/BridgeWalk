"""Open-corpus setting: ONE shared corpus for all three benchmarks = the three test corpora + every training corpus +
all MuSiQue-train paragraphs + all HotpotQA-train (distractor) paragraphs, de-duplicated on exact (title, text).
OpenIE triples are kept only where they already exist (the test and training corpora); the added passages receive no LLM
processing at all, so their entities are their titles and the text-grounded mentions of the shared entity vocabulary.
Writes data/open/{corpus,triples}.jsonl and data/open_<ds>/test.jsonl (+ symlinks to the shared corpus/index)."""
import glob, hashlib, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ["musique", "2wikimultihopqa", "hotpotqa"]


def main():
    seen, docs, trip = {}, [], {}

    def add(pid, title, text, triples=None):
        k = (title.strip(), " ".join(text.split()))
        if k in seen:
            return
        pid = pid or hashlib.md5((k[0] + "\t" + k[1]).encode()).hexdigest()[:16]
        seen[k] = pid; docs.append({"pid": pid, "title": k[0], "text": k[1]})
        if triples:
            trip[pid] = triples

    for d in [f"{ROOT}/data/{ds}" for ds in DS] + [f"{ROOT}/data/train/{ds}" for ds in DS]:
        t = {}
        if os.path.exists(f"{d}/triples.jsonl"):
            for l in open(f"{d}/triples.jsonl", encoding="utf-8"):
                r = json.loads(l); t[r["pid"]] = r["triples"]
        for l in open(f"{d}/corpus.jsonl", encoding="utf-8"):
            r = json.loads(l); add(r["pid"], r["title"], r["text"], t.get(r["pid"]))
        print(d, len(docs), flush=True)
    import gzip
    for l in gzip.open(f"{ROOT}/data/raw/musique_train_paras.jsonl.gz", "rt", encoding="utf-8"):
        p = json.loads(l); add(None, p["title"], p["text"])
    print("musique-train", len(docs), flush=True)
    for f in sorted(glob.glob(f"{ROOT}/data/raw_large/hotpot_train_v1.1.json")):
        for r in json.load(open(f, encoding="utf-8")):
            for title, sents in r["context"]:
                add(None, title, "".join(sents))
        print(f, len(docs), flush=True)
    for f in sorted(glob.glob(f"{ROOT}/data/raw_large/*train-*.parquet")):
        import pyarrow.parquet as pq
        for ctx in pq.read_table(f, columns=["context"]).column("context").to_pylist():
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                add(None, title, "".join(sents))
        print(f, len(docs), flush=True)
    os.makedirs(f"{ROOT}/data/open", exist_ok=True)
    with open(f"{ROOT}/data/open/corpus.jsonl", "w", encoding="utf-8") as fo:
        for d in docs:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(f"{ROOT}/data/open/triples.jsonl", "w", encoding="utf-8") as fo:
        for p, t in trip.items():
            fo.write(json.dumps({"pid": p, "triples": t}, ensure_ascii=False) + "\n")
    for ds in DS:
        o = f"{ROOT}/data/open_{ds}"; os.makedirs(o, exist_ok=True)
        os.system(f"cp {ROOT}/data/{ds}/test.jsonl {o}/test.jsonl")
        for n in ["corpus.jsonl", "index.pt", "text_mentions.pt"]:
            if not os.path.lexists(f"{o}/{n}"):
                os.symlink(f"{ROOT}/data/open/{n}", f"{o}/{n}")
        os.makedirs(f"{ROOT}/results/open_{ds}", exist_ok=True)
    print("open corpus", len(docs), "with triples", len(trip))


if __name__ == "__main__":
    main()
