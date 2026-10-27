"""MultiHop-RAG (news, non-Wikipedia) -> unified format for zero-shot domain transfer.
Articles are split into ~120-word passages along sentence boundaries; passage titles are "<article title> (part k)" so that
gold labels are passage-level. Gold passages = passages containing an evidence fact of the query.
data/mhrag/corpus.jsonl, data/mhrag/test.jsonl (1,000 answerable queries, seed 0)."""
import json, os, re, random, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/data/raw/mhrag"
random.seed(0)


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def main():
    arts = json.load(open(f"{RAW}/corpus.json", encoding="utf-8"))
    qs = json.load(open(f"{RAW}/MultiHopRAG.json", encoding="utf-8"))
    corpus, by_title = [], {}
    for a in arts:
        sents = re.split(r"(?<=[.!?])\s+", a["body"].strip())
        parts, cur = [], []
        for s in sents:
            cur.append(s)
            if sum(len(x.split()) for x in cur) >= 120:
                parts.append(" ".join(cur)); cur = []
        if cur:
            parts.append(" ".join(cur))
        for k, t in enumerate(parts):
            title = f"{a['title']} (part {k + 1})"
            pid = hashlib.md5((title + t).encode()).hexdigest()[:16]
            corpus.append({"pid": pid, "title": title, "text": t, "src": a["source"]})
            by_title.setdefault(a["title"], []).append((title, norm(t)))
    out, miss = [], 0
    pool = [q for q in qs if q["question_type"] != "null_query" and q["evidence_list"]]
    random.shuffle(pool)
    for i, q in enumerate(pool):
        gold = []
        for ev in q["evidence_list"]:
            fn = norm(ev["fact"])
            cands = by_title.get(ev["title"], [])
            hit = [t for t, txt in cands if fn[:80] in txt] or [t for t, txt in cands if fn[-60:] in txt]
            if hit:
                gold.append(hit[0])
        gold = list(dict.fromkeys(gold))
        if len(gold) < 2:
            miss += 1; continue
        out.append({"qid": f"mh{i}", "question": q["query"], "answers": [q["answer"]], "gold_titles": gold,
                    "type": q["question_type"]})
        if len(out) >= 1000:
            break
    os.makedirs(f"{ROOT}/data/mhrag", exist_ok=True)
    with open(f"{ROOT}/data/mhrag/corpus.jsonl", "w", encoding="utf-8") as f:
        for r in corpus:
            f.write(json.dumps({k: r[k] for k in ("pid", "title", "text")}, ensure_ascii=False) + "\n")
    with open(f"{ROOT}/data/mhrag/test.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print("passages", len(corpus), "queries", len(out), "skipped(<2 gold located)", miss,
          collections.Counter(r["type"] for r in out), collections.Counter(len(r["gold_titles"]) for r in out))


if __name__ == "__main__":
    main()
