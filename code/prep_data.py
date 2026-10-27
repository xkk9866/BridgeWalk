"""Convert HippoRAG-format test corpora/questions + original train splits into a unified jsonl format.

data/<ds>/corpus.jsonl : {"pid","title","text"}
data/<ds>/test.jsonl   : {"qid","question","answers":[...],"gold_titles":[...],"gold_triples":[...] (2wiki only)}
data/train/<ds>/corpus.jsonl, questions.jsonl : per-question candidate pools (for training the propagation network)
"""
import json, os, glob, random, hashlib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HR = os.path.join(ROOT, "repos/HippoRAG/reproduce/dataset")
HF = os.path.expanduser("~/.cache/huggingface/hub")
random.seed(0)


def w(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pid_of(title, text):
    return hashlib.md5((title + "\n" + text).encode("utf-8")).hexdigest()[:16]


def prep_test():
    for ds in ["musique", "2wikimultihopqa", "hotpotqa"]:
        corpus = json.load(open(f"{HR}/{ds}_corpus.json", encoding="utf-8"))
        rows, seen = [], set()
        for c in corpus:
            p = pid_of(c["title"], c["text"])
            if p in seen:
                continue
            seen.add(p)
            rows.append({"pid": p, "title": c["title"], "text": c["text"]})
        w(f"{ROOT}/data/{ds}/corpus.jsonl", rows)
        qs = json.load(open(f"{HR}/{ds}.json", encoding="utf-8"))
        out = []
        for q in qs:
            if ds == "musique":
                ans = [q["answer"]] + q.get("answer_aliases", [])
                gold = [p["title"] for p in q["paragraphs"] if p["is_supporting"]]
                gtexts = [p["paragraph_text"] for p in q["paragraphs"] if p["is_supporting"]]
                out.append({"qid": q["id"], "question": q["question"], "answers": ans, "gold_titles": gold,
                            "gold_texts": gtexts, "decomp": [d["question"] for d in q.get("question_decomposition", [])]})
            elif ds == "2wikimultihopqa":
                ans = [q["answer"]]
                gold = list(dict.fromkeys(t for t, _ in q["supporting_facts"]))
                ctx = {t: " ".join(s) for t, s in q["context"]}
                out.append({"qid": q["_id"], "question": q["question"], "answers": ans, "gold_titles": gold,
                            "gold_texts": [ctx[t] for t in gold if t in ctx], "type": q.get("type"),
                            "gold_triples": q.get("evidences", [])})
            else:
                gold = list(dict.fromkeys(t for t, _ in q["supporting_facts"]))
                ctx = {t: "".join(s) for t, s in q["context"]}
                out.append({"qid": q["_id"], "question": q["question"], "answers": [q["answer"]], "gold_titles": gold,
                            "gold_texts": [ctx[t] for t in gold if t in ctx], "type": q.get("type")})
        w(f"{ROOT}/data/{ds}/test.jsonl", out)
        print(ds, len(rows), len(out))


def prep_train(n_per=1200):
    """Training pools from official train splits (disjoint from the test questions)."""
    test_ids = set()
    for ds in ["musique", "2wikimultihopqa", "hotpotqa"]:
        for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8"):
            test_ids.add(json.loads(l)["qid"])
    pools = {}
    # HotpotQA distractor train
    f = sorted(glob.glob(f"{HF}/datasets--hotpot_qa/snapshots/*/distractor/train-0000*.parquet"))
    df = pd.concat([pd.read_parquet(x) for x in f]).sample(frac=1, random_state=0)
    df = df[df["level"] == "hard"]
    rows = []
    for _, r in df.iterrows():
        if r["id"] in test_ids:
            continue
        titles, sents = r["context"]["title"], r["context"]["sentences"]
        paras = [{"title": t, "text": "".join(s)} for t, s in zip(titles, sents)]
        gold = list(dict.fromkeys(r["supporting_facts"]["title"]))
        rows.append({"qid": r["id"], "question": r["question"], "answers": [r["answer"]], "paras": paras, "gold_titles": gold})
        if len(rows) >= n_per:
            break
    pools["hotpotqa"] = rows
    # 2Wiki train
    f = sorted(glob.glob(f"{HF}/datasets--framolfese--2WikiMultihopQA/snapshots/*/data/train-*.parquet"))
    df = pd.concat([pd.read_parquet(x) for x in f]).sample(frac=1, random_state=0)
    rows = []
    for _, r in df.iterrows():
        if r["id"] in test_ids:
            continue
        ctx = r["context"]
        titles, sents = ctx["title"], ctx["sentences"]
        paras = [{"title": t, "text": " ".join(s)} for t, s in zip(titles, sents)]
        gold = list(dict.fromkeys(r["supporting_facts"]["title"]))
        rows.append({"qid": r["id"], "question": r["question"], "answers": [r["answer"]], "paras": paras, "gold_titles": gold})
        if len(rows) >= n_per:
            break
    pools["2wikimultihopqa"] = rows
    # MuSiQue train (jsonl downloaded separately)
    mp = f"{ROOT}/data/raw/musique_ans_v1.0_train.jsonl"
    if os.path.exists(mp):
        lines = [json.loads(l) for l in open(mp, encoding="utf-8")]
        random.shuffle(lines)
        rows = []
        for q in lines:
            if q["id"] in test_ids or not q.get("answerable", True):
                continue
            paras = [{"title": p["title"], "text": p["paragraph_text"]} for p in q["paragraphs"]]
            sup = [d["paragraph_support_idx"] for d in q["question_decomposition"] if d.get("paragraph_support_idx") is not None]
            gold = [q["paragraphs"][i]["title"] for i in sup] or [p["title"] for p in q["paragraphs"] if p["is_supporting"]]
            rows.append({"qid": q["id"], "question": q["question"], "answers": [q["answer"]] + q.get("answer_aliases", []),
                         "paras": paras, "gold_titles": gold, "gold_idx": sup})
            if len(rows) >= n_per:
                break
        pools["musique"] = rows
    for ds, rows in pools.items():
        corpus, qs = {}, []
        for r in rows:
            pids = []
            for p in r["paras"]:
                pid = pid_of(p["title"], p["text"])
                corpus[pid] = {"pid": pid, "title": p["title"], "text": p["text"]}
                pids.append(pid)
            if r.get("gold_idx"):  # hop order from the MuSiQue decomposition
                gold_pids = list(dict.fromkeys(pid_of(r["paras"][i]["title"], r["paras"][i]["text"]) for i in r["gold_idx"]))
            else:
                gold_pids = [pid_of(p["title"], p["text"]) for p in r["paras"] if p["title"] in r["gold_titles"]]
            qs.append({"qid": r["qid"], "question": r["question"], "answers": r["answers"], "pool": pids,
                       "gold_pids": gold_pids, "gold_titles": r["gold_titles"], "ordered": bool(r.get("gold_idx"))})
        w(f"{ROOT}/data/train/{ds}/corpus.jsonl", list(corpus.values()))
        w(f"{ROOT}/data/train/{ds}/questions.jsonl", qs)
        print("train", ds, len(qs), len(corpus))


if __name__ == "__main__":
    prep_test()
    prep_train()
