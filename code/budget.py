"""Strict reader-budget packing with the reader's own tokenizer.
The context string is built exactly as reader.context() builds it; every line is counted with the reader
tokenizer, and the final string is re-tokenised and trimmed so that len(tokens) <= B always holds.

python budget.py <ds> <src_result_name> <dst_name> <B> [order]
  order = rank  : pack passages in the method's ranking order (all baselines)
  order = marg  : pack by expected-coverage gain per token using item marginals stored in the result ("marg")"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.environ.get("READER_TOK", "/home/yaokang/research/testimony_rag/models/Qwen2.5-7B-Instruct-AWQ")
_tok = None


def tok():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(TOK)
    return _tok


def render(C, pas, facts, sents=None):
    """identical to reader.context()"""
    import threads
    return threads.render(C, pas, facts, sents)


def pack_units(C, units, B):
    """Sentence-level packing.  `units` is an ordered list of (pid, sentence index | None); None stands for the whole
    passage.  Units are added in order while the rendered context stays within B reader tokens.  Passages appear in
    the order of their first admitted unit; sentences of a passage are shown in document order."""
    import threads
    pas, sel, whole = [], {}, set()

    def ctx(pas_, sel_):
        return render(C, pas_, [], {str(p): v for p, v in sel_.items() if p not in whole})
    for p, j in units:
        if p in whole or (j is not None and j in sel.get(p, ())):
            continue
        pas2 = pas if p in sel or p in whole else pas + [p]
        sel2 = dict(sel)
        if j is None:
            sel2[p] = list(range(len(threads.split(C.texts[p]))))
            whole.add(p)
            fits = ntok(ctx(pas2, sel2)) <= B
            if not fits:
                whole.discard(p)
        else:
            sel2[p] = sel.get(p, []) + [j]
            fits = ntok(ctx(pas2, sel2)) <= B
        if fits:
            pas, sel = pas2, sel2
            if ntok(ctx(pas, sel)) >= B - 4:                # full: nothing else can be added
                break
    out = {str(p): sorted(set(v)) for p, v in sel.items() if p not in whole}
    assert ntok(render(C, pas, [], out)) <= B
    return pas, out


def ntok(s):
    return len(tok()(s)["input_ids"]) if s else 0


def pack(C, order, B, facts=()):
    """greedily add passages in `order` while the rendered context stays within B reader tokens"""
    pas, fs = [], []
    for f in facts:
        if ntok(render(C, pas, fs + [f])) <= B:
            fs.append(f)
    for p in order:
        if p in pas:
            continue
        if ntok(render(C, pas + [p], fs)) <= B:
            pas.append(p)
    assert ntok(render(C, pas, fs)) <= B
    return pas, fs


def main():
    from unicor import Corpus
    ds, src, dst, B = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    R = json.load(open(f"{ROOT}/results/{ds}/{src}.json"))
    out = []
    for it in R["items"]:
        order = it.get("pack_order") or it["rank"]
        pas, fs = pack(C, order, B, it.get("pack_facts", []))
        o = dict(it); o["pas"], o["facts"] = pas, fs
        out.append(o)
    R2 = dict(R); R2["items"] = out; R2["method"] = dst; R2["budget"] = B
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    for k in (2, 5):   # always recompute recall from the ranking (sources may carry placeholders)
        R2[f"R@{k}"] = 100 * sum(len(set(q["gold_titles"]) & {C.titles[p] for p in o["rank"][:k]}) / len(q["gold_titles"]) for q, o in zip(qs, out)) / len(qs)
    json.dump(R2, open(f"{ROOT}/results/{ds}/{dst}.json", "w"))
    print(ds, dst, "B", B, "avgP", round(sum(len(o["pas"]) for o in out) / len(out), 2), flush=True)


if __name__ == "__main__":
    main()
