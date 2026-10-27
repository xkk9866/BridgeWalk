"""QA reading with int4 (AWQ) LLM + evaluation (EM/F1, RelCov, Trace, tokens).
python reader.py <ds> <method> [<method> ...]   -> results/<ds>/<method>.qa.json"""
import json, os, re, string, sys, collections, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from vllm import LLM, SamplingParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("RESDIR", "results")
MODEL = os.environ.get("READER", "/home/yaokang/research/testimony_rag/models/Qwen2.5-7B-Instruct-AWQ")
QUANT = dict(quantization="awq_marlin") if "AWQ" in MODEL else dict(quantization="bitsandbytes", load_format="bitsandbytes")  # int4 only (project constraint: int-quantised, <=14B)
if os.environ.get("FP16"):
    sys.exit("FP16 runs disabled: only int-quantised models are used")  # int4 NF4 weight quantization
SYS = ("You are a precise question answering assistant. Use the evidence (passages and knowledge-graph facts, each "
       "fact is tagged with the passage it comes from) to answer the question. Think briefly step by step, then give "
       "the final answer on the last line as 'Answer: <short answer>'. The short answer should be an entity, date, "
       "number or yes/no, with no explanation.")


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(p, g):
    pt, gt = normalize(p).split(), normalize(g).split()
    common = collections.Counter(pt) & collections.Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(pt), ns / len(gt)
    return 2 * pr * rc / (pr + rc)


def context(C, item):
    """passages are rendered whole unless the item selects some of their sentences (item["sents"], see threads.py)"""
    import threads
    return threads.render(C, item["pas"], item["facts"], item.get("sents"))


def main():
    from unicor import Corpus, norm_text
    ds, methods = sys.argv[1], sys.argv[2:]
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    # shared GPU: memory other processes allocate while vLLM profiles is charged to vLLM, so claim the headroom that is
    # free at start (between GMU and 0.6 of the card, always leaving 1.5 GB to other users)
    free, total = torch.cuda.mem_get_info()
    gmu = float(os.environ.get("GMU", 0.85))
    gmu = max(gmu, min(0.6, (free - 1.5 * 2**30) / total))
    llm = LLM(MODEL, **QUANT, max_model_len=8192, gpu_memory_utilization=gmu, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=256)
    summary = {}
    for meth in methods:
        R = json.load(open(f"{ROOT}/{RES}/{ds}/{meth}.json"))
        items = R["items"]
        ctxs = [context(C, it) for it in items]
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYS},
                                            {"role": "user", "content": f"Evidence:\n{c}\n\nQuestion: {q['question']}"}],
                                           tokenize=False, add_generation_prompt=True) for c, q in zip(ctxs, qs)]
        _t0 = time.time(); outs = llm.generate(prompts, sp); gen_s = (time.time() - _t0) / len(prompts)
        em, ff, toks, trace, relcov, preds = [], [], [], [], [], []
        chain, insuff_ok, hop = [], [], collections.defaultdict(list)
        for q, it, c, o in zip(qs, items, ctxs, outs):
            txt = o.outputs[0].text.strip()
            m = re.findall(r"Answer:\s*(.*)", txt)
            pred = (m[-1] if m else txt.split("\n")[-1]).strip().strip(".")
            preds.append(pred)
            em.append(max(float(normalize(pred) == normalize(a)) for a in q["answers"]))
            ff.append(max(f1(pred, a) for a in q["answers"]))
            toks.append(len(tok(c)["input_ids"]))
            full = set(q["gold_titles"]) <= {C.titles[p] for p in it["pas"]}
            chain.append(float(full))
            if em[-1] == 1 and not full:
                insuff_ok.append(1.0)
            elif em[-1] == 1:
                insuff_ok.append(0.0)
            ht = q["qid"].split("__")[0][:4] if ds == "musique" else (q.get("type") or "all")
            hop[ht].append((em[-1], ff[-1], float(full), toks[-1]))
            if it["facts"]:
                ps = set(it["pas"])
                trace.append(np.mean([any(p in ps for p in C.f_prov[f]) for f in it["facts"]]))
            if q.get("gold_triples"):
                units = [norm_text(C.titles[p] + " " + C.texts[p]) for p in it["pas"]] + [norm_text(C.facts[f]) for f in it["facts"]]
                cov = []
                for s, r, o_ in q["gold_triples"]:
                    a, b = norm_text(s), norm_text(o_)
                    cov.append(any(a in u and b in u for u in units))
                relcov.append(np.mean(cov))
        res = {"gen_s_per_q": gen_s, "out_tokens": float(np.mean([len(o.outputs[0].token_ids) for o in outs])), "EM": 100 * np.mean(em), "F1": 100 * np.mean(ff), "tokens": float(np.mean(toks)),
               "ChainCtx": 100 * float(np.mean(chain)), "CorrectButInsufficient": 100 * float(np.mean(insuff_ok)) if insuff_ok else None,
               "by_type": {k: [round(100 * float(np.mean([x[i] for x in v])), 2) for i in range(3)] + [round(float(np.mean([x[3] for x in v])), 1), len(v)] for k, v in hop.items()},
               "Trace": 100 * float(np.mean(trace)) if trace else None,
               "RelCov": 100 * float(np.mean(relcov)) if relcov else None,
               "R@2": R["R@2"], "R@5": R["R@5"], "latency_ms": 1000 * R["latency_s"],
               "avgP": float(np.mean([len(i["pas"]) for i in items])), "avgF": float(np.mean([len(i["facts"]) for i in items]))}
        summary[meth] = res
        json.dump({"summary": res, "preds": preds, "em": em, "f1": ff}, open(f"{ROOT}/{RES}/{ds}/{meth}.qa.json", "w"))
        print(ds, meth, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
