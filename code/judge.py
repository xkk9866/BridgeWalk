"""Audit graph quality with an LLM judge (int4 Qwen2.5-7B):
 (a) fact entailment: is the OpenIE triple entailed by its source passage?  (300 random facts / dataset)
 (b) synonymy precision: do two entities joined by a synonymy edge denote the same thing? (300 random edges / dataset)
 (c) facts emitted by UniCoR v3 vs Fixed fusion: entailment of facts actually shown to the reader."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from vllm import LLM, SamplingParams
from unicor import Corpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/yaokang/sra/models/Qwen2.5-7B-Instruct"
random.seed(0)
llm = LLM(MODEL, quantization="bitsandbytes", load_format="bitsandbytes", max_model_len=4096,
          gpu_memory_utilization=float(os.environ.get("GMU", 0.85)), seed=0)
tok = llm.get_tokenizer(); sp = SamplingParams(temperature=0, max_tokens=4)


def ask(prompts):
    ps = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    return [o.outputs[0].text.strip().lower().startswith("yes") for o in llm.generate(ps, sp)]


out = {}
for ds in ["musique", "2wikimultihopqa", "hotpotqa"]:
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    fs = random.sample(range(C.nF), 300)
    pr = [f"Passage: {C.titles[C.f_prov[f][0]]}: {C.texts[C.f_prov[f][0]]}\n\nClaim (subject | relation | object): {C.facts[f]}\n\n"
          "Is the claim stated or directly entailed by the passage? Answer Yes or No." for f in fs]
    ent = ask(pr)
    syn = C.syn.tolist(); es = random.sample(syn, min(300, len(syn)))
    ps = [f"Do the two names refer to the same real-world entity or concept?\nA: {C.ents[a]}\nB: {C.ents[b]}\nAnswer Yes or No." for a, b in es]
    sy = ask(ps)
    res = {"fact_entail": 100 * sum(ent) / len(ent), "syn_precision": 100 * sum(sy) / len(sy), "n_syn_edges": len(syn)}
    for m in ["unicor3", "fixed_fusion"]:
        fpath = f"{ROOT}/results/{ds}/{m}.json"
        if not os.path.exists(fpath):
            continue
        items = json.load(open(fpath))["items"]
        shown = [f for it in items for f in it["facts"]]
        shown = random.sample(shown, min(300, len(shown)))
        pr = [f"Passage: {C.titles[C.f_prov[f][0]]}: {C.texts[C.f_prov[f][0]]}\n\nClaim (subject | relation | object): {C.facts[f]}\n\n"
              "Is the claim stated or directly entailed by the passage? Answer Yes or No." for f in shown]
        res[f"entail_shown_{m}"] = 100 * sum(ask(pr)) / max(1, len(pr))
    out[ds] = res
    print(ds, res, flush=True)
json.dump(out, open(f"{ROOT}/results/graph_audit.json", "w"), indent=1)
