"""OpenIE over all corpora with int4 (AWQ) Qwen2.5-7B-Instruct via vLLM.
Writes data/<name>/triples.jsonl: {"pid", "triples": [[s,r,o],...]}"""
import json, os, sys
from vllm import LLM, SamplingParams
sys.path.insert(0, os.path.dirname(__file__))
from openie import SYS, DEMO_IN, DEMO_OUT, parse

MODEL = os.environ.get("LLM", "/home/yaokang/sra/models/Qwen2.5-7B-Instruct")
QUANT = dict(quantization="bitsandbytes", load_format="bitsandbytes")  # int4 NF4 weight quantization
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA = sys.argv[1:] or ["musique", "2wikimultihopqa", "hotpotqa", "train/hotpotqa", "train/2wikimultihopqa"]


def main():
    llm = LLM(MODEL, **QUANT, max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=384)
    for c in CORPORA:
        out = f"{ROOT}/data/{c}/triples.jsonl"
        docs = [json.loads(l) for l in open(f"{ROOT}/data/{c}/corpus.jsonl", encoding="utf-8")]
        old = [l for l in open(out, encoding="utf-8")] if os.path.exists(out) else []
        done = {json.loads(l)["pid"] for l in old}
        docs = [d for d in docs if d["pid"] not in done]
        if not docs:
            continue
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": DEMO_IN},
                                            {"role": "assistant", "content": DEMO_OUT},
                                            {"role": "user", "content": f"Title: {d['title']}\n{d['text'][:3000]}"}],
                                           tokenize=False, add_generation_prompt=True) for d in docs]
        res = llm.generate(prompts, sp)
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            f.writelines(old)
            for d, r in zip(docs, res):
                f.write(json.dumps({"pid": d["pid"], "triples": parse(r.outputs[0].text)}, ensure_ascii=False) + "\n")
        os.replace(out + ".tmp", out)
        print("done", c, len(docs), flush=True)


if __name__ == "__main__":
    main()
