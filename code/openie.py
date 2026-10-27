"""Batched OpenIE (named entities + relation triples) with a 4-bit/8-bit quantized local LLM.

Usage: python openie.py --corpus passages.jsonl --out triples.jsonl [--model Qwen/Qwen2.5-7B-Instruct --quant int4]
Each input line: {"pid": str, "title": str, "text": str}
Each output line: {"pid": str, "triples": [[s, r, o], ...]}
"""
import argparse, json, os, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SYS = ("You are an information extraction system. Extract factual relation triples from the passage. "
       "Each triple is (subject, relation, object). Subjects and objects must be specific named entities, "
       "dates, numbers or short noun phrases copied from the passage; resolve pronouns to the entity they refer to. "
       "Relations are short verb phrases. Output one triple per line in the form: subject | relation | object. "
       "Output nothing else.")
DEMO_IN = ("Title: Radio City (Indian radio station)\nRadio City is India's first private FM radio station and was "
           "started on 3 July 2001. It plays Hindi, English and regional songs. It was owned by Music Broadcast Limited.")
DEMO_OUT = ("Radio City | is | India's first private FM radio station\nRadio City | started on | 3 July 2001\n"
            "Radio City | plays | Hindi songs\nRadio City | plays | English songs\nRadio City | plays | regional songs\n"
            "Radio City | owned by | Music Broadcast Limited")


def load_llm(model, quant):
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if quant == "int4":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=False)
        m = AutoModelForCausalLM.from_pretrained(model, quantization_config=q, device_map="cuda:0", torch_dtype=torch.float16)
    elif quant == "int8":
        q = BitsAndBytesConfig(load_in_8bit=True)
        m = AutoModelForCausalLM.from_pretrained(model, quantization_config=q, device_map="cuda:0", torch_dtype=torch.float16)
    else:
        m = AutoModelForCausalLM.from_pretrained(model, device_map="cuda:0", torch_dtype=torch.bfloat16)
    m.eval()
    return tok, m


def build_prompt(tok, title, text):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": DEMO_IN}, {"role": "assistant", "content": DEMO_OUT},
            {"role": "user", "content": f"Title: {title}\n{text}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse(out):
    trips = []
    for line in out.split("\n"):
        parts = [p.strip(" .\"'") for p in line.split("|")]
        if len(parts) != 3 or not all(parts):
            continue
        if max(len(p) for p in parts) > 120:
            continue
        trips.append(parts)
    # dedup keep order
    seen, res = set(), []
    for t in trips:
        k = tuple(x.lower() for x in t)
        if k not in seen:
            seen.add(k); res.append(t)
    return res[:40]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", default="int4")
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    docs = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["pid"] for l in open(a.out, encoding="utf-8")}
    docs = [d for d in docs if d["pid"] not in done]
    if a.limit:
        docs = docs[:a.limit]
    # sort by length for efficient batching
    docs.sort(key=lambda d: len(d["text"]))
    tok, m = load_llm(a.model, a.quant)
    fo = open(a.out, "a", encoding="utf-8")
    t0, n = time.time(), 0
    for i in range(0, len(docs), a.bs):
        b = docs[i:i + a.bs]
        prompts = [build_prompt(tok, d["title"], d["text"][:2500]) for d in b]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")
        gen = m.generate(**enc, max_new_tokens=a.max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.pad_token_id)
        outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for d, o in zip(b, outs):
            fo.write(json.dumps({"pid": d["pid"], "triples": parse(o)}, ensure_ascii=False) + "\n")
        fo.flush()
        n += len(b)
        el = time.time() - t0
        print(f"{n}/{len(docs)} {el:.0f}s {n/el:.2f} doc/s", flush=True)


if __name__ == "__main__":
    main()
