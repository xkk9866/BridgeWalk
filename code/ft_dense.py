"""Supervised dense baseline: fine-tune bge-base on the SAME 3,600 training questions UniCoR uses.
Positives = gold passages, hard negatives = in-pool non-gold passages, in-batch + hard-negative InfoNCE (MNRL).
Plain PyTorch loop (no `datasets` dependency)."""
import json, os, random
import torch, torch.nn.functional as F
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QPREF = "Represent this sentence for searching relevant passages: "
random.seed(0); torch.manual_seed(0)
ex = []
for ds in ["hotpotqa", "2wikimultihopqa", "musique"]:
    corp = {}
    for l in open(f"{ROOT}/data/train/{ds}/corpus.jsonl", encoding="utf-8"):
        d = json.loads(l); corp[d["pid"]] = f"{d['title']}: {d['text']}"
    for l in open(f"{ROOT}/data/train/{ds}/questions.jsonl", encoding="utf-8"):
        q = json.loads(l)
        neg = [p for p in q["pool"] if p not in q["gold_pids"]]
        for gp in q["gold_pids"]:
            if gp in corp and neg:
                ex.append((QPREF + q["question"], corp[gp], corp[random.choice(neg)]))
print("triples", len(ex), flush=True)
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
m.max_seq_length = 256
opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
bs, epochs = 32, 2
steps = epochs * (len(ex) // bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-5, total_steps=steps + 5, pct_start=0.1)


def emb(texts):
    f = m.tokenize(texts); f = {k: v.cuda() for k, v in f.items()}
    return F.normalize(m(f)["sentence_embedding"], dim=-1)


m.train(); step = 0
for ep in range(epochs):
    random.shuffle(ex)
    for i in range(0, len(ex) - bs + 1, bs):
        b = ex[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = emb([x[0] for x in b]); d = emb([x[1] for x in b] + [x[2] for x in b])
            loss = F.cross_entropy((q @ d.T).float() / 0.05, torch.arange(len(b), device="cuda"))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        if step % 200 == 0:
            print(ep, step, float(loss), flush=True)
m.save(f"{ROOT}/ckpt/bge_ft")
print("saved", flush=True)
