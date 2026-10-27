"""Trained MDR baseline, hop-2 encoder: fine-tune (from the hop-1 fine-tuned bge) to retrieve hop t+1 from [q ; hop-t passage],
on the same 3,600 training questions. Chains are ordered by the MuSiQue decomposition, otherwise by hop-1 dense similarity.
In-batch + hard-negative InfoNCE, plain PyTorch."""
import json, os, random
import torch, torch.nn.functional as F
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QPREF = "Represent this sentence for searching relevant passages: "
random.seed(0); torch.manual_seed(0)
m = SentenceTransformer(f"{ROOT}/ckpt/bge_ft", device="cuda"); m.max_seq_length = 384
ex = []
for ds in ["hotpotqa", "2wikimultihopqa", "musique"]:
    corp = {}
    for l in open(f"{ROOT}/data/train/{ds}/corpus.jsonl", encoding="utf-8"):
        d = json.loads(l); corp[d["pid"]] = f"{d['title']}: {d['text']}"
    for l in open(f"{ROOT}/data/train/{ds}/questions.jsonl", encoding="utf-8"):
        q = json.loads(l); g = [p for p in q["gold_pids"] if p in corp]
        if len(g) < 2:
            continue
        if not q.get("ordered"):
            with torch.no_grad():
                e = m.encode([QPREF + q["question"]] + [corp[p] for p in g], normalize_embeddings=True, convert_to_tensor=True)
            g = [g[i] for i in (e[1:] @ e[0]).argsort(descending=True).tolist()]
        neg = [p for p in q["pool"] if p not in q["gold_pids"]]
        for a, b in zip(g[:-1], g[1:]):
            if neg:
                ex.append((QPREF + q["question"] + " " + corp[a][:600], corp[b], corp[random.choice(neg)]))
print("hop transitions", len(ex), flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
bs, epochs = 32, 2
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-5, total_steps=epochs * (len(ex) // bs) + 5, pct_start=0.1)


def emb(t):
    f = m.tokenize(t); f = {k: v.cuda() for k, v in f.items()}
    return F.normalize(m(f)["sentence_embedding"], dim=-1)


m.train()
for ep in range(epochs):
    random.shuffle(ex)
    for i in range(0, len(ex) - bs + 1, bs):
        b = ex[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = emb([x[0] for x in b]); d = emb([x[1] for x in b] + [x[2] for x in b])
            loss = F.cross_entropy((q @ d.T).float() / 0.05, torch.arange(len(b), device="cuda"))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
m.save(f"{ROOT}/ckpt/mdr_hop2"); print("saved", flush=True)
