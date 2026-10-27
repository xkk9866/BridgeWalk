"""Train the ChainCoR verifier: a cross-encoder that scores (question, passage_i, KG bridge facts, passage_j)
as a candidate evidence chain. Group-wise softmax (1 gold chain vs. 15 graph-proposed distractor chains).
python chain_train.py --mode pair|pairnb|point --out ckpt/chain_pair_s0 --seed 0"""
import argparse, json, os, random, time
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "BAAI/bge-reranker-v2-m3"


def ptxt(c, n=900):
    return f"{c['title']}: {c['text'][:n]}"


def pair_doc(row, i, j, mode):
    a, b = (i, j) if i < j else (j, i)
    ca, cb = row["cands"][a], row["cands"][b]
    br = row["bridges"].get(f"{a},{b}", []) if mode == "pair" else []
    mid = (" [bridge] " + " ; ".join(br) + " [next] ") if br else " [next] "
    return ptxt(ca, 700) + mid + ptxt(cb, 700)


def groups(rows, mode, nneg=15):
    out = []
    for r in rows:
        K, gold = len(r["cands"]), set(r["gold"])
        if mode == "point":
            neg = [k for k in range(K) if k not in gold]
            for gpos in gold:
                ns = random.sample(neg, min(nneg, len(neg)))
                out.append((r["question"], [ptxt(r["cands"][gpos])] + [ptxt(r["cands"][k]) for k in ns]))
            continue
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        pos = [p for p in pairs if p[0] in gold and p[1] in gold]
        hard = [p for p in pairs if (p[0] in gold) != (p[1] in gold)]
        easy = [p for p in pairs if p[0] not in gold and p[1] not in gold]
        for pp in pos[:3]:
            ns = random.sample(hard, min(nneg // 2 + 2, len(hard)))
            ns += random.sample(easy, min(nneg - len(ns), len(easy)))
            out.append((r["question"], [pair_doc(r, *pp, mode)] + [pair_doc(r, *p, mode) for p in ns]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pair"); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--maxlen", type=int, default=512)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(f"{ROOT}/ckpt/{os.environ.get('CHAINDATA', 'chain_train')}.jsonl", encoding="utf-8")]
    if a.mode == "joint":   # one model scores single passages (support) and pairs (chain links) in one logit space
        G = groups(rows, "pairnb") + groups(rows, "point")
    else:
        G = groups(rows, a.mode)
    print("groups", len(G), flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).cuda()
    m.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(G); acc = 4
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps // acc + 5, pct_start=0.1)
    m.train(); t0 = time.time(); k = 0
    for ep in range(a.epochs):
        random.shuffle(G)
        for q, docs in G:
            enc = tok([q] * len(docs), docs, truncation="only_second", max_length=a.maxlen, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = m(**enc).logits.squeeze(-1).float()
            loss = F.cross_entropy(s.unsqueeze(0), torch.zeros(1, dtype=torch.long, device="cuda")) / acc
            loss.backward(); k += 1
            if k % acc == 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(); sched.step()
            if k % 400 == 0:
                print(ep, k, round(float(loss) * acc, 4), f"{time.time() - t0:.0f}s", flush=True)
    m.save_pretrained(a.out); tok.save_pretrained(a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
