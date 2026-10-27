"""ChainCoR inference.
python chain_infer.py <ds> <model_dir|zeroshot> <mode: pair|pairnb|point> <name> [delta]
Writes results/<ds>/<name>_B{300,450,600,800}.json (+ <name>_fB* with bridge facts in the context for pair modes)."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unicor import Corpus
from chain_train import ptxt, pair_doc, BASE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS = [300, 450, 600, 800]


@torch.no_grad()
def score(m, tok, q, docs, bs=48):
    out = []
    for i in range(0, len(docs), bs):
        enc = tok([q] * len(docs[i:i + bs]), docs[i:i + bs], truncation="only_second", max_length=512, padding=True,
                  return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(m(**enc).logits.squeeze(-1).float().cpu())
    return torch.cat(out).numpy()


def main():
    ds, mdir, mode, name = sys.argv[1:5]
    delta = float(sys.argv[5]) if len(sys.argv) > 5 else 1.5
    path = BASE if mdir == "zeroshot" else mdir
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, torch_dtype=torch.bfloat16).cuda().eval()
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    rows = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand{os.environ.get('CANDSUF', '')}.jsonl", encoding="utf-8")]
    items, t0 = [], time.time()
    for q, r in zip(qs, rows):
        K = len(r["cands"]); pids = [c["pid"] for c in r["cands"]]
        if mode == "point":
            s = score(m, tok, r["question"], [ptxt(c) for c in r["cands"]])
            rank, chain, links = list(np.argsort(-s)), [], []
        else:
            pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
            s = score(m, tok, r["question"], [pair_doc(r, i, j, mode) for i, j in pairs])
            S = np.full((K, K), -1e9)
            for (i, j), v in zip(pairs, s):
                S[i, j] = S[j, i] = v
            a, b = pairs[int(np.argmax(s))]; best = float(np.max(s))
            chain, links = [a, b], [(min(a, b), max(a, b))]
            while len(chain) < 4:                      # extend the chain while a link is nearly as strong as the best
                cand = [(max(S[k, c] for c in chain), k) for k in range(K) if k not in chain]
                v, k = max(cand)
                if v < best - delta:
                    break
                c = max(chain, key=lambda c: S[k, c]); chain.append(k); links.append((min(k, c), max(k, c)))
            mk = S.max(1)
            rank = chain + [k for k in np.argsort(-mk) if k not in chain]
        rank = [int(k) for k in rank]
        it = {"qid": q["qid"], "rank": [pids[k] for k in rank], "chain": [pids[k] for k in chain],
              "bridge_ids": [f for (i, j) in links for f in r.get("bridge_ids", {}).get(f"{i},{j}", [])[:1]]}
        items.append(it)
    lat = (time.time() - t0) / len(qs)
    for B in BUDGETS:
        for fact_mode in ([False, True] if mode == "pair" else [False]):
            out = []
            for it in items:
                used, pas, facts = 0.0, [], []
                if fact_mode:
                    for f in it["bridge_ids"]:
                        if used + float(C.f_cost[f]) <= B:
                            facts.append(f); used += float(C.f_cost[f])
                for p in it["rank"]:
                    if used + float(C.p_cost[p]) <= B:
                        pas.append(p); used += float(C.p_cost[p])
                out.append({"qid": it["qid"], "rank": it["rank"][:20], "pas": pas, "facts": facts, "sel_p": it["chain"]})
            r2 = np.mean([len(set(q["gold_titles"]) & {C.titles[p] for p in o["rank"][:2]}) / len(q["gold_titles"]) for q, o in zip(qs, out)])
            r5 = np.mean([len(set(q["gold_titles"]) & {C.titles[p] for p in o["rank"][:5]}) / len(q["gold_titles"]) for q, o in zip(qs, out)])
            cc = np.mean([float(set(q["gold_titles"]) <= {C.titles[p] for p in o["pas"]}) for q, o in zip(qs, out)])
            tag = f"{name}_{'f' if fact_mode else ''}B{B}"
            json.dump({"method": tag, "R@2": 100 * r2, "R@5": 100 * r5, "ChainCtx": 100 * cc, "latency_s": lat, "items": out},
                      open(f"{ROOT}/results/{ds}/{tag}.json", "w"))
            print(f"{ds} {tag:28s} R@2 {100*r2:.1f} R@5 {100*r5:.1f} ChainCtx {100*cc:.1f} avgP {np.mean([len(o['pas']) for o in out]):.2f} "
                  f"avgF {np.mean([len(o['facts']) for o in out]):.2f} lat {1000*lat:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
