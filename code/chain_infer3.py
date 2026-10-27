"""Question-rooted evidence trees (QRET) and controlled ordering comparison.
A jointly trained cross-encoder gives, in one logit space, support scores u_i = s(q,[c_i]) and chain-link scores
S_ij = s(q,[c_i </s> c_j]). QRET adds the question as a root vertex r with edge weights w(r,i)=u_i, w(i,j)=S_ij, and grows
the maximum spanning tree from r (Prim): every evidence passage enters either because it directly supports the question
or because it is a verified link to evidence already chosen. The insertion order is packed under the exact reader budget.
Controls with the same candidates and the same scores: unrooted MST, pairwise greedy, pointwise, MMR, coverage greedy.
python chain_infer3.py <ds> <verifier_dir> <name> <cand_suffix>"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unicor import Corpus
from chain_train import pair_doc, ptxt
from chain_infer import score
from chain_infer2 import prim_order
import budget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS = [300, 450, 600, 800]


def qret_order(u, S):
    K = len(u)
    best = u.copy(); order = []
    while len(order) < K:
        b = best.copy(); b[order] = -np.inf
        k = int(np.argmax(b)); order.append(k)
        best = np.maximum(best, S[k])
    return order


def greedy_pair(S):
    i, j = np.unravel_index(np.argmax(S), S.shape)
    mu = S.max(1)
    return [int(i), int(j)] + [int(k) for k in np.argsort(-mu) if k not in (i, j)]


def mmr(u, E, lam=0.5):
    sim = E @ E.T; order = []
    while len(order) < len(u):
        sc = [lam * u[k] - (1 - lam) * (max(sim[k, order]) if order else 0) if k not in order else -np.inf for k in range(len(u))]
        order.append(int(np.argmax(sc)))
    return order


def coverage(u, S):
    sig = lambda x: 1 / (1 + np.exp(-x))
    pu, pS = sig(u), sig(S); order = []
    while len(order) < len(u):
        sc = [pu[k] + (pS[k, order].sum() if order else 0) if k not in order else -np.inf for k in range(len(u))]
        order.append(int(np.argmax(sc)))
    return order


def main():
    ds, vdir, name, suf = sys.argv[1:5]
    tok = AutoTokenizer.from_pretrained(vdir)
    m = AutoModelForSequenceClassification.from_pretrained(vdir, num_labels=1, torch_dtype=torch.bfloat16).cuda().eval()
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    rows = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand{suf}.jsonl", encoding="utf-8")]
    orders = {k: [] for k in ["qret", "mst", "greedy", "point", "mmr", "cover"]}
    t0 = time.time(); scores = []
    for r in rows:
        K = len(r["cands"]); pids = [c["pid"] for c in r["cands"]]
        u = score(m, tok, r["question"], [ptxt(c) for c in r["cands"]])
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        s = score(m, tok, r["question"], [pair_doc(r, i, j, "pairnb") for i, j in pairs])
        S = np.full((K, K), -1e9)
        for (i, j), v in zip(pairs, s):
            S[i, j] = S[j, i] = v
        E = torch.nn.functional.normalize(C.P[pids].float(), dim=-1).numpy()
        for k, od in [("qret", qret_order(u, S)), ("mst", prim_order(S)), ("greedy", greedy_pair(S)),
                      ("point", list(np.argsort(-u))), ("mmr", mmr(u, E)), ("cover", coverage(u, S))]:
            orders[k].append([pids[i] for i in od])
        scores.append({"u": [round(float(x), 3) for x in u]})
    lat = (time.time() - t0) / len(rows)
    for k, ods in orders.items():
        for B in BUDGETS:
            items = []
            for q, od in zip(qs, ods):
                pas, _ = budget.pack(C, od, B)
                items.append({"qid": q["qid"], "rank": od, "pas": pas, "facts": [], "sel_p": od[:2]})
            r2 = np.mean([len(set(q["gold_titles"]) & {C.titles[p] for p in o["rank"][:2]}) / len(q["gold_titles"]) for q, o in zip(qs, items)])
            r5 = np.mean([len(set(q["gold_titles"]) & {C.titles[p] for p in o["rank"][:5]}) / len(q["gold_titles"]) for q, o in zip(qs, items)])
            cc = np.mean([float(set(q["gold_titles"]) <= {C.titles[p] for p in o["pas"]}) for q, o in zip(qs, items)])
            tag = f"{name}_{k}_SB{B}"
            json.dump({"method": tag, "R@2": 100 * r2, "R@5": 100 * r5, "ChainCtx": 100 * cc, "latency_s": lat, "items": items},
                      open(f"{ROOT}/results/{ds}/{tag}.json", "w"))
            print(f"{ds} {tag:28s} R@2 {100*r2:.1f} R@5 {100*r5:.1f} Chain {100*cc:.1f} lat {1000*lat:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
