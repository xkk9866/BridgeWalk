"""Final ChainCoR inference: pairwise chain verification + parameter-free maximum-spanning-tree (Prim) evidence ordering
+ strict reader-token budget packing.
python chain_infer2.py <ds> <verifier_dir> <name> <cand_suffix>
Writes results/<ds>/<name>_SB{300,450,600,800}.json and <name>_prop_SB* (proposer order, no verifier)."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unicor import Corpus
from chain_train import pair_doc
from chain_infer import score
import budget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS = [300, 450, 600, 800]


def prim_order(S):
    """Grow a maximum spanning tree of the verifier-weighted candidate graph from its heaviest edge;
    the insertion order is the evidence order (no thresholds, chain length decided by the budget)."""
    K = S.shape[0]
    i, j = np.unravel_index(np.argmax(S), S.shape)
    order = [int(i), int(j)]
    best = np.maximum(S[i], S[j])
    while len(order) < K:
        best[order] = -np.inf
        k = int(np.argmax(best)); order.append(k)
        best = np.maximum(best, S[k])
    return order


def evaluate(C, qs, items, tag, ds, lat):
    r2 = np.mean([len(set(q["gold_titles"]) & {C.titles[p] for p in o["rank"][:2]}) / len(q["gold_titles"]) for q, o in zip(qs, items)])
    r5 = np.mean([len(set(q["gold_titles"]) & {C.titles[p] for p in o["rank"][:5]}) / len(q["gold_titles"]) for q, o in zip(qs, items)])
    cc = np.mean([float(set(q["gold_titles"]) <= {C.titles[p] for p in o["pas"]}) for q, o in zip(qs, items)])
    json.dump({"method": tag, "R@2": 100 * r2, "R@5": 100 * r5, "ChainCtx": 100 * cc, "latency_s": lat, "items": items},
              open(f"{ROOT}/results/{ds}/{tag}.json", "w"))
    print(f"{ds} {tag:26s} R@2 {100*r2:.1f} R@5 {100*r5:.1f} Chain {100*cc:.1f} avgP {np.mean([len(o['pas']) for o in items]):.2f} lat {1000*lat:.0f}ms", flush=True)


LINK = []


def v_floor(S):
    return float(S[S > -1e8].min()) if (S > -1e8).any() else 0.0


def main():
    ds, vdir, name, suf = sys.argv[1:5]
    tok = AutoTokenizer.from_pretrained(vdir)
    m = AutoModelForSequenceClassification.from_pretrained(vdir, num_labels=1, torch_dtype=torch.bfloat16).cuda().eval()
    C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    rows = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand{suf}.jsonl", encoding="utf-8")]
    orders, t0 = [], time.time()
    for r in rows:
        K = len(r["cands"])
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        s = score(m, tok, r["question"], [pair_doc(r, i, j, "pairnb") for i, j in pairs])
        S = np.full((K, K), -1e9)
        for (i, j), v in zip(pairs, s):
            S[i, j] = S[j, i] = v
        if os.environ.get("KGMASK"):           # only pairs directly linked in the KG may be compared / connected
            ps = [c["pid"] for c in r["cands"]]
            for i, j in pairs:
                a, b = ps[i], ps[j]
                linked = bool(set(C.p_ents[a]) & set(C.p_ents[b])) or b in C.subj_out[a] or a in C.subj_out[b]
                LINK.append(linked)
                if not linked:
                    S[i, j] = S[j, i] = S[i, j] - 1e6
        orders.append([r["cands"][k]["pid"] for k in prim_order(S)])
    lat = (time.time() - t0) / len(rows)
    if LINK:
        print(ds, name, "fraction of candidate pairs linked in the KG: %.3f" % (sum(LINK) / len(LINK)), flush=True)
    for B in BUDGETS:
        items, pitems = [], []
        for q, r, od in zip(qs, rows, orders):
            pas, _ = budget.pack(C, od, B)
            items.append({"qid": q["qid"], "rank": od, "pas": pas, "facts": [], "sel_p": od[:2]})
            po = [c["pid"] for c in r["cands"]]
            ppas, _ = budget.pack(C, po, B)
            pitems.append({"qid": q["qid"], "rank": po, "pas": ppas, "facts": []})
        evaluate(C, qs, items, f"{name}_SB{B}", ds, lat)
        evaluate(C, qs, pitems, f"{name}_prop_SB{B}", ds, 0.0)


if __name__ == "__main__":
    main()
