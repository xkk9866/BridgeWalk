"""Revision statistics: subject-link significance (seed-pooled paired bootstrap on EM, exact McNemar on chain coverage),
per-hop / per-type results, and positive-pair statistics of the verifier training data. -> results/rev_stats.json"""
import json, os, math, collections
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ["musique", "2wikimultihopqa", "hotpotqa"]
rng = np.random.default_rng(0)


def qa(ds, m, res="results"):
    f = f"{ROOT}/{res}/{ds}/{m}.qa.json"
    return json.load(open(f)) if os.path.exists(f) else None


def chain(ds, m, T, qs):
    R = json.load(open(f"{ROOT}/results/{ds}/{m}.json"))["items"]
    return np.array([float(set(q["gold_titles"]) <= {T[p] for p in o["pas"]}) for q, o in zip(qs, R)])


def boot(d, B=10000):
    idx = rng.integers(0, len(d), (B, len(d))); bs = d[idx].mean(1)
    return {"diff": round(100 * float(d.mean()), 2), "ci95": [round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)],
            "p": round(float(2 * min((bs <= 0).mean(), (bs >= 0).mean())), 4)}


def mcnemar(a, b):
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum()); n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"ours_only": n01, "base_only": n10, "p_exact": p}


def main():
    out = {"subject_links": {}, "per_type": {}, "positives": {}}
    for ds in DS:
        T = torch.load(f"{ROOT}/data/{ds}/index.pt", weights_only=False)["titles"]
        qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
        out["subject_links"][ds] = {}
        for B in (450, 600):
            s0, s1, ns = qa(ds, f"chainS_s0_SB{B}"), qa(ds, f"chainS_s1_SB{B}"), qa(ds, f"chainNS_SB{B}")
            if s0 and s1 and ns:
                pooled = (np.array(s0["em"]) + np.array(s1["em"])) / 2 - np.array(ns["em"])
                out["subject_links"][ds][f"EM_B{B}_seedpooled"] = boot(pooled)
            c0, cn = chain(ds, f"chainS_s0_SB{B}", T, qs), chain(ds, f"chainNS_SB{B}", T, qs)
            c1 = chain(ds, f"chainS_s1_SB{B}", T, qs)
            out["subject_links"][ds][f"Chain_B{B}"] = {"ours_s0": round(100 * c0.mean(), 1), "ours_s1": round(100 * c1.mean(), 1),
                                                       "no_subj": round(100 * cn.mean(), 1), "mcnemar_s0": mcnemar(c0, cn), "mcnemar_s1": mcnemar(c1, cn)}
        # per-type EM / chain coverage at B=450 for the main methods
        ty = [(q["qid"].split("__")[0][:4] if ds == "musique" else q.get("type") or "all") for q in qs]
        out["per_type"][ds] = {}
        for m in ["chainS_s0", "point", "rerankZS", "gnn", "hipporag2", "mdr", "dense"]:
            r = qa(ds, f"{m}_SB450")
            if not r:
                continue
            ch = chain(ds, f"{m}_SB450", T, qs)
            agg = collections.defaultdict(list)
            for t, e, c in zip(ty, r["em"], ch):
                agg[t].append((e, c))
            out["per_type"][ds][m] = {t: [round(100 * np.mean([x[0] for x in v]), 1), round(100 * np.mean([x[1] for x in v]), 1), len(v)] for t, v in agg.items()}
    rows = [json.loads(l) for l in open(f"{ROOT}/ckpt/chain_train_v7.jsonl", encoding="utf-8")]
    hop = collections.Counter(len(r["gold"]) for r in rows)
    out["positives"] = {"questions_by_gold_size": dict(hop),
                        "positive_pairs_per_question": {k: math.comb(k, 2) for k in hop},
                        "groups_used_per_question": "min(3, C(k,2)) gold pairs -> one 16-way group each"}
    json.dump(out, open(f"{ROOT}/results/rev_stats.json", "w"), indent=1)
    print(json.dumps(out["subject_links"], indent=0)[:3000])


if __name__ == "__main__":
    main()
