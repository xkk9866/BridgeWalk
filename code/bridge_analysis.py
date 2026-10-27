"""Mechanism analysis for BridgeWalk -> results/bridge_analysis.json
 * bridge hit rate: along the MAP chain, for every gold passage that still has gold passages after it, does one of the
   K followed bridges lead (through the mention graph) to a remaining gold passage?  Compared with the entities closest
   to the question (qbridge) and with the oracle (any mentioned entity leads there).
 * halting: mean MAP-chain length by number of gold hops, with and without the learned halting read-out.
 * support size actually scored per question (from the stored chains is not possible, so it is recomputed here)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from bridgewalk import MentionGraph, ptxt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ["musique", "2wikimultihopqa", "hotpotqa"]


def reach_set(G, e):
    s = set(G.M[e])
    for e2 in G.C.e_syn[e][:3]:
        s |= set(G.M[e2])
    return s


def main():
    out = {}
    for ds in DS:
        C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
        G = MentionGraph(C, f"{ROOT}/data/{ds}/text_mentions.pt")
        qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
        gold = [{i for i, t in enumerate(C.titles) if t in set(q["gold_titles"])} for q in qs]
        hops = [len(set(q["gold_titles"])) for q in qs]
        key = [q["qid"].split("__")[0][:4] if ds == "musique" else (q.get("type") or "all") for q in qs]
        out[ds] = {}
        for m in ["bridge_full", "bridge_qbridge", "bridge_nostop", "bridges1_full", "bridge_allment"]:
            f = f"{ROOT}/results/{ds}/{m}_SB450.json"
            if not os.path.exists(f):
                continue
            items = json.load(open(f))["items"]
            hit, orc, n = 0, 0, 0
            for it, gs in zip(items, gold):
                ch, hb = it["chain"], it.get("chain_br", [])
                for i, c in enumerate(ch[:-1]):
                    rem = gs - set(ch[:i + 1])
                    if c not in gs or not rem or i >= len(hb):
                        continue
                    n += 1
                    hit += any(reach_set(G, e) & rem for e in hb[i])
                    pre = "" if i == 0 else "x"
                    ments = G.spans(ptxt(C, c, 900 if not pre else 600))
                    orc += any(reach_set(G, e) & rem for e in ments)
            lens = np.array([len(it["chain"]) for it in items])
            res = {"bridge_hit": 100 * hit / max(n, 1), "bridge_oracle": 100 * orc / max(n, 1), "n_transitions": n,
                   "len_mean": float(lens.mean()),
                   "len_by_hops": {str(h): float(lens[np.array(hops) == h].mean()) for h in sorted(set(hops))},
                   "len_by_type": {k: float(lens[np.array(key) == k].mean()) for k in sorted(set(key))},
                   "corr_len_hops": float(np.corrcoef(lens, hops)[0, 1]) if len(set(hops)) > 1 and lens.std() > 0 else None,
                   "lat_ms": 1000 * float(np.mean([it.get("lat", 0) for it in items]))}
            out[ds][m] = res
            print(ds, m, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    json.dump(out, open(f"{ROOT}/results/bridge_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
