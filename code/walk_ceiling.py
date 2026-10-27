"""Ceiling study for verifier-guided graph walks: how often does the gold chain enter the candidate set when the
proposer's top-12 is expanded through KG bridge links vs. through text-similarity neighbours (same expansion size)?"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from index import embed_queries, norm_ent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class KGNbr:
    """Passage-passage bridge links read off the grounded evidence graph:
    a -> b if a mentions b's title entity, b mentions a's title entity, or they share a rare entity."""

    def __init__(self, C, rare=5):
        self.C = C
        eidx = {e: i for i, e in enumerate(C.ents)}
        self.title_e = [eidx.get(norm_ent(t)) for t in C.titles]
        self.e_title = {}
        for p, e in enumerate(self.title_e):
            if e is not None:
                self.e_title.setdefault(e, []).append(p)
        self.rare = rare

    def __call__(self, a):
        C, out = self.C, set()
        for e in C.p_ents[a]:
            for b in self.e_title.get(e, []):          # a mentions b's subject
                out.add(b)
            if len(C.e_pass[e]) <= self.rare:           # shared rare entity
                out.update(C.e_pass[e])
        te = self.title_e[a]
        if te is not None:
            out.update(C.e_pass[te])                     # b mentions a's subject
        out.discard(a)
        return out


def main():
    nb_k, seed_k = 8, 6
    for ds in ["musique", "2wikimultihopqa", "hotpotqa"]:
        C = Corpus(f"{ROOT}/data/{ds}/index.pt")
        N = KGNbr(C)
        qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
        rows = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand_v6.jsonl", encoding="utf-8")]
        big = [json.loads(l) for l in open(f"{ROOT}/results/{ds}/chain_cand_k40.jsonl", encoding="utf-8")]
        Q = embed_queries([q["question"] for q in qs]).cuda()
        Pn = C.P.float()
        res = {"top12": [], "kg": [], "text": [], "kg2": [], "size_kg": [], "size_text": [], "text_matched": [], "proposer_matched": []}
        for q, r, qe, rb in zip(qs, rows, Q, big):
            gold = set(q["gold_titles"])
            base = [c["pid"] for c in r["cands"]]
            sq = (C.P @ qe).float()
            kg, tx = set(base), set(base)
            txl = []
            for a in base[:seed_k]:
                nb = list(N(a) - set(base))
                if nb:
                    nb = sorted(nb, key=lambda b: -float(sq[b]))[:nb_k]; kg.update(nb)
                sim = (Pn @ Pn[a]); sim[base] = -9
                nbt = sim.topk(nb_k + 1).indices.tolist()[1:nb_k + 1]
                tx.update(nbt); txl.append(nbt)
            # size-matched controls: same number of candidates as the KG walk
            n = len(kg); txm = list(base)
            for rnd in range(nb_k):
                for nbt in txl:
                    if rnd < len(nbt) and nbt[rnd] not in txm and len(txm) < n:
                        txm.append(nbt[rnd])
            pm = [c["pid"] for c in rb["cands"]][:n]
            kg2 = set(kg)                                 # second walk step from the best expanded nodes
            for a in sorted(kg - set(base), key=lambda b: -float(sq[b]))[:seed_k]:
                nb = list(N(a) - kg2)
                if nb:
                    kg2.update(sorted(nb, key=lambda b: -float(sq[b]))[:nb_k // 2])
            T = lambda S: {C.titles[p] for p in S}
            res["top12"].append(gold <= T(base)); res["kg"].append(gold <= T(kg)); res["text"].append(gold <= T(tx))
            res["kg2"].append(gold <= T(kg2)); res["size_kg"].append(len(kg)); res["size_text"].append(len(tx))
            res["text_matched"].append(gold <= T(txm)); res["proposer_matched"].append(gold <= T(pm))
        print(ds, {k: round(100 * np.mean(v), 1) if not k.startswith("size") else round(float(np.mean(v)), 1) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
