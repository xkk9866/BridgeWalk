"""Run all retrieval methods on a test dataset; writes results/<ds>/<method>.json with selected evidence.
python run_retrieval.py <ds> [methods...]"""
import json, os, sys, time, math
import numpy as np, torch, scipy.sparse as sps
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus, UniCoR
from unicor2 import UniCoR2
from index import embed_queries

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = float(os.environ.get("BUDGET", 1200))
FILL = float(os.environ.get("FILL", 0))
CAND = dict(n_p=40, n_exp=30, max_p=96) if os.environ.get("WIDE") else {}


def fill_budget(C, pas, facts, budget=BUDGET):
    """Truncate an evidence list to the shared token budget (passages first, then facts, in rank order)."""
    used, P, F = 0.0, [], []
    for p in pas:
        c = float(C.p_cost[p])
        if used + c <= budget:
            P.append(p); used += c
    for f in facts:
        c = float(C.f_cost[f])
        if used + c <= budget:
            F.append(f); used += c
    return P, F


class Graphs:
    def __init__(self, C):
        self.C = C
        nE, nP = C.nE, C.nP
        r, c = [], []
        for a, b in C.f_args:
            r += [a, b]; c += [b, a]
        for a, b in C.syn.tolist():
            r += [a, b]; c += [b, a]
        A = sps.coo_matrix((np.ones(len(r)), (r, c)), shape=(nE, nE)).tocsr()
        A.data[:] = 1.0
        self.Aent = self._rownorm(A)
        # entity+passage graph for HippoRAG2-style PPR
        m = C.ment.numpy()
        rr = r + list(m[:, 0]) + list(m[:, 1] + nE)
        cc = c + list(m[:, 1] + nE) + list(m[:, 0])
        A2 = sps.coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(nE + nP, nE + nP)).tocsr()
        A2.data[:] = 1.0
        self.A2 = self._rownorm(A2)
        self.Ment = sps.coo_matrix((np.ones(len(m)), (m[:, 0], m[:, 1])), shape=(nE, nP)).tocsr()
        df = np.asarray(self.Ment.sum(1)).ravel()
        self.spec = 1.0 / np.maximum(df, 1)

    @staticmethod
    def _rownorm(A):
        d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
        return sps.diags(1 / d) @ A

    @staticmethod
    def ppr(AT, seed, alpha=0.5, it=30):
        s = seed / max(seed.sum(), 1e-12); x = s.copy()
        for _ in range(it):
            x = (1 - alpha) * s + alpha * (AT @ x)
        return x


def bm25_scores(C):
    from rank_bm25 import BM25Okapi
    tok = lambda s: s.lower().replace(",", " ").replace(".", " ").split()
    return BM25Okapi([tok(t + " " + x) for t, x in zip(C.titles, C.texts)]), tok


def main():
    ds = sys.argv[1]
    methods = sys.argv[2:] or ["bm25", "dense", "kg_only", "fixed_fusion", "rrf", "hipporag", "hipporag2", "unicor"]
    C = Corpus(f"{ROOT}/data/{ds}/index.pt")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    Q = embed_queries([q["question"] for q in qs]).cuda()
    os.makedirs(f"{ROOT}/results/{ds}", exist_ok=True)
    G = None
    for meth in methods:
        name, ck = meth, None
        if meth.startswith("unicor"):
            ck = f"{ROOT}/ckpt/{meth}.pt"
            st = torch.load(ck, weights_only=False)
            if st.get("v2"):
                cfg = dict(st["cfg"]); K = cfg.pop("K")
                model = UniCoR2(d=st.get("d", 256), L=st["L"], K=int(os.environ.get("KTEST", K)), **cfg).cuda()
            else:
                model = UniCoR(L=st["L"], **st["cfg"]).cuda()
            model.load_state_dict(st["state"]); model.eval()
        if meth in ("hipporag", "hipporag2") and G is None:
            G = Graphs(C); AT_ent = G.Aent.T.tocsr(); AT2 = G.A2.T.tocsr()
        if meth == "bm25":
            bm, tok = bm25_scores(C)
        out, t0 = [], time.time()
        for q, qe in zip(qs, Q):
            sp = (C.P @ qe).float(); sf = (C.F @ qe).float()
            facts = []
            if meth == "bm25":
                rank = np.argsort(-bm.get_scores(tok(q["question"])))[:50].tolist()
                pas = rank[:5]
            elif meth == "dense":
                rank = sp.topk(50).indices.tolist(); pas = rank[:5]
            elif meth == "kg_only":
                ftop = sf.topk(40).indices.tolist()
                rank = list(dict.fromkeys([p for f in ftop for p in C.f_prov[f]]))
                pas, facts = [], ftop
            elif meth == "fixed_fusion":
                rank = sp.topk(50).indices.tolist(); pas = rank[:3]; facts = sf.topk(15).indices.tolist()
            elif meth == "rrf":
                bmr = None
                ftop = sf.topk(30).indices.tolist()
                frank = list(dict.fromkeys([p for f in ftop for p in C.f_prov[f]]))
                drank = sp.topk(50).indices.tolist()
                sc = {}
                for lst in (drank, frank):
                    for i, p in enumerate(lst):
                        sc[p] = sc.get(p, 0) + 1 / (60 + i)
                rank = sorted(sc, key=lambda p: -sc[p])[:50]; pas = rank[:3]; facts = ftop[:10]
            elif meth == "hipporag":
                es = (C.E @ qe).float(); v, ix = es.topk(5)
                seed = np.zeros(C.nE); seed[ix.cpu().numpy()] = v.cpu().numpy() * G.spec[ix.cpu().numpy()]
                x = G.ppr(AT_ent, seed)
                ps = G.Ment.T @ (x * G.spec)
                rank = np.argsort(-ps)[:50].tolist(); pas = rank[:5]
            elif meth == "hipporag2":
                v, ix = sf.topk(5)
                seed = np.zeros(C.nE + C.nP)
                for f, w in zip(ix.tolist(), v.tolist()):
                    for e in C.f_args[f]:
                        seed[e] += w * G.spec[e]
                spn = sp.cpu().numpy(); spn = (spn - spn.min()) / (spn.max() - spn.min() + 1e-9)
                seed[C.nE:] = 0.05 * spn
                x = G.ppr(AT2, seed)
                rank = np.argsort(-x[C.nE:])[:50].tolist(); pas = rank[:5]
            elif meth.startswith("unicor"):
                g = C.candidates(qe, **CAND)
                if model.cfg["use_dec"]:
                    r = model.retrieve(g, budget=BUDGET)
                else:
                    r = model.retrieve(g, budget=BUDGET, topk_nodec=(3, 15))
                rank, pas, facts = r["rank"], r["sel_p"], r["sel_f"]
                if FILL:  # budget-matched: extend the decoded set with the network's own passage ranking
                    used = sum(float(C.p_cost[p]) for p in pas) + sum(float(C.f_cost[f]) for f in facts)
                    pas = list(pas)
                    for p in rank:
                        if p not in pas and used + float(C.p_cost[p]) <= FILL:
                            pas.append(p); used += float(C.p_cost[p])
            if not meth.startswith("unicor"):
                pas, facts = fill_budget(C, pas, facts)
            out.append({"qid": q["qid"], "rank": [int(x) for x in rank[:20]], "pas": [int(x) for x in pas],
                        "facts": [int(x) for x in facts]})
        el = (time.time() - t0) / len(qs)
        # retrieval recall
        r2, r5 = [], []
        for q, o in zip(qs, out):
            gold = set(q["gold_titles"])
            for k, acc in ((2, r2), (5, r5)):
                got = {C.titles[p] for p in o["rank"][:k]}
                acc.append(len(gold & got) / len(gold))
        res = {"method": meth, "R@2": 100 * np.mean(r2), "R@5": 100 * np.mean(r5), "latency_s": el, "items": out}
        tag = meth + (os.environ.get("TAG", ""))
        res["method"] = tag
        json.dump(res, open(f"{ROOT}/results/{ds}/{tag}.json", "w"))
        print(f"{ds} {meth:20s} R@2 {res['R@2']:.1f} R@5 {res['R@5']:.1f} lat {el*1000:.1f}ms "
              f"avgP {np.mean([len(o['pas']) for o in out]):.2f} avgF {np.mean([len(o['facts']) for o in out]):.2f}", flush=True)


if __name__ == "__main__":
    main()
