"""Evaluate learned retrievers (UniCoR v2/v3, supervised GNN, plain pointer, fine-tuned dense).
python run_v3.py <ds> <ckpt_name|dense_ft> [...]
env: TAG (suffix), STOP=suff:0.5|fixed:3|thresh:0.3, BUDGET, FILL (budget-matched reader context),
     INJECT=1 (force a top-ranked non-gold first pick: error-propagation test), NP (candidate pool size)
Writes results/<ds>/<name><TAG>.json in the run_retrieval format plus extra diagnostics."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus, UniCoR
from unicor2 import UniCoR2
from unicor3 import UniCoR3
from index import embed_queries, QPREF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = float(os.environ.get("BUDGET", 1200)); FILL = float(os.environ.get("FILL", 0))
NP = int(os.environ.get("NP", 40))
CAND = dict(n_p=NP, n_exp=30, max_p=96)


def load(name):
    st = torch.load(f"{ROOT}/ckpt/{name}.pt", weights_only=False)
    cfg = dict(st["cfg"])
    if st.get("v3"):
        K = cfg.pop("K"); m = UniCoR3(d=st.get("d", 256), L=st["L"], K=K, **cfg)
    elif st.get("v2"):
        K = cfg.pop("K"); m = UniCoR2(d=st.get("d", 256), L=st["L"], K=K, **cfg)
    else:
        m = UniCoR(L=st["L"], **cfg)
    m.load_state_dict(st["state"]); m.cuda().eval()
    if st.get("v3") and os.environ.get("STOP"):
        mode, par = os.environ["STOP"].split(":"); m.stop_mode = (mode, float(par))
    return m


def main():
    ds, names = sys.argv[1], sys.argv[2:]
    tag = os.environ.get("TAG", "")
    C = Corpus(f"{ROOT}/data/{ds}/index.pt")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
    t_q = time.time(); Q = embed_queries([q["question"] for q in qs]).cuda(); t_q = (time.time() - t_q) / len(qs)
    for name in names:
        out, tc, te = [], 0.0, 0.0
        if name == "dense_ft":
            from sentence_transformers import SentenceTransformer
            enc = SentenceTransformer(f"{ROOT}/ckpt/bge_ft", device="cuda")
            Pft = torch.tensor(enc.encode([f"{t}: {x}" for t, x in zip(C.titles, C.texts)], batch_size=256,
                                          normalize_embeddings=True), device="cuda")
            Qft = torch.tensor(enc.encode([QPREF + q["question"] for q in qs], normalize_embeddings=True), device="cuda")
            for q, qe in zip(qs, Qft):
                rank = (Pft @ qe).topk(50).indices.tolist(); pas, used = [], 0.0
                for p in rank[:5]:
                    if used + float(C.p_cost[p]) <= (FILL or BUDGET):
                        pas.append(p); used += float(C.p_cost[p])
                out.append({"qid": q["qid"], "rank": rank[:20], "pas": pas, "facts": []})
        else:
            m = load(name)
            for q, qe in zip(qs, Q):
                t0 = time.time(); g = C.candidates(qe, **CAND); torch.cuda.synchronize(); t1 = time.time()
                inj = None
                if os.environ.get("INJECT"):
                    gold = set(q["gold_titles"])
                    wrong = [i for i in g["sp"].argsort(descending=True).tolist() if C.titles[g["P"][i]] not in gold]
                    inj = wrong[0] if wrong else None
                if isinstance(m, UniCoR3):
                    r = m.retrieve(g, budget=BUDGET, inject_first=inj)
                elif m.cfg["use_dec"]:
                    r = m.retrieve(g, budget=BUDGET)
                else:
                    r = m.retrieve(g, budget=BUDGET, topk_nodec=(int(os.environ.get("TOPK", 5)), 0))
                torch.cuda.synchronize(); tc += t1 - t0; te += time.time() - t1
                pas, facts = list(r["sel_p"]), list(r["sel_f"])
                if FILL:
                    used = sum(float(C.p_cost[p]) for p in pas) + sum(float(C.f_cost[f]) for f in facts)
                    for p in r["rank"]:
                        if p not in pas and used + float(C.p_cost[p]) <= FILL:
                            pas.append(p); used += float(C.p_cost[p])
                out.append({"qid": q["qid"], "rank": [int(x) for x in r["rank"][:20]], "pas": [int(x) for x in pas],
                            "facts": [int(x) for x in facts], "sel_p": [int(x) for x in r["sel_p"]]})
        r2, r5, cc, ccs, prec = [], [], [], [], []
        for q, o in zip(qs, out):
            gold = set(q["gold_titles"])
            r2.append(len(gold & {C.titles[p] for p in o["rank"][:2]}) / len(gold))
            r5.append(len(gold & {C.titles[p] for p in o["rank"][:5]}) / len(gold))
            ctx = {C.titles[p] for p in o["pas"]}
            cc.append(float(gold <= ctx))
            sp_ = {C.titles[p] for p in o.get("sel_p", o["pas"])}
            ccs.append(float(gold <= sp_))
            prec.append(len(gold & ctx) / max(1, len(o["pas"])))
        res = {"method": name + tag, "R@2": 100 * np.mean(r2), "R@5": 100 * np.mean(r5), "latency_s": tc + te,
               "lat_cand_ms": 1000 * tc / len(qs), "lat_net_ms": 1000 * te / len(qs), "lat_qenc_ms": 1000 * t_q,
               "ChainCtx": 100 * np.mean(cc), "ChainSel": 100 * np.mean(ccs), "EvPrec": 100 * np.mean(prec),
               "items": out}
        res["latency_s"] = (tc + te) / len(qs)
        json.dump(res, open(f"{ROOT}/results/{ds}/{name}{tag}.json", "w"))
        print(f"{ds} {name+tag:28s} R@2 {res['R@2']:.1f} R@5 {res['R@5']:.1f} ChainSel {res['ChainSel']:.1f} "
              f"ChainCtx {res['ChainCtx']:.1f} Prec {res['EvPrec']:.1f} avgP {np.mean([len(o['pas']) for o in out]):.2f} "
              f"avgF {np.mean([len(o['facts']) for o in out]):.2f} cand {res['lat_cand_ms']:.1f}ms net {res['lat_net_ms']:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
