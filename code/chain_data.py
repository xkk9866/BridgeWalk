"""ChainCoR data: graph-proposed candidate passages + KG bridge facts for every candidate pair.
python chain_data.py train|<test_ds> [K]
 train  -> ckpt/chain_train.jsonl   (one row per question: query, candidates, gold, pair bridges)
 <ds>   -> results/<ds>/chain_cand.jsonl
Proposer = supervised grounded GNN (gnn_s0) over the question's candidate graph."""
import json, os, sys, random
import torch
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from unicor2 import UniCoR2
from index import embed_queries

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = dict(n_p=40, n_exp=30, max_p=int(os.environ.get("MAXP", 96)))


def load_gnn():
    st = torch.load(f"{ROOT}/ckpt/{os.environ.get('GNN', 'gnn_s0')}.pt", weights_only=False)
    cfg = dict(st["cfg"]); K = cfg.pop("K"); nosyn = cfg.pop("no_syn", False)
    m = UniCoR2(d=st.get("d", 256), L=st["L"], K=K, **cfg); m.load_state_dict(st["state"])
    m.cfg["no_syn"] = nosyn
    return m.cuda().eval()


def bridges(C, a, b, k=2):
    """KG facts linking passage a and passage b: facts of one passage whose arguments are mentioned by the other."""
    ea, eb = set(C.p_ents[a]), set(C.p_ents[b])
    out = []
    for x, other in ((a, eb), (b, ea)):
        for f in C.p_facts[x]:
            if any(e in other for e in C.f_args[f]):
                out.append(f)
    return sorted(set(out), key=lambda f: len(C.facts[f]))[:k]


@torch.no_grad()
def candidates(m, C, qe, K, force=()):
    if os.environ.get("PROPOSER") == "dense":          # ablation: text-only proposer (no graph)
        sp = (C.P @ qe.to(C.P.device)).float()
        idx = sp.topk(K).indices.tolist()
        return idx, [float(sp[i]) for i in idx], None
    g = C.candidates(qe, force_p=force, **CAND)
    if os.environ.get("NOSYN"):                          # ablation: no synonymy edges in the candidate graph
        keep = g["et"] != 6; g["ei"], g["et"] = g["ei"][:, keep], g["et"][keep]
    st = m.encode(g)
    z = st["z"][:len(g["P"])]
    order = z.argsort(descending=True).tolist()[:K]
    H = st["h"][:len(g["P"])]
    allz = {g["P"][i]: float(z[i]) for i in range(len(g["P"]))}
    Hd = {g["P"][i]: H[i].float().cpu() for i in range(len(g["P"]))}
    Hd["_z"] = allz
    return [g["P"][i] for i in order], [float(z[i]) for i in order], Hd


_EIDX = {}


def title_ent(C, p):
    if id(C) not in _EIDX:
        _EIDX[id(C)] = {e: i for i, e in enumerate(C.ents)}
    from index import norm_ent
    return _EIDX[id(C)].get(norm_ent(C.titles[p]))


def pair_struct(C, a, b):
    """KG structure between two passages: shared (incl. synonymous) entities, fact bridges in each direction,
    title mentions in each direction. Returns features and the shared entity strings used as span markers."""
    ea, eb = set(C.p_ents[a]), set(C.p_ents[b])
    shared = ea & eb
    syn = {e for e in ea for e2 in C.e_syn[e] if e2 in eb}
    bab = sum(1 for f in C.p_facts[a] if any(e in eb for e in C.f_args[f]))
    bba = sum(1 for f in C.p_facts[b] if any(e in ea for e in C.f_args[f]))
    ta, tb = title_ent(C, a), title_ent(C, b)
    feat = [len(shared), len(syn), bab, bba, float(tb is not None and tb in ea), float(ta is not None and ta in eb)]
    names = sorted({C.ents[e] for e in shared | syn if len(C.ents[e]) > 2}, key=len, reverse=True)[:4]
    return feat, names


def pack(C, ps, zs, H=None):
    br, bi = {}, {}
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            b = bridges(C, ps[i], ps[j])
            if b:
                br[f"{i},{j}"] = [C.facts[f] for f in b]; bi[f"{i},{j}"] = [int(f) for f in b]
    row = {"cands": [{"pid": int(p), "title": C.titles[p], "text": C.texts[p], "z": z} for p, z in zip(ps, zs)],
           "bridges": br, "bridge_ids": bi}
    if os.environ.get("V6"):
        pf, pn = {}, {}
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                f, n = pair_struct(C, ps[i], ps[j]); pf[f"{i},{j}"] = f; pn[f"{i},{j}"] = n
        row["pairfeat"], row["shared"] = pf, pn
        if H is not None:
            for c, p in zip(row["cands"], ps):
                c["h"] = [round(x, 4) for x in H[p].tolist()]
    return row


def main():
    what = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    m = load_gnn()
    if what == "train":
        out = open(f"{ROOT}/ckpt/{os.environ.get('TRAINOUT', 'chain_train')}.jsonl", "w")
        for ds in ["hotpotqa", "2wikimultihopqa", "musique"]:
            C = Corpus(f"{ROOT}/data/train/{ds}/index.pt")
            qs = [json.loads(l) for l in open(f"{ROOT}/data/train/{ds}/questions.jsonl", encoding="utf-8")]
            Q = embed_queries([q["question"] for q in qs])
            for q, qe in zip(qs, Q):
                gp = [C.pid2i[p] for p in q["gold_pids"] if p in C.pid2i]
                if len(gp) < 2:
                    continue
                ps, zs, H = candidates(m, C, qe, K, force=gp)
                missing = [p for p in gp if p not in ps]   # training: make sure every gold hop is a candidate
                slot = len(ps) - 1
                for p in missing:
                    while slot >= 0 and ps[slot] in gp:
                        slot -= 1
                    if slot < 0:
                        break
                    ps[slot], zs[slot] = p, (H["_z"][p] if H is not None else min(zs) - 1.0)
                    slot -= 1
                if not all(p in ps for p in gp):
                    continue
                o = sorted(range(len(ps)), key=lambda k: -zs[k])   # canonical proposer order, as at test time
                ps, zs = [ps[k] for k in o], [zs[k] for k in o]
                row = pack(C, ps, zs, H)
                row.update(qid=q["qid"], ds=ds, question=q["question"], gold=[ps.index(p) for p in gp if p in ps],
                           n_gold=len(gp), ordered=bool(q.get("ordered")),
                           gold_order=[ps.index(p) for p in gp if p in ps])
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("train", ds, flush=True)
    else:
        ds = what
        C = Corpus(f"{ROOT}/data/{ds}/index.pt")
        qs = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/test.jsonl", encoding="utf-8")]
        Q = embed_queries([q["question"] for q in qs])
        with open(f"{ROOT}/results/{ds}/chain_cand{os.environ.get('CANDSUF', '')}.jsonl", "w") as out:
            for q, qe in zip(qs, Q):
                ps, zs, H = candidates(m, C, qe, K)
                row = pack(C, ps, zs, H); row.update(qid=q["qid"], question=q["question"])
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("test", ds, flush=True)


if __name__ == "__main__":
    main()
