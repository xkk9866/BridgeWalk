"""Train UniCoR on HotpotQA-train + 2Wiki-train pools (MuSiQue is held out entirely: zero-shot).
python train.py --out ckpt/unicor.pt [ablation flags]"""
import argparse, json, os, random, time, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus, UniCoR, norm_text
from unicor2 import UniCoR2
from unicor3 import UniCoR3
WIDE = dict(n_p=40, n_exp=30, max_p=int(os.environ.get("MAXP", 96)))
from index import embed_queries

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gold_facts_for(C, p, question, answers, gold_titles):
    keys = set(norm_text(question).split())
    targets = [norm_text(a) for a in answers] + [norm_text(t) for t in gold_titles]
    scored = []
    for f in C.p_facts[p]:
        sc = 0
        for e in C.f_args[f]:
            en = C.ents[e]
            if any(en and (en in t or t in en) for t in targets if t):
                sc += 2
            elif len(set(en.split()) & keys) >= max(1, len(en.split()) // 2):
                sc += 1
        if sc >= 2:
            scored.append((sc, f))
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:2]]


def build_examples(name):
    C = Corpus(f"{ROOT}/data/train/{name}/index.pt")
    qs = [json.loads(l) for l in open(f"{ROOT}/data/train/{name}/questions.jsonl", encoding="utf-8")]
    Q = embed_queries([q["question"] for q in qs])
    ex = []
    for q, qe in zip(qs, Q):
        gp = [C.pid2i[p] for p in q["gold_pids"] if p in C.pid2i]
        if not gp:
            continue
        g = C.candidates(qe, force_p=gp, **WIDE)
        pos = {p: i for i, p in enumerate(g["P"])}
        fpos = {f: i + len(g["P"]) for i, f in enumerate(g["Fl"])}
        order = gp if q.get("ordered") else sorted(gp, key=lambda p: -float(C.P[p].float() @ qe.cuda().float()))
        seq, gold_f = [], []
        for p in order:
            seq.append(pos[p])
            for f in gold_facts_for(C, p, q["question"], q["answers"], q["gold_titles"]):
                if f in fpos:
                    seq.append(fpos[f]); gold_f.append(fpos[f])
        ex.append(dict(g=g, seq=seq, gold_p=[pos[p] for p in gp], gold_f=gold_f, name=name))
    return C, ex


def to_dev(g, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in g.items()}


def recall_at(model, ex, k=5):
    model.eval(); r = []
    for e in ex:
        out = model.retrieve(to_dev(e["g"], "cuda") | {"X": e["g"]["X"].cuda().float()})
        gold = {e["g"]["P"][i] for i in e["gold_p"]}
        rr = len(gold & set(out["rank"][:k])) / len(gold)
        if getattr(model, "stop_mode", None) is not None:   # v3: also reward complete chains inside the decoded set
            rr = 0.5 * rr + 0.5 * float(gold <= set(out["sel_p"]))
        r.append(rr)
    model.train()
    return sum(r) / len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/ckpt/unicor.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--no_dec", action="store_true"); ap.add_argument("--no_prov", action="store_true")
    ap.add_argument("--no_gate", action="store_true"); ap.add_argument("--no_red", action="store_true")
    ap.add_argument("--no_ground", action="store_true"); ap.add_argument("--no_gloss", action="store_true")
    ap.add_argument("--no_stop", action="store_true"); ap.add_argument("--hop", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v2", action="store_true"); ap.add_argument("--v3", action="store_true"); ap.add_argument("--gran", action="store_true"); ap.add_argument("--no_syn", action="store_true"); ap.add_argument("--kappa", type=float, default=1.0); ap.add_argument("--soft_prov", action="store_true"); ap.add_argument("--no_conf", action="store_true"); ap.add_argument("--tf", action="store_true"); ap.add_argument("--n_train", type=int, default=0); ap.add_argument("--no_diff", action="store_true")
    ap.add_argument("--no_restart", action="store_true"); ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--data", default="hotpotqa,2wikimultihopqa,musique"); ap.add_argument("--d", type=int, default=256)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    cache = f"{ROOT}/ckpt/train_examples_{a.data.replace(',', '_')}{os.environ.get('CACHESUF', '')}.pt"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if os.path.exists(cache):
        ex = torch.load(cache, weights_only=False, map_location="cpu")
        for e in ex:
            e["g"]["X"] = e["g"]["X"].half()
    else:
        ex = []
        for n in a.data.split(","):
            ex += build_examples(n)[1]
        torch.save(ex, cache)
    random.shuffle(ex)
    val, tr = ex[:300], ex[300:]
    if a.n_train:
        tr = tr[:a.n_train]
    print("train", len(tr), "val", len(val), "avg nodes", sum(e["g"]["X"].shape[0] for e in ex) / len(ex), flush=True)
    if a.v3:
        m = UniCoR3(d=a.d, L=a.L, K=a.K, use_dec=not a.no_dec, use_prov=not a.no_prov, use_diff=not a.no_diff,
                    use_restart=not a.no_restart, use_red=not a.no_red, use_ground=not a.no_ground, use_stop=True,
                    hard_prov=not a.soft_prov, conf_restart=not a.no_conf, onpolicy=not a.tf, gran=a.gran, kappa=a.kappa).cuda()
    elif a.v2:
        m = UniCoR2(d=a.d, L=a.L, K=a.K, use_dec=not a.no_dec, use_prov=not a.no_prov, use_diff=not a.no_diff,
                    use_restart=not a.no_restart, use_red=not a.no_red, use_ground=not a.no_ground,
                    use_stop=not a.no_stop).cuda()
        if a.no_syn:
            m.cfg["no_syn"] = True
    else:
      m = UniCoR(L=a.L, use_dec=not a.no_dec, use_prov=not a.no_prov, use_gate=not a.no_gate, use_red=not a.no_red,
               use_ground=not a.no_ground, use_stop=not a.no_stop, hop_refine=a.hop).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tr) // 8 + 10, pct_start=0.1)
    best, t0 = -1, time.time()
    for ep in range(a.epochs):
        random.shuffle(tr); tot = 0.0
        for i in range(0, len(tr), 8):
            beta = max(0.5, 1 - ep / max(1, a.epochs - 1) * 0.5)
            kw = dict(beta=beta) if a.v3 else {}
            loss = sum(m.forward_loss(to_dev(e["g"], "cuda") | {"X": e["g"]["X"].cuda().float()}, e["seq"], e["gold_p"], e["gold_f"], w_ground=0 if a.no_gloss else 0.5, **kw)
                       for e in tr[i:i + 8]) / len(tr[i:i + 8])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
            tot += float(loss)
        r5 = recall_at(m, val)
        print(f"ep{ep} loss {tot / (len(tr) / 8):.4f} val R@5 {r5:.4f} {time.time() - t0:.0f}s", flush=True)
        if r5 > best:
            best = r5; torch.save({"state": m.state_dict(), "cfg": m.cfg, "L": a.L, "v2": a.v2 or a.v3, "v3": a.v3, "d": a.d}, a.out)
    print("best", best)


if __name__ == "__main__":
    main()
