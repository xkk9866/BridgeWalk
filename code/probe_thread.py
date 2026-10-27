"""Probe (no training): does showing the reader only the link-bearing sentence of every non-final chain passage help
under strict budgets?  Uses BridgeWalk's MAP chain and its chosen bridges (results/<ds>/bw_SB450.json).
python probe_thread.py <ds> <src> <dst>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from unicor import Corpus
from bridgewalk import MentionGraph, ptxt
import budget, threads

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ds, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
C = Corpus(f"{ROOT}/data/{ds}/index.pt", device="cpu")
G = MentionGraph(C, f"{ROOT}/data/{ds}/text_mentions.pt")
R = json.load(open(f"{ROOT}/results/{ds}/{src}_SB450.json"))


def link_sentence(c, e, nxt):
    s = ptxt(C, c, 10 ** 6); off = len(C.titles[c]) + 2
    sp = G.spans(s).get(e, [])
    if not sp:
        return None
    return threads.sentence_of(C.texts[c], max(0, sp[0][0] - off))


for B in (300, 450, 600):
    items = []
    for it in R["items"]:
        ch, hb = it["chain"], it.get("chain_br", [])
        units = []
        for i, c in enumerate(ch):
            if i + 1 < len(ch) and i < len(hb) and hb[i]:
                nxt = ch[i + 1]
                cand = [e for e in hb[i] if nxt in set(G.M[e]).union(*[G.M[e2] for e2 in C.e_syn[e][:3]])] or hb[i][:1]
                j = link_sentence(c, cand[0], nxt)
                units.append((c, j))
            else:
                units.append((c, None))
        units += [(p, None) for p in it["rank"] if p not in ch]
        pas, sents = budget.pack_units(C, units, B)
        items.append({"qid": it["qid"], "rank": it["rank"], "pas": pas, "sents": sents, "facts": []})
    json.dump({"method": f"{dst}_SB{B}", "R@2": R["R@2"], "R@5": R["R@5"], "latency_s": R["latency_s"], "items": items},
              open(f"{ROOT}/results/{ds}/{dst}_SB{B}.json", "w"))
    print(ds, dst, B, "avg passages", sum(len(i["pas"]) for i in items) / len(items), flush=True)
