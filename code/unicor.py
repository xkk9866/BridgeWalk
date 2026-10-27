"""UniCoR: Unified Co-Retrieval Network.
One network = query-conditioned grounded propagation encoder over passage/fact/entity nodes
            + budget-aware evidence set decoder (pointer + STOP), trained with a single objective.
"""
import math, os, re, string
import torch, torch.nn as nn, torch.nn.functional as Fn

NP, NF, NE = 0, 1, 2
# edge types (directed): 0 f->p prov,1 p->f prov,2 f->e arg,3 e->f arg,4 e->p ment,5 p->e ment,6 e->e syn,
#                       7 p->p subject link (p mentions the subject entity of p'), 8 its reverse
SUBJ = bool(os.environ.get("SUBJ"))
N_ET = 9 if SUBJ else 7


def _norm_ent(s):
    s = s.lower().strip()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return " ".join(s.split())


def norm_text(s):
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


class Corpus:
    """Holds a Grounded Evidence Graph and builds per-query candidate subgraphs."""

    def __init__(self, path, device="cuda"):
        d = torch.load(path, weights_only=False)
        self.__dict__.update(d)
        self.dev = device
        self.P = d["P"].to(device); self.F = d["F"].to(device); self.E = d["E"].to(device)
        nF, nP, nE = len(self.facts), len(self.pids), len(self.ents)
        self.nP, self.nF, self.nE = nP, nF, nE
        self.pid2i = {p: i for i, p in enumerate(self.pids)}
        self.title2i = {}
        for i, t in enumerate(self.titles):
            self.title2i.setdefault(t, i)
        self.f_prov = [[] for _ in range(nF)]; self.p_facts = [[] for _ in range(nP)]
        for f, p in d["prov"].tolist():
            self.f_prov[f].append(p); self.p_facts[p].append(f)
        self.e_pass = [[] for _ in range(nE)]; self.p_ents = [[] for _ in range(nP)]
        for e, p in d["ment"].tolist():
            self.e_pass[e].append(p); self.p_ents[p].append(e)
        self.e_syn = [[] for _ in range(nE)]
        for a, b in d["syn"].tolist():
            self.e_syn[a].append(b)
        self.f_args = d["f_args"].tolist()
        self.p_cost = torch.tensor([len(t.split()) * 1.35 + 8 for t in self.texts], device=device)
        # subject links of the grounded evidence graph: passage a -> passage b if a mentions b's subject (title) entity
        eidx = {e: i for i, e in enumerate(self.ents)}
        self.title_e = [eidx.get(_norm_ent(t)) for t in self.titles]
        e_title = {}
        for p_, e_ in enumerate(self.title_e):
            if e_ is not None:
                e_title.setdefault(e_, []).append(p_)
        self.subj_out = [sorted({b for e_ in self.p_ents[a] for b in e_title.get(e_, []) if b != a}) for a in range(nP)]
        self.f_cost = torch.tensor([len(f.split()) * 1.35 + 4 for f in self.facts], device=device)

    @torch.no_grad()
    def candidates(self, q, n_p=30, n_f=60, n_exp=15, max_p=64, max_f=160, force_p=()):
        q = q.to(self.dev)
        sp = (self.P @ q).float(); sf = (self.F @ q).float() if self.nF else torch.zeros(0, device=self.dev)
        P = sp.topk(min(n_p, self.nP)).indices.tolist()
        topf = sf.topk(min(n_f, self.nF)).indices.tolist() if self.nF else []
        pset = dict.fromkeys(P)
        for f in topf:
            for p in self.f_prov[f]:
                pset.setdefault(p)
        # entity expansion -> bridge passages
        cand = set()
        for f in topf[:20]:
            for e in self.f_args[f]:
                for e2 in [e] + self.e_syn[e]:
                    cand.update(self.e_pass[e2][:50])
        cand = [p for p in cand if p not in pset]
        if cand:
            cs = sp[torch.tensor(cand, device=self.dev)]
            for j in cs.topk(min(n_exp, len(cand))).indices.tolist():
                pset.setdefault(cand[j])
        if SUBJ:   # subject-link expansion: passages whose subject is mentioned by / mentions a top passage
            seeds = list(pset)[:20]
            sc = set()
            for a in seeds:
                sc.update(self.subj_out[a])
                te = self.title_e[a]
                if te is not None:
                    sc.update(self.e_pass[te][:50])
            sc = [p for p in sc if p not in pset]
            if sc:
                cs = sp[torch.tensor(sc, device=self.dev)]
                for j in cs.topk(min(30, len(sc))).indices.tolist():
                    pset.setdefault(sc[j])
        for p in force_p:
            pset.setdefault(p)
        P = list(pset)
        if len(P) > max_p:
            keep = set(force_p)
            rest = sorted([p for p in P if p not in keep], key=lambda p: -sp[p].item())
            P = list(keep) + rest[:max_p - len(keep)]
        Pset = set(P)
        fl = list(dict.fromkeys([f for p in P for f in self.p_facts[p]]))
        if len(fl) > max_f:
            fs = sf[torch.tensor(fl, device=self.dev)]
            fl = [fl[j] for j in fs.topk(max_f).indices.tolist()]
        E = list(dict.fromkeys([e for f in fl for e in self.f_args[f]] +
                               [e for p in P for e in self.p_ents[p][:1]]))
        pi = {p: i for i, p in enumerate(P)}; fi = {f: i + len(P) for i, f in enumerate(fl)}
        ei = {e: i + len(P) + len(fl) for i, e in enumerate(E)}
        src, dst, et = [], [], []

        def add(a, b, t):
            src.append(a); dst.append(b); et.append(t)
        for f in fl:
            for p in self.f_prov[f]:
                if p in Pset:
                    add(fi[f], pi[p], 0); add(pi[p], fi[f], 1)
            for e in self.f_args[f]:
                add(fi[f], ei[e], 2); add(ei[e], fi[f], 3)
        for p in P:
            for e in self.p_ents[p]:
                if e in ei:
                    add(ei[e], pi[p], 4); add(pi[p], ei[e], 5)
        for e in E:
            for e2 in self.e_syn[e]:
                if e2 in ei:
                    add(ei[e], ei[e2], 6)
        if SUBJ:
            for a in P:
                for b in self.subj_out[a]:
                    if b in Pset:
                        add(pi[a], pi[b], 7); add(pi[b], pi[a], 8)
        dv = self.dev
        X = torch.cat([self.P[P], self.F[fl] if fl else self.P[:0], self.E[E]]).float()
        ntype = torch.tensor([NP] * len(P) + [NF] * len(fl) + [NE] * len(E), device=dv)
        cost = torch.cat([self.p_cost[P], self.f_cost[fl] if fl else self.p_cost[:0], torch.zeros(len(E), device=dv)])
        # incidence: evidence item -> entities it covers
        inc_r, inc_c = [], []
        for p in P:
            for e in self.p_ents[p]:
                if e in ei:
                    inc_r.append(pi[p]); inc_c.append(ei[e] - len(P) - len(fl))
        for f in fl:
            for e in self.f_args[f]:
                inc_r.append(fi[f]); inc_c.append(ei[e] - len(P) - len(fl))
        nC = len(P) + len(fl)
        inc = torch.zeros(nC, len(E), device=dv)
        if inc_r:
            inc[torch.tensor(inc_r, device=dv), torch.tensor(inc_c, device=dv)] = 1.0
        prov = torch.zeros(len(fl), len(P), device=dv)
        pr = [(fi[f] - len(P), pi[p]) for f in fl for p in self.f_prov[f] if p in Pset]
        if pr:
            a = torch.tensor(pr, device=dv); prov[a[:, 0], a[:, 1]] = 1.0
        return dict(P=P, Fl=fl, E=E, X=X, ntype=ntype, q=q.float(), cost=cost, inc=inc, prov=prov,
                    ei=torch.tensor([src, dst], device=dv, dtype=torch.long).reshape(2, -1),
                    et=torch.tensor(et, device=dv, dtype=torch.long), sp=sp[P], sf=sf[fl] if fl else sf[:0])


def scatter_softmax(logit, index, n):
    mx = torch.full((n,) + logit.shape[1:], -1e9, device=logit.device).scatter_reduce(
        0, index.view(-1, *([1] * (logit.dim() - 1))).expand_as(logit), logit, "amax", include_self=True)
    ex = (logit - mx[index]).exp()
    den = torch.zeros((n,) + logit.shape[1:], device=logit.device).index_add_(0, index, ex)
    return ex / (den[index] + 1e-9)


class GPLayer(nn.Module):
    """Query-conditioned heterogeneous grounded propagation layer (multi-head, edge-type FiLM gates)."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.Wm = nn.Parameter(torch.randn(N_ET, d, d) * (1 / math.sqrt(d)))
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d)
        self.gate = nn.Linear(d, N_ET * d)
        self.rel_bias = nn.Embedding(N_ET, heads)
        self.ffn = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, h, qh, ei, et):
        n, d = h.shape
        if ei.shape[1] == 0:
            return h
        s, t = ei
        msg = torch.einsum("nd,tdk->tnk", h, self.Wm)[et, s]
        g = torch.sigmoid(self.gate(qh).view(N_ET, d))[et]  # query modulates each relation channel
        msg = (msg * g).view(-1, self.h, self.dh)
        qv = self.Wq(h[t] * qh).view(-1, self.h, self.dh)
        kv = self.Wk(h[s]).view(-1, self.h, self.dh)
        a = (qv * kv).sum(-1) / math.sqrt(self.dh) + self.rel_bias(et)
        a = scatter_softmax(a, t, n)
        agg = torch.zeros(n, self.h, self.dh, device=h.device).index_add_(0, t, a.unsqueeze(-1) * msg).view(n, d)
        return self.ln(h + self.ffn(torch.cat([h, agg], -1)))


class UniCoR(nn.Module):
    def __init__(self, din=768, d=256, L=3, use_dec=True, use_prov=True, use_gate=True, use_red=True,
                 use_ground=True, use_stop=True, hop_refine=False):
        super().__init__()
        self.cfg = dict(use_dec=use_dec, use_prov=use_prov, use_gate=use_gate, use_red=use_red,
                        use_ground=use_ground, use_stop=use_stop, hop_refine=hop_refine)
        self.inp = nn.ModuleList([nn.Sequential(nn.Linear(2 * din + 1, d), nn.GELU(), nn.Linear(d, d)) for _ in range(3)])
        self.qp = nn.Linear(din, d)
        self.layers = nn.ModuleList([GPLayer(d) for _ in range(L)])
        self.score = nn.ModuleList([nn.Linear(d, 1) for _ in range(3)])
        # decoder
        self.type_emb = nn.Embedding(3, d)
        self.s0 = nn.Linear(2 * d, d)
        self.gru = nn.GRUCell(d, d)
        self.W1 = nn.Linear(d, d, bias=False); self.W2 = nn.Linear(d, d); self.w = nn.Linear(d, 1, bias=False)
        self.ctrl = nn.Linear(d, 3)  # state-dependent redundancy / grounding / budget-pressure strengths
        self.hop = GPLayer(d) if hop_refine else None  # evidence-conditioned re-propagation
        self.stop = nn.Sequential(nn.Linear(2 * d + 2, d), nn.GELU(), nn.Linear(d, 1))

    # ---------------- encoder ----------------
    def encode(self, g):
        X, q, nt = g["X"], g["q"], g["ntype"]
        cos = (X @ q).unsqueeze(-1)
        f = torch.cat([X, X * q, cos], -1)
        h = torch.zeros(X.shape[0], self.qp.out_features, device=X.device)
        for t in range(3):
            m = nt == t
            if m.any():
                h[m] = self.inp[t](f[m])
        qh = self.qp(q)
        ei, et = g["ei"], g["et"]
        if not self.cfg["use_prov"]:
            keep = et > 1; ei, et = ei[:, keep], et[keep]
        self._edges = (ei, et)
        for lay in self.layers:
            qg = qh if self.cfg["use_gate"] else torch.ones_like(qh) * 3.0
            h = lay(h, qg, ei, et)
        z = torch.zeros(X.shape[0], device=X.device)
        for t in range(3):
            m = nt == t
            if m.any():
                z[m] = self.score[t](h[m]).squeeze(-1)
        return h, z, qh

    # ---------------- decoder ----------------
    def decode(self, g, h, z, qh, budget, gold_seq=None, max_steps=16):
        """gold_seq: list of candidate indices (in [0,nC)) for teacher forcing; returns (loss or selection)."""
        nPc, nFc = len(g["P"]), len(g["Fl"])
        nC = nPc + nFc
        hc, zc = h[:nC], z[:nC]
        he, ze = h[nC:], z[nC:]
        cost = g["cost"][:nC]
        isF = torch.zeros(nC, device=h.device); isF[nPc:] = 1
        hn = Fn.normalize(hc, dim=-1)
        ew = torch.sigmoid(ze)
        s = torch.tanh(self.s0(torch.cat([qh, hc.mean(0) if nC else qh], -1)))
        sel = torch.zeros(nC, dtype=torch.bool, device=h.device)
        maxsim = torch.zeros(nC, device=h.device)
        ground = torch.zeros(nC, device=h.device)
        covered = torch.zeros(len(g["E"]), device=h.device)
        used = 0.0
        W1h = self.W1(hc)
        losses, order, n_steps = [], [], 0
        T = len(gold_seq) + 1 if gold_seq is not None else max_steps
        for t in range(T):
            lam = Fn.softplus(self.ctrl(s))
            rho = used / budget
            if self.hop is not None and t > 0:
                # re-propagate relevance over the graph conditioned on what has been selected so far
                hr = self.hop(h, s, *self._edges)
                W1h = self.W1(hr[:nC])
            rel = self.w(torch.tanh(W1h + self.W2(s))).squeeze(-1) + zc
            self._last_rel = rel
            logit = rel
            if self.cfg["use_red"]:
                logit = logit - lam[0] * 4 * maxsim
            if self.cfg["use_ground"]:
                logit = logit + lam[1] * 2 * isF * ground
            logit = logit - lam[2] * (cost / 100.0) * rho
            mask = sel.clone()
            if gold_seq is None:
                mask |= (used + cost) > budget
            logit = logit.masked_fill(mask, -1e4)
            unc = ew * (1 - covered.clamp(max=1))
            u = (unc.unsqueeze(-1) * he).sum(0) / (ew.sum() + 1e-6) if len(g["E"]) else torch.zeros_like(s)
            stop_l = self.stop(torch.cat([s, u, torch.stack([unc.sum() / (ew.sum() + 1e-6), torch.tensor(rho, device=s.device)])]))
            if not self.cfg["use_stop"] and gold_seq is None:
                stop_l = stop_l * 0 + (1e4 if bool(mask.all()) else -1e4)
            full = torch.cat([logit, stop_l])
            if gold_seq is not None:
                tgt = gold_seq[t] if t < len(gold_seq) else nC
                losses.append(Fn.cross_entropy(full.unsqueeze(0), torch.tensor([tgt], device=h.device)))
                c = tgt
            else:
                c = int(full.argmax())
            if c == nC:
                break
            order.append(c)
            sel[c] = True
            used += float(cost[c])
            maxsim = torch.maximum(maxsim, (hn @ hn[c]).clamp(min=0))
            if c < nPc and nFc:
                ground = torch.maximum(ground, torch.cat([ground.new_zeros(nPc), g["prov"][:, c]]))
            covered = covered + g["inc"][c]
            s = self.gru((hc[c] + self.type_emb.weight[int(isF[c])]).unsqueeze(0), s.unsqueeze(0)).squeeze(0)
        if gold_seq is not None:
            return torch.stack(losses).mean()
        return order

    def forward_loss(self, g, gold_seq, gold_p, gold_f, budget=1200.0, w_node=1.0, w_ground=0.5):
        h, z, qh = self.encode(g)
        nPc, nFc = len(g["P"]), len(g["Fl"])
        loss = 0.0
        zP = z[:nPc]
        if gold_p:
            loss = loss + w_node * -(Fn.log_softmax(zP, 0)[torch.tensor(gold_p, device=z.device)]).mean()
        if nFc and gold_f:
            zF = z[nPc:nPc + nFc]
            loss = loss + w_node * 0.5 * -(Fn.log_softmax(zF, 0)[torch.tensor([f - nPc for f in gold_f], device=z.device)]).mean()
        if nFc and w_ground > 0:
            # grounding consistency: a fact is only as credible as its best provenance passage
            pz = (g["prov"] * torch.sigmoid(zP).detach().unsqueeze(0)).max(1).values
            has = g["prov"].sum(1) > 0
            if has.any():
                loss = loss + w_ground * Fn.mse_loss(torch.sigmoid(z[nPc:nPc + nFc])[has], pz[has])
        if self.cfg["use_dec"]:
            loss = loss + self.decode(g, h, z, qh, budget, gold_seq=gold_seq)
        return loss

    @torch.no_grad()
    def retrieve(self, g, budget=1200.0, max_steps=16, topk_nodec=(5, 0)):
        h, z, qh = self.encode(g)
        nPc, nFc = len(g["P"]), len(g["Fl"])
        if self.cfg["use_dec"]:
            order = self.decode(g, h, z, qh, budget, max_steps=max_steps)
        else:  # node-score top-k under the same token budget (score-then-select)
            order, used = [], 0.0
            idx = z[:nPc].argsort(descending=True).tolist()[:topk_nodec[0]]
            idx += (z[nPc:nPc + nFc].argsort(descending=True) + nPc).tolist()[:topk_nodec[1]]
            for c in idx:
                if used + float(g["cost"][c]) <= budget:
                    order.append(c); used += float(g["cost"][c])
        sel_p = [g["P"][c] for c in order if c < nPc]
        sel_f = [g["Fl"][c - nPc] for c in order if c >= nPc]
        rs = self._last_rel[:nPc] if (self.cfg["use_dec"] and self.cfg.get("hop_refine")) else z[:nPc]
        rank = sel_p + [g["P"][i] for i in rs.argsort(descending=True).tolist() if g["P"][i] not in sel_p]
        return dict(order=order, sel_p=sel_p, sel_f=sel_f, rank=rank)
