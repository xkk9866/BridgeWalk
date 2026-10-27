"""UniCoR v2: Learned Grounded Diffusion + selection-as-restart set decoding.

Encoder: L untied query-modulated grounded propagation layers produce node states h; a weight-tied, mass-conserving
         diffusion operator with restart then propagates a scalar relevance mass r over the grounded graph:
             r^{k+1} = (1-a) r^0 + a * T_q^T r^k,   T_q(u->v) = softmax_{v in N(u)} <A h_u * g_type(q), B h_v> + b_type
         (Personalized PageRank is the special case of uniform T and hand-set seeds; here T and r^0 are learned and
          query-conditioned, and the tied operator extrapolates to more hops than seen in training).
Decoder: budget-aware pointer over passages/facts with STOP; after each selection the chosen item is injected into the
         restart distribution and the same diffusion operator is re-run, so the next hop is retrieved conditioned on the
         evidence chain built so far.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as Fn
from unicor import GPLayer, scatter_softmax, N_ET


class Diffusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = nn.Linear(d, d, bias=False); self.B = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, N_ET * d)
        self.tb = nn.Embedding(N_ET, 1)
        self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid -> 0.5 damping at init (HippoRAG 2 setting)

    def transition(self, h, qh, ei, et):
        s, t = ei
        d = h.shape[1]
        g = torch.sigmoid(self.gate(qh).view(N_ET, d))[et]
        logit = ((self.A(h)[s] * g) * self.B(h)[t]).sum(-1) / math.sqrt(d) + self.tb(et).squeeze(-1)
        return scatter_softmax(logit, s, h.shape[0])  # normalised over out-edges of the source: mass conserving

    def run(self, r0, T, ei, K):
        s, t = ei
        a = torch.sigmoid(self.alpha)
        r = r0
        for _ in range(K):
            r = (1 - a) * r0 + a * torch.zeros_like(r0).index_add(0, t, T * r[s])
        return r


class UniCoR2(nn.Module):
    def __init__(self, din=768, d=256, L=2, K=10, use_dec=True, use_prov=True, use_diff=True, use_restart=True,
                 use_red=True, use_ground=True, use_stop=True):
        super().__init__()
        self.cfg = dict(use_dec=use_dec, use_prov=use_prov, use_diff=use_diff, use_restart=use_restart,
                        use_red=use_red, use_ground=use_ground, use_stop=use_stop, K=K)
        self.inp = nn.ModuleList([nn.Sequential(nn.Linear(2 * din + 1, d), nn.GELU(), nn.Linear(d, d)) for _ in range(3)])
        self.qp = nn.Linear(din, d)
        self.layers = nn.ModuleList([GPLayer(d) for _ in range(L)])
        self.score = nn.ModuleList([nn.Linear(d, 1) for _ in range(3)])
        self.seed = nn.ModuleList([nn.Linear(d, 1) for _ in range(3)])
        self.diff = Diffusion(d)
        self.wr = nn.Parameter(torch.tensor(1.0))
        self.mu = nn.Parameter(torch.tensor(0.0))
        self.type_emb = nn.Embedding(3, d)
        self.s0 = nn.Linear(2 * d, d)
        self.gru = nn.GRUCell(d, d)
        self.W1 = nn.Linear(d, d, bias=False); self.W2 = nn.Linear(d, d); self.w = nn.Linear(d, 1, bias=False)
        self.ctrl = nn.Linear(d, 3)
        self.stop = nn.Sequential(nn.Linear(2 * d + 3, d), nn.GELU(), nn.Linear(d, 1))

    def _per_type(self, mods, h, nt):
        z = torch.zeros(h.shape[0], device=h.device)
        for t in range(3):
            m = nt == t
            if m.any():
                z[m] = mods[t](h[m]).squeeze(-1)
        return z

    def encode(self, g):
        X, q, nt = g["X"], g["q"], g["ntype"]
        f = torch.cat([X, X * q, (X @ q).unsqueeze(-1)], -1)
        h = torch.zeros(X.shape[0], self.qp.out_features, device=X.device)
        for t in range(3):
            m = nt == t
            if m.any():
                h[m] = self.inp[t](f[m])
        qh = self.qp(q)
        ei, et = g["ei"], g["et"]
        if not self.cfg["use_prov"]:
            keep = et > 1; ei, et = ei[:, keep], et[keep]
        if self.cfg.get("no_syn"):
            keep = et != 6; ei, et = ei[:, keep], et[keep]
        for lay in self.layers:
            h = lay(h, qh, ei, et)
        z = self._per_type(self.score, h, nt)
        st = dict(h=h, qh=qh, ei=ei, et=et, z_node=z)
        if self.cfg["use_diff"] and ei.shape[1]:
            r0 = torch.softmax(self._per_type(self.seed, h, nt), 0)
            T = self.diff.transition(h, qh, ei, et)
            r = self.diff.run(r0, T, ei, self.cfg["K"])
            st.update(r0=r0, T=T)
            z = z + self.wr * torch.log(r * X.shape[0] + 1e-4)
        st["z"] = z
        return st

    def _rel_mass(self, st, r0):
        r = self.diff.run(r0, st["T"], st["ei"], self.cfg["K"])
        return self.wr * torch.log(r * r0.shape[0] + 1e-4)

    def decode(self, g, st, budget, gold_seq=None, max_steps=16):
        h, z, qh = st["h"], st["z"], st["qh"]
        nPc, nFc = len(g["P"]), len(g["Fl"])
        nC = nPc + nFc
        hc, zc, he = h[:nC], z[:nC], h[nC:]
        ze = z[nC:]
        cost = g["cost"][:nC]
        isF = torch.zeros(nC, device=h.device); isF[nPc:] = 1
        hn = Fn.normalize(hc, dim=-1)
        ew = torch.sigmoid(ze)
        s = torch.tanh(self.s0(torch.cat([qh, hc.mean(0)], -1)))
        sel = torch.zeros(nC, dtype=torch.bool, device=h.device)
        maxsim = torch.zeros(nC, device=h.device)
        ground = torch.zeros(nC, device=h.device)
        covered = torch.zeros(len(g["E"]), device=h.device)
        inject = torch.zeros(h.shape[0], device=h.device)
        used, losses, order = 0.0, [], []
        W1h = self.W1(hc)
        dyn = torch.zeros(nC, device=h.device)
        T_ = len(gold_seq) + 1 if gold_seq is not None else max_steps
        for t in range(T_):
            lam = Fn.softplus(self.ctrl(s))
            rho = used / budget
            rel = self.w(torch.tanh(W1h + self.W2(s))).squeeze(-1) + zc + dyn
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
            u = (unc.unsqueeze(-1) * he).sum(0) / (ew.sum() + 1e-6)
            nsel = torch.tensor([unc.sum() / (ew.sum() + 1e-6), rho, len(order) / 8.0], device=h.device)
            stop_l = self.stop(torch.cat([s, u, nsel]))
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
            order.append(c); sel[c] = True; used += float(cost[c])
            maxsim = torch.maximum(maxsim, (hn @ hn[c]).clamp(min=0))
            if c < nPc and nFc:
                ground = torch.maximum(ground, torch.cat([ground.new_zeros(nPc), g["prov"][:, c]]))
            covered = covered + g["inc"][c]
            s = self.gru((hc[c] + self.type_emb.weight[int(isF[c])]).unsqueeze(0), s.unsqueeze(0)).squeeze(0)
            if self.cfg["use_diff"] and self.cfg["use_restart"] and "T" in st:
                # selection-as-restart: the chosen evidence becomes a new source of relevance mass
                inject = inject.clone(); inject[c] = 1.0
                r0 = st["r0"] + torch.sigmoid(self.mu) * inject / inject.sum()
                dyn = self._rel_mass(st, r0 / r0.sum())[:nC] - (z[:nC] - st["z_node"][:nC])
        if gold_seq is not None:
            return torch.stack(losses).mean()
        return order

    def forward_loss(self, g, gold_seq, gold_p, gold_f, budget=1200.0, w_node=1.0, w_ground=0.5):
        st = self.encode(g)
        z = st["z"]
        nPc, nFc = len(g["P"]), len(g["Fl"])
        loss = 0.0
        zP = z[:nPc]
        if gold_p:
            loss = loss + w_node * -(Fn.log_softmax(zP, 0)[torch.tensor(gold_p, device=z.device)]).mean()
        if nFc and gold_f:
            zF = z[nPc:nPc + nFc]
            loss = loss + w_node * 0.5 * -(Fn.log_softmax(zF, 0)[torch.tensor([f - nPc for f in gold_f], device=z.device)]).mean()
        if nFc and w_ground > 0:
            pz = (g["prov"] * torch.sigmoid(zP).detach().unsqueeze(0)).max(1).values
            has = g["prov"].sum(1) > 0
            if has.any():
                loss = loss + w_ground * Fn.mse_loss(torch.sigmoid(z[nPc:nPc + nFc])[has], pz[has])
        if self.cfg["use_dec"]:
            loss = loss + self.decode(g, st, budget, gold_seq=gold_seq)
        return loss

    @torch.no_grad()
    def retrieve(self, g, budget=1200.0, max_steps=16, topk_nodec=(5, 0)):
        st = self.encode(g)
        z = st["z"]
        nPc, nFc = len(g["P"]), len(g["Fl"])
        if self.cfg["use_dec"]:
            order = self.decode(g, st, budget, max_steps=max_steps)
            rs = self._last_rel[:nPc]
        else:
            order, used = [], 0.0
            idx = z[:nPc].argsort(descending=True).tolist()[:topk_nodec[0]]
            idx += (z[nPc:nPc + nFc].argsort(descending=True) + nPc).tolist()[:topk_nodec[1]]
            for c in idx:
                if used + float(g["cost"][c]) <= budget:
                    order.append(c); used += float(g["cost"][c])
            rs = z[:nPc]
        sel_p = [g["P"][c] for c in order if c < nPc]
        sel_f = [g["Fl"][c - nPc] for c in order if c >= nPc]
        rank = sel_p + [g["P"][i] for i in rs.argsort(descending=True).tolist() if g["P"][i] not in sel_p]
        return dict(order=order, sel_p=sel_p, sel_f=sel_f, rank=rank)
