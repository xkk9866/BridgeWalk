"""UniCoR v3: sufficiency-driven feasible-set decoding.

Shares the v2 encoder (grounded propagation + learned diffusion). The decoder is re-founded as one sequential decision:
  * feasibility  : a fact is selectable only after one of its source passages is selected (hard provenance constraint),
                   so every emitted fact is traceable by construction;
  * sufficiency  : a head psi(S_t) estimates whether the current set already contains a complete evidence chain;
                   decoding stops when psi >= tau (or nothing feasible fits the budget) -- stopping is tied to evidence
                   state, not to the length of oracle sequences;
  * value        : the pointer is trained with an order-invariant multi-reference set loss (any still-missing gold item is
                   a correct next action) on mixed teacher/on-policy rollouts, which removes the unique-order assumption and
                   the teacher-forcing / inference mismatch;
  * restart      : the selected item re-enters the shared diffusion as restart mass weighted by the decoder's own
                   confidence, so low-confidence (possibly wrong) early picks inject little mass.
"""
import random
import torch, torch.nn as nn, torch.nn.functional as Fn
from unicor2 import UniCoR2


class UniCoR3(UniCoR2):
    def __init__(self, hard_prov=True, conf_restart=True, onpolicy=True, gran=False, kappa=1.0, **kw):
        super().__init__(**kw)
        if gran:
            hard_prov = False   # granular mode: a fact is cited by source id instead of duplicating its passage
        self.cfg.update(hard_prov=hard_prov, conf_restart=conf_restart, onpolicy=onpolicy, gran=gran, kappa=kappa)
        d = self.qp.out_features
        self.suff = nn.Sequential(nn.Linear(2 * d + 3, d), nn.GELU(), nn.Linear(d, 1))
        self.tau = 0.5
        self.stop_mode = ("suff", None)   # ("suff",tau) | ("fixed",k) | ("thresh",theta)

    def decode(self, g, st, budget, gold=None, max_steps=16, beta=1.0, inject_first=None):
        h, z, qh = st["h"], st["z"], st["qh"]
        nPc, nFc = len(g["P"]), len(g["Fl"])
        nC = nPc + nFc
        dev = h.device
        hc, zc, he, ze = h[:nC], z[:nC], h[nC:], z[nC:]
        cost = g["cost"][:nC]
        isF = torch.zeros(nC, device=dev); isF[nPc:] = 1
        hn = Fn.normalize(hc, dim=-1)
        ew = torch.sigmoid(ze)
        s = torch.tanh(self.s0(torch.cat([qh, hc.mean(0)], -1)))
        sel = torch.zeros(nC, dtype=torch.bool, device=dev)
        maxsim = torch.zeros(nC, device=dev); ground = torch.zeros(nC, device=dev)
        covered = torch.zeros(len(g["E"]), device=dev)
        inject = torch.zeros(h.shape[0], device=dev)
        used, order, losses, psis = 0.0, [], [], []
        W1h = self.W1(hc); dyn = torch.zeros(nC, device=dev)
        train = gold is not None
        if train:
            gold_p = set(i for i in gold if i < nPc); gold_all = set(gold)
            T_ = len(gold) + 3
            # evidence requirements (one per gold hop) with multi-granular satisfiers: the passage or any of its gold facts
            reqs = [{p} | {f for f in gold_all if f >= nPc and nFc and float(g['prov'][f - nPc, p]) > 0} for p in sorted(gold_p)]
            done_req = [False] * len(reqs)
        else:
            T_ = max_steps
        for t in range(T_):
            lam = Fn.softplus(self.ctrl(s)); rho = used / budget
            rel = self.w(torch.tanh(W1h + self.W2(s))).squeeze(-1) + zc + dyn
            self._last_rel = rel
            logit = rel
            if self.cfg["use_red"]:
                logit = logit - lam[0] * 4 * maxsim
            if self.cfg["use_ground"]:
                logit = logit + lam[1] * 2 * isF * ground
            logit = logit - lam[2] * (cost / 100.0) * rho
            mask = sel.clone()
            if self.cfg["hard_prov"] and nFc:
                mask |= (isF > 0) & (ground <= 0)          # infeasible: fact without a selected source passage
            if self.cfg.get("gran") and nFc:
                mask |= (isF > 0) & (ground > 0)           # redundant: its source passage is already in the set
            if not train:
                mask |= (used + cost) > budget
            logit = logit.masked_fill(mask, -1e4)
            unc = ew * (1 - covered.clamp(max=1))
            u = (unc.unsqueeze(-1) * he).sum(0) / (ew.sum() + 1e-6)
            feat = torch.stack([unc.sum() / (ew.sum() + 1e-6), torch.tensor(rho, device=dev),
                                torch.tensor(len(order) / 8.0, device=dev)])
            psi_l = self.suff(torch.cat([s, u, feat])).squeeze(-1)
            prob = torch.softmax(logit, 0)
            if train and self.cfg.get("gran"):
                complete = all(done_req)
                psis.append(Fn.binary_cross_entropy_with_logits(psi_l, torch.tensor(float(complete), device=dev)))
                if complete:
                    break
                rem = sorted({c for r, dn in zip(reqs, done_req) if not dn for c in r if not bool(mask[c])})
                if not rem:
                    break
                ri = torch.tensor(rem, device=dev)
                lw = -self.cfg["kappa"] * cost[ri] / 100.0          # prefer the cheapest sufficient granularity
                losses.append(-(torch.logsumexp(Fn.log_softmax(logit, 0)[ri] + lw, 0) - torch.logsumexp(lw, 0)))
                if (not self.cfg["onpolicy"]) or random.random() < beta:
                    c = rem[int((logit[ri] + lw).argmax())]
                else:
                    c = int(logit.argmax())
                for k, r in enumerate(reqs):
                    if c in r:
                        done_req[k] = True
            elif train:
                complete = gold_all <= set(order) or (gold_p <= set(order) and not [c for c in gold_all - set(order) if not bool(mask[c])])
                psis.append(Fn.binary_cross_entropy_with_logits(psi_l, torch.tensor(float(complete), device=dev)))
                if complete:
                    break
                rem = [c for c in gold_all - set(order) if not bool(mask[c])]
                if not rem:
                    break
                ri = torch.tensor(rem, device=dev)
                losses.append(-torch.logsumexp(Fn.log_softmax(logit, 0)[ri], 0))   # order-invariant set loss
                if (not self.cfg["onpolicy"]) or random.random() < beta:
                    c = rem[int(logit[ri].argmax())]                                  # model-preferred gold item
                else:
                    c = int(logit.argmax())                                           # on-policy (may be wrong)
            else:
                mode, par = self.stop_mode
                if bool(mask.all()):
                    break
                if t == 0 and inject_first is not None:
                    c = inject_first
                else:
                    if mode == "suff" and t > 0 and torch.sigmoid(psi_l) >= (par if par is not None else self.tau):
                        break
                    if mode == "fixed" and sum(1 for o in order if o < nPc) >= par:
                        break
                    if mode == "thresh" and t > 0 and float(prob.max()) < par:
                        break
                    c = int(logit.argmax())
            order.append(c); sel[c] = True; used += float(cost[c])
            maxsim = torch.maximum(maxsim, (hn @ hn[c]).clamp(min=0))
            if c < nPc and nFc:
                ground = torch.maximum(ground, torch.cat([ground.new_zeros(nPc), g["prov"][:, c]]))
            covered = covered + g["inc"][c]
            s = self.gru((hc[c] + self.type_emb.weight[int(isF[c])]).unsqueeze(0), s.unsqueeze(0)).squeeze(0)
            if self.cfg["use_diff"] and self.cfg["use_restart"] and "T" in st:
                inject = inject.clone()
                inject[c] = float(prob[c].detach()) if self.cfg["conf_restart"] else 1.0
                r0 = st["r0"] + torch.sigmoid(self.mu) * inject / len(order)
                dyn = self._rel_mass(st, r0 / r0.sum())[:nC] - (z[:nC] - st["z_node"][:nC])
        if train:
            ls = (torch.stack(losses).mean() if losses else z.sum() * 0)
            return ls + torch.stack(psis).mean()
        return order

    def forward_loss(self, g, gold_seq, gold_p, gold_f, budget=1200.0, w_node=1.0, w_ground=0.5, beta=1.0):
        st = self.encode(g)
        z = st["z"]; nPc, nFc = len(g["P"]), len(g["Fl"])
        loss = -(Fn.log_softmax(z[:nPc], 0)[torch.tensor(gold_p, device=z.device)]).mean() * w_node
        if nFc and gold_f:
            loss = loss + w_node * 0.5 * -(Fn.log_softmax(z[nPc:nPc + nFc], 0)[torch.tensor([f - nPc for f in gold_f], device=z.device)]).mean()
        if nFc and w_ground > 0:
            pz = (g["prov"] * torch.sigmoid(z[:nPc]).detach().unsqueeze(0)).max(1).values
            has = g["prov"].sum(1) > 0
            if has.any():
                loss = loss + w_ground * Fn.mse_loss(torch.sigmoid(z[nPc:nPc + nFc])[has], pz[has])
        if self.cfg["use_dec"]:
            loss = loss + self.decode(g, st, budget, gold=list(gold_p) + list(gold_f), beta=beta)
        return loss

    @torch.no_grad()
    def retrieve(self, g, budget=1200.0, max_steps=16, topk_nodec=(5, 0), inject_first=None):
        st = self.encode(g)
        nPc = len(g["P"])
        order = self.decode(g, st, budget, max_steps=max_steps, inject_first=inject_first)
        rs = self._last_rel[:nPc]
        sel_p = [g["P"][c] for c in order if c < nPc]
        sel_f = [g["Fl"][c - nPc] for c in order if c >= nPc]
        rank = sel_p + [g["P"][i] for i in rs.argsort(descending=True).tolist() if g["P"][i] not in sel_p]
        return dict(order=order, sel_p=sel_p, sel_f=sel_f, rank=rank)
