**Problem.** Muon (val_loss 2.1995, the strongest baseline) orthogonalizes the update from the
*instantaneous* momentum geometry — it whitens this step's direction but accumulates no running estimate
of the gradient's cross-coordinate curvature, and leaves embeddings / LM head on plain AdamW. The next
move on the geometry trend is a *history-aware, non-diagonal* preconditioner: accumulated second-order
structure on both sides of each weight matrix.

**Key idea.** Shampoo is the tractable accumulated preconditioner — running L = ΣGGᵀ, R = ΣGᵀG, step
L^{-1/2} G R^{-1/2} = (L⊗R)^{-1/2} vectorized — but its L^{-1/2}, R^{-1/2} need an eigendecomposition
affordable only every f steps, and since Shampoo's adaptivity *is* that refresh, it goes stale and
degrades as f grows. The unlock: Shampoo (power 1/2, trace-corrected) equals running a diagonal adaptive
optimizer in the *eigenbasis* of L and R — the eigendecomposition only computes the basis; the
rescaling in that basis is just the second moment of the rotated gradient. So **SOAP** refreshes the
basis rarely but runs a full **Adam in that basis, updating its second moment every step** — Shampoo's
accumulated curvature, continuously adaptive, with one new hyperparameter over AdamW (the refresh
frequency f) and graceful behavior at large f.

**Finale edit.** Mirror Muon's structure in the same `configure_optimizers` contract: a `SOAP` instance
over the 2D *hidden* weights (attention/MLP projections — `c_attn`, `c_proj`, `c_fc`) wrapped with a
`torch.optim.AdamW` over the embedding / LM head / 1D params, via a `CombinedOptimizer` exposing the
merged `param_groups`. SOAP's effective LR is in Adam's range (the rotated-space step *is* an Adam step),
so it rides the base cosine schedule with no large `lr_scale`; the one genuinely new knob is
`precondition_frequency`. This distilled task scaffold keeps the first moment in original coordinates
and rotates it with the current basis each step; by linearity this is equivalent to canonical SOAP's
in-basis first moment between refreshes, and the rotated second moment is reindexed when the basis axes
are sorted. Weight decay 0.1 decoupled on the 2D side; `get_lr` and grad-clip untouched.

**The bar (no run yet — finale).** Must clear Muon's val_loss 2.1995 and its downstream accuracies
(arc_easy 60.19, hellaswag 36.85, winogrande 52.17) on the single-seed FineWeb run at 12,030 iterations.
What I would validate: val_loss below 2.1995 (primary), corroborated by WikiText-2 / LAMBADA perplexity
and the four downstream accuracies past Muon's; and — since the one new knob is the refresh frequency —
stability of the result as f is raised, the property that distinguishes SOAP from the Shampoo it builds
on.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — finale: SOAP (2D hidden) + AdamW (rest)
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [(n, p) for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [(n, p) for n, p in param_dict.items() if p.dim() < 2]
        num_decay_params = sum(p.numel() for _, p in decay_params)
        num_nodecay_params = sum(p.numel() for _, p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # 2D hidden projection weights -> SOAP; embeddings/head/1D -> AdamW (same split as Muon)
        soap_params = [p for n, p in decay_params
                       if 'wte' not in n and 'wpe' not in n and 'lm_head' not in n]
        adam_decay_params = [p for n, p in decay_params
                             if 'wte' in n or 'wpe' in n or 'lm_head' in n]
        adam_nodecay_params = [p for _, p in nodecay_params]

        class SOAP(torch.optim.Optimizer):
            """AdamW run in the eigenbasis of Shampoo's (L, R) preconditioner. 2D layers only."""
            def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), eps=1e-8,
                         weight_decay=0.0, precondition_frequency=10, shampoo_beta=-1.0):
                defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                precondition_frequency=precondition_frequency,
                                shampoo_beta=shampoo_beta)
                super().__init__(params, defaults)

            @staticmethod
            def _eigh_basis(P):
                P32 = P.float()
                eye = torch.eye(P32.shape[0], device=P32.device, dtype=P32.dtype)
                _, Q = torch.linalg.eigh(P32 + 1e-30 * eye)
                return torch.flip(Q, dims=[1]).to(P.dtype)     # descending eigenvalue order

            @staticmethod
            def _qr_basis(P, Q_prev, V, dim):
                # sort the carried basis by estimated eigenvalue, keeping V aligned to the axes
                P32, Q32 = P.float(), Q_prev.float()
                est_eig = torch.diag(Q32.T @ P32 @ Q32)
                idx = torch.argsort(est_eig, descending=True)
                V = V.index_select(dim, idx)
                Q32 = Q32[:, idx]
                Q32, _ = torch.linalg.qr(P32 @ Q32)            # one power-iteration step + QR
                return Q32.to(Q_prev.dtype), V

            @torch.no_grad()
            def step(self):
                for group in self.param_groups:
                    beta1, beta2 = group['betas']
                    lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
                    f = group['precondition_frequency']
                    shampoo_beta = group['shampoo_beta'] if group['shampoo_beta'] >= 0 else beta2
                    for p in group['params']:
                        if p.grad is None or p.grad.dim() != 2:
                            continue
                        G = p.grad
                        state = self.state[p]
                        m, n = G.shape
                        if 'step' not in state:                # init preconditioner, skip first update
                            state['step'] = 0
                            state['exp_avg'] = torch.zeros_like(p)
                            state['exp_avg_sq'] = torch.zeros_like(p)
                            state['L'] = torch.zeros(m, m, device=G.device, dtype=G.dtype)
                            state['R'] = torch.zeros(n, n, device=G.device, dtype=G.dtype)
                            state['L'].add_(G @ G.T, alpha=1 - shampoo_beta)
                            state['R'].add_(G.T @ G, alpha=1 - shampoo_beta)
                            state['QL'] = self._eigh_basis(state['L'])
                            state['QR'] = self._eigh_basis(state['R'])
                            continue
                        QL, QR = state['QL'], state['QR']
                        M, V = state['exp_avg'], state['exp_avg_sq']
                        state['step'] += 1
                        t = state['step']

                        G_rot = QL.T @ G @ QR                  # rotate gradient into eigenbasis
                        M.mul_(beta1).add_(G, alpha=1 - beta1) # first moment in original space
                        M_rot = QL.T @ M @ QR                  # rotate momentum in
                        V.mul_(beta2).addcmul_(G_rot, G_rot, value=1 - beta2)  # 2nd moment in basis, every step

                        denom = V.sqrt().add_(eps)
                        step_size = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                        N = QL @ (M_rot / denom) @ QR.T        # rotate the Adam step back

                        p.add_(N, alpha=-step_size)
                        if wd > 0:
                            p.add_(p, alpha=-lr * wd)          # decoupled weight decay

                        state['L'].mul_(shampoo_beta).add_(G @ G.T, alpha=1 - shampoo_beta)
                        state['R'].mul_(shampoo_beta).add_(G.T @ G, alpha=1 - shampoo_beta)
                        if t % f == 0:                        # refresh basis only every f steps
                            QL, V = self._qr_basis(state['L'], QL, V, 0)  # reindex 2nd moment to new axes
                            QR, V = self._qr_basis(state['R'], QR, V, 1)
                            state['QL'], state['QR'], state['exp_avg_sq'] = QL, QR, V

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        soap_opt = SOAP([{'params': soap_params}],
                        lr=learning_rate, betas=betas, weight_decay=weight_decay,
                        precondition_frequency=10)
        adam_groups = [
            {'params': adam_decay_params, 'weight_decay': weight_decay},
            {'params': adam_nodecay_params, 'weight_decay': 0.0},
        ]
        adam_opt = torch.optim.AdamW(adam_groups, lr=learning_rate, betas=betas, **extra_args)

        class CombinedOptimizer:
            def __init__(self, optimizers):
                self.optimizers = optimizers
                self.param_groups = []
                for opt in optimizers:
                    self.param_groups.extend(opt.param_groups)
            def zero_grad(self, set_to_none=True):
                for opt in self.optimizers:
                    opt.zero_grad(set_to_none=set_to_none)
            def step(self):
                for opt in self.optimizers:
                    opt.step()
            def state_dict(self):
                return [opt.state_dict() for opt in self.optimizers]

        print("using SOAP (2D hidden) + AdamW combined optimizer")
        return CombinedOptimizer([soap_opt, adam_opt])
```
