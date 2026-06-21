Muon is the strongest baseline: val_loss 2.1995, edging Lion's 2.2028 and well below NAdam's 2.3231, with the best downstream accuracies of the three (arc_easy 60.19, hellaswag 36.85, winogrande 52.17). The ladder now shows a clean trend — each rung that changes the update *geometry* to respect more of the weight matrix's structure does better: NAdam (diagonal, magnitude-aware) → Lion (uniform per entry) → Muon (uniform per singular direction). The trend points at the caveat I logged building Muon: it orthogonalizes from the *instantaneous* momentum geometry alone. Muon's update is the polar factor $UV^\top$ of the current momentum $M$ — a whitening of *this* step, carrying no memory of how the gradient's covariance has looked over training, so two different curvature landscapes that happen to produce the same momentum this step get the same Muon update. The next move is a *history-aware, non-diagonal* preconditioner: accumulated second-order structure on both sides of each weight matrix.

The ideal object is full-matrix Adagrad: vectorize $g=\text{vec}(G)$, accumulate $H=\sum gg^\top$, and step $H^{-1/2}g$, using every pairwise correlation among the $mn$ coordinates. It is hopeless — $H$ is $mn\times mn$, a million-by-million matrix for a $1024\times1024$ weight whose inverse square root I would have to keep current — and Adam's answer of keeping only the diagonal of $H$ is exactly why it is cheap and exactly why it is blind. Shampoo is the tractable middle: approximate $H$ by a Kronecker product of a left factor $L=\sum GG^\top\in\mathbb{R}^{m\times m}$ (row correlations) and a right factor $R=\sum G^\top G\in\mathbb{R}^{n\times n}$ (column correlations), giving the step

$$W \leftarrow W - \eta\, L^{-1/2} G R^{-1/2},$$

which vectorized is preconditioning by $(L\otimes R)^{-1/2}$ (power $1/2$ rather than the original $1/4$ matches the optimal Kronecker approximation of the Adagrad preconditioner, with $L$ trace-normalized so the step carries a $\text{Trace}(L)^{1/2}$ scalar). This is the history-aware preconditioner Muon lacks: $L$ and $R$ *accumulate* the gradient covariance across steps rather than reading one momentum snapshot.

But Shampoo has a cost that decides the rung. To form $L^{-1/2}$ and $R^{-1/2}$ I need an eigendecomposition, cubic in the side length, so no one does it every step — they refresh the roots every $f$ steps and reuse stale ones in between. The trouble is that *Shampoo's adaptivity is the eigen-refresh*: between refreshes the preconditioner is frozen, the only thing that changes it is the periodic recomputation. So as I push $f$ up to keep overhead tolerable the preconditioner goes stale and the benefit erodes — the lever I would pull to make it affordable is the one that kills it. On a 355M-parameter run at a fixed 12,030-iteration budget I cannot afford a frequent eigendecomposition, so naive Shampoo forces me into the stale-and-degrading regime.

The unlock is that Shampoo is secretly diagonal in the right coordinates. Rotate the gradient into the eigenbasis of the two factors: let $Q_L$ hold $L$'s eigenvectors (eigenvalues $\lambda_i$) and $Q_R$ hold $R$'s ($\mu_j$), and set $G' = Q_L^\top G Q_R$. In that basis Shampoo's trace-corrected preconditioner scales coordinate $(i,j)$ by $(\lambda_i\mu_j/\sum\lambda)^{-1/2}$ — a diagonal rescaling. That is suspiciously close to Adafactor's row-column scaling, which scales $(i,j)$ by $(A_iC_j/\sum A)^{-1/2}$ with $A_i,C_j$ the row/column marginals of the squared rotated gradient. They coincide exactly: with $u_i$ the $i$-th eigenvector of $L$, $A_i = \mathbb{E}[\sum_j (u_i^\top G v_j)^2] = \mathbb{E}[\|u_i^\top G\|^2] = u_i^\top \mathbb{E}[GG^\top] u_i = u_i^\top L u_i = \lambda_i$, and symmetrically $C_j=\mu_j$. So Shampoo (power $1/2$, trace correction, dataset-average factors) is *identical* to running a diagonal adaptive optimizer in the eigenbasis of its own preconditioner — and the row/column marginals of $G'\odot G'$ are precisely those $\lambda_i,\mu_j$. The expensive eigendecomposition was never the adaptivity; it was just computing the basis. Once in that basis, the preconditioning is a cheap elementwise rescaling — the second moment of the rotated gradient.

That separation is the whole idea. I propose SOAP: Adam run in Shampoo's eigenbasis. The eigen-refresh only computes the *basis*, and the basis drifts slowly, so I refresh it rarely — every $f$ steps, and even then via one cheap power-iteration step warm-started from the previous eigenvectors plus a QR, not a full eigendecomposition. But in that slowly-drifting basis I run a full adaptive optimizer, updating its second moment **every step** on the rotated gradient. Because the rescaling now lives in a per-step elementwise EMA — the thing that costs nothing — the preconditioner keeps adapting between basis refreshes, precisely what Shampoo could not do: Shampoo froze the rescaling along with the basis, this one does not. And since I am free to pick the diagonal optimizer in that basis, I use Adam rather than Adafactor's rank-1 factorization — Adafactor only helps memory and is itself an approximation; with Adam I keep the true elementwise second moment in the rotated space. The result is history-aware ($L,R$ accumulate covariance), non-diagonal (the rotation couples coordinates), and — unlike Shampoo — gracefully tolerant of a large refresh period, with only *one* hyperparameter beyond AdamW, the frequency $f$.

The per-layer order of operations is where the idea lives. For an $m\times n$ weight I keep four things: the two factors $L$ ($m\times m$) and $R$ ($n\times n$), their eigenbases $Q_L,Q_R$, and Adam's two moments. At step $t$: rotate the gradient into the basis, $G' = Q_L^\top G Q_R$, the coordinate system where the preconditioner is diagonal; update the first moment as the same original-space momentum expressed in the current basis, $M' = Q_L^\top(\beta_1 M + (1-\beta_1)G)Q_R$, so the invariant across a basis change is the original-space momentum (project back before the change, project in after); update the second moment directly in the rotated space *every step*, $V = \beta_2 V + (1-\beta_2)(G'\odot G')$ — the rescaling Shampoo froze and I do not; take the Adam step in the rotated frame, $N' = M'/(\sqrt V + \epsilon)$, with bias corrections folded into the step size as $\sqrt{1-\beta_2^t}/(1-\beta_1^t)$; rotate the result back, $N = Q_L N' Q_R^\top$, and apply it with decoupled weight decay; update the factors by EMA, $L = \beta_2 L + (1-\beta_2)GG^\top$, $R = \beta_2 R + (1-\beta_2)G^\top G$; and only when $t \bmod f = 0$ refresh the eigenbases. The refresh is cheap: form $S = LQ_\text{prev}$ (one matmul) and orthonormalize by QR, $Q = \text{QR}(S)$ — one power-iteration step warm-started from the previous eigenvectors; a genuine eigendecomposition is needed only on the first refresh to initialize $Q$. One bookkeeping detail the structure forces: before the QR, sort the old basis by the estimated eigenvalues $\text{diag}(Q^\top L Q)$ and reorder the matching axis of $V$ the same way, so the diagonal variance table follows the coordinates it belongs to. Setting $Q_L=Q_R=I$ for a layer reduces the whole thing exactly to Adam — a sanity check that this really is Adam in a rotated space.

The finale has to be a fill of the same `configure_optimizers` contract, and the cleanest fit mirrors Muon's structure from rung 3. Like Muon, SOAP applies only to the 2D *hidden* weights — the attention and MLP projections `c_attn`, `c_proj`, `c_fc` — where the operator/covariance story holds. The embedding and LM head are huge on the vocabulary side ($\approx50$k), so maintaining and eigendecomposing a $50\text{k}\times50\text{k}$ factor is prohibitive, and the 1D LayerNorm parameters have no left/right matrix structure at all; both go to plain AdamW, exactly as in Muon, via the same name-based split (exclude `wte`/`wpe`/`lm_head` for the structured optimizer, route those plus the 1D params to AdamW). I reuse Muon's `CombinedOptimizer` shape: a `SOAP` instance over the hidden 2D weights wrapped with a `torch.optim.AdamW` over the rest, exposing merged `param_groups` so the loop's per-group LR scaling reaches both.

The LR handling is where SOAP is *simpler* than Muon, and the difference is worth stating because Muon needed the complexity. Muon's orthogonalized step had unit-ish RMS, so it needed a large `lr_scale` (0.02) and a `CONFIG_OVERRIDES` bump to 1e-3 to keep the two sides commensurate. SOAP's effective step *is* an Adam step in the rotated space — its RMS sits in Adam's own range — so it rides the base cosine schedule directly alongside AdamW with no large `lr_scale`. The one genuinely new knob is `precondition_frequency`, the $f$ above: a moderate value keeps the per-step overhead small while the every-step $V$ update preserves adaptivity, which is the whole property that distinguishes this from the Shampoo it is built on. Weight decay stays the substrate's 0.1, decoupled, on the 2D side; the cosine schedule and grad-clip are untouched, so no `CONFIG_OVERRIDES` are required. The one piece of the full method the harness lets me drop is the distributed-sharding machinery — on two GPUs each SOAP layer keeps its factors and bases whole on its device. One scaffold simplification to be explicit about: this fill keeps the first moment in original coordinates and rotates it with the current basis each step; by linearity that is equivalent to canonical SOAP's in-basis first moment between refreshes, and the rotated second moment is reindexed when the basis axes are sorted.

So the delta from Muon is the move from an *instantaneous* whitening to an *accumulated, history-aware* non-diagonal preconditioner: Muon orthogonalizes this step's momentum, discarding the singular values of a single snapshot; SOAP runs a continuously-adapting Adam in a coordinate system built from the running gradient covariance on both sides of the matrix, keeping the cross-coordinate curvature Muon throws away and adapting the rescaling every step. As the endpoint this rung carries no measured result of its own; the bar is concrete and already on the board — Muon's val_loss 2.1995 and its downstream accuracies — and the reason to expect SOAP can clear it follows the trend the ladder has shown three times: each rung that lets the update respect more of the weight matrix's structure has lowered the loss, and SOAP respects strictly more than Muon. What I would validate on this single-seed FineWeb run at the fixed 12,030-iteration budget is whether val_loss lands below 2.1995 (the primary signal), corroborated by WikiText-2 and LAMBADA perplexity falling below 37.98 and 60.08 and the four downstream accuracies moving past Muon's; and, since the one new knob is the refresh frequency, whether the result stays stable as $f$ is raised — the cheapest confirmation that the per-step $V$ update, not the eigen-refresh, is carrying the adaptivity here.

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
