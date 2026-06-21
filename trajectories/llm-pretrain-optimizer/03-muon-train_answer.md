Lion did what the geometry argument said: val_loss dropped to 2.2028 from NAdam's 2.3231, a clear structural improvement, with the downstream accuracies rising in step (arc_easy 58.21, hellaswag 35.64). Equalizing the step magnitude across coordinates and leaning on the regularizing noise of the sign was the right kind of move, and the large reliable batch made the sign trustworthy. But the number sits against the ceiling I flagged going in: Lion equalizes magnitude *per coordinate*, treating each weight matrix as a bag of $m\cdot n$ independent scalars. The sign is blind to the fact that a hidden layer's weight is a linear *operator*. That residual blindness is the gap, and the lever this rung pulls is to make the update see the matrix as a matrix.

A concrete fact about the updates I am actually producing points straight at the fix. Take the SGD-momentum or Lion-style updates for the 2D weights of this transformer and look at their singular spectrum: they are nearly low-rank — a huge condition number, a few singular directions carrying almost the entire magnitude, a long tail getting almost nothing. Lion's sign does not address this; it equalizes *entries*, not *directions*, so a sign update can still be dominated by a handful of operator directions while starving the rest. But those rare directions can still matter — there is no reason the loss only cares about the top few. A good matrix update should equalize the step across *singular directions*, not across entries, and that is exactly what a per-entry sign cannot deliver.

I propose Muon: orthogonalize the momentum to its polar factor and step along that. Make the choice precise by reading any first-order optimizer as steepest descent under a chosen norm. Build the local model $\langle G, \Delta W\rangle + \tfrac{\lambda}{2}\|\Delta W\|^2$ and minimize over $\Delta W$, where $G$ is the (momentum) gradient. Splitting $\Delta W = c\cdot T$ into magnitude $c\ge0$ and unit direction $\|T\|=1$ gives $c\langle G,T\rangle + \tfrac\lambda2 c^2$, minimized by $T = -\arg\max_{\|T\|=1}\langle G,T\rangle$ and $c = \|G\|_\dagger/\lambda$, so $\Delta W = -(\|G\|_\dagger/\lambda)\arg\max_{\|T\|=1}\langle G,T\rangle$ — the whole personality of the optimizer is in *which norm* I pick. Lion drops out as a special case: under the $\ell_\infty$ norm on the flattened weights, maximizing $\sum G_{ij}T_{ij}$ with every $|T_{ij}|\le1$ gives $T_{ij}=\text{sign}(G_{ij})$, dual norm $\|G\|_1$. The sign is steepest descent under $\ell_\infty$ — a statement about individual entries that throws away the matrix structure. The fix is to pick a better norm.

A weight matrix maps an input space to an output space, both locally Euclidean, so the natural yardstick is the induced $\ell_2\to\ell_2$ operator norm — the **spectral norm**, the largest singular value, the worst-case stretch. Redo steepest descent with $\|\cdot\|$ the spectral norm. Write the SVD $G = U\Sigma V^\top = \sum_i \sigma_i u_i v_i^\top$; then $\langle G,T\rangle = \sum_i \sigma_i (u_i^\top T v_i)$, and $\|T\|_2=1$ implies $u_i^\top T v_i \le 1$, so $\langle G,T\rangle \le \sum_i \sigma_i$, attained by $T = \sum_i u_i v_i^\top = UV^\top$. So the optimal direction is

$$T^\star = UV^\top,$$

the SVD of $G$ with every singular value replaced by 1 — the orthogonal polar factor, the matrix version of sign. And this is exactly the equalization the spectra were begging for: the raw momentum $G=U\Sigma V^\top$ pours most of its magnitude into the top few $\sigma_i$; replacing $\Sigma$ by the identity gives every singular direction an equal-size step, finally feeding the rare low-$\sigma$ directions Lion's sign was starving. Two independent characterizations land on the same object — $UV^\top$ is both the spectral-norm steepest-descent direction and the closest semi-orthogonal matrix to $G$ in Frobenius distance — which is reassuring that it is not an artifact of one variational problem.

Computing $UV^\top$ by an SVD or Shampoo-style inverse roots every step is too slow and too precision-hungry for a GPU inner loop. The cheapest GPU primitive is matmul in bfloat16, so I want an iteration that converges to $UV^\top$ using only matmuls. The lever: an *odd* matrix polynomial leaves $U$ and $V$ fixed while acting on the singular values as a scalar polynomial — $p(X) = XX^\top X$ sends $U\Sigma V^\top$ to $U\Sigma^3 V^\top$ — so iterating $X_{k+1} = aX_k + b\,X_kX_k^\top X_k + c\,(X_kX_k^\top)^2 X_k$ evolves the singular values under $g(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ while never moving $U,V$. I tune $(a,b,c)$ to drive every $\sigma$ toward 1 with a large slope at zero (so the many small singular values lift fast), accepting that they end up scattered around 1 rather than exactly 1 — equalization to within a factor of $\sim$3 is all I need, and the loss does not care about the difference. The tuned $(3.4445, -4.7750, 2.0315)$ give $g'(0)\approx3.44$, and **5 iterations** suffice. The one precondition: the iteration only contracts inside its basin, so I Frobenius-normalize first ($X_0 = G/\|G\|_F$ puts every $\sigma\le1$), which is free of consequence for the direction since scaling $G$ scales all $\sigma$ equally and leaves $U,V$ alone.

This runs only on the 2D *hidden* matrices — the attention and MLP projections — because the derivation assumed a weight that acts as an operator on a Euclidean hidden space. The embedding table and the LM head are technically 2D but do not act that way (a vocabulary lookup and a class scorer are not operators whose singular directions I should equalize), and the 1D parameters have no matrix structure at all; those keep an adaptive per-entry optimizer.

This task's Muon diverges from the generic method in several harness-specific ways, and the trajectory has to land *that* implementation. First, momentum: the canonical form is SGD-momentum $M=\mu M+G$, but this task uses an **EMA momentum with a Nesterov blend** — `buf.lerp_(g, 1−momentum)` so $\text{buf}=(1-\mu)g+\mu\,\text{buf}$, then `nesterov_g = g.lerp(buf, momentum)` so the orthogonalized input is $(1-\mu)g+\mu\,\text{buf}$, a look-ahead, with $\mu=0.95$ (the same look-ahead idea from rung 1, now feeding the orthogonalization rather than a diagonal step). Second, specific to GPT-2's fused attention: the `c_attn` weight is a single $3\cdot n_\text{embd}\times n_\text{embd}$ matrix holding Q, K, V stacked — three separate operators, so orthogonalizing the stack as one is wrong. The edit **splits any matrix whose row count is 3× its column count into three parts**, runs Newton–Schulz on each, concatenates, and uses $\text{scale} = \max(1,3)^{0.5}=\sqrt3$. Third, the shape-dependent scale: the canonical method cancels the orthogonalized update's $\sqrt{1/\max(A,B)}$ RMS with $\sqrt{\max(A,B)}$ and a 0.2 prefactor to match AdamW's RMS; *this* edit instead uses $\text{scale} = \max(1, A/B)^{0.5}$ — a ratio correction with no explicit 0.2 — and absorbs the overall magnitude into the learning rate. Dropping the 0.2 RMS-match matters: I cannot lean on Muon and AdamW sharing one learning rate by construction, which is why the LR handling is so explicit.

That LR handling is the fourth and most harness-coupled difference. The orthogonalized step has unit-ish RMS, so it wants a much larger LR than Adam ($\sim$0.02 vs $\sim6\times10^{-4}$). The edit sets `muon_base_lr = 0.02`, computes `muon_lr_scale = muon_base_lr / learning_rate`, and attaches it as `lr_scale` on the Muon group; the substrate's loop multiplies every group's LR by its `lr_scale` each step, so Muon rides the *shape* of the cosine schedule at $\sim$0.02 peak while the AdamW side rides it at base rate — the `lr_scale` hook used as advertised. To keep the base rate sane for the AdamW side, the edit sets `CONFIG_OVERRIDES = {'learning_rate': 1e-3}`, bumping the substrate's 6e-4 to 1e-3 — the first time the ladder touches a `CONFIG_OVERRIDES` value. Fifth, the two optimizers are combined: a `CombinedOptimizer` holds a `Muon` instance for the 2D projection weights (identified by *excluding* names with `wte`/`wpe`/`lm_head`) and a `torch.optim.AdamW` for the embedding/head 2D params plus all 1D params, exposing merged `param_groups` so the per-group LR scaling reaches both. Weight decay 0.1 is applied decoupled inside Muon (before the update) and via AdamW for the rest. The one piece of the full method the harness lets me drop is the distributed reduce-scatter/gather machinery — on two GPUs each Muon param is orthogonalized whole on its device.

So the delta from Lion is the deepest geometry change yet: Lion equalized the update *per entry* ($\ell_\infty$ steepest descent), Muon equalizes it *per singular direction* (spectral-norm steepest descent), precisely the structure Lion was blind to. Giving the starved directions a full unit of update should let the model use more of its capacity, so I expect val_loss below 2.2028 and arc_easy/hellaswag past 58.21/35.64 — likely the strongest of the three baselines. The caveat I carry forward: Muon orthogonalizes from the *instantaneous* momentum geometry alone — it whitens this step's direction but accumulates no running estimate of the gradient's cross-coordinate curvature, and it leaves the embedding and head on plain AdamW. If a rung beyond this is needed, the place to look is a *history-aware*, non-diagonal preconditioner — accumulated second-order structure on both sides of the matrix — rather than a per-step orthogonalization.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 3: Muon (2D hidden) + AdamW (rest)
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [(n, p) for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [(n, p) for n, p in param_dict.items() if p.dim() < 2]
        num_decay_params = sum(p.numel() for _, p in decay_params)
        num_nodecay_params = sum(p.numel() for _, p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Separate 2D projection weights (for Muon) from rest (for AdamW)
        muon_params = [p for n, p in decay_params
                       if 'wte' not in n and 'wpe' not in n and 'lm_head' not in n]
        adam_decay_params = [p for n, p in decay_params
                            if 'wte' in n or 'wpe' in n or 'lm_head' in n]
        adam_nodecay_params = [p for _, p in nodecay_params]

        class Muon(torch.optim.Optimizer):
            """Muon — MomentUm Orthogonalized by Newton-schulz."""
            def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
                defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
                super().__init__(params, defaults)

            @staticmethod
            def zeroth_power_via_newtonschulz5(G, steps=5):
                assert G.ndim == 2
                a, b, c = (3.4445, -4.7750, 2.0315)
                X = G.bfloat16()
                X = X / (X.norm() + 1e-7)
                if G.size(0) > G.size(1):
                    X = X.T
                for _ in range(steps):
                    A = X @ X.T
                    X = a * X + b * (A @ X) + c * (A @ (A @ X))
                if G.size(0) > G.size(1):
                    X = X.T
                return X

            @torch.no_grad()
            def step(self):
                for group in self.param_groups:
                    lr = group['lr']; momentum = group['momentum']; wd = group.get('weight_decay', 0.0)
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        if wd > 0:
                            p.mul_(1 - lr * wd)
                        g = p.grad
                        state = self.state[p]
                        if len(state) == 0:
                            state['momentum_buffer'] = torch.zeros_like(g)
                        buf = state['momentum_buffer']
                        buf.lerp_(g, 1.0 - momentum)                 # EMA momentum
                        nesterov_g = g.lerp(buf, momentum)           # Nesterov look-ahead
                        if nesterov_g.dim() == 2:
                            orig_shape = nesterov_g.shape
                            if orig_shape[0] == 3 * orig_shape[1]:   # split fused QKV
                                parts = nesterov_g.split(orig_shape[1])
                                update = torch.cat([
                                    self.zeroth_power_via_newtonschulz5(part, steps=group['ns_steps'])
                                    for part in parts])
                                scale = max(1, orig_shape[0] // orig_shape[1]) ** 0.5
                            else:
                                update = self.zeroth_power_via_newtonschulz5(nesterov_g, steps=group['ns_steps'])
                                scale = max(1, orig_shape[0] / orig_shape[1]) ** 0.5
                            p.data.add_(update.to(p.dtype), alpha=-lr * scale)
                        else:
                            p.add_(buf, alpha=-lr)

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        muon_base_lr = 0.02
        muon_lr_scale = muon_base_lr / learning_rate
        muon_opt = Muon([{'params': muon_params, 'lr_scale': muon_lr_scale}],
                        lr=muon_base_lr, momentum=0.95, weight_decay=0.1)
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

        print(f"using Muon (lr={muon_base_lr}, scale={muon_lr_scale:.1f}) + AdamW combined optimizer")
        return CombinedOptimizer([muon_opt, adam_opt])

# ... and in the __main__ config block:
    # CONFIG_OVERRIDES: override training hyperparameters for your method.
    CONFIG_OVERRIDES = {'learning_rate': 1e-3}
```
