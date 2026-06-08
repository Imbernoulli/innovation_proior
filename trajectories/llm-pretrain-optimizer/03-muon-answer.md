**Problem.** Lion (val_loss 2.2028) equalizes the step *per entry* via the sign, but still treats a
weight matrix as a bag of scalars — blind to its operator structure, so a few singular directions can
dominate the actual change. The next geometry change is to equalize the step *per singular direction*.

**Key idea.** Read an optimizer step as steepest descent under a norm. The sign is steepest descent
under ℓ∞ (per-entry); the norm that respects a weight matrix as an operator is the **spectral norm**,
whose steepest-descent direction is the polar factor `UVᵀ` of the (momentum) gradient — the SVD with
every singular value set to 1. This orthogonalization gives every singular direction a unit step,
finally feeding the rare directions a per-entry method starves. Compute `UVᵀ` with a 5-step
Newton–Schulz iteration (matmuls only, bfloat16). Applied to 2D hidden weights; AdamW for
embeddings / LM head / 1D params.

**Step-3 edit (this task's Muon, not the canonical recipe).** EMA momentum with a Nesterov blend
(`buf ← (1−μ)g + μ·buf`; orthogonalize `(1−μ)g + μ·buf`, μ = 0.95), not plain SGD-momentum. GPT-2's
fused `c_attn` (3·n_embd × n_embd) is **split into Q/K/V**, each orthogonalized separately, scale `√3`;
other matrices use `scale = max(1, A/B)^0.5` (a ratio correction, not the canonical `0.2·√max(A,B)`).
Muon runs at `muon_base_lr = 0.02` via an `lr_scale = 0.02/lr` group (riding the cosine schedule's
shape); `CONFIG_OVERRIDES = {'learning_rate': 1e-3}` raises the AdamW-side base. A `CombinedOptimizer`
wraps Muon (2D projection weights) + AdamW (wte/wpe/lm_head + 1D), weight decay 0.1 decoupled.

**What to watch.** The most principled per-matrix step on the ladder; expect val_loss below Lion's
2.2028 and downstream accuracies (arc_easy, hellaswag) past 58.21 / 35.64 — likely the strongest
baseline. Caveat for any further rung: Muon orthogonalizes from the *instantaneous* momentum geometry
only — no accumulated, history-aware non-diagonal preconditioner — and leaves embeddings/head on AdamW.

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
