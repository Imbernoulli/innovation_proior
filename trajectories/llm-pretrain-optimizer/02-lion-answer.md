**Problem.** NAdam landed at the AdamW floor (val_loss 2.3231) because refining the momentum of a
*diagonal* step leaves the step diagonal. To move validation loss, change the update geometry. The
cheapest geometry change: replace the per-coordinate adaptive magnitude with a uniform one.

**Key idea.** Lion steps every coordinate by the same magnitude — the *sign* of a momentum-blended
gradient — dropping Adam's second moment entirely (one buffer, not two). The sign discards gradient
magnitude, injecting uniform update noise that acts as a regularizer (favoring flatter minima), and it
likes the large reliable batch this run already has. Two momentum constants do two jobs: the *step* is
`sign(β₁·m + (1−β₁)·g)` (recency-weighted), while the *buffer* updates on its own slower constant
`m ← β₂·m + (1−β₂)·g` (longer memory).

**Step-2 edit (this task's Lion, not the paper recipe).** The harness fixes choices the standalone
method would set differently: betas are the substrate's **(0.9, 0.95)** (β₂ = 0.95, a *shorter* buffer
memory than Lion's usual 0.99); the learning rate is **0.3 ×** the schedule value (not 0.1×); weight
decay is the substrate's **0.1 on 2D / 0 on 1D**, *not* raised, applied as a decoupled shrink **before**
the sign step. Default dimension-based grouping kept; `get_lr` and `CONFIG_OVERRIDES` untouched (the 0.3
factor is in the constructor).

**What to watch.** A uniform sign step changes what the update is, so expect a clear drop below NAdam's
2.3231 and downstream accuracies rising with it. Ceiling caveat: the sign step still treats a weight
matrix as a bag of scalars — equalized in magnitude, but blind to the matrix's operator structure. That
blindness is the next rung's target.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 2: Lion (sign-momentum), task config
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        class Lion(torch.optim.Optimizer):
            """Lion optimizer — sign-based updates with EMA momentum."""
            def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
                defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
                super().__init__(params, defaults)
            @torch.no_grad()
            def step(self):
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        grad = p.grad
                        state = self.state[p]
                        if len(state) == 0:
                            state['exp_avg'] = torch.zeros_like(p)
                        exp_avg = state['exp_avg']
                        beta1, beta2 = group['betas']
                        # Weight decay first (decoupled, before update)
                        if group['weight_decay'] != 0:
                            p.mul_(1 - group['lr'] * group['weight_decay'])
                        update = exp_avg * beta1 + grad * (1 - beta1)
                        p.add_(torch.sign(update), alpha=-group['lr'])
                        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        optimizer = Lion(optim_groups, lr=learning_rate * 0.3, betas=betas)
        print("using Lion optimizer")
        return optimizer
```
