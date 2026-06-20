**Problem (from step 9).** The recipe is a tight balance: every throughput lever (progressive resizing,
ColOut, stochastic depth) cashes in accuracy that a quality lever must refill to stay above the 76.6% target.
The binding constraint is now *quality margin* — more generalization in hand would let the schedule run even
leaner. The stacked penalty-style regularizers are at diminishing returns; the finale needs to change *what
kind of minimum* is found, not add another penalty.

**Key idea — Sharpness-Aware Minimization (SAM).** Flat minima generalize because the test surface is a
shifted train surface: a sharp minimum's loss spikes under the shift, a flat one stays low. So minimize the
*worst-case* loss in a ρ-ball around θ: `min_θ max_{‖ε‖≤ρ} L(θ+ε)`. First-order, the worst nearby point is
`ε* ≈ ρ·∇L(θ)/‖∇L(θ)‖` (steepest ascent to the ball's edge), and the objective's gradient is `∇L` evaluated
*at θ+ε*. So each step: (1) gradient at θ → form ε*, ascend to θ+ε*; (2) gradient at θ+ε*; undo the
perturbation back to θ; (3) base optimizer (DecoupledSGDW) steps with the second gradient. Descending on the
worst nearby point drags the whole neighborhood down — flattening.

**Why it works.** It directly optimizes for robustness to the train→test perturbation, the thing that
separates generalizing from non-generalizing minima — a different axis from the stacked regularizers, so it
keeps paying off. Cost: two forward-backward passes per step (≈ halves throughput). Managed with an
`interval` — run the full SAM step once every k steps, ordinary steps in between — the throughput-vs-quality
dial; `interval=10` with `rho` scaled up (vs `rho=0.05` at `interval=1`) is the practical sweet spot.
Distributed nicety: skip the all-reduce on the first gradient (it only locates ε* locally), recovering
throughput and adding helpful stochasticity. SAM helps only in the overfitting regime (data seen multiple
times) — ImageNet multi-epoch classification qualifies; single-pass LM pretraining does not. It's the largest
generalization lever in the recipe, and that margin lets every speed lever below it run aggressively and
still land on 76.6% top-1.

**Change / code.** Wrap the base optimizer in the SAM optimizer; the core is the two-step ascent/descent and
the interval-gated `step` driven by a forward-backward closure.

```python
@torch.no_grad()
def first_step(self):
    grad_norm = self._grad_norm()
    for group in self.param_groups:
        scale = group['rho'] / (grad_norm + group['epsilon'])
        for p in group['params']:
            if p.grad is None:
                continue
            e_w = p.grad * scale.to(p)
            p.add_(e_w)                       # climb to the local maximum "w + e(w)"
            self.state[p]['e_w'] = e_w

@torch.no_grad()
def second_step(self):
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None or 'e_w' not in self.state[p]:
                continue
            p.sub_(self.state[p]['e_w'])      # back to "w" from "w + e(w)"
    self.base_optimizer.step()                # the sharpness-aware update

@torch.no_grad()
def step(self, closure=None):
    assert closure is not None, 'SAM requires a closure (full forward-backward pass)'
    closure = torch.enable_grad()(closure)
    loss = None
    if (self.global_step + 1) % self.interval == 0:
        loss = closure(ddp_sync=False)        # gradient at (w), per-GPU, no sync
        if loss:
            self.first_step()                 # set weights to (w + e(w))
            loss_dict = {}
            if closure(loss_dict=loss_dict):  # gradient at (w + e(w))
                self.second_step()            # reset to (w) and step base optimizer
            else:
                self.sub_e_w()                # second pass failed: restore (w)
    else:
        loss = closure()                      # ordinary single-gradient step
        if loss:
            self.base_optimizer.step()
    self.global_step += 1
    return loss
```
