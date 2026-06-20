**Problem (from prior art).** The floor recipe trains ResNet-50 with stock SGD, where weight decay is
folded into the gradient as L2 regularization and the per-step shrink is η·λ·θ — the decay is silently
scaled by the learning rate. Since the whole ladder is about aggressively sweeping the LR and the schedule
to chase the time-accuracy frontier, this couples regularization strength to the learning rate: every LR
change secretly changes the effective weight decay, so regularization can't be tuned honestly.

**Key idea.** Decouple weight decay from the learning rate. Instead of adding λθ to the gradient (so SGD
scales it by η), apply weight decay as a separate multiplicative shrink of the parameters each step —
`θ ← (1 − λ′)·θ` — and run the SGD/momentum step on the *bare* loss gradient. The shrink factor λ′ no
longer rides the LR schedule, so learning rate and regularization become independent knobs. Crucially, the
decay must *not* be re-tied to the scheduler (the common "schedule the weight decay alongside the LR" trick
reintroduces the same coupling through the initial LR).

**Why it works.** `θ ← θ − η·(∇L + λθ)` applies shrink η·λ·θ, which equals the clean shrink λ′·θ only at
one learning rate; once η varies (warmup, decay, sweeps), the L2 form drifts. Applying the decay as a
fixed multiplicative factor on the parameter removes the η dependence, so two runs with different peak LRs
get the same regularization. Consequence: the right numeric `weight_decay` is now *smaller* (it absorbed
the η factor that used to multiply it) — for ResNet-50, ~2e-3 rather than the larger coupled value. This is
a correctness fix, not a quality jump: properly tuned it has no known downside, and it makes every
downstream LR/schedule sweep mean what it says.

**Change / code.** Swap `torch.optim.SGD` for `DecoupledSGDW`. The decay is applied directly to the
parameter (`param.mul_(1 - decay_factor * weight_decay)`) and is *separate* from the gradient step
(`param.add_(d_p, alpha=-lr)`); `decay_factor = lr / initial_lr` follows the LR schedule's *shape* but is
not scaled by `lr` itself.

```python
from composer.optim import DecoupledSGDW

optimizer = DecoupledSGDW(
    model.parameters(),
    lr=0.05,
    momentum=0.9,
    weight_decay=2.0e-3,   # small: no longer multiplied by lr
)

# The functional core (DecoupledSGDW.sgdw): decay is a separate multiplicative
# shrink, applied to the parameter, NOT folded into the gradient.
@staticmethod
def sgdw(params, d_p_list, momentum_buffer_list, *,
         weight_decay, momentum, lr, initial_lr, dampening, nesterov):
    for i, param in enumerate(params):
        d_p = d_p_list[i]

        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            d_p = d_p.add(buf, alpha=momentum) if nesterov else buf

        if weight_decay != 0:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            param.mul_(1 - decay_factor * weight_decay)   # decoupled shrink

        param.add_(d_p, alpha=-lr)                        # bare-gradient step
```
