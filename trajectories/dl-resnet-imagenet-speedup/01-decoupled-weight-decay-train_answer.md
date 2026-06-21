The floor I am building on is a vanilla ResNet-50 / ImageNet run with stock SGD-with-momentum, and the whole ladder ahead is going to lean hard on tuning the learning rate, the schedule, and the regularization. So before I stack a single speedup I want the optimizer to be telling me the truth, and there is one place where it is not. In the textbook recipe, "weight decay" and "L2 regularization" are treated as synonyms: you add $\tfrac{\lambda}{2}\|\theta\|^2$ to the loss, its gradient contributes $+\lambda\theta$, and SGD subtracts $\eta\,(g + \lambda\theta)$. The per-step pull toward zero is therefore $\eta\cdot\lambda\cdot\theta$ — the decay is silently multiplied by the learning rate. Since $\eta$ sweeps from near zero up to its peak and back down over a warmup-then-decay schedule, the *effective* weight decay rides that schedule up and down, and two runs with the same $\lambda$ but different peak learning rates carry genuinely different regularization. That coupling is poison for everything I am about to do: every time I scale the schedule or retune the peak LR to chase the time-accuracy frontier, I would also, without meaning to, be changing the regularization strength. I cannot tune "how much regularization" independently from "how fast I learn" when the two knobs are physically the same knob wearing two labels.

The fix I propose is **Decoupled Weight Decay**, applied to SGD as `DecoupledSGDW`. The principle is to stop pushing the decay through the gradient and instead apply it directly to the parameters as a separate multiplicative shrink. Write the two updates side by side to see why they differ. The L2-regularized form is

$$\theta \leftarrow \theta - \eta\,(\nabla L(\theta) + \lambda\theta) = \theta - \eta\,\nabla L(\theta) - \eta\,\lambda\,\theta,$$

so the shrink applied per step is $(\eta\lambda)\,\theta$. What I actually want weight decay to be is a fixed multiplicative shrink of the weights each step, independent of the gradient and of $\eta$:

$$\theta \leftarrow (1 - \lambda')\,\theta - \eta\,\nabla L(\theta).$$

These coincide only if $\lambda' = \eta\lambda$ — that is, at exactly one learning rate. The moment $\eta$ varies through warmup, decay, or a sweep, the L2 form drifts away from the clean decay form. Applying the decay as a standalone factor $\lambda'$ on the parameter removes the $\eta$ dependence entirely, so two runs with different peak LRs get the same regularization and the two knobs become independent.

There is a trap I have to avoid even when reaching for a stock "decoupled" optimizer. The naive way frameworks implement it is to schedule the weight decay alongside the learning rate, scaling the decay down in lockstep as the scheduler scales $\eta$ down. That reintroduces exactly the coupling I am trying to kill, just tied now to the *initial* learning rate. The version I want applies a genuinely schedule-independent shrink: the decay factor follows the LR schedule's *shape* (so the relative annealing is preserved) but is not itself scaled by $\eta$. Concretely the implementation uses `decay_factor = lr / initial_lr`, which tracks the schedule's shape, and multiplies the parameter by `1 - decay_factor * weight_decay` as a step entirely separate from the bare-gradient update `param.add_(d_p, alpha=-lr)`.

One numeric consequence follows directly and I have to respect it when tuning. Because the decay is no longer multiplied by $\eta$, the value of $\lambda'$ that gives the right regularization will be *smaller* than the old coupled $\lambda$ — it has to absorb the $\eta$ factor that used to scale it up, so the right $\lambda'$ is roughly $\eta\lambda$ in magnitude. Drop in the old value and I will badly over-regularize; the move is to switch the optimizer and re-tune from scratch. For ResNet-50 the setting that lands is $\text{lr}=0.05$, momentum $0.9$, and $\text{weight\_decay}\approx 2\times 10^{-3}$ — small precisely because it is no longer being scaled up by the learning rate. This rung does not buy a headline accuracy jump; it is a correctness fix with no known downside once properly tuned. What it buys is leverage: from here on, every LR and schedule sweep I run downstream means exactly what it says, which is the honest foundation the rest of the stack gets built on.

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
