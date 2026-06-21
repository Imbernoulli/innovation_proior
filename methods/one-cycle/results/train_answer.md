What wastes the most time when I train a deep residual network is not the optimizer, the loss, or the data — it is the single knob I am allowed to move per step, the learning rate, and the way the standard recipe uses it. With SGD and momentum at a rate around $0.1$, every run draws the same curve: a burst of progress over the first handful of epochs, then a long flat plateau where each iteration buys almost nothing, and only when I divide the rate by ten does accuracy lurch upward, plateau again, drop again. Two or three of those drops and eighty thousand iterations have gone by, most of them spent in plateaus. Step decay is monotone non-increasing — it never raises the rate, so it never crosses the flat valley quickly; it just grinds through with smaller and smaller steps, demands hand-chosen milestone epochs per dataset and architecture, and keeps the rate at conventionally small values the whole way, never probing the large-rate band at all. The cyclical-learning-rate work showed that a temporary rate increase can help — its triangular policy oscillates linearly between a lower and an upper bound for the whole run, justified by the saddle-point argument that a larger step traverses a plateau a small step stalls on — but it runs many full up-and-down cycles between two fixed bounds and never drives the rate below the lower one. SGDR cosine-decays from high to low and then jumps back to the top to restart, a sawtooth, rather than ramping once. Warmup-then-decay is only an up-leg followed by ordinary decay with momentum left constant. Each gets a piece; none reaches a target accuracy in far fewer iterations while preserving generalization, and none sets its bounds from a single short pre-run instead of a grid search.

The shape I want comes straight off the loss topology. The trajectory makes steep fast progress early, then enters a long nearly-flat valley (and saddle plateaus, where the gradient is small in many directions and descent stalls), and only at the end threads a narrow trough down to a minimum. Overlay the learning rate on that: in the steep early region a large step would overshoot, so I want the rate *small*; in the flat valley a small rate times a tiny slope is a microscopic step, so I want it *large*, to cross the valley fast and punch through the saddles; threading the final trough, a large rate would bounce me out, so I want it *small* again. Small, then large, then small — one rise and one fall, not a monotone decay and not many cycles. And because the trough is reached only at the very end, the fall should not stop where the rise began: it should keep going, annealing the rate orders of magnitude *below* the start for the remaining iterations, so that after the high-rate phase has carried SGD into a wide flat region the deep anneal drops it into a steeper minimum inside that region and locks it in. This is curriculum (easy-to-hard-to-settle) welded to simulated annealing (cool to a quiet low-temperature search) inside a single cycle. I propose this as the one-cycle policy, "1cycle": one cosine climb from a small initial rate to a large maximum over roughly the first $30\%$ of the run, then a long cosine descent from the maximum down to a rate far below where it started.

The whole thing hinges on "large" meaning genuinely large, not $0.15$ instead of $0.1$. To find out how large without a per-architecture grid search I use the LR range test: one short pre-run that starts the rate near zero and increases it linearly, plotting test accuracy against the rising rate. While the rate is small accuracy climbs, then holds, then turns ragged and falls once the rate is too large; the rate at that turn is the largest usable value, the upper bound, and a value a factor of three or four below it serves as a lower cyclic bound — one curve, the bound read straight off it. On a 56-layer residual net on CIFAR-10 the accuracy stays high over a band of rates an order of magnitude past the conventional ceiling, out near $3$, which is what makes the cross-the-valley phase fast enough to remove most of the plateau. A second-order argument independently puts the optimum that high. Locally $f(\theta) \approx f(\theta_0) + (\theta-\theta_0)^\top \nabla f + \tfrac{1}{2}(\theta-\theta_0)^\top H (\theta-\theta_0)$, and the full Hessian $H$ is $O(N^2)$ and hopeless to form, but only the curvature along the descent direction matters, a directional second derivative estimated by the finite difference $H(\theta) = \lim_{\delta\to 0}[\nabla f(\theta+\delta) - \nabla f(\theta)]/\delta$. The secant optimal per-weight rate is $\varepsilon^* \approx (\theta_{i+1}-\theta_i)/(\nabla f(\theta_{i+1}) - \nabla f(\theta_i))$; substituting the SGD relation $\theta_{i+1} = \theta_i - \varepsilon\,\nabla f(\theta_i)$ to trade gradients for weight differences rewrites it in terms of three sequential weight snapshots,

$$\varepsilon^* = \varepsilon \cdot \frac{\theta_{i+1}-\theta_i}{2\,\theta_{i+1} - \theta_i - \theta_{i+2}},$$

collapsed to a global rate by summing $|\text{numerators}|$ and $|\text{denominators}|$ over weights separately — absolute values rather than squares, to keep positivity without rescaling. The estimate lands around $2$ to $6$ for ResNet-56, and it is large precisely because its denominator, the finite-difference curvature proxy, is *small*: small curvature means a wide, flat minimum, which is exactly the flat-minimum-generalizes-better story arriving from another direction.

That a rate that large does not wreck final accuracy is the load-bearing observation, and it is also from the range test, read through training and test loss separately: over the band of large rates the *training* loss rises while the *test* loss falls, so the generalization gap shrinks as the rate grows. By the working definition that regularization is any modification reducing generalization error without reducing training error, a large learning rate *is* a regularizer. The mechanism is the SGD gradient-noise scale

$$g \approx \frac{\varepsilon\,N}{B\,(1-m)},$$

with $\varepsilon$ the rate, $N$ the dataset size, $B$ the batch, $m$ the momentum: a larger rate injects more noise, and the flat-minima line of work says higher noise pushes SGD toward wider, flatter, better-generalizing regions, while the small rate at the end cools it into the trough. This forces a precondition I would otherwise miss. If the large rate is now carrying much of the regularization, and I leave weight decay and dropout at their textbook strengths, I am over-regularizing and the network underfits or the large rate becomes unusable — which is exactly why naive attempts to just turn the rate up fail. The principle is therefore *balance*: when I add regularization through the rate I must remove some elsewhere — reduce weight decay (and possibly dropout, possibly enlarge the batch) so the total noise budget stays balanced. Weight decay itself I do *not* cycle; it is a fixed penalty doing steady regularization, set once by a small grid search on short runs, lower than usual, with a useful tell — the right value shows a *small* amount of early overfitting, while a value showing none is too large.

The legs of the cycle are cosine rather than linear. Linear ramps are a fine simple default, but a cosine eases into and out of the extremes, spending a little more time near the top and bottom and avoiding an abrupt slope change at the peak — the most dangerous moment, when the rate is largest. The interpolation from `start` to `end` as a fraction `pct` runs $0\to 1$ is

$$\text{anneal}(\text{start},\text{end},\text{pct}) = \text{end} + \frac{\text{start}-\text{end}}{2}\big(\cos(\pi\,\text{pct}) + 1\big),$$

which is `start` at $\text{pct}=0$ (where $\cos 0 = 1$) and `end` at $\text{pct}=1$ (where $\cos\pi = -1$), gliding monotonically between. The up-leg is short and the down-leg long because the steep early region is short and the flat valley plus final descent is most of the run, so $\text{pct\_start} \approx 0.3$ climbs for the first third and descends over the last two-thirds. The climb starts well below the peak via $\text{initial\_lr} = \text{max\_lr}/\text{div\_factor}$ with a division factor around $25$, so the warmup genuinely begins small, and the descent does not stop there but continues to $\text{min\_lr} = \text{initial\_lr}/\text{final\_div\_factor}$ with a final factor around $10^4$ — the annihilation tail, orders of magnitude below the start.

Momentum is the piece I almost left at $0.9$, which would be a mistake, because in the update

$$v_{i+1} = m\,v_i - \varepsilon\,\nabla L(\theta_i), \qquad \theta_{i+1} = \theta_i + v_{i+1},$$

the velocity is a running average of past gradients and the displacement scales with *both* $\varepsilon$ and $m$ — they push on the same step-size lever, and a high constant momentum behaves like a pseudo-increasing rate. In the up-leg I am already driving $\varepsilon$ toward a very large value; holding $m$ high there would stack two amplifiers and the noise scale, with its $(1-m)$ in the denominator, would blow past stability. So during the high-rate phase momentum should be *low* — both to keep the step controlled and to weight the *current* gradient over stale history, which is what I want when the large rate is meant to explore new directions toward the flat region rather than be dragged along the old one. As the rate falls and stability is no longer at risk, momentum should rise again to accelerate the descent into the trough. Momentum therefore cycles *inversely* to the rate, run through the very same cosine `anneal` with endpoints swapped per leg: as $\text{lr}$ goes $\text{initial\_lr}\to\text{max\_lr}$, momentum goes $0.95\to 0.85$; as $\text{lr}$ goes $\text{max\_lr}\to\text{min\_lr}$, momentum goes $0.85\to 0.95$. I cannot set these bounds from a sweep the way I set the rate — a momentum range test is uninformative, since test loss just keeps decreasing as momentum rises toward $1$ with no clean peak — so the band $\text{max\_momentum}=0.95$ down to $\text{base\_momentum}=0.85$ comes from the coupling argument and a small search.

Everything lives in a per-step function from training progress to $(\text{learning rate},\text{momentum})$; the SGD-with-momentum update itself is untouched. I precompute the three derived rates and lay out two phases, each with its start/end rates and start/end momenta; at each step I find the active phase, compute the within-phase fraction `pct`, and cosine-anneal both the rate and the momentum across it. Setting `three_phase=True` instead uses a symmetric up/down cycle plus a separate annihilation third phase, but the two-phase form is the common default. The bounds are supplied by the linear LR range test, and the schedule sets `lr` and `momentum` (or, for an Adam-style optimizer, `beta1`) on each parameter group once per batch.

```python
import math


def _format_param(name, optimizer, value):
    groups = optimizer.param_groups
    if isinstance(value, (list, tuple)):
        if len(value) != len(groups):
            raise ValueError(f"{name} must match optimizer.param_groups")
        return list(value)
    return [value] * len(groups)


def _annealing_cos(start, end, pct):
    """Cosine anneal from `start` (pct=0) to `end` (pct=1)."""
    cos_out = math.cos(math.pi * pct) + 1.0       # 2 at pct=0, 0 at pct=1
    return end + (start - end) / 2.0 * cos_out


def _annealing_linear(start, end, pct):
    """Linear anneal from `start` to `end` as pct goes 0 -> 1."""
    return (end - start) * pct + start


class OneCycleLR:
    """1cycle policy. Sets lr and, inversely, momentum once per batch.
    Two phases by default; three_phase=True uses a symmetric cycle plus annihilation."""

    def __init__(self, optimizer, max_lr, total_steps,
                 pct_start=0.3, anneal_strategy="cos",
                 cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                 div_factor=25.0, final_div_factor=1e4, three_phase=False):
        if total_steps <= 0 or not isinstance(total_steps, int):
            raise ValueError("total_steps must be a positive integer")
        if not isinstance(pct_start, float) or pct_start < 0 or pct_start > 1:
            raise ValueError("pct_start must be a float in [0, 1]")
        if anneal_strategy not in {"cos", "linear"}:
            raise ValueError("anneal_strategy must be 'cos' or 'linear'")

        self.optimizer = optimizer
        self.total_steps = total_steps
        self.cycle_momentum = cycle_momentum
        self.use_beta1 = "betas" in optimizer.defaults if cycle_momentum else False
        self.anneal = _annealing_cos if anneal_strategy == "cos" else _annealing_linear

        max_lrs = _format_param("max_lr", optimizer, max_lr)
        base_moms = _format_param("base_momentum", optimizer, base_momentum)
        max_moms = _format_param("max_momentum", optimizer, max_momentum)
        for group, group_max_lr, group_base_mom, group_max_mom in zip(
            optimizer.param_groups, max_lrs, base_moms, max_moms
        ):
            group["initial_lr"] = group_max_lr / div_factor
            group["max_lr"] = group_max_lr
            group["min_lr"] = group["initial_lr"] / final_div_factor
            if cycle_momentum:
                group["base_momentum"] = group_base_mom
                group["max_momentum"] = group_max_mom
                if self.use_beta1:
                    group["betas"] = (group_max_mom, group["betas"][1])
                else:
                    group["momentum"] = group_max_mom

        if three_phase:
            self.phases = [
                dict(end=float(pct_start * total_steps) - 1,
                     lr0="initial_lr", lr1="max_lr",
                     m0="max_momentum", m1="base_momentum"),
                dict(end=float(2 * pct_start * total_steps) - 2,
                     lr0="max_lr", lr1="initial_lr",
                     m0="base_momentum", m1="max_momentum"),
                dict(end=total_steps - 1,
                     lr0="initial_lr", lr1="min_lr",
                     m0="max_momentum", m1="max_momentum"),
            ]
        else:
            self.phases = [
                dict(end=float(pct_start * total_steps) - 1,
                     lr0="initial_lr", lr1="max_lr",
                     m0="max_momentum", m1="base_momentum"),
                dict(end=total_steps - 1,
                     lr0="max_lr", lr1="min_lr",
                     m0="base_momentum", m1="max_momentum"),
            ]

        self.last_step = -1
        self.step()                                   # apply initial_lr / max_momentum at step 0

    def step(self):
        self.last_step += 1
        step_num = self.last_step
        if step_num > self.total_steps:
            raise ValueError("stepped past total_steps")

        start = 0.0
        for i, ph in enumerate(self.phases):
            end = ph["end"]
            if step_num <= end or i == len(self.phases) - 1:
                pct = (step_num - start) / (end - start)        # 0..1 within the phase
                lrs = []
                for group in self.optimizer.param_groups:
                    lr = self.anneal(group[ph["lr0"]], group[ph["lr1"]], pct)
                    group["lr"] = lr
                    lrs.append(lr)
                    if self.cycle_momentum:
                        momentum = self.anneal(group[ph["m0"]], group[ph["m1"]], pct)
                        if self.use_beta1:
                            group["betas"] = (momentum, group["betas"][1])
                        else:
                            group["momentum"] = momentum
                return lrs
            start = end


def lr_range_test(set_lr, train_one_batch, lr_start, lr_end, num_steps):
    """One short pre-run sweeping lr up linearly; pick max_lr from where the
    accuracy curve peaks / turns ragged (lower cyclic bound = peak / 3 or 4)."""
    history = []
    for s in range(num_steps):
        pct = s / (num_steps - 1)
        lr = _annealing_linear(lr_start, lr_end, pct)
        set_lr(lr)
        loss, acc = train_one_batch()
        history.append((lr, loss, acc))
    return history
```
