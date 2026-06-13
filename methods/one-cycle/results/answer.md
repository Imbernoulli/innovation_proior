# One-Cycle (1cycle) learning-rate policy, distilled

The 1cycle policy trains a deep network with a single learning-rate cycle: ramp the rate
*up* from a small value to a large maximum, then *down* again, and then keep annealing it
several orders of magnitude *below* where it started for the rest of the run — while cycling
the momentum coefficient inversely to the rate. The large rate in the middle acts as a strong
regularizer (driving SGD toward wide, flat minima that generalize well), so other forms of
regularization must be reduced to keep the total balanced. The intended effect is
super-convergence: reaching the target accuracy in far fewer iterations without giving up
generalization. Only the per-iteration learning rate (and momentum)
changes — the optimizer, loss, data pipeline, and SGD update rule are untouched.

## Problem it solves

Step-decay (hold a small global rate, divide by ten at hand-chosen milestones) spends most of
its iterations on flat plateaus making little progress, requires per-architecture milestone
tuning, and never probes large learning rates. The goal: a schedule that reaches a target
accuracy in far fewer iterations, sets its bounds from one short pre-run instead of a grid
search, and generalizes well.

## Key ideas

- **One cycle, shaped by the loss topology.** Training is steep early, then crosses a long
  flat valley (and saddle plateaus), then threads a narrow trough. So the rate should be
  *small → large → small*: small to begin converging without overshoot, large to cross the
  valley fast and punch through saddles, small to settle the trough. One rise, one fall —
  not a small monotone decay, and not many cycles.
- **A final annihilation tail.** After the fall, keep driving the rate orders of magnitude
  below the initial value for the remaining iterations. The large-rate phase carries SGD into
  a wide flat region; the deep anneal then drops into a steeper minimum inside it. This tail
  is what distinguishes 1cycle from a plain triangular cycle, from warmup-then-decay, and from
  SGDR's restart sawtooth.
- **Bounds from the LR range test.** Sweep the rate upward over one short pre-run and plot
  accuracy vs. rate; the largest rate before accuracy turns ragged is `max_lr`. (On deep
  residual nets this peak is an order of magnitude above the usual ~0.1.) No grid search.
- **Large rate = regularization ⇒ rebalance.** Across a band of large rates the *training*
  loss rises while the *test* loss falls — the generalization gap shrinks, the signature of
  regularization. From the SGD noise scale `g ≈ eps·N / (B·(1−m))`, a larger rate means more
  noise, hence wider, flatter, better-generalizing minima. Because the rate now carries much
  of the regularization, other regularizers (weight decay, dropout) must be *reduced* to keep
  the total balanced — the central principle that makes large rates usable.
- **Cyclical momentum, inverse to the rate.** In the SGD-with-momentum update
  `v ← m·v − eps·∇L ; θ ← θ + v`, the displacement scales with both `eps` and `m`, so they
  push on the same lever (and `g ∝ 1/(1−m)`). When the rate is being ramped up, momentum is
  ramped *down* (0.95 → 0.85) to stay stable and weight current gradients (new directions
  toward the flat region) over stale history; when the rate falls, momentum rises back to
  accelerate the descent. A momentum *range test* is uninformative (loss decreases
  monotonically with momentum), so momentum is set by this coupling, not a sweep.
- **Weight decay is constant, not cycled** — a fixed, reduced, grid-searched value. The best
  value shows a *small* amount of early overfitting; no overfitting means it is too large.

## Estimating the optimal rate (simplified Hessian-free)

A by-product justification that large rates are right. Locally
`f(θ) ≈ f(θ₀) + (θ−θ₀)ᵀ∇f + ½(θ−θ₀)ᵀ H (θ−θ₀)`. The full Hessian `H` is `O(N²)` and
unnecessary — only the curvature along the descent direction matters, estimated by a
finite difference `H(θ) = lim_{δ→0}[∇f(θ+δ) − ∇f(θ)]/δ`. The secant optimal per-weight rate
is `eps* ≈ (θ_{i+1} − θ_i)/(∇f(θ_{i+1}) − ∇f(θ_i))`; substituting
`θ_{i+1} = θ_i − eps·∇f(θ_i)` rewrites it from three sequential weight snapshots:

```
eps* = eps · (θ_{i+1} − θ_i) / (2·θ_{i+1} − θ_i − θ_{i+2}).
```

Collapse to a global rate by summing |numerators| and |denominators| over weights separately
(absolute values, to keep positivity, rather than squares). The estimate comes out in the
range ~2–6 for a ResNet-56 on CIFAR-10. A large estimate corresponds to a *small* curvature
denominator — i.e. SGD sitting in a wide, flat minimum, matching the generalization argument.

## Final schedule (two-phase cosine, the fastai / PyTorch default)

With `initial_lr = max_lr / div_factor`, `min_lr = initial_lr / final_div_factor`, cosine
interpolation `anneal(a, b, p) = b + (a−b)/2·(cos(πp)+1)`:

```
phase 1 (first pct_start of steps): lr: initial_lr → max_lr,  momentum: max_mom → base_mom
phase 2 (remaining steps):          lr: max_lr → min_lr,      momentum: base_mom → max_mom
```

Defaults: `pct_start = 0.3`, `anneal = cos`, `div_factor = 25`, `final_div_factor = 1e4`,
`max_momentum = 0.95`, `base_momentum = 0.85`. The step is taken per *batch*. (Setting
`three_phase=True` uses a symmetric up/down cycle plus a separate annihilation third phase;
the two-phase form above is the common default.)

## Working code

Faithful to `torch.optim.lr_scheduler.OneCycleLR`. The schedule sets `lr` and `momentum`
(or Adam-style beta1) on each optimizer parameter group after each batch; the optimizer
update itself is unchanged.

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

## Relation to prior schedules

- **Step decay / piecewise-constant**: small monotone-decreasing rate; 1cycle replaces it
  with one rise-then-fall and a deep anneal to attack the long plateau directly.
- **Triangular cyclical LR** (the parent): many cycles between two fixed bounds; 1cycle uses a
  single cycle and drives the rate far below the lower bound at the end. Shares the LR range
  test for picking bounds.
- **Warmup + decay**: only the up-leg plus ordinary decay, constant momentum; 1cycle adds the
  annihilation tail and inverse momentum cycling.
- **SGDR (cosine warm restarts)**: cosine descents with jumps back to the maximum; 1cycle does
  not restart — its tail keeps descending.
