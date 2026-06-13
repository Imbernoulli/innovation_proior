# Cosine annealing (with warm restarts), distilled

Cosine annealing is a learning-rate schedule for SGD-with-momentum training of deep nets. Within a run it
decays the learning rate from a high value `eta_max` down to a low value `eta_min` along a half-cosine
curve, and (optionally) periodically *warm-restarts* by snapping the rate back up to `eta_max`. It
replaces the hand-tuned multi-step staircase with a smooth, low-parameter curve, gives usable checkpoints
at cycle troughs, and — when restarts are used — can produce a sequence of distinct candidate models in a
single run.

## Problem it solves

The schedule of the scalar learning rate dominates deep-net training quality, with the optimizer
(SGD+momentum), model, augmentation, and weight decay held fixed. The prevailing schedule is a
piecewise-constant staircase (e.g. multiply `eta` by 0.2 at epochs 60/120/160 of 200, or divide by 10 at
plateaus), which is discontinuous, requires hand-picking drop milestones and a drop factor tied to a
pre-fixed budget, lags the region the iterate is actually in between drops, and yields a usable solution
essentially only at the final epoch. The goal is a schedule that is smooth, needs few hyperparameters, has
good anytime behavior, and can deliver multiple good solutions per run.

## Key idea

Two moves, derived from the restart literature:

- **Emulate a restart by raising the learning rate (warm restart).** A restart in momentum optimization is
  a reset of the acceleration phase — stop refining, explore broadly again. Rather than literally zeroing
  the momentum (which discards hard-won velocity) or triggering on a noisy stochastic loss/gradient test
  (unreliable per minibatch), simply *schedule* the learning rate back up to `eta_max`. This keeps state,
  needs no momentum surgery, and requires no detection; the size of the LR jump softly controls how much
  prior velocity survives.
- **Decay along a half-cosine within each run.** A cycle should start flat and high (sustained exploratory
  steps) and end flat and low (fine convergence, and a clean restart point). Writing the within-cycle rate
  as `eta = eta_min + (eta_max - eta_min)·g(s)` with `s = T_cur/T_i in [0,1]`, the requirements are
  `g(0)=1`, `g(1)=0`, `g'(0)=g'(1)=0`, monotone decreasing. The half-cosine `g(s) = (1/2)(1 + cos(pi·s))`
  satisfies all four exactly (linear has constant slope and corners; exponential is steep at the top;
  polynomial `(1-s)^p` is flat only at the bottom). Among the S-curves meeting the four conditions, the
  cosine is the parameter-free half-period with the right flat endpoints.

## Final schedule

Within the `i`-th run:

```
eta_t = eta_min^i + (1/2) (eta_max^i - eta_min^i) (1 + cos(pi · T_cur / T_i))
```

- `T_cur` = epochs (updated per batch, so it can be fractional) since the last restart.
- `T_i` = length of the current run.
- `T_cur = 0`  -> `cos(0) = 1`  -> `eta_t = eta_max`  (the restart kick / exploratory start).
- `T_cur = T_i` -> `cos(pi) = -1` -> `eta_t = eta_min` (the fine-convergence trough).
- Restart schedule: `T_i = T_0 · T_mult^{i}`. `T_mult = 1` is a fixed period; `T_mult > 1` (e.g.
  `T_0=10, T_mult=2`) grows each run, giving good anytime performance (a first annealed solution early,
  progressively longer refinement after) — the "increase the budget per restart" pattern from
  restart-based gradient-free search.
- `eta_max^i`, `eta_min^i` kept fixed across `i` to minimize hyperparameters; `eta_min` is usually 0, so
  each cycle anneals all the way down (sharp final fit, maximal-contrast restart kick).
- Recommendation (incumbent): since a warm restart transiently worsens performance, report the iterate at
  the *end* of the last completed run (`eta = eta_min`, the cosine trough), not the literal last iterate —
  which needs no validation split.

## Single-run special case (smooth cosine decay)

With `T_mult = 1`, one cycle over the whole budget (`T_i = total_epochs`), `eta_min = 0`,
`eta_max = base_lr`, the schedule reduces to a single half-cosine from `base_lr` to `0`:

```
eta_t = base_lr · (1/2) (1 + cos(pi · epoch / total_epochs))
```

Two hyperparameters (`base_lr`, `total_epochs`), no milestones, no drop factor — a smooth drop-in
replacement for the staircase. This is the plain "cosine annealing" schedule (it is exactly the
closed form of PyTorch's `CosineAnnealingLR` with `eta_min=0`).

## Working code

Single-run cosine decay, filling the schedule slot of the SGD+momentum harness:

```python
import math


def get_lr(epoch, total_epochs, base_lr, config):
    """Cosine annealing of the learning rate over the whole run.

    A single half-cosine from base_lr (epoch 0) to 0 (final epoch): starts flat and
    high (broad, exploratory steps), accelerates through the middle, and eases gently
    to ~0 at the end (fine convergence).

        eta = base_lr * 0.5 * (1 + cos(pi * epoch / total_epochs))
    """
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
```

General cosine annealing with warm restarts, written in the same state variables as the common PyTorch
scheduler:

```python
import math


class CosineAnnealingWarmRestarts:
    """Within run i, anneal each group's base lr to eta_min along a half-cosine;
    after the run length T_i is reached, wrap T_cur and grow T_i by T_mult.

        eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * T_cur / T_i))
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("T_0 must be a positive integer")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("T_mult must be an integer >= 1")
        if not isinstance(eta_min, (float, int)):
            raise ValueError("eta_min must be a number")

        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.eta_min = eta_min
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.T_cur = last_epoch
        self.last_epoch = last_epoch
        self.step(0 if last_epoch < 0 else last_epoch)

    def get_lr(self):
        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min)
            * (1.0 + math.cos(math.pi * self.T_cur / self.T_i))
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        # epoch may be fractional, e.g. epoch + batch_index / batches_per_epoch.
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur -= self.T_i
                self.T_i *= self.T_mult
        else:
            if epoch < 0:
                raise ValueError("epoch must be non-negative")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log(
                        epoch / self.T_0 * (self.T_mult - 1) + 1,
                        self.T_mult,
                    ))
                    self.T_cur = (
                        epoch
                        - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                    )
                    self.T_i = self.T_0 * self.T_mult**n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        return lrs
```

## Why cosine over the alternatives

| shape | `g(0)=1` | `g(1)=0` | `g'(0)=0` (stay high) | `g'(1)=0` (gentle floor / clean restart) |
|-------|:--------:|:--------:|:---------------------:|:----------------------------------------:|
| linear `1-s` | yes | yes | no (slope -1) | no (slope -1; corner at restart) |
| exponential `c^s` | yes | no (-> `c`, never 0) | no (steepest at top) | no (slope `c·ln c`, only relatively flatter) |
| polynomial `(1-s)^p`, `p>1` | yes | yes | no (slope `-p`) | yes |
| half-cosine `(1/2)(1+cos(pi s))` | yes | yes | yes | yes |

The half-cosine is the parameter-free member of the family of monotone S-curves with zero slope at both
ends (the cubic smoothstep `1-(3s^2-2s^3)` also satisfies all four conditions but adds an arbitrary degree
choice); its slope is steepest at the midpoint, giving a sustained high-rate
exploratory phase, a brisk transition, and a gentle convergent landing.
