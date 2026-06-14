# Trapezoidal (constant LR + cooldown) schedule, distilled

The trapezoidal schedule — also called constant-LR-with-cooldown, or warmup-stable-decay (WSD) —
replaces the single cosine cycle used for LLM pretraining with three phases: a linear **warmup** to
the peak learning rate, a long **constant plateau** at the peak, and a short final **cooldown** that
anneals the rate down to a floor (typically zero). The key decoupling is that the plateau can be run
without knowing the eventual stopping point; when a stopped model is needed, a short cooldown can be
branched from the chosen checkpoint.

## Problem it solves

A cosine schedule's decay must be stretched to the total number of steps to be optimal (Hoffmann et
al. 2022), so the intermediate checkpoints of one long run overestimate the loss of length-matched
shorter runs. Scaling-law fits, data-mixture studies, and ablations that need quality at several
lengths therefore require a separate model trained from scratch per length, and a finished cosine
run cannot be cleanly extended (re-warming spikes the loss). The goal: a schedule that (1) keeps the
high-rate exploration phase independent of the final training length, (2) adds a detachable final
annealing phase when a stopped model is needed, and (3) leaves peak rate, final rate, and cooldown
length as tunable knobs.

## Key idea

Hold the learning rate constant for the main body of training (length-agnostic exploration) and only
anneal it in a short final cooldown (the "commit to a basin" phase) that can be launched from any
checkpoint at any step. Formally, with peak `eta_max`, total horizon `N`, warmup `N_warmup`, and
cooldown `N_decay`:

```
eta(n) = (n / N_warmup) * eta_max                          if n < N_warmup        (warmup)
eta(n) = eta_max                                           if N_warmup < n <= N - N_decay   (plateau)
eta(n) = f(n, N, N_decay) * eta_max                        if n > N - N_decay     (cooldown)
```

with `f` monotonically decreasing from 1 to ~0 over the cooldown. Linear cooldown:
`f = 1 - (n - (N - N_decay)) / N_decay`. The (1-sqrt) refinement:
`f = 1 - sqrt((n - (N - N_decay)) / N_decay)`.

## Why each choice

- **Constant plateau + short cooldown instead of cosine.** Cosine's value is "explore high, then
  cool enough to commit"; that shape is incidental. A flat plateau makes the exploration phase
  independent of when training stops, and the cooldown is a detachable tail. The total length only
  determines where the cooldown begins.
- **Why a cooldown is needed (vs. constant only).** For convex Lipschitz objectives the last iterate
  of a *constant*-step SGD carries an extra `ln T` factor in the schedule-dependent bound term: with
  constant `eta` and gradient norm `G`,
  `eta * G^2 * sum_{k=1}^{T-1} 1/(T-k) = eta * G^2 * H_{T-1} ≈ eta * G^2 * ln T`. Letting the late
  `eta_k` shrink suppresses exactly those late summands, so a cooldown removes the logarithmic term
  that a constant schedule leaves in the last-iterate analysis.
- **Why linear cooldown.** Linear decay to zero is the worst-case-optimal last-iterate schedule
  (Defazio et al. 2023): SGD with `eta_t = (D / (G * sqrt(T))) * (1 - t/T)` gives
  `E[f(x_T) - f*] <= D*G / sqrt(T)`, the optimal `O(1/sqrt(T))` rate with no `ln T`. It "emulates
  iterate averaging" — each gradient's contribution to the final point is weighted as in a uniform
  average.
- **Why (1-sqrt) is a useful alternative.** Worst-case optimality is for the adversarial gradient
  sequence; the realized sequence can have structure. Since `sqrt(x) >= x` on `[0, 1]`,
  `1 - sqrt(x) <= 1 - x`: the square-root cooldown drops faster than linear at the beginning and
  then spends a longer tail near the floor. Sweeping `1 - x^a` with `a` around `0.5` tests that
  front-loaded-decay regime; very small `a` values drop the rate too far too early.
- **Cooldown length.** Use a short tunable tail, commonly in the 10-20% range, but keep it as
  `fract_decay`; for longer runs, the relevant quantity can be enough absolute decay steps rather
  than a fixed percentage.
- **Plateau height.** Tune below a peak that is only briefly visited by a decaying schedule, since
  the plateau holds the full rate for much longer.
- **Final rate.** Cooling to zero is the clean theoretical endpoint, but the floor should remain a
  separate knob (`final_lr_factor` / `min_lr`) rather than being tied to the peak.
- **Warmup.** A short linear ramp-in (the mirror of the cooldown) avoids early blow-ups from large
  unreliable gradients on a fresh model.

## Weight-space diagnostic

The intended picture is that the plateau finds a useful region while high-rate noise prevents the
last iterate from settling. A line interpolation between pre- and post-cooldown checkpoints is the
diagnostic: a barrier would indicate that the cooldown escaped to a different region, while a smooth
path indicates that the tail simply descended into the connected basin already reached by the
plateau checkpoint.

## Replacing the cooldown with averaging (optional)

Stochastic weight averaging along the plateau denoises the returned weights and behaves like a
decayed-rate schedule (Sandler et al. 2023; Izmailov et al. 2018). It is a useful readout along the
trajectory, while the explicit cooldown remains the direct way to lower the late step sizes in the
last-iterate bound.

## Implications for scaling laws

Train each model size once at a constant rate for the longest length, saving plateau checkpoints; get
shorter-length stopped models by launching cooldown branches from the corresponding checkpoints. The
length axis changes from K from-scratch runs to one long run plus K short tails.

## Working code

The schedule is a single stateless function the training loop calls each step to set the rate. A
faithful implementation returns a multiplicative factor on `max_lr`; `get_lr` below wraps that form
to fill the scaffold slot:

```python
import math


def wsd_schedule(n_iterations, final_lr_factor=0.0, n_warmup=1000,
                 init_div_factor=100, fract_decay=0.1,
                 decay_type="linear"):
    n_anneal_steps = int(fract_decay * n_iterations)
    n_hold = n_iterations - n_anneal_steps

    def schedule(step):
        if step < n_warmup:
            return step / n_warmup + (1 - step / n_warmup) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < n_iterations:
            x = (step - n_hold) / n_anneal_steps
            if decay_type == "linear":
                return final_lr_factor + (1 - final_lr_factor) * (1 - x)
            elif decay_type == "exp":
                return final_lr_factor ** x
            elif decay_type == "cosine":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 + math.cos(math.pi * x)
                ) * 0.5
            elif decay_type == "miror_cosine":
                cosine_value = final_lr_factor + (1 - final_lr_factor) * (
                    1 + math.cos(math.pi * x)
                ) * 0.5
                linear_value = final_lr_factor + (1 - final_lr_factor) * (1 - x)
                return linear_value * 2 - cosine_value
            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (1 - x ** 2)
            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (1 - math.sqrt(x))
            else:
                raise ValueError(
                    "decay type must be one of "
                    "['cosine', 'miror_cosine', 'linear', 'exp', 'square', 'sqrt']"
                )
        else:
            return final_lr_factor

    return schedule


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr,
           frac_decay=0.1, decay_type="linear", warmup_div=100):
    final_lr_factor = min_lr / learning_rate
    schedule = wsd_schedule(
        n_iterations=lr_decay_iters,
        final_lr_factor=final_lr_factor,
        n_warmup=warmup_iters,
        init_div_factor=warmup_div,
        fract_decay=frac_decay,
        decay_type=decay_type,
    )
    return learning_rate * schedule(it)
```
