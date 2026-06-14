# Uncertainty Weighting, distilled

Uncertainty weighting (homoscedastic / task-dependent uncertainty weighting) combines several
task losses in a shared-representation multi-task network by treating each task's loss as a
negative log-likelihood with its own learned observation-noise scale `σ_i`. The derivation weights
each task by inverse variance `1/σ_i²` and adds a logarithmic scale penalty; the canonical
log-variance implementation uses `s_i = log σ_i²` and optimises `Σ_i exp(-s_i) L_i + s_i`, so the
relative task weights are learned jointly with the network instead of hand-tuned and cannot collapse
like bare learnable weights.

## Problem it solves

In multi-task deep learning the total loss `L_total = Σ_i w_i L_i` is acutely sensitive to the
weights `w_i`, which depend on each task's measurement units and label noise and differ by orders
of magnitude. Finding good weights by grid search costs one full training run per grid point and
explodes combinatorially with the number of tasks. Making the `w_i` free trainable parameters fails:
`∂L_total/∂w_i = L_i ≥ 0`, so SGD drives every `w_i → 0` and the model learns nothing. The goal is a
principled, automatically learned, data-adaptive relative weighting that cannot collapse.

## Key idea

Read each loss as a negative log-likelihood with a per-task observation-noise scale `σ_i`, and learn
`σ_i` jointly with the weights `W` by maximising the joint likelihood.

- **Regression** uses a Gaussian likelihood `p(y | f^W(x)) = N(f^W(x), σ²)`, whose NLL is
  `(1/2σ²)||y − f^W(x)||² + (1/2)log σ²`. So a regression task contributes
  `(1/2σ_i²) L_i + log σ_i` with `L_i = ||y_i − f^W(x)||²`.
- **Classification** scales the logits by `1/σ²` before the softmax — a Boltzmann/Gibbs distribution
  with temperature `σ²` — so `p(y | f^W(x), σ) = Softmax((1/σ²) f^W(x))`. Its NLL is
  `(1/σ²) L_i + log[ Σ_{c'} exp((1/σ²) f_{c'}) / (Σ_{c'} exp f_{c'})^{1/σ²} ]`, with `L_i` the
  unscaled cross-entropy `-log Softmax(f^W(x))_c`. Under the approximation
  `(1/σ) Σ_{c'} exp((1/σ²) f_{c'}) ≈ (Σ_{c'} exp f_{c'})^{1/σ²}` (exact as `σ → 1`) the bracket
  collapses to `log σ`, giving `(1/σ_i²) L_i + log σ_i`.
- **Inverse-variance weighting:** the coefficient `1/σ_i²` down-weights noisy / large-scale tasks
  and up-weights well-determined ones — the statistically correct way to fuse measurements of
  differing precision.
- **The logarithmic scale regulariser** is the anti-collapse term: sending `σ_i → ∞` to zero out a
  task's weight drives the log term to `+∞`, so the likelihood forbids declaring a task infinitely
  noisy. This is exactly what a bare learnable `w_i` lacks. For the Gaussian term,
  `∂[(1/2σ_i²)L_i + log σ_i]/∂σ_i = -L_i/σ_i³ + 1/σ_i`, so the stationary point is
  `σ_i² = L_i`. The implemented `exp(-s_i)L_i + s_i` has the same fixed point via
  `∂/∂s_i = -exp(-s_i)L_i + 1`. Each task's learned variance tracks its current loss, so the rule is
  dynamic during training rather than one fixed grid weight for the whole run.

## Stability and initialisation

Train the **log-variance** `s_i := log σ_i²` rather than `σ_i`. Then `1/σ_i² = exp(-s_i)` (always
positive, no divide-by-zero) and `s_i` is unconstrained (plain SGD can step it). The Gaussian NLL
term `(1/2)exp(-s)L + (1/2)s` is strictly convex in `s` with minimum at `s = log L`; the canonical
implemented term `exp(-s)L + s` is the same scalar objective multiplied by two and has the same
minimum. Initialise `s_i = 0` (i.e. `σ_i² = 1`, equal weighting) — a neutral start that needs no
tuning.

## Final objective

For a mix of one continuous (Gaussian) output `y_1` and one discrete (softmax) output `y_2`:

```
L(W, σ_1, σ_2) = (1/2σ_1²) L_1(W) + (1/σ_2²) L_2(W) + log σ_1 + log σ_2
```

with `L_1 = ||y_1 − f^W(x)||²` (Euclidean) and `L_2 = −log Softmax(y_2, f^W(x))` (cross-entropy).
The regression coefficient carries the `1/2` from the Gaussian NLL; the classification coefficient
carries the full `1/σ²` from the logit temperature scaling. That asymmetry is part of the derivation.
The compact canonical implementation uses the uniform log-variance form for already-reduced task
losses:

```
L_impl = Σ_i exp(-s_i) L_i + s_i,    s_i = log σ_i²
```

## Working code

Filling the combination-rule slot of the multi-task harness, with one log-variance Parameter per
task registered so the optimiser trains it jointly with the network:

```python
import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Homoscedastic uncertainty weighting of K task losses.

    Learns one log-variance s_i = log(sigma_i^2) per task. Each task loss is
    weighted by its precision exp(-s_i) and regularized by + s_i:
        L = sum_i  exp(-s_i) * L_i  +  s_i
    exp(-s_i) > 0 (no divide-by-zero); s_i is unconstrained so SGD steps it
    freely; + s_i forbids the sigma -> inf (weight -> 0) collapse that a bare
    learnable weight w_i in sum_i w_i L_i suffers."""

    def __init__(self, num_tasks=2):
        super().__init__()
        # s_i = log(sigma_i^2), init 0  =>  sigma_i^2 = 1  =>  equal weighting.
        # Objective is convex in s with optimum at s = log(L_i), so init is robust.
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        losses = [fine_loss, coarse_loss]
        total = sum(
            torch.exp(-self.log_vars[i]) * losses[i] + self.log_vars[i]
            for i in range(len(losses))
        )
        return total
```

## Relation to prior methods

- **Naive weighted sum** `Σ_i w_i L_i` with fixed / hand-tuned `w_i`: this method replaces the manual
  grid search with a learned, dynamic, data-adaptive weighting.
- **Learnable bare weights** `Σ_i w_i L_i` with trainable `w_i`: collapse to `w_i = 0`; this method
  adds the logarithmic scale regulariser that the likelihood supplies, which prevents the collapse.
- **Heteroscedastic / mean-variance regression** (Nix & Weigend 1994): the same Gaussian-NLL move
  with a learned noise, but there `σ(x)` is input-dependent and attenuates individual data points
  within one task; here `σ_i` is a single constant per task and sets the relative weight *between*
  tasks (homoscedastic / task-dependent uncertainty).
