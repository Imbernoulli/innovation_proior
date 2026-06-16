# Hyperband, distilled

Hyperband is a hyperparameter-optimization method that speeds up random search by adaptive
resource allocation and early stopping. It treats each configuration as an arm of a
non-stochastic infinite-armed bandit, uses Successive Halving as an early-stopping inner loop,
and wraps it in an outer loop that hedges over the "`n` versus `B/n`" tradeoff — the unknown
choice between many cheaply-evaluated configurations and few fully-trained ones — by running
Successive Halving for a geometric grid of bracket sizes. It needs only two inputs, the maximum
per-configuration resource `R` and a halving rate `eta` (default 3), makes no parametric
assumption about convergence, and loses only a logarithmic factor relative to Successive Halving
run with the optimal (unknowable) number of configurations.

## Problem it solves

Find the configuration `x ∈ X` minimizing validation error using as little total resource
(iterations, data-subsample size, or features) as possible, where each train-and-validate is the
expensive unit, the objective is a black box of unknown smoothness, and the convergence rate of
each configuration's loss curve and the distribution of good configurations are both unknown.

## Key idea

Cast configurations as arms whose loss `ell_{i,k}` after `k` units of resource converges to a
terminal value `nu_i`. Let `gamma(j)` be the smallest non-increasing envelope with
`sup_i |ell_{i,j} - nu_i| <= gamma(j)`; two configurations are separable once
`gamma(j) <= (nu_i - nu_1)/2`, so configuration `i` needs resource
`tau_i = gamma^{-1}((nu_i - nu_1)/2)` to be told apart from the best.

- **Successive Halving (SH)** allocates resource adaptively: given budget `B` and `n` configs,
  proceed in rounds; each round train all survivors equally, then keep the best fraction
  `1/eta`. Resource escalates geometrically onto survivors, so easy-to-reject configs are
  cut after tiny resource. SH pays roughly a *sum* of per-config separation costs, versus
  uniform allocation's `n ×` the *hardest* config's cost.
- **The `n` versus `B/n` problem.** SH needs `n` as input; the optimal `n` depends on the
  unknown envelope `gamma` and the unknown distribution `F` of terminal values. Estimating them
  would reimport the fragile convergence assumptions the bandit framing was meant to avoid.
- **Hyperband's move.** Don't choose `n` — hedge. Run SH for a geometric grid of bracket sizes
  `s = s_max, ..., 0` (a two-dimensional doubling trick over `n` and `B`). The most aggressive
  bracket `s = s_max` starts with many configs at minimum resource (maximal early stopping); the
  least aggressive bracket `s = 0` is plain random search at full resource. There are only
  `s_max + 1 = floor(log_eta R) + 1 = O(log R)` brackets, so hedging costs only a log factor.

## Algorithm (finite horizon)

Inputs: `R` (max resource per configuration), `eta >= 2` (default 3).
Initialize `s_max = floor(log_eta R)`, `B = (s_max + 1) R`.

```
for s in {s_max, s_max - 1, ..., 0}:
    n = ceil( (B/R) * eta^s / (s+1) )      # starting pool: many configs for large s
    r = R * eta^(-s)                       # minimum resource so r*eta^s = R
    T = sample n configurations i.i.d. (uniform by default)
    for i in {0, 1, ..., s}:               # Successive Halving inner loop
        n_i = floor(n * eta^(-i))          # configs alive this round
        r_i = r * eta^i                    # resource each receives
        L   = { run_then_return_val_loss(t, r_i) : t in T }
        T   = top_k(T, L, floor(n_i / eta))   # promote the best 1/eta
return the configuration with the smallest validation loss observed
```

Required callbacks: `sample` (i.i.d. configuration draw), `run_then_return_val_loss(config,
resource)` (the expensive train+validate), `top_k(configs, losses, k)` (rank by loss). For
`R = 81, eta = 3`: `s_max = 4`, five brackets; bracket `s = 4` runs `(81, 27, 9, 3, 1)` configs
at `(1, 3, 9, 27, 81)` resource; bracket `s = 0` runs 5 configs at 81 (random search). One sweep
costs `(s_max + 1) B`; repeat as budget allows.

## Why these choices

- `eta = 3` (or 4): each round keeps `1/eta`, doing `floor(log_eta n) + 1` rounds; the
  bracket-overhead constant is minimized near `eta = e ≈ 2.718`, rounded for convenience, and
  results are insensitive to it.
- `n = ceil((B/R) eta^s/(s+1))`: the `eta^s` makes aggressive brackets large; the `1/(s+1)`
  divides by the number of rounds so every bracket spends ~`B`.
- `r = R eta^{-s}`: minimum resource so that after `s` promotions (each `× eta`) the survivor
  reaches exactly `R`.
- `s_max + 1` brackets: logarithmic in `R`, hence cheap to hedge across all tradeoffs.
- Uniform i.i.d. sampling: the analysis needs only i.i.d. draws from a stationary distribution,
  so uniform is the assumption-free default; any smarter sampler is an optional, orthogonal
  improvement.

## Guarantees

Successive Halving (infinite horizon): if `B > z_SH` with
```
z_SH = 2 ceil(log2 n) max_{i>=2} i (1 + gamma^{-1}(max{eps/4, (nu_i - nu_1)/2}))
     <= 2 ceil(log2 n) (n + sum_i gamma^{-1}(max{eps/4, (nu_i - nu_1)/2}))
```
SH returns an arm with `nu_hat - nu_1 <= eps/2`. Under `gamma(j) ≈ (1/j)^{1/alpha}` and
`F(x) ≈ (x - nu_*)^beta`, SH's budget scales like `Delta^{-max{alpha,beta}}` versus uniform
allocation's `Delta^{-(alpha+beta)}`.

Hyperband: after total budget `T`,
`nu_hat_T - nu_* <= c (logbar(T)^3 logbar(log(T)/delta) / T)^{1/max{alpha,beta}}` with
`c = exp(O(max{alpha,beta}))` and `logbar(x) = log(x) log log(x)` — i.e. only log factors worse
than SH run with the optimal `n`, and matching known best-arm lower bounds (infinite-armed and
finite-`K`-armed) up to log factors in the stochastic special case (`alpha = 2`).

## Relation to prior methods

- **Random search** = the least aggressive Hyperband bracket (`s = 0`): a fixed pool trained to
  full resource with no early stopping. It is the conservative end the hedge falls back to.
- **Successive Halving** = a single Hyperband bracket. Hyperband is exactly SH plus the outer
  hedge over the bracket size, removing SH's need to pre-select `n`.
- **Bayesian optimization (TPE/SMAC/Spearmint)** optimizes *which* configuration to evaluate and
  trains each to completion; Hyperband instead optimizes *how much* resource each configuration
  earns and is orthogonal — a model-based sampler can replace uniform sampling inside it.

## Working code

Finite-horizon Hyperband, calling Successive Halving as a subroutine and matching the standard
implementation structure:

```python
import numpy as np
from math import log, ceil


def successive_halving(space, n, r, s, eta, rng,
                       sample_configuration,
                       run_then_return_val_loss, top_k):
    """One bracket: n configs start at minimum resource r; over s+1 rounds keep the
    top 1/eta while multiplying resource by eta, so the survivor reaches r*eta^s."""
    T = [sample_configuration(space, rng) for _ in range(n)]  # n i.i.d. configurations
    seen = []
    for i in range(s + 1):                                 # rounds 0..s
        n_i = int(n * eta ** (-i))                         # configs alive: floor(n eta^-i)
        r_i = r * eta ** i                                 # resource each gets: r eta^i
        losses = [run_then_return_val_loss(t, r_i) for t in T]   # the expensive op
        seen.extend(zip(T, losses, [r_i] * len(T)))
        T = top_k(T, losses, int(n_i / eta))               # promote best floor(n_i/eta)
    return seen


def hyperband(space, R, eta, rng,
              sample_configuration,
              run_then_return_val_loss, top_k):
    """Hedge SH over a geometric grid of bracket sizes, from maximal early stopping
    (s=s_max) to plain random search (s=0). Each bracket uses ~B; n is never chosen."""
    s_max = int(log(R) / log(eta))                         # #brackets - 1
    B = (s_max + 1) * R                                    # budget per bracket
    seen = []
    for s in reversed(range(s_max + 1)):                   # s = s_max ... 0
        n = int(ceil((B / R) * eta ** s / (s + 1)))        # ceil((B/R) eta^s/(s+1))
        r = R * eta ** (-s)                                # min resource: r*eta^s = R
        seen.extend(successive_halving(space, n, r, s, eta, rng,
                                       sample_configuration,
                                       run_then_return_val_loss, top_k))
    return min(seen, key=lambda clr: clr[1])[0]            # best config by loss
```
