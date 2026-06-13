# KL-UCB, distilled

KL-UCB is an optimistic index policy for the stochastic multi-armed bandit. For each arm it
computes an upper confidence bound by inverting the *Kullback-Leibler* large-deviation rate of
the sample mean — the largest mean `q` whose Bernoulli-KL distance from the empirical mean
`mu_hat_a` is within a `(log t)/N_a` deviation budget — and pulls the arm of highest bound. It
replaces UCB1's additive Hoeffding bonus with a KL ball, which makes the confidence width
asymmetric and self-tightening near reward `0` or `1`. The result is provably optimal for
Bernoulli rewards (it matches the Lai-Robbins lower bound) while keeping a distribution-free
guarantee for all `[0, 1]`-bounded rewards, with no horizon- or problem-dependent tuning.

## Problem it solves

Stochastic `K`-armed bandit, rewards i.i.d. and bounded in `[0, 1]` (any bounded reward
rescales). Minimize regret `R_n = sum_{a : mu_a < mu*} Delta_a E[N_a(n)]`, `Delta_a = mu* -
mu_a`, with an online, horizon-free, tuning-free *index* policy whose per-arm pull count comes
as close as possible to the information-theoretic floor.

## Key idea

The per-arm index is the KL upper confidence bound

```
U_a(t) = max{ q in [0, 1] : N_a * d(mu_hat_a, q) <= c log t },     mu_hat_a = S_a / N_a,
```

in the practical pure-log implementation. The theorem uses the exploration level
`log t + 3 log log t` instead. Here `d(p, q) = p log(p/q) + (1 - p) log((1 -
p)/(1 - q))` is the Bernoulli KL divergence (conventions `0 log 0 = 0`, `x
log(x/0) = +inf`). Pull `argmax_a U_a(t)`.

- `d` is the **exact Chernoff large-deviation rate** of a sample mean: `P(mu_hat >= mu + eps)
  <= exp(-n d(mu + eps, mu))`. Inverting this tail gives a confidence bound whose width
  `U_a - mu_hat_a` **shrinks automatically as `mu_hat_a` approaches `0` or `1`**, because `q ->
  d(mu_hat_a, q)` gets steeper there. UCB1's `sqrt(2 log t / N_a)` is the inverse of the
  Pinsker lower bound `2(p-q)^2 <= d(p,q)` — symmetric, range-only, loose near the boundary.
  **UCB1 is the Pinsker-relaxed special case** of KL-UCB.
- It is **distribution-free** even though `d` is Bernoulli: for any `X in [0, 1]` with mean
  `mu`, the convex function `f(x) = exp(lambda x) - x(exp(lambda) - 1) - 1` vanishes at `0` and
  `1`, so `f <= 0` on `[0, 1]`, giving `E[exp(lambda X)] <= 1 - mu + mu exp(lambda)` — the
  Bernoulli MGF. The Bernoulli is the *least concentrated* bounded law of a given mean, so its
  rate `d` upper-bounds everyone's deviations.

## Exploration constant `c`

- In the theorem's notation, the exploration function is `log t + c log log t`, and `c = 3`
  is the proved finite-time choice: this `delta` makes the optimal-arm-underestimation term
  `O(log log n)`. Results hold for any `c >= 3`; an `(1+eps) log t` variant also works, but
  for `eps = 0.1` it only dominates `log t + 3 log log t` past `t > 2*10^51`.
- In practical code the `log log t` correction is dropped. SMPyBandits writes the resulting
  pure-log exploration as `c log t` and defaults to `c = 1`, i.e. budget `log t`.

## Final algorithm

```
Pull each arm once.
For t = K+1, K+2, ...:
    A_t = argmax_a  max{ q in [0,1] : N_a * d(S_a/N_a, q) <= c log t }     # c = 1 in practice
    observe reward r; N_{A_t} += 1; S_{A_t} += r
```

The inner `max` (KL inversion) has no closed form, but `q -> d(p, q)` is strictly convex,
zero at `q = p`, and strictly increasing on `[p, 1]`, so the bound is the unique root of
`d(p, q) = (c log t)/N_a` (or the right endpoint). Solve by bisection or Newton, seeding the upper endpoint
with the Pinsker bound `min(1, p + sqrt((c log t)/(2 N_a)))`.

## Regret guarantee

With theorem exploration `log t + 3 log log t`, for every sub-optimal arm and any `eps > 0`,

```
E[N_n(a)] <= (1 + eps) log n / d(mu_a, mu*) + C_1 log log n + C_2(eps)/n^{beta(eps)},
```

hence `limsup_n E[R_n]/log n <= sum_{a : mu_a < mu*} (mu* - mu_a) / d(mu_a, mu*)`. For
Bernoulli rewards this matches the **Lai-Robbins lower bound** (asymptotic optimality); by the
bounded-to-Bernoulli MGF lemma the same bound holds for *all* `[0, 1]` rewards. Since
`d(mu_a, mu*) >= 2 Delta_a^2` (Pinsker), KL-UCB improves UCB's leading constant, with the
largest gain when means are near `0` or `1`. The two pillars of the proof are (i) a self-normalized
supermartingale `W_t = exp(lambda S(t) - N(t) psi_mu(lambda))` plus a geometric *peeling* of
the random pull count `N(n)`, giving `P(u(n) < mu) <= e ceil(delta log n) exp(-delta)`; and
(ii) a sample-count decomposition split at `K_n = floor((1+eps)(log n + 3 log log n) /
d(mu_a, mu*))` with a Chernoff geometric tail.

## Exponential-family variant

The proof only uses an MGF bound, so swap `d` for any one-parameter exponential family's
Legendre rate `d(x, mu(theta)) = sup_lambda { lambda x - log E_theta[exp(lambda X)] }`, which
equals that family's KL divergence `KL(p_beta, p_theta) = mu(beta)(beta - theta) - b(beta) +
b(theta)` (for `x = mu(beta)`). Then KL-UCB is Lai-Robbins-optimal for that family. Recipes:
exponential `d(x, y) = x/y - 1 - log(x/y)`; Poisson `d(x, y) = y - x + x log(x/y)`; Gaussian
fixed variance recovers UCB's additive `sqrt` bonus. An upper bound on the true `d` also gives
a valid, slightly looser policy.

## Working code

Faithful to the canonical SMPyBandits implementation: a Bernoulli KL `d`, a bisection KL
inverter seeded by the Pinsker (Gaussian) upper bracket, and an index policy with `c = 1`.

```python
import math
import numpy as np

_EPS = 1e-15  # truncate means to [_EPS, 1 - _EPS]; d = +inf at the exact 0/1 endpoints


def kl_bernoulli(p, q):
    """Bernoulli KL divergence d(p, q) = p log(p/q) + (1-p) log((1-p)/(1-q))."""
    p = min(max(p, _EPS), 1 - _EPS)
    q = min(max(q, _EPS), 1 - _EPS)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def klucb(mu_hat, budget, kl=kl_bernoulli, upper=1.0, precision=1e-6, max_iter=50):
    """Generic KL-UCB inversion: max{ q in [mu_hat, upper] : kl(mu_hat, q) <= budget }.
    Bisection on the increasing branch q -> kl(mu_hat, q)."""
    lower = mu_hat
    for _ in range(max_iter):
        if upper - lower <= precision:
            break
        mid = 0.5 * (lower + upper)
        if kl(mu_hat, mid) > budget:
            upper = mid
        else:
            lower = mid
    return 0.5 * (lower + upper)


def klucb_bern(mu_hat, budget, precision=1e-6):
    """KL-UCB index for Bernoulli (and any [0,1]-bounded) rewards.
    Pinsker/Gaussian seed for the upper bracket: d(p,q) >= 2(p-q)^2 => U <= p + sqrt(budget/2)."""
    upper = min(1.0, mu_hat + math.sqrt(budget / 2.0))
    return klucb(mu_hat, budget, kl=kl_bernoulli, upper=upper, precision=precision)


class KLUCB:
    """KL-UCB index policy. Index for arm a at time t:
        U_a(t) = max{ q : N_a * d(mu_hat_a, q) <= c log t },   mu_hat_a = S_a / N_a.
    Pull argmax_a U_a(t); pull each unpulled arm first (index +inf)."""

    def __init__(self, K, c=1.0, tolerance=1e-4):
        self.K = K
        self.c = c                       # pure-log exploration multiplier; default c=1
        self.tolerance = tolerance
        self.counts = np.zeros(K)        # N_a
        self.rewards = np.zeros(K)       # S_a

    def reset(self):
        self.counts[:] = 0
        self.rewards[:] = 0

    def select_arm(self, t, context=None):
        if t < self.K:
            return t                      # round-robin init
        budget_t = self.c * math.log(max(t, 1))     # delta = c log t
        best_arm, best_index = 0, -math.inf
        for a in range(self.K):
            if self.counts[a] < 1:
                return a                  # never pulled -> index +inf
            mu_hat = self.rewards[a] / self.counts[a]
            index = klucb_bern(mu_hat, budget_t / self.counts[a], self.tolerance)  # U_a(t)
            if index > best_index:
                best_index, best_arm = index, a
        return best_arm

    def update(self, arm, reward, context=None):
        self.counts[arm] += 1
        self.rewards[arm] += reward


# Exponential-family rates: drop the matching d into `klucb(..., kl=...)`.
def kl_exponential(x, y):
    """d(x, y) = x/y - 1 - log(x/y) for exponential rewards."""
    if x <= 0 or y <= 0:
        return math.inf
    return x / y - 1.0 - math.log(x / y)


def kl_poisson(x, y):
    """d(x, y) = y - x + x log(x/y) for Poisson rewards."""
    y = max(y, _EPS)
    if x <= 0:
        return y
    return y - x + x * math.log(x / y)
```
