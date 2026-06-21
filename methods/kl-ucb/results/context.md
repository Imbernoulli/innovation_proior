# Context: index policies for the stochastic multi-armed bandit (circa 2009-2011)

## Research question

An agent faces a slot machine with `K` arms. At each round `t = 1, 2, ..., n` she pulls one
arm `A_t in {1, ..., K}` and receives a reward `X_t`; conditional on the arm choices, the
rewards from arm `a` are i.i.d. with unknown mean `mu_a`, and are *bounded* — say in `[0, 1]`
(any bounded reward can be rescaled to `[0, 1]`). She knows nothing about the reward
distributions beyond this boundedness. The best arm is any `a*` with `mu_{a*} = max_a mu_a`.
Her performance is the regret

```
R_n = n * mu_{a*} - E[ sum_{t=1}^n X_t ]  =  sum_{a : mu_a < mu_{a*}} (mu_{a*} - mu_a) * E[N_a(n)],
```

where `N_a(n)` is the number of pulls of arm `a` up to time `n` and `Delta_a = mu_{a*} - mu_a`
is its sub-optimality gap. So regret is a gap-weighted count of sub-optimal pulls, and the
whole game is to keep `E[N_a(n)]` small for every sub-optimal arm. This is the
exploration-versus-exploitation dilemma: pull the apparent best arm to cash in, but pull the
others enough to be sure they really are worse.

The question is how to design a simple *index* policy — one that computes a single number from
each arm's own data and pulls the arm of highest index — that is online and horizon-free, with
a per-arm pull count as close as possible to the information-theoretic floor established by
Lai and Robbins (1985).

## Background

The multi-armed bandit is the archetypal model of sequential decision-making under
uncertainty, studied since Gittins (1979) showed that, in a Bayesian discounted setting,
optimal policies take the special form of a per-arm *dynamic allocation index* — compute a
number for each arm from its own draws, pull the largest. This is the conceptual seed of all
*index policies*. The "optimistic" reading of an index is that instead of the plain estimate
`mu_hat_a` the agent uses an *upper confidence bound* (UCB) on `mu_a` and acts as if every
arm were as good as its data still allow — optimism automatically forces exploration of
under-sampled arms (their bounds are wide) and exploitation of well-sampled good ones (their
bounds are tight and high).

**The information-theoretic floor.** Lai and Robbins (1985), extended to multi-parameter
models by Burnetas and Katehakis (1997), proved a lower bound that any *uniformly good*
policy (one whose regret is sub-polynomial, `o(n^alpha)` for every `alpha > 0`, on every
instance) must obey. The bound is a change-of-measure argument: to stop pulling a
sub-optimal arm `a`, the agent must accumulate enough evidence to rule out the "most
confusing" alternative world in which `a`'s distribution is nudged up until it becomes the
best — and the number of samples needed to distinguish the true world from that alternative
is governed by a Kullback-Leibler divergence. Concretely,

```
liminf_{n -> inf}  E[N_a(n)] / log n  >=  1 / Kinf(nu_a, mu_{a*}),
   where  Kinf(nu_a, x) = inf{ KL(nu_a, nu') : E[nu'] > x },
```

and hence `liminf_n E[R_n]/log n >= sum_{a : mu_a < mu_{a*}} Delta_a / Kinf(nu_a, mu_{a*})`.
For *Bernoulli* arms the most-confusing-instance divergence `Kinf` collapses to the Bernoulli
KL divergence

```
d(p, q) = p log(p/q) + (1 - p) log((1 - p)/(1 - q)),
```

(with the conventions `0 log 0 = 0`, `0 log(0/0) = 0`, `x log(x/0) = +inf` for `x > 0`), so
the per-arm floor is `1 / d(mu_a, mu_{a*})`. This `d` is the load-bearing object of the whole
area. Two of its properties matter throughout. First, for fixed `p` the map `q -> d(p, q)` is
strictly convex, equals `0` at `q = p`, and is strictly increasing on `[p, 1]` (and
decreasing on `[0, p]`) — a U-shape with its floor at the empirical mean. Second, *Pinsker's
inequality* `d(p, q) >= 2(p - q)^2` lower-bounds it by a parabola, with the gap growing as
`p` or `q` approaches `0` or `1`; equivalently `d` is *steeper than quadratic* near the
boundary, and a Taylor expansion gives `d(p, q) = (p - q)^2 / (2 p (1 - p)) + o((p-q)^2)`, so
near the boundary the local curvature `1 / (2 p (1 - p))` is much larger than the uniform `2`
of the Pinsker parabola.

**The concentration that makes `d` the natural currency.** For i.i.d. Bernoulli `X_1, ...,
X_n` with mean `mu` and sample mean `mu_hat`, Chernoff's method gives the exponential tail
bounds `P(mu_hat >= mu + eps) <= exp(-n d(mu + eps, mu))` and `P(mu_hat <= mu - eps) <= exp(-n
d(mu - eps, mu))` — the *exact* large-deviation rate, not the `exp(-2 n eps^2)` that Pinsker
(or Hoeffding) would give. So `n d(mu_hat, mu)` is the natural "number of nats of surprise" in
an observed deviation, and inverting the tail produces a confidence region whose width
*shrinks automatically as the mean approaches `0` or `1`*, where a Bernoulli is intrinsically
less variable. The prevailing finite-time analyses of the time, however, were built on
Hoeffding's inequality, i.e. on the quadratic Pinsker proxy, which throws this asymmetry
away.

## Baselines

These are the prior index (and elimination) policies a new method would be measured against.

**UCB1 (Auer, Cesa-Bianchi & Fischer, 2002).** The canonical optimistic index. After pulling
each arm once, at step `t` pull

```
A_t = argmax_a  mu_hat_a + sqrt( 2 log t / N_a ),    mu_hat_a = S_a / N_a.
```

The additive bonus `sqrt(2 log t / N_a)` is a Hoeffding/sub-Gaussian confidence radius:
treating a `[0, 1]` reward as having variance proxy `1/4`, the half-width that holds with
probability `~ 1 - t^{-4}` is `sqrt(2 log t / N_a)`. Auer et al. prove the finite-time bound

```
E[R_n] <= sum_{a : mu_a < mu_{a*}} 8 log n / Delta_a  +  (1 + pi^2/3) sum_a Delta_a,
```

so `E[N_a(n)] <= 8 log n / Delta_a^2 + C`. This is `O(log n)`, horizon-free, and tuning-free.
The bonus is *symmetric* and depends only on `N_a` and the `[0, 1]` range.

**UCB2 (Auer et al., 2002).** A re-tuned variant with leading constant `(1 + epsilon)/2`
instead of `8`, attaining the right `1/2` factor against the *quadratic* divergence. It
carries a parameter `alpha` whose right value depends on the problem and the horizon.

**UCB-Tuned (Auer et al., 2002).** Replaces the variance proxy `1/4` by an *empirical*
per-arm variance estimate inside the bonus, so well-behaved arms get tighter intervals.
Empirically strong.

**UCB-V (Audibert, Munos & Szepesvári, 2009).** Puts an *empirical Bernstein* bonus in the
index, with a non-asymptotic correction term of order `3 log t / N_a` demanded by Bennett's
and Bernstein's inequalities.

**MOSS (Audibert & Bubeck, 2010).** An improved UCB using an exploration term of the form
`log( t / (K N_a) )_+`, achieving the *distribution-free minimax-optimal* rate
`O(sqrt(K n))`.

**DMED — Deterministic Minimum Empirical Divergence (Honda & Takemura, 2010).** A
large-deviations *elimination* policy: maintain a list of arms close enough to the current
empirical best to still be plausibly optimal, and play them. The closeness test uses the
empirical divergence `N_a d(mu_hat_a, max_b mu_hat_b) < log t`. DMED is first-order optimal
for bounded-support models and requires the rate function of the reward law.

## Evaluation settings

The natural yardsticks already in use for bandit policies, all measuring expected cumulative
regret (or the closely-tied expected number of sub-optimal pulls) as a function of the
horizon, typically plotted against `log` time:

- **Two-arm Bernoulli**: a simple instance with means such as `mu_1 = 0.9`, `mu_2 = 0.8`;
  track the mean number of draws of the sub-optimal arm vs. time. Because the regret
  distribution is poorly concentrated (rare runs where the best arm is under-estimated early
  produce `O(n)` regret), reliable estimates need many independent runs — on the order of
  `N = 50,000`, with `N` much larger than the horizon `n`, and box-plots (not just means) to
  expose the upper tail.
- **Ten-arm low-reward Bernoulli**: a harder, application-flavored instance where all means
  are small (e.g. an optimal arm at `0.1` and groups of arms at `0.05, 0.02, 0.01`) — the
  rare-event regime where the asymmetry of the confidence width should matter most.
- **Bounded non-Bernoulli rewards**: e.g. exponential rewards truncated to `[0, x_max]`, to
  test a policy on non-binary, non-discrete, bounded reward distributions and on whether a
  variant matched to the reward family does better.
- Protocol: identical instances across policies; metric is mean cumulative regret with
  central and extreme quantiles over the many runs; the per-arm Lai-Robbins floor
  `Delta_a / d(mu_a, mu_{a*})` drawn as a reference line.

## Code framework

The policy plugs into a standard bandit harness. The harness already provides: the per-arm
sufficient statistics (a pull count `N_a` and a reward sum `S_a` per arm, so `mu_hat_a =
S_a / N_a`), the round counter `t`, and the loop that asks the policy which arm to pull, draws
the reward, and feeds it back. It also already provides a numerical primitive for the
Bernoulli KL divergence and a generic *confidence-bound* helper — given an empirical mean, a
count, and a per-arm deviation budget, return the largest mean consistent with that budget by
inverting the (convex, increasing) divergence with a root-finder. What is *not* settled is
the index itself: exactly which number to compute per arm from `(N_a, S_a, t)`, and how the
deviation budget scales with `t` and `N_a`. That is the single empty slot.

```python
import numpy as np


def divergence_bernoulli(p, q):
    """KL(Bernoulli(p) || Bernoulli(q)), edges clipped to [eps, 1-eps]."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def largest_consistent_mean(mu_hat, n, budget):
    """Largest mean q in [mu_hat, 1] whose divergence from mu_hat costs at most `budget`:
       max { q : n * divergence_bernoulli(mu_hat, q) <= budget }.
       Found by root-finding on the increasing branch q -> d(mu_hat, q) (convex, increasing
       on [mu_hat, 1]), with the Pinsker/Gaussian bound as a safe upper endpoint."""
    if n == 0:
        return 1.0
    threshold = budget / n
    upper = min(1.0, mu_hat + np.sqrt(threshold / 2.0))
    # bisection / Newton on q in (mu_hat, upper) for divergence_bernoulli(mu_hat, q) == threshold;
    # if the entire bracket is inside the ball, the helper returns the upper endpoint.
    return _root_find(lambda q: divergence_bernoulli(mu_hat, q) - threshold, mu_hat, upper)


class BanditPolicy:
    """Index policy. Pulls the arm of highest index; index built from each arm's own data."""

    def __init__(self, K, context_dim=0):
        self.K = K
        self.counts = np.zeros(K)      # N_a
        self.rewards = np.zeros(K)     # S_a

    def reset(self):
        self.counts[:] = 0
        self.rewards[:] = 0

    def select_arm(self, t, context=None):
        # Pull each arm once first.
        if t < self.K:
            return t
        best_arm, best_index = 0, -np.inf
        for a in range(self.K):
            if self.counts[a] == 0:
                return a
            mu_hat = self.rewards[a] / self.counts[a]
            # TODO: the per-arm index we will design -- the one number to maximize,
            #       built from (mu_hat, self.counts[a], t).
            index = None  # pass
            if index > best_index:
                best_index, best_arm = index, a
        return best_arm

    def update(self, arm, reward, context=None):
        self.counts[arm] += 1
        self.rewards[arm] += reward
```

The harness supplies `(N_a, S_a, t)` and the divergence machinery; `select_arm` is where the
index rule will live.
