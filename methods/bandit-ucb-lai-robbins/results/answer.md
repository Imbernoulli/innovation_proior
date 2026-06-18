# Lai-Robbins Lower Bound And UCB

## Source Status

The original 1985 Lai-Robbins article remains a strict primary-source blocker: it was identified by DOI and metadata, but no local full-text PDF was retrieved. The formulas below are therefore grounded in later faithful sources that explicitly restate or derive the Lai-Robbins lower bound, plus Auer-Cesa-Bianchi-Fischer's finite-time UCB1 theorem.

## Lower Bound

For an unstructured stochastic bandit class `E = M_1 x ... x M_K`, any policy consistent over `E`, and any suboptimal arm `i`,

```text
liminf_{n -> infinity} E[T_i(n)] / log n >= 1 / d_inf(P_i, mu*, M_i),
d_inf(P_i, mu*, M_i) = inf { D(P_i || Q) : Q in M_i and mean(Q) > mu* }.
```

For a regular one-parameter family with a unique optimal parameter `theta*`, this gives

```text
liminf_{n -> infinity} R_n / log n
  >= sum_{i: Delta_i > 0} Delta_i / KL(theta_i, theta*).
```

Interpretation: a bad arm must be sampled until the policy accumulates about `log n` likelihood-ratio evidence against the closest alternative world where that arm is best.

For Bernoulli arms the denominator is `kl(mu_i, mu*)`. For Gaussian rewards with known variance `sigma^2`, the denominator is `Delta_i^2/(2 sigma^2)`, so the pull-count lower-bound constant is `2 sigma^2 / Delta_i^2`.

## UCB1 Rule

The simple finite-time rule here is Auer-Cesa-Bianchi-Fischer UCB1, not the cyclic Lai-Robbins 1985 allocation rule. Sample each arm once. At round `t`, choose

```text
argmax_i hat_mu_i + sqrt(2 log t / T_i(t-1)).
```

For rewards in `[0,1]`,

```text
E[T_i(n)] <= 8 log n / Delta_i^2 + 1 + pi^2/3,
R_n <= sum_{i: Delta_i > 0} 8 log n / Delta_i
       + (1 + pi^2/3) sum_i Delta_i.
```

A refined 1-subgaussian anytime UCB index from later analyses uses `f(t)=1+t log^2(t)` and satisfies

```text
limsup R_n / log n <= sum_{i: Delta_i > 0} 2 / Delta_i,
```

matching the information lower-bound constant for unit-variance Gaussian rewards.

For Bernoulli or exponential-family rewards, closing the Lai-Robbins information constant generally requires KL-shaped confidence sets such as KL-UCB rather than this quadratic Hoeffding radius.

## Code Artifact

```python
import numpy as np

def select_ucb1(stats, t):
    """Auer-Cesa-Bianchi-Fischer UCB1 selector.

    `t` is the 1-based current round, and `stats` contains observations
    from the previous `t-1` rounds.
    """
    unplayed = np.flatnonzero(stats.counts == 0)
    if unplayed.size:
        return int(unplayed[0])
    index = stats.means + np.sqrt(2.0 * np.log(t) / stats.counts)
    return int(np.argmax(index))

def select_asymptotic_ucb(stats, t):
    """Later 1-subgaussian asymptotic UCB with f(t)=1+t log^2(t)."""
    unplayed = np.flatnonzero(stats.counts == 0)
    if unplayed.size:
        return int(unplayed[0])
    log_t = np.log(max(t, 2))
    f_t = 1.0 + t * log_t * log_t
    index = stats.means + np.sqrt(2.0 * np.log(f_t) / stats.counts)
    return int(np.argmax(index))
```
