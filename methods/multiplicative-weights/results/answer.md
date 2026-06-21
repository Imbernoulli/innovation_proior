# Multiplicative Weights Update Method

## Algorithm

Given `n` decisions, a horizon `T`, costs `m_i(t) in [-1,1]`, and `eta <= 1/2`:

```text
Initialize w_i(1) = 1 for all i.
For t = 1, ..., T:
  Phi(t) = sum_i w_i(t)
  p_i(t) = w_i(t) / Phi(t)
  sample/play from p(t)
  observe the full cost vector m(t)
  w_i(t+1) = w_i(t) * (1 - eta * m_i(t)) for every i
```

The learner's expected cost on round `t` is `m(t) . p(t)`.

## Regret Guarantee

For every fixed decision `i`,

`sum_t m(t) . p(t) <= sum_t m_i(t) + eta sum_t |m_i(t)| + ln(n)/eta`.

If `|m_i(t)| <= 1`, then regret against the best fixed decision is at most

`eta T + ln(n)/eta`.

When `sqrt(ln(n)/T) <= 1/2`, choosing `eta = sqrt(ln(n)/T)` gives regret at most

`2 sqrt(T ln n)`.

For shorter horizons, keep `eta <= 1/2` and use `eta T + ln(n)/eta` directly.

## Proof Skeleton

Let `Phi(t) = sum_i w_i(t)`.

Upper bound:

`Phi(t+1) = Phi(t)(1 - eta m(t) . p(t)) <= Phi(t) exp(-eta m(t) . p(t))`,

so

`Phi(T+1) <= n exp(-eta sum_t m(t) . p(t))`.

Lower bound for any fixed decision `i`:

`Phi(T+1) >= prod_t (1 - eta m_i(t))`.

Split rounds by the sign of `m_i(t)` and use

`(1 - eta)^x <= 1 - eta x` for `x in [0,1]`,

`(1 + eta)^(-x) <= 1 - eta x` for `x in [-1,0]`.

Taking logs and applying

`ln(1/(1-eta)) <= eta + eta^2`,

`ln(1+eta) >= eta - eta^2`,

yields the displayed regret bound.

## Useful Variants

Hedge uses `w_i(t+1) = w_i(t) exp(-eta m_i(t))`, with `eta <= 1`, and gives

`sum_t m(t) . p(t) <= sum_t m_i(t) + eta sum_t (m(t))^2 . p(t) + ln(n)/eta`.

The gains form flips the sign, using

`w_i(t+1) = w_i(t)(1 + eta m_i(t))`,

with `eta <= 1`, and guarantees

`sum_t m(t) . p(t) >= sum_t m_i(t) - eta sum_t |m_i(t)| - ln(n)/eta`.

The relative-entropy form views the update as reducing `RE(p || p(t))` toward any comparator distribution `p`; with convex restrictions on allowed distributions, a relative-entropy projection preserves that progress.

## Instantiations

Zero-sum games: rows are decisions, a best-response column is the cost vector, and the average play gives `epsilon`-optimal mixed strategies in `O(log(n)/epsilon^2)` oracle calls.

LP feasibility: constraints are decisions, the oracle satisfies one weighted average constraint `p^T A x >= p^T b`, and the theoretical cost of constraint `i` is `(A_i x - b_i)/rho`. Positive satisfaction lowers future weight; negative violation raises it. In the gains-form implementation this same sign convention appears as reward `b_i - A_i x`, so violated constraints get positive reward under `1 + eta reward`. Averaging oracle outputs gives an `epsilon`-feasible point in `O(ell rho log(m)/epsilon^2)` oracle calls under an `(ell,rho)` bounded oracle.

Set cover: with `eta = 1`, covered elements drop to zero weight, so the rule becomes greedy set cover and gives a `ceil(ln n)` approximation.

Boosting: training examples are decisions. Correctly classified examples lose relative emphasis, misclassified examples gain relative emphasis, and a majority vote of weak hypotheses drives training error below `epsilon` after `ceil((2/gamma^2) ln(1/epsilon))` rounds under a `gamma`-weak learner.

## Code Artifact

The implementation artifact is in `code/j2kun_mwua.py` and `code/j2kun_linear.py`, which match the canonical `j2kun/mwua` raw files. It uses the gains form for the generic update, `weights[i] *= 1 + learningRate * reward(...)`. The LP wrapper makes constraints the experts, uses `reward(i, x) = b_i - A_i x`, queries the weighted-average oracle with `A.T @ weights` and `weights.dot(b)`, binary-searches over the objective value, and averages the returned points.
