# Doubly Robust Off-Policy Evaluation

The estimator is not a weighted average of the direct method and importance sampling. It is a single estimating equation: a model baseline plus an importance-weighted residual.

## Contextual-Bandit Form

For logged samples `(x_i, a_i, r_i, p_i)`, target policy `pi`, behavior propensity estimate `p_hat`, and reward model `q_hat`, define

`q_hat(x,pi) = sum_a pi(a|x) q_hat(x,a)`

and

`V_hat_DR = (1/n) sum_i [ q_hat(x_i,pi) + (pi(a_i|x_i) / p_hat_i) * (r_i - q_hat(x_i,a_i)) ]`.

For a deterministic target policy, this is the Dudik-Langford-Li estimator

`q_hat(x,pi(x)) + I(pi(x)=a) / p_hat(a|x,h) * (r_a - q_hat(x,a))`.

If `Delta(a,x) = q_hat(x,a) - q(x,a)` and `delta(a,x) = 1 - p(a|x) / p_hat(a|x)`, then, with nuisance estimates independent of the evaluation fold,

`Bias(V_hat_DR) = E[sum_a pi(a|x) Delta(a,x) delta(a,x)]`.

For deterministic `pi`, this is `E[Delta delta]`. Thus the estimator is unbiased if the reward model is correct or if the behavior propensities are correct. Its importance-weighting variance penalty is scaled by squared model error, not squared reward magnitude.

## Sequential RL Form

Let `rho_t = pi_1(a_t|s_t) / pi_0(a_t|s_t)`, fit `Q_hat`, and define

`V_hat(s) = sum_a pi_1(a|s) Q_hat(s,a)`.

Initialize `V_DR^0 = 0` and recurse backward:

`V_DR^{H+1-t} = V_hat(s_t) + rho_t * (r_t + gamma V_DR^{H-t} - Q_hat(s_t,a_t))`.

Step-wise IS is the special case `Q_hat = 0`. With `Q_hat = Q`, the reducible action-mismatch variance disappears; the remaining transition/reward variance is the lower-bound floor in Jiang-Li's tree-MDP analysis.

## Minimal Implementation

```python
import numpy as np

def doubly_robust_bandit(rewards, actions, pscore, contexts, pi_target, q_hat):
    vals = []
    for r, a, p, x in zip(rewards, actions, pscore, contexts):
        pi_x = pi_target(x)
        q_x = q_hat(x)
        baseline = float(np.dot(pi_x, q_x))
        iw = pi_x[a] / p
        vals.append(baseline + iw * (r - q_x[a]))
    return float(np.mean(vals))

def doubly_robust_sequential(trajectory, gamma, pi_target, pi_behavior, q_hat, v_hat):
    value = 0.0
    for s, a, r in reversed(trajectory):
        rho = pi_target(s)[a] / pi_behavior(s)[a]
        value = v_hat(s) + rho * (r + gamma * value - q_hat(s, a))
    return value
```

Use sample splitting or cross-fitting: `q_hat`, `v_hat`, and any estimated propensities must be fit independently of the evaluation samples for the cancellation argument to apply.
