# REINFORCE: Score-Function Policy-Gradient Estimator

## Core Estimator

For a stochastic unit or policy with sampled output `y` and probability law `g_theta(y)`, optimize expected reinforcement

```text
J(theta) = E[r].
```

Use the log-derivative identity:

```text
grad_theta g_theta(y) = g_theta(y) grad_theta log g_theta(y)
```

Then the one-sample estimator

```text
g_hat(theta) = (r - b) grad_theta log g_theta(y)
```

is unbiased for `grad_theta J(theta)` when `b` is independent of the current sampled output under the conditioning used by the unit. The baseline does not bias the estimator because

```text
E[grad_theta log g_theta(Y)] = 0.
```

## Williams Unit Form

For unit `i`,

```text
g_i(y_i | w_i, x_i) = Pr(Y_i = y_i | w_i, x_i)
e_ij = d log g_i(Y_i | w_i, x_i) / d w_ij
Delta w_ij = alpha_ij (r - b_ij) e_ij
```

Williams's theorem shows that

```text
E[Delta w_ij | W] = alpha_ij d E[r | W] / d w_ij
```

for each weight under the stated assumptions. With a common positive `alpha`, the expected update is exactly proportional to the gradient of expected reinforcement.

## Useful Special Cases

Bernoulli unit:

```text
d log g(y) / d p = (y - p) / (p(1 - p))
```

Bernoulli logistic unit:

```text
e_ij = (y_i - p_i) x_j
Delta w_ij = alpha (r - b) (y_i - p_i) x_j
```

Episode or trajectory:

```text
g_hat(theta) = (R - b) sum_t grad_theta log pi_theta(a_t | s_t)
```

Reward-to-go or advantage form:

```text
g_hat(theta) = sum_t A_t grad_theta log pi_theta(a_t | s_t)
```

Gaussian unit:

```text
d log g / d mu = (y - mu) / sigma^2
d log g / d sigma = ((y - mu)^2 - sigma^2) / sigma^3
```

## Code Artifact

The stored Spinning Up examples implement the estimator as a loss whose gradient is the negative policy-gradient estimate:

```python
logp = get_policy(obs).log_prob(act)
loss = -(logp * weights).mean()
```

In full VPG:

```python
loss_pi = -(logp * adv).mean()
```

Here `weights` or `adv` is the return, reward-to-go, or advantage multiplier. This is the modern code form of "log-probability score times reward."

## What The Method Contributes

REINFORCE turns scalar reinforcement for stochastic units into a valid stochastic gradient estimator. Its essential move is to use the score of the sampled action, `grad log probability`, as the local eligibility and multiply it by reward or advantage. The estimator is unbiased, supports output-independent baselines, works for discrete or continuous stochastic units, and extends to policies by summing log-probability scores over time.

The method does not solve variance or convergence by itself. Baselines, critics, reward-to-go, and advantage estimates are practical variance-reduction layers around the same unbiased score-function core.
