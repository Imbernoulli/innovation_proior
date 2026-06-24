# Generalized Advantage Estimation (GAE)

## Problem

Policy gradient methods optimize a stochastic policy with an estimator
$g=\mathbb{E}[\sum_t \Psi_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)]$.
Raw returns make $\Psi_t$ extremely noisy over long horizons. The ideal
low-variance multiplier is the advantage
$A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)$, but the advantage is unknown and must
be estimated from sampled rewards and an imperfect value function.

GAE is the two-parameter advantage-estimation family that uses $\gamma$ as a
credit-horizon / variance-reduction parameter and $\lambda$ as the interpolation
between one-step bootstrapping and Monte Carlo credit assignment.

## Estimator

For an approximate value function $V$, define the discounted TD residual
$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t).
$$
Summing $k$ residuals telescopes:
$$
\hat A_t^{(k)}
= \sum_{l=0}^{k-1}\gamma^l\delta_{t+l}^V
= -V(s_t)+r_t+\gamma r_{t+1}+\cdots+\gamma^{k-1}r_{t+k-1}
  +\gamma^k V(s_{t+k}).
$$
This is a $k$-step return minus a state baseline. As $k$ grows, the bootstrap
tail where value-function error enters the future return is discounted more
heavily. The $-V(s_t)$ term may make the conditional advantage estimate offset
from $A^{\pi,\gamma}$ when $V$ is imperfect, but it is state-only and therefore
does not bias the policy gradient.

GAE takes the geometrically weighted average of all $k$-step estimators:
$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
= (1-\lambda)\sum_{k=1}^{\infty}\lambda^{k-1}\hat A_t^{(k)}
= \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V.
$$

Limits:

- $\lambda=0$: $\hat A_t=\delta_t^V$, low variance, biased unless
  $V=V^{\pi,\gamma}$.
- $\lambda=1$: $\hat A_t=\sum_{l\ge0}\gamma^l r_{t+l}-V(s_t)$, Monte Carlo
  return minus a baseline, $\gamma$-just for any $V$ but high variance.

$\gamma<1$ biases the gradient even with a perfect value function because it
targets $A^{\pi,\gamma}$ rather than the undiscounted advantage. $\lambda<1$
adds policy-gradient bias only through value-function error; with
$V=V^{\pi,\gamma}$, every $\lambda$ is $\gamma$-just.

Reward shaping gives the same formula: with potential $\Phi=V$,
$\tilde r=r+\gamma V(s')-V(s)=\delta^V$, so GAE is the
$\gamma\lambda$-discounted return of shaped rewards.

## Algorithm

At iteration $i$:

1. Roll out $\pi_{\theta_i}$ and evaluate $V_{\phi_i}(s_t)$ on the sampled states.
2. Compute $\delta_t^V=r_t+\gamma V_{\phi_i}(s_{t+1})-V_{\phi_i}(s_t)$.
3. Compute advantages by the backward recursion
   $\hat A_t=\delta_t^V+\gamma\lambda\hat A_{t+1}$, resetting at terminals.
4. Update the policy in a KL trust region. Equivalently, maximize
   $\frac1N\sum_n\frac{\pi_\theta(a_n\mid s_n)}
   {\pi_{\theta_i}(a_n\mid s_n)}\hat A_n$ with mean KL constrained, giving an
   ascent direction proportional to $F^{-1}g$; implementations commonly minimize
   the negative surrogate, whose descent direction is $-F^{-1}\nabla\ell$.
5. Fit the value function to Monte Carlo discounted returns
   $\hat V_t=\sum_{l\ge0}\gamma^l r_{t+l}$ under a value-function trust region.

Advantages must be computed using the old value function. The paper's algorithm
updates $\theta$ before $\phi$; rllab computes advantages, then fits the baseline
before calling the policy optimizer, but the current policy step still uses
advantages from the old baseline. The value target is the discounted return, not
$\hat A_t+V(s_t)$; the latter is the TD($\lambda$)-style target, which was tried
without improvement over the Monte Carlo target.

The natural-gradient direction $F^{-1}g$ used for the policy step is also what
least-squares projection of the advantage onto the compatible features
$\nabla_\theta\log\pi(a_t\mid s_t)$ yields: minimizing
$\sum_t\|\mathbf r\cdot\nabla_\theta\log\pi(a_t\mid s_t)-\hat A_t\|^2$ gives the
normal-equation solution $\mathbf r=F^{-1}g$ when $\hat A_t$ is $\gamma$-just,
with $F=\frac1N\sum_t\nabla\log\pi\,\nabla\log\pi^\top$ the empirical Fisher
information and $g=\frac1N\sum_t\nabla\log\pi\,\hat A_t$ the policy gradient.
GAE supplies a $\gamma$-just (or near-$\gamma$-just) $\hat A_t$ to plug in. This
is why a state-value $V$ with the $\lambda$ knob is preferred over a
parameterized $Q$-function: $V$ has a lower-dimensional input and is easier to
fit, and it permits the full $\lambda\in[0,1]$ interpolation, whereas an estimate
$Q(s,a)-V(s)$ is locked to the high-bias one-step corner ($\lambda=0$), whose
bias is empirically prohibitive on high-dimensional control.

## Reference Implementation

This is the rllab-compatible computation for complete paths, with `last_value=0`
at terminal boundaries. Passing a nonzero `last_value` is the usual extension for
truncated rollouts.

```python
import numpy as np
import scipy.signal


def discount_cumsum(x, discount):
    # out[t] - discount*out[t+1] = x[t]: a one-pole IIR filter run in reverse
    # time. out[t] = x[t] + discount*x[t+1] + discount^2*x[t+2] + ...
    return scipy.signal.lfilter([1], [1, -float(discount)], x[::-1], axis=0)[::-1]


def compute_gae_path(rewards, values, gamma, lam, last_value=0.0):
    """GAE for one path.

    rewards: r_0 .. r_{T-1}
    values:  V(s_0) .. V(s_{T-1}) from the old value function
    last_value: V(s_T) for a truncated path; 0 for a terminal path
    """
    values_ext = np.append(values, last_value)
    deltas = rewards + gamma * values_ext[1:] - values_ext[:-1]
    advantages = discount_cumsum(deltas, gamma * lam)

    rewards_ext = np.append(rewards, last_value)
    value_targets = discount_cumsum(rewards_ext, gamma)[:-1]
    return advantages, value_targets
```

For a rollout array spanning multiple episodes:

```python
def compute_gae_batch(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    value_targets = np.zeros(T, dtype=np.float32)

    lastgaelam = 0.0
    running_return = last_value
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]

        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        advantages[t] = lastgaelam

        running_return = rewards[t] + gamma * nonterminal * running_return
        value_targets[t] = running_return

    return advantages, value_targets
```
