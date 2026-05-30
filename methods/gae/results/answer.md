# Generalized Advantage Estimation (GAE)

## Problem

Policy gradient methods optimize a parameterized stochastic policy by ascending
the expected return with the score-function estimator
$g=\mathbb{E}[\sum_t \Psi_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)]$. The
estimator's variance grows with the horizon because each action's credit is
confounded with the rewards caused by all other actions and by environment
noise. Using the advantage $\Psi_t=A^\pi(s_t,a_t)=Q^\pi-V^\pi$ gives nearly the
lowest variance among the standard choices, but $A^\pi$ is unknown and must be
estimated from an imperfect value function. GAE is a one-parameter family of
advantage estimators that smoothly trades the bias of short bootstrapped
estimates against the variance of full Monte-Carlo returns.

## Key idea

Let $V$ be an approximate value function and define the TD residual
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$
Summing $k$ residuals telescopes into a $k$-step return minus a baseline,
$\hat A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}=-V(s_t)+r_t+\dots+\gamma^{k-1}r_{t+k-1}+\gamma^k V(s_{t+k})$,
whose bias (carried entirely by the bootstrap term $\gamma^k V(s_{t+k})$) shrinks
as $k$ grows while its variance rises. GAE takes the exponentially-weighted
average of all of them,
$$
\hat A_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{k=1}^{\infty}\lambda^{k-1}\,\hat A_t^{(k)},
$$
which collapses to a single discounted sum of TD residuals:
$$
\boxed{\;\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}\;}
$$

The two parameters play distinct roles:

- $\gamma\in[0,1]$ sets the credit-assignment horizon and the scale of $V^{\pi,\gamma}$. It biases the gradient even when $V$ is exact (it estimates $A^{\pi,\gamma}$ rather than the undiscounted advantage).
- $\lambda\in[0,1]$ trades bias against variance and biases the gradient *only* through the inaccuracy of $V$.

**Limits.**
$$
\lambda=0:\quad \hat A_t = \delta_t = r_t+\gamma V(s_{t+1})-V(s_t)\quad\text{(one-step TD; low variance, high bias unless $V$ exact)}
$$
$$
\lambda=1:\quad \hat A_t = \sum_{l=0}^{\infty}\gamma^l\delta_{t+l}=\sum_{l=0}^{\infty}\gamma^l r_{t+l}-V(s_t)\quad\text{(Monte-Carlo minus baseline; high variance, unbiased for any $V$)}
$$

**$\gamma$-just property.** An estimator $\hat A_t=Q_t(s_{t:\infty},a_{t:\infty})-b_t(s_{0:t},a_{0:t-1})$ with $\mathbb{E}[Q_t\mid s_t,a_t]=Q^{\pi,\gamma}(s_t,a_t)$ yields an unbiased estimate of the discounted policy gradient $g^\gamma$, because $\mathbb{E}_{a_t}[\nabla_\theta\log\pi(a_t\mid s_t)]=0$ kills any past-measurable baseline. Hence GAE$(\gamma,1)$ is unbiased for any $V$, and GAE$(\gamma,0)$ only for $V=V^{\pi,\gamma}$.

**Reward-shaping view.** With potential $\Phi=V$, the shaped reward
$\tilde r=r+\gamma V(s')-V(s)=\delta$, and GAE is the $\gamma\lambda$-discounted
return of the shaped MDP. Shaping with a good $V$ concentrates the response
function near zero delay; the steeper discount $\gamma\lambda$ then cuts the
long-delay noise.

## Full algorithm

Each iteration, using value function $V_{\phi_i}$:

1. Simulate the current policy $\pi_{\theta_i}$ for a batch of $N$ timesteps.
2. Compute $\delta_t=r_t+\gamma V_{\phi_i}(s_{t+1})-V_{\phi_i}(s_t)$ at all timesteps.
3. Compute $\hat A_t=\sum_{l}(\gamma\lambda)^l\delta_{t+l}$ (backward recursion $\hat A_t=\delta_t+\gamma\lambda\,\hat A_{t+1}$).
4. Policy update by trust-region policy optimization:
   $\max_\theta \frac1N\sum_n \frac{\pi_\theta(a_n\mid s_n)}{\pi_{\theta_i}(a_n\mid s_n)}\hat A_n$
   s.t. $\frac1N\sum_n D_{\mathrm{KL}}(\pi_{\theta_i}(\cdot\mid s_n)\|\pi_\theta(\cdot\mid s_n))\le\epsilon$,
   solved as $\theta-\theta_i\propto -F^{-1}g$ (natural gradient) via CG with Fisher-vector products and a line search.
5. Value-function update under its own trust region:
   $\min_\phi\sum_n\|V_\phi(s_n)-\hat V_n\|^2$ s.t.
   $\frac1N\sum_n\frac{\|V_\phi(s_n)-V_{\phi_i}(s_n)\|^2}{2\sigma^2}\le\epsilon$,
   with $\hat V_n=\hat A_n+V_{\phi_i}(s_n)$, solved by CG with Gauss-Newton/Fisher-vector products.

The policy is updated with the **old** value function $V_{\phi_i}$, and $V$ is
refit afterward — fitting $V$ first would drive the residuals toward zero and
bias the gradient toward zero.

## Code

```python
import numpy as np


def discount_cumsum(x, discount):
    """out[t] = x[t] + discount*x[t+1] + discount^2*x[t+2] + ..., O(T)."""
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        out[t] = running
    return out


def compute_gae(rewards, values, last_value, gamma, lam):
    """GAE for one episode.

    rewards    : r_0 .. r_{T-1}
    values     : V(s_0) .. V(s_{T-1})
    last_value : V(s_T) bootstrap (0 if s_T terminal)
    returns    : advantages A_t and value targets A_t + V(s_t)
    """
    vals_next = np.append(values[1:], last_value)
    deltas = rewards + gamma * vals_next - values         # TD residual delta_t
    advantages = discount_cumsum(deltas, gamma * lam)     # A_t = sum (gamma*lam)^l delta_{t+l}
    returns = advantages + values                         # value-regression targets
    return advantages, returns


def compute_gae_batch(rewards, values, dones, last_value, gamma, lam):
    """Backward-recursion form over a rollout spanning multiple episodes.
    dones[t]=1 if s_{t+1} is terminal (no bootstrap across it)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def train(env, policy, value_fn, n_iters, batch_steps, gamma, lam, kl_policy, kl_vf):
    for _ in range(n_iters):
        # roll out the CURRENT policy; advantages use the OLD value function
        batch = collect_batch(env, policy, value_fn, batch_steps)

        advs, rets = [], []
        for ep in batch.episodes:
            last_v = 0.0 if ep.terminated else value_fn.predict(ep.last_state)
            a, r = compute_gae(ep.rewards, ep.values, last_v, gamma, lam)
            advs.append(a); rets.append(r)
        advs = (lambda a: (a - a.mean()) / (a.std() + 1e-8))(np.concatenate(advs))
        rets = np.concatenate(rets)

        # KL-trust-region policy step (natural-gradient direction, CG + FVP)
        trpo_step(policy, batch.states, batch.actions, advs, kl_policy)
        # value-function trust-region fit, AFTER the policy step
        value_fn.trust_region_fit(batch.states, rets, kl_vf)
```

Typical settings: $\gamma\in[0.99,0.995]$, $\lambda\in[0.9,0.99]$. The best
$\lambda$ is generally smaller than the best $\gamma$, because $\lambda$
introduces far less bias than $\gamma$ for a reasonably accurate value function.
