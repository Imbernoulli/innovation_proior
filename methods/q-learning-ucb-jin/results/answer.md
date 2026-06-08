# Q-learning with UCB exploration, distilled

## Problem

In a tabular episodic MDP $(\mathcal S,\mathcal A,H,\mathbb P,r)$ — $S$ states, $A$ actions, horizon $H$, step-dependent kernels $\mathbb P_1,\dots,\mathbb P_H$, rewards in $[0,1]$, no simulator and no mid-episode reset — does a *model-free* algorithm (a Q-table, nothing more) achieve $\sqrt T$ regret, matching model-based methods? The answer is yes: **Q-learning with a UCB exploration bonus**, provided the learning rate weights recent samples ($\alpha_t=\Theta(H/t)$, not $1/t$).

## Key idea

Two coupled fixes turn plain Q-learning from exponential-in-$H$ to near-optimal:

1. **Optimistic, directed exploration.** Replace $\varepsilon$-greedy (which needs $\Omega(A^{H/2})$ episodes on a combination lock) with an added UCB bonus $b_t$ to the Bellman target, making $Q^k\ge Q^\star$ at all times; act greedily under this upper-confidence value.
2. **A recent-weighting learning rate** $\alpha_t=\frac{H+1}{H+t}$. The induced cumulative weights $\alpha_t^i=\alpha_i\prod_{j=i+1}^t(1-\alpha_j)$ track the still-changing next-layer value $V_{h+1}$ instead of uniform-averaging stale targets the way $1/t$ does. The decisive algebraic property is $\sum_{t=i}^\infty\alpha_t^i=1+\tfrac1H$, which caps the per-layer error amplification at $(1+\tfrac1H)^H\le e$.

## The algorithm (UCB-Hoeffding)

Initialize $Q_h(x,a)\leftarrow H$, $N_h(x,a)\leftarrow 0$. With $\iota=\log(SAT/p)$:

For episode $k=1,\dots,K$, receive $x_1$; for $h=1,\dots,H$:
- take $a_h\leftarrow\arg\max_{a'}Q_h(x_h,a')$, observe $x_{h+1}$;
- $t=N_h(x_h,a_h)\mathrel{+}=1$;  $\ \ b_t\leftarrow c\sqrt{H^3\iota/t}$;  $\ \ \alpha_t\leftarrow\frac{H+1}{H+t}$;
- $Q_h(x_h,a_h)\leftarrow(1-\alpha_t)Q_h(x_h,a_h)+\alpha_t\big[r_h(x_h,a_h)+V_{h+1}(x_{h+1})+b_t\big]$;
- $V_h(x_h)\leftarrow\min\{H,\ \max_{a'}Q_h(x_h,a')\}$.

## Learning-rate properties (Lemma)

For $\alpha_t^0=\prod_{j=1}^t(1-\alpha_j)$, $\alpha_t^i=\alpha_i\prod_{j=i+1}^t(1-\alpha_j)$, with $\sum_{i=1}^t\alpha_t^i=1$:
- (a) $\frac{1}{\sqrt t}\le\sum_{i=1}^t\frac{\alpha_t^i}{\sqrt i}\le\frac{2}{\sqrt t}$ — keeps the bonus accumulator $\beta_t=2\sum_i\alpha_t^i b_i=\Theta(\sqrt{H^3\iota/t})$;
- (b) $\max_i\alpha_t^i\le\frac{2H}{t}$ and $\sum_{i=1}^t(\alpha_t^i)^2\le\frac{2H}{t}$ — controls the variance of the weighted martingale;
- (c) $\sum_{t=i}^\infty\alpha_t^i=1+\frac1H$ — the constant cross-layer amplification factor.

## Regret theorem (UCB-Hoeffding)

With $b_t=c\sqrt{H^3\iota/t}$, with probability $1-p$,
$$\mathrm{Regret}(K)\le O\!\big(\sqrt{H^4SAT\,\iota}\big)=\tilde O\!\big(\sqrt{H^4SAT}\big).$$

### Proof sketch
- **Optimism.** Unfolding the update, $(Q_h^k-Q_h^\star)(x,a)=\alpha_t^0(H-Q_h^\star)+\sum_i\alpha_t^i\big[(V_{h+1}^{k_i}-V_{h+1}^\star)(x_{h+1}^{k_i})+[(\hat{\mathbb P}_h^{k_i}-\mathbb P_h)V_{h+1}^\star](x,a)+b_i\big]$. The middle stochastic term is an $\alpha$-weighted martingale-difference sum of single-sample Bellman backups of $V^\star\in[0,H]$; Azuma-Hoeffding plus (b) bounds it by $c\sqrt{H^3\iota/t}$ ($H^2$ from the value range squared, one $H$ from $\sum(\alpha_t^i)^2\le 2H/t$). Choosing $b_t$ at exactly this scale gives $\beta_t=2\sum_i\alpha_t^i b_i\in[2c\sqrt{H^3\iota/t},4c\sqrt{H^3\iota/t}]$ and, by induction on $h$, $0\le(Q_h^k-Q_h^\star)\le\alpha_t^0H+\sum_i\alpha_t^i(V_{h+1}^{k_i}-V_{h+1}^\star)(x_{h+1}^{k_i})+\beta_t$.
- **Regret recursion.** With $\delta_h^k=(V_h^k-V_h^{\pi_k})(x_h^k)$, $\phi_h^k=(V_h^k-V_h^\star)(x_h^k)\ge0$ (optimism $\Rightarrow$ Regret $\le\sum_k\delta_1^k$), the per-step bound $\delta_h^k\le\alpha_t^0H+\sum_i\alpha_t^i\phi_{h+1}^{k_i}+\beta_t-\phi_{h+1}^k+\delta_{h+1}^k+\xi_{h+1}^k$, with $\xi_{h+1}^k=[(\mathbb P_h-\hat{\mathbb P}_h^k)(V_{h+1}^\star-V_{h+1}^{\pi_k})](x_h^k,a_h^k)$, regroups (each $\phi_{h+1}^{k_i}$ collects total weight $\sum_{t=i}^\infty\alpha_t^i=1+\tfrac1H$ by (c)) into $\sum_k\delta_h^k\le SAH+(1+\tfrac1H)\sum_k\delta_{h+1}^k+\sum_k(\beta_{n_h^k}+\xi_{h+1}^k)$ using $\phi\le\delta$.
- **Telescope.** Recursing $h=1..H$, $(1+\tfrac1H)^H\le e$ gives $\sum_k\delta_1^k\le O(H^2SA+\sum_{h,k}(\beta+\xi))$. Pigeonhole: $\sum_k\beta_{n_h^k}\le O(\sqrt{H^3SAK\iota})=O(\sqrt{H^2SAT\iota})$ per layer $\Rightarrow O(\sqrt{H^4SAT\iota})$ total; Azuma on $\sum_{h,k}\xi\le cH\sqrt{T\iota}$. Absorbing the additive $H^2SA$ yields the bound. $\square$

## Bernstein refinement (UCB-Bernstein)

Scaling the bonus by the online **empirical variance** $W_t$ of next-state values (running $\sum V_{h+1}$, $\sum V_{h+1}^2$) and exploiting that the **total variance per episode is $O(H^2)$** not $O(H^3)$ (law of total variance: $\sum_{k,h}\mathbb V_h V_{h+1}^{\pi_k}\le O(HT+H^3\iota)$) saves one $\sqrt H$. With $\beta_t=\min\{c_1(\sqrt{\tfrac Ht(W_t+H)\iota}+\tfrac{\sqrt{H^7SA}\,\iota}{t}),c_2\sqrt{\tfrac{H^3\iota}{t}}\}$ and $b_t=\tfrac{\beta_t-(1-\alpha_t)\beta_{t-1}}{2\alpha_t}$, with probability $1-p$,
$$\mathrm{Regret}(K)\le O\!\big(\sqrt{H^3SAT\,\iota}+\sqrt{H^9S^3A^3}\,\iota^2\big)=\tilde O\!\big(\sqrt{H^3SAT}\big).$$

## Lower bound and PAC corollary

Any algorithm on this setting (distinct $\mathbb P_h$) has regret $\Omega(\sqrt{H^2SAT})$ ($H$ JAO chains in series; only $T/H$ samples per step). So UCB-H is within $H$ and UCB-B within $\sqrt H$ of optimal — model-free, matching model-based up to a single $\sqrt H$. Regret-to-PAC: choosing $\pi=\pi_k$ at random returns an $\varepsilon$-optimal policy in $\tilde O(H^5SA/\varepsilon^2)$ (UCB-H) and $\tilde O(H^4SA/\varepsilon^2)$ (UCB-B) samples. Time $O(T)$, space $O(SAH)$ — strictly better than the $\tilde O(TS^2A)$ time / $O(S^2AH)$ space of model-based methods.

## Executable form (UCB-Hoeffding)

```python
import numpy as np

def q_learning_ucb_hoeffding(env, K, c=1.0, p=0.1):
    """Model-free, O(SAH) space. env exposes S, A, H, reset()->x, step(h,x,a)->(r,x')."""
    S, A, H = env.S, env.A, env.H
    iota = np.log(S * A * (K * H) / p)              # log(SAT/p)
    Q = np.full((H, S, A), float(H))                # optimistic initialization
    V = np.zeros((H + 1, S)); V[:H] = H             # V_{H+1} == 0
    N = np.zeros((H, S, A), dtype=int)
    for k in range(K):
        x = env.reset()
        for h in range(H):
            a = int(Q[h, x].argmax())               # greedy on optimistic Q
            r, x_next = env.step(h, x, a)
            N[h, x, a] += 1
            t = N[h, x, a]
            alpha = (H + 1) / (H + t)                # recent-weighting rate
            b = c * np.sqrt(H**3 * iota / t)         # UCB-Hoeffding bonus
            Q[h, x, a] = (1 - alpha) * Q[h, x, a] + alpha * (r + V[h + 1, x_next] + b)
            V[h, x] = min(H, Q[h, x].max())          # clip to [0,H]
            x = x_next
    return Q
```
