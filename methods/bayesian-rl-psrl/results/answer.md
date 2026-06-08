# Posterior Sampling for Reinforcement Learning (PSRL)

## Problem

An agent acts in an unknown finite-horizon MDP $M^*=(\mathcal{S},\mathcal{A},R^{M^*},P^{M^*},\tau,\rho)$ over repeated episodes of length $\tau$, observing rewards and transitions but not the dynamics. The goal is to minimize regret against the optimal policy,
$$\text{Regret}(T,\pi)=\sum_{k=1}^{\lceil T/\tau\rceil}\sum_s\rho(s)\big(V^{M^*}_{\mu^*,1}(s)-V^{M^*}_{\mu_k,1}(s)\big),$$
with sublinear ($\tilde O(\sqrt T)$) growth, low computation, and the ability to encode a prior — without the statistical conservatism and joint-optimization cost of optimism-based methods.

## Key idea

Maintain a prior over MDPs. Each episode, **sample one MDP from the posterior and follow its optimal policy for the whole episode.** Exploration is driven by the variance of the posterior: each policy is played with the probability it is optimal, so plausibly-optimal policies keep being tried and ruled-out ones are dropped — no optimism bonus, no confidence-set construction. Sampling **once per episode** (not per timestep) is essential: it commits the agent to a coherent, temporally-extended plan, enabling deep exploration that a per-step resample destroys.

## Algorithm

> **PSRL.** Input: prior $f$ over MDPs.
> For each episode $k=1,2,\dots$:
> 1. Sample $M_k\sim f(\cdot\mid H_{t_k})$ (posterior given history).
> 2. Compute $\mu_k=\mu^{M_k}$, the optimal policy of $M_k$ (finite-horizon backward induction).
> 3. For $j=1,\dots,\tau$: take $a_t=\mu_k(s_t,j)$, observe $r_t,s_{t+1}$.
> 4. Update the posterior with the episode's observations.

With Dirichlet posteriors over transition rows (conjugate to multinomial) and normal-gamma over rewards (conjugate to normal), step 1 is closed-form and step 2 is one ordinary MDP solve.

## Main result

**Theorem (Bayesian regret).** If $f$ is the prior distribution of $M^*$, then for PSRL with episode length $\tau$,
$$\mathbb{E}\big[\text{Regret}(T,\pi^{\rm PS}_\tau)\big]=O\big(\tau\,S\sqrt{AT\log(SAT)}\big).$$
This holds for any prior over MDPs. By Markov's inequality, $\text{Regret}(T)/T^\alpha\to_p 0$ for any $\alpha>\tfrac12$, and the same $o_p(T^\alpha)$ statement holds conditioned on $M^*$ lying in any prior-positive family of MDPs.

## Proof sketch

**1. Posterior-sampling lemma.** Given $H_{t_k}$, the sampled MDP $M_k$ and the true MDP $M^*$ are identically distributed (the posterior *is* the belief about $M^*$). Hence for any measurable mapping $g$ that may be chosen using $H_{t_k}$,
$$\mathbb{E}[g(M^*)\mid H_{t_k}]=\mathbb{E}[g(M_k)\mid H_{t_k}],\qquad\text{and}\quad \mathbb{E}[g(M^*)]=\mathbb{E}[g(M_k)].$$

**2. Regret equivalence (implicit optimism).** Define the surrogate $\tilde\Delta_k=\sum_s\rho(s)\big(V^{M_k}_{\mu_k,1}(s)-V^{M^*}_{\mu_k,1}(s)\big)$ — the sampled MDP's claimed value of $\mu_k$ minus its value under the real dynamics. Then $\Delta_k-\tilde\Delta_k=\sum_s\rho(s)\big(V^{M^*}_{\mu^*,1}(s)-V^{M_k}_{\mu_k,1}(s)\big)$ applies the same optimal-value functional $g(M)=\sum_s\rho(s)V^M_{\mu^M,1}(s)$ to $M^*$ and $M_k$; by the lemma $\mathbb{E}[\Delta_k-\tilde\Delta_k\mid H_{t_k}]=0$, so
$$\mathbb{E}\Big[\textstyle\sum_k\Delta_k\Big]=\mathbb{E}\Big[\textstyle\sum_k\tilde\Delta_k\Big].$$
The unobservable optimal policy $\mu^*$ is eliminated; $\tilde\Delta_k$ has the form of an optimism term, obtained without constructing one.

**3. Bellman-error reduction.** With $\mathcal{T}^M_\mu V(s)=\overline R^M_\mu(s)+\sum_{s'}P^M_\mu(s'|s)V(s')$ and the DP recursion $V^M_{\mu,i}=\mathcal{T}^M_{\mu(\cdot,i)}V^M_{\mu,i+1}$, set $X_i=V^{M_k}_{\mu_k,i}-V^{M^*}_{\mu_k,i}$ and
$$d_{t_k+i}=X_{i+1}(s_{t_k+i+1})-\sum_{s'}P^*_{\mu_k(\cdot,i)}(s'|s_{t_k+i})X_{i+1}(s').$$
Then $\mathbb{E}[d_{t_k+i}\mid M^*,M_k,H_{t_k+i}]=0$, and telescoping gives
$$(V^{M_k}_{\mu_k,1}-V^{M^*}_{\mu_k,1})(s_{t_k+1})=\sum_{i=1}^\tau(\mathcal{T}^k_{\mu_k(\cdot,i)}-\mathcal{T}^*_{\mu_k(\cdot,i)})V^k_{\mu_k,i+1}(s_{t_k+i})-\sum_{i=1}^\tau d_{t_k+i}.$$
So $\mathbb{E}[\tilde\Delta_k\mid M^*,M_k]$ equals the expected sum of one-step Bellman errors *along the visited trajectory* — no optimal policy, no unvisited states.

**4. Confidence sets (analysis only).** Define $\mathcal{M}_k=\{M:\lVert\hat P^t_a(\cdot|s)-P^M_a(\cdot|s)\rVert_1\le\beta_k(s,a)\ \&\ |\hat R^t_a(s)-R^M_a(s)|\le\beta_k(s,a)\ \forall (s,a)\}$ with
$$\beta_k(s,a)=\sqrt{\tfrac{14\,S\log(2SAm\,t_k)}{\max\{1,N_{t_k}(s,a)\}}},$$
chosen (as in UCRL2 with $\delta=1/m$) so $\mathbb{P}(M^*\notin\mathcal{M}_k)\le 1/m$. Because $\mathcal{M}_k$ is $H_{t_k}$-measurable, the lemma gives $\mathbb{P}(M_k\notin\mathcal{M}_k\mid H_{t_k})=\mathbb{P}(M^*\notin\mathcal{M}_k\mid H_{t_k})$: the sample lies in the set exactly as often as the truth. A one-step Bellman error obeys
$$|(\mathcal{T}^k-\mathcal{T}^*)V|\le|r_k-r^*|+\lVert P_k-P^*\rVert_1\lVert V\rVert_\infty\le 2\beta_k+\tau\cdot2\beta_k$$
on the good event ($\lVert V\rVert_\infty\le\tau$), and it is capped by a universal constant times $\tau$ when $\beta_k>1$. Thus, hiding only numerical constants,
$$\mathbb{E}\Big[\textstyle\sum_k\tilde\Delta_k\Big]\le C\tau\,\mathbb{E}\sum_k\sum_i\min\{\beta_k(s_{t_k+i},a_{t_k+i}),1\}+2\tau.$$

**5. Summing the widths.** Split by $N_{t_k}\le\tau$ (at most $2\tau SA$ such steps) vs $N_{t_k}>\tau$ (where $N_t+1\le 2N_{t_k}$). For the latter,
$$\sum_t (N_t(s_t,a_t)+1)^{-1/2}\le\sum_{s,a}\int_0^{N_{T+1}(s,a)}x^{-1/2}dx=2\sum_{s,a}\sqrt{N_{T+1}(s,a)}\le 2\sqrt{SAT}$$
by Cauchy-Schwarz ($\sum_{s,a}N_{T+1}=T$), and the frozen-count swap adds only a factor $\sqrt2$. Multiplying by the $\sqrt{14S\log(SAT)}$ inside $\beta_k$ gives a large-count contribution $O(S\sqrt{AT\log(SAT)})$. The small-count contribution is $2\tau SA$, and the cap by $T$ handles that burn-in:
$$\min\Big\{C\tau\textstyle\sum_k\sum_i\min\{\beta_k,1\},T\Big\}\le C'\tau S\sqrt{AT\log(SAT)}.$$
Combining gives $\mathbb{E}[\text{Regret}]=O(\tau S\sqrt{AT\log(SAT)})$. $\;\blacksquare$

The extra $\sqrt S$ over the $\sqrt{SAT}$ lower bound sits inside $\beta_k$ — the cost of controlling an $L^1$ ball over $S$ successor states — matching UCRL2 ($D\!\leftrightarrow\!\tau$), but achieved with one MDP solve per episode and no optimization over a family of MDPs.

## Implementation

```python
import numpy as np

class DirichletNormalGammaBelief:
    """Conjugate posterior over an unknown finite-horizon MDP.
    Dirichlet over transition rows; normal-gamma over rewards."""
    def __init__(self, S, A, horizon, alpha0=None, mu0=1.0, lam0=1.0, a0=1.0, b0=1.0):
        self.S, self.A, self.H = S, A, horizon
        a0d = (1.0 / S) if alpha0 is None else alpha0
        self.dir = np.full((S, A, S), a0d)
        self.mu  = np.full((S, A), mu0)
        self.lam = np.full((S, A), lam0)
        self.a   = np.full((S, A), a0)
        self.b   = np.full((S, A), b0)

    def update(self, s, a, r, s_next):
        self.dir[s, a, s_next] += 1.0
        lam, mu = self.lam[s, a], self.mu[s, a]
        self.mu[s, a]  = (lam * mu + r) / (lam + 1.0)
        self.b[s, a]  += 0.5 * lam / (lam + 1.0) * (r - mu) ** 2
        self.a[s, a]  += 0.5
        self.lam[s, a] = lam + 1.0

    def sample_mdp(self):
        P = np.empty((self.S, self.A, self.S))
        R = np.empty((self.S, self.A))
        for s in range(self.S):
            for a in range(self.A):
                P[s, a] = np.random.dirichlet(self.dir[s, a])
                prec = np.random.gamma(self.a[s, a], 1.0 / self.b[s, a])
                R[s, a] = np.random.normal(self.mu[s, a],
                                           1.0 / np.sqrt(self.lam[s, a] * prec))
        return R, P

def finite_horizon_optimal_policy(R, P, S, A, horizon):
    V = np.zeros((horizon + 1, S))
    mu = np.zeros((horizon, S), dtype=int)
    for i in reversed(range(horizon)):
        Q = R + P @ V[i + 1]
        mu[i] = Q.argmax(axis=1)
        V[i]  = Q.max(axis=1)
    return mu

def psrl(env, S, A, horizon, n_episodes):
    belief = DirichletNormalGammaBelief(S, A, horizon)
    for k in range(n_episodes):
        R, P = belief.sample_mdp()                                 # sample one MDP
        mu   = finite_horizon_optimal_policy(R, P, S, A, horizon)  # plan for it
        s = env.reset()
        for i in range(horizon):                                   # fixed policy / episode
            a = mu[i][s]
            r, s_next = env.step(s, a)
            belief.update(s, a, r, s_next)
            s = s_next
    return belief
```
