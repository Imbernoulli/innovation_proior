# The occupancy-measure linear program for MDPs

## Problem

For a finite MDP $M=(S,A,P,r,\mu,\gamma)$ with $\gamma\in[0,1)$, find a policy $\pi$ maximizing the discounted return $J(\pi)=\mathbb{E}[\sum_{t\ge0}\gamma^t r(s_t,a_t)\mid s_0\sim\mu,\pi]$. Dynamic programming characterizes the optimum through the *nonlinear* Bellman optimality equation, and the return is *non-convex* viewed as a function of $\pi$. The goal is to recast planning as a **linear program**, exposing convex-duality structure and the full LP toolbox.

## Key idea

Re-encode a policy by its **discounted state–action occupancy measure** — its $\gamma$-discounted visitation frequencies
$$\lambda^\pi(s,a)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t\,\mathbb{P}[s_t=s,a_t=a\mid s_0\sim\mu,\pi].$$
In these coordinates the return is **linear**, $J(\pi)=\tfrac{1}{1-\gamma}\langle\lambda^\pi,r\rangle$, and the set of achievable occupancies is a **polytope** cut out by linear **Bellman-flow (continuity) constraints**. Planning becomes maximizing a linear objective over that polytope — a linear program — and an optimal policy is recovered by normalizing the optimal occupancy within each state.

## The two linear programs

**Primal (value-function LP).** $V^\star$ is the pointwise-least solution of the Bellman inequalities $V\ge TV$ (since $T$ is monotone and a $\gamma$-contraction, $V\ge TV\Rightarrow V\ge V^\star$). Strictly positive objective weights pin $V=V^\star$; the start-distribution weights below make $V^\star$ an optimizer and pin the optimal start value, with possible non-uniqueness on states irrelevant to $\mu$:
$$\min_{V}\ (1-\gamma)\sum_s\mu(s)V(s)\quad\text{s.t.}\quad V(s)\ge r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\ \ \forall s,a.$$
$|S|$ variables, $|S||A|$ constraints.

**Dual (occupancy-measure LP).** One nonnegative variable $\lambda(s,a)$ per primal constraint; $V$-stationarity in the Lagrangian yields the flow constraints:
$$\max_{\lambda\ge0}\ \sum_{s,a}\lambda(s,a)\,r(s,a)\quad\text{s.t.}\quad \sum_a\lambda(s,a)=(1-\gamma)\mu(s)+\gamma\sum_{s',a'}P(s\mid s',a')\lambda(s',a')\ \ \forall s.$$
$|S||A|$ variables, $|S|+|S||A|$ constraints. In matrix form, with $E$ the $(s,a)\!\to\!s$ copy matrix and $P$ the transition matrix: primal $\min(1-\gamma)\langle\mu,V\rangle$ s.t. $EV\ge r+\gamma PV$; dual $\max\langle\lambda,r\rangle$ s.t. $E^\top\lambda=(1-\gamma)\mu+\gamma P^\top\lambda,\ \lambda\ge0$.

## Occupancy-measure theorem and policy recovery

1. **Each policy is feasible.** $\lambda^\pi$ is a probability distribution ($\sum_{s,a}\lambda^\pi=(1-\gamma)\sum_t\gamma^t=1$) and satisfies $\lambda^\pi(s,a)=(1-\gamma)\mu(s)\pi(a\mid s)+\gamma\pi(a\mid s)\sum_{s',a'}P(s\mid s',a')\lambda^\pi(s',a')$; summing over $a$ gives the flow constraint.
2. **The flow polytope is exactly the set of occupancies.** Any feasible $\lambda\ge0$ has $\sum_{s,a}\lambda=1$ (summing the constraints uses $\sum_{s'}P=1$). Define $\pi_\lambda(a\mid s)=\lambda(s,a)/\sum_b\lambda(s,b)$ (arbitrary where the state marginal is $0$). Then $\lambda$ solves the same occupancy fixed-point system that $\lambda^{\pi_\lambda}$ uniquely solves: the induced operator $M_{\pi_\lambda}$ has column sums one, so $\|\gamma M_{\pi_\lambda}\|_1=\gamma<1$ and $I-\gamma M_{\pi_\lambda}$ is invertible. Hence $\lambda^{\pi_\lambda}=\lambda$. No phantom feasible points.
3. **Objective is the return.** $(1-\gamma)V^\pi(\mu)=\langle\lambda^\pi,r\rangle$.
4. **Recovery + optimality.** Solve the dual for $\lambda^\star$; set $\pi^\star(a\mid s)=\lambda^\star(s,a)/\sum_b\lambda^\star(s,b)$. By strong LP duality $\langle\lambda^\star,r\rangle=(1-\gamma)\mu^\top V^\star=(1-\gamma)J^\star$, and because $\lambda^\star$ is the occupancy of $\pi^\star$, this policy is optimal. Complementary slackness: $\lambda^\star(s,a)>0\Rightarrow$ the Bellman constraint is tight at $(s,a)\Rightarrow a$ is greedy for $V^\star$ — occupancy mass sits only on optimal actions. The occupancy $\lambda$ is the LP-dual of the value function $V$, and vice versa.
5. **Determinism for free.** A linear objective over the polytope attains its max at a vertex, and each vertex is an occupancy of a deterministic policy: expose the vertex with some reward vector, then the Bellman equation for that reward has a deterministic greedy optimizer whose occupancy maximizes the same linear objective; uniqueness of the exposed vertex makes the two occupancies equal. Although randomized policies are allowed to convexify the problem, an optimal deterministic policy always exists.

## $\gamma$ bookkeeping

$\gamma$ multiplies the recirculated inflow in the flow constraint, mirroring the discount on the future value in the primal. $(1-\gamma)$ plays one role from two sides: the normalizer making $\lambda$ a probability distribution (vs. un-normalized $\hat\lambda$ with total mass $1/(1-\gamma)$), equivalently the primal weight $c=(1-\gamma)\mu$ that scales the source term so the constraints sum to one. Dropping it gives the equivalent un-normalized LP (source $\mu$, objective the raw return); the recovered policy is unchanged because the recovery ratio is scale-invariant. The average-cost version replaces the discounted source-and-recirculation equation with statistical-equilibrium balance $\sum_a x(i,a)=\sum_{i',a'}P(i\mid i',a')x(i',a')$ and normalization $\sum_{i,a}x(i,a)=1$.

## Reference implementation

```python
import numpy as np
from scipy.optimize import linprog
# MDP: P[s,a,s'] = Pr(s'|s,a); r[s,a]; mu[s]; gamma in [0,1).

def solve_occupancy_dual_LP(P, r, mu, gamma):
    """max <lambda, r> s.t. sum_a lambda(s,a) = (1-gamma) mu(s) + gamma sum_{s',a'} P(s|s',a') lambda(s',a'),
    lambda >= 0.  (Bellman-flow / continuity constraints.)"""
    nS, nA = r.shape
    idx = lambda s, a: s * nA + a
    c = -r.reshape(-1)                                  # minimize -<lambda,r> == maximize <lambda,r>
    A_eq = np.zeros((nS, nS * nA)); b_eq = (1.0 - gamma) * mu
    for s in range(nS):
        for a in range(nA):
            A_eq[s, idx(s, a)] += 1.0                   # outflow placed on state s
        for sp in range(nS):
            for ap in range(nA):
                A_eq[s, idx(sp, ap)] -= gamma * P[sp, ap, s]   # discounted inflow from predecessors
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (nS * nA), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x.reshape(nS, nA)

def policy_from_occupancy(lam):
    """pi*(a|s) = lambda*(s,a) / sum_a lambda*(s,a)."""
    m = lam.sum(axis=1, keepdims=True)
    return np.where(m > 1e-12, lam / np.maximum(m, 1e-12), 1.0 / lam.shape[1])

def solve_primal_value_LP(P, r, mu, gamma):
    """min (1-gamma)<mu,V> s.t. V(s) >= r(s,a) + gamma sum_s' P(s'|s,a) V(s')."""
    nS, nA = r.shape
    A_ub = np.zeros((nS * nA, nS)); b_ub = np.zeros(nS * nA); row = 0
    for s in range(nS):
        for a in range(nA):
            A_ub[row, :] = gamma * P[s, a, :]; A_ub[row, s] -= 1.0; b_ub[row] = -r[s, a]; row += 1
    res = linprog((1.0 - gamma) * mu, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * nS, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x

def optimal_policy(P, r, mu, gamma):
    return policy_from_occupancy(solve_occupancy_dual_LP(P, r, mu, gamma))
```

Solving either LP gives the same optimal start value; solving the occupancy LP also gives the policy directly by the statewise normalization rule.
