We are given a fully known finite Markov decision process — states $S$, actions $A$, transition probabilities $P(s'\mid s,a)$, one-step rewards $r(s,a)$, an initial distribution $\mu$, and a discount $\gamma\in[0,1)$ — and we want a stationary policy $\pi(a\mid s)$ that maximizes the discounted return $J(\pi)=\mathbb{E}\!\left[\sum_{t\ge0}\gamma^t r(s_t,a_t)\mid s_0\sim\mu,\pi\right]$. The textbook characterization is recursive: the optimal value obeys the Bellman optimality equation $V^*(s)=\max_a\!\big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\big]$. That equation is correct and has a unique fixed point because the Bellman operator is monotone and a $\gamma$-contraction in max norm, but its shape is hostile to mathematical programming. The statewise $\max_a$ is the obstruction: it packs a whole family of affine lookaheads into one nonlinear expression. The standard iterative remedies all inherit that nonlinearity in one form or another. Value iteration just applies the optimality operator repeatedly and slows to a crawl as $\gamma\to1$; policy iteration converges fast but its outer loop still rests on a nonlinear greedy action-selection step; and optimizing directly over the policy is worse still, because once $\pi$ is fixed the controlled chain has transition matrix $P_\pi(s,s')=\sum_a\pi(a\mid s)P(s'\mid s,a)$ and value $(I-\gamma P_\pi)^{-1}r_\pi$, so the return is a curved, global function of the policy parameters through that inverse. Relaxing the $\max$ into one inequality per action does linearize the constraints in $V$, but it leaves us with a value-only program that certifies a number and still owes us a separate, after-the-fact greedy step to extract the decision rule. What we actually need is a single finite optimization with a linear objective and linear constraints whose variables *are* the controlled behavior, so that solving it returns a policy directly and exactly, with the start distribution $\mu$ and the discount $\gamma$ kept explicit.

I propose the Occupancy-Measure LP: change variables from policies to state-action visitation flow, and the entire control problem becomes a linear program whose feasible region is exactly the set of achievable controlled behaviors. The starting point is the inequality relaxation of Bellman. Saying $V(s)$ dominates the best one-step lookahead is the same as saying it dominates every action's lookahead, $V(s)\ge r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')$ for all $(s,a)$, which is linear in $V$. This relaxation is tight, not loose: if $V\ge TV$ then monotonicity gives $V\ge TV\ge T^2V\ge\cdots$ and contraction gives $T^kV\to V^*$, so every feasible $V$ lies coordinatewise above $V^*$; since $V^*$ itself satisfies all the inequalities, it is the *least* feasible vector. Minimizing any strictly positive weighted sum of coordinates therefore recovers $V^*$. Choosing the weights to be the start-specific $(1-\gamma)\mu(s)$ gives the value primal
$$\min_V\ (1-\gamma)\sum_s\mu(s)V(s)\quad\text{s.t.}\quad V(s)\ge r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\ \ \forall (s,a),$$
whose optimum is the true $\mu$-start value (states irrelevant to $\mu$ may not be uniquely pinned, which is exactly the right notion of exactness — decisions where $\mu$ never sends mass do not matter).

This is still a program over values, so the policy is not yet a variable. The decisive move is to dualize. There is one Bellman inequality per state-action pair, so the dual has one nonnegative variable $\lambda(s,a)$ per pair. Writing each constraint as $r(s,a)+\gamma P_{s,a}V-V(s)\le0$ and collecting the coefficient of a fixed free $V(s)$ in the Lagrangian, boundedness in that free variable forces the equality
$$\sum_a\lambda(s,a)=(1-\gamma)\mu(s)+\gamma\sum_{s',a'}P(s\mid s',a')\,\lambda(s',a'),$$
and the rest of the Lagrangian is the linear reward $\sum_{s,a}\lambda(s,a)r(s,a)$. So the dual is to maximize $\sum_{s,a}\lambda(s,a)r(s,a)$ subject to that equality and $\lambda\ge0$. The constraint reads as a conservation law: the left side is all the mass present at state $s$ summed over the actions chosen there, and the right side is its two sources — new mass injected from the start distribution, $(1-\gamma)\mu(s)$, plus old mass arriving from predecessor pairs $(s',a')$, discounted by $\gamma$ and routed through $P(s\mid s',a')$. This is not a dual accident; it is the continuity equation for discounted state-action flow.

What makes this exact rather than a relaxation is that the dual variable has a closed identity. Define, for a fixed policy, the normalized discounted occupancy
$$\lambda^\pi(s,a)=(1-\gamma)\sum_{t\ge0}\gamma^t\Pr{}_\pi[s_t=s,a_t=a\mid s_0\sim\mu].$$
The prefactor $(1-\gamma)$ is exactly the normalizer, since $\sum_{s,a}\lambda^\pi=(1-\gamma)\sum_t\gamma^t=1$ — total mass is one probability distribution over $(s,a)$. At time zero the mass at $(s,a)$ is $(1-\gamma)\mu(s)\pi(a\mid s)$, and at later times mass arrives into $s$ from all predecessors and then picks action $a$ with probability $\pi(a\mid s)$, giving $\lambda^\pi(s,a)=\pi(a\mid s)\big[(1-\gamma)\mu(s)+\gamma\sum_{s',a'}P(s\mid s',a')\lambda^\pi(s',a')\big]$; summing over $a$ yields precisely the dual equality, so every policy produces a feasible nonnegative flow. The converse is the load-bearing direction, the one that guarantees no phantom feasible points. Take any $\lambda\ge0$ satisfying the flow equalities and sum them over $s$: the left side is $\sum_{s,a}\lambda(s,a)$ and the right side is $(1-\gamma)+\gamma\sum_{s',a'}\lambda(s',a')$ (because $\sum_sP(s\mid s',a')=1$), which since $\gamma<1$ forces $\sum_{s,a}\lambda(s,a)=1$. Now condition the joint mass to read off an action rule, $\pi_\lambda(a\mid s)=\lambda(s,a)/\sum_b\lambda(s,b)$ where the marginal is positive and anything where it is zero — the only sensible read-off, since if $\lambda$ is joint state-action mass the policy must be the conditional action distribution. Multiplying the flow equality by $\pi_\lambda(a\mid s)$ gives $\lambda(s,a)=\pi_\lambda(a\mid s)\big[(1-\gamma)\mu(s)+\gamma\sum_{s',a'}P(s\mid s',a')\lambda(s',a')\big]$ on positive-marginal states, and on zero-marginal states the bracketed source-plus-inflow term is itself zero so any action distribution still gives zero mass. That is exactly the occupancy fixed-point equation for $\pi_\lambda$, and for a fixed policy that affine, $\gamma$-discounted equation has a unique solution. Hence $\lambda=\lambda^{\pi_\lambda}$: the feasible region of the dual is *precisely* the occupancy polytope of stationary policies, with nothing extra introduced by the relaxation. The return is linear in this variable too, $\sum_{s,a}\lambda^\pi(s,a)r(s,a)=(1-\gamma)\,\mathbb{E}_\pi[\sum_{t\ge0}\gamma^tr(s_t,a_t)]$, so the policy nonlinearity was never approximated away — it was moved into a larger variable whose valid values are cut out by linear flow constraints, and the controlled dynamics that were hidden inside $(I-\gamma P_\pi)^{-1}$ are now written as mass conservation over pairs.

The two programs then close the loop by strong LP duality. The value primal has $V^*$ optimal with objective $(1-\gamma)\mu^\top V^*$; the occupancy dual attains the same value. Solving the dual for $\lambda^*$ and returning $\pi^*(a\mid s)=\lambda^*(s,a)/\sum_b\lambda^*(s,b)$ yields a policy whose occupancy is $\lambda^*$, so its scaled return equals the dual objective and therefore its return from $\mu$ equals the optimal dynamic-programming value from $\mu$ — exact agreement, not approximation. Complementary slackness supplies the certificate: wherever $\lambda^*(s,a)>0$, the Bellman inequality for $(s,a)$ is tight, $V^*(s)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')$, so positive flow lands only on actions greedy with respect to $V^*$, and the value vector and the flow are dual certificates for the same problem. Because the objective is linear over a polytope, an optimum can be taken at a vertex, which in the finite MDP corresponds to a deterministic optimal policy. The same structural idea predates the discounted form: Manne's average-cost inventory LP optimizes joint state-decision probabilities $x_{ij}$ under nonnegativity, normalization, and a statistical-equilibrium balance constraint; the discounted version here simply swaps steady-state equilibrium for injected-and-discounted flow, "state-action mass at $s$ = start-source mass + discounted predecessor inflow." The implementation builds both programs with matching signs: the value primal encodes each inequality as $\gamma P[s,a,:]V-V[s]\le-r[s,a]$, the occupancy dual feeds `linprog` the objective $-r$ flattened with equality $\sum_a\lambda(s,a)-\gamma\sum_{s',a'}P(s\mid s',a')\lambda(s',a')=(1-\gamma)\mu(s)$, and the policy is recovered by normalizing each positive state marginal (uniform on zero-mass states).

```python
"""Exact discounted MDP planning via value and occupancy linear programs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class MDP:
    """Finite discounted MDP with dense transition and reward arrays."""

    P: np.ndarray
    r: np.ndarray
    mu: np.ndarray
    gamma: float

    def __post_init__(self) -> None:
        if self.P.ndim != 3:
            raise ValueError("P must have shape (n_states, n_actions, n_states)")
        if self.r.shape != self.P.shape[:2]:
            raise ValueError("r must have shape (n_states, n_actions)")
        if self.mu.shape != (self.P.shape[0],):
            raise ValueError("mu must have shape (n_states,)")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if not np.allclose(self.P.sum(axis=2), 1.0):
            raise ValueError("each P[s, a, :] must sum to 1")
        if not np.isclose(self.mu.sum(), 1.0):
            raise ValueError("mu must sum to 1")


def solve_value_primal(mdp: MDP) -> np.ndarray:
    """Solve min (1-gamma)<mu,V> subject to Bellman inequalities."""

    n_states, n_actions = mdp.r.shape
    rows = n_states * n_actions
    A_ub = np.zeros((rows, n_states))
    b_ub = np.zeros(rows)

    row = 0
    for s in range(n_states):
        for a in range(n_actions):
            # r(s,a) + gamma P(s,a,:) V <= V(s)
            A_ub[row, :] = mdp.gamma * mdp.P[s, a, :]
            A_ub[row, s] -= 1.0
            b_ub[row] = -mdp.r[s, a]
            row += 1

    res = linprog(
        (1.0 - mdp.gamma) * mdp.mu,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * n_states,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def solve_occupancy_dual(mdp: MDP) -> np.ndarray:
    """Solve max <lambda,r> over discounted state-action flow constraints."""

    n_states, n_actions = mdp.r.shape
    n_vars = n_states * n_actions

    def idx(s: int, a: int) -> int:
        return s * n_actions + a

    A_eq = np.zeros((n_states, n_vars))
    b_eq = (1.0 - mdp.gamma) * mdp.mu

    for s in range(n_states):
        for a in range(n_actions):
            A_eq[s, idx(s, a)] += 1.0
        for prev_s in range(n_states):
            for prev_a in range(n_actions):
                A_eq[s, idx(prev_s, prev_a)] -= (
                    mdp.gamma * mdp.P[prev_s, prev_a, s]
                )

    res = linprog(
        -mdp.r.reshape(-1),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * n_vars,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x.reshape(n_states, n_actions)


def policy_from_occupancy(occupancy: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Recover pi(a|s) = lambda(s,a) / sum_b lambda(s,b)."""

    if occupancy.ndim != 2:
        raise ValueError("occupancy must have shape (n_states, n_actions)")
    n_actions = occupancy.shape[1]
    state_mass = occupancy.sum(axis=1, keepdims=True)
    default_policy = np.full_like(occupancy, 1.0 / n_actions, dtype=float)
    return np.divide(
        occupancy,
        np.maximum(state_mass, tol),
        out=default_policy,
        where=state_mass > tol,
    )


def solve_optimal_policy(mdp: MDP) -> tuple[np.ndarray, np.ndarray]:
    """Return an optimal policy and its normalized discounted occupancy."""

    occupancy = solve_occupancy_dual(mdp)
    return policy_from_occupancy(occupancy), occupancy


if __name__ == "__main__":
    # Tiny two-state check: the LP recovers the action with the better long-run flow.
    P = np.array(
        [
            [[0.8, 0.2], [0.1, 0.9]],
            [[0.0, 1.0], [0.6, 0.4]],
        ],
        dtype=float,
    )
    r = np.array([[1.0, 0.0], [0.0, 2.0]])
    mu = np.array([1.0, 0.0])
    mdp = MDP(P=P, r=r, mu=mu, gamma=0.9)
    V = solve_value_primal(mdp)
    pi, lam = solve_optimal_policy(mdp)
    print("V:", np.round(V, 6))
    print("occupancy:", np.round(lam, 6))
    print("policy:", np.round(pi, 6))
```
