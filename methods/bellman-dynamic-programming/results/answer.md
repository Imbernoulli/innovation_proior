# Dynamic Programming and the Principle of Optimality

## Core Statement

The method solves multistage decision problems by replacing optimization over complete future decision sequences with recursive optimization over the current decision and a value function on states.

The principle of optimality is the key step:

> After an initial decision has led to a new state, the remaining decisions in an optimal policy must be optimal for the subproblem starting from that new state.

If the continuation were not optimal, replacing it with a better continuation would improve the original policy.

## Value Equations

For a deterministic finite-horizon process where action `k` sends state `p` to `T_k(p)`:

$$
f_N(p)=\max_k f_{N-1}(T_k(p)), \qquad N=2,3,\ldots
$$

For a stochastic transition kernel `G_k(p,dz)`:

$$
f_N(p)=\max_k \int f_{N-1}(z)\,G_k(p,dz), \qquad N=2,3,\ldots
$$

With immediate reward and discount factor `alpha`:

$$
V(p)=\max_k\left\{r(p,k)+\alpha\int V(z)\,P(dz\mid p,k)\right\}.
$$

For continuous deterministic control,

$$
f_T=\max_v\{F(c,v)+G(c,v)f_c\}.
$$

Interior maximizers satisfy `F_v+G_v f_c=0`; constraints stay inside the feasible maximization.

## Algorithms

Finite horizon: start from terminal values and recurse backward, choosing an optimal current action at each state.

Discounted infinite horizon: define the Bellman operator

$$
(TV)(p)=\max_k\left\{r(p,k)+\alpha\int V(z)\,P(dz\mid p,k)\right\}.
$$

For bounded rewards on a finite state-action model, if `0 <= alpha < 1`, then

$$
\|TW-TV\|_\infty\le \alpha\|W-V\|_\infty,
$$

so `T` has a unique fixed point on bounded value functions and value iteration converges. Policy iteration alternates policy evaluation with greedy improvement and produces monotone improvement under the same discounted finite-state assumptions.

## Artifact

No canonical historical reference implementation was retrieved; Bellman's artifact is mathematical. The local executable artifact in `code/bellman_dynamic_programming.py` implements the retrieved equations:

- deterministic finite-horizon backward induction
- stochastic finite-horizon backward induction
- discounted value iteration
- discounted policy iteration

The mathematical artifact is summarized in `refs/final_artifact/bellman-value-equations.md`.

## Limitation

The method removes the exponential blow-up over full histories or action sequences when the current state is a sufficient summary. It does not remove the cost of representing the value function over a high-dimensional state space; that is the curse of dimensionality.
