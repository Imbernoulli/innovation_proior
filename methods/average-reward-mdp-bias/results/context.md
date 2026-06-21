# Context: optimizing long-run reward in a process that never stops

## Research question

A controller faces a system that runs *forever*. Each period it observes a state `s` from a
finite set `S`, picks an action `a` from a finite set `A`, collects an immediate reward
`r(s,a)`, and the system moves to `s'` with probability `p(s'|s,a)`. There is no terminal
state and no episode boundary — a queue keeps receiving customers, a machine keeps being
maintained or replaced, a routing policy keeps cycling. The question is: **which stationary
policy is best for a task that has no end?**

The standard tool for infinite-horizon problems is the *discounted* total reward
`Σ_{t≥0} γ^t r_t` with `0 ≤ γ < 1`, where the factor `γ` keeps an infinite sum of bounded
rewards finite and encodes an effective horizon of roughly `1/(1−γ)` periods. For a
genuinely unending, recurrent task the quantity one wants to optimize directly is the
**long-run average reward per step**,

```
g = lim_{N→∞} (1/N) · E[ Σ_{t=0}^{N-1} r_t ],
```

so the task is to characterize when an optimal policy for this criterion exists and to give a
way to compute it.

## Background

**Finite Markov decision processes.** A policy `π` (stationary, deterministic, `π:S→A`)
induces a Markov chain on `S` with transition matrix `P_π`, where `P_π(s,s') = p(s'|s,π(s))`,
and a reward vector `r_π(s) = r(s,π(s))`. `P_π` is row-stochastic: `P_π 1 = 1`. The theory of
finite Markov chains classifies states into **recurrent** (revisited forever with probability 1)
and **transient** (eventually abandoned). The recurrent states split into one or more
**recurrent (ergodic) classes**; each class has a unique **stationary distribution** `d`
solving `d P_π = d`, `d 1 = 1`, and by the ergodic theorem the time-average of any function
along a trajectory converges to its expectation under `d`. An MDP is **unichain** if every
policy's chain has a single recurrent class (plus possibly transient states), **communicating**
if every pair of states is mutually reachable under some policy, and **multichain** otherwise.

**Discounted dynamic programming.** For `γ < 1` the value of a policy,
`V_π(s) = E[ Σ_{t≥0} γ^t r_t | s_0=s ]`, is the unique solution of the linear Bellman equation

```
V_π = r_π + γ P_π V_π     ⇔     (I − γ P_π) V_π = r_π,
```

and `I − γ P_π` is invertible because its eigenvalues are `1 − γ λ_i` with `|λ_i| ≤ 1` and
`γ < 1`, so none vanish; the **Neumann series** `(I − γ P_π)^{-1} = Σ_{k≥0} γ^k P_π^k`
converges. The Bellman *optimality* equation
`V*(s) = max_a [ r(s,a) + γ Σ_{s'} p(s'|s,a) V*(s') ]` has a unique solution `V*`, its
operator `T_γ` is a `γ`-contraction in the max-norm, **value iteration** `V ← T_γ V` converges
geometrically, and the greedy policy with respect to `V*` is optimal. This machinery — Bellman's
equation, value iteration, the contraction — is the prevailing toolbox into which any new
criterion is set.

**Discounting on cyclic tasks.** On a continuing task the relative ranking of two policies under
the discounted criterion depends on `γ`. Consider a controller in a state where it can enter one
of two cycles. One cycle pays reward `5` immediately and then every `5` steps (average `1` per
step); the other pays reward `10` four steps later and then every `5` steps (average `2` per
step). The discounted values are `5/(1−γ^5)` and `10γ^4/(1−γ^5)`, so the first cycle has the
larger discounted value whenever `γ < 2^{-1/4} ≈ 0.8409`, even though the second has the larger
long-run average. As `γ → 1`, `V_γ(s) → ∞` for every policy, since an undiscounted unending
stream of bounded rewards diverges.

**The state-independence of the average.** For a unichain policy the average reward `g` is the
*same constant for every starting state*: the recurrent class is visited forever, so its
time-average reward (a single number `d · r_π` under the stationary distribution `d`) dominates,
while the finite reward accumulated over the transient prefix is divided by `N → ∞` and vanishes.
So the gain of a unichain policy is a scalar `g`, not a function of `s`.

## Baselines

**Discounted MDP solution (Bellman; value iteration / policy iteration for `γ<1`).** Core idea:
fix `γ<1`, solve `V* = T_γ V*` by the contraction fixed point. Math: `T_γ` is a max-norm
`γ`-contraction, value iteration converges at rate `γ`, policy iteration alternates
`(I−γP_π)V_π = r_π` (a nonsingular linear solve) with greedy improvement.

**Howard's policy iteration for the average criterion (Howard, 1960).** Core idea: alternate
*value-determination* (solve a linear system for a per-step gain plus relative values) and
*policy improvement* (greedy on the relative values). Math (as reproduced in the survey
literature, e.g. Mahadevan 1996, citing Howard 1960): for a fixed policy `π`, solve for a scalar
`g` and a vector `V` the `|S|` equations

```
V(s) + g = r(s, π(s)) + Σ_{s'} p(s'|s,π(s)) V(s'),
```

pinning the extra degree of freedom by setting `V(reference) = 0`; then improve
`π'(s) = argmax_a [ r(s,a) + Σ_{s'} p(s'|s,a) V(s') ]`. Howard proved this converges in finitely
many steps to a policy of maximal average reward.

**Ergodic Markov-chain / stationary-distribution evaluation.** Core idea: to *evaluate* one
policy's average reward, compute the chain's stationary distribution `d` (`dP_π = d`) and read
off `g = d · r_π`. Math: linear-algebra solve for the left eigenvector of `P_π`.

**Schwartz-style average-adjusted ("R-learning") value (Schwartz, 1993).** Core idea: replace
the discounted return with the average-adjusted return `Σ_t (r_t − g)`, learning value estimates
and `g` separately, as a model-free learning rule for continuing tasks.

## Evaluation settings

The natural yardsticks are small finite MDPs where the criteria can be checked by hand and the
distinctions made visible: (i) **two-action MDPs with a single recurrent state** where two
policies share the same average reward but differ in the finite reward accumulated reaching it;
(ii) **grid-world / queueing / maintenance-style continuing tasks** where the process cycles
forever and success is measured continuously, matching operations-research practice; (iii)
**periodic chains** (recurrent classes of period `> 1`), which stress whether an iterative scheme
converges at all. The relevant metrics are the policy's gain `g`, its per-state relative values,
and — for an iterative solver — the **span seminorm** `sp(V) = max_s V(s) − min_s V(s)`, which
measures the spread of the relative values while ignoring any common additive drift.

## Code framework

The scaffold starts from a finite-MDP container, the stationary-distribution solve, and the
discounted backups and loops. It leaves empty the solver a continuing-task criterion has to fill.

```python
import numpy as np

class MDP:
    """Finite MDP: states 0..n-1, actions 0..m-1.
    P[a] is an (n x n) row-stochastic transition matrix; R[a] is length-n reward."""
    def __init__(self, P, R):
        self.P = P            # list/array of (n,n) stochastic matrices, one per action
        self.R = R            # list/array of length-n reward vectors, one per action
        self.n = P[0].shape[0]
        self.m = len(P)

# --- Markov-chain utility ---------------------------------------------------
def stationary_distribution(Ppi):
    """Left eigenvector d with d Ppi = d, d·1 = 1 (single recurrent class)."""
    n = Ppi.shape[0]
    A = np.vstack([Ppi.T - np.eye(n), np.ones(n)])
    b = np.concatenate([np.zeros(n), [1.0]])
    return np.linalg.lstsq(A, b, rcond=None)[0]

# --- discounted machinery ---------------------------------------------------
def discounted_backup(mdp, V, gamma):
    """One discounted Bellman optimality backup: (T_gamma V)(s)."""
    Q = np.stack([mdp.R[a] + gamma * mdp.P[a] @ V for a in range(mdp.m)], axis=0)
    return Q.max(axis=0)

# --- the slot the continuing-task criterion will fill -----------------------
def solve(mdp):
    """Compute a best stationary policy for the long-run-average objective.
    # TODO: the criterion and the machinery to solve it go here.
    """
    pass
```
