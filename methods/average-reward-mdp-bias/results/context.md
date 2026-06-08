# Context: optimizing long-run reward in a process that never stops

## Research question

A controller faces a system that runs *forever*. Each period it observes a state `s` from a
finite set `S`, picks an action `a` from a finite set `A`, collects an immediate reward
`r(s,a)`, and the system moves to `s'` with probability `p(s'|s,a)`. There is no terminal
state and no episode boundary — a queue keeps receiving customers, a machine keeps being
maintained or replaced, a routing policy keeps cycling. The question is: **which stationary
policy is best for a task that has no end?**

The standard tool for infinite-horizon problems is the *discounted* total reward
`Σ_{t≥0} γ^t r_t` with `0 ≤ γ < 1`. But for a genuinely unending, recurrent task the discount
is uncomfortable. The factor `γ` was introduced for one technical reason — to keep an infinite
sum of bounded rewards finite — yet it silently encodes an effective horizon of roughly
`1/(1−γ)` periods and downweights the long-run behaviour that *is* the whole point of a
continuing task. Worse, it can rank policies *wrongly*: a policy whose rewards arrive sooner
can beat a policy that earns strictly more per step in the long run. So the goal a solution
must hit is: directly optimize the **long-run average reward per step**,

```
g = lim_{N→∞} (1/N) · E[ Σ_{t=0}^{N-1} r_t ],
```

characterize when an optimal policy exists, and give a way to compute it — without smuggling
in an arbitrary discount.

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
equation, value iteration, the contraction — is the prevailing wisdom and the toolbox into which
any new criterion must fit.

**The pain point that motivates a different criterion.** The discounted criterion can pick a
long-run-worse policy. Concretely: a controller sits in a state where it can enter one of two
cycles. One cycle pays reward `5` immediately and then every `5` steps (average `1` per step);
the other pays reward `10` four steps later and then every `5` steps (average `2` per step).
The long-run-better choice is unambiguous — the second cycle. But the discounted values are
`5/(1−γ^5)` and `10γ^4/(1−γ^5)`, so the first cycle has the larger discounted value whenever
`γ < 2^{-1/4} ≈ 0.8409`, purely because its smaller reward arrives sooner and discounting
overweights "sooner." One can push `γ` toward `1` to suppress the effect, but then
`V_γ(s) → ∞` for *every* policy (an unending stream of bounded rewards, undiscounted,
diverges), so the ranking is the difference of two quantities both blowing up — numerically and
conceptually unstable. This is a known, documented failure mode of discounting on
cyclic/continuing tasks: it "encourages short-term gains over long-term benefits," and finding a
`γ` close enough to `1` to avoid it, while still keeping values finite and learnable, is fragile.

**The state-independence of the average.** For a unichain policy the average reward `g` is the
*same constant for every starting state*: the recurrent class is visited forever, so its
time-average reward (a single number `d · r_π` under the stationary distribution `d`) dominates;
the finite reward accumulated over the transient prefix is divided by `N → ∞` and vanishes. So
the gain of a unichain policy is a scalar `g`, not a function of `s`. A scalar, by itself, cannot
say *which states are better to be in* — it carries no ranking information across states. That
gap is exactly what a second object will have to fill.

**The Laurent picture of the discount limit.** The discounted value, viewed as a function of
`γ` near `1`, is singular: `(I − γ P_π)^{-1}` has a pole at `γ = 1` because `I − P_π` is
singular (`P_π 1 = 1` makes `1` an eigenvalue, so `I − P_π` annihilates the constant vector).
The resolvent of a stochastic matrix is known to admit a **Laurent expansion** about `γ = 1`:
a `1/(1−γ)` pole term, a constant term, and higher vanishing terms. This is the analytic bridge
between the discounted world (which works) and the average-reward world (the goal): whatever the
average-reward objects are, they should be the *coefficients* of this expansion.

## Baselines

**Discounted MDP solution (Bellman; value iteration / policy iteration for `γ<1`).** Core idea:
fix `γ<1`, solve `V* = T_γ V*` by the contraction fixed point. Math: `T_γ` is a max-norm
`γ`-contraction, value iteration converges at rate `γ`, policy iteration alternates
`(I−γP_π)V_π = r_π` (a nonsingular linear solve) with greedy improvement. **Gap it leaves:**
needs a `γ < 1`; on a continuing task `γ` is an unprincipled knob that can invert the long-run
ranking of policies and whose "right" value is task-dependent and fragile.

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
many steps to a policy of maximal average reward. **Gap it leaves:** it finds a policy of maximal
*gain*, but does not, on its own, discriminate *among* equal-gain policies — when several policies
tie on the long-run average (e.g. several routes that all reach a goal), policy iteration can stop
at one that is not the best on the *finite* reward accrued along the way.

**Ergodic Markov-chain / stationary-distribution evaluation.** Core idea: to *evaluate* one
policy's average reward, compute the chain's stationary distribution `d` (`dP_π = d`) and read
off `g = d · r_π`. Math: linear-algebra solve for the left eigenvector of `P_π`. **Gap it
leaves:** gives the scalar `g` only — no per-state value, no improvement direction, and nothing
to break ties between policies of equal `g`.

**Schwartz-style average-adjusted ("R-learning") value (Schwartz, 1993).** Core idea: replace
the discounted return with the average-adjusted return `Σ_t (r_t − g)`, learning a relative value
and `g` separately. **Gap it leaves (as a baseline for the characterization being sought):** it is
a *learning* heuristic; it presumes, rather than derives, that the right object is a relative value
offset by `g`, and it does not by itself pin down the finer optimality ordering among gain-equal
policies.

## Evaluation settings

The natural yardsticks are small finite MDPs where the criteria can be checked by hand and the
distinctions made visible: (i) **two-action MDPs with a single recurrent state** where two
policies share the same average reward but differ in the finite reward accumulated reaching it —
the canonical setting for exposing that gain alone is too coarse; (ii) **grid-world / queueing /
maintenance-style continuing tasks** where the process cycles forever and "success is measured
continuously," matching operations-research practice; (iii) **periodic chains** (recurrent classes
of period `> 1`), which stress whether an iterative scheme converges at all. The relevant metrics
are the policy's gain `g`, its per-state relative values, and — for an iterative solver — the
**span seminorm** `sp(V) = max_s V(s) − min_s V(s)`, which measures the spread of the relative
values while ignoring any common additive drift.

## Code framework

The scaffold starts from a finite-MDP container, the stationary-distribution solve, and the
discounted backups and loops. It leaves empty the two slots a continuing-task solution has to
fill — the *evaluation* of a policy by something richer than a single discounted value, and the
*backup map* used to iterate toward the optimum.

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

# --- the slots the continuing-task criterion will fill ----------------------
def evaluate_policy(mdp, pi):
    """Evaluate a fixed policy for the continuing-task criterion.
    Must return whatever object(s) actually characterize a never-ending task —
    a single discounted value will not do here.
    # TODO: this is the object the method has to discover.
    """
    pass

def continuing_backup(mdp, V):
    """One backup of the iteration map appropriate to a task with no horizon.
    # TODO: the map the method will iterate, plus whatever normalization keeps
    #       the iterates from drifting off to infinity.
    """
    pass

def solve(mdp):
    """Driver: iterate evaluation + greedy improvement (policy iteration), or
    iterate the backup map to a fixed point (value iteration).
    # TODO: assemble from the two slots above once they are known.
    """
    pass
```
