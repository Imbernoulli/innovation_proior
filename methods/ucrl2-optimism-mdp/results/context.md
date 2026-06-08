# Context: provable exploration in unknown average-reward MDPs

## Research question

A learner is dropped into an unknown finite Markov decision process. At each step it sits in a state `s`, picks an action `a`, collects a random reward in `[0,1]` with mean `r̄(s,a)`, and is carried to a new state `s'` drawn from an unknown transition law `p(·|s,a)`. The MDP is undiscounted: success is the long-run average reward, and the benchmark is the optimal average reward `ρ*` of the best stationary policy.

The precise quantity to control is the **total regret accumulated while learning**:

```
Δ(M, A, s, T) = T·ρ*(M) − Σ_{t=1..T} r_t .
```

This is deliberately *not* the quality of the policy the learner eventually outputs. It charges the learner for every reward it misses *during* exploration, including being stranded in an unrewarding region of the MDP. The exploration/exploitation dilemma is therefore in-built: explore too little and you never find the good states; explore too much and you bleed regret wandering.

The goal a solution must hit: regret that grows like `√T` (so per-step regret `→ 0` at the fastest rate concentration allows), with *no* prior knowledge of the MDP beyond the counts of states `S` and actions `A`, and with the dependence on the MDP's difficulty captured by a parameter that describes **only its transition structure** — not a mixing time of an unknown optimal policy that the learner would have to be told in advance. Why it matters: average-reward learning with continued wandering (no resets, no episodes given by the environment) is the standard model for an agent acting indefinitely in one world, and before a `√T` bound existed the field only had sample-complexity guarantees that, converted to regret, stalled at `T^{2/3}`.

## Background

**The bandit special case and "optimism in the face of uncertainty."** A multi-armed bandit is an MDP with one state and `K` actions. Auer, Cesa-Bianchi & Fischer (2002) gave UCB1: keep, for each arm `i`, the empirical mean `x̄_i` and a confidence radius, and at each round play the arm maximizing the **upper confidence bound**

```
x̄_i + sqrt( 2 ln n / T_i )
```

where `T_i` is the number of pulls of arm `i` and `n` the round. The radius comes from a Chernoff–Hoeffding tail: `x̄_i` is within `sqrt(2 ln n / T_i)` of `μ_i` with overwhelming probability. Acting on the optimistic estimate forces exploration automatically: an under-pulled arm has a large radius, hence a large index, hence gets tried; once pulled enough its radius shrinks below the gap and it is dropped. The analysis shows each suboptimal arm `j` is pulled at most `8 ln n / Δ_j²` times (`Δ_j = μ* − μ_j`), giving `Σ_j (8/Δ_j) ln n` gap-dependent regret and `O(√(Kn))` gap-independent regret. The decisive conceptual content: *a confidence interval, used optimistically, is by itself a complete exploration rule* — no separate exploration schedule is needed. This is the principle that begs to be lifted out of the single-state world.

**What is hard about an MDP that a bandit doesn't have.** Two structural facts. First, a bandit's "model" is a vector of scalar means; an MDP's model also has a *transition law* per `(s,a)` — an unknown distribution over `S` next-states. Estimating a distribution from `n` samples concentrates in `L1` only at rate `sqrt(S/n)` (Weissman, Ordentlich, Seroussi, Verdú, Weinberger 2003): for an `m`-outcome distribution, `P(||p̂−p||_1 ≥ ε) ≤ (2^m − 2) exp(−nε²/2)`. So learning a transition vector costs an extra `√S` over learning a scalar mean. Second, in a bandit a mistake costs you one round; in an MDP a mistake can *transport you* into a region from which it takes many steps to return to where reward lives. The natural measure of this is the **diameter**: for a pair of states `s ≠ s'`, the expected hitting time under the best policy aimed at `s'`, maximized over pairs,

```
D(M) = max_{s ≠ s'} min_{π} E[ T(s' | M, π, s) ] .
```

`D` depends only on the transition structure (it is finite exactly when the MDP is *communicating*). An MDP with `S` states and `A` actions has `D ≥ log_A S − 3` (you need at least logarithmically many steps to address `S` states with `A` choices per step). The diameter is the price of a single planning error: a wrong action can cost up to `D` steps before recovery, so one expects it to multiply the exploration term in any regret bound.

**Average-reward MDP theory that already exists.** Undiscounted value iteration (Puterman 1994): iterate `u_{i+1}(s) = max_a { r(s,a) + Σ_{s'} p(s'|s,a) u_i(s') }`; for a communicating, aperiodic MDP the increments `u_{i+1} − u_i` converge to the constant optimal gain `ρ*·1`, and `u_i` (recentred) converges to the *bias* vector, related to gain and rewards through the Poisson equation `λ = r − ρ·1 + Pλ`. Crucially, the **span** of the bias of an optimal policy is at most `D`. Aperiodicity matters: a periodic optimal policy can make plain value iteration fail to converge (Puterman §8.5, §9.4), and one then needs an aperiodicity transformation.

**Diagnostic facts about the prior exploration algorithms.** E3 and R-max are *correct* — they learn near-optimal policies in polynomial time — but their guarantees are PAC: they reach an `ε`-optimal average reward with probability `1−δ` after time polynomial in `1/ε, 1/δ, S, A` and a mixing time. The polynomial dependence on `ε` is of order `1/ε³`; converting a `1/ε³` sample-complexity bound into a regret bound yields, at best, `T^{2/3}` regret — strictly worse than `√T`. Moreover both algorithms require, **as an input parameter**, the `ε`-return mixing time `T_mix^ε` of an optimal policy (the time for the optimal policy's running average reward to come within `ε` of `ρ*`); one can build MDPs of diameter `D` whose mixing time is `≈ D`, and if the learner guesses `T_mix` too small the bounds become exponential in the true mixing time. So two concrete shortcomings stand out before any new method: (i) the wrong exponent in `T`, and (ii) dependence on a hard-to-know mixing parameter rather than on transition structure alone.

## Baselines

**E3 (Kearns & Singh, 2002).** The first provably near-optimal polynomial-time RL algorithm for general MDPs. It partitions states into "known" (visited enough that the estimated dynamics are accurate) and "unknown." It maintains two model MDPs — one for *exploitation* (known states keep their estimated rewards) and one for *exploration* (known states are zeroed and an absorbing "explore" reward is placed on the unknown frontier) — and **explicitly decides at each known state** whether to exploit (if a near-optimal exploitation policy exists in the known-MDP) or to explore (run the exploration policy to reach an unknown state and gather a new sample). Performance is stated in terms of the `ε`-return mixing time `T`, which must be supplied. Gap left open: explicit explore-vs-exploit branching plus a required mixing-time input; PAC `1/ε³` sample complexity → `T^{2/3}` regret; large exponents in `S, A`.

**R-max (Brafman & Tennenholtz, 2002).** Simplifies E3's explicit branching into an *implicit* one. Maintain a full model; initialize every not-yet-known `(s,a)` **optimistically** to the maximal reward `R_max` (with a self-loop), and solve for the optimal policy of this fictitious model. Acting on it, the agent is guaranteed to *either* be near-optimal in the true MDP *or* visit an unknown `(s,a)` and learn — "implicit explore-or-exploit." This is the first formal justification of the optimistic-initialization heuristic and removes the explicit branch. Gap left open: optimism here is a coarse all-or-nothing flag on each `(s,a)` (known → use estimate, unknown → assume `R_max`), not a graded confidence region; still a PAC bound (`T^{2/3}` regret), still parameterized by a mixing time, large `S,A` exponents.

**MBIE (Strehl & Littman, 2005/2008).** Also computes an optimistic policy from confidence intervals around estimated parameters — conceptually the closest relative — but only for the *discounted* reward setting, where effectively a finite horizon matters and the notion of regret used measures rewards along the learner's own trajectory rather than against the optimal policy's trajectory. It supplies the `O(S)`-time inner routine for maximizing a linear function over an `L1`-ball of transition vectors. Gap left open: discounted only; no average-reward `√T` regret; the trajectory-based regret can be trivially zero (e.g. at discount `0`) for a policy that never moves to where reward is.

**Index policies for ergodic MDPs (Burnetas & Katehakis 1997; Tewari & Bartlett 2008, OLP).** Choose actions optimistically using confidence bounds **on the current state only**, achieving asymptotically logarithmic regret. Gaps: they assume *ergodic* MDPs (every policy visits every state), the bounds are gap-dependent and hide an additive term **exponential in the number of states**, and confidence is applied per-state rather than to the whole planning problem.

**UCRL (Auer & Ortner, 2007).** The direct predecessor: it already implements "optimism over a confidence set of MDPs," but its regret bound carries large exponents, depends on a *mixing time* rather than the diameter, and assumes the MDP is *ergodic*. Gap left open: bring the exponents down, replace the mixing time with the smaller, structure-only diameter, and weaken ergodicity to the natural finite-diameter (communicating) assumption.

## Evaluation settings

The natural yardstick is the regret `Δ(M, A, s, T) = T·ρ*(M) − Σ_{t≤T} r_t` of an online learner that wanders continuously (no environment-given resets or episodes) in a single unknown communicating MDP, measured against the optimal average reward `ρ*`. Instances are finite tabular MDPs specified by `(S, A, p(·|s,a), r̄(s,a))` with rewards in `[0,1]`; difficulty is indexed by `S`, `A`, the diameter `D`, and the horizon `T`. The companion metrics are: the **sample complexity** (number of steps until per-step regret is below `ε` with probability `≥ 1−δ`), and, in the gap-dependent regime, a logarithmic-in-`T` bound driven by the average-reward gap `g` between the best and second-best policy. A matching **information-theoretic lower bound** — the worst-case regret achievable by *any* algorithm on some MDP with given `S, A, D` — is the other side of the yardstick. A variant setting allows the MDP to change a bounded number `ℓ` of times, with regret measured against the per-segment optimal policies.

## Code framework

The available scaffold already has a tabular MDP simulator, empirical count-based estimators, the two concentration tails (Hoeffding for scalar means, the `L1` deviation bound for distributions), and average-reward undiscounted value iteration on a *fixed* MDP. The open slots are the confidence-set construction, the optimistic planner, and the wandering loop with whatever rule decides when to recompute the policy.

```python
import numpy as np

# --- reusable primitives ---------------------------------------------------

class TabularMDP:
    """Simulator interface to an unknown MDP: only S, A are exposed; step() samples
    the hidden reward and transition."""
    def __init__(self, S, A): self.S, self.A = S, A
    def reset(self): ...                  # returns an initial state
    def step(self, s, a): ...             # returns (reward in [0,1], next_state)

def hoeffding_radius(n, log_term):
    """Scalar-mean confidence half-width from a Hoeffding tail."""
    return np.sqrt(log_term / (2 * max(1, n)))

def l1_distribution_radius(n, S, log_term):
    """L1 confidence radius for an S-outcome empirical distribution
    (distribution-learning costs an extra sqrt(S) over a scalar mean)."""
    return np.sqrt(S * log_term / max(1, n))

def average_reward_value_iteration(r, P, tol):
    """Undiscounted VI on a FIXED communicating MDP: returns (gain, bias, greedy
    policy). Iterate u_{i+1}(s) = max_a [ r(s,a) + sum_s' P(s'|s,a) u_i(s') ];
    stop when span(u_{i+1}-u_i) < tol."""
    ...

# --- open algorithm slots ---------------------------------------------------

def build_confidence_set(rhat, phat, N, t, S, A, delta):
    """TODO: confidence region of plausible MDPs around the empirical
    (rhat, phat) given counts N at time t."""
    pass

def plan_optimistic(conf_set, tol):
    """TODO: among all MDPs in conf_set, find the one with the largest optimal
    average reward and a near-optimal policy for it."""
    pass

def learn(mdp: TabularMDP, T, delta):
    """TODO: wander for T steps; periodically rebuild the confidence set,
    re-plan an optimistic policy, and follow it. The rule that decides WHEN to
    re-plan is part of the method."""
    pass
```
