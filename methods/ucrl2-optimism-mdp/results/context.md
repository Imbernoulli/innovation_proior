# Context: provable exploration in unknown average-reward MDPs

## Research question

A learner is dropped into an unknown finite Markov decision process. At each step it sits in a state `s`, picks an action `a`, collects a random reward in `[0,1]` with mean `r̄(s,a)`, and is carried to a new state `s'` drawn from an unknown transition law `p(·|s,a)`. The MDP is undiscounted: success is the long-run average reward, and the benchmark is the optimal average reward `ρ*` of the best stationary policy.

The precise quantity to control is the **total regret accumulated while learning**:

```
Δ(M, A, s, T) = T·ρ*(M) − Σ_{t=1..T} r_t .
```

This charges the learner for every reward it misses during exploration, including being stranded in an unrewarding region of the MDP. The exploration/exploitation dilemma is therefore in-built: explore too little and you never find the good states; explore too much and you bleed regret wandering.

The central question is: how should a learner choose actions, from data gathered so far, to keep this regret small over a long horizon `T`?

## Background

**The bandit special case and "optimism in the face of uncertainty."** A multi-armed bandit is an MDP with one state and `K` actions. Auer, Cesa-Bianchi & Fischer (2002) gave UCB1: keep, for each arm `i`, the empirical mean `x̄_i` and a confidence radius, and at each round play the arm maximizing the **upper confidence bound**

```
x̄_i + sqrt( 2 ln n / T_i )
```

where `T_i` is the number of pulls of arm `i` and `n` the round. The radius comes from a Chernoff–Hoeffding tail: `x̄_i` is within `sqrt(2 ln n / T_i)` of `μ_i` with overwhelming probability. Acting on the optimistic estimate forces exploration automatically: an under-pulled arm has a large radius, hence a large index, hence gets tried; once pulled enough its radius shrinks below the gap and it is dropped. The analysis shows each suboptimal arm `j` is pulled at most `8 ln n / Δ_j²` times (`Δ_j = μ* − μ_j`), giving `Σ_j (8/Δ_j) ln n` gap-dependent regret and `O(√(Kn))` gap-independent regret. In the single-state world, then, a confidence interval used optimistically doubles as the exploration rule: an under-pulled arm has a large radius and so is tried, with no separate exploration schedule written down.

**What is hard about an MDP that a bandit doesn't have.** Two structural facts. First, a bandit's "model" is a vector of scalar means; an MDP's model also has a *transition law* per `(s,a)` — an unknown distribution over `S` next-states. Estimating a distribution from `n` samples concentrates in `L1` only at rate `sqrt(S/n)` (Weissman, Ordentlich, Seroussi, Verdú, Weinberger 2003): for an `m`-outcome distribution, `P(||p̂−p||_1 ≥ ε) ≤ (2^m − 2) exp(−nε²/2)`. So learning a transition vector costs an extra `√S` over learning a scalar mean. Second, in a bandit a mistake costs you one round; in an MDP a mistake can *transport you* into a region from which it takes many steps to return to where reward lives. The natural measure of this is the **diameter**: for a pair of states `s ≠ s'`, the expected hitting time under the best policy aimed at `s'`, maximized over pairs,

```
D(M) = max_{s ≠ s'} min_{π} E[ T(s' | M, π, s) ] .
```

`D` depends only on the transition structure (it is finite exactly when the MDP is *communicating*). An MDP with `S` states and `A` actions has `D ≥ log_A S − 3` (you need at least logarithmically many steps to address `S` states with `A` choices per step). A wrong action can carry the learner into a region from which it takes up to `D` steps to return to where reward lives — so a single planning error can cost on the order of `D` steps, a price a bandit never pays.

**Average-reward MDP theory that already exists.** Undiscounted value iteration (Puterman 1994): iterate `u_{i+1}(s) = max_a { r(s,a) + Σ_{s'} p(s'|s,a) u_i(s') }`; for a communicating, aperiodic MDP the increments `u_{i+1} − u_i` converge to the constant optimal gain `ρ*·1`, and `u_i` (recentred) converges to the *bias* vector, related to gain and rewards through the Poisson equation `λ = r − ρ·1 + Pλ`. The relevant scalar attached to the bias is its **span** `max_s λ(s) − min_s λ(s)`. Aperiodicity matters: a periodic optimal policy can make plain value iteration fail to converge (Puterman §8.5, §9.4), and one then needs an aperiodicity transformation.

**Prior exploration algorithms.** E3 and R-max are *correct* — they learn near-optimal policies in polynomial time — but their guarantees are PAC: they reach an `ε`-optimal average reward with probability `1−δ` after time polynomial in `1/ε, 1/δ, S, A` and a mixing time. Both algorithms require, as an input parameter, the `ε`-return mixing time `T_mix^ε` of an optimal policy (the time for the optimal policy's running average reward to come within `ε` of `ρ*`); one can build MDPs of diameter `D` whose mixing time is `≈ D`.

## Baselines

**E3 (Kearns & Singh, 2002).** The first provably near-optimal polynomial-time RL algorithm for general MDPs. It partitions states into "known" (visited enough that the estimated dynamics are accurate) and "unknown." It maintains two model MDPs — one for *exploitation* (known states keep their estimated rewards) and one for *exploration* (known states are zeroed and an absorbing "explore" reward is placed on the unknown frontier) — and **explicitly decides at each known state** whether to exploit (if a near-optimal exploitation policy exists in the known-MDP) or to explore (run the exploration policy to reach an unknown state and gather a new sample). Performance is stated in terms of the `ε`-return mixing time `T`, which must be supplied.

**R-max (Brafman & Tennenholtz, 2002).** Simplifies E3's explicit branching into an *implicit* one. Maintain a full model; initialize every not-yet-known `(s,a)` **optimistically** to the maximal reward `R_max` (with a self-loop), and solve for the optimal policy of this fictitious model. Acting on it, the agent is guaranteed to *either* be near-optimal in the true MDP *or* visit an unknown `(s,a)` and learn — "implicit explore-or-exploit." This is the first formal justification of the optimistic-initialization heuristic and removes the explicit branch.

**MBIE (Strehl & Littman, 2005/2008).** Also computes an optimistic policy from confidence intervals around estimated parameters — conceptually the closest relative — but only for the *discounted* reward setting, where effectively a finite horizon matters and the notion of regret used measures rewards along the learner's own trajectory rather than against the optimal policy's trajectory. It supplies the `O(S)`-time inner routine for maximizing a linear function over an `L1`-ball of transition vectors.

**Index policies for ergodic MDPs (Burnetas & Katehakis 1997; Tewari & Bartlett 2008, OLP).** Choose actions optimistically using confidence bounds **on the current state only**, achieving asymptotically logarithmic regret. They assume *ergodic* MDPs (every policy visits every state), the bounds are gap-dependent and hide an additive term **exponential in the number of states**, and confidence is applied per-state rather than to the whole planning problem.

**UCRL (Auer & Ortner, 2007).** The direct predecessor: it already implements "optimism over a confidence set of MDPs," with a regret bound parameterized by a mixing time and under the assumption that the MDP is *ergodic* (every policy visits every state).

## Evaluation settings

The natural yardstick is the regret `Δ(M, A, s, T) = T·ρ*(M) − Σ_{t≤T} r_t` of an online learner that wanders continuously (no environment-given resets or episodes) in a single unknown communicating MDP, measured against the optimal average reward `ρ*`. Instances are finite tabular MDPs specified by `(S, A, p(·|s,a), r̄(s,a))` with rewards in `[0,1]`; difficulty is indexed by `S`, `A`, the diameter `D`, and the horizon `T`. The companion metrics are: the **sample complexity** (number of steps until per-step regret is below `ε` with probability `≥ 1−δ`), and, in the gap-dependent regime, a logarithmic-in-`T` bound driven by the average-reward gap `g` between the best and second-best policy. A matching **information-theoretic lower bound** — the worst-case regret achievable by *any* algorithm on some MDP with given `S, A, D` — is the other side of the yardstick. A variant setting allows the MDP to change a bounded number `ℓ` of times, with regret measured against the per-segment optimal policies.

## Code framework

The available scaffold already has a tabular MDP simulator, empirical count-based estimators, the two concentration tails (Hoeffding for scalar means, the `L1` deviation bound for distributions), and average-reward undiscounted value iteration on a *fixed* MDP. What remains open is the online learning procedure that wanders for `T` steps in the unknown MDP.

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

# --- open algorithm slot ----------------------------------------------------

def learn(mdp: TabularMDP, T, delta):
    """TODO: wander for T steps in the unknown MDP, choosing actions from the
    data gathered so far, and keep the total regret small."""
    pass
```
