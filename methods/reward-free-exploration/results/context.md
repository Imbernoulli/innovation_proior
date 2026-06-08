# Context

## Research question

In episodic tabular reinforcement learning an agent interacts with an unknown
Markov decision process (MDP) over horizon `H`, with `S` states and `A` actions,
and wants an `ε`-optimal policy for a reward function `r`. Exploration is the
hard part: a state carrying high reward may sit behind a precise sequence of
actions, and a random walk can take time exponential in the depth to reach the
corner of the environment where reward accumulates (Li 2012). The provably
sample-efficient algorithms that solve this — E³, R-max, UCBVI, UCB-Q, EULER —
all interleave exploration with optimizing **one fixed** reward `r`: the bonus
that drives exploration is computed from the same value function that is being
maximized.

That coupling is a problem in the regime that actually occurs in practice. Reward
functions are rarely handed down once and for all; they are engineered by trial
and error to elicit a desired behavior — reward shaping, constrained-RL
formulations where one sweeps a Lagrange multiplier (Altman 1999; Achiam et al.
2017; Tessler et al. 2018; Miryoosefi et al. 2019). Each new candidate reward
sends a reward-coupled algorithm back to collect fresh trajectories from scratch.
If there are many rewards of interest, paying the full exploration cost once per
reward is enormously wasteful.

The precise goal is to **explore once, with no reward signal at all**, collecting
a dataset `D` from the MDP; and then, when an arbitrary reward `r` is revealed
afterward — possibly chosen adversarially, possibly many of them in sequence — to
compute an `ε`-optimal policy for `r` from `D` alone, with no further interaction.
A solution must answer: what does "good coverage" of the data have to guarantee so
that *every* reward function is simultaneously well served, and how many
exploration episodes does achieving that coverage cost?

## Background

**The episodic MDP and its value functions.** A non-stationary episodic
`MDP(S, A, H, P, r)` has time-indexed transitions `P_h(·|s,a)` and per-step
rewards `r_h(s,a) ∈ [0,1]`. A policy `π = {π_h: S → Δ_A}` has value
`V^π_h(s) = E_π[Σ_{h'≥h} r_{h'} | s_h = s]` and action value `Q^π_h(s,a)`,
linked by the Bellman equations `Q^π_h = r_h + P_h V^π_{h+1}` and
`V^π_h(s) = Q^π_h(s, π_h(s))`; the optimal policy `π*` attains
`V*_h(s) = sup_π V^π_h(s)`. Because rewards lie in `[0,1]`, values lie in `[0,H]`.
Write `P^π_h(s)` for the probability of occupying state `s` at step `h` under `π`,
and `P^π_h(s,a)` for the state-action occupancy.

**Optimism under uncertainty / PAC-MDP.** The dominant paradigm for provably
efficient exploration is optimism. E³ (Kearns & Singh 2002) and R-max (Brafman &
Tennenholtz 2002) maintain a *known set* `K` of state-action pairs visited enough
times to estimate their transitions, and build an optimistic surrogate MDP in
which unknown pairs are maximally rewarding; planning in the surrogate either
exploits what is known or is provably driven toward the unknown. The analysis
rests on an **escape-probability pigeonhole**: the agent can leave the known set
at most `O(mSA)` times, where `m` is the per-pair visitation threshold, so there
can be only `O(mSA/ε)` episodes in which the probability of escaping `K` exceeds
`ε` (Kakade 2003, Lemma 8.5.2). UCBVI (Azar et al. 2017) and UCB-Q (Jin et al.
2018) instead attach Hoeffding/Bernstein **bonuses** to the empirical Bellman
update and obtain near-optimal `Õ(√(SAH²·T))`-type regret for a fixed reward.

**Value-dependent (problem-dependent) regret — EULER.** Worst-case regret bounds
scale with the horizon range `H` of the value function. EULER (Zanette &
Brunskill 2019) sharpens this: by carrying computable upper/lower bounds on the
value function and applying an empirical-Bernstein bonus driven by the *variance*
of the next-state value, its regret scales with the actual optimal value rather
than the worst-case range. Concretely, over `K` episodes its cumulative regret
is `Õ(√(𝒢·SAHK) + S²AH⁴)`, where `𝒢` upper-bounds the sum of optimal-value
variances and, when rewards are bounded by a small quantity `G`, can be taken as
`G`-scaled rather than `H`-scaled. The leading term thus shrinks when the optimal
value `V*_1(s_1)` is small. This problem-dependence is the load-bearing property:
it means that *learning to reach a hard-to-reach state* — where the maximum
reaching probability is small — is correspondingly cheap.

**The simulation / value-difference lemma.** For two MDPs `M'`, `M''` sharing a
policy `π`, the value gap telescopes (Dann et al. 2017, Lemma E.15):
`V'_h(s) − V''_h(s) = E_{M'',π}[ Σ_{i≥h} (r'_i − r''_i) + (P'_i − P''_i)V'_{i+1} ]`.
When the two MDPs share the reward and differ only in transitions, the value gap
is exactly `(P̂ − P)` applied to the value function, weighted by the occupancy of
`π`. This reduces "how wrong is my plan?" to "how wrong are my transition
estimates, measured against where `π` actually goes?"

**Concentration for transition estimates.** Estimating `P_h(·|s,a)` from counts
gives, for a fixed bounded function `G: S → [0,H]`, that `(P̂_h − P_h)G(s,a)`
concentrates at rate `H/√(N_h(s,a))` by Hoeffding, sharpened to a
variance-dependent rate by Bernstein. A **self-bounding** structure recurs in
these analyses: the empirical estimate minimizes a squared Bellman error, so the
error random variable has variance controlled by its own mean, which Bernstein
turns into a fast `1/N` rate. Chen & Jiang (2019) use exactly such a Bernstein
self-bounding argument for batch RL, and observe that covering all deterministic
policies together with their Q-values needs only `(A/ε)^S` balls rather than the
`(1/ε)^{SA}` balls needed to cover all Q-values blindly.

**Reaching probability and the empirical fact that some states are unreachable.**
Unlike a bandit, where any arm can be pulled at will, an MDP can contain states
that no policy reaches with non-negligible probability: in a chain with a
low-probability branch, a target state may have maximum reaching probability
`10^{-6}` under the best policy, and others may be unreachable outright. This is a
geometric fact about MDPs: `max_π P^π_h(s)` can be arbitrarily small.
Any honest exploration guarantee must therefore be *relative to reachability* — we
cannot demand uniform accuracy at states we provably cannot reach.

## Baselines

**R-max run with no reward (PAC-MDP coverage).** One can repurpose a PAC-MDP
algorithm for reward-free exploration by zeroing out the reward inside the known
set and rewarding `1` on unknown pairs, so the agent is driven purely to expand
`K` (call this ZeroRMax). Optimism plus the escape-probability pigeonhole give a
coverage guarantee: after enough episodes, with high probability a uniformly
sampled episode has small escape probability, and the empirical model on `K` is
accurate. Pushing the value-difference lemma through three MDPs (the truth `M`,
the truncated `M_K` with self-loops on unknown states, and the empirical `M̂_K`)
yields suboptimality `H³ε_escape + Õ(H⁴√(S/m))`. Setting this to `ε` forces
`m = Ω(SH⁸/ε²)` and an exploration budget
`N = Ω(H^{11} S²A / (ε³ p) · log²)`. The gaps: the sample complexity scales as
`ε^{−3}` and polynomially (not logarithmically) in the failure probability `1/p`,
and the worst-case `H` powers are large. Reward-dependent exploration is simply
the wrong objective for producing coverage.

**Function-approximation exploration specialized to the tabular case.** Methods
that explore for the harder function-approximation problem (Du et al. 2019, which
plans to visit unexplored states with an iteratively refined model; Misra et al.
2019, which uses model-free dynamic programming to reach all latent states) can be
specialized to tabular MDPs and do guarantee coverage. Their gaps: suboptimal
sample complexity, and a requirement that every state be reachable with
significant probability — exactly the assumption that fails when some states have
tiny maximum reaching probability.

**Maximum-entropy exploration.** Hazan et al. (2018) find, with a Frank-Wolfe
scheme, a policy whose state-occupancy measure `d_π` approximately maximizes
entropy `(1/S)Σ_s log d_π(s)`. An exact maximizer has a coverage property similar
to what one wants: it spreads occupancy over the state space. The gaps: only an
approximate optimizer is guaranteed, and the analysis does not say how the
optimization error feeds into coverage. A short calculation shows the entropy
objective must be driven to error `O(1/S)` before the ratio
`d_{π̃}(s)/d_{π̂}(s)` can be controlled — at which point the sample complexity
scales with `S⁵`. And the result is not end-to-end: it stops at the exploratory
policy and never closes the loop to a planning guarantee for an arbitrary reward.

**Batch RL with a coverage/concentrability assumption.** Given an *a priori*
logged dataset with bounded concentrability — the ratio of any target occupancy to
the logging distribution is bounded — fitted-Q / approximate dynamic programming
and policy-gradient methods compute near-optimal policies (Munos & Szepesvári
2008; Antos et al. 2008; Chen & Jiang 2019; Agarwal et al. 2019). This is the
right *consumer* of good coverage and supplies the planning oracle. The gap it
leaves open is precisely the question above: it assumes a well-covering logging
policy exists and is given, and provides no recipe to produce one.

## Evaluation settings

The natural yardstick is the tabular episodic MDP with `S` states, `A` actions,
horizon `H`, rewards normalized to `[0,1]`, and accuracy `ε`, failure probability
`p`. Performance is the number of exploration episodes `K` (each a length-`H`
trajectory under Protocol: pick `π_k`, sample `s_1 ∼ P_1`, roll out `H` steps
observing transitions but no reward) needed so that, in a subsequent planning
phase with no further interaction, the returned policy satisfies
`E_{s_1∼P_1}[V*_1(s_1; r) − V^{π̂}_1(s_1; r)] ≤ ε` for any revealed reward `r` —
and simultaneously for an arbitrary, possibly adaptively chosen, sequence of
rewards. The relevant comparison point is the known sample complexity of
single-reward tabular RL, `Θ̃(SAH²/ε²)` (Dann & Brunskill 2015), against which the
reward-free cost is to be measured; the matching lower-bound machinery (packing
constructions, Fano's inequality, Wald's identity) is the standard tool for
showing optimality.

## Code framework

The scaffold contains an episodic-MDP simulator that implements the reward-free
protocol, an episodic regret-minimizing explorer used as a black box (a
PAC/regret oracle), empirical transition counting, and an approximate MDP solver
for a supplied model.

```python
import numpy as np

class EpisodicMDP:
    def __init__(self, P, H, p1):
        self.P, self.H, self.p1 = P, H, p1
        self.S, self.A = P[0].shape[0], P[0].shape[1]
    def rollout(self, policy, rng):
        s = rng.choice(self.S, p=self.p1)
        traj = []
        for h in range(self.H):
            a = rng.choice(self.A, p=policy[h][s])
            s_next = rng.choice(self.S, p=self.P[h][s, a])
            traj.append((h, s, a, s_next))
            s = s_next
        return traj

def value_iteration(P, r, H):
    # exact optimal policy for a supplied (P, r): the planning oracle
    ...

def regret_minimizing_explorer(env, reward, N0, rng):
    # episodic regret/PAC oracle: returns a set of N0 policies for the supplied reward
    ...

# ---- open slots -------------------------------------------------------------
def explore(env, N0, N, rng):
    # TODO: collect a dataset with no reward such that ANY reward can be
    # planned for afterward. What reward (if any) do we feed the explorer per
    # target, how do we assemble an exploratory policy set, and what data
    # distribution does it induce?
    pass

def plan(D, S, A, H, reward):
    # TODO: from D and a revealed reward, return a near-optimal policy.
    pass
```
