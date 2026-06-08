# Context

## Research question

I am given demonstrations of a purposeful agent acting over time — paths through some
planning space: a driver's routes through a road network, an arm's trajectories, a
navigator's sequence of moves. I want to recover not just *what* the agent did but
*why* — the underlying objective that makes the behavior sensible — so that I can
predict, complete, and generalize that behavior to new situations rather than parrot it.

The natural formalism is a Markov Decision Process: states `s_i`, actions `a_i`,
transition dynamics, and a reward (a "cost" / "utility") the agent is taken to be
optimizing. Standard (forward) reinforcement learning takes the reward as given and
produces optimal behavior. Here I have the behavior and lack the reward. This is the
*inverse* problem: from observed near-optimal behavior, recover the reward function
being optimized.

Why recover the reward at all, instead of directly cloning the state→action map the
agent used? Because the reward is the most succinct, robust, and transferable
description of the task. A direct mapping memorizes the demonstrated states; the reward
generalizes to states never demonstrated, to changed dynamics, to new goals. The whole
enterprise of imitation/apprenticeship learning rests on this presupposition.

The thing a solution must achieve: take noisy, possibly imperfect demonstrations and a
known MDP structure, and produce a single, well-defined probabilistic model of behavior
whose reward explains the demonstrations — with a learning procedure that is principled
(not a heuristic tie-breaker), tractable on large planning spaces (hundreds of thousands
of states), and that comes with a performance guarantee tying the learned behavior to the
demonstrated behavior.

## Background

**The MDP / reward-optimization frame.** An agent's behavior is a trajectory
`ζ = (s_1, a_1, s_2, ...)` in a planning space modeled as an MDP. The agent is assumed
to optimize a function that linearly maps each state's features `f_{s_j} ∈ ℝ^k` to a
state reward via reward weights `θ`. The reward of a whole path is the sum of state
rewards along it, equivalently `θ` applied to the path **feature counts**
`f_ζ = Σ_{s_j ∈ ζ} f_{s_j}`: `reward(f_ζ) = θ·f_ζ`. Demonstrations give an empirical
expected feature count `f̄ = (1/m) Σ_i f_{ζ̃_i}` over the `m` demonstrated paths.

**Feature expectations are sufficient statistics for value.** For a policy `π`, define
its expected (discounted) feature counts `μ(π) = E[Σ_t γ^t φ(s_t) | π]`. Because the
reward is linear in features, the policy's expected value is `w·μ(π)`. So two behaviors
with the *same* feature expectations have the *same* value under *every* linear reward.
Matching feature expectations is therefore necessary and sufficient to match
performance for linear rewards.

**The maximum-entropy principle (Jaynes 1957).** A separate, older idea from statistical
inference: when only expectation constraints `⟨f_r⟩ = Σ_i p_i f_r(x_i)` (plus
normalization `Σ p_i = 1`) are known, the distribution is underdetermined. Jaynes's
resolution is to choose, among all distributions meeting the constraints, the one of
**maximum entropy** `H(p) = −Σ_i p_i ln p_i` — the unique distribution that is
"maximally noncommittal with regard to missing information," assigning no preference
beyond what the constraints force. With the sign convention
`J = H + Σ_r α_r(Σ_i p_i f_r(x_i) − ⟨f_r⟩) + β(Σ_i p_i − 1)`, stationarity gives the
exponential-family / Boltzmann form `p_i = Z(α)^{-1} exp(Σ_r α_r f_r(x_i))`, with
partition function `Z(α) = Σ_i exp(Σ_r α_r f_r(x_i))`. Two identities fall out:
`⟨f_r⟩ = ∂ ln Z / ∂α_r` and `Cov[f] = ∂² ln Z / ∂α∂αᵀ` (positive semidefinite). This is
the machinery for turning "match these expectations, commit to nothing else" into a
concrete distribution.

**Exponential-family / log-linear models elsewhere.** The same global-normalization,
exp-of-a-linear-score form appears in conditional random fields for sequence labeling,
where it is known that *locally* normalized sequence models suffer **label bias**:
probability mass is conserved locally at each step, so states with fewer outgoing
branches concentrate mass regardless of their scores. A globally normalized model does
not have this pathology — a fact about the design space worth keeping in view when a
sequence (path) distribution is needed.

**The diagnostic that motivates everything: recovering a reward is ill-posed.** The
inverse problem is degenerate. The set of reward functions under which a given
demonstrated policy is optimal is large; in particular the all-zero reward (and any
constant reward) makes *every* policy optimal, so it always "explains" the
demonstrations. Many distinct reward weights, and many distinct policies / mixtures of
policies, reproduce the same demonstrated feature counts. There is no information in
"make the demonstration optimal" or in "match feature counts" alone that selects one
answer. This ambiguity is the wall a solution has to get through.

## Baselines

**Inverse reinforcement learning, Ng & Russell 2000.** Frames the problem precisely:
given an MDP and an observed optimal policy `π`, find rewards `R` making `π` optimal.
Their characterization (their Theorem 3) gives the full solution set: `π ≡ a_1` is
optimal iff `(P_{a_1} − P_a)(I − γ P_{a_1})^{-1} R ⪰ 0` for all actions `a`. The
explicit value of this result is also its limitation: the set is huge and degenerate
(`R = 0` is always in it). To pick one reward they add a **heuristic**: maximize the sum
of margins `Σ_s (Q*(s, a_1) − max_{a ≠ a_1} Q*(s, a))` between the demonstrated action
and the next-best action, optionally with an `ℓ_1` penalty, solved as a linear program.
Gaps: the tie-break is an arbitrary margin objective with no probabilistic meaning; it
needs the *full* optimal policy as input; and it has no graceful story for demonstrations
that are themselves suboptimal or noisy.

**Apprenticeship learning via IRL, Abbeel & Ng 2004.** Sidesteps recovering the "true"
reward and instead aims only to match performance. Using feature expectations `μ(π)` and
`value = w·μ(π)`, it iterates an IRL step (a max-margin or projection step finding a
weight vector `w` on which the expert currently beats the candidate policies by a margin
`t`) with an RL step (solve the MDP under that `w`). On termination it returns a **mixture
of policies** whose feature expectations match the expert's, with a guarantee that the
mixture's value is within `ε` of the expert's under the unknown reward, and a bound of
`O(k/((1−γ)²ε²) · log(k/((1−γ)ε)))` iterations. Gaps: the output is a *mixture*, not a
single coherent stochastic policy; feature matching is satisfied by *many* different
policies and mixtures, so the procedure does not pin down a unique behavior and offers no
principled way to choose among the matching distributions; and the max-margin step is
again an arbitrary tie-break, fragile when the demonstrations are imperfect.

**Locally normalized / action-based probabilistic IRL.** A line of probabilistic models
assigns probability to each action locally, e.g. `P(action a | s) ∝ exp(Q*(s, a))`,
normalizing per state. Core idea: make demonstrated actions likely under a softmax of
action values. Gap: because normalization is *local* (per state / per branch), these
models exhibit **label bias** — paths through low-branching-factor regions accrue
probability for structural reasons unrelated to reward, so the highest-reward path need
not be the most probable path, and behaviors with equal reward can receive unequal
probability. They also tend to require solving the MDP to obtain `Q*` and inherit the
ambiguity without resolving it.

**Maximum margin planning, Ratliff, Bagnell & Zinkevich 2006.** Casts reward recovery as
structured maximum-margin prediction: learn reward weights so the demonstrated path beats
alternatives by a margin under a structured-loss convex objective, needing only oracle
access to an MDP solver. Gap: when no single reward makes the demonstrated behavior both
optimal and clearly better than alternatives — which happens whenever the demonstrator is
imperfect or the planner captures only part of the relevant state space — the
margin-based objective has nothing clean to latch onto.

## Evaluation settings

The natural proving ground is real sequential decision behavior modeled as a known-structure
MDP. The largest such setting is **driver route modeling on a real road network**: a
deterministic MDP over the road graph surrounding a city (on the order of 300,000 states /
road segments and 900,000 actions / intersection transitions), with an absorbing
destination state. Demonstrations are **collected GPS traces** of taxi drivers' daily
driving — raw position data fit to the road network with a particle filter and segmented
into distinct trips, then split into training and held-out test trips. Road-segment
**features** describe each state: road type (interstate / highway / local), speed category,
number of lanes, and transition type at intersections (straight, left, right, hard left,
hard right) — discretized into a fixed bank of counts per segment.

Smaller controlled settings exist as standard yardsticks for IRL: discrete **gridworlds**
(e.g. an N×N grid with sparse/region rewards, noisy compass-direction actions with some
probability of slipping) where an expert's optimal policy under a hidden reward is sampled
for trajectories, and continuous control tasks such as **mountain-car**. Natural **metrics**
for the route setting: fraction of a predicted most-likely path's distance that matches the
held-out actual path; the percentage of test examples whose predicted path matches at least
90% of the true path's distance; and the average log-probability assigned to held-out
paths. For destination inference, posterior prediction accuracy as a function of the
fraction of the path observed.

## Code framework

The primitives that already exist: an MDP container (states, actions, transition
probabilities, planning horizon), a feature map over states, expert demonstrations as
state sequences, an empirical feature-count routine, and a numerical optimizer (gradient
ascent). The missing bridge from demonstrations to a reward is a behavior model and an
efficient expected-count routine.

```python
import numpy as np

class MDP:
    """Planning space: states, actions, transition dynamics, features."""
    def __init__(self, n_states, n_actions, transition, feature_matrix, horizon):
        self.n_states = n_states          # |S|
        self.n_actions = n_actions        # |A|
        self.transition = transition      # P(s'|s,a), shape (S, A, S)
        self.feature_matrix = feature_matrix  # f_{s}, shape (S, k)
        self.horizon = horizon            # fixed planning horizon for path sums

def empirical_feature_counts(feature_matrix, trajectories):
    """f̄ = average feature counts over demonstrated paths."""
    fe = np.zeros(feature_matrix.shape[1])
    for traj in trajectories:
        for s in traj:
            fe += feature_matrix[s]
    return fe / len(trajectories)

# --- empty slots ---------------------------------------------------------------

def behavior_distribution(mdp, theta):
    """The distribution over behavior that the recovered weights θ induce,
    used to predict the agent and to compute expected feature counts.
    # TODO: what object resolves the reward ambiguity, and what form does it take?
    """
    pass

def expected_feature_counts(mdp, theta):
    """Expected feature counts under the current model — the quantity that must
    equal f̄ at the solution; the hard part is computing it without enumerating
    exponentially many paths.
    # TODO: an efficient computation over the planning horizon, with the
    #       remaining-horizon index handled explicitly.
    """
    pass

def recover_reward(mdp, trajectories, epochs, lr):
    """Fit reward weights θ to the demonstrations.
    # TODO: an objective whose optimum makes the model match the demonstrations,
    #       with a usable gradient driving gradient ascent.
    """
    theta = np.random.uniform(size=(mdp.feature_matrix.shape[1],))
    f_bar = empirical_feature_counts(mdp.feature_matrix, trajectories)
    for _ in range(epochs):
        pass  # TODO: gradient step from f_bar and the model's expected counts
    return theta
```
