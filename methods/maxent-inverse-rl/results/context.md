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
inference about underdetermined distributions: when the available knowledge fixes only
some expectation values `⟨f_r⟩ = Σ_i p_i f_r(x_i)` (plus normalization `Σ p_i = 1`) and
leaves the rest open, infinitely many distributions are consistent with it. Jaynes's
principle selects, among all distributions meeting the constraints, the one of
**maximum entropy** `H(p) = −Σ_i p_i ln p_i` — "maximally noncommittal with regard to
missing information," assigning no preference beyond what the constraints force. It is an
established device for committing to a single distribution under expectation constraints.

**Locally vs. globally normalized sequence models.** In sequence-labeling models built
from exp-of-a-linear-score scores (e.g. conditional random fields), it is known that
*locally* normalized models suffer **label bias**: probability mass is conserved locally
at each step, so states with fewer outgoing branches concentrate mass regardless of their
scores. Globally normalized models do not have this pathology. This is an established
contrast in the design of probabilistic sequence models.

**Ill-posedness of reward recovery.** The inverse problem is degenerate. The set of
reward functions under which a given demonstrated policy is optimal is large; in
particular the all-zero reward (and any constant reward) makes *every* policy optimal,
so it always "explains" the demonstrations. Many distinct reward weights, and many
distinct policies / mixtures of policies, reproduce the same demonstrated feature counts.

## Baselines

**Inverse reinforcement learning, Ng & Russell 2000.** Frames the problem precisely:
given an MDP and an observed optimal policy `π`, find rewards `R` making `π` optimal.
Their characterization (their Theorem 3) gives the full solution set: `π ≡ a_1` is
optimal iff `(P_{a_1} − P_a)(I − γ P_{a_1})^{-1} R ⪰ 0` for all actions `a`. To
pick one reward they add a **heuristic**: maximize the sum of margins
`Σ_s (Q*(s, a_1) − max_{a ≠ a_1} Q*(s, a))` between the demonstrated action and the
next-best action, optionally with an `ℓ_1` penalty, solved as a linear program.

**Apprenticeship learning via IRL, Abbeel & Ng 2004.** Sidesteps recovering the "true"
reward and instead aims only to match performance. Using feature expectations `μ(π)` and
`value = w·μ(π)`, it iterates an IRL step (a max-margin or projection step finding a
weight vector `w` on which the expert currently beats the candidate policies by a margin
`t`) with an RL step (solve the MDP under that `w`). On termination it returns a **mixture
of policies** whose feature expectations match the expert's, with a guarantee that the
mixture's value is within `ε` of the expert's under the unknown reward, and a bound of
`O(k/((1−γ)²ε²) · log(k/((1−γ)ε)))` iterations.

**Locally normalized / action-based probabilistic IRL.** A line of probabilistic models
assigns probability to each action locally, e.g. `P(action a | s) ∝ exp(Q*(s, a))`,
normalizing per state. Core idea: make demonstrated actions likely under a softmax of
action values.

**Maximum margin planning, Ratliff, Bagnell & Zinkevich 2006.** Casts reward recovery as
structured maximum-margin prediction: learn reward weights so the demonstrated path beats
alternatives by a margin under a structured-loss convex objective, needing only oracle
access to an MDP solver.

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
ascent). What is still missing is the bridge from demonstrations to a reward.

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

def recover_reward(mdp, trajectories, epochs, lr):
    """Fit reward weights θ to the demonstrations.
    # TODO: the bridge from demonstrations to a reward goes here.
    """
    theta = np.random.uniform(size=(mdp.feature_matrix.shape[1],))
    f_bar = empirical_feature_counts(mdp.feature_matrix, trajectories)
    for _ in range(epochs):
        pass  # TODO: gradient step
    return theta
```
