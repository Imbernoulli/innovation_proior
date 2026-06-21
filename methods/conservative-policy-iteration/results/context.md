# Context

## Research question

We want to solve reinforcement-learning problems large enough that we cannot enumerate the state space, which forces us to use *approximate* algorithms — function approximators for value functions, restricted parametric policy classes, sampling instead of sweeps. Working only from samples (a simulator we can query, or trajectories), we want an algorithm that improves a policy and whose cost is measured in an approximation parameter ε and the horizon 1/(1−γ) rather than in the size of the state space. The questions of interest are: what performance measure does the algorithm move, how do we check from samples that a candidate update moves it, and what level of performance is reached after how many updates.

## Background

The setting is a finite Markov decision process (S, D, A, R, P): start-state distribution D, reward R(s,a) ∈ [0, R], transitions P(s'; s, a). We work in the γ-discounted case, 0 ≤ γ < 1, and use *normalized* values so they stay bounded as γ → 1:

  V_π(s) = (1−γ) E[ Σ_t γ^t R(s_t,a_t) | π, s ] ∈ [0, R],
  Q_π(s,a) = (1−γ) R(s,a) + γ E_{s'}[V_π(s')],
  A_π(s,a) = Q_π(s,a) − V_π(s) ∈ [−R, R]   (the *advantage*; centered: Σ_a π(a|s) A_π(s,a) = 0).

The object that ties everything together is the **discounted future-state distribution** of π started from a distribution μ,

  d_{π,μ}(s) = (1−γ) Σ_t γ^t Pr(s_t = s ; π, μ),

a genuine probability distribution over states (the 1−γ normalizes it); as γ → 1 it approaches the stationary distribution. Two scalar objectives sit on top: η_D(π) = E_{s~D}[V_π(s)], the discounted return from the true start distribution, and more generally η_μ(π) = E_{s~μ}[V_π(s)] for any state distribution μ. A policy that is optimal simultaneously maximizes V_π(s) at every state, hence maximizes both.

A capability assumed available is a **restart distribution**: the agent can draw its next state from a fixed distribution μ of our choosing. This is weaker than a full generative model (we do not get to query an arbitrary (s,a) for its next-state distribution) but much stronger than "irreversible" single-trajectory experience. A restart lets us gather information about states the current policy would rarely reach on its own.

Two facts about *existing* methods anchor the setting:

- **Approximate dynamic programming.** Exact policy iteration computes Q_π and switches to the deterministic greedy policy π'(a|s)=1 iff a∈argmax_a Q_π(s,a); this improves the policy and converges. With *approximate* values the available guarantee is a max-norm statement: for a greedy π' built from an approximation with L∞ value error ε,

    V_{π'}(s) ≥ V_π(s) − 2γε/(1−γ)   for all s.

  The bound lower-bounds the new value, permitting a change of up to 2γε/(1−γ) in either direction. The same shape appears as the Williams & Baird (1993) Bellman-error bound, V_π ≥ V* − 2B_J/(1−γ). The penalty has the 1/(1−γ) horizon factor because a greedy swap replaces the policy *at every state simultaneously*, so the worst-case local error propagates over the whole horizon.

- **Policy gradient.** For a parametric class {π_θ}, the policy-gradient theorem (Sutton, McAllester, Singh & Mansour 2000) gives

    ∂η/∂θ = (1/(1−γ)) Σ_s d_{π,D}(s) Σ_a (∂π(s,a)/∂θ) Q_π(s,a),

  whose decisive feature is that *no term ∂d_π/∂θ appears* — an infinitesimal parameter step changes the visitation distribution only infinitesimally, so η improves along the gradient. The gradient direction is estimated from on-policy samples. In a length-n chain MDP where random actions tend to move away from the goal, the expected hitting time under undirected exploration is exponential in n (for n = 50, on the order of 10^15). In a two-state diagnostic example with Gibbs policies, increasing the self-loop probability at one state drives the stationary probability of the other state from 0.2 down to about 10^-7. In both, η_D weights each state's improvement by how often the *current* policy visits it, through d_{π,D}.

## Baselines

- **Exact / approximate value-function methods (policy iteration, value iteration; Bertsekas & Tsitsiklis 1996).** Core idea: iterate policy-evaluation then greedy improvement. Exact versions have convergence-rate bounds; with function approximation the available guarantee is the max-norm one above, V_{π'} ≥ V_π − 2γε/(1−γ).

- **Policy-gradient methods (Sutton et al. 2000; Williams' REINFORCE 1992).** Core idea: parameterize the policy and ascend ∂η/∂θ = (1/(1−γ))Σ_s d_π(s) Σ_a ∂π(s,a)/∂θ · Q_π(s,a); small θ steps keep the visitation distribution nearly fixed and η improves along the gradient. The gradient direction is estimated from on-policy samples drawn under d_{π,D}.

- **Sparse-sampling / generative-model planners (Kearns, Mansour & Ng 1999).** Core idea: build a depth-H lookahead tree by querying a generative model, getting a policy that is ε-near-optimal with cost (A·H/ε)^{O(H log(H/ε))} — independent of |S| but exponential in the horizon H = 1/(1−γ). It assumes a full generative model rather than restart access, and its lower bounds establish that |S|-independent guarantees are possible and that some exploration cost is unavoidable.

## Evaluation settings

The natural yardsticks at this time are MDPs constructed to stress exploration and approximation: the length-n chain in which random actions tend to increase distance to a goal (so undirected exploration takes time exponential in n); small two- and few-state MDPs with Gibbs/Boltzmann table-lookup policies π(a|s) ∝ exp(θ_{sa}), in which a stationary distribution can collapse toward zero at a state; and the generative-model / restart-model access pattern in which an algorithm is charged by the number of calls (trajectories) it makes. The performance quantities of interest are the discounted return η_D and the more uniform η_μ, and the algorithmic costs are the number of policy updates and the number of sampled trajectories, measured as a function of the approximation parameter ε and the horizon 1/(1−γ) rather than |S|.

## Code framework

The available scaffold has simulator/restart access and value and advantage estimation by rollouts. The policy-update step is left unresolved.

```python
import numpy as np

class RestartMDP:
    """Simulator with a restart distribution mu (weaker than a full generative model)."""
    def __init__(self, mu, gamma, n_actions, R):
        self.mu, self.gamma, self.n_actions, self.R = mu, gamma, n_actions, R
    def restart(self):       # draw a start state s ~ mu
        raise NotImplementedError
    def step(self, s, a):    # return (s_next, reward)
        raise NotImplementedError

def sample_future_state(mdp, policy):
    """Draw s ~ d_{policy, mu}: roll out from mu, stop at each step w.p. (1-gamma)."""
    s = mdp.restart()
    while np.random.rand() >= (1 - mdp.gamma):
        a = policy.sample(s)
        s, _ = mdp.step(s, a)
    return s

def estimate_advantage(mdp, policy, s, a):
    """Normalized Q_policy(s,a) by rollout estimation, minus V_policy(s); A = Q - V."""
    raise NotImplementedError  # TODO: rollout estimator for A_policy(s,a)

def fit_candidate(mdp, policy):
    """Regression: fit advantages, return a candidate policy pi'."""
    raise NotImplementedError  # TODO: regression -> candidate pi'

class Policy:
    def sample(self, s):     # TODO: action distribution at s
        raise NotImplementedError

def policy_update(policy, candidate, advantage_estimate):
    """Produce the next policy from the current policy and the candidate."""
    raise NotImplementedError  # TODO: the update rule

def learn(mdp, policy):
    while True:
        candidate = fit_candidate(mdp, policy)
        adv = ...                      # TODO: a stopping quantity to test the candidate
        if adv <= EPS:                 # TODO: halting criterion
            return policy
        policy = policy_update(policy, candidate, adv)
```
