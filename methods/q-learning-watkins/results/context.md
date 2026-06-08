# Context: learning to act optimally from delayed reward, without a model

## Research question

An agent moves through a world, at each step observing a state, choosing an action, and receiving a numerical reward. It wants to act so as to maximise its total reward over the long run — where rewards arriving later matter less, discounted by a factor γ per step. The catch that makes this hard is *delay*: the action that earns a reward may have been taken many steps earlier, so the agent has to solve a credit-assignment problem across time. A dog learns to obey commands for food; a lion learns the exact moment to break cover and charge; these are learned from the consequences of acting, not from a textbook of the world's dynamics.

The precise problem is this. The world is a controlled Markov process: from state x under action a the agent receives a reward with mean ρ(x,a) and moves to state y with probability P_xy[a]. The agent wants an optimal policy — a rule for choosing actions that maximises, from every state, the expected discounted return E[Σ_{k≥0} γ^k r_{t+k}]. Dynamic programming already tells us how to compute such a policy *if ρ and P are known*. The open problem is to learn an optimal policy from raw experience — a stream of (state, action, reward) triples — when ρ and P are **not** known, and ideally without ever estimating them. A solution would have to (i) work purely from sampled transitions, (ii) handle stochastic rewards and transitions, (iii) be cheap enough to run online, step by step, and (iv) come with a guarantee that it actually finds the optimal policy and does not merely wander.

## Background

**Markov decision processes and the value function.** For a fixed policy f, define the value of a state as the expected discounted return obtained by starting at x and following f forever:
V_f(x) = E[ r_0 + γ r_1 + γ² r_2 + ··· ]. Because discounting is exponential, V_f satisfies a one-step consistency equation,
V_f(x) = ρ(x,f) + γ Σ_y P_xy[f] V_f(y),
which in a finite world is a set of |S| linear equations. So evaluating a *given* policy is mechanical if the model is known.

**Bellman's dynamic programming and optimality.** Bellman & Dreyfus (1962) (and Ross 1983, Howard 1960) show there is an optimal stationary policy f* whose value V* is as large as achievable from every state simultaneously, and that V* obeys the Bellman optimality equation
V*(x) = max_a { ρ(x,a) + γ Σ_y P_xy[a] V*(y) }.
Two theorems organise everything. Define the *action-value* of a under policy f as the return from doing a once and then following f: Q_f(x,a) = ρ(x,a) + γ Σ_y P_xy[a] V_f(y), so Q_f(x,f(x)) = V_f(x). The **policy-improvement theorem**: if a policy g satisfies Q_f(x,g) ≥ V_f(x) for all x, then g is uniformly at least as good as f, V_g ≥ V_f everywhere. The **optimality theorem**: if a policy can no longer be improved this way — if V*(x) = max_a Q*(x,a) and f*(x) attains the max — then V* and Q* are the unique optimal value and action-value functions. These give two classical algorithms, both requiring the model: *policy iteration* (evaluate f, then set f to be greedy w.r.t. Q_f, repeat) and *value iteration*, which sweeps
V^n(x) := max_a { ρ(x,a) + γ Σ_y P_xy[a] V^{n-1}(y) }
and converges, V^n → V*, because the update is a contraction: with M = max_x |U(x) − V*(x)|, one backup at any single state y gives |U'(y) − V*(y)| ≤ γ M. The factor γ < 1 shrinks the worst-case error every sweep. Crucially, the sweeps need not be synchronous — value iteration still converges if states are updated in any order, provided each is updated often enough.

**Why the model is the obstacle.** A value function needs only |S| numbers; a stochastic policy at most |S||A|. The transition model needs as many as |S|²|A| numbers, and (Howard 1960) representing it is what limits the classical approach in practice. There is a subtler problem too: to build a transition model the agent must decide which features of the world to represent and which to ignore — but it cannot know which features matter for the *optimal policy* without having already found that policy. A model that is too coarse yields a policy that is optimal for the model yet wrong for the world; a model that is too detailed wastes resources. Model-based learning replaces the world with an internal model and does "mental experiments" on it, and that substitution is exactly where it can go wrong.

**Temporal-difference prediction.** Sutton (1988) introduced temporal-difference (TD) methods for *learning to predict* the value of a fixed policy from experience, with no model. The agent keeps an approximation U to V_f and, instead of waiting for the full return, bootstraps: it forms a *prediction difference* e_t = r_t + γ U(x_{t+1}) − U(x_t) and nudges U(x_t) toward r_t + γ U(x_{t+1}). The general TD(λ) target blends n-step *corrected truncated returns*
r_t^(n) = r_t + γ r_{t+1} + ··· + γ^{n-1} r_{t+n-1} + γ^n U_{t+n}(x_{t+n})
with weights proportional to λ^n; λ=0 gives the one-step target r_t + γU(x_{t+1}), λ=1 the full Monte-Carlo return. The decisive property is *error reduction*: the expected corrected return is closer to V_f than U is, max_x |E[r_t^(n)] − V_f(x)| ≤ γ^n K where K = max_x|U(x) − V_f(x)|. So bootstrapping off a current estimate provably contracts the error of a *prediction* toward the true value of the policy being followed. Earlier reinforcement-learning systems — Samuel's checker player, Holland's bucket brigade, Barto–Sutton–Anderson's (1983) adaptive heuristic critic, Witten (1977), Sutton (1984) — used related ideas: an estimated value function feeding a separately represented policy (action strengths or probabilities) that is nudged up for actions that turn out better than expected.

**Stochastic approximation.** Robbins & Monro (1951) showed how to find the root θ of an equation M(x) = α when M can only be sampled noisily: take observations Y(x_n) and step x_{n+1} = x_n + a_n(α − Y(x_n)). It converges to θ with probability one provided the step sizes satisfy Σ_n a_n = ∞ (so the iterate can travel any distance and never stops adjusting) and Σ_n a_n² < ∞ (so the accumulated sampling noise has finite variance and averages out). This is the template for turning a noisy sampled target into a consistent estimate by averaging with a decreasing step size.

## Baselines

- **Value iteration / policy iteration (Bellman, Howard 1960, Ross 1983).** Compute V* or Q* by sweeping the Bellman optimality backup, or by alternating exact policy evaluation with greedy improvement. Provably correct and the gold standard for the *known-model* problem. Gap: both need ρ and P explicitly; policy iteration re-solves |S| linear equations every round; neither learns from sampled experience.

- **Certainty-equivalence / model-estimating control (Sato, Abe & Takeda 1988).** Estimate ρ and P from observed transitions while running DP on the current estimates, acting as if the estimated model were exact. Gap: assuming certainty equivalence "costs dearly in the early stages of learning" (Barto & Singh 1990) when the model is still poor; and it pays the full price of representing the transition model — the very object that is large and hard to choose.

- **TD prediction, TD(λ) (Sutton 1984, 1988).** Learn the value of a *fixed* policy from experience by bootstrapping temporally successive predictions; lower variance and shorter delay than averaging Monte-Carlo returns. Gap: it is *prediction*, not control. It evaluates the policy you follow; it does not by itself say which action is optimal, and a state's value is only meaningful relative to a policy. Used inside a learning controller it requires a *separate* policy representation.

- **Learning automata / policy-with-value-function controllers (Witten 1977; Barto, Sutton & Anderson 1983; Wheeler & Narendra 1986; Sutton 1984; Anderson 1987).** Represent the policy explicitly as action strengths or probabilities and adjust it using an estimated value function (e.g. nudge probabilities toward actions whose return exceeded the state's value). The adaptive heuristic critic learns a policy and a value function together. Gap: two concurrent adaptive processes — value estimation and policy improvement — that may interact during learning and prevent convergence; no convergence guarantee to the optimal policy was available for these schemes, and because action-values are not represented, the proof techniques for value estimation cannot be applied. On the (non-Markov, coarsely discretised) pole-balancer, the pure policy-learning variant gave disappointing results.

## Evaluation settings

The natural yardstick is a finite, discrete controlled Markov process (finite state and action sets), with bounded rewards and a discount factor 0 < γ < 1, against which a learned policy's value is compared to the dynamic-programming optimum V* / Q*. Tabular (look-up-table) representation of the learned values. Standard small testbeds of the period for sequential learning under delayed reward: gridworld navigation tasks with goal states; the cart–pole / pole-balancing control task (Barto–Sutton–Anderson formulation); blackjack-style episodic tasks with a terminal reward (Widrow et al. 1972). The relevant quantities to track are whether the learned action-values approach Q*, whether the implied greedy policy approaches an optimal policy, and how the learning rate schedule and the requirement that every state–action pair be tried repeatedly affect convergence.

## Code framework

A tabular learner can already keep a store of numbers indexed by state–action pairs, query an environment for a sampled next state and reward, use a discount γ, use a learning-rate schedule, and choose exploratory actions. The agent runs a loop over experienced transitions and, on each one, revises its stored numbers. The unresolved design choice is the target to revise toward and the stored quantity that makes the optimal policy readable off directly.

```python
import numpy as np
from collections import defaultdict

# A discrete environment yields (next_state, reward) for (state, action).
# A discount factor, decreasing learning-rate schedule, and exploratory
# behaviour rule are available. Values are held in a look-up table.

class Estimates:
    def __init__(self, n_actions):
        # table of numbers indexed by state (and, to be decided, by action)
        self.table = defaultdict(lambda: np.zeros(n_actions))

    def value_of(self, state):
        # TODO: how to summarise the stored numbers at a state into the
        # quantity that drives both the target and the choice of action
        pass

def behaviour_action(estimates, state, n_actions, epsilon):
    # an exploratory rule over actions; need NOT be the policy whose value we want
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    # TODO: greedy choice w.r.t. the stored numbers
    pass

def learn(env, n_actions, gamma, alpha_schedule, epsilon, n_steps):
    est = Estimates(n_actions)
    state = env.reset()
    for t in range(n_steps):
        action = behaviour_action(est, state, n_actions, epsilon)
        next_state, reward, done = env.step(action)
        alpha = alpha_schedule(state, action)
        # TODO: the update rule --- the target to move the stored estimate toward,
        #       and which stored quantity is revised
        pass
        state = env.reset() if done else next_state
    return est
```
