# Context: Model-Free Optimal Control From Delayed Reward

## Research Question

An agent repeatedly observes a state, chooses an action, receives a reward, and moves to another state. The environment is a finite controlled Markov process: the next-state distribution and expected reward depend only on the current state and chosen action. The objective is to learn behavior that maximizes expected discounted return from every state, with discount factor `0 < gamma < 1`, while the reward function and transition probabilities are not given.

The central difficulty is delayed credit assignment. A useful reward may arrive many steps after the action that made it possible, and exploratory actions are necessary before the agent knows which actions are useful.

## Background

Classical dynamic programming solves the known-model version of the control problem. For a fixed policy, the value of a state is the expected discounted return obtained by following that policy. Because the process is Markov and discounting is exponential, the value function satisfies a one-step consistency equation. If the reward means and transition probabilities are known, policy evaluation is a finite system of linear equations.

Optimal control adds Bellman's optimality principle: the best value at a state is obtained by comparing the available actions after accounting for immediate reward and the best continuation value. This yields value iteration and policy iteration. Those methods require the model.

Temporal-difference learning addresses prediction without a model. Instead of waiting for a full return, a learner can bootstrap from a sampled reward and a current estimate of the next state's value. Sutton's temporal-difference analysis shows why this is useful for prediction: corrected truncated returns reduce error toward the value of the policy being evaluated. Robbins-Monro stochastic approximation gives the step-size conditions under which noisy sampled targets can be averaged consistently.

## Baselines

Known-model value iteration and policy iteration compute the dynamic-programming optimum, but require the reward and transition model explicitly. They establish the benchmark, not a direct learning method from raw experience.

Certainty-equivalence control estimates the model from data and then plans in that estimated model.

Temporal-difference prediction learns the value of a fixed policy from sampled experience.

Actor-critic and learning-automata style controllers keep a value estimate and a separately represented policy or action-strength system. They can improve behavior empirically.

Monte Carlo control waits for sampled returns before changing estimates. It avoids a model and can learn from complete episodes.

## Evaluation Settings

The main setting is a finite tabular controlled Markov process with bounded rewards and discount `0 < gamma < 1`. The learner receives sampled transitions only. It should converge to the optimal state-action values and therefore induce an optimal greedy policy, provided every state-action pair is sampled often enough and the learning rates decrease according to stochastic-approximation conditions.

Important diagnostics are whether the learned estimates match the dynamic-programming optimum and whether the induced greedy behavior becomes optimal. Small grid tasks, episodic absorbing tasks, and tabular control examples are natural sanity checks because the exact dynamic-programming solution can be computed for comparison.

## Code Framework

```python
import numpy as np
from collections import defaultdict


class TabularControl:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.scores = defaultdict(lambda: np.zeros(n_actions))

    def greedy_action(self, state):
        # TODO: choose the action implied by the current state-action scores.
        raise NotImplementedError

    def state_summary(self, state):
        # TODO: summarize the best continuation score available at this state.
        raise NotImplementedError


def exploratory_action(agent, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(agent.n_actions)
    return agent.greedy_action(state)


def learn(env, n_actions, gamma, alpha_schedule, epsilon, n_steps):
    agent = TabularControl(n_actions)
    state = env.reset()

    for t in range(n_steps):
        action = exploratory_action(agent, state, epsilon)
        next_state, reward, done = env.step(action)
        alpha = alpha_schedule(state, action)

        # TODO: define the sampled control target and update exactly one
        # state-action entry from the observed transition.
        raise NotImplementedError

        state = env.reset() if done else next_state

    return agent
```
