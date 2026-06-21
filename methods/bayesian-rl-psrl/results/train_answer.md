The problem is exploration in an unknown finite-horizon Markov decision process. The learner knows the state and action sets and the horizon, but not the transition probabilities or reward distribution. Each episode begins from a fixed start-state distribution, and the agent must choose actions for the whole horizon. The catch is that actions produce both reward and information, and the value of that information may only appear several steps later. The benchmark is the policy that would be optimal if the true model were known, so every learning rule must trade short-term reward against model discovery.

Existing ideas each have a failure mode. Greedy control with maximum-likelihood estimates treats uncertainty as if it has already disappeared. Once early samples make a useful action look bad, the learner can stop collecting the evidence that would correct the estimate, and the policy can lock into a local mistake. Random dithering prevents complete lock-in, but random moves are not a coherent experiment: they do not necessarily carry the agent through the sequence of states that would reveal the missing information. The fully Bayesian answer is to plan in the belief state, but that requires integrating over models, observations, and future policies, which is computationally intractable for all but the smallest problems. Optimistic algorithms such as UCRL2 solve the exploration problem with strong guarantees, but they do so by explicitly constructing confidence sets and acting with respect to the most favorable plausible model, which adds analytical and algorithmic machinery that is not part of the simple probability-matching intuition.

The method I propose is Posterior Sampling for Reinforcement Learning, abbreviated PSRL. The algorithm is simple. Maintain a Bayesian posterior distribution over finite-horizon MDPs. At the start of each episode, sample one MDP from the posterior conditioned on all history so far. Solve that sampled MDP exactly, for example by backward induction, to obtain an optimal policy for the sampled model. Execute that policy for the entire episode without resampling, observe the resulting rewards and transitions, and update the posterior. That is the whole algorithm.

Why this works is less obvious than it looks. In a multi-armed bandit, probability matching means sampling a parameter vector from the posterior and taking the action that is best under that sample. Actions that remain plausibly optimal keep being tried; actions that the evidence rules out fade away. In an MDP we cannot simply resample at every step, because a sampled model might say that reward is reached only after a long chain of correct decisions. Changing the sampled world after one step would destroy the experiment before it reaches the evidence. PSRL fixes this by sampling one whole model and committing to its optimal policy for the entire episode. The exploration stays temporally coherent and goal-directed, but the computational cost is just one planning call per episode rather than planning in the exponentially larger belief space.

The theoretical guarantee comes from posterior symmetry. At the beginning of an episode, conditioned on the observed history, the true MDP and the sampled MDP have exactly the same distribution because both are drawn from the same posterior. Therefore, for any model functional that depends only on the history, the conditional expectation under the true MDP equals the conditional expectation under the sampled MDP. Applying this symmetry to the optimal-value functional removes the unknown true optimal policy from the regret expression. Instead of comparing the learner against a counterfactual optimal policy, we compare the value of the sampled policy in the sampled MDP against the value of that same policy in the true MDP. The surrogate depends only on the policy actually executed.

From there the analysis proceeds by dynamic programming. The value difference telescopes into one-step Bellman errors evaluated at the states and actions the agent actually visits, while the transition-noise term disappears in conditional expectation. If the sampled transition and reward laws are close to the true laws at the visited state-action pairs, the per-episode error is small. As visits accumulate, the empirical confidence radii shrink. The confidence sets are used only in the proof; the algorithm itself never constructs them or maximizes over them. This separation is the distinctive feature: PSRL has the exploration guarantees of optimistic RL while remaining a pure probability-matching rule.

```python
import numpy as np


class PSRLEnv:
    def __init__(self, n_states, n_actions, horizon, seed=0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        # Unknown true transition and reward models.
        self.true_P = self.rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
        self.true_R = self.rng.beta(2, 5, size=(n_states, n_actions))

    def reset(self):
        return 0

    def step(self, s, a):
        s_next = self.rng.choice(self.n_states, p=self.true_P[s, a])
        r = self.rng.binomial(1, self.true_R[s, a])
        return s_next, r


class PSRLAgent:
    def __init__(self, n_states, n_actions, horizon, seed=0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        # Dirichlet prior with uniform pseudocounts for transitions.
        self.transition_counts = np.ones((n_states, n_actions, n_states))
        # Beta(1, 1) prior for binary rewards.
        self.reward_success = np.ones((n_states, n_actions))
        self.reward_failure = np.ones((n_states, n_actions))

    def sample_mdp(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                P[s, a] = self.rng.dirichlet(self.transition_counts[s, a])
        R = self.rng.beta(self.reward_success, self.reward_failure)
        return P, R

    def plan(self, P, R):
        # Finite-horizon backward induction.
        V = np.zeros((self.horizon + 1, self.n_states))
        policy = np.zeros((self.horizon, self.n_states), dtype=int)
        for t in range(self.horizon - 1, -1, -1):
            for s in range(self.n_states):
                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    q_values[a] = R[s, a] + np.dot(P[s, a], V[t + 1])
                best_a = int(np.argmax(q_values))
                policy[t, s] = best_a
                V[t, s] = q_values[best_a]
        return policy

    def update(self, trajectory):
        for s, a, r, s_next in trajectory:
            self.transition_counts[s, a, s_next] += 1
            if r > 0:
                self.reward_success[s, a] += 1
            else:
                self.reward_failure[s, a] += 1


def train(n_states=5, n_actions=2, horizon=10, n_episodes=1000, seed=0):
    env = PSRLEnv(n_states, n_actions, horizon, seed=seed)
    agent = PSRLAgent(n_states, n_actions, horizon, seed=seed + 1)
    total_reward = 0
    for _ in range(n_episodes):
        P, R = agent.sample_mdp()
        policy = agent.plan(P, R)
        s = env.reset()
        trajectory = []
        ep_reward = 0.0
        for t in range(horizon):
            a = int(policy[t, s])
            s_next, r = env.step(s, a)
            trajectory.append((s, a, r, s_next))
            ep_reward += r
            s = s_next
        agent.update(trajectory)
        total_reward += ep_reward
    return total_reward / n_episodes


if __name__ == "__main__":
    avg_reward = train()
    print(f"Average reward: {avg_reward:.3f}")
```
