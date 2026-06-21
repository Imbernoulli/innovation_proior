I present Thompson Sampling as the canonical method for balancing exploration and exploitation in online decision problems. The core idea is simple but subtle: instead of committing to a point estimate or following a fixed exploration schedule, I maintain a posterior distribution over the unknown reward parameter of each action, and I choose each action with probability equal to its posterior probability of being the best one. This turns uncertainty itself into the exploration mechanism.

The setting is a sequential allocation problem. At each round I must choose one of several actions, often called arms. Each choice yields a noisy reward whose distribution depends on an unknown parameter associated with that arm. I want to maximize cumulative reward, but I also need to learn which arms are good. The dilemma is that the only way to learn is to try arms, and every try on a suboptimal arm costs reward. Thompson Sampling resolves this by making the probability of trying an arm match the current evidence that the arm might be optimal.

For the Bernoulli bandit, the most transparent case, each arm has an unknown success probability p. I start with a uniform prior on each p, which is the Beta(1,1) distribution. After observing r successes and s failures on an arm, Bayes's rule gives the posterior Beta(r+1, s+1). The posterior mean is a natural point estimate, but Thompson Sampling does not play the posterior mean. Instead, at each decision I draw one sample from each arm's posterior and play the arm whose sampled value is largest. Because the draw is high exactly when the posterior places mass on high success probabilities, the probability that arm k is selected equals the posterior probability that arm k has the highest true success probability. This is probability matching in action.

Why is probability matching a good decision rule? Consider the two-arm case. Let P be the posterior probability that arm 1 is better than arm 2, and let Q = 1-P. If I assign arm 1 with probability P and arm 2 with probability Q, then the expected probability that the next chosen arm is inferior is 2PQ. This is at most one half, and it is strictly less than one half whenever the evidence favors either arm. A fixed alternation rule would always have inferior-selection probability one half. A greedy rule that always picks the current posterior-best arm would have inferior-selection probability Q if arm 1 appears best, which can be close to one half or worse when samples are small and the apparent best is wrong. Thompson Sampling automatically interpolates between exploration and exploitation through P itself, without tuning a schedule or confidence width.

For two arms there is even a closed-form expression for P. If arm i has r_i successes and s_i failures with n_i = r_i + s_i, then the probability that arm 2 is better than arm 1 is a ratio of binomial sums divided by a normalizing binomial coefficient. In practice, especially with many arms, we do not need this closed form: drawing one sample per posterior and taking the argmax gives the same marginal selection probabilities and is computationally trivial.

The same principle extends beyond Bernoulli bandits. When rewards are contextual, the best arm can depend on a feature vector x that changes each round. Here I maintain a Gaussian posterior over each arm's parameter vector theta_a and model the expected reward as x dot theta_a. At each round I sample a plausible theta_tilde_a from the posterior of each arm, score each arm as x dot theta_tilde_a, and play the arm with the highest score. This is Linear Thompson Sampling. The posterior covariance is naturally large in directions where an arm has been observed little, so exploration is targeted in context space rather than being uniform across arms. This allows the method to adapt to structured environments where a fixed best arm does not exist.

Thompson Sampling also adapts to non-stationary environments where arm qualities change over time. One simple and effective modification is to discount older observations in the posterior. Before each update I multiply the Beta parameters by a factor gamma slightly less than one, pulling them back toward the prior, and then add the new observation. This gives recent observations exponentially more weight, creating an effective memory of about 1/(1-gamma) rounds. A small discount lets the method forget stale segments without needing explicit changepoint detection. Clamping the parameters above the prior values keeps a floor of uncertainty and prevents collapse.

The regret of Thompson Sampling for Bernoulli bandits scales as O(log T / Delta_a) for suboptimal arms, matching the Lai-Robbins lower bound order up to constants. Empirically it often outperforms deterministic index policies at finite horizons because its randomization avoids the front-loaded exploration that fixed confidence bonuses impose. The method is also conceptually compact: it needs only a posterior update and a sampling step, with no horizon tuning or problem-specific constants beyond the prior.

The following Python script illustrates the method on a small Bernoulli bandit. I simulate three arms with true success probabilities 0.8, 0.5, and 0.2. I compare Thompson Sampling against a purely random policy and a greedy policy that always plays the arm with the highest empirical success rate. Over repeated short runs Thompson Sampling typically accumulates more reward than random and avoids the early lock-in mistakes that can trap the greedy rule. The code is self-contained and prints the average reward per step for each method.

```python
import numpy as np

np.random.seed(0)

def thompson_trial(true_probs, n_steps):
    K = len(true_probs)
    successes = np.ones(K)
    failures = np.ones(K)
    total_reward = 0.0
    for _ in range(n_steps):
        samples = np.random.beta(successes, failures)
        arm = int(np.argmax(samples))
        reward = 1 if np.random.rand() < true_probs[arm] else 0
        if reward:
            successes[arm] += 1
        else:
            failures[arm] += 1
        total_reward += reward
    return total_reward / n_steps

def random_trial(true_probs, n_steps):
    K = len(true_probs)
    total_reward = 0.0
    for _ in range(n_steps):
        arm = np.random.randint(K)
        reward = 1 if np.random.rand() < true_probs[arm] else 0
        total_reward += reward
    return total_reward / n_steps

def greedy_trial(true_probs, n_steps):
    K = len(true_probs)
    successes = np.zeros(K)
    failures = np.zeros(K)
    total_reward = 0.0
    for t in range(n_steps):
        if t < K:
            arm = t
        else:
            arm = int(np.argmax(successes / (successes + failures)))
        reward = 1 if np.random.rand() < true_probs[arm] else 0
        successes[arm] += reward
        failures[arm] += 1 - reward
        total_reward += reward
    return total_reward / n_steps

true_probs = np.array([0.8, 0.5, 0.2])
n_steps = 500
n_trials = 200

methods = {
    "Thompson Sampling": thompson_trial,
    "Random": random_trial,
    "Greedy": greedy_trial,
}

for name, fn in methods.items():
    avg = np.mean([fn(true_probs, n_steps) for _ in range(n_trials)])
    print(f"{name}: average reward per step = {avg:.4f}")
```

I call this method Thompson Sampling because it samples from the posterior to decide which action to try. The name captures the essential operation: drawing a plausible parameter from each action's posterior and selecting the action whose draw is best. Whether implemented with Beta posteriors for binary rewards, Gaussian posteriors for contextual linear models, or discounted posteriors for changing environments, the governing law remains the same. Posterior uncertainty about which action is optimal becomes the randomization law for the next action. This makes exploration targeted, automatic, and self-extinguishing as evidence accumulates.
