## The Exploration Setting

A learner acts in an unknown Markov decision process over repeated finite episodes. The state and action sets and the horizon are known; the transition probabilities and reward distribution are not. Each episode begins from a fixed start-state distribution. Each action produces both reward and information about rewards or transitions, and the value of that information may only appear several steps later. The benchmark is the policy that would be optimal if the true model were known, so a learning rule trades short-term reward against model discovery.

## Point Estimates and Dithering

One approach is greedy control with maximum-likelihood estimates: maintain point estimates of the transitions and rewards and act optimally with respect to them. A common variant adds random dithering, such as epsilon-greedy action selection, to keep visiting actions the current estimate would otherwise abandon.

## Probability Matching in Bandits

In multi-armed bandits, probability matching selects actions in proportion to the posterior probability that they are best. Concretely, one draws a parameter vector from the posterior and takes the action that is optimal under that draw. Actions that remain plausibly best keep being tried; actions the evidence rules out are selected with vanishing probability. Each action in a bandit is itself a complete one-step experiment.

## Bayesian Control

In sequential control, the fully Bayesian planner reasons over beliefs, future observations, and future courses of action — planning in the belief state by integrating over models, observations, and policies. Sparse-sampling and value-of-information schemes approximate this belief-space planning. A separate family makes the agent optimistic to drive data collection: bonus-based methods add an exploration reward, and confidence-set methods such as UCRL2 construct a set of plausible models and act with respect to the most favorable one. Confidence-set algorithms come with finite-time regret guarantees from the explicit choice of an optimistic model.

## Bayesian Model Sampling

A Bayesian alternative is to sample a model hypothesis from the posterior, solve it, and retain it over a trial so exploration stays goal-directed (Strens, 2000, "Bayesian Dynamic Programming"). This has older empirical support. The question taken up here is the finite-time analysis: how to characterize the regret of a Bayesian model-sampling rule for episodic reinforcement learning, relative to the policy that is optimal for the unknown true MDP.
