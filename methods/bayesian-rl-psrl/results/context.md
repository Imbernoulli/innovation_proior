## The Exploration Contract

A learner acts in an unknown Markov decision process over repeated finite episodes. Each action produces both reward and information about rewards or transitions, and the value of that information may only appear several steps later. The benchmark is the policy that would be optimal if the true model were known, so every learning rule must trade short-term reward against model discovery.

## Point Estimates Fail Quietly

Greedy control with maximum-likelihood estimates treats uncertainty as if it has already disappeared. Once early samples make a useful action look bad, the learner can stop collecting the evidence that would correct the estimate. Random dithering can prevent complete lock-in, but it is not the same as planning a coherent test of a possible model.

## Probability Matching Before Control

In bandits, probability matching gives a different answer: sample actions in proportion to the posterior chance that they are best. It keeps trying actions that remain plausible and naturally fades away from actions that the evidence rules out. This principle is simple in one-step problems because each action is itself a complete experiment.

## Bayesian Control Was Too Large

In sequential control, the Bayesian ideal is much larger. A fully Bayes-adaptive planner must reason over beliefs, future observations, and future courses of action. Sparse-sampling and value-of-information approximations try to manage that burden, while bonus and sampled-set methods make the agent optimistic enough to gather data. These routes show how to use uncertainty, but they add planning or optimism structures that are not part of plain probability matching.

## The Proof Gap

Confidence-set algorithms had finite-time guarantees because they explicitly chose optimistic models. A simpler Bayesian model-sampling rule had attractive behavior and older empirical support, but its regret analysis still seemed to require comparison with a policy selected for the unknown true model. The open proof problem was to show efficient exploration without making the learner choose from a confidence set or adding an explicit bonus.
