## Research Question

Can a continuous-control actor-critic use a much larger policy and value network without changing
the reinforcement-learning algorithm itself? The setting is a replay-buffer agent on state-based
control tasks: observations are low-dimensional but heterogeneous, actions are continuous, and the
training budget is fixed.

## Why Larger MLPs Behave Differently

The standard actor-critic network is usually a shallow MLP. Bootstrapped value learning, off-policy
replay, and a drifting state distribution make the target less stable than a supervised label. When
this plain MLP is widened or deepened, the critic operates under the Bellman backup with shifting
targets and a non-stationary replay buffer.

## What Prior Work Already Gives

Several algorithmic tools already exist. SAC gives a stable stochastic off-policy backbone with
entropy regularization. TD3-style clipped double Q reduces overestimation. High update-to-data
methods, ensembles, distributional heads, periodic resets, and optimistic exploration can improve
sample efficiency, but they also change the training protocol. Other work uses normalization or
spectral constraints to make deeper value networks less unstable. These results show that
architecture matters.

## Architectural Observations From Supervised Learning

Large supervised models are usually trained with residual paths, normalization, and appropriate
initialization. Residual paths make identity-like behavior cheap. Normalization keeps activations
on a controlled scale. ReLU feedforward blocks and careful initialization shape the frequency
content of random networks before training begins. For reinforcement learning, input statistics
add another consideration: each observation coordinate can have a different physical scale, and
the distribution shifts as the policy changes.

## Setting For A Network Architecture Study

A controlled experiment keeps the RL objective, target construction, replay buffer, exploration
scheme, and target-network update fixed. Only the actor and critic function class changes. The
same network pattern applies to the policy input and to the critic input, where the critic receives
the action concatenated to the observation.
