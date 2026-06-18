## Research Question

Can a continuous-control actor-critic use a much larger policy and value network without changing
the reinforcement-learning algorithm itself? The setting is a replay-buffer agent on state-based
control tasks: observations are low-dimensional but heterogeneous, actions are continuous, and the
training budget is fixed. The desired outcome is not just a larger model that trains, but a larger
model whose extra parameters reliably improve sample efficiency and final return.

## Why Larger MLPs Break

The standard actor-critic network is usually a shallow MLP. That choice is not accidental:
bootstrapped value learning, off-policy replay, and a drifting state distribution make the target
less stable than a supervised label. When this plain MLP is widened or deepened, the critic can fit
temporary structure in the replay buffer and feed its own error back through the Bellman target.
The policy then optimizes against that distorted critic. In this regime, raw capacity can amplify
the deadly triad instead of producing the scaling behavior seen in vision or language.

## What Prior Work Already Gives

Several algorithmic tools already attack parts of the problem. SAC gives a stable stochastic
off-policy backbone with entropy regularization. TD3-style clipped double Q reduces overestimation.
High update-to-data methods, ensembles, distributional heads, periodic resets, and optimistic
exploration can improve sample efficiency, but they also change the training protocol. Other work
uses normalization or spectral constraints to make deeper value networks less unstable. These
results show that architecture matters, but they do not isolate a minimal plug-in network change
that makes parameter scaling itself useful.

## Architectural Clues Before The Answer

Large supervised models usually avoid memorizing noise because their architecture and optimizer bias
training toward simpler functions. Residual paths make identity-like behavior cheap. Normalization
keeps activations on a controlled scale. ReLU feedforward blocks and careful initialization shape
the frequency content of random networks before training begins. For reinforcement learning, input
statistics add another issue: each observation coordinate can have a different physical scale, and
the distribution shifts as the policy changes. Any scalable actor-critic architecture has to manage
both the input scale and the internal residual stream.

## Constraints For A Clean Test

The clean experiment keeps the RL objective, target construction, replay buffer, exploration scheme,
and target-network update fixed. Only the actor and critic function class should change. The same
network pattern must work for the policy input and for the critic input, where the critic receives
the action concatenated to the observation. The strongest evidence would be a critic that can grow
substantially beyond the usual small MLP and become better, not merely less unstable, while the
backbone algorithm remains recognizable as the same actor-critic.
