# Context

## Research Question

Sparse-reward reinforcement learning needs directed exploration. A policy trained only on
environment reward can receive an all-zero learning signal for a very long time, especially in
games where meaningful rewards are separated by long action sequences and risky intermediate
states. The desired extra signal is an intrinsic reward: high when the agent reaches a state unlike
its past experience, low when the state is familiar.

The mechanism has to work from high-dimensional image observations and scale with modern RL
training: many parallel actors, long rollouts, and billions of frames. A practical bonus therefore
cannot require a dense state table, expensive posterior inference, or per-transition retraining
measurements. It should be computed by batched neural-network inference and should plug into an
ordinary policy-gradient optimizer.

The central difficulty is that novelty is not the same as unpredictability. Some parts of an
environment remain hard to predict forever because they are stochastic, partially observed, or
irrelevant noise. A curiosity bonus that rewards that irreducible error can become an attractor
instead of an exploration guide.

## Prior Mechanisms

Tabular count bonuses assign larger reward to states with smaller visit counts, often using
functions such as \(1/N(s)\) or \(1/\sqrt{N(s)}\). In image-based environments almost every raw
observation is unique, so count methods need generalization. Pseudo-count methods get that
generalization from learned density models over observations, but the density model is a substantial
extra system and is awkward to scale across many actors.

Prediction-error curiosity replaces density estimation with a supervised prediction problem. A
forward model predicts the next observation or its features from the current observation and action;
large error marks a transition as novel because a model trained on past experience has not yet fit
that region. This is cheap and neural-network-friendly, but it is vulnerable to the "noisy-TV"
failure: if the target is intrinsically random, the prediction error never decays.

Learning-progress methods try to fix that by rewarding improvement in the predictor rather than raw
error. That suppresses permanent stochastic error, but measuring improvement is more expensive and
less clean in a large distributed rollout system.

Random features and randomized prior functions provide another nearby ingredient. A randomly
initialized network can define a rich feature map, and a random prior function can make supervised
uncertainty estimates behave more like posterior samples. The open question is whether a random
prediction problem can be made into a simple novelty signal without importing the noisy target
failure.

## Error Sources

A prediction error can be high for four different reasons. First, the predictor has seen too little
nearby data; this is the useful epistemic part and is the intended novelty signal. Second, the target
function is stochastic; this is the aleatoric part that creates noisy-TV traps. Third, the prediction
problem is misspecified: the necessary inputs are missing, or the function class cannot fit the
target. Fourth, optimization and learning dynamics may fail even when the target is representable.

Only the first source should drive exploration. The target prediction problem should therefore avoid
stochastic labels and avoid missing-input structure. It should also be simple enough that fitting
visited observations is plausible, while still leaving generalization to unseen regions dependent on
data rather than on a perfect global shortcut.

Episode boundaries create a separate design constraint. Novelty is useful even across game-over
boundaries: dying after a risky attempt should not make all future novelty disappear from the
intrinsic return. Environment reward is different; treating extrinsic return as non-episodic can make
an agent exploit early rewards by deliberately restarting. The two reward streams need different
termination handling.

## Evaluation Frame

The relevant benchmark is hard-exploration Atari with image observations, sparse rewards, sticky
actions for nondeterminism, no demonstrations, and no access to emulator state. Montezuma's Revenge
is the main stress case because plain stochastic exploration rarely discovers sustained positive
reward and because room transitions expose the stochastic-transition failure of forward prediction.

The method should be judged by extrinsic episodic return, by rooms or regions discovered, and by
ablations that isolate the intrinsic reward from environment reward. Comparisons should include a
plain policy optimizer, pseudo-count or count-style methods where available, and dynamics-prediction
curiosity under the same optimizer and preprocessing.

Implementation details are part of the evaluation frame: observation normalization, reward-scale
normalization, episode-boundary handling, and auxiliary-model update rate can change behavior enough
to determine whether the policy ever leaves the first room.

## Code Frame

The existing trainer has a vectorized Atari environment, PPO rollouts, clipped policy updates,
generalized advantage estimation, and a convolutional policy/value network. The missing component is
the intrinsic reward module and the way its returns are combined with environment returns.

```python
import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.encoder = make_atari_encoder(in_channels=4)
        self.pi = nn.Linear(512, n_actions)
        self.value = None  # TODO: value head or heads

    def forward(self, frame_stack):
        h = self.encoder(frame_stack / 255.0)
        raise NotImplementedError


class IntrinsicBonus(nn.Module):
    def reward(self, next_observation):
        raise NotImplementedError

    def loss(self, observations):
        raise NotImplementedError


def normalize_bonus_observation(single_frame, running_stats):
    raise NotImplementedError


def compute_returns_and_advantages(rewards, dones, value_predictions):
    raise NotImplementedError
```
