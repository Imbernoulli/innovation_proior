# Context

## Research question

Deep networks are beginning to give RL agents rich enough representations to act from raw pixels, and the current working recipe pairs them with experience replay to stabilize training. The setting is how to train deep-network RL agents on a stream of experience so that the updates remain stable, including for on-policy methods such as actor-critic.

## Background

**Online deep RL and correlated updates.** In a single stream of experience, consecutive states are highly correlated, so gradients point in similar directions for long stretches. Stochastic gradient descent treats samples as roughly independent. The bootstrap target is computed from the same parameters being updated, and the data distribution shifts as the policy changes.

**What replay does.** A replay buffer samples past transitions uniformly, mixing time steps so consecutive gradients are decorrelated, averaging over past policies to make the data distribution more stationary, and reusing each transition multiple times. A slowly-updated target network further stabilizes the bootstrap target.

**On-policy vs. off-policy and replay.** Uniform replay is valid for off-policy updates: the stored data was produced by older policies, and the update rule remains correct regardless of the behavior policy. Q-learning satisfies this. On-policy methods such as Sarsa, policy gradient, and actor-critic have updates that are unbiased only for data drawn from the current policy. The replay-based deep RL successes to date are off-policy value methods.

**Supporting concepts.** Discounted return, action-value and state-value functions, bootstrapped n-step returns, policy-gradient identity with a state-value baseline yielding an advantage actor-critic update, RMSProp/Adam-style adaptive stepsizes, and entropy regularization to prevent premature saturation.

## Baselines

**Q-learning with a deep network and experience replay (DQN).** Approximate Q* with a CNN trained on squared TD error over transitions sampled uniformly from a replay buffer, using a periodically-frozen target network. The method is off-policy and replay-based.

**Massively distributed asynchronous DQN (Gorila).** Scale DQN by replicating actors and learners across ~130 machines and shipping gradients asynchronously to a central parameter server. Each process keeps its own replay.

**One-step Q-learning (value-based, off-policy).** Update Q(s,a) directly toward r + γ max_a' Q(s',a'). A reward touches the single state-action pair that produced it; credit propagates through repeated bootstrapping.

**One-step Sarsa (value-based, on-policy).** Like one-step Q-learning, but the target uses the action actually taken, evaluating the behavior policy.

**n-step / multi-step Q-learning.** Update toward an n-step return so one reward directly affects several preceding state-action pairs, trading one-step bias against full-return variance.

**REINFORCE and actor-critic (policy-based, on-policy).** Parameterize the policy directly and ascend its expected return, using a learned value function as a baseline to reduce variance.

## Evaluation settings

The natural yardstick is the Arcade Learning Environment (Atari 2600) from raw frames, using the standard DQN preprocessing (grayscale, 84×84, frame stack) and an action repeat of 4. The standard network is the DQN CNN, or a shared conv body with separate softmax policy and linear value heads for policy-and-value agents. Other relevant testbeds include continuous-control physics tasks (cart-pole, pendulum, reacher, cheetah, hopper, walker, gripper, etc.) from low-dimensional state and from pixels, the TORCS 3D car-racing simulator, and first-person 3D maze navigation from vision. Metrics are average episode score/return against wall-clock training time and environment frames consumed. Key protocols are robustness over many random learning rates and initializations, and scaling behavior as parallel resources grow. Standard discounting γ = 0.99.

## Code framework

A deep-learning library supplies autodiff modules and optimizers; an Atari environment wrapper supplies resized visual observations.

```python
import math
import os
import time
from collections import deque

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces.box import Box

def create_atari_env(env_name):
    # TODO: wrap the environment and expose a normalized visual observation tensor
    pass

def _process_frame42(frame):
    # TODO: crop, resize, grayscale, and scale a raw frame
    pass

class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        # TODO: declare the visual observation shape
        pass

    def _observation(self, observation):
        # TODO: transform one observation frame
        pass

class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        # TODO: initialize running observation statistics
        pass

    def _observation(self, observation):
        # TODO: normalize one observation with running statistics
        pass

def normalized_columns_initializer(weights, std=1.0):
    # TODO: initialize output layers with a controlled column norm
    pass

def weights_init(module):
    # TODO: initialize convolutional and linear layers
    pass

class Agent(nn.Module):
    def __init__(self, num_inputs, action_space):
        super().__init__()
        # TODO: build the network that maps an observation to the
        #       quantities the chosen update rule needs
        pass

    def forward(self, inputs):
        # TODO: produce those quantities for one observation
        pass

# TODO: assemble the learning algorithm and run it on the environment.
```
