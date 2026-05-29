## Research question

How do we learn a control policy over a **continuous, possibly high-dimensional, real-valued action space** — joint torques on a robot arm, steering/throttle on a car, the actuators of a walking biped — directly from a model-free reinforcement-learning signal, with training that is stable and that does not require a per-timestep optimization over actions? Deep value functions already give a workable recipe for high-dimensional observations when the action set is small and discrete. The open problem is the action side: a continuous action vector cannot be enumerated, and the greedy maximization used by value-based control becomes an expensive nonlinear optimization problem.

A solution has to (1) act in real time with one cheap forward pass per timestep; (2) scale to action vectors with many dimensions; (3) remain stable when the value function is a large nonlinear neural network; (4) learn off-policy so that old exploratory data can be reused; and ideally (5) keep the same core implementation across a wide range of physical-control tasks.

## Background

**The Markov decision process and value functions.** An agent observes `s_t`, takes `a_t`, receives `r_t`, and transitions to `s_{t+1}`. With discount `γ ∈ [0,1]`, the return is `R_t = Σ_{k≥0} γ^k r_{t+k}` and the control objective is expected return from the start distribution. The action-value function for a policy `π` is `Q^π(s,a) = E[R_t | s_t=s, a_t=a]`. It obeys
`Q^π(s,a) = E_{r,s'}[r + γ E_{a'~π(·|s')} Q^π(s',a')]`.
For a deterministic policy `μ`, the action expectation collapses:
`Q^μ(s,a) = E_{r,s'}[r + γ Q^μ(s', μ(s'))]`.
At terminal transitions, the continuation term is zero, so a sampled bootstrap target uses `r + γ(1-d)Q(...)`.

**Value-based control and its action bottleneck.** Q-learning (Watkins & Dayan, 1992) learns a value function for greedy control. With a function approximator, its target has the form `y = r + γ max_{a'} Q(s',a')`, and the greedy policy is `argmax_a Q(s,a)`. With a small finite action set this is cheap: evaluate one value per action and take the largest. With continuous actions, `argmax_a Q(s,a)` is a non-convex optimization over a real vector and would have to be solved both at action-selection time and inside every bootstrap target.

**Deep value functions can be stabilized.** The deep Q-network line (Mnih et al., 2013, 2015) made large neural Q-learning practical in discrete-action domains by using a replay buffer and a target network. Replay stores transitions and samples uniform minibatches, breaking the short-range correlation of consecutive environment steps and reusing data. A target network computes bootstrap targets with parameters that do not move on the same optimizer step as the fitted network, reducing the feedback loop in which a regressor chases a target produced by itself.

**The diagnostic that motivates the continuous-action case.** Discretizing a continuous action vector invokes the curse of dimensionality: a 7-DOF arm with just three choices per joint already has `3^7 = 2187` joint actions, and finer torque control increases the grid exponentially. Discretization also discards metric structure: nearby torques become unrelated symbols. Keeping actions continuous preserves that structure, but then a neural `Q(s,a)` does not provide a cheap greedy action. The missing ingredient is a way to represent and improve the greedy action without solving a fresh inner maximization at every step.

**Exploration noise for deterministic controllers.** A deterministic controller needs an explicit behavior-noise process during data collection. Independent Gaussian action noise is the simplest option and is easy to implement as `a = μ(s) + σξ`. For inertial physical systems, a temporally correlated process is also a natural prior: the zero-mean Ornstein-Uhlenbeck Euler update with unit step is `x ← x + θ_ou(-x) + σξ`, so perturbations persist for several timesteps while still reverting toward zero.

## Baselines

**Q-learning with deep networks (DQN), discrete actions.** The core algorithm learns `Q(s,a)` by mean-squared Bellman error, uses replay for minibatch decorrelation, and uses a target network for stable bootstrap targets. Its gap is the finite action head: both acting and training require a cheap `max_a`, which continuous actions do not provide.

**Deterministic policy gradient (DPG; Silver et al., 2014).** A deterministic policy `μ_θ(s)` can be improved by the chain rule through the critic. If `ρ^μ` is the unnormalized discounted state visitation measure, the deterministic policy gradient is
`∇_θ J = ∫ ρ^μ(s) [∇_θ μ_θ(s)] [∇_a Q^μ(s,a)]_{a=μ_θ(s)} ds`.
If the state distribution is normalized to `d^μ=(1-γ)ρ^μ`, the same expression appears with a factor `1/(1-γ)`, which is normally absorbed into the learning rate. The important structural point is that the expectation is over states only, with no action integral and therefore no action-space importance ratio when samples come from a noisy behavior policy. Off-policy actor-critic uses the same chain-rule integrand sampled under the behavior/replay state distribution; formally this is a behavior-weighted approximation that drops a downstream term involving `∇_θ Q^μ`, so state coverage still matters. The original deterministic-gradient work used smaller function approximators and did not solve the large-neural-network stability problem.

**NFQCA (Hafner & Riedmiller, 2011).** Neural fitted Q-iteration with continuous actions uses actor-critic style updates and relies on batch learning for stability. Full-batch retraining over accumulated data is expensive at scale, while straightforward online minibatch neural actor-critic remains fragile without the stabilizers from deep value learning.

**Stochastic policy-gradient actor-critic.** A stochastic policy can be optimized by `E[∇_θ log π_θ(a|s) Q^π(s,a)]`. This is general, but in high-dimensional continuous action spaces the action sampling term increases variance, and off-policy variants introduce importance ratios over actions. The gap is a lower-variance continuous-control update that still permits replay.

## Evaluation settings

The natural benchmarks are simulated continuous-control tasks in physics engines such as MuJoCo: pendulum and cart-pole swing-up, reaching and manipulation with multi-DOF arms, locomotion tasks such as cheetah, hopper, walker, and quadruped control, and driving tasks with steering/throttle/brake actions. Observations may be low-dimensional state vectors or rendered images; rewards give per-step control feedback such as distance-to-goal shaping, forward progress, action costs, and fall penalties. Policies are judged by undiscounted or discounted episodic return over fixed-length rollouts, with separate evaluation rollouts run without exploration noise. Natural comparison points are uniform random behavior, model-predictive controllers with access to simulator dynamics, discrete-action deep value methods where discretization is feasible, and stochastic policy-search baselines.

## Code framework

The available primitives are an autodiff deep-learning library, MLP layers, ReLU and `tanh` nonlinearities, Adam, NumPy arrays, and an environment with `reset()`, `step(a)`, `observation_space`, and continuous `action_space`. The scaffold leaves one big slot for the learned components, with the standard off-policy plumbing (a transition store, an interaction loop, an update step) sketched around it.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

def combined_shape(length, shape=None):
    # TODO
    pass

class Policy(nn.Module):
    # Maps a state to a real-valued action vector within the action bounds.
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        # TODO: the policy network we will design
        pass
    def forward(self, obs):
        # TODO
        pass

class ValueFunction(nn.Module):
    # Estimates expected return for a (state, action) pair.
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        # TODO: the value network we will design
        pass
    def forward(self, obs, act):
        # TODO
        pass

class Controller(nn.Module):
    # Holds the learned policy and value modules and exposes an action method.
    def __init__(self, observation_space, action_space, hidden_sizes):
        super().__init__()
        # TODO
        pass
    def act(self, obs):
        # TODO
        pass

class TransitionBuffer:
    # FIFO store of (s, a, r, s', done) for off-policy minibatch sampling.
    def __init__(self, obs_dim, act_dim, size):
        # TODO
        pass
    def store(self, obs, act, rew, next_obs, done):
        # TODO
        pass
    def sample_batch(self, batch_size):
        # TODO
        pass

def value_loss(data):
    # TODO: how do we form a regression target for a continuous-action value
    #       function, and what does the policy contribute to it?
    pass

def policy_loss(data):
    # TODO: how do we improve a continuous-action policy against the value function?
    pass

def select_action(obs, exploring):
    # TODO: deterministic action plus some exploration mechanism
    pass

def update_step(data):
    # TODO: update the value function, update the policy, then update any
    #       auxiliary copies used by the bootstrap target.
    pass

def train(env_fn, controller=Controller, controller_kwargs=dict(), seed=0,
          steps_per_epoch=4000, epochs=100, replay_size=int(1e6),
          gamma=0.99, pi_lr=1e-4, q_lr=1e-3, batch_size=64, max_ep_len=1000):
    # Standard off-policy loop: act -> store -> sample minibatch -> update.
    # TODO: instantiate the networks, optimizers, and any auxiliary copies the
    #       learner needs; fill in the update and the target-forming machinery,
    #       plus whatever exploration schedule and update cadence the method needs.
    pass
```
