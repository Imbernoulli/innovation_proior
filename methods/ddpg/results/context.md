## Research question

How do we learn a control policy over a **continuous, possibly high-dimensional, real-valued action space** — joint torques on a robot arm, steering/throttle on a car, the actuators of a walking biped — directly from a model-free reinforcement-learning signal? Deep value functions already give a workable recipe for high-dimensional observations when the action set is small and discrete, where greedy control evaluates one value per action. With continuous actions the action vector is a real vector that cannot be enumerated, and the greedy maximization used by value-based control is a nonlinear optimization over that vector. The question is how to drive a model-free continuous-control policy from this signal with a training loop that reuses old exploratory data and stays usable across a range of physical-control tasks.

## Background

**The Markov decision process and value functions.** An agent observes `s_t`, takes `a_t`, receives `r_t`, and transitions to `s_{t+1}`. With discount `γ ∈ [0,1]`, the return is `R_t = Σ_{k≥0} γ^k r_{t+k}` and the control objective is expected return from the start distribution. The action-value function for a policy `π` is `Q^π(s,a) = E[R_t | s_t=s, a_t=a]`. It obeys
`Q^π(s,a) = E_{r,s'}[r + γ E_{a'~π(·|s')} Q^π(s',a')]`.
For a deterministic policy `μ`, the action expectation collapses:
`Q^μ(s,a) = E_{r,s'}[r + γ Q^μ(s', μ(s'))]`.
At terminal transitions, the continuation term is zero, so a sampled bootstrap target uses `r + γ(1-d)Q(...)`.

**Value-based control.** Q-learning (Watkins & Dayan, 1992) learns a value function for greedy control. With a function approximator, its target has the form `y = r + γ max_{a'} Q(s',a')`, and the greedy policy is `argmax_a Q(s,a)`. With a small finite action set this is cheap: evaluate one value per action and take the largest. With continuous actions, `argmax_a Q(s,a)` is an optimization over a real vector, entering both at action-selection time and inside every bootstrap target.

**Deep value functions can be stabilized.** The deep Q-network line (Mnih et al., 2013, 2015) made large neural Q-learning practical in discrete-action domains by using a replay buffer and a target network. Replay stores transitions and samples uniform minibatches, breaking the short-range correlation of consecutive environment steps and reusing data. A target network computes bootstrap targets with parameters that do not move on the same optimizer step as the fitted network, reducing the feedback loop in which a regressor chases a target produced by itself.

**Discretizing a continuous action vector.** A 7-DOF arm with just three choices per joint already has `3^7 = 2187` joint actions, and finer torque control increases the grid exponentially. Discretization also turns nearby torques into unrelated symbols. Keeping actions continuous preserves that metric structure, and a neural `Q(s,a)` then represents value as a smooth function of the real action vector.

**Exploration noise for deterministic controllers.** A deterministic controller needs an explicit behavior-noise process during data collection. Independent Gaussian action noise is the simplest option and is easy to implement as `a = μ(s) + σξ`. For inertial physical systems, a temporally correlated process is also a natural prior: the zero-mean Ornstein-Uhlenbeck Euler update with unit step is `x ← x + θ_ou(-x) + σξ`, so perturbations persist for several timesteps while still reverting toward zero.

## Baselines

**Q-learning with deep networks (DQN), discrete actions.** The core algorithm learns `Q(s,a)` by mean-squared Bellman error, uses replay for minibatch decorrelation, and uses a target network for stable bootstrap targets. Both acting and training use a `max_a` over a finite action head.

**Deterministic policy gradient (DPG; Silver et al., 2014).** A deterministic policy `μ_θ(s)` can be improved by the chain rule through the critic. If `ρ^μ` is the unnormalized discounted state visitation measure, the deterministic policy gradient is
`∇_θ J = ∫ ρ^μ(s) [∇_θ μ_θ(s)] [∇_a Q^μ(s,a)]_{a=μ_θ(s)} ds`.
If the state distribution is normalized to `d^μ=(1-γ)ρ^μ`, the same expression appears with a factor `1/(1-γ)`, which is normally absorbed into the learning rate. The expectation is over states only, with no action integral and therefore no action-space importance ratio when samples come from a noisy behavior policy. Off-policy actor-critic uses the same chain-rule integrand sampled under the behavior/replay state distribution; formally this is a behavior-weighted approximation that drops a downstream term involving `∇_θ Q^μ`. The original deterministic-gradient work used smaller function approximators.

**NFQCA (Hafner & Riedmiller, 2011).** Neural fitted Q-iteration with continuous actions uses actor-critic style updates and relies on batch learning, with full-batch retraining over accumulated data.

**Stochastic policy-gradient actor-critic.** A stochastic policy can be optimized by `E[∇_θ log π_θ(a|s) Q^π(s,a)]`. This is general; the update samples actions from `π_θ`, and off-policy variants carry importance ratios over actions.

## Evaluation settings

The natural benchmarks are simulated continuous-control tasks in physics engines such as MuJoCo: pendulum and cart-pole swing-up, reaching and manipulation with multi-DOF arms, locomotion tasks such as cheetah, hopper, walker, and quadruped control, and driving tasks with steering/throttle/brake actions. Observations may be low-dimensional state vectors or rendered images; rewards give per-step control feedback such as distance-to-goal shaping, forward progress, action costs, and fall penalties. Policies are judged by undiscounted or discounted episodic return over fixed-length rollouts, with separate evaluation rollouts run without exploration noise. Natural comparison points are uniform random behavior, model-predictive controllers with access to simulator dynamics, discrete-action deep value methods where discretization is feasible, and stochastic policy-search baselines.

## Code framework

The available primitives are an autodiff deep-learning library, MLP layers, ReLU and `tanh` nonlinearities, Adam, NumPy arrays, and an environment with `reset()`, `step(a)`, `observation_space`, and continuous `action_space`. The scaffold leaves the learned policy, the state-action value function, the replay store, the bootstrap target, and the update rule as empty slots inside a standard off-policy interaction loop.

```python
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

def combined_shape(length, shape=None):
    # TODO
    pass

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class MLPActor(nn.Module):
    # Maps a state to a real-valued action vector within the action bounds.
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        # TODO
        pass
    def forward(self, obs):
        # TODO
        pass

class MLPQFunction(nn.Module):
    # Estimates expected return for a (state, action) pair.
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # TODO
        pass
    def forward(self, obs, act):
        # TODO
        pass

class MLPActorCritic(nn.Module):
    # Holds the learned policy and value modules and exposes an action method.
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        # TODO
        pass
    def act(self, obs):
        # TODO
        pass

class ReplayBuffer:
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

def train(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
          steps_per_epoch=4000, epochs=100, replay_size=int(1e6),
          gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
          batch_size=100, start_steps=10000, update_after=1000,
          update_every=50, act_noise=0.1, max_ep_len=1000):
    # TODO: instantiate the live modules, auxiliary bootstrap copies,
    #       replay buffer, and optimizers.
    def compute_loss_q(data):
        # TODO: form the value regression target.
        pass
    def compute_loss_pi(data):
        # TODO: improve the policy against the value function.
        pass
    def update(data):
        # TODO: update the value function, update the policy, then update
        #       auxiliary bootstrap copies.
        pass
    def get_action(obs, noise_scale):
        # TODO: choose an action and add the behavior exploration signal.
        pass
    # TODO: run the off-policy loop: act -> store -> sample minibatch -> update.
    pass
```
