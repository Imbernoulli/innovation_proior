# Research question

By mid-2017 there are several credible families for training a neural-network policy on sequential decision tasks: value-based deep Q-learning, vanilla policy gradients, trust-region methods, and parallel actor-critic. The setting of interest is policy optimization for deep neural-network policies that must work across both continuous-control robotics and discrete pixel games, using the first-order stochastic-gradient / Adam machinery and automatic differentiation that everyone already has.

A central quantity in this setting is the surrogate objective that on-policy methods ascend: an estimate of how much the expected return improves when the current policy replaces the data-generating one. This estimate is built from a batch of experience collected under one policy and remains a faithful proxy for the true objective only while the new policy stays close to that data-generating policy. The broad question is how to turn a batch of on-policy rollouts into a policy-improvement update — and how many gradient steps to take per batch — within a plain first-order optimization loop.

# Background

The object being optimized is the expected discounted return of a stochastic policy, η(π_θ) = E[Σ_t γ^t R_t]. The workhorse is the score-function (likelihood-ratio) gradient (Williams 1992; Sutton et al. 2000):

    ∇_θ η = E_t[ ∇_θ log π_θ(a_t|s_t) · Ψ_t ],

where Ψ_t measures how good action a_t was. The lowest-variance natural choice is the **advantage** A(s_t,a_t) = Q(s_t,a_t) − V(s_t), which subtracts a state-dependent baseline V(s_t) so the estimator's variance is governed by how much *better* than average an action was, not by the absolute return scale. The estimator used in deep RL is ĝ = Ê_t[ ∇_θ log π_θ(a_t|s_t) Â_t ]. Because modern implementations rely on automatic differentiation, one does not hand-code this gradient; one writes a scalar surrogate L^PG(θ) = Ê_t[ log π_θ(a_t|s_t) Â_t ] whose gradient equals ĝ and lets the framework differentiate it.

Resting underneath everything is the conservative-policy-iteration analysis of Kakade and Langford (2002). They asked how much η changes when policy π_old is replaced by a new policy π, and produced an exact identity: η(π) − η(π_old) equals the expected advantage of π_old taken under π's *own* state distribution. That distribution is unknown — it is the very thing being produced — so they approximate it by the old state distribution, giving the surrogate

    L_{π_old}(π) = η(π_old) + E_{s∼ρ_old, a∼π}[ A_{π_old}(s,a) ],

and they bound the error of that swap by a term that grows with how far π strays from π_old (a total-variation / KL distance times a constant). The surrogate matches η to first order at π = π_old and is a **lower bound** on true improvement as long as the policy change stays small. Closeness between successive policies is the condition under which the surrogate tracks the true objective.

The bridge from "a∼π" data to "a∼π_old" data is **importance sampling**: introduce the probability ratio r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t), so the surrogate becomes Ê_t[ r_t(θ) Â_t ], with r_t(θ_old) = 1 at the start of each update. As π_θ drifts from π_old, the ratios fan out and their variance grows.

# Baselines

**Deep Q-learning (DQN, Mnih et al. 2015).** Learns an action-value function Q(s,a;θ) with a neural net using experience replay and a target network; updates by minimizing the temporal-difference error toward R + γ max_{a'} Q(s',a';θ⁻). It learns Atari from raw pixels and is built around a discrete argmax over actions.

**Vanilla policy gradients (REINFORCE-style, Williams 1992).** Ascend ĝ = Ê_t[∇_θ log π_θ(a_t|s_t) Â_t] directly, equivalently optimize L^PG. General, simple, and handle continuous and discrete actions alike; each collected sample feeds one gradient step.

**Trust Region Policy Optimization (TRPO, Schulman et al. 2015b).** Operationalizes the Kakade–Langford bound. Each update it solves a constrained problem,

    maximize_θ   Ê_t[ r_t(θ) Â_t ]
    subject to   Ê_t[ KL[ π_old(·|s_t), π_θ(·|s_t) ] ] ≤ δ,

where the KL ball bounds the distributional move. It solves this over a million-parameter network by linearizing the objective (gradient g), taking a quadratic approximation of the KL constraint whose Hessian is the Fisher information matrix F, and stepping in the natural-gradient direction F⁻¹g scaled to land on the δ boundary, θ = θ_old + √(2δ / (gᵀF⁻¹g)) · F⁻¹g. F is never formed; conjugate gradient solves F x = g using only Fisher-vector products (a couple of backprops each), followed by a backtracking line search to enforce the true KL constraint and verify improvement. It is reliable for continuous control, takes one constrained step per batch, and defines the KL on the policy's output distribution.

**Parallel actor-critic (A3C/A2C, Mnih et al. 2016).** Runs many actor-learners, each interacting with its own environment copy, accumulating gradients for a shared policy/value network. Uses a fixed rollout length T much shorter than an episode, with n-step bootstrapped returns for the advantage:

    Â_t = −V(s_t) + R_t + γ R_{t+1} + ... + γ^{T−t−1} R_{T−1} + γ^{T−t} V(s_T).

Two reusable ingredients: a single network that **shares parameters** between a policy head and a value head (so the value loss is folded into the objective), and an **entropy bonus** S[π_θ](s) added to the objective to discourage premature collapse to a deterministic policy. A2C is the synchronous GPU-friendly variant: wait for all N workers to finish T steps, batch the NT samples, do one synchronous update.

**Generalized Advantage Estimation (GAE, Schulman et al. 2015a).** A variance-reduced advantage estimator built on a learned value function. With the one-step TD residual δ_t = R_t + γ V(s_{t+1}) − V(s_t) (itself a one-step advantage estimate), GAE forms Â_t^{(γ,λ)} = Σ_{l≥0} (γλ)^l δ_{t+l}. λ is a bias-variance knob: λ→0 gives δ_t (low variance, biased by V's errors); λ→1 telescopes to the Monte-Carlo advantage Σ_l γ^l R_{t+l} − V(s_t) (unbiased, high variance); intermediate λ ≈ 0.95 with γ = 0.99 works well. Truncated to a length-T rollout it reduces to the A3C n-step estimator at λ = 1, so it is a strict generalization. This is the standard advantage estimator in this harness.

**Cross-entropy method (CEM, Szita & Lőrincz 2006)** and **adaptive-stepsize vanilla PG.** Gradient-free black-box optimization over policy parameters (CEM) and a policy gradient whose Adam stepsize is rescaled by the realized KL each batch. Both are sometimes effective on continuous control and serve as additional yardsticks.

# Evaluation settings

The yardstick is a mix of continuous-control and discrete pixel benchmarks. For continuous control, **OpenAI Gym** (Brockman et al. 2016) wrapping the **MuJoCo** physics engine (Todorov et al. 2012) provides simulated robotics tasks — HalfCheetah, Hopper, Walker2d, Reacher, Swimmer, InvertedPendulum, InvertedDoublePendulum, the "-v1" suite — typically trained for on the order of one million timesteps, scored by average total reward over the most recent episodes, often shifted and scaled so a random policy scores 0 and the best result 1 to permit averaging across environments and seeds. Higher-dimensional 3D humanoid locomotion / steering tasks (run, steer, get up, possibly while being pelted) are available via the Roboschool simulator. For discrete control, the **Arcade Learning Environment** (Bellemare et al. 2015), the 49-game Atari benchmark, supplies the pixel domain, with two standard scoring metrics — average reward per episode over the whole training period (favoring fast learning) and over the last 100 episodes (favoring final performance). Protocol conventions of the era include several random seeds per environment, learning curves over a fixed timestep/frame budget, and comparison against tuned implementations of the prior methods above (TRPO, the cross-entropy method, vanilla policy gradient with adaptive step size, A2C, A2C with a trust region, ACER).

# Code framework

The substrate is a bare on-policy policy-gradient harness: collect rollouts from parallel environments, estimate advantages, then perform some policy-improvement update. The open slot is how a batch of rollouts becomes a parameter update. Concretely the pieces are:

**Environments and data pipeline.** Gym exposes `reset()` / `step(action)` over vectorized parallel environments (`gym.vector.SyncVectorEnv`), wrapped with episode-statistics recording and, for continuous control, observation normalization, observation clipping to ±10, reward normalization (discounted-return scaling), reward clipping to ±10, and action clipping to the box bounds. The data-collection skeleton is the A2C one: N actors each roll the current policy for a fixed T steps into preallocated buffers (`obs`, `actions`, `logprobs`, `rewards`, `dones`, `values`), producing an NT-sample batch per iteration.

**Optimizer and autodiff.** First-order **Adam** (Kingma & Ba 2014) is the optimizer, typically with epsilon set to 1e-5 rather than the framework default 1e-8. Everything runs through an autodiff framework: one builds a scalar loss, calls `loss.backward()`, clips the *global* gradient norm (a max-norm cap, commonly 0.5, that rescales the whole gradient vector if it spikes), and calls `optimizer.step()`.

**Network primitives.** For continuous control, a small fully-connected MLP with two hidden layers of 64 units and tanh nonlinearities, outputting the mean of a Gaussian action distribution, paired with a separate **state-independent** learnable log-standard-deviation parameter (one vector shared across all states); a twin MLP outputs the scalar value V(s). Weight initialization is **orthogonal**, with layer-specific gains: √2 on hidden layers, 0.01 on the policy output layer, and 1.0 on the value output layer. For Atari, a DQN/A3C-style convolutional trunk feeds a categorical action head.

**Advantage estimation.** Generalized advantage estimation is the prior-art estimator: a reverse scan over the rollout computes δ_t = R_t + γ V(s_{t+1})(1−done) − V(s_t) and Â_t = δ_t + γλ(1−done)Â_{t+1}, with value targets V_t^targ = Â_t + V(s_t); the (1−done) mask zeroes the bootstrap across episode boundaries.

**The scaffold.** The open pieces are the policy/value module and the rule that turns collected rollouts into a parameter update:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # orthogonal init; caller passes std=0.01 for the policy head, 1.0 for the value head
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Policy and value module for the on-policy harness."""
    def __init__(self, envs):
        super().__init__()
        # TODO: build the action head, value head, and exploration parameter
        raise NotImplementedError

    def get_value(self, x):
        raise NotImplementedError  # TODO

    def get_action_and_value(self, x, action=None):
        # TODO: return action, log pi(a|s), entropy, V(s)
        raise NotImplementedError


def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda):
    # prior-art advantage estimator: A_t = delta_t + gamma*lambda*(1-done)*A_{t+1}
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns


def update(agent, optimizer, rollouts):
    # TODO: the policy-improvement objective we'll design.
    # Input: a batch of rollouts (obs, actions, old log-probs, advantages,
    # value targets, old values). Output: first-order parameter update(s).
    raise NotImplementedError


# ---- training loop scaffold ----
# agent = Agent(envs)
# optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
# for iteration in range(num_iterations):
#     # (optional) anneal the learning rate over the run
#     # 1) roll out N actors x T steps under no_grad into preallocated buffers
#     rollouts = collect_rollouts(agent, envs, num_steps)
#     # 2) advantages via the prior-art estimator
#     advantages, returns = compute_gae(rollouts.rewards, rollouts.values,
#                                       rollouts.dones, next_value, next_done,
#                                       gamma=0.99, gae_lambda=0.95)
#     # 3) improve the policy on this batch
#     update(agent, optimizer, rollouts.with(advantages, returns))
#     # 4) the updated policy becomes the next data-generating policy
```

The policy/value module and `update()` are the two open pieces: the model parameterization and the rule that turns a batch of on-policy rollouts into first-order parameter updates.
