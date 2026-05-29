# Context

## Research question

How can we build a model-free deep reinforcement learning algorithm for continuous state and
action spaces that is simultaneously **sample-efficient** and **stable**?

Two failure modes dominate continuous-control deep RL at this point. The first is sample
complexity: the most reliable algorithms are on-policy, and on-policy learning discards every
batch of experience after a single gradient step, so even modest tasks burn through millions of
environment interactions. The second is brittleness: the off-policy algorithms that *could* reuse
old data are notoriously hard to stabilize — small changes in learning rate, exploration constant,
or target-update rate swing performance between "solved" and "no progress," and the basin of good
hyperparameters shrinks as the task grows. On the hardest high-dimensional benchmarks, the
sample-efficient off-policy method fails to make any progress at all while a slower on-policy
method still works.

A solution would have to combine off-policy data reuse (for efficiency) with a training procedure
that is robust across random seeds and needs almost no per-task tuning, and it would have to scale
to high-dimensional action spaces (e.g. a 21-DoF humanoid) where the existing off-policy approach
breaks down.

## Background

**Actor-critic from policy iteration.** Actor-critic methods descend from policy iteration, which
alternates *policy evaluation* (compute the value function of the current policy) with *policy
improvement* (use that value function to produce a better policy). At scale neither step is run to
convergence; instead the value function (critic) and policy (actor) are optimized jointly. The
classical actor update is the on-policy policy gradient (Sutton; Peters & Schaal 2008).

**The maximum-entropy RL framework.** Standard RL maximizes the expected return
Σ_t E[r(s_t,a_t)]. The maximum-entropy framework (Ziebart 2008, 2010; Todorov 2008; Toussaint
2009; Rawlik 2012; Fox, Pakman & Tishby 2016) augments this with the policy's entropy:

  J(π) = Σ_t E_{(s_t,a_t)~ρ_π}[ r(s_t,a_t) + α·H(π(·|s_t)) ].

The temperature α trades reward against randomness; as α→0 the standard objective is recovered.
The motivations, established in prior work, are concrete facts about how such
policies behave: (i) entropy-seeking policies explore more widely while abandoning clearly
unpromising directions; (ii) they can represent *multiple* modes of near-optimal behavior,
assigning comparable probability to comparably-good actions instead of arbitrarily committing to
one; (iii) Ziebart (2010) shows maximum-entropy policies are robust to model and estimation error;
and (iv) prior work reports improved exploration under this objective. In the maximum-entropy
framework the optimal solution is characterized by a *soft* Bellman equation in which the usual
hard maximum over actions is replaced by a log-sum-exp ("soft max"), and the optimal policy is an
energy-based (Boltzmann) distribution proportional to the exponentiated value.

**The reparameterization trick.** To differentiate an expectation E_{a~π_θ}[f(a)] with respect to
the distribution parameters θ when f is itself a differentiable function, one can express the
sample as a deterministic transform of fixed noise, a = f_θ(ε; s) with ε drawn from a fixed
distribution (Kingma & Welling 2013). The gradient then flows through the sample, giving a
low-variance estimator. The alternative — the likelihood-ratio / score-function estimator
(Williams 1992), ∇_θ E[f] = E[f·∇_θ log π_θ] — does not require f to be differentiable but is high
variance, and ignores any gradient information available in f.

**Bootstrapping with target networks.** Regressing a value network onto a bootstrapped target that
depends on the same network is unstable; using a slowly-tracking copy of the weights as the target
(a periodically-copied or exponentially-averaged "target network," Mnih 2015) stabilizes the
regression.

**Overestimation bias.** Value-based methods that take a maximum (or a max-like operation) over a
noisy value estimate are biased upward (van Hasselt 2010). Maintaining two independent value
estimators and using the smaller of the two reduces this positive bias.

## Baselines

**Deep deterministic policy gradient — DDPG (Lillicrap 2015), the deep form of DPG (Silver 2014).**
DDPG is the popular off-policy actor-critic for continuous control. It learns a Q-function by
minimizing the mean-squared Bellman error against a target
y = r + γ·Q_targ(s', μ_targ(s')), and — because the maximization max_a Q(s,a) needed by Q-learning
is intractable in continuous action spaces — it trains a *deterministic* actor μ_θ(s) to
approximate the maximizer, updated through the chain rule
∇_θ Q(s, μ_θ(s)) = ∇_a Q · ∇_θ μ_θ. It is at once an approximate Q-learning method and a
deterministic actor-critic. Gap it leaves open: the deterministic actor has no intrinsic
exploration, so time-correlated (Ornstein-Uhlenbeck) or Gaussian noise must be injected and tuned;
the actor–Q interplay is extremely brittle and hyperparameter-sensitive; the Q-function
overestimates; and the method fails to scale to high-dimensional tasks, where on-policy methods
still win.

**Stochastic value gradients, zero-step — SVG(0) (Heess 2015).** Uses the reparameterization trick
to push value gradients through a *stochastic* actor. But it optimizes the standard
maximum-expected-return objective (no entropy term) and uses no separate value network.

**Energy-based / soft Q-learning (Haarnoja 2017; Fox 2016; Ziebart 2008).** These solve the
maximum-entropy problem by directly learning the optimal soft Q-function, from which the optimal
policy is recovered as the Boltzmann distribution π ∝ exp(Q/α). The soft value function is the
log-sum-exp V(s) = α·log ∫ exp(Q(s,a)/α) da. Gap it leaves open: in continuous action spaces there
is no closed-form way to sample from exp(Q), so an auxiliary sampling network must be trained (via
amortized Stein variational gradient descent) to approximate draws from the energy-based policy.
This is a Q-learning method, not a true actor-critic: the Q-function targets the *optimal* Q*, and
the "actor" is only an approximate sampler whose quality bounds convergence; the inference
procedure is complex and a source of instability. These methods generally do not exceed the
performance of the strong off-policy baseline when learning from scratch.

**On-policy policy gradients with entropy regularization — TRPO (Schulman 2015), PPO (Schulman
2017), A3C (Mnih 2016).** Stable and effective; many of them add an entropy *regularizer* to the
policy-gradient objective (entropy as a bonus, not as the quantity being maximized in the value).
Gap they leave open: being on-policy, they cannot reuse past experience and so are
sample-inefficient; the large batch sizes they need on complex tasks make them slow.

**Double Q-learning / clipped double-Q (van Hasselt 2010; concurrent TD3, Fujimoto 2018).**
Maintains two Q-estimators and uses the minimum to suppress the overestimation bias that degrades
value-based methods.

## Evaluation settings

The natural yardstick is the suite of continuous-control tasks from the OpenAI Gym benchmark
(Brockman 2016) built on the MuJoCo simulator, together with the rllab implementation of the
high-dimensional Humanoid task (Duan 2016). Representative tasks span a range of action
dimensionalities — Hopper (3), Walker2d (6), HalfCheetah (6), Ant (8), Humanoid-v1 (17), and
Humanoid-rllab (21) — chosen so that the easy tasks are solvable by many algorithms while the
high-dimensional ones (especially the 21-DoF humanoid) are known to be very hard for off-policy
methods. The protocol: train several instances per algorithm under different random seeds,
periodically run evaluation rollouts (e.g. one every 1000 environment steps), and report the
total average return over training, summarizing both the mean and the spread across seeds — so that
both sample efficiency (how fast return rises) and stability (how tightly seeds agree) are visible.

## Code framework

A minimal off-policy actor-critic harness already has a replay buffer, an MLP builder, neural
policy/value modules, optimizers, and a loop that collects transitions, stores them, samples
minibatches, and performs gradient updates.

```python
import torch
import torch.nn as nn
import numpy as np

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class ReplayBuffer:
    """Store transitions (s, a, r, s', done) for off-policy updates."""
    def __init__(self, obs_dim, act_dim, size):
        self.obs  = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.act  = np.zeros((size, act_dim), dtype=np.float32)
        self.rew  = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
    def store(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = o, a, r
        self.obs2[i], self.done[i] = o2, d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        to = lambda x: torch.as_tensor(x, dtype=torch.float32)
        return dict(obs=to(self.obs[idx]), act=to(self.act[idx]),
                    rew=to(self.rew[idx]), obs2=to(self.obs2[idx]),
                    done=to(self.done[idx]))

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        # TODO: choose the policy parameterization
        pass
    def forward(self, obs, deterministic=False, with_logprob=True):
        # TODO: produce an action and any training signal needed by the update
        pass

class ActionValueNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        # TODO: define the state-action value estimator
        pass
    def forward(self, obs, act):
        # TODO: return a scalar value for each (state, action)
        pass

class StateValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        # TODO: define the state value estimator if the update needs one
        pass
    def forward(self, obs):
        # TODO: return a scalar value for each state
        pass

class OffPolicyAgent:
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256, 256),
                 gamma=0.99, tau=0.005, lr=3e-4):
        # TODO: instantiate policy/value modules and optimizers
        pass
    def update(self, data):
        # TODO: compute losses, apply optimizer steps, and update target weights
        pass
    def act(self, obs, deterministic=False):
        # TODO: return a NumPy action for environment interaction
        pass

def reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs2, reward, terminated, truncated, info = out
        return obs2, reward, terminated or truncated, info
    obs2, reward, done, info = out
    return obs2, reward, done, info

def train(env, agent, steps=int(1e6), batch_size=256, start_steps=10000):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buffer = ReplayBuffer(obs_dim, act_dim, int(1e6))
    o = reset_env(env)
    for t in range(steps):
        a = env.action_space.sample() if t < start_steps else agent.act(o)
        o2, r, d, _ = step_env(env, a)
        buffer.store(o, a, r, o2, d)
        o = reset_env(env) if d else o2
        if t >= start_steps and buffer.size >= batch_size:
            agent.update(buffer.sample(batch_size))
```
