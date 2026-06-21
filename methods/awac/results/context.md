# Context

## Research question

In computer vision and NLP the dominant recipe is to pre-train on a large, general dataset and then
fine-tune on the task at hand. Reinforcement learning typically collects data online from scratch
every time, which is costly wherever real interaction is expensive — robotics above all. The goal is
an RL method that can **pre-train from a fixed offline dataset and then keep improving with a small
amount of online interaction**.

The offline dataset is arbitrary: it may be expert demonstrations, suboptimal trajectories, data from
related tasks, or pure random exploration, and the method should not treat any one type in a
privileged way. The question is how a single algorithm can pre-train on such a static dataset and
then fine-tune with a small online budget, using the same update throughout rather than choosing in
advance between "offline only" and "online only".

## Background

**Actor-critic and policy iteration.** Actor-critic methods (Konda & Tsitsiklis 2000) alternate
*policy evaluation* — estimate the critic `Q^π` for the current policy by repeatedly applying the
Bellman operator `B^π Q(s,a) = r(s,a) + γ E_{s'}E_{a'~π}[Q(s',a')]`, in practice minimizing the
Bellman error `E_D[(Q_φ(s,a) − y)²]` with `y = r + γ E_{s',a'}[Q_{φ̄}(s',a')]` against a target network
— and *policy improvement* — update the actor to maximize `E_{s~D}[E_{a~π_θ}[Q_φ(s,a)]]`. With a
replay buffer this is off-policy and can reuse stored transitions, which is what makes it
sample-efficient. The advantage `A^π(s,a) = Q^π(s,a) − V^π(s)` with `V^π(s) = E_{a~π}[Q^π(s,a)]`
measures how much better an action is than the policy's average; since `V` does not depend on the
action, maximizing `E_π[Q]` is the same as maximizing `E_π[A]`.

**Bootstrapping in off-policy actor-critic.** The improvement step samples actions *from the policy*
and bootstraps the TD target at actions `a' ~ π`. When `a'` falls outside the dataset distribution,
`Q_{φ̄}(s', a')` is an extrapolation, and the error is backed up through the Bellman recursion; the
actor in turn maximizes the estimate. This bootstrap behavior is well documented for off-policy
actor-critic trained on static data.

**Maximum-likelihood / EM views of policy improvement.** A line of work (RWR, Peters & Schaal 2007;
REPS, Peters et al. 2010; MARWIL, Wang et al. 2018; MPO, Abdolmaleki et al. 2018; AWR, Peng et al.
2019) casts policy improvement as a KL-constrained optimization, `max_π E_{a~π}[A(s,a)]` subject to
`KL(π‖π_β) ≤ ε`. Enforcing the KKT conditions gives the closed-form non-parametric optimum
`π*(a|s) ∝ π_β(a|s)·exp(A(s,a)/λ)`, and projecting it onto a parametric policy yields an
advantage-weighted maximum-likelihood (supervised) update. These methods differ in how they get the
advantage and how they handle the projection.

**Overestimation control.** Twin Q-functions with a minimum target (Fujimoto et al. 2018, TD3) and a
Polyak-averaged target network (Mnih et al. 2015) are standard fixes for the upward bias of
bootstrapped value learning; SAC (Haarnoja et al. 2018) is the standard off-policy actor-critic for
continuous control built on them.

## Baselines

**Soft actor-critic, off-policy (SAC, with prior data in the buffer).** A standard off-policy
continuous-control actor-critic: twin critics and an entropy-regularized stochastic actor. The offline
data is used by simply placing it in the replay buffer; the policy can optionally be BC-pretrained
before learning begins.

**Explicit-constraint offline RL — BCQ, BEAR, BRAC, ABM.** These add the constraint `D(π_θ,π_β) ≤ ε`
by *fitting an explicit parametric behavior model* `π̂_β` from the data by maximum likelihood, then
using it as a penalty (BRAC), a sampler (BCQ, ABM), or a support constraint (BEAR).

**Advantage-weighted regression (AWR, Peng et al. 2019) and related MC methods (MARWIL, RWR).** Apply
the advantage-weighted maximum-likelihood update, estimating the value function of the *behavior*
policy `V^{π_β}` by Monte-Carlo returns or TD(λ).

**On-policy fine-tuning from demonstrations (DAPG, Rajeswaran et al. 2018).** BC-initialize, then
fine-tune with an on-policy policy gradient.

## Evaluation settings

The natural testbeds are challenging continuous-control robotics tasks where exploration is hard and
prior data is genuinely useful: sparse-reward dexterous manipulation in MuJoCo (a 28-DoF hand spinning
a pen, opening a door, relocating a ball) with binary task-completion rewards and offline data made of
a handful of human demonstrations plus trajectories from a behavior-cloned policy; a simulated Sawyer
tabletop-pushing task with a dataset of random-process trajectories (to test learning from suboptimal
data); the standard Gym MuJoCo locomotion benchmarks (halfcheetah, hopper, walker2d, ant) with
demonstrations as offline data; and real-world robots (a 3-fingered claw rotating a valve, a
4-fingered hand repositioning an object, a Sawyer arm opening a drawer). The protocol: pre-train on
the offline dataset for a fixed number of gradient steps, then run online fine-tuning, plotting
performance against the number of online timesteps and reading off both the offline-only starting
point and the slope of online improvement. A controlled diagnostic on HalfCheetah-v2 (15 expert demos
+ 100 behavior-cloned suboptimal trajectories) provides a small, fixed setting for comparing methods.

## Code framework

The primitives that already exist: an MLP builder, twin state-action critics with target copies, a
diagonal Gaussian stochastic policy whose mean is bounded with `tanh`, Adam, a Polyak target-update,
a replay buffer shared between offline data and online transitions, and an off-policy training loop. What does not yet exist is the
policy-improvement step itself. That is the stub.

```python
import copy
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__(); self.net = mlp([obs_dim + act_dim, *hidden, 1])
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden)
        self.q2 = Critic(obs_dim, act_dim, hidden)
    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__()
        self.trunk = mlp([obs_dim, *hidden])
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
    def dist(self, s):
        h = self.trunk(s)
        return Normal(torch.tanh(self.mu(h)), self.log_std(h).clamp(-6, 0).exp())
    def log_prob(self, s, a):
        return self.dist(s).log_prob(a).sum(-1)
    def sample(self, s):
        return self.dist(s).rsample()


def critic_td_loss(batch, critic, target_critic, policy, discount):
    # TODO: twin-Q Bellman target with the next action from the current policy
    pass


def policy_improvement_loss(batch, critic, policy, hp):
    # TODO: the policy-improvement objective
    pass


def update(batch, critic, target_critic, policy, opts, hp):
    # TODO: critic TD step, then actor step, then Polyak target sync
    pass
```
