# Context

## Research question

Every actor-critic algorithm carries two learned quantities — a policy `π` and a value function `V` —
and a practical implementation must decide whether to **share network parameters** between them or
keep **separate networks**. Both choices have a real cost.

Sharing has a clear upside: features learned for one objective help the other, and on high-dimensional
inputs (e.g. vision) a single representation that serves both is far more efficient than learning two.
But sharing forces two compromises. First, the two losses must be combined with a relative weight, and
however that weight is chosen, the gradient of one objective can **interfere** with the other —
optimizing the value can degrade the policy and vice versa. Second, a shared trunk all but requires
`π` and `V` to be trained on the **same data with the same sample reuse** (the same number of
optimization epochs per batch), which is an arbitrary coupling. Separate networks remove both problems
— no interference, independent sample reuse — but throw away the feature sharing entirely.

The question: can we **keep the feature sharing of a shared network while recovering the
non-interference and the independent sample reuse of separate networks**? Two facts about on-policy
actor-critic make this worth chasing. (1) When parameters are shared, policy↔value interference
measurably hurts performance — yet on high-dimensional benchmarks sharing is also necessary, so simply
separating the networks is not a free fix. (2) Value-function optimization tolerates a far higher
level of sample reuse than policy optimization does; much of the benefit that on-policy methods get
from extra epochs over a batch is really the *value function* being better trained, not the policy.

## Background

**On-policy actor-critic and PPO.** PPO (Schulman et al. 2017) is the standard on-policy method. It
trains the policy with the clipped surrogate
`L^clip = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1−ε, 1+ε)Â_t)]`, where `r_t(θ) = π_θ(a_t|s_t)/π_old(a_t|s_t)`
and `Â_t` is an advantage estimate; in practice it optimizes `L^clip + β_S·S[π]` with an entropy bonus
`S`. It trains the value with `L^value = Ê_t[½(V(s_t) − V̂^targ_t)²]`. Both `Â` and `V̂^targ` come from
GAE (Schulman et al. 2016): `Â_t = Σ_l (γλ)^l δ_{t+l}` with `δ_t = r_t + γV(s_{t+1}) − V(s_t)`, and
`V̂^targ_t = V(s_t) + Â_t`. PPO also proposed an adaptive-KL-penalty variant as an alternative to
clipping. The policy objective is largely interchangeable across the actor-critic family — TRPO
(trust-region step), ACKTR (K-FAC trust region), V-MPO and AWR (move the policy toward an
exponentiated-advantage reweighting of actions) — and any of them could in principle fill the policy
slot.

**The value function as a representation learner / auxiliary task.** A line of work treats value
prediction as a source of useful features. Jaderberg et al. (2017, UNREAL) add auxiliary prediction
tasks to improve representations; Bellemare et al. (2019) study adversarial value functions as a
representation-learning auxiliary; Lyle et al. (2019) argue the gains of distributional RL
(Bellemare et al. 2017) come largely from the richer value-as-auxiliary signal. The common thread:
fitting value targets shapes a network's features in ways that help control.

**Distillation and behavioral cloning.** A behavioral-cloning / distillation loss `KL(π_old ‖ π_θ)`
forces `π_θ` to match a reference policy's outputs; minimizing it lets a network's *representation*
change while its *policy* is held nearly fixed. ITER (Igl et al. 2020) alternates a standard RL phase
with a distillation phase — periodically distilling the policy and value teachers into freshly
initialized student networks to reduce the impact of non-stationarity and improve generalization —
establishing the pattern of interleaving RL with a separate distillation phase.

**Off-policy replay for sample efficiency.** SAC (Haarnoja et al. 2018), DDPG (Lillicrap et al. 2015),
and ACER (Wang et al. 2017) use replay buffers to reuse data via off-policy updates; SAC also uses
*separate* policy and value networks, to avoid the very interference described above.

## Baselines

**PPO with a shared network (Cobbe et al. 2020 tuning).** The standard, highly-tuned baseline: one
shared trunk feeding a policy head and a value head, trained with `L^clip + β_S·S[π] + vfcoef·L^value`,
GAE advantages, and a single sample-reuse setting (epochs per batch) applied to both objectives. A
hyperparameter sweep is used to pick a near-optimal value-loss weight and a near-optimal number of
epochs. Gap: the value-loss weight is a forced compromise that risks policy↔value interference, and a
single epochs-per-batch setting couples policy and value sample reuse — empirically the method's
apparent need for several epochs is the value function being under-trained, not the policy.

**PPO with separate networks.** Removes the interference and the weight, but on high-dimensional
benchmarks performs *worse* than the shared baseline because it loses feature sharing. Gap: gives up
the representation sharing that is critical on vision inputs.

**On-policy alternatives — TRPO, ACKTR, A3C/IMPALA, V-MPO, AWR.** Different policy objectives within
the same actor-critic frame; all face the identical shared-vs-separate decision and the same coupled
sample reuse. Gap: none of them decouples policy and value training while preserving shared features.

## Evaluation settings

The benchmark is Procgen (Cobbe et al. 2020): 16 procedurally generated, visually diverse 2D games
with image observations, designed so that improvements are expected to transfer broadly and so that
shared visual representations matter. The protocol: train for a fixed total of environment timesteps
(on the order of 10⁸) across parallel vectorized workers, with reward normalization so that discounted
returns have roughly unit variance, and report sample efficiency (return as a function of timesteps),
typically averaged over a few seeds with standard deviation across runs. Natural ablation axes — for
isolating the two observations above — are the level of policy sample reuse, the level of value sample
reuse, the frequency of any auxiliary optimization, and the choice of clipping versus a KL penalty.

## Code framework

The primitives that already exist: a (possibly shared) convolutional encoder feeding a policy head and
a value head, GAE, a PPO clipped-surrogate policy update with an entropy bonus, a value-regression
loss, Adam, and an on-policy rollout loop over vectorized environments. What does not yet exist is any
mechanism to train the policy and value with *different* sample reuse while keeping shared features, or
any separate phase of optimization beyond the PPO update. Those are the stubs.

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Convolutional trunk producing a feature vector from an observation."""
    def __init__(self, ob_space, outsize=256):
        super().__init__()
        # TODO: conv stack (e.g. an Impala-style ResNet encoder)
        pass
    def forward(self, ob):
        pass


class ActorCritic(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        # TODO: encoder(s) + a policy head + a value head
        #       (whether they share a trunk is exactly the open question)
        pass
    def forward(self, ob):
        # TODO: return (action distribution, value prediction)
        pass


def compute_gae(rewards, values, dones, gamma, lam):
    # advantage Â_t = Σ (γλ)^l δ_{t+l}, δ_t = r_t + γ V(s_{t+1}) − V(s_t)
    # value target V̂^targ_t = V(s_t) + Â_t
    ...


def ppo_losses(model, batch, clip_param, ent_coef, vf_coef):
    # clipped surrogate + entropy bonus + value MSE
    ...


def policy_phase(model, rollouts, opt, hp):
    # TODO: PPO update — but with what sample reuse for policy vs value,
    #       and on shared or separate parameters?
    pass


def auxiliary_phase(model, buffer, opt, hp):
    # TODO: a separate optimization phase (currently nonexistent) that could
    #       train value features without disturbing the policy
    pass


def train(venv, model, hp):
    # TODO: the outer loop — what alternates, and how often?
    pass
```
