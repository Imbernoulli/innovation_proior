# Context: offline reinforcement learning for continuous control (circa 2020-2021)

## Research question

We want to learn a control policy from a fixed, previously collected dataset of transitions
`D = {(s, a, r, s')}`, with no further interaction with the environment. This is the offline
(historically "batch") setting, and it matters because in robotics, healthcare, and other
real-world domains, collecting fresh data is expensive, slow, or unsafe, while logged data
already exists. The state and action spaces are continuous vectors (joint torques, not a
handful of discrete moves), so the policy is a function `π: S → A` and the value function a
critic `Q(s, a)` over a continuous action argument.

The difficulty is specific. Off-policy actor-critic methods are in principle applicable
offline — they already learn from a replay buffer — but they degrade badly when that buffer is
frozen. The cause is a documented failure mode: the critic is asked to evaluate state-action
pairs whose actions are not represented in `D`, where its output is an unconstrained
extrapolation; because temporal-difference learning bootstraps each target from the next
state's estimate, that error is backed up through the Bellman recursion, and because the actor
is moved to *increase* the critic's value, the policy is pulled toward exactly the actions the
critic happens to over-rate. Online, fresh interaction would refute an over-valued action;
offline there is no such correction, so the error compounds. A solution would have to keep the
learned policy's actions close enough to the data that the critic is only ever queried where it
has support, *without* sacrificing the value-based exploitation that lets a policy improve on
the data-generating behavior — and it must do so under a sharp practical constraint particular
to the offline setting: every additional component or hyperparameter is far more expensive than
online, because there is no environment to validate it against. The yardstick is a method that
is robust across datasets of very different reward scales and data quality, with as little
added machinery and tuning as possible.

## Background

**MDPs, value functions, and off-policy actor-critic.** A task is a Markov decision process
`(S, A, R, p, γ)`; the objective is the expected discounted return `E_π[Σ_t γ^t r_{t+1}]`,
measured by the action-value `Q^π(s, a) = E_π[Σ_t γ^t r_{t+1} | s_0=s, a_0=a]`, which satisfies
the Bellman equation `Q^π(s,a) = r + γ E_{s',a'}[Q^π(s',a')]`. In continuous action spaces one
cannot enumerate `max_a Q`; the deterministic policy gradient (Silver et al. 2014) instead
moves a deterministic actor in the direction the critic says value increases,
`∇_φ J(φ) = E_s[ ∇_a Q(s,a)|_{a=π_φ(s)} · ∇_φ π_φ(s) ]`. The critic supplies the gradient and
the actor performs a soft, local maximization of `Q`. With a deep critic fit by TD regression,
a replay buffer, and slowly-updated target networks, this is off-policy and sample-efficient.

**Extrapolation error in the batch setting.** When the replay buffer is a fixed dataset
collected by some other (possibly unknown) process, the improvement step samples actions from
the current policy and the critic target bootstraps at `a' = π(s')`. Whenever `a'` falls
outside the dataset's action distribution, `Q(s', a')` is extrapolated with no data to ground
it; the error propagates through the Bellman recursion, and the actor — maximizing the estimate
— steers toward the over-valued out-of-distribution actions (Fujimoto et al. 2019). This was
measured directly: dropping a fixed dataset into a strong off-policy learner yields performance
no better than learning from scratch, because the learner cannot safely use data uncorrelated
with its own policy. The accepted diagnosis is that the learned policy must be kept close to the
data-generating ("behavior") policy so the critic is only queried where it has support; the
solution family carries many names — batch-constrained, KL-control, behavior-regularized,
policy-constraint — distinguished by *how* that closeness is imposed.

**Behavior cloning.** The other way to get a policy from a dataset is pure imitation: fit
`π(s) ≈ a` by supervised regression on `D` (Pomerleau 1991). Behavior cloning needs no value
function and never queries out-of-distribution actions, so it cannot suffer extrapolation
error — but it is ceilinged by the data, copying good and bad actions alike, and cannot improve
on the behavior that produced the dataset.

**Diagnostic findings about existing offline systems.** Two observations
about the state of the art frame the problem. First, the recent strong offline methods are not,
on inspection, "simple": each modifies an underlying online algorithm (TD3 or SAC) not only with
its named algorithmic idea but with a stack of unannounced implementation changes — altered
network architectures, separate actor learning rates, actor pre-training phases, removal of the
entropy term, reward bonuses, max-over-sampled-actions at evaluation. When those implementation
adjustments are stripped back to the base algorithm, measured performance drops sharply across
many datasets — i.e. the headline results lean heavily on the un-justified extras, which are
hard to attribute and, lacking a train/validation split, plausibly tuned on the evaluation data
itself. These extras also more than double wall-clock training time relative to the base
algorithm (logsumexp over many sampled actions; training a separate generative model). Second,
offline-trained policies examined show large variance in return both
within a single evaluation and across nearby evaluation checkpoints, in contrast to online
policies that converge to low-variance behavior; this appears tied to distributional shift and
poor generalization to unobserved states, and is a property of the offline setting rather than
of any one algorithm.

## Baselines

These are the prior methods a new offline algorithm would be measured against and react to.

**Behavior cloning (Pomerleau 1991).** Supervised regression `min_π E_{(s,a)~D}[(π(s) - a)^2]`
(or a likelihood for stochastic policies). Simple, value-free, no out-of-distribution queries.
**Gap:** it imitates the dataset wholesale and cannot exceed the behavior that generated it; on
mixed or suboptimal data it copies the bad actions along with the good.

**Online off-policy actor-critic — DDPG (Lillicrap et al. 2015) and TD3 (Fujimoto et al. 2018).**
DDPG couples a deterministic actor with a deep critic, a replay buffer, and target networks.
TD3 hardens it against the function-approximation pathologies of continuous-control value
learning with three pieces. *Clipped double-Q:* keep twin critics `Q_{θ1}, Q_{θ2}` and build
the target from the smaller, `y = r + γ min_{i=1,2} Q_{θ'_i}(s', ã)`, so the target can never
introduce overestimation beyond the standard Q-learning target. *Target policy smoothing:*
perturb the target action, `ã = π_{φ'}(s') + ε`, `ε ~ clip(N(0,σ), -c, c)` with
`σ=0.2·a_max`, `c=0.5·a_max`, so the critic fits the value of a small region around the
target action rather than a narrow peak — a SARSA-flavored regularizer. *Delayed policy
updates:* update the actor and the soft target networks (`τ=5e-3`) only every `d=2` critic
steps, letting the value estimate settle relative to the slowly moving policy. Architecture is
`256×256` ReLU MLPs, Adam at `3e-4`, batch 256, `γ=0.99`. **Gap (offline):** designed for
online interaction, it has no mechanism to keep the actor inside the data distribution, so on a
frozen dataset its critic extrapolates at the actions the actor selects and the policy collapses
onto over-valued out-of-distribution actions.

**Batch-constrained Q-learning — BCQ (Fujimoto et al. 2019).** The work that named
extrapolation error. It restricts the actor to actions close to the data by fitting a
conditional variational autoencoder as a generative model of the behavior policy, sampling
candidate actions from it, perturbing them within a small range, and selecting among them with
the critic. **Gap:** it depends on an accurate generative model of the behavior policy and adds
that model plus a perturbation network and their hyperparameters; the constraint is only as good
as the density estimate.

**Behavior-regularized policy optimization — BEAR, BRAC (Kumar et al. 2019; Wu et al. 2019).**
Fit an explicit parametric behavior model `π̂_β` by maximum likelihood and constrain the actor
toward it — as a divergence penalty `D(π_θ, π̂_β)` (KL in BRAC), or a support constraint via
MMD (BEAR). **Gap:** the constraint hinges on a fitted `π̂_β`; an inaccurate behavior model
makes the constraint either leaky or over-conservative, and the fit, its divergence choice, and
its strength are extra moving parts to tune without an environment to check them.

**Conservative critic — CQL (Kumar et al. 2020).** Rather than constrain the policy, regularize
the *critic*: add a term that pushes down `Q` on out-of-distribution actions while pushing it up
on dataset actions, so the value function itself becomes pessimistic about unsupported actions.
**Gap:** the conservative term is approximated with a logsumexp over many sampled actions
(extra compute), and the reported results also rely on an actor pre-training phase, a
max-over-sampled-actions evaluation rule, removal of the SAC entropy term, and modified
architecture/learning-rate — a substantial stack of adjustments beyond the named idea.

**Behavior-model-with-offset — Fisher-BRC (Kostrikov et al. 2021).** State-of-the-art at the
time: train a separate generative behavior model, reparameterize the critic as that model's log
density plus a learned offset, regularize the offset with a Fisher-divergence gradient penalty,
and add a constant reward bonus. **Gap:** strong, but it carries a generative model, an offset
network, a gradient-penalty term, a reward bonus, and the entropy-removal/architecture changes —
many components and hyperparameters, and roughly double the base algorithm's runtime.

**Advantage-weighted regression — AWR/AWAC (Peng et al. 2019; Nair et al. 2020).** Extract a
policy by weighted behavior cloning, weighting each dataset action by `exp(A(s,a)/β)` so the
policy imitates the high-advantage actions more strongly. Stays inside the data by construction.
**Gap:** the policy is still pinned to a (weighted) imitation objective with an advantage
estimate and a temperature `β` to set, and the weighting can be conservative on broad datasets.

## Evaluation settings

The standard yardstick is the D4RL benchmark (Fu et al. 2021) of offline datasets for OpenAI
Gym MuJoCo continuous-control tasks (Todorov et al. 2012; Brockman et al. 2016): HalfCheetah,
Hopper, Walker2d, and Ant, each with continuous state and action vectors and the environment's
native reward. For each task, datasets of differing quality are provided — `random` (a random
policy), `medium` (a partially-trained policy), `medium-replay` (the replay buffer of training
up to medium), `medium-expert` (a mix), and `expert`. The metric is the D4RL normalized score,
scaled so that 0 is a random policy and 100 is an expert policy. The protocol: train for one
million gradient steps, evaluate periodically (every 5000 steps) by averaging undiscounted
return over 10 episodes with exploration noise off, and report the average over the final
evaluations across several random seeds. Wall-clock training time on a single GPU is a secondary
axis of comparison, since avoiding added compute is part of the goal. Comparisons are run from a
common framework with author-provided implementations re-run where possible for an identical
evaluation.

## Code framework

The primitives below already exist: a deep-learning library with autodiff and Adam, MLP layers,
an MSE loss, a replay buffer that samples minibatches of `(state, action, next_state, reward,
not_done)` from the fixed dataset, and the standard TD3 actor-critic machinery (deterministic
actor with target, twin critics with targets, clipped double-Q target with target policy
smoothing, delayed actor and soft target updates). What is *not* settled is how — if at all —
the policy is kept faithful to the dataset, and whether the raw dataset features are used as-is;
those slots are left open.

```python
import copy, torch, torch.nn as nn, torch.nn.functional as F


class Actor(nn.Module):
    # deterministic policy s -> a, tanh-squashed to the action range
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, s):
        a = F.relu(self.l1(s)); a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    # twin state-action value estimators (clipped double-Q)
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256); self.l2 = nn.Linear(256, 256); self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 256); self.l5 = nn.Linear(256, 256); self.l6 = nn.Linear(256, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        q1 = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
        q2 = self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
        return q1, q2

    def Q1(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))


def prepare_dataset(replay_buffer):
    # The dataset is fixed and given up front.
    # TODO: any preprocessing of the dataset features we decide to do.
    return replay_buffer


class Agent:
    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.max_action, self.discount, self.tau = max_action, discount, tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.total_it = 0

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            # The plain off-policy actor objective maximizes the critic over the policy's actions.
            # TODO: the actor objective we will use here, given a fixed dataset.
            actor_loss = None
            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
            # soft target updates theta' <- tau*theta + (1-tau)*theta'
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
```

The two open slots are the dataset-preprocessing step and the actor objective.
