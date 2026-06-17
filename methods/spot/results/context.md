# Context: policy-constraint offline RL on continuous control (circa 2021-2022)

## Research question

We are handed a fixed dataset `D = {(s, a, r, s')}` collected by some unknown behavior policy
`pi_beta`, with no ability to interact with the environment, and asked to learn the best
possible continuous-control policy from it alone. The obvious move is to run an off-the-shelf
off-policy actor-critic (DDPG/TD3-style): fit `Q_theta(s, a)` by minimizing the Bellman error
against a bootstrapped target, and train a deterministic actor `pi_phi` to maximize
`Q_theta(s, pi_phi(s))`. This fails, and fails in a specific, diagnosable way. The Bellman
target `r + gamma * Q_target(s', pi(s'))` evaluates the critic at actions `pi(s')` that the
actor proposes — and nothing ties those actions to the data. When the actor proposes an action
the dataset never contained, `Q_theta` is being asked to extrapolate, and a function
approximator's extrapolation on an out-of-distribution (OOD) action is unconstrained; it can be
arbitrarily large. The actor, maximizing `Q`, is then *drawn toward* exactly those OOD actions
with erroneously high value, the inflated values get bootstrapped back through the Bellman
backup, and the error compounds over iterations until the value estimates and the policy
diverge. This is *extrapolation error*, and it is the central obstacle of learning from a fixed
batch.

The precise goal is therefore a continuous-control offline RL method that (1) keeps the learned
policy's actions inside the region where the behavior policy actually put mass — the *support*
of `pi_beta` — so the critic is not queried far off-distribution; (2) avoids replacing that support
condition with an over-restrictive imitation objective; (3) stays *pluggable* — implementable as a
small, non-intrusive modification on top of a standard off-policy algorithm, so it adds no
inference-time machinery beyond a single forward pass of the policy and can inherit improvements
from online RL; and (4) leaves the offline and online learning objectives close enough that an
offline-pretrained actor-critic can continue learning once fresh interaction becomes available.
Each existing family below achieves a subset; none achieves all four at once.

## Background

By this time the field state is well established. The MDP is `(S, A, rho_0, p, r, gamma)`; the
goal is to maximize `E_pi[sum_t gamma^t r(s_t, a_t)]`. The optimal value function `Q*(s,a)` is
the fixed point of the Bellman optimality operator `T Q(s,a) = E_{s'}[r + gamma max_{a'} Q(s',a')]`,
and for continuous actions the intractable `max_{a'}` is replaced by an actor: actor-critic
methods fit `Q_theta` by minimizing the mean-squared Bellman error
`E_D[(Q_theta(s,a) - r - gamma Q_target(s', pi_phi(s')))^2]` and update the actor by the
deterministic policy gradient (Silver et al. 2014), `min_phi E_s[-Q_theta(s, pi_phi(s))]`.

The diagnostic finding that organizes the whole field is the one above: applied naively to a
fixed batch, these off-policy methods produce overestimated `Q` on OOD actions and the policy
collapses (Fujimoto et al. 2019). The accepted cure is the **support constraint**: in state `s`,
only permit actions with `eps`-support under the behavior policy, i.e. actions in the set
`{a : pi_beta(a|s) > eps}`. Kumar et al. (2019) made this precise with a *distribution-constrained*
backup operator that restricts the maximization in the Bellman backup to a policy set `Pi`, and
proved a performance bound exposing a tradeoff: shrinking the allowed set toward the data reduces
the error term but enlarges a suboptimality term, while growing the set does the reverse — so the
support threshold is a single lever balancing "stay where the data is" against "keep enough room
to be optimal." Specialized to the support set `Pi_eps = {pi : pi(a|s)=0 whenever pi_beta(a|s)<eps}`,
the relevant statement is that the supported optimal value function `Q*_eps` is suboptimal by at
most `alpha(eps)/(1-gamma)`, where `alpha(eps) = max_{s,a}|T Q*(s,a) - T_eps Q*(s,a)|` measures how
much the support restriction perturbs a single backup. This result about the
support-constrained problem tells us that
constraining to the support costs only a controlled, contraction-amplified amount of optimality.

Two further pieces of background are load-bearing. First, **TD3** (Fujimoto et al. 2018)
hardened deterministic actor-critic against overestimation with three devices: *clipped
double-Q* (keep two critics, use `min` of the two in the target so an overestimating critic
cannot inflate the bootstrap), *target policy smoothing* (add clipped Gaussian noise to the
target action, `pi_target(s') + clip(noise)`, a SARSA-style smoothing that prevents the critic
from latching onto a sharp spurious peak), and *delayed policy updates* (update the actor and
targets only every `d` critic steps, so the actor chases a settled value estimate). These are
exactly the properties one wants when extrapolation error is the enemy, even off-line. Second,
**conditional variational autoencoders** (Kingma & Welling 2013; Sohn et al. 2015) are flexible
neural density models: a latent-variable model `p_psi(a|s) = ∫ p_psi(a|z,s) p(z|s) dz` with fixed
prior `p(z|s)=N(0,I)`, trained by maximizing the evidence lower bound (ELBO)
`log p_psi(a|s) >= E_{q_phi(z|a,s)}[log p_psi(a|z,s)] - KL(q_phi(z|a,s) || p(z|s)) =: -L_ELBO`.
The bound is loose by exactly `KL(q_phi(z|a,s) || p_psi(z|a,s)) >= 0`, and Burda et al. (2015)
showed an `L`-sample importance-weighted estimator
`log (1/L) sum_l p_psi(a, z_l|s)/q_phi(z_l|a,s)` that lower-bounds `log p_psi(a|s)` and tightens
monotonically as `L` grows. VAEs are already the standard behavior model in offline RL because
they capture near-arbitrary, multimodal action distributions, where a single Gaussian cannot.

## Baselines

These are the prior methods a new offline method is measured against and reacts to.

**BCQ (Fujimoto et al. 2019).** The first deep method to name and attack extrapolation error.
It fits a CVAE generative model of the behavior policy, then defines the policy implicitly: sample
`N` candidate actions from the VAE, apply a learned bounded perturbation `xi_phi(s, a_i)` to each,
and *act by argmax of `Q`* over the perturbed candidates,
`pi(s) = argmax_{a_i + xi_phi(s,a_i)} Q_theta(s, a_i + xi_phi(s,a_i))`, `a_i ~ VAE`. The value
update uses clipped double-Q. **Gap:** the constraint is welded into the policy parameterization,
so *every action selection* — at training and at deployment — requires sampling the generative
model, perturbing, and scoring with the critic. That is intrusive and slow at inference, couples
the algorithm tightly to its generative component, and muddies which piece
earns the performance.

**PLAS (Zhou et al. 2020) and EMaQ (Ghasemipour et al. 2021).** Variations on the same
parameterization theme. PLAS runs the policy in the *latent* space of the generative model and
decodes, `pi(s) = D_beta(pi_phi(s))`, so latent actions map to in-distribution actions by
construction. EMaQ strips BCQ's perturbation model and just takes `argmax_{a_i} Q(s, a_i)` over
VAE samples. **Gap:** same intrusive structure — the constraint lives in the architecture and the
inference path runs through secondary generative/critic components.

**BEAR (Kumar et al. 2019).** The first to argue for constraining *support* rather than the
full distribution: matching distributions is overly restrictive (if `pi_beta` is near-uniform,
matching it forces near-random behavior even when the data supports a strong policy), whereas one
only needs the learned policy's actions to lie *in the support* of `pi_beta`. It is implemented as
a *regularizer*: penalize the actor by the *sampled maximum mean discrepancy* (MMD) between actions
drawn from `pi_phi` and dataset actions, keeping it under a threshold via a Lagrange multiplier.
**Gap:** MMD is a distributional distance, not a density/support condition; BEAR's claim that it
constrains *support* rests on an empirically-observed property that, *at small sample counts*, the
sampled MMD between a distribution and a uniform distribution over its support can be lower than
between the distribution and itself. That is a fragile, sample-count-dependent surrogate; in
practice the constraint is loose and OOD actions leak through, and the method is unstable.

**TD3+BC (Fujimoto et al. 2021).** The minimalist regularization baseline: take TD3 and add a
single behavior-cloning term to the actor,
`max_phi E_{(s,a)~D}[lambda Q(s, pi(s)) - (pi(s) - a)^2]`, plus state normalization. To stop the
relative weight of the penalty from depending on the (reward-dependent) scale of `Q`, it folds a
normalization into the multiplier: `lambda = alpha / ((1/N) sum_i |Q(s_i, a_i)|)`, with `alpha=2.5`,
the mean absolute `Q` over the minibatch (detached, not differentiated). Pluggable, fast, one
line of change. **Gap:** the BC penalty `(pi(s) - a)^2` is again a *distributional* closeness to
the single logged action — it pulls the policy toward imitating the data point rather than toward
"any action the behavior policy would plausibly have taken," so on multimodal or suboptimal data
it over-constrains and cannot exploit the freedom the support genuinely allows.

So the prior art splits cleanly: the **parameterization** family (BCQ/PLAS/EMaQ) gets a genuine
density/support handle on the constraint but pays an intrusive, slow inference path; the
**regularization** family (BEAR/TD3+BC) is pluggable and fast but enforces a *divergence/imitation*
proxy that does not coincide with the support condition the theory actually asks for. Neither is
simultaneously pluggable *and* a direct match to the density-based support constraint.

## Evaluation settings

The natural yardsticks:

- **D4RL benchmark (Fu et al. 2020), `-v2` datasets.** Standard offline RL suite. Gym-MuJoCo
  locomotion (HalfCheetah, Hopper, Walker2d) in `medium`, `medium-replay`, and `medium-expert`
  variants — datasets of differing optimality, the `medium`/`medium-replay` ones being suboptimal
  and the natural stress test for whether a method can *improve over* the behavior policy rather
  than merely imitate it. The much harder **AntMaze** domains (umaze / medium / large, `play` and
  `diverse`) are sparse-reward navigation tasks requiring "stitching" of suboptimal trajectory
  fragments to reach a goal, so imitation-only objectives are a weak fit.
- **Metric:** D4RL normalized score (0 = random policy, 100 = expert), averaged over evaluation
  trajectories (e.g. 10 for Gym-MuJoCo, 100 for AntMaze) and reported over multiple seeds.
- **Continuous-control conventions:** states are commonly normalized to zero mean / unit variance
  on the locomotion tasks; for AntMaze a reward shift of `-1` is standard. The sparse-reward maze
  tasks stress whether a method can combine useful pieces of suboptimal trajectories rather than
  merely clone individual actions.
- **Offline-to-online protocol:** pretrain on the fixed dataset for a fixed budget of gradient
  steps, then continue learning with fresh environment interaction for a further budget of steps,
  adding collected transitions to the replay buffer and tracking the normalized score throughout
  both phases. The yardstick is whether the online phase improves substantially on the offline
  initialization without an early collapse.
- **Cost:** inference runtime (time to roll out a trajectory) and training time, since a
  selling point of pluggable methods is a single policy forward pass at action time.

## Code framework

A candidate solution can start from a standard offline-to-online actor-critic harness:
fixed-size MLP networks, Adam optimizers, a replay buffer, a twin-critic Bellman update with target
networks and soft (`tau`) target updates, a deterministic actor trained by the deterministic policy
gradient, and an outer loop that first consumes the fixed dataset and later, if permitted, appends
fresh interaction data. The unresolved part is the constraint mechanism; the scaffold keeps that
piece as a neutral slot rather than deciding whether it should be architectural, distributional,
density-based, pessimistic, or something else.

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Deterministic policy pi_phi(s) -> a in [-max_action, max_action]. 2x256 MLP."""
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, s):
        return self.max_action * self.net(s)


class Critic(nn.Module):
    """Twin Q(s, a) heads. Each 2x256 MLP returns (batch, 1)."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1_l1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_l2 = nn.Linear(256, 256)
        self.q1_l3 = nn.Linear(256, 1)
        self.q2_l1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_l2 = nn.Linear(256, 256)
        self.q2_l3 = nn.Linear(256, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)
        q2 = F.relu(self.q2_l1(sa))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)
        return q1, q2

    def Q1(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        return self.q1_l3(q1)


class AuxiliaryModule(nn.Module):
    """Optional module for whatever constraint mechanism is chosen."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # TODO: the auxiliary object, if the solution needs one
        pass

    def fit_step(self, s, a):
        # TODO: one fitting step, if needed
        pass

    def penalty(self, s, a):
        # TODO: per-sample actor penalty, if the solution uses one
        pass


def soft_update(target, source, tau):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


class OfflineOnlineAlgorithm:
    """Twin-critic deterministic actor-critic with target smoothing, delayed
    actor updates, soft target updates — the standard off-policy substrate.
    The mechanism that keeps the policy faithful to the data is left open."""

    def __init__(self, state_dim, action_dim, max_action, device="cuda",
                 discount=0.99, tau=5e-3, policy_noise=0.2, noise_clip=0.5,
                 policy_freq=2):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.auxiliary = AuxiliaryModule(state_dim, action_dim).to(device)  # may stay unused
        self.discount, self.tau = discount, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action
        self.total_it = 0

    def pretrain(self, replay_buffer, batch_size):
        # TODO: any fitting required before actor-critic training
        pass

    def train(self, batch, is_online=False):
        self.total_it += 1
        state, action, reward, next_state, done, *_ = batch
        not_done = 1 - done

        # Critic update: clipped double-Q target with target policy smoothing
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = (F.mse_loss(current_q1, target_q)
                       + F.mse_loss(current_q2, target_q))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed actor update
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critic.Q1(state, pi)
            # TODO: replace this placeholder with the chosen constrained actor objective.
            actor_loss = -q.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

    def on_online_start(self):
        # TODO: any transition adjustment when fresh online data starts arriving
        pass
```

A concrete solution has to fill the auxiliary slot, any required pretraining, the actor objective,
and any offline-to-online transition rule; everything else is the standard substrate.
