# ReBRAC, distilled

ReBRAC ("Revisited BRAC") is a minimalist offline RL algorithm for continuous control: TD3
with a **decoupled behavior-cloning penalty on both the actor loss and the critic bootstrap
target**, the TD3+BC value normalization on the actor, and a small set of revisited design
choices (LayerNorm in the critic, three hidden layers, larger batches with scaled learning
rate, an increased discount on long sparse-reward tasks). It is the deterministic-policy,
squared-distance instantiation of the behavior-regularized actor-critic framework with the two
penalty coefficients separated, applied on top of the minimalist TD3+BC baseline. No
secondary networks, no ensembles, no generative behavior model.

## Problem it solves

Learn the best policy from a fixed dataset `D` of `(s, a, r, s')` with no environment
interaction. Bootstrapped Q-learning overestimates `Q` at out-of-distribution next actions;
the maximizing policy chases the overestimate off-distribution; values diverge. ReBRAC
suppresses this OOD overestimation on both the acting policy and the critic's bootstrap while
keeping the method minimal and each design choice individually ablatable.

## Key idea

Inject a behavior penalty in the two places the behavior-regularized framework names, with the
divergence specialized to a deterministic policy as a squared distance (so no learned behavior
model is needed), and let the two strengths be **independent**:

- **Actor (policy regularization), `beta_1`** — penalize the policy for deviating from the
  logged action, with TD3+BC value normalization so one coefficient transfers across reward
  scales:
  `L_actor = mean( beta_1 (pi(s) - a)^2 - lambda Q(s, pi(s)) )`,
  `lambda = stopgrad( 1 / mean|Q(s, pi(s))| )` (when `normalize_q`).
- **Critic (value penalty), `beta_2`** — pessimize the clipped-double-Q bootstrap toward the
  dataset's own next action `a_hat'`:
  `a' = clip(pi_target(s') + clip(N(0, sigma), -c, c), -1, 1)` (target smoothing),
  `next_q = min_{i=1,2} Q_{theta'_i}(s', a') - beta_2 (a' - a_hat')^2`,
  `y = r + gamma (1 - done) next_q`,
  `L_critic = sum_i mean( (Q_{theta_i}(s, a) - y)^2 )`.

`beta_1` and `beta_2` are **decoupled** and tuned per environment (the original framework used
one shared coefficient). The actor penalty is the load-bearing one; the critic penalty
completes the two-sided regularization but contributes less on most tasks.

## Revisited design choices and why

- **LayerNorm in the critic only.** With LayerNorm before the output head `w`, the last hidden
  feature has bounded norm, so for any (including OOD) `(s,a)`:
  `|Q(s,a)| = |w^T relu(psi(s,a))| <= ||w|| ||relu(psi(s,a))|| <= ||w|| ||psi(s,a)|| <= ||w||`.
  The OOD value is capped by the head weight norm, killing the runaway extrapolation that
  feeds overestimation. The actor is left without inter-layer normalization (its tanh output
  is already bounded and `beta_1` pulls it to data).
- **Three hidden layers, width 256, ReLU.** Depth helps the value/policy fit on a large
  static dataset; the two-layer base is a holdover. Saturates around 3-5 layers; drops at 6.
- **Larger batch + scaled learning rate on dense locomotion.** Batch 1024, lr `1e-3` for
  Gym-MuJoCo (lower-variance gradients, faster convergence). Kept small (256, lr `1e-4` /
  `3e-4`) on sparse-reward / harder domains where larger batches hurt — a per-domain knob.
- **Discount.** `gamma = 0.999` on long sparse-reward tasks, else `0.99`. For a length-`L`
  episode with a single terminal reward, the signal reaches the start discounted by `gamma^L`:
  `0.99^1000 ~= 4e-5` (erased) vs `0.999^1000 ~= 0.37` (survives). Pure horizon arithmetic.
- **Drop state-feature normalization** (used by TD3+BC) to stay online-compatible at
  negligible offline cost.
- **Inherited from TD3, unchanged:** clipped double-Q (`min` over twins), target policy
  smoothing (`sigma = 0.2`, `c = 0.5`), delayed actor/target updates (`policy_freq = 2`), soft
  targets (`tau = 5e-3`), Adam.

## Defaults

`tau = 5e-3`, `policy_noise (sigma) = 0.2`, `noise_clip (c) = 0.5`, `policy_freq = 2`,
hidden dim 256, 3 hidden layers, ReLU, Adam. Per-env `(beta_1, beta_2)`, learning rate, batch
size, and `gamma` as above. (E.g. halfcheetah-medium-v2: `beta_1 = 0.001`, `beta_2 = 0.01`,
lr `1e-3`, batch 1024, `gamma = 0.99`.)

## Working code

Faithful to the canonical (CORL/JAX) implementation, written in the PyTorch shape of the
TD3+BC base it builds on. The dataset must store the next action `a_hat'` for the critic
penalty.

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicActor(nn.Module):
    """pi(s) = max_action * tanh(net(s)). 3 x 256, ReLU, no LayerNorm."""

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )

    def forward(self, state):
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state, device="cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().numpy().flatten()


class Critic(nn.Module):
    """Q(s, a). 3 x 256, ReLU, LayerNorm between layers (bounds OOD |Q|)."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class ReBRAC:
    def __init__(self, state_dim, action_dim, max_action,
                 actor_bc_coef=0.01, critic_bc_coef=0.01,   # beta_1, beta_2 (decoupled, per-env)
                 discount=0.99, tau=5e-3, lr=1e-3,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 normalize_q=True, device="cuda"):
        self.device = device
        self.beta_1, self.beta_2 = actor_bc_coef, critic_bc_coef
        self.discount, self.tau, self.max_action = discount, tau, max_action
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_freq, self.normalize_q = policy_freq, normalize_q
        self.total_it = 0

        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

    def train(self, batch):
        self.total_it += 1
        states, actions, rewards, next_states, dones, next_actions_data = batch
        not_done = 1.0 - dones.squeeze(-1)
        rewards = rewards.squeeze(-1)

        # ---- critic update (every step) ----
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action)
            next_q = torch.min(self.critic_1_target(next_states, next_actions),
                               self.critic_2_target(next_states, next_actions))
            bc_penalty = ((next_actions - next_actions_data) ** 2).sum(-1)   # value penalty
            next_q = next_q - self.beta_2 * bc_penalty
            target_q = rewards + not_done * self.discount * next_q

        critic_loss = (F.mse_loss(self.critic_1(states, actions), target_q)
                       + F.mse_loss(self.critic_2(states, actions), target_q))
        self.critic_1_opt.zero_grad(); self.critic_2_opt.zero_grad()
        critic_loss.backward()
        self.critic_1_opt.step(); self.critic_2_opt.step()

        # ---- delayed actor + target update ----
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            q = self.critic_1(states, pi)
            bc_mse = ((pi - actions) ** 2).sum(-1)                            # policy regularization
            lmbda = 1.0
            if self.normalize_q:
                lmbda = 1.0 / (q.abs().mean().detach() + 1e-8)                # TD3+BC normalization
            actor_loss = (self.beta_1 * bc_mse - lmbda * q).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            for net, tgt in ((self.critic_1, self.critic_1_target),
                             (self.critic_2, self.critic_2_target),
                             (self.actor, self.actor_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
```

## Relation to prior methods

- **TD3+BC** = ReBRAC with only the actor penalty (no critic value penalty), a single coupled
  coefficient with `alpha = 2.5`, two layers, no LayerNorm, batch 256, and added state
  normalization.
- **BRAC** = the general two-location framework (value penalty + policy regularization) with a
  *shared* coefficient, a stochastic policy, and a density divergence (KL/MMD/Wasserstein)
  against a learned behavior model. ReBRAC is its deterministic-policy, squared-distance,
  decoupled-coefficient specialization.
- **TD3** supplies the unchanged backbone: clipped double-Q, target policy smoothing, delayed
  updates, soft targets.
