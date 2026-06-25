# FOCAL, distilled

FOCAL (Fully-Offline Context-based Actor-critic meta-RL) is an end-to-end, model-free algorithm for
offline meta-reinforcement learning: adapt to unseen tasks from static, pre-collected data with no
environment interaction at train or test time. It pairs a **deterministic, mean-aggregating context
encoder** trained by a **negative-power distance metric learning (DML) loss** with a
**behavior-regularized SAC** actor-critic, and **decouples the encoder from the value learning at the
gradient level** (the task variable `z` is detached when it feeds the actor and critic).

## Problem it solves

Offline meta-RL has two coupled difficulties. (1) Offline RL bootstrapping error: the value of
out-of-distribution `(s,a)` is never corrected, so backups over the learned policy's OOD actions
diverge. (2) Task inference with no exploration: encode a few logged transitions into a task variable,
condition control on it, and learn the encoder jointly with control — robustly and from few samples.

## Key ideas

**Deterministic encoder from task-transition correspondence.** For deterministic MDPs satisfying
task-transition correspondence — `P_1(·|s,a)=P_2(·|s,a)` and `R_1(s,a)=R_2(s,a) ⟺ T_1=T_2` — the pair
`(P,R)` identifies the task pointwise, so a logged transition can in principle reveal task identity
under that assumption. The
encoder should therefore be **permutation-invariant** (use the mean of per-transition embeddings) and
**deterministic** (no information bottleneck, no product-of-Gaussians): there is no irreducible
uncertainty to model and no exploration that probabilistic posterior-sampling would serve.

**Why separation is necessary (continuity).** A value network `Q_θ(s,a,z)` is continuous: close `z`
forces close output `Q`. Two distinct tasks have genuinely different true `Q` values, so if their
embeddings are close the conditioned value functions become unrepresentable. Distinct tasks' `z` must
be pushed apart for control to be learnable — task inference geometry is a precondition, not a luxury.

**The inverse-power DML loss (the central new piece).** The classic contrastive loss
`1{same}‖q_i−q_j‖² + 1{diff} max(0, m−‖q_i−q_j‖)²` degenerates: the margin spring has bounded
short-range force and is zero beyond `m`; the deeper failure is that a squared-distance objective is
*exactly* variance maximization, `Σ_{i≠j}(x_i−x_j)² = 2N²·Var(X)`, a global statistic that a degenerate
(e.g. Bernoulli) distribution can maximize while merging several tasks into one pile. Replace the
repulsive term with a **negative power** of distance:
```
L_dml(x_i, x_j; q) = 1{y_i=y_j} ‖q_i−q_j‖₂²  +  1{y_i≠y_j} · β / ( ‖q_i−q_j‖₂ⁿ + ε ),   n ≥ 0, ε > 0.
```
For `n>0` this is an epsilon-capped short-range repulsive penalty: the unregularized `1/d^n`
intuition is Coulomb-like, but `ε` makes the value finite at zero. The `n=0` case is degenerate
because the different-task term is the constant `β/(1+ε)` and gives no separation gradient. The useful
cases are `n=1` (inverse) and `n=2` (inverse-square); `n=2` is analogous to Cauchy graph
embedding. With the latent bounded by tanh to `(−1,1)^l`, the inverse-power terms directly penalize
close different-task pairs instead of only increasing a global variance statistic. `q` is the **mean**
transition embedding of a task; the repository implements the inverse-square case with a
dimension-normalized squared distance.

**Gradient-level decoupling.** Offline value functions can reach huge magnitudes; if the encoder also
received Bellman gradients they would swamp the DML gradient and the embedding would collapse. So the
encoder is trained **only** by `L_dml`, and `z` is **detached** when consumed by the actor and critic.
(Unlike PEARL, which decoupled by *data sampling* to support exploration — unnecessary offline.)

**Behavior-regularized control (BRAC).** A task-augmented behavior-regularized SAC handles
bootstrapping: a divergence `D(π_θ, π_b)` enters the critic target (value penalty) and/or actor
(policy regularization). `D` is a **dual-form KL** estimated by a learned discriminator `g`
(minimax + gradient penalty), avoiding a cloned behavior density. The max-entropy term is kept (helps
when distinct actions give the same `(s',r)`). `α` is tuned per environment (adaptively, vs a
divergence threshold), with a wide spread (0 on easy tasks up to `~10⁶` on Ant-Fwd-Back).

## Algorithm

Three objectives, three optimizers, updated separately each step:
```
φ ← φ − α₁ ∇_φ Σ_{ij} L_dml(c_i, c_j)        # encoder: distance metric loss only
θ ← θ − α₂ ∇_θ Σ_i  L_actor(b_i, z̄)          # actor:  max E[Q(s,z̄,a)] − α D̂,  z detached
ψ ← ψ − α₃ ∇_ψ Σ_i  L_critic(b_i, z̄)         # critic: (r + γ Q̄^D(s',z̄,a') − Q(s,z̄,a))²
```
At test time on a held-out task: draw logged transitions as context, `z = mean(encoder(context))`,
roll out `π_θ(a|s,z)` deterministically — no exploration.

## Defaults and why

- **Latent dim** is 5 or 20: Sparse-Point-Robot and Half-Cheetah-Fwd-Back use
  5, while Half-Cheetah-Vel and Ant-Fwd-Back use 20; the power-law ablation reduces Half-Cheetah-Vel
  to 5 for speed.
- **Networks**: the main configuration uses a width-200 depth-3 context encoder and width-300 depth-3 policy,
  Q, V, and discriminator networks; the repository default is width 256 but the launcher builds the
  context encoder as `[200, 200, 200]` with tanh output.
- **DML constants**: Eq. 13 has coefficient `β`; the main configuration uses `β=1`, while the power-law
  ablation uses `(β, ε) = (1, 0.1)` for inverse-square, `(2, 0.1)` for inverse, `(8, 0.1)` for linear,
  and `(16, 0.1)` for square. The repository's actual `z_loss` has a separate whole-loss multiplier
  `z_loss_weight` (default 10) and implements `1 / (mean(diff²) + 100ε)`.
- **Control constants**: reward scale is 100 for the point-robot tasks and 5 for MuJoCo locomotion;
  discount is 0.9 for point-robot tasks and 0.99 otherwise; soft target `τ=0.005`; repo default
  policy/Q/V/context learning rates are `3e-4` and discriminator learning rate is `1e-4`.
- **Behavior reg `α`**: the main configuration reports 0 for Sparse-Point-Robot, 500 for
  Half-Cheetah-Vel, `10^6` for Ant-Fwd-Back, and 500 for Half-Cheetah-Fwd-Back. The repository JSONs
  initialize 50 for Half-Cheetah-Vel and Walker-2D-Params, disable BRAC for Point-Robot-Wind, and use
  adaptive clipping through `alpha_max`.

## Working code

Faithful to the canonical implementation: a tanh-bounded deterministic mean-aggregating encoder, the
repository's inverse-square `z_loss` variant of Eq. 13, and a SAC + dual-form-KL-BRAC update where
critic/value/policy prediction paths use detached `z` when `allow_backward_z=False` and target values
are evaluated under `torch.no_grad()`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FOCALAgent(nn.Module):
    """Canonical shape: PEARLAgent with IB disabled, tanh encoder, mean z."""

    def __init__(self, latent_dim, context_encoder, policy, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_encoder = context_encoder
        self.policy = policy
        self.use_ib = kwargs.get('use_information_bottleneck', False)
        self.recurrent = kwargs.get('recurrent', False)
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

    def clear_z(self, num_tasks=1):
        self.z_means = self.z.new_zeros(num_tasks, self.latent_dim)
        self.z_vars = self.z.new_zeros(num_tasks, self.latent_dim)
        self.z = self.z_means

    def detach_z(self):
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def infer_posterior(self, context, task_indices=None):
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        if self.use_ib:
            raise NotImplementedError("FOCAL disables the PEARL information bottleneck.")
        self.z_means = torch.mean(params, dim=1)
        self.z_vars = torch.std(params, dim=1)
        self.z = self.z_means

    def forward(self, obs, context, task_indices=None):
        self.infer_posterior(context, task_indices=task_indices)
        t, b, _ = obs.size()
        obs_flat = obs.view(t * b, -1)
        task_z = torch.cat([z.repeat(b, 1) for z in self.z], dim=0)
        in_ = torch.cat([obs_flat, task_z], dim=1)
        policy_outputs = self.policy(t, b, in_, reparameterize=True, return_log_prob=True)
        task_z_vars = torch.cat([z.repeat(b, 1) for z in self.z_vars], dim=0)
        return policy_outputs, task_z, task_z_vars


def repo_z_loss(task_z, indices, b, epsilon=1e-3):
    """Exact repository z_loss: RMSE attraction plus inverse-square repulsion."""
    pos_loss, neg_loss = 0.0, 0.0
    pos_cnt, neg_cnt = 0, 0
    for i in range(len(indices)):
        zi = task_z[i * b]
        for j in range(i + 1, len(indices)):
            zj = task_z[j * b]
            d_sq = torch.mean((zi - zj) ** 2)                  # mean squared diff / dim
            if indices[i] == indices[j]:                       # same task -> attract
                pos_loss += torch.sqrt(d_sq + epsilon)
                pos_cnt += 1
            else:                                              # different task -> repel
                neg_loss += 1.0 / (d_sq + epsilon * 100)
                neg_cnt += 1
    return pos_loss / (pos_cnt + epsilon) + neg_loss / (neg_cnt + epsilon)


class FOCALAlgorithm:
    """Decoupled training: encoder by z_loss only; SAC + BRAC on detached z."""

    def __init__(self, agent, env, train_tasks, replay_buffer, enc_replay_buffer,
                 qf1, qf2, vf, c_network, divergence, config):
        self.agent = agent
        self.qf1, self.qf2, self.vf = qf1, qf2, vf
        self.target_vf = vf.copy()
        self.train_tasks = train_tasks
        self.replay_buffer, self.enc_replay_buffer = replay_buffer, enc_replay_buffer
        self.c = c_network
        self.divergence = divergence
        self.batch_size = config.get('batch_size', 256)
        self.discount = config.get('discount', 0.99)
        self.reward_scale = config.get('reward_scale', 5.0)
        self.soft_target_tau = config.get('soft_target_tau', 0.005)
        self.z_loss_weight = config.get('z_loss_weight', 10.0)      # whole z_loss multiplier
        self._alpha_var = torch.tensor(config.get('alpha_init', 500.0), requires_grad=True)
        self.alpha_max = config.get('alpha_max', 2000.0)
        self.alpha_lr = config.get('alpha_lr', 1.0)
        self.target_divergence = config.get('target_divergence', 0.05)
        self.use_brac = config.get('use_brac', True)
        self.use_value_penalty = config.get('use_value_penalty', False)
        self.max_entropy = config.get('max_entropy', True)
        self.allow_backward_z = config.get('allow_backward_z', False)
        lr = 3e-4
        self.context_optimizer = optim.Adam(agent.context_encoder.parameters(), lr=lr)
        self.policy_optimizer  = optim.Adam(agent.policy.parameters(),          lr=lr)
        self.qf1_optimizer     = optim.Adam(qf1.parameters(),                   lr=lr)
        self.qf2_optimizer     = optim.Adam(qf2.parameters(),                   lr=lr)
        self.vf_optimizer      = optim.Adam(vf.parameters(),                    lr=lr)
        self.c_optimizer       = optim.Adam(self.c.parameters(),                lr=1e-4)

    @property
    def get_alpha(self):
        return torch.clamp(self._alpha_var, 0.0, self.alpha_max)

    def _take_step(self, indices, context):
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = sample_sac_batch(
            self.replay_buffer, indices, self.batch_size)

        # encode context -> per-task z (encoder gradients live here)
        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        obs_f, act_f, next_f = (x.view(t * b, -1) for x in (obs, actions, next_obs))

        # --- BRAC dual-form KL: train discriminator c, estimate divergence ---
        div_estimate = self.divergence.dual_estimate(obs_f, new_actions, act_f, task_z)
        c_loss = self.divergence.dual_critic_loss(obs_f, new_actions, act_f, task_z)
        self.c_optimizer.zero_grad(); c_loss.backward(retain_graph=True); self.c_optimizer.step()

        # --- encoder update: z_loss ONLY ---
        self.context_optimizer.zero_grad()
        z_loss = self.z_loss_weight * repo_z_loss(task_z, indices, b)
        z_loss.backward(retain_graph=True)
        self.context_optimizer.step()

        # --- critic update; canonical FlattenMlp call is net(t, b, *inputs) ---
        z_for_q = task_z if self.allow_backward_z else task_z.detach()
        q1 = self.qf1(t, b, obs_f, act_f, z_for_q)
        q2 = self.qf2(t, b, obs_f, act_f, z_for_q)
        v_pred = self.vf(t, b, obs_f, z_for_q)
        with torch.no_grad():
            target_v = self.target_vf(t, b, next_f, task_z)
            if self.use_brac and self.use_value_penalty:
                target_v = target_v - self.get_alpha * div_estimate
        rewards_f = rewards.view(self.batch_size * num_tasks, -1) * self.reward_scale
        terms_f = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_f + (1. - terms_f) * self.discount * target_v
        qf_loss = torch.mean((q1 - q_target) ** 2) + torch.mean((q2 - q_target) ** 2)
        self.qf1_optimizer.zero_grad(); self.qf2_optimizer.zero_grad()
        qf_loss.backward(); self.qf1_optimizer.step(); self.qf2_optimizer.step()

        # --- value function update (max-entropy target, kept even offline) ---
        min_q = torch.min(self.qf1(t, b, obs_f, new_actions, task_z.detach()),
                          self.qf2(t, b, obs_f, new_actions, task_z.detach()))
        v_target = min_q - log_pi if self.max_entropy else min_q
        vf_loss = F.mse_loss(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad(); vf_loss.backward(); self.vf_optimizer.step()
        soft_update(self.vf, self.target_vf, self.soft_target_tau)

        # --- policy update on detached z, with behavior policy regularization ---
        if self.use_brac:
            if self.max_entropy:
                policy_loss = (log_pi - min_q + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-min_q + self.get_alpha.detach() * div_estimate).mean()
        else:
            policy_loss = (log_pi - min_q).mean() if self.max_entropy else -min_q.mean()
        policy_loss = policy_loss + 1e-3 * (policy_mean ** 2).mean() \
                                  + 1e-3 * (policy_log_std ** 2).mean()
        self.policy_optimizer.zero_grad(); policy_loss.backward(); self.policy_optimizer.step()

        alpha_loss = -(self._alpha_var * (div_estimate - self.target_divergence).detach()).mean()
        alpha_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            self._alpha_var.grad.zero_()
```
