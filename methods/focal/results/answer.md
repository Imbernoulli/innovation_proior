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
`(P,R)` identifies the task, so a single transition tuple in principle reveals task identity. The
encoder should therefore be **permutation-invariant** (use the mean of per-transition embeddings) and
**deterministic** (no information bottleneck, no product-of-Gaussians): there is no irreducible
uncertainty to model and no exploration that probabilistic posterior-sampling would serve.

**Why separation is necessary (continuity).** A value network `Q_θ(s,a,z)` is continuous: close `z`
forces close output `Q`. Two distinct tasks have genuinely different true `Q` values, so if their
embeddings are close the conditioned value functions become unrepresentable. Distinct tasks' `z` must
be pushed apart for control to be learnable — task inference geometry is a precondition, not a luxury.

**The inverse-power DML loss (the contribution).** The classic contrastive loss
`1{same}‖q_i−q_j‖² + 1{diff} max(0, m−‖q_i−q_j‖)²` degenerates: the margin spring is weak at small
distance and zero beyond `m`, and — the deeper failure — a squared-distance objective is *exactly*
variance maximization, `Σ_{i≠j}(x_i−x_j)² = 2N²·Var(X)`, a global statistic that a degenerate (e.g.
Bernoulli) distribution maximizes while merging several tasks into one pile. Replace the repulsive
term with a **negative power** of distance:
```
L_dml(x_i, x_j; q) = 1{y_i=y_j} ‖q_i−q_j‖₂²  +  1{y_i≠y_j} · β / ( ‖q_i−q_j‖₂ⁿ + ε ),   n ≥ 0, ε > 0.
```
This is a Coulomb-like repulsive potential: strongest at short range, per-pair, so it cannot be
satisfied by global variance and forces *every* pair of distinct task clusters apart. `n=2` coincides
with Cauchy graph embedding (better local-topology preservation than the quadratic/Laplacian form).
With the latent bounded by tanh to `(−1,1)^l` ("conducting box"), like charges settle to the
edges/corners — maximal separation. `q` is the **mean** transition embedding of a task; experiments
use `n ∈ {1 (inverse), 2 (inverse-square)}`, with inverse-square strongest.

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

- **Latent dim** 5 (reward-varying tasks) to 20 (dynamics-varying): `z` only encodes `(P,R)` structure.
- **Encoder** width 200, depth 3 MLP; **SAC heads** width 300, depth 3; latent tanh-bounded to `(−1,1)^l`.
- **DML**: `β=1` (match attractive/repulsive scales near half the per-dim range), small `ε` floor,
  `n` inverse-square; **DML weight** `~10` (it is the encoder's sole signal).
- **Reward scale** 5 (locomotion) / larger (sparse); **discount** 0.9–0.99; **soft target τ** 0.005;
  **lr** `1e-3`–`3e-4`; **discriminator lr** `1e-4`.
- **Behavior reg `α`**: 0 (Sparse-Point), ~500 (cheetah), `~10⁶` (Ant-Fwd-Back), tuned adaptively.

## Working code

Faithful to the canonical implementation: a deterministic mean-aggregating encoder, the Eqn 13 DML
loss, and a SAC + dual-form-KL-BRAC update with `z` detached from the actor/critic.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy


class FOCALAgent(nn.Module):
    """Deterministic context encoder with mean aggregation; SAC heads condition on z."""

    def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
                 reward_dim=1, use_next_obs_in_context=False, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_next_obs_in_context = use_next_obs_in_context
        context_input_dim = obs_dim + action_dim + reward_dim
        if use_next_obs_in_context:
            context_input_dim += obs_dim
        # deterministic encoder: one transition -> latent_dim (no IB, no Gaussian params)
        self.context_encoder = build_mlp(context_input_dim, latent_dim,
                                         hidden_dim=200, n_layers=3)
        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
        self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.vf = build_vf(obs_dim, latent_dim, net_size)
        self.target_vf = copy.deepcopy(self.vf)
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self._context = None

    def clear_z(self, num_tasks=1):
        self.z = torch.zeros(num_tasks, self.latent_dim)
        self._context = None

    def infer_posterior(self, context):
        # embed every transition, then MEAN over transitions -> deterministic z per task
        embeddings = self.context_encoder(context)
        embeddings = embeddings.view(context.size(0), -1, self.latent_dim)
        self.z = torch.mean(embeddings, dim=1)                 # (num_tasks, latent_dim)

    def adapt(self):
        if self._context is not None:
            self.infer_posterior(self._context)

    def detach_z(self):
        self.z = self.z.detach()                               # cut gradient path to encoder

    def get_action(self, obs, deterministic=False):
        in_ = torch.cat([torch.as_tensor(obs)[None], self.z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2, self.vf, self.target_vf]


def dml_loss(task_z, indices, b, beta=1.0, eps=1e-3, n=2.0):
    """FOCAL Eqn 13 distance metric learning loss.
    Same-task pairs attract (squared distance); different-task pairs repel
    (inverse-power potential ~ Coulomb). n=2 -> inverse-square (Cauchy form)."""
    pos_loss, neg_loss = 0.0, 0.0
    pos_cnt, neg_cnt = 0, 0
    for i in range(len(indices)):
        zi = task_z[i * b]
        for j in range(i + 1, len(indices)):
            zj = task_z[j * b]
            d_sq = torch.mean((zi - zj) ** 2)                  # mean squared diff / dim
            if indices[i] == indices[j]:                       # same task -> attract
                pos_loss += torch.sqrt(d_sq + eps)
                pos_cnt += 1
            else:                                              # different task -> repel
                neg_loss += beta / (d_sq ** (n / 2.0) + eps * 100)
                neg_cnt += 1
    return pos_loss / (pos_cnt + eps) + neg_loss / (neg_cnt + eps)


class FOCALAlgorithm:
    """Decoupled training: encoder by DML loss only; SAC + BRAC on detached z."""

    def __init__(self, agent, env, train_tasks, replay_buffer, enc_replay_buffer,
                 divergence, config):
        self.agent = agent
        self.train_tasks = train_tasks
        self.replay_buffer, self.enc_replay_buffer = replay_buffer, enc_replay_buffer
        self.divergence = divergence                              # dual-form KL discriminator
        self.batch_size = config.get('batch_size', 256)
        self.meta_batch = config.get('meta_batch', 16)
        self.discount = config.get('discount', 0.99)
        self.reward_scale = config.get('reward_scale', 5.0)
        self.soft_target_tau = config.get('soft_target_tau', 0.005)
        self.z_loss_weight = config.get('z_loss_weight', 10.0)
        self.alpha = config.get('alpha', 500.0)                  # behavior reg strength
        self.use_value_penalty = config.get('use_value_penalty', False)
        lr = 3e-4
        self.context_optimizer = optim.Adam(agent.context_encoder.parameters(), lr=lr)
        self.policy_optimizer  = optim.Adam(agent.policy.parameters(),          lr=lr)
        self.qf1_optimizer     = optim.Adam(agent.qf1.parameters(),             lr=lr)
        self.qf2_optimizer     = optim.Adam(agent.qf2.parameters(),             lr=lr)
        self.vf_optimizer      = optim.Adam(agent.vf.parameters(),              lr=lr)
        self.c_optimizer       = optim.Adam(self.divergence.parameters(),       lr=1e-4)

    def _take_step(self, indices, context):
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = sample_sac_batch(
            self.replay_buffer, indices, self.batch_size)

        # encode context -> per-task z (encoder gradients live here)
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        obs_f, act_f, next_f = (x.view(t * b, -1) for x in (obs, actions, next_obs))

        # --- BRAC dual-form KL: train discriminator, estimate divergence ---
        div_estimate = self.divergence.dual_estimate(obs_f, new_actions, act_f, task_z)
        c_loss = self.divergence.dual_critic_loss(obs_f, new_actions, act_f, task_z)
        self.c_optimizer.zero_grad(); c_loss.backward(retain_graph=True); self.c_optimizer.step()

        # --- encoder update: DML loss ONLY (z detached everywhere downstream) ---
        self.context_optimizer.zero_grad()
        z_loss = self.z_loss_weight * dml_loss(task_z, indices, b)
        z_loss.backward(retain_graph=True)
        self.context_optimizer.step()

        # --- critic update on detached z, with behavior value penalty ---
        with torch.no_grad():
            target_v = self.agent.target_vf(next_f, task_z.detach())
            if self.use_value_penalty:
                target_v = target_v - self.alpha * div_estimate     # BRAC value penalty
        q1 = self.agent.qf1(obs_f, act_f, task_z.detach())
        q2 = self.agent.qf2(obs_f, act_f, task_z.detach())
        rewards_f = rewards.view(self.batch_size * num_tasks, -1) * self.reward_scale
        terms_f = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_f + (1. - terms_f) * self.discount * target_v
        qf_loss = torch.mean((q1 - q_target) ** 2) + torch.mean((q2 - q_target) ** 2)
        self.qf1_optimizer.zero_grad(); self.qf2_optimizer.zero_grad()
        qf_loss.backward(); self.qf1_optimizer.step(); self.qf2_optimizer.step()

        # --- value function update (max-entropy target, kept even offline) ---
        min_q = torch.min(self.agent.qf1(obs_f, new_actions, task_z.detach()),
                          self.agent.qf2(obs_f, new_actions, task_z.detach()))
        v_pred = self.agent.vf(obs_f, task_z.detach())
        v_target = min_q - log_pi
        vf_loss = F.mse_loss(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad(); vf_loss.backward(); self.vf_optimizer.step()
        soft_update(self.agent.vf, self.agent.target_vf, self.soft_target_tau)

        # --- policy update on detached z, with behavior policy regularization ---
        policy_loss = (log_pi - min_q + self.alpha * div_estimate.detach()).mean()
        policy_loss = policy_loss + 1e-3 * (policy_mean ** 2).mean() \
                                  + 1e-3 * (policy_log_std ** 2).mean()
        self.policy_optimizer.zero_grad(); policy_loss.backward(); self.policy_optimizer.step()
```
