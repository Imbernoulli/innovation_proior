# FastTD3

## Problem

Train humanoid control policies (whole-body locomotion, dexterous manipulation; continuous
actions in `[-1, 1]`, dozens of action dimensions) that are simultaneously **fast in wall-clock
time**, **capable**, **stable**, and **simple** — on a single GPU. On-policy PPO is fast with
parallel simulation but sample-inefficient; the sample-efficient off-policy line is complex and
slow because high update-to-data ratios trip the deadly triad (bootstrapping + function
approximation + off-policy) and need architectural stabilizers.

## Key idea

Take **TD3** unchanged in spirit and make the speed/stability/exploration come from **data, not
architecture**. The recipe:

1. **Massively parallel environments.** Step `num_envs` (128 for HumanoidBench; up to thousands
   on GPU-native sims) in parallel, each with its own exploration noise and init state. The
   diverse data cures TD3's weak deterministic exploration *without* making the policy
   stochastic, and fills the replay buffer fast.
2. **Large-batch updates.** Sample a very large batch (32,768) so each gradient step averages
   the bootstrapped critic loss over a diverse, low-variance cross-section of the buffer →
   stable critic. Fewer, more informative updates ⇒ less total wall-clock.
3. **Distributional (categorical / C51) critic.** Learn the return distribution `Z(s,a)` over a
   fixed support of `num_atoms` atoms (101) in `[v_min, v_max]` (`±250`), via the categorical
   projection and cross-entropy loss. Keeps separated value modes (e.g. survive vs. fall)
   distinct ⇒ more stable, more capable critic.
4. **Clipped Double Q over distributions.** Twin critics; bootstrap with the *entire
   distribution* of whichever critic has the smaller **mean** value (pessimism vs.
   overestimation). In the no-LayerNorm setting, the empirical ablation favors the min over
   averaging.
5. **Plain descending MLP, no LayerNorm / no residual; low UTD.** Data diversity tames the
   triad, so stabilizers only cost wall-clock; low UTD (`num_updates` 2–8 per 128+ env steps)
   avoids early overfitting and scales without extra machinery.

Carried over from TD3: target policy smoothing (clipped noise on the target action), delayed
policy updates (`policy_frequency`), soft critic-target updates `θ' ← (1-τ)θ' + τθ`,
deterministic policy gradient actor objective. Plus: per-env **mixed exploration** noise sampled in
`[std_min, std_max]` with large `std_max` (no per-task σ tuning); replay buffer sized
`N × num_envs` and held on GPU; bootstrap mask `(truncations | ~dones)`; AdamW + weight decay +
cosine LR; AMP (bfloat16) + `torch.compile` for speed.

## Distributional critic — the load-bearing math

Fixed support `z_i = v_min + i·Δz`, `i = 0..N-1`, `Δz = (v_max - v_min)/(N-1)`. The critic emits
logits; `p_i(s,a) = softmax(logits)_i`; scalar value `Q = Σ_i p_i z_i`.

**Categorical projection of the Bellman target.** For each target atom `z_j` with prob
`p_j(s',a')`, the shifted location is `Tz_j = clip(r + bootstrap·discount·z_j, v_min, v_max)`.
Its grid-index position `b = (Tz_j - v_min)/Δz` has integer neighbors `l = ⌊b⌋`, `u = ⌈b⌉`;
distribute its mass by linear interpolation (farther neighbor gets the smaller share):

```
m_l += p_j · (u - b)
m_u += p_j · (b - l)
```

The weights `(u-b) + (b-l) = 1` for adjacent integers, so probability is conserved. When
`l == u`, choose an adjacent pair before scattering: for `b=0`, use `(l,u)=(0,1)`, and for
integer `b=k>0`, use `(l,u)=(k-1,k)`. The complementary weight is zero, so all mass lands on
the original atom and total probability is still conserved. Equivalently, the projected target is the tent kernel
`(ΦT̂Z)_i = Σ_j [1 - |Tz_j - z_i|/Δz]_0^1 · p_j`.

**Loss = cross-entropy to the projected target** (minimizes `D_KL(m ‖ p_θ)`):
`L = -Σ_i m_i · log_softmax(logits)_i`, averaged over the batch.

## Faithful implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class DistributionalQNetwork(nn.Module):
    def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device), nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min, self.v_max, self.num_atoms = v_min, v_max, num_atoms

    def forward(self, obs, actions):
        return self.net(torch.cat([obs, actions], 1))

    def projection(self, obs, actions, rewards, bootstrap, discount, q_support, device):
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()
        is_int = (l == u)
        l_mask = is_int & (l > 0)
        u_mask = is_int & (l == 0)
        l = torch.where(l_mask, l - 1, l)
        u = torch.where(u_mask, u + 1, u)
        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
                  .unsqueeze(1).expand(batch_size, self.num_atoms).long())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        return proj_dist


class Critic(nn.Module):
    def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device=None):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device)
        self.qnet2 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device)
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))
        self.device = device

    def forward(self, obs, actions):
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(self, obs, actions, rewards, bootstrap, discount):
        q1 = self.qnet1.projection(obs, actions, rewards, bootstrap, discount, self.q_support, self.q_support.device)
        q2 = self.qnet2.projection(obs, actions, rewards, bootstrap, discount, self.q_support, self.q_support.device)
        return q1, q2

    def get_value(self, probs):
        return torch.sum(probs * self.q_support, dim=1)


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, num_envs, init_scale, hidden_dim,
                 std_min=0.05, std_max=0.8, device=None):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device), nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(nn.Linear(hidden_dim // 4, n_act, device=device), nn.Tanh())
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        noise_scales = torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs
        self.device = device

    def forward(self, obs):
        return self.fc_mu(self.net(obs))

    def explore(self, obs, dones=None, deterministic=False):
        if dones is not None and dones.sum() > 0:
            new_scales = torch.rand(self.n_envs, 1, device=obs.device) * (self.std_max - self.std_min) + self.std_min
            self.noise_scales.copy_(torch.where(dones.view(-1, 1) > 0, new_scales, self.noise_scales))
        act = self(obs)
        if deterministic:
            return act
        return act + torch.randn_like(act) * self.noise_scales
```

### Critic and actor updates (TD3 + distributional + clipped double Q)

```python
def update_main(data, logs_dict):
    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
        observations = data["observations"]
        next_observations = data["next"]["observations"]
        if envs.asymmetric_obs:
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
        else:
            critic_observations = observations
            next_critic_observations = next_observations
        actions = data["actions"]
        rewards = data["next"]["rewards"]
        dones = data["next"]["dones"].bool()
        truncations = data["next"]["truncations"].bool()
        bootstrap = (~dones).float() if args.disable_bootstrap else (truncations | ~dones).float()

        clipped_noise = torch.randn_like(actions).mul(policy_noise).clamp(-noise_clip, noise_clip)
        next_state_actions = (actor(next_observations) + clipped_noise).clamp(action_low, action_high)
        discount = args.gamma ** data["next"]["effective_n_steps"]

        with torch.no_grad():
            qf1_proj, qf2_proj = qnet_target.projection(
                next_critic_observations, next_state_actions, rewards, bootstrap, discount
            )
            qf1_val = qnet_target.get_value(qf1_proj)
            qf2_val = qnet_target.get_value(qf2_proj)
            if args.use_cdq:
                qf_target_dist = torch.where(qf1_val.unsqueeze(1) < qf2_val.unsqueeze(1), qf1_proj, qf2_proj)
                qf1_target_dist = qf2_target_dist = qf_target_dist
            else:
                qf1_target_dist, qf2_target_dist = qf1_proj, qf2_proj

        qf1, qf2 = qnet(critic_observations, actions)
        qf1_loss = -torch.sum(qf1_target_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
        qf2_loss = -torch.sum(qf2_target_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
        qf_loss = qf1_loss + qf2_loss

    q_optimizer.zero_grad(set_to_none=True)
    scaler.scale(qf_loss).backward()
    scaler.unscale_(q_optimizer)
    if args.use_grad_norm_clipping:
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            qnet.parameters(), max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
        )
    else:
        critic_grad_norm = torch.tensor(0.0, device=device)
    scaler.step(q_optimizer)
    scaler.update()
    logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
    logs_dict["qf_loss"] = qf_loss.detach()
    logs_dict["qf_max"] = qf1_val.max().detach()
    logs_dict["qf_min"] = qf1_val.min().detach()
    return logs_dict


def update_pol(data, logs_dict):
    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
        critic_observations = data["critic_observations"] if envs.asymmetric_obs else data["observations"]
        qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
        qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
        qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
        qf_value = torch.minimum(qf1_value, qf2_value) if args.use_cdq else (qf1_value + qf2_value) / 2.0
        actor_loss = -qf_value.mean()

    actor_optimizer.zero_grad(set_to_none=True)
    scaler.scale(actor_loss).backward()
    scaler.unscale_(actor_optimizer)
    if args.use_grad_norm_clipping:
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
        )
    else:
        actor_grad_norm = torch.tensor(0.0, device=device)
    scaler.step(actor_optimizer)
    scaler.update()
    logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
    logs_dict["actor_loss"] = actor_loss.detach()
    return logs_dict


@torch.no_grad()
def soft_update(src, tgt, tau):
    src_ps = [p.data for p in src.parameters()]
    tgt_ps = [p.data for p in tgt.parameters()]
    torch._foreach_mul_(tgt_ps, 1.0 - tau)
    torch._foreach_add_(tgt_ps, src_ps, alpha=tau)
```

## Default hyperparameters (HumanoidBench)

`num_envs=128`, `batch_size=32768`, `buffer_size = N × num_envs`, `gamma=0.99`, `tau=0.1`,
`num_atoms=101`, `v_min=-250`, `v_max=250`, `critic_hidden_dim=1024` (→ 1024/512/256),
`actor_hidden_dim=512` (→ 512/256/128), `policy_noise=0.001`, `noise_clip=0.5`,
mixed exploration `std_min=0.001`, `std_max=0.4` in the HumanoidBench script (the class has
broader defaults), `num_updates=2`
(low UTD), `policy_frequency=2` (delayed actor update), AdamW `lr=3e-4`, `weight_decay=0.1`,
cosine LR schedule, observation normalization, `use_cdq=True`, AMP (bfloat16) + `torch.compile`.
Deterministic actions at evaluation via `actor(obs)`.
