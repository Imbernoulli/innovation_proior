## Research question

Train a humanoid to stand, walk, and run — continuous actions clipped to `[-1, 1]`, tens of action dimensions — and get the **highest mean episode return within a fixed, small budget**: 100,000 gradient steps with 128 parallel environments on a single GPU, deterministic actions at evaluation. The object of design is the **off-policy algorithm itself**: the actor architecture and exploration, the critic architecture and value estimation, and the update rules that tie them together. The training infrastructure is fixed. The question is which fill of that algorithm surface learns the most capable locomotion policy under exactly this budget.

## Prior art / Background / Baselines

The relevant baselines and the gap each leaves:

- **REINFORCE / vanilla policy gradient (Williams 1992).** Estimates the policy gradient from Monte-Carlo returns and takes one gradient step per sample. Gap: high variance and no data reuse make it sample-inefficient.
- **TRPO (Schulman et al. 2015).** Optimizes within a KL-divergence trust region using Fisher-vector products. Gap: each update is expensive and second-order, so data reuse is limited and wall-clock cost is high.
- **DDPG (Lillicrap et al. 2016).** Uses a deterministic actor and a Q-critic with a replay buffer and target networks. Gap: brittle in practice due to critic overestimation and exploration collapse.
- **TD3 (Fujimoto et al. 2018).** Extends DDPG with clipped twin critics, target-policy smoothing, and delayed actor updates. Gap: still relies on narrow additive action noise, so exploration can be slow on hard locomotion tasks.
- **SAC (Haarnoja et al. 2018).** Trains a stochastic actor with a maximum-entropy objective and automatic temperature tuning. Gap: entropy maximization over a high-dimensional action space is difficult, and the stochastic policy can underperform a well-explored deterministic one at this scale.
- **Distributional value estimation (C51, Bellemare et al. 2017).** Models the full return distribution rather than a scalar value. Gap: a richer critic target still leaves open how to pair it with exploration and actor-architecture choices. The substrate's critic is distributional.

## Fixed substrate / Code framework

A scaled off-policy training loop is frozen. It provides: 128 parallel HumanoidBench environments stepped on the GPU; a GPU-resident replay buffer (`SimpleReplayBuffer`) with per-env capacity and n-step returns; running-statistic observation normalization (`EmpiricalNormalization`) applied to actor, critic, and next observations; bfloat16 autocast and `torch.compile` for speed; the experience-collection step (policy noise → env step → store `(obs, action, next_obs, reward, truncation, done)` with the correct time-limit-vs-termination bootstrap mask); `num_updates=2` gradient steps per env step, `policy_frequency=2` delayed actor updates, and a Polyak `soft_update`; cosine LR annealing; and deterministic-evaluation logging over 3 rollouts. The fixed loop calls the editable `build_algorithm`, `update_critic`, `update_actor`, and `soft_update` once per update.

## Editable interface

Exactly one region of `custom_algorithm.py` (lines 50–331) is editable: the `Actor` and `Critic` network classes, `build_algorithm()`, `update_critic()`, `update_actor()`, and `soft_update()`. The fixed loop already normalizes observations, samples the batch, masks bootstraps, and steps the schedulers; the editable functions receive a normalized `data` TensorDict and the AMP context and must return loss/diagnostic dicts.

The starting point is the default scaffold — a TD3-style fill: a deterministic descending-MLP actor with per-env mixed Gaussian exploration noise, twin categorical distributional critics (101 atoms, support `[-250, 250]`) with the projected cross-entropy target and clipped double-Q, target-policy smoothing in the critic target, and the deterministic-policy-gradient actor objective (ascend the min of the two distributional means).

```python
# EDITABLE region of custom_algorithm.py (lines 50-331) — default fill (TD3-style)
class Actor(nn.Module):
    """Deterministic actor: descending MLP + tanh head, per-env Gaussian exploration noise."""
    def __init__(self, n_obs, n_act, num_envs, device, hidden_dim=512,
                 init_scale=0.01, std_min=0.001, std_max=0.4):
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
        self.device_ = device

    def forward(self, obs):
        return self.fc_mu(self.net(obs))

    def explore(self, obs, dones=None, deterministic=False):
        if dones is not None and dones.sum() > 0:
            new_scales = torch.rand(self.n_envs, 1, device=obs.device) * (self.std_max - self.std_min) + self.std_min
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(torch.where(dones_view, new_scales, self.noise_scales))
        act = self(obs)
        if deterministic:
            return act
        return act + torch.randn_like(act) * self.noise_scales


class DistributionalQNetwork(nn.Module):
    """Categorical distributional Q-network: emits atom logits; projects the Bellman target."""
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
        target_z = (rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support).clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long(); u = torch.ceil(b).long()
        is_int = (l == u); l_mask = is_int & (l > 0); u_mask = is_int & (l == 0)
        l = torch.where(l_mask, l - 1, l); u = torch.where(u_mask, u + 1, u)
        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device).unsqueeze(1).expand(batch_size, self.num_atoms).long()
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        return proj_dist


class Critic(nn.Module):
    """Twin distributional critics with clipped double Q-learning."""
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


def build_algorithm(n_obs, n_act, num_envs, device, args):
    actor = Actor(n_obs=n_obs, n_act=n_act, num_envs=num_envs, device=device,
                  hidden_dim=args.actor_hidden_dim, init_scale=args.init_scale,
                  std_min=args.std_min, std_max=args.std_max)
    critic = Critic(n_obs=n_obs, n_act=n_act, num_atoms=args.num_atoms, v_min=args.v_min,
                    v_max=args.v_max, hidden_dim=args.critic_hidden_dim, device=device)
    critic_target = Critic(n_obs=n_obs, n_act=n_act, num_atoms=args.num_atoms, v_min=args.v_min,
                           v_max=args.v_max, hidden_dim=args.critic_hidden_dim, device=device)
    critic_target.load_state_dict(critic.state_dict())
    actor_optimizer = optim.AdamW(actor.parameters(), lr=torch.tensor(args.actor_learning_rate, device=device), weight_decay=args.weight_decay)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=torch.tensor(args.critic_learning_rate, device=device), weight_decay=args.weight_decay)
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=args.total_timesteps, eta_min=torch.tensor(args.actor_learning_rate_end, device=device))
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=args.total_timesteps, eta_min=torch.tensor(args.critic_learning_rate_end, device=device))
    return {"actor": actor, "critic": critic, "critic_target": critic_target,
            "actor_optimizer": actor_optimizer, "critic_optimizer": critic_optimizer,
            "actor_scheduler": actor_scheduler, "critic_scheduler": critic_scheduler}


def update_critic(data, components, args, scaler, amp_enabled, amp_device_type, amp_dtype):
    actor, critic, critic_target = components["actor"], components["critic"], components["critic_target"]
    critic_optimizer = components["critic_optimizer"]
    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
        observations = data["observations"]; next_observations = data["next"]["observations"]
        actions = data["actions"]; rewards = data["next"]["rewards"]
        dones = data["next"]["dones"].bool(); truncations = data["next"]["truncations"].bool()
        bootstrap = (truncations | ~dones).float()
        clipped_noise = torch.randn_like(actions).mul(args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
        next_state_actions = (actor(next_observations) + clipped_noise).clamp(-1.0, 1.0)
        discount = args.gamma ** data["next"]["effective_n_steps"]
        with torch.no_grad():
            qf1_next_proj, qf2_next_proj = critic_target.projection(next_observations, next_state_actions, rewards, bootstrap, discount)
            qf1_next_val = critic_target.get_value(qf1_next_proj); qf2_next_val = critic_target.get_value(qf2_next_proj)
            qf_next_dist = torch.where(qf1_next_val.unsqueeze(1) < qf2_next_val.unsqueeze(1), qf1_next_proj, qf2_next_proj)
            qf1_next_dist = qf2_next_dist = qf_next_dist
        qf1, qf2 = critic(observations, actions)
        qf1_loss = -torch.sum(qf1_next_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
        qf2_loss = -torch.sum(qf2_next_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
        qf_loss = qf1_loss + qf2_loss
    critic_optimizer.zero_grad(set_to_none=True)
    scaler.scale(qf_loss).backward(); scaler.unscale_(critic_optimizer); scaler.step(critic_optimizer); scaler.update()
    return {"qf_loss": qf_loss.detach(), "qf1_next_val": qf1_next_val}


def update_actor(data, components, args, scaler, amp_enabled, amp_device_type, amp_dtype):
    actor, critic = components["actor"], components["critic"]; actor_optimizer = components["actor_optimizer"]
    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
        qf1, qf2 = critic(data["observations"], actor(data["observations"]))
        qf1_value = critic.get_value(F.softmax(qf1, dim=1)); qf2_value = critic.get_value(F.softmax(qf2, dim=1))
        actor_loss = -torch.minimum(qf1_value, qf2_value).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    scaler.scale(actor_loss).backward(); scaler.unscale_(actor_optimizer); scaler.step(actor_optimizer); scaler.update()
    return {"actor_loss": actor_loss.detach()}


@torch.no_grad()
def soft_update(src, tgt, tau):
    src_ps = [p.data for p in src.parameters()]; tgt_ps = [p.data for p in tgt.parameters()]
    torch._foreach_mul_(tgt_ps, 1.0 - tau); torch._foreach_add_(tgt_ps, src_ps, alpha=tau)
```

## Evaluation settings

Three HumanoidBench locomotion tasks — **h1hand-stand-v0** (standing balance), **h1hand-walk-v0** (walking), **h1hand-run-v0** (running, hidden) — each at a fixed budget of 100,000 gradient steps with 128 parallel environments on a single GPU. The metric on every task is **mean episode return over 3 evaluation rollouts** at the end of training (`mean_reward_h1hand_{stand,walk,run}_v0`), higher is better; the task score is the geometric mean across the three settings. Actions are continuous in `[-1, 1]` and evaluation actions are deterministic via `actor(obs)`.
