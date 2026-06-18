**Problem.** FastTD3 wins by investing in the *algorithm* (deterministic exploitation + fleet
exploration + experience reuse) but deliberately not in the *network* — a plain ReLU MLP, on the bet
that data diversity tames the triad so architecture is unnecessary. That leaves a plain MLP critic
that does not reliably convert capacity into a clean value fit on the sharper, longer-horizon value
surfaces (walk, run), where unregularized capacity overfits a moving bootstrapped target.

**Key idea (finale: SimBa, Lee et al. 2025).** Keep the entire FastTD3 backbone and change only the
*function class* of the actor and critic, importing the simplicity-bias architecture that makes a
network scale safely. The encoder, applied identically to actor and critic, is: a linear embedding →
N **pre-LayerNorm residual blocks** (`x + Linear_down(ReLU(Linear_up(LayerNorm(x))))`, 4× inverted
bottleneck) → a **final LayerNorm**, then the original head. The residual skip makes a simple
near-linear map the free default and any nonlinearity an opt-in correction — exactly the
regularization a bootstrapped critic chasing a moving target needs. Running-stat observation
normalization (the third SimBa ingredient) is already supplied by the fixed loop's
`EmpiricalNormalization`, so it is inherited, not duplicated. Everything algorithmic — distributional
critic, projection, clipped double-Q, target smoothing, per-env noise, delayed actor updates — stays
exactly FastTD3.

**Why it should beat FastTD3.** The critic is made the larger of the two (2 blocks / width 512 vs the
actor's 1 block / width 128) because its target is the harder one, and the residual + LayerNorm + He
init combination lets that capacity *help* where FastTD3's plain MLP regressed. The change is confined
to the network, so any gain is attributable to the architecture. Bar: not worse than FastTD3 on stand
(little headroom), better on walk and run (the hard value surfaces). Falsifiable: if scaling the
critic with this encoder does not help on run, the simplicity-bias premise is wrong.

**Hyperparameters (canonical SimBa).** Actor: 1 residual block, hidden 128. Critic (each twin): 2
residual blocks, hidden 512. Block: pre-LN → Linear(d→4d) → ReLU → Linear(4d→d) → add. Embedding:
orthogonal gain 1.0; block linears: He-normal, zero bias; actor head: orthogonal gain 0.01 (start
near-zero action). Distributional head: 101 atoms, $[-250, 250]$. Optimizer/loop inherited from the
substrate: AdamW lr 3e-4, weight_decay 0.1, cosine anneal, $\tau=0.1$, policy_freq 2, num_updates 2.

```python
# EDITABLE region of custom_algorithm.py — finale: SimBa encoder bodies on the FastTD3 backbone
class ResidualBlock(nn.Module):
    """Pre-LayerNorm inverted-bottleneck residual block (the SimBa simplicity-bias block)."""
    def __init__(self, hidden_dim, device=None):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, device=device)
        self.fc_up = nn.Linear(hidden_dim, hidden_dim * 4, device=device)
        self.fc_down = nn.Linear(hidden_dim * 4, hidden_dim, device=device)
        nn.init.kaiming_normal_(self.fc_up.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc_down.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc_up.bias); nn.init.zeros_(self.fc_down.bias)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = torch.relu(self.fc_up(x))
        x = self.fc_down(x)
        return res + x


class SimbaEncoder(nn.Module):
    """Linear embed -> N residual blocks -> final LayerNorm (obs normalization is the loop's job)."""
    def __init__(self, n_in, hidden_dim, num_blocks, device=None):
        super().__init__()
        self.embed = nn.Linear(n_in, hidden_dim, device=device)
        nn.init.orthogonal_(self.embed.weight, gain=1.0); nn.init.zeros_(self.embed.bias)
        self.blocks = nn.ModuleList(ResidualBlock(hidden_dim, device) for _ in range(num_blocks))
        self.norm_out = nn.LayerNorm(hidden_dim, device=device)

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm_out(x)


class Actor(nn.Module):
    """Deterministic actor: SimBa encoder body + tanh head, per-env Gaussian exploration noise."""
    def __init__(self, n_obs, n_act, num_envs, device, hidden_dim=128, num_blocks=1,
                 init_scale=0.01, std_min=0.001, std_max=0.4):
        super().__init__()
        self.n_act = n_act
        self.encoder = SimbaEncoder(n_obs, hidden_dim, num_blocks, device)
        self.fc_mu = nn.Linear(hidden_dim, n_act, device=device)
        nn.init.orthogonal_(self.fc_mu.weight, gain=init_scale)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        noise_scales = torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs

    def forward(self, obs):
        return torch.tanh(self.fc_mu(self.encoder(obs)))

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
    """Categorical critic with a SimBa encoder body over [obs, act]."""
    def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, num_blocks=2, device=None):
        super().__init__()
        self.encoder = SimbaEncoder(n_obs + n_act, hidden_dim, num_blocks, device)
        self.head = nn.Linear(hidden_dim, num_atoms, device=device)
        self.v_min, self.v_max, self.num_atoms = v_min, v_max, num_atoms

    def forward(self, obs, actions):
        return self.head(self.encoder(torch.cat([obs, actions], 1)))

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
    def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device=None):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, 2, device)
        self.qnet2 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, 2, device)
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
    # SimBa canonical sizes: actor 1 block / width 128, critic 2 blocks / width 512.
    actor = Actor(n_obs=n_obs, n_act=n_act, num_envs=num_envs, device=device,
                  hidden_dim=128, num_blocks=1, init_scale=args.init_scale,
                  std_min=args.std_min, std_max=args.std_max)
    critic = Critic(n_obs, n_act, args.num_atoms, args.v_min, args.v_max, 512, device)
    critic_target = Critic(n_obs, n_act, args.num_atoms, args.v_min, args.v_max, 512, device)
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
        qf1, qf2 = critic(observations, actions)
        qf1_loss = -torch.sum(qf_next_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
        qf2_loss = -torch.sum(qf_next_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
        qf_loss = qf1_loss + qf2_loss
    critic_optimizer.zero_grad(set_to_none=True)
    scaler.scale(qf_loss).backward(); scaler.unscale_(critic_optimizer); scaler.step(critic_optimizer); scaler.update()
    return {"qf_loss": qf_loss.detach()}


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
