**Problem.** PPO is weakest because on-policy learning discards every transition. The cheapest way
past it is to start reusing experience — move onto the off-policy surface and put the replay buffer
to work — while getting exploration from the objective rather than a hand-tuned noise schedule.

**Key idea.** FastSAC: a maximum-entropy off-policy actor-critic re-expressed on this task's
distributional substrate. A tanh-squashed Gaussian actor (LayerNorm+SiLU body) explores by
optimizing reward *plus* policy entropy; the entropy lives inside the soft value, so it enters the
**distributional** bootstrap by subtracting $\alpha\log\pi(a'|s')$ from the reward fed to the
projection (clamp / floor-ceil split / cross-entropy unchanged). Clipped double-Q selects the whole
target distribution whose mean is smaller. The temperature $\alpha$ is a dual variable servoing
expected entropy onto $\bar H=-\dim(A)$ (a thermostat), so no per-task tuning. The actor updates
every step (exploration is in the policy, not a delayed schedule); evaluation uses the deterministic
mean $\tanh(\mu)$.

**Why it works.** Experience reuse converts the same 12.8M frames into far more gradient signal than
PPO; entropy-driven exploration suits the long-horizon coordination (walk, run) where PPO fell
furthest behind. LayerNorm+SiLU stabilizes the off-policy critic now pushed harder; the
distributional critic keeps separated return modes (survive vs fall) distinct.

**Hyperparameters.** Actor hidden 512 (→256→128), critic hidden 1024 (→512→256), each Linear +
LayerNorm + SiLU; num_atoms 101, support $[-250, 250]$; tanh-Gaussian log-std clamp $[-5, 2]$;
init entropy coef 0.2, target entropy $-\dim(A)$; AdamW lr 3e-4, weight_decay 0.1, cosine anneal;
$\tau=0.1$, num_updates 2, gamma 0.99.

```python
# EDITABLE region of custom_algorithm.py — step 2: FastSAC (stochastic max-entropy, distributional)
LOG_STD_MIN, LOG_STD_MAX = -5, 2


class Actor(nn.Module):
    """Stochastic tanh-Gaussian actor with LayerNorm + SiLU body."""
    def __init__(self, n_obs, n_act, num_envs, device, hidden_dim=512,
                 init_scale=0.01, std_min=0.001, std_max=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device), nn.LayerNorm(hidden_dim // 2, device=device), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device), nn.LayerNorm(hidden_dim // 4, device=device), nn.SiLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim // 4, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim // 4, n_act, device=device)
        self.n_act = n_act

    def forward(self, obs):
        x = self.net(obs)
        return self.fc_mean(x), torch.clamp(self.fc_logstd(x), LOG_STD_MIN, LOG_STD_MAX)

    def get_action(self, obs):
        mean, log_std = self.forward(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1)

    def deterministic_action(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean)

    def forward_eval(self, obs):  # the loop's eval calls actor(obs); route to the mean action
        return self.deterministic_action(obs)

    def explore(self, obs, dones=None, deterministic=False):
        if deterministic:
            return self.deterministic_action(obs)
        action, _ = self.get_action(obs)
        return action


class DistributionalQNetwork(nn.Module):
    """Categorical critic with LayerNorm + SiLU; entropy folded into the projection's reward."""
    def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device), nn.LayerNorm(hidden_dim // 2, device=device), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device), nn.LayerNorm(hidden_dim // 4, device=device), nn.SiLU(),
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
    actor = Actor(n_obs, n_act, num_envs, device, hidden_dim=args.actor_hidden_dim)
    critic = Critic(n_obs, n_act, args.num_atoms, args.v_min, args.v_max, args.critic_hidden_dim, device)
    critic_target = Critic(n_obs, n_act, args.num_atoms, args.v_min, args.v_max, args.critic_hidden_dim, device)
    critic_target.load_state_dict(critic.state_dict())
    log_alpha = torch.tensor(float(np.log(0.2)), device=device, requires_grad=True)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=torch.tensor(args.actor_learning_rate, device=device), weight_decay=args.weight_decay)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=torch.tensor(args.critic_learning_rate, device=device), weight_decay=args.weight_decay)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.actor_learning_rate)
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=args.total_timesteps, eta_min=torch.tensor(args.actor_learning_rate_end, device=device))
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=args.total_timesteps, eta_min=torch.tensor(args.critic_learning_rate_end, device=device))
    return {"actor": actor, "critic": critic, "critic_target": critic_target,
            "actor_optimizer": actor_optimizer, "critic_optimizer": critic_optimizer,
            "actor_scheduler": actor_scheduler, "critic_scheduler": critic_scheduler,
            "log_alpha": log_alpha, "alpha_optimizer": alpha_optimizer, "target_entropy": -float(n_act)}


def update_critic(data, components, args, scaler, amp_enabled, amp_device_type, amp_dtype):
    actor, critic, critic_target = components["actor"], components["critic"], components["critic_target"]
    critic_optimizer = components["critic_optimizer"]; alpha = components["log_alpha"].exp().detach()
    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
        observations = data["observations"]; next_observations = data["next"]["observations"]
        actions = data["actions"]; rewards = data["next"]["rewards"]
        dones = data["next"]["dones"].bool(); truncations = data["next"]["truncations"].bool()
        bootstrap = (truncations | ~dones).float()
        discount = args.gamma ** data["next"]["effective_n_steps"]
        with torch.no_grad():
            next_actions, next_log_prob = actor.get_action(next_observations)
            qf1_next_proj, qf2_next_proj = critic_target.projection(
                next_observations, next_actions, rewards - alpha * next_log_prob, bootstrap, discount)
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
    log_alpha = components["log_alpha"]; alpha_optimizer = components["alpha_optimizer"]
    alpha = log_alpha.exp().detach()
    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
        new_actions, log_prob = actor.get_action(data["observations"])
        qf1_a, qf2_a = critic(data["observations"], new_actions)
        qf1_v = critic.get_value(F.softmax(qf1_a, dim=1)); qf2_v = critic.get_value(F.softmax(qf2_a, dim=1))
        actor_loss = (alpha * log_prob - torch.minimum(qf1_v, qf2_v)).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    scaler.scale(actor_loss).backward(); scaler.unscale_(actor_optimizer); scaler.step(actor_optimizer); scaler.update()
    alpha_loss = -(log_alpha * (log_prob.detach() + components["target_entropy"])).mean()
    alpha_optimizer.zero_grad(); alpha_loss.backward(); alpha_optimizer.step()
    return {"actor_loss": actor_loss.detach(), "alpha_loss": alpha_loss.detach()}


@torch.no_grad()
def soft_update(src, tgt, tau):
    src_ps = [p.data for p in src.parameters()]; tgt_ps = [p.data for p in tgt.parameters()]
    torch._foreach_mul_(tgt_ps, 1.0 - tau); torch._foreach_add_(tgt_ps, src_ps, alpha=tau)
```
