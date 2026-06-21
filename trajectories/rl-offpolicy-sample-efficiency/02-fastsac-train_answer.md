PPO landed where its structure said it would — weakest of the ladder — because on-policy learning discards every transition, so each of the 12.8M frames feeds at most a few gradient steps and is then thrown away, and the gap to an off-policy actor-critic opened widest on the harder-to-explore behaviors, walk and run, where long-horizon gait coordination needs the sample efficiency that experience reuse provides. The diagnosis is precise and it is not "PPO is unstable" — the clip makes PPO reliable — it is "PPO's learning throughput per frame is too low." The fix is not a better policy-gradient trick; it is to start reusing transitions, which means moving onto the editable off-policy surface and putting the replay buffer to work, while getting exploration from the objective rather than a hand-tuned per-task noise schedule.

I propose FastSAC: a maximum-entropy off-policy actor-critic re-expressed on this task's distributional substrate. I deliberately reach for the maximum-entropy line rather than the deterministic DDPG/TD3 line, because a deterministic actor explores by a thin band of additive noise, and on a humanoid with tens of contact-rich action dimensions "the current policy plus a little Gaussian noise" is a narrow data distribution — I do not want to trade PPO's throughput failure for a different exploration failure. In the max-entropy formulation exploration is not bolted on; it is a term the policy *optimizes*: maximize reward plus the entropy of the policy,

$$J(\pi)=\sum_t\mathbb{E}\big[r(s_t,a_t)+\alpha\,\mathcal H(\pi(\cdot|s_t))\big].$$

A policy paid for entropy keeps probability mass on every action that looks comparably good instead of collapsing onto one prematurely — directed-by-the-objective exploration that needs no per-task tuning. Crucially the entropy must live *inside* the value, not be sprinkled on the actor loss: the soft state value is $V(s)=\mathbb{E}_{a\sim\pi}[Q(s,a)-\alpha\log\pi(a|s)]$, and the soft Bellman target bootstraps on that $V$, so the future entropy I will collect shows up in the value of acting now and shapes long-horizon behavior — exactly the regime (walk, run) where PPO struggled.

Here is the care this substrate forces, because its critic is not the textbook scalar one. The critic emits 101 atom logits over the fixed support $[-250, 250]$ and is trained by cross-entropy against a *projected* target distribution, so I cannot use the usual soft-Bellman MSE target $y=r+\gamma(\min Q'-\alpha\log\pi')$. I have to fold the entropy bonus into the distributional projection, and the clean way is to push it into the *reward* the projection operates on. The projection computes, for each support atom $z$, the shifted location $Tz=r+\text{bootstrap}\cdot\text{discount}\cdot z$, clamps it to $[-250,250]$, and splits its mass onto the two neighboring grid atoms by linear interpolation; cross-entropy then trains the current critic toward that projected target. If I replace the reward $r$ with the entropy-augmented reward $r-\alpha\log\pi(a'|s')$ — evaluated at the next action $a'$ sampled from the current stochastic actor — then the soft value's entropy term enters the distributional bootstrap exactly as the theory demands, and everything downstream (clamp, floor/ceil split, cross-entropy) is unchanged. The clipped double-Q logic survives the move to distributions the FastTD3 way: read each target critic's scalar mean $\sum_i p_i z_i$ and keep the *whole distribution* belonging to whichever critic has the smaller mean; the min selects a distribution by its expectation, and that selected distribution is the cross-entropy target for both critics.

The actor mirrors this. It is a tanh-squashed Gaussian: the net outputs a mean and a log-std clamped to a sane band $[-5, 2]$; I draw a reparameterized pre-activation $u\sim\mathcal N(\mu,\sigma)$, squash with $\tanh$ to bound the action, and correct the log-prob for the squash by subtracting $\sum_i\log(1-\tanh^2(u_i)+\varepsilon)$ so the entropy term is exact. The actor objective is the reparameterized $\mathbb{E}[\alpha\log\pi(a|s)-\min(Q_1,Q_2)]$, where the two $Q$ values are the support-weighted means of the two critics' predicted distributions at the actor's sampled action — ascend the value, pay the entropy price. Because exploration now comes from the stochastic policy itself, the actor updates *every* gradient step rather than on TD3's delayed `policy_frequency` schedule; the loop runs the actor update inside the same inner loop as the critic. At evaluation the loop calls the deterministic readout $\tanh(\mu)$, the mean action — the right deterministic projection of an entropy-trained policy. Keeping the two entropy channels straight is what makes this a faithful soft actor-critic rather than a reward-shaping hack: one channel is the $-\alpha\log\pi(a'|s')$ inside the bootstrap (future-entropy credit), the other is the $\alpha\log\pi(a|s)$ in the actor loss.

The temperature $\alpha$ decides whether this works across stand, walk, and run without per-task tuning, so I do not hand-set it. I recast it as a dual variable that servos expected entropy onto a target: constrain $\mathbb{E}[-\log\pi]\ge\bar H$, solve the Lagrangian, and get $\alpha\leftarrow\arg\min_\alpha\mathbb{E}[-\alpha(\log\pi+\bar H)]$. I parameterize $\log\alpha$ for positivity and take one gradient step per update; the gradient sign makes it a thermostat — if the policy is too deterministic (entropy below $\bar H$), $\alpha$ rises and forces more exploration; if too random, $\alpha$ falls and lets the policy commit. The target is the scale-aware heuristic $\bar H=-\dim(\mathcal A)$, one nat per action dimension, so it scales with the humanoid's action count and needs no sweep. I initialize the entropy coefficient at $0.2$ and adapt $\log\alpha$ from there with its own Adam optimizer — modest because the augmented reward $r-\alpha\log\pi(a'|s')$ is what gets clamped to $[-250, 250]$, and a poorly-scaled $\alpha$ would push it outside the support and silently distort the target; starting at $0.2$ and letting the dual update pull it down as the policy sharpens is the safe regime.

Two substrate-specific choices lean toward the FastTD3 design philosophy. First, the architecture is not a bare MLP: both actor and critic use LayerNorm and SiLU bodies (a descending stack $512\to256\to128$ for the actor, $1024\to512\to256$ for the critic, each Linear followed by LayerNorm then SiLU). This is a deliberate stabilizer for off-policy value learning — LayerNorm controls feature magnitudes so the bootstrapped critic does not amplify its own scale drift, and SiLU is smoother than ReLU — and it matters precisely because moving off-policy pushes more gradient signal through the critic, the regime where the deadly triad bites without normalization. Second, the optimizer and reuse rate stay matched to the substrate's fast-and-stable settings: AdamW with weight decay $0.1$, cosine LR annealing, `num_updates=2` gradient steps per env step, and a fast target update $\tau=0.1$ — not SAC's classic small-$\tau$, single-update settings, but the ones the loop is built for.

Against PPO the advantage is structural and should be decisive: FastSAC reuses every transition out of the replay buffer many times instead of discarding it after a few epochs, so on the same 12.8M frames it extracts far more learning, and its exploration suits long-horizon coordination rather than undirected Gaussian wandering. So I expect it to clear PPO on all three tasks and open the largest margin on walk and run, closing the shortfall the PPO feedback diagnosed. The two-sided risk is against a *deterministic* off-policy method on this same substrate: maximizing entropy over a high-dimensional humanoid action is genuinely hard, and a stochastic actor holding a tanh-Gaussian over tens of dimensions can be noisier to optimize than a deterministic actor whose exploration is supplied for free by the fleet of 128 parallel environments. If such an actor explores adequately through parallelism alone, it could exploit the value function more aggressively and edge FastSAC out — especially on stand, where exploration is cheap and exploitation is what matters. That comparison is the next rung.

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
