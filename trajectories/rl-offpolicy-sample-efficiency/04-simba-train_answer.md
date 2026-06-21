FastTD3 is the strongest of the three, and the reason it wins tells me exactly where its ceiling is. It beat FastSAC by dropping the entropy tax and getting exploration for free from 128 parallel environments, and it beat PPO by reusing experience while exploiting the value function greedily with a deterministic actor. Both wins are about the *algorithm* — the losses, the targets, the exploration. What FastTD3 deliberately did *not* invest in is the network: it uses a plain descending ReLU MLP with no normalization, on the explicit bet that data diversity (the fast-filling buffer plus the 32,768 batch) tames the deadly triad so architecture is unnecessary. That bet is right for being fast and simple, but it leaves a specific opening. On stand the value function is gentle and the plain critic is plenty; on walk and especially run the return surface is sharper, more contact-driven, and longer-horizon, and a bare MLP critic chasing a moving bootstrapped target there is exactly the regime where unregularized capacity overfits the transient target and feeds the error back. FastTD3's answer to instability was "more diverse data, smaller-than-tempting network." The complementary answer it left on the table is "let the network scale, by giving it the bias that makes scaling safe."

I propose SimBa (Lee et al. 2025): keep the entire FastTD3 backbone untouched and change only the *function class* of the actor and critic, importing a simplicity-bias architecture that makes a larger network help instead of hurt. The motivation is that in supervised vision and language, scaling up parameters reliably improves results — not because the networks are small but because modern architectures carry an implicit *simplicity bias* from standard components (normalization, residual paths, careful init) that steer optimization toward simple, generalizable functions even when heavily over-parameterized. Deep RL has not inherited this: the standard actor-critic is a bare MLP, and widening it usually regresses because the value target moves (it is bootstrapped through the critic's own next-state value) and the data distribution drifts (the policy is changing), so raw capacity fits the transient target too hard and feeds the error back. FastTD3's plain MLP is precisely this bare network. The hypothesis I act on is that the missing thing is not RL-specific — it is the simplicity-inducing components — so importing exactly those, and nothing algorithm-specific, should let FastTD3's critic finally scale. The edit surface fits this perfectly: I can redesign the `Actor` and `Critic` networks while leaving `update_critic`, `update_actor`, the clipped double-Q logic, the distributional projection, the per-env exploration, and the whole fixed loop exactly as FastTD3 has them.

The encoder I install, applied identically to actor and critic, has three load-bearing ingredients. First, running-statistic observation normalization — but the substrate *already* provides this: `EmpiricalNormalization` runs in the fixed loop and standardizes actor obs, critic obs, and next obs online before every update. The canonical form of this architecture pairs the residual body with its own running-stat input normalization (RSNorm), but on this scaffold that normalization is a fixed-loop service, so my edit is the *body* and the input normalization is inherited rather than duplicated. Second, the residual feedforward block — the ingredient FastTD3's plain MLP lacks and the one that does the real work. Instead of a stack of fresh affine-plus-ReLU layers with no direct input-to-output path, each block computes a correction and *adds the input back*:

$$\text{out}=x+\text{Linear}_{\text{down}}\big(\text{ReLU}(\text{Linear}_{\text{up}}(\text{LayerNorm}(x)))\big).$$

The addition creates a linear identity pathway, so the simplest function a block can represent — the identity — is free, and any nonlinearity is an additive opt-in correction. That is simplicity bias made structural: the easy near-linear map is the default, complex behavior is something the optimizer must actively choose. For a bootstrapped value function chasing a moving target, defaulting to simple is exactly the regularization that stops the overfitting FastTD3's bare MLP was exposed to on the hard tasks. The block body is the standard inverted bottleneck: pre-LayerNorm the input, a linear map up to four times the width, a ReLU, a linear map back down, then the residual add. The *pre*-LN keeps the residual stream clean — the identity branch carries the un-normalized signal forward while each correction is computed from a normalized version — and the $4\times$ expansion is where the parameters that *scale* live, so widening the critic grows the bottleneck and the capacity rides along. Third, a final LayerNorm after the last block, before the head: the residual stream accumulates (it is the input plus every block's correction), so its magnitude grows with depth, and one LayerNorm right before the head standardizes the stream regardless of block count.

Initialization is the quiet piece that makes the deep stack train from step one, and I set it deliberately. The embedding linear that lifts the input into the residual stream gets orthogonal init at unit gain — a clean, well-conditioned start that neither inflates nor shrinks the representation. The two linears inside each block get He (Kaiming) normal init, the variance-preserving choice for ReLU, so the block's correction starts at a sane magnitude and neither vanishes nor explodes as blocks stack. The actor's final tanh head keeps FastTD3's small-init convention (orthogonal, tiny gain) so the deterministic policy starts near zero action — that head behavior is part of the backbone and I do not disturb it. The point of the discipline is that residual + LayerNorm + He init is the *combination* that lets a deep, wide stack train; drop any one and the same architecture can stall.

The sizing is the whole reason to do this: I want capacity to *convert*. The value function is the harder object — it has to fit a sharp, bootstrapped, contact-driven return surface — while the policy is smoother, so I make the critic the larger of the two. The canonical configuration is one residual block at hidden width 128 for the actor and two residual blocks at hidden width 512 for each twin critic, and that asymmetry is exactly the bet: the critic gets the extra capacity, the residual/normalized structure makes that capacity safe, and the actor stays lean. Against FastTD3's critic (a $1024\to512\to256$ plain MLP, roughly two affine layers of real depth with no skip) the SimBa critic is a genuinely deeper, normalized, residual network of comparable parameter scale but with the simplicity bias that lets the depth help. The optimizer pairs with it: AdamW with a modest weight decay is itself a simplicity-bias regularizer (it pulls weights toward small norm, toward simpler functions) and composes with the residual structure; the substrate already uses AdamW with weight decay $0.1$ and cosine annealing, in the same regime as the canonical SimBa decay, so I keep the substrate's value rather than introduce a new knob — the architectural change is the variable under test.

Everything else stays exactly FastTD3. The critic is still twin categorical distributional networks over 101 atoms on $[-250, 250]$; the projection, clamp, floor/ceil split, and cross-entropy target are unchanged; clipped double-Q still keeps the whole distribution whose mean is smaller; the actor is still deterministic with the per-env mixed exploration noise and target-policy smoothing, and it still updates on the delayed `policy_frequency` schedule; `num_updates=2`, $\tau=0.1$, the bootstrap mask $(\text{truncations}\,|\,\sim\!\text{dones})$, and the fixed loop's observation normalization all carry over. The only thing that changes between FastTD3 and this finale is that the network bodies become SimBa encoders — embedding, residual blocks, final LayerNorm — instead of plain MLPs. That confinement is the point: it makes the comparison a clean test of the architecture and nothing else, and it is what makes the claim "this is stronger than FastTD3" attributable to the simplicity-bias body rather than to any algorithmic change.

There is one substrate friction worth naming. FastTD3 is anti-architecture for *speed* — plain MLPs are cheaper per update, and the residual encoder adds LayerNorms and a $4\times$ bottleneck that cost wall-clock. Under a fixed *gradient-step* budget (100,000 steps) that cost is acceptable, because the budget is counted in steps, not seconds, so a heavier-but-better-converging critic is a fair trade; the risk is only that the extra compute does not pay for itself within 100k steps if the plain MLP was already near its ceiling on a given task. That risk is highest on stand, where FastTD3's plain critic is likely already sufficient, and lowest on run, where the harder value surface is exactly what the scalable critic should fit better. So the bar this finale must clear is: not worse than FastTD3 on stand (little headroom, the architecture mostly has to avoid hurting), and better than FastTD3 on walk and especially run — the tasks whose sharper, longer-horizon value surfaces are where a plain MLP critic overfits its moving target and a simplicity-biased residual critic should convert its capacity into a cleaner fit and a higher final return. The mechanism is falsifiable in the same way the central claim is: if scaling the critic with this encoder did not help on run — if the residual/normalized critic matched or trailed FastTD3's plain MLP there — then the premise that the missing ingredient is simplicity bias rather than something RL-intrinsic would be wrong.

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
