FastSAC did what its structure promised on the easy axis and exposed the trouble on the hard one. As an off-policy method it reuses the replay buffer, so it cleared PPO comfortably across stand, walk, and run, closing the sample-efficiency shortfall the PPO step diagnosed. But it sits below where a deterministic actor would land, and the reason is the entropy tax: a stochastic tanh-Gaussian actor must maximize entropy over tens of contact-rich action dimensions, a genuinely hard optimization, and it pays for it twice — the $-\alpha\log\pi$ term inside the distributional bootstrap and the entropy penalty in the actor loss — both of which pull the policy away from greedily exploiting the value function. On stand, where exploration is cheap and exploitation is the whole game, that tax is pure cost; on walk and run the entropy helps less than hoped because the limiting factor is the *precision* of a learned gait, not the breadth of exploration. So I am paying for exploration I can get for free elsewhere, and that payment caps the policy below a pure-exploitation actor. The move is to drop the stochastic actor and the entropy machinery entirely and go deterministic — provided I can solve the exploration problem that made me reach for entropy in the first place.

I propose FastTD3: a deterministic off-policy actor-critic on the same distributional substrate, betting that the substrate already solves exploration — 128 parallel environments, each running the same deterministic actor with its own independent noise and random start, manufacture a broad data distribution with no entropy objective at all. The backbone is TD3: a deterministic actor $\mu(s)$ pushed uphill on the critic by the deterministic policy gradient, twin critics with a clipped-min bootstrap, target-policy smoothing, and delayed actor updates. The deterministic policy gradient is the source of the exploitation advantage — I differentiate the critic with respect to the action and backprop into the actor, so the actor moves straight toward the critic's argmax with no score-function variance and no entropy term holding it back. That is precisely the greedy exploitation FastSAC sacrificed. But a deterministic actor that just maximizes the critic is exactly what made DDPG diverge: the critic, bootstrapped through a target that effectively maximizes over actions, overestimates, and the actor eagerly chases the inflated value. TD3's clipped double-Q is the fix, and it transfers directly to this substrate's distributional critic the FastTD3 way — keep two categorical critics, project both target distributions, read each one's scalar mean $\sum_i p_i z_i$, and keep the *whole distribution* belonging to whichever critic has the smaller mean. The min selects a distribution by its expectation, clamping the more-overestimated of the two, and that selected distribution is the cross-entropy target for both critics. Underestimation, unlike overestimation, does not get chased and amplified by the policy, which is why the min is the right pessimism.

The crucial difference from FastSAC's critic target is what enters the projection's reward. FastSAC subtracted $\alpha\log\pi(a'|s')$ to fold in entropy; here there is no entropy term, so the projection operates on the *raw* reward, and the next action fed to the target is not a stochastic sample but the deterministic actor's output with target-policy-smoothing noise added: $a'=\operatorname{clip}(\mu(s')+\varepsilon,\,-1,1)$ with $\varepsilon$ a small clipped Gaussian. Target smoothing averages the critic over a small neighborhood of the target action so the policy cannot exploit a needle-thin spurious peak in the approximate distributional value — a SARSA-flavored regularizer saying nearby actions should have similar value. It matters more for a deterministic actor than a stochastic one, because the deterministic actor will otherwise drive straight at whatever sharp maximum the critic happens to have. The actor objective is then the pure deterministic-policy-gradient one: read the support-weighted means of the two critics at the actor's noise-free action, take the clipped-double-Q minimum, and ascend it — $\text{actor\_loss}=-\min(Q_1,Q_2).\text{mean()}$, no entropy, no log-prob, no temperature. And the actor updates on the *delayed* schedule, every `policy_frequency` critic steps rather than every step: the two-timescale trick lets the critic settle before each policy move so the actor is not chasing a critic that is still thrashing — the opposite of SAC's every-step actor update, and appropriate because a deterministic actor has nothing damping it but the delay.

Now the exploration that justifies dropping entropy. The actor is deterministic, so on its own it would trace one thin trajectory; the substrate fixes this by parallelism. Each of the 128 environments runs the same actor but adds its own Gaussian exploration noise, and — the FastTD3 touch — each environment draws its *own* noise scale once from a range $[\text{std\_min}, \text{std\_max}]$ and resamples that scale when its episode ends. So the fleet spans a spread of exploration aggressiveness simultaneously: some envs explore timidly, some boldly, and I never have to find the single right noise scale for a task because the fleet covers the range. This mixed per-env noise is the deterministic answer to FastSAC's entropy — exploration comes from a fleet of noisy copies of a deterministic policy rather than from the policy itself being stochastic — so I keep the clean deterministic-policy-gradient exploitation *and* get a broad data distribution. On a high-dimensional humanoid action this is the better trade, because injecting noise into 128 parallel rollouts is trivial whereas maximizing entropy over the action space is hard; that asymmetry is exactly what should let FastTD3 edge past FastSAC.

The architecture is where I make the other deliberate departure from FastSAC, and it goes against instinct. FastSAC used LayerNorm and SiLU to stabilize its off-policy critic; FastTD3's philosophy is that stabilization should come from *data*, not architecture. With a fast-filling, diverse replay buffer (128 envs filling it every step) and the substrate's huge 32,768 batch, each gradient update is low-variance and close to on-distribution, so the deadly triad is tamed by data diversity rather than by normalization. So the actor and critic are plain descending-width MLPs with ReLU and no LayerNorm — $512\to256\to128$ for the actor, $1024\to512\to256$ for the critic — and the deterministic actor's final tanh head is small-initialized so it starts near zero action. This is lighter and faster than FastSAC's normalized body, and the claim is that on this substrate it loses nothing in stability because the diversity is doing that job. Everything else stays matched to the substrate exactly as FastSAC had it — AdamW with weight decay $0.1$, cosine LR annealing, `num_updates=2` per env step, fast target update $\tau=0.1$, the categorical critic with 101 atoms over $[-250, 250]$, and the bootstrap mask $(\text{truncations}\,|\,\sim\!\text{dones})$ so time-limit truncations still bootstrap while true terminations do not. The FastTD3 fill is therefore the scaffold's *default* — this rung is the substrate's own baseline, and the honest framing is that I am recovering it as the strongest of the three fills by arguing my way from PPO through FastSAC to it, not adding to it.

Against FastSAC, FastTD3 should win because it removes the entropy tax while keeping exploration through the fleet, so I expect it above FastSAC on all three tasks, with the clearest margin on stand — where exploitation matters most and FastSAC's entropy was pure cost — and a narrower but still positive margin on walk and run, where some of FastSAC's entropy was doing useful exploration so the gap should shrink. Against PPO the margin should be the largest of all three methods, since FastTD3 combines experience reuse with aggressive deterministic exploitation. The risk that would falsify the ordering is if the plain MLP without LayerNorm proves *less* stable than FastSAC's normalized body on the harder-to-fit walk and run value surfaces — if the data-diversity-tames-the-triad claim fails at this scale, FastTD3's critic could be noisier and lose to FastSAC on run even while winning on stand. That precise failure mode — a plain MLP critic that does not convert its capacity into a clean value fit on the hardest task — is the opening the next rung attacks.

```python
# EDITABLE region of custom_algorithm.py — step 3: FastTD3 (deterministic, distributional) = default
class Actor(nn.Module):
    """Deterministic actor: descending ReLU MLP + tanh head, per-env Gaussian exploration noise."""
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
    critic = Critic(n_obs, n_act, args.num_atoms, args.v_min, args.v_max, args.critic_hidden_dim, device)
    critic_target = Critic(n_obs, n_act, args.num_atoms, args.v_min, args.v_max, args.critic_hidden_dim, device)
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
