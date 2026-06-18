**Problem.** Turn a fixed set of expert MuJoCo transitions (no reward) into a reward the fixed PPO loop
can optimize. Behavioral cloning learns no reward and compounds covariate-shift error; full inverse RL
needs an inner RL loop; linear apprenticeship matching cannot pin a nonlinear expert. The first real
rung wants trajectory-level (occupancy) matching with no inner RL loop and no restrictive cost class.

**Key idea.** Occupancy-measure duality collapses regularized IRL-then-RL into one objective: make the
policy's visited distribution $\rho_\pi$ match the expert's $\rho_E$ under a divergence read off a binary
classifier. A discriminator $D$ separates expert from policy transitions; the policy is rewarded for
fooling it; at the optimum the occupancies match (JS divergence $\to 0$). The discriminator is the
adaptive cost — no inner RL loop, no linear class — and matching the full occupancy (not per-step
decisions) is what evades BC's compounding error.

**Why it works / scaffold reality.** The trust-region role the textbook method gives a TRPO policy step
is already filled by the substrate's clipped PPO, so the module supplies only the reward
$-\log(1-D)=-\texttt{logsigmoid}(-f)$ — expert-like transitions get positive reward. Two scaffold-forced
additions: (1) a **RunningNorm on the obs inputs** so the discriminator cannot cheat on raw observation
scale (untrained policies visit extreme magnitudes); without it the imitation-library GAIL configs do not
train. (2) Because `irl_batch_size` / `n_irl_updates_per_round` are fixed outside the edit region, bump
the **effective disc updates and batch inside `update()`** (`_inner_updates=4`, `_batch_mult=4`) to keep
the discriminator co-trained with the fast-moving policy.

**Hyperparameters.** Discriminator MLP $[s,a,s']\to256\to256\to1$ (ReLU); Adam at `args.irl_lr`; BCE
expert=1/policy=0; reward $-\texttt{logsigmoid}(-f)$; Welford obs running-norm refreshed each round.

**What to watch.** GAIL is the most principled but most fragile rung — the min-max game saturates on
clean demos. Expect it weakest, splitting by terminal structure: HalfCheetah (no terminal) limps to a
modest score; Hopper and Walker2d (terminate on falling) near-collapse once the reward saturates. That
fragility is the gap step 2 must fix with a *structured* reward.

```python
class _RunningMeanStd:
    """Welford running mean/std for input normalization."""

    def __init__(self, shape, device, eps=1e-4):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var = torch.ones(shape, device=device, dtype=torch.float32)
        self.count = eps

    @torch.no_grad()
    def update(self, x):
        if x.numel() == 0:
            return
        x = x.detach().to(self.mean.device, dtype=torch.float32).reshape(-1, self.mean.shape[-1])
        batch_count = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False) if batch_count > 1 else torch.zeros_like(self.var)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class RewardNetwork(nn.Module):
    """GAIL discriminator over (s,a,s'). Inputs normalized by running stats."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        input_dim = obs_dim + action_dim + obs_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self._obs_rms = None
        self._update_norm = True

    def _ensure_rms(self, ref_tensor):
        if self._obs_rms is None:
            self._obs_rms = _RunningMeanStd((self.obs_dim,), ref_tensor.device)

    def update_obs_norm(self, obs, next_obs):
        if self._obs_rms is None:
            self._ensure_rms(obs)
        if self._update_norm:
            self._obs_rms.update(obs)
            self._obs_rms.update(next_obs)

    def _norm_obs(self, x):
        if self._obs_rms is None:
            return x
        return self._obs_rms.normalize(x)

    def forward(self, state, action, next_state):
        x = torch.cat([self._norm_obs(state), action, self._norm_obs(next_state)], dim=-1)
        return self.net(x).squeeze(-1)


class IRLAlgorithm:
    """GAIL — Generative Adversarial Imitation Learning.

    Discriminator D(s,a,s') is trained to output high logits for expert
    transitions, low for policy. Policy reward is the imitation-library
    standard transform: -log(1 - D) = -logsigmoid(-logit).
    """

    def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
        self.reward_net = reward_net
        self.expert_demos = expert_demos
        self.device = device
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.irl_lr)
        self.total_updates = 0

        # Bump effective disc updates / batch (args are FIXED).
        self._inner_updates = 4
        self._batch_mult = 4

    def compute_reward(self, obs, acts, next_obs):
        with torch.no_grad():
            logits = self.reward_net(obs, acts, next_obs)
        return -F.logsigmoid(-logits)

    def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
        self.total_updates += 1
        bs = self.args.irl_batch_size * self._batch_mult

        self.reward_net.update_obs_norm(policy_obs, policy_next_obs)

        n_expert = len(self.expert_demos["obs"])
        n_policy = len(policy_obs)

        last = {}
        for _ in range(self._inner_updates):
            expert_idx = torch.randint(0, n_expert, (bs,))
            expert_obs = self.expert_demos["obs"][expert_idx]
            expert_acts = self.expert_demos["acts"][expert_idx]
            expert_next_obs = self.expert_demos["next_obs"][expert_idx]

            policy_idx = torch.randint(0, n_policy, (bs,))
            gen_obs = policy_obs[policy_idx]
            gen_acts = policy_acts[policy_idx]
            gen_next_obs = policy_next_obs[policy_idx]

            expert_logits = self.reward_net(expert_obs, expert_acts, expert_next_obs)
            gen_logits = self.reward_net(gen_obs, gen_acts, gen_next_obs)

            logits = torch.cat([expert_logits, gen_logits], dim=0)
            labels = torch.cat([
                torch.ones(bs, device=self.device),
                torch.zeros(bs, device=self.device),
            ], dim=0)

            loss = F.binary_cross_entropy_with_logits(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                acc = ((logits > 0).float() == labels).float().mean().item()
            last = {"irl_loss": loss.item(), "disc_acc": acc}

        return last
```
