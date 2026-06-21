The module slot is the entire problem, and the scaffold default does nothing in it: an MLP over $[s,a,s']$ trained against a literal zero loss, so the reward net never moves and the policy ascends pure noise. That is not even a floor I can read — it is a non-run. The first real rung has to be the simplest *learning* fill of this contract that turns the fixed pile of expert transitions into a usable reward. The two standard routes both fail this exact setting. Behavioral cloning fits $\pi(a\mid s)$ from the expert $(s,a)$ pairs but learns no reward at all, and structurally it scores single-timestep decisions under the expert's own state distribution — the moment the learner's small errors push it off that distribution, errors compound and the regret over a horizon $T$ grows like $\varepsilon T^2$ rather than $\varepsilon T$. Maximum-causal-entropy IRL fixes the compounding error by scoring whole trajectories, but its maximum-likelihood gradient has a negative phase that is an expectation under the model's own trajectory distribution — the intractable partition function $Z$ — which the classic solvers estimate by running a full RL problem in an inner loop for every cost update, an expense the continuous-control budget cannot pay. Apprenticeship learning escapes that inner loop by matching expected features over a small linear cost class, but it only pins the expert down if a cost that truly explains the expert lives in that class, which for a low-dimensional linear class on MuJoCo it does not.

I propose GAIL, generative adversarial imitation learning: the method that sits between these, getting trajectory-level (occupancy) matching with no inner RL loop and no restrictive linear cost class. The reframing that delivers it is occupancy-measure duality. Write a policy not as $\pi(a\mid s)$ but as its occupancy measure $\rho_\pi(s,a)=\sum_t\gamma^t P(s_t=s,a_t=a\mid\pi)$, the discounted distribution of state-action pairs it actually visits. Expected cost is linear in $\rho$, the set of valid occupancies is a convex polytope (Bellman flow constraints), and policies and occupancies are in bijection. In these variables, $\psi$-regularized IRL followed by RL collapses — with no cost function ever materialized — into a single optimization: find the policy whose occupancy $\rho_\pi$ is closest to the expert's $\rho_E$, where "closest" is the convex conjugate $\psi^*$ of the cost regularizer. Choosing the regularizer *is* choosing the distance. A constant regularizer forces exact matching everywhere, degenerate on finitely many samples; a cost-class indicator gives back the linear apprenticeship learners that cannot imitate exactly. What I want is a regularizer whose induced distance is a smooth, finite divergence between the two occupancy distributions, minimized only when they are equal.

A divergence between two distributions is exactly what a binary classifier reads off. Train a discriminator $D$ to tell expert $(s,a,s')$ transitions from policy transitions; the better it separates them, the further apart the occupancies, and when it is reduced to chance they match. With the logistic surrogate, the optimal-classifier objective is, up to a constant, the Jensen-Shannon divergence between $\rho_\pi$ and $\rho_E$ — zero iff the occupancies are equal. So the imitation objective becomes adversarial: the discriminator separates expert from policy transitions, the policy (the generator) adjusts to fool it, and the discriminator *is* the adaptive cost supplied fresh each round — no inner RL loop, no linear cost class. When $D$ can no longer separate them, the policy has matched the expert's full visited occupancy rather than per-step decisions, so it evades BC's compounding covariate shift.

Landing this in the scaffold forces several departures from the textbook adversarial-imitation story, and the contract is rigid enough that I have to respect each. First, the textbook algorithm alternates a discriminator step with a *trust-region* (TRPO/natural-gradient) policy step, because a raw, over-aggressive policy gradient on a noisy adversarial reward lets the policy lurch into garbage. But the policy learner here is the fixed clipped-PPO loop, which I cannot touch — and PPO's clipped surrogate is itself a trust region (a flat spot past $\pm\epsilon$), so the trust-region role is already filled by the substrate. My module supplies only the reward. Second, the reward I hand PPO. I train $D$ with binary cross-entropy on the logit $f(s,a,s')$, expert labeled $1$ and policy $0$, and serve PPO the imitation-library standard transform $-\log(1-D)$. With $D=\sigma(f)$, $1-D=\sigma(-f)$, so

$$-\log(1-D) = -\log\sigma(-f) = -\,\texttt{logsigmoid}(-f),$$

the exact form in `compute_reward`. This branch gives expert-like transitions *positive* reward rather than merely avoiding a penalty, which is right when the policy is scored on accumulated return — I want it paid for visiting expert-like states, not just spared a cost.

Third, and the part the scaffold forces that the generic method never mentions, the discriminator can cheat. Early in training an untrained MuJoCo policy flails into extreme joint angles and velocities, so the discriminator can classify expert-vs-policy on raw observation *magnitude* alone, learn nothing about behavior, and hand back a reward that just says "be small." The fix is a running mean/std normalization on the reward net's observation inputs: a Welford `_RunningMeanStd` updated each round on the freshest policy rollout, so the discriminator sees whitened observations and must separate on *structure*, not scale. `RewardNetwork` holds a lazily-initialized `_obs_rms`, normalizes both $s$ and $s'$ before the MLP, and `update_obs_norm` refreshes the stats each `update()`; the action is left unnormalized because it is already bounded. Without input normalization the imitation-library GAIL configs do not train at all. Fourth, the budget knobs. The scaffold's `irl_batch_size` and `n_irl_updates_per_round` are fixed outside my editable region, and a single small discriminator step per round against a fast-moving PPO policy lets the discriminator fall behind and the game destabilize. Since I cannot edit the args, I bump them *inside* `update()`: an inner loop of `_inner_updates=4` discriminator gradient steps on a `_batch_mult=4` larger effective batch, resampling fresh expert and policy minibatches each inner step. The architecture is the default-shaped MLP, $[s,a,s']\to256\to256\to1$ with ReLU, kept inside the parameter budget the scaffold enforces.

One thing I want stated plainly, because it is the point of running this rung first: GAIL is the most *principled* method on the ladder — exact occupancy matching, the right divergence — but also the most *fragile*, and the fragility is structural to the min-max game, not a bug I can tune away from inside this contract. The discriminator and policy chase each other; if the discriminator gets too strong too fast, its logits saturate, $-\log(1-D)$ flattens, and the policy reward loses its gradient. I have no trust-region control over the policy (PPO's clip is generic, not tuned to the adversarial reward), and the reward is recomputed every step from a moving discriminator, so the PPO value function is chasing a non-stationary target. On clean, dense MuJoCo demos this dominates: the demonstrations are tight and easy to separate, so the discriminator tends to *win* — high accuracy, saturated logits, a vanishing or adversarially-shaped reward — and the policy collapses to a degenerate gait the saturated reward no longer penalizes. My falsifiable expectation, against which the next rung's numbers will be read: GAIL should be the weakest learner and fail unevenly by exactly this saturation mechanism — on HalfCheetah, which has no terminal state and accumulates return over a fixed-length episode even from a partly-collapsed gait, a modest score in the low thousands; on Hopper and Walker2d, which terminate the moment the body falls, near-collapse toward zero. If that split appears, the diagnosis is confirmed and the next rung must make the recovered reward *structured* enough to stop saturating.

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
