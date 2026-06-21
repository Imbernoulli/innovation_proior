The data this harness hands me is unlike the locomotion benchmarks where most offline algorithms were tuned: the D4RL `human-v1` buffer holds roughly twenty-five human teleoperation trajectories on a 24-DoF hand, a thin near-expert tube of states and actions embedded in a 24-to-30-dimensional action space. I get one million gradient steps at batch 256, I never touch the environment, and I am judged by the D4RL normalized score on Pen, Hammer and Door. The two failure modes are sharp and opposite. Lean on value-based RL the online way, and the critic's bootstrap will query actions the demos never contain at a state; the network extrapolates there — almost always upward, since the actor is trained to maximize $Q$ — and that inflated value backs up through the Bellman recursion until the policy is chasing pure extrapolation off the demo tube. Lean entirely on imitation, and I am capped at the demonstrators; on twenty-five trajectories that cap is real but not catastrophic, which is exactly why behavior cloning is a serious competitor here and the thing I have to beat. I want the cheapest algorithm that does *some* value-based improvement over the demos while staying provably inside their support. The prior offline fixes (BCQ, Fujimoto et al. 2019; BEAR, Kumar et al. 2019; BRAC, Wu et al. 2019) all enforce a constraint $D(\pi, \pi_\beta) \le \varepsilon$ by *fitting a behavior model* $\hat\pi_\beta$ — but on twenty-five narrow trajectories in a 30-dim action space, fitting an accurate $\hat\pi_\beta$ is itself a hard density-estimation problem, and a constraint that pins the policy to a *bad* estimate is worse than none. I want the stay-near-data behavior without ever fitting a behavior model.

I propose AWAC — Advantage-Weighted Actor-Critic. The idea is to write the constrained policy-improvement step exactly and solve it, so the behavior model cancels analytically rather than having to be fit. I want to push up the advantage $A^{\pi_k}(s,a) = Q^{\pi_k}(s,a) - V^{\pi_k}(s)$ (maximizing $\mathbb{E}_\pi[Q]$ equals maximizing $\mathbb{E}_\pi[A]$, since $V$ does not depend on the action), subject to a KL trust region around the behavior policy:

$$\pi_{k+1} = \arg\max_\pi \; \mathbb{E}_{a\sim\pi}\big[ A^{\pi_k}(s,a) \big] \quad \text{s.t.} \quad \mathrm{KL}\big(\pi(\cdot|s) \,\|\, \pi_\beta(\cdot|s)\big) \le \varepsilon, \quad \int \pi(a|s)\,da = 1.$$

Form the Lagrangian with multiplier $\lambda$ on the KL and $\alpha$ on normalization, differentiate with respect to the value $\pi(a|s)$ at a single action, and set to zero: $A(s,a) - \lambda(\log\pi - \log\pi_\beta + 1) - \alpha = 0$. Solving for $\log\pi$ and folding the action-independent constants into a per-state normalizer $Z(s)$ gives the optimal constrained policy as the behavior policy reweighted by the exponentiated advantage,

$$\pi^*(a|s) = \frac{1}{Z(s)}\,\pi_\beta(a|s)\,\exp\!\Big( \frac{A^{\pi_k}(s,a)}{\lambda} \Big).$$

Here $\lambda$, the multiplier on the KL, is a temperature: small $\lambda$ sharpens toward the highest-advantage actions (aggressive improvement), large $\lambda$ flattens toward $\pi_\beta$ (cautious, BC-like). The behavior model is still sitting in $\pi^*$, and the decisive step is the projection onto my parametric actor $\pi_\theta$ — specifically the *direction* of the KL in that projection is what makes $\pi_\beta$ cancel. Project by minimizing the *forward* KL, averaged over the data states: $\arg\min_\theta \mathbb{E}_\rho[\mathrm{KL}(\pi^* \| \pi_\theta)] = \arg\min_\theta \mathbb{E}_\rho \mathbb{E}_{a\sim\pi^*}[-\log\pi_\theta(a|s)]$, since only the $-\log\pi_\theta$ term depends on $\theta$. I cannot sample $\pi^*$ directly, but $\pi^*$ is just $\pi_\beta$ reweighted, so I importance-sample from the buffer: $\mathbb{E}_{a\sim\pi^*}[-\log\pi_\theta] = \mathbb{E}_{a\sim\pi_\beta}[(\pi^*/\pi_\beta)(-\log\pi_\theta)] = \mathbb{E}_{a\sim\pi_\beta}[\frac{1}{Z(s)}\exp(A/\lambda)(-\log\pi_\theta)]$. The $\pi_\beta$ factor cancels, and the actor update becomes a *weighted maximum likelihood* on samples drawn straight from the buffer:

$$\theta \leftarrow \arg\max_\theta \; \mathbb{E}_{(s,a)\sim\text{buffer}}\Big[ \exp\!\Big(\frac{A^{\pi_k}(s,a)}{\lambda}\Big)\,\log\pi_\theta(a|s) \Big].$$

This is supervised learning on the dataset's own actions, each $(s,a)$ weighted by its exponentiated advantage. The constraint is enforced *implicitly*: reweighting the buffer's actions can never put mass on an action the data did not contain, yet it concentrates that mass on the high-advantage ones. No behavior model anywhere, and — crucially for narrow human data — the actor never queries $Q$ at a policy-proposed off-support action during improvement. The reverse KL would drag both the behavior model and the OOD-$Q$ query back in (it evaluates $\log\pi_\beta$ and samples $a\sim\pi_\theta$), so forward KL is exactly the right call: it is what lets me sample from the buffer and cancel $\pi_\beta$. The per-state $Z(s) = \mathbb{E}_{a\sim\pi_\theta}[\exp(A/\lambda)]$ in the weight I drop, normalizing weights across the minibatch instead; it is a per-*state* factor, so it only reweights how much different states count, not how actions compete within a state, and estimating it would inject variance like a degenerate importance weight.

That leaves the critic and the advantage. I want $A = Q - V$ from an off-policy bootstrapped $Q^\pi$ of the *current* policy — this is what makes the method improve past one step, unlike Monte-Carlo behavior-value methods. I bootstrap a twin-$Q$ TD target with the $\min$ of the targets and a Polyak target update to keep overestimation in check: $y = r + \gamma\,\min_i \bar{Q}_i(s', a')$ with $a' \sim \pi(\cdot|s')$. And since $V(s) = \mathbb{E}_{a\sim\pi}[Q(s,a)]$, I estimate it by evaluating the critics at an action sampled from the current policy and taking the same $\min$, so $A(s,a) = Q(s,a) - \min_i Q_i(s, a_\pi)$ with $a_\pi \sim \pi(\cdot|s)$. Per step: a critic update on the bootstrapped MSE for both twins, then the advantage-weighted MLE actor update with detached critics, with the weight $\exp(A/\lambda)$ clipped at $100$ so a handful of huge advantages cannot dominate the loss, then a soft update of both target critics at $\tau = 5\times10^{-3}$.

Several choices are forced by this harness specifically. This is a *purely offline* run — the method's headline strength, that the same update flows from offline pre-training into online fine-tuning unchanged, is simply unused, leaving the offline half that is most exposed to the implicit constraint being only as good as the advantage estimate. Batch and width are fixed at 256, so I cannot reach for the larger batch or wider nets that smoothed the advantage-weighting variance elsewhere. The temperature is the most consequential knob: I set $\lambda = 0.1$ to match the reference manipulation configuration, knowing it is a sharp, high-variance reweighting on twenty-five trajectories. And the policy parameterization is *not* the squashed Tanh-Gaussian the scaffold ships: advantage-weighted MLE evaluates $\log\pi_\theta(a|s)$ on dataset actions, and a TanhTransform makes that log-prob awkward near the action-box boundary where the demos sit. So I use a plain Gaussian over a $3\times256$ trunk with a *state-independent* `log_std` clamped to $[-20, 2]$, computing the log-prob directly with no Tanh correction — the actor clamps its samples into $[-1,1]$ but the density is a clean Normal. The critic returns its output *un-squeezed*, shape `(batch, 1)`, because the advantage and TD arithmetic here keep the trailing dimension, and I keep separate optimizers for the two critics. I turn on state normalization (`CONFIG_OVERRIDES = {"normalize": True}`), whitening the 24-to-30-dim hand state to help the MLPs.

The weak spot I expect to expose is that the advantage is $Q - V$ with $V$ read off a single policy sample, so the "how good is this action relative to alternatives" signal is exactly as noisy as one stochastic critic query; and the actor is pinned to demo actions by an MLE with no explicit machinery for *staying calibrated* when the data is this thin. So I anticipate a respectable but volatile Pen — a seed whose critic happens to calibrate the demo advantages flies, one whose critic is mis-calibrated collapses — with near-floor Hammer and near-zero Door, since the long precise contact sequence has to flow through that same noisy single-sample advantage and the implicit constraint welds the policy to the demonstrations. The direction that fix points is to build the constraint into the *target construction*, so the critic itself never trusts an off-support next action.

```python
# EDITABLE region of custom_adroit.py — step 1: AWAC
CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": True}


class Actor(nn.Module):
    """AWAC GaussianPolicy — 3x256 MLP, state-independent log_std, Normal + clamp."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 min_log_std: float = -20.0, max_log_std: float = 2.0):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std

    def _get_policy(self, state: torch.Tensor):
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        return torch.distributions.Normal(mean, log_std.exp())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        return policy.log_prob(action).sum(-1, keepdim=True)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(-1.0, 1.0)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        return action_t[0].cpu().numpy()


class Critic(nn.Module):
    """Q-function Q(s, a). 3x256 MLP, returns (batch, 1)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._mlp(torch.cat([state, action], dim=-1))


class OfflineAlgorithm:
    """AWAC — Advantage Weighted Actor-Critic for offline RL."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_lr=3e-4, orthogonal_init=True, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        self.awac_lambda = 0.1
        self.exp_adv_max = 100.0

        self.actor = Actor(state_dim, action_dim, 256).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_1 = Critic(state_dim, action_dim, 256).to(device)
        self.critic_2 = Critic(state_dim, action_dim, 256).to(device)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        log_dict: Dict[str, float] = {}

        # Critic update
        with torch.no_grad():
            next_actions, _ = self.actor(next_states)
            q_next = torch.min(
                self.target_critic_1(next_states, next_actions),
                self.target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self.discount * (1.0 - dones) * q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Actor update (advantage-weighted)
        with torch.no_grad():
            pi_action, _ = self.actor(states)
            v = torch.min(self.critic_1(states, pi_action), self.critic_2(states, pi_action))
            q = torch.min(self.critic_1(states, actions), self.critic_2(states, actions))
            adv = q - v
            weights = torch.clamp_max(torch.exp(adv / self.awac_lambda), self.exp_adv_max)

        action_log_prob = self.actor.log_prob(states, actions)
        actor_loss = (-action_log_prob * weights).mean()
        log_dict["actor_loss"] = actor_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

        return log_dict
```
