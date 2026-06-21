AWAC confirmed its design and exposed its limit. Hammer-expert recovered to a mean of 126.8 (almost no seed spread) — removing the deterministic-maximizer OOD query did stop SPOT's collapse to 2.5, so on clean expert data an actor that only reweights logged actions retains competence. Pen-cloned held at 63.7, a touch above SPOT. But hammer-cloned *collapsed* to 0.336 — dead flat, near zero on every seed, below even SPOT's one-lucky-seed 19.7. Hammer-cloned is mostly noise with rare good demonstrations buried in it, and AWAC reweights only logged actions by $\exp(\text{adv}/0.1)$; with that sharp temperature the weight concentrates on a tiny handful of transitions, and if the critic cannot reliably tell the rare good actions from the noise the weights smear uselessly and the policy never finds the manipulation behavior. The binding constraint has moved: SPOT's problem was the actor's OOD query; AWAC fixed that but left the *critic* doing only behavior-policy evaluation. Its $Q$ is $Q^{\pi_\beta}$ — it learns the value of the *average* behavior action — and its only improvement lever is one reweighting step away from $\pi_\beta$, so it cannot *stitch* the rare good fragments scattered across noisy trajectories into a value function for the good behavior. The improvement must move *into the value function itself*, while still never querying an unseen action.

I propose **IQL** — implicit Q-learning. Name precisely what the safe-but-insufficient piece is: SARSA's MSE fits $Q(s,a)$ to the *mean* of the TD targets over dataset actions, which is exactly why it only evaluates $\pi_\beta$. What I want in the backup is not the mean over $a'$ but the value of the *best in-support action* — a max restricted to actions the behavior policy could produce at $s'$, which improves (so iterating it does real dynamic programming and can stitch) yet never reaches an OOD action. The obstacle is that I cannot *compute* that restricted max directly: to take a max over in-support $a'$ I would enumerate or sample actions and query $Q$ at each — back to the OOD evaluation that broke SPOT.

What makes IQL work is reframing the max as a regression statistic. Fix a state $s$; as $a'$ ranges over $\pi_\beta(\cdot\mid s)$, the quantity $Q(s, a')$ is a *random variable* whose randomness comes from the action. SARSA's MSE gives its *mean*; the *maximum over the support* is what improvement needs. So I need a statistic that sits high in this action-induced distribution, estimable from the dataset actions at $s$ without evaluating $Q$ anywhere except at those in-data actions. Mean regression gives the mean; the upper tail comes from **expectile regression**. The $\tau$-expectile minimizes the asymmetric squared loss

$$L_2^\tau(u) = \big|\tau - \mathbf{1}(u<0)\big|\cdot u^2,$$

weighting a positive residual (a sample above the estimate) by $\tau$ and a negative one by $1-\tau$. At $\tau = 0.5$ both weights are $\tfrac12$ and the minimizer is the mean; for $\tau > 0.5$ the upper samples dominate and the estimate climbs toward the top of the distribution; as $\tau \to 1$ it approaches the supremum of the support. The argument is short: every expectile lies within the support and shares its supremum, the expectile is monotone non-decreasing in $\tau$, and a bounded monotone function has a $\tau\to1$ limit the asymmetry pushes to $x^*$. So the upper expectile, in the limit, *is* the in-support max — expressed as a regression I can do with SGD on in-sample data only.

The directest move — swapping the SARSA MSE for the expectile loss on the TD residual $r + \gamma Q(s',a') - Q(s,a)$ — breaks, and the break dictates the architecture. That target carries *two* sources of randomness: the action $a' \sim \pi_\beta(\cdot\mid s')$, which I *want* to be optimistic over (the best $a'$ is the improvement signal), and the stochastic transition $s' \sim p(\cdot\mid s,a)$, which I emphatically do *not*. An upper expectile rewards high targets indiscriminately, so it would reward a target that is high merely because the environment got lucky with the next state — conflating "there exists a better action here" with "I rolled well," and optimism over dynamics compounds over the horizon into a runaway value. So I separate the two with a value network. $V_\psi(s)$ does purely the action-expectile with the transition held fixed,

$$L_V(\psi) = \mathbb{E}_{(s,a)\sim D}\big[\,L_2^\tau\big(Q_{\hat\theta}(s,a) - V_\psi(s)\big)\,\big],$$

where both $s$ and $a$ come from $D$, so for a given $s$ the only randomness in the target is the action and $V_\psi(s)$ becomes the $\tau$-expectile of $Q$ over dataset actions — optimistic over actions, with no dynamics in it. Then I back this up into $Q$ with an *ordinary* MSE that averages the transition honestly,

$$L_Q(\theta) = \mathbb{E}_{(s,a,s')\sim D}\big[\,(r + \gamma V_\psi(s') - Q_\theta(s,a))^2\,\big].$$

The MSE is correct precisely because $V_\psi(s')$ has already done the optimistic action selection; what remains is to average $\gamma V_\psi(s')$ over $s'$, and a mean is the right way to average dynamics. The division of labor is the whole point: $V$ takes the upper expectile over actions, $Q$ takes the mean over transitions, both losses touch only dataset $(s,a,s')$, no policy appears in value training, and no OOD action is ever queried. This is genuine multi-step dynamic programming — $V_\tau$ is monotone in $\tau$, bounded by the in-support optimum, and converges to it as $\tau\to1$, spanning SARSA ($\tau = 0.5$, what AWAC's critic effectively did) to in-support Q-learning. That spectrum is the lever hammer-cloned was missing: at higher $\tau$ the critic propagates the value of rare good downstream states *backward across transitions from different noisy trajectories*, stitching the fragments AWAC could not.

Policy extraction obeys the same commandment — never query $Q$ at an unseen action. I cannot argmax $Q$ (searches OOD) or do DDPG-style ascent (evaluates $Q$ at the policy's possibly-OOD actions), so I reweight the dataset's own actions with advantage-weighted regression,

$$L_\pi(\phi) = \mathbb{E}_{(s,a)\sim D}\big[\,\exp\!\big(\beta\,(Q_{\hat\theta}(s,a) - V_\psi(s))\big)\cdot\log\pi_\phi(a\mid s)\,\big],$$

weight clipped to $\le 100$. This is the same family AWAC used — and that is fine, because the OOD-safe extraction was never the problem; the fix for hammer-cloned is *what advantage feeds it*. AWAC's advantage came from a $Q^\pi$ that only evaluated $\pi_\beta$; IQL's comes from a $Q$ that has done in-support dynamic programming, so the weights now reward actions good *for the stitched optimal-in-support policy*, not just good relative to the behavior average. Same extraction, far stronger signal.

In this harness the IQL fill is specific. The actor is a $2\times256$ `GaussianPolicy` with a Tanh output activation and a state-independent `log_std` (`nn.Parameter`), a plain `Normal` (not `TanhTransform`), with **dropout 0.1** in the actor MLP and a **CosineAnnealingLR** schedule over the 1M offline steps — both stabilizers that matter on the noisy `cloned` data. The critic is `TwinQ` (two $2\times256$ MLPs, squeezed, with a `.both()` for the twin outputs), and `ValueFunction` is a $2\times256$ squeezed MLP. The hyperparameters match the CORL Adroit config: $\texttt{iql\_tau} = 0.8$ (the expectile, high enough to do real improvement on the manipulation tasks), $\beta = 3.0$ (advantage temperature, softer than AWAC's effective $1/0.1 = 10$ sharpness, which on noisy data spreads the weights over more transitions rather than betting on a few), and $\texttt{exp\_adv\_max} = 100$. The update order per step is V (expectile) → Q (MSE onto $r + \gamma V(s')$) → Polyak the target critic → policy (advantage-weighted), with the actor LR cosine-stepping. The decisive transition fact: `on_online_start` is a *no-op* — IQL needs no special handling at the handoff, for the same reason AWAC did not (value training is policy-free and the same update runs offline and online as the buffer grows). IQL keeps AWAC's transition-robustness while replacing the critic's behavior-evaluation with in-support dynamic programming.

```python
def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class Actor(nn.Module):
    """IQL GaussianPolicy — 2x256 MLP with Tanh output, state-independent log_std, Normal dist."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hidden_dim: int = 256, n_hidden: int = 2, dropout: float = 0.1):
        super().__init__()
        dims = [state_dim] + [hidden_dim] * n_hidden + [action_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.max_action = max_action
        self._log_std_min = -20.0
        self._log_std_max = 2.0

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(self._log_std_min, self._log_std_max))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class TwinQ(nn.Module):
    """Twin Q-functions Q1(s,a), Q2(s,a). 2x256 MLPs, squeezed output."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim + action_dim] + [hidden_dim] * n_hidden + [1]

        def _build_mlp():
            layers = []
            for i in range(len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(dims[-2], dims[-1]))
            return nn.Sequential(*layers)

        self.q1 = _build_mlp()
        self.q2 = _build_mlp()

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    """State value function V(s). 2x256 MLP, squeezed output."""

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim] + [hidden_dim] * n_hidden + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.v = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state).squeeze(-1)


class OfflineOnlineAlgorithm:
    """IQL — Implicit Q-Learning for offline-to-online RL."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        self.iql_tau = 0.8
        self.beta = 3.0
        self.exp_adv_max = 100.0

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=int(1e6)
        )

        self.qf = TwinQ(state_dim, action_dim).to(device)
        self.q_target = deepcopy(self.qf).requires_grad_(False).to(device)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=critic_lr)

        self.vf = ValueFunction(state_dim).to(device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            target_q = self.q_target(states, actions)
        v = self.vf(states)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        with torch.no_grad():
            next_v = self.vf(next_states)
        targets = rewards + (1.0 - dones) * self.discount * next_v.detach()
        qs = self.qf.both(states, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        soft_update(self.q_target, self.qf, self.tau)

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.exp_adv_max)
        policy_out = self.actor(states)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape mismatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

        return log_dict

    def select_action(self, state: np.ndarray) -> np.ndarray:
        return self.actor.act(state, self.device)

    def on_online_start(self):
        # IQL needs no special handling at the offline-to-online transition
        pass
```
