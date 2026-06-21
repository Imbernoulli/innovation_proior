SPOT confined the actor to the behavior support and got a respectable but uneven Pen-cloned (mean 56.4, with one seed falling to 23) and a hammer-cloned driven almost entirely by one lucky run (mean 19.7, the other two near zero). The number that diagnoses it is hammer-expert: it *collapsed* to 2.5, near random, on the *cleanest* data — clean expert demonstrations, where the VAE's support is a thin tube the density penalty should have kept the actor exactly inside. The collapse on the cleanest data is the tell. SPOT's deterministic TD3 actor still maximizes $Q(s, \pi(s))$, so its improvement step queries the critic at a self-proposed action; on expert data, once the cooled $\lambda$ opens the tube during the online phase and the actor steps a little off it, the critic has no real feedback to correct an over-valued OOD action and the value runs away. The binding problem is not the policy's support per se but the *deterministic maximizer + bootstrapped critic exposing itself to OOD values during improvement*. So the move is to keep the implicit-constraint idea but stop letting the actor's improvement step ever query $Q$ at a self-proposed action at all — and to drop the brittle separately-fit VAE, whose frozen support is either too tight (expert) or too broad (cloned).

I propose **AWAC** — advantage-weighted actor-critic. Start from the KL-constrained policy-improvement problem and solve it exactly, watching whether the behavior model can be made to vanish. Since $\max_\pi \mathbb{E}_\pi[Q]$ equals $\max_\pi \mathbb{E}_\pi[A]$ for the advantage $A(s,a) = Q(s,a) - V(s)$ (the baseline $V$ is action-independent), at iteration $k$ I solve

$$\pi_{k+1} = \arg\max_\pi\ \mathbb{E}_{a\sim\pi}\big[A^{\pi_k}(s,a)\big]\quad\text{s.t.}\quad \mathrm{KL}\big(\pi(\cdot\mid s)\,\Vert\,\pi_\beta(\cdot\mid s)\big) \le \varepsilon.$$

Forming the Lagrangian with multiplier $\lambda$ on the KL and $\alpha$ on normalization, writing the KL as $\int \pi(\log\pi - \log\pi_\beta)$, and differentiating with respect to $\pi(a\mid s)$ at a single action, the $\mathbb{E}_\pi[A]$ term gives $A(s,a)$, the $-\lambda\,\mathrm{KL}$ term gives $-\lambda(\log\pi - \log\pi_\beta + 1)$ (the $+1$ from differentiating $\pi\log\pi$), and normalization gives $-\alpha$. Setting the sum to zero and folding the action-independent constants into a per-state normalizer yields

$$\pi^*(a\mid s) = \frac{1}{Z(s)}\,\pi_\beta(a\mid s)\,\exp\!\Big(\frac{A(s,a)}{\lambda}\Big).$$

The optimal constrained policy is the behavior policy reweighted by the exponentiated advantage, with $\lambda$ a temperature: small $\lambda$ sharpens toward the highest-advantage actions, large $\lambda$ flattens toward $\pi_\beta$. The behavior model is still in there — the projection step is where it has to die.

What makes AWAC work is the *direction* of the projection onto the parametric policy $\pi_\theta$. Minimizing the **forward** KL $\mathrm{KL}(\pi^* \,\Vert\, \pi_\theta)$ over data states reduces to $\arg\min_\theta \mathbb{E}_{a\sim\pi^*}[-\log\pi_\theta(a\mid s)]$, the only $\theta$-dependent term. I cannot sample $\pi^*$ directly, but $\pi^*$ is just $\pi_\beta$ reweighted, so I importance-sample from the buffer: $\mathbb{E}_{a\sim\pi^*}[-\log\pi_\theta] = \mathbb{E}_{a\sim\pi_\beta}\big[(\pi^*/\pi_\beta)(-\log\pi_\theta)\big]$, and $\pi^*/\pi_\beta = (1/Z(s))\exp(A/\lambda)$ — the $\pi_\beta$ factor *cancels*. No behavior model is left. The actor update becomes a weighted maximum likelihood on samples drawn straight from the buffer,

$$\theta_{k+1} = \arg\max_\theta\ \mathbb{E}_{(s,a)\sim\beta}\big[\log\pi_\theta(a\mid s)\cdot\exp(A^{\pi_k}(s,a)/\lambda)\big],$$

supervised learning where each observed $(s,a)$ is weighted by its exponentiated advantage. The constraint is enforced *implicitly*: reweighting the buffer's own actions can never put mass on an action the data did not contain, yet it concentrates mass on the high-advantage ones. This is exactly the fix for the SPOT collapse — the improvement step now *reweights logged actions* instead of *maximizing $Q$ at a self-proposed action*. The actor never asks the critic "what is the value of this new action I invented?"; it only asks "of the actions I actually saw, which were good?" The OOD query the deterministic maximizer could not avoid is gone.

The forward direction is load-bearing, and the contrast with reverse KL shows why. Reverse KL $\mathrm{KL}(\pi_\theta \,\Vert\, \pi^*) = \mathbb{E}_{a\sim\pi_\theta}[\log\pi_\theta - \log\pi_\beta - A/\lambda + \log Z]$ needs two things I am fleeing: it evaluates $\log\pi_\beta$ — a density model, exactly the brittle VAE — and it samples actions from $\pi_\theta$, which offline are the possibly-OOD actions that make $Q$ extrapolate, exactly the hammer-expert collapse. The direction that lets me sample from the buffer and cancel $\pi_\beta$ is also the direction that removes both failure mechanisms at once. The per-state normalizer $Z(s) = \mathbb{E}_{a\sim\pi_\theta}[\exp(A/\lambda)]$ I drop: estimating it empirically *hurts* (the estimation error injects variance like a degenerate importance weight), and there is a clean argument it is benign — $Z(s)$ is a per-*state* factor, so it only reweights how much different states count, not how actions compete within a state, and the buffer's state distribution is already off from what $\pi_\theta$ will visit. I normalize the weights across the minibatch instead.

That leaves the critic, and here is the second deliberate departure from the SARSA/AWR lineage: I bootstrap $Q^\pi$ of the *current* policy off-policy — twin-Q with a $\min$ target and a Polyak target network to control overestimation — rather than a Monte-Carlo $V^{\pi_\beta}$ of the behavior policy. The Monte-Carlo route only supports one step of improvement away from $\pi_\beta$ and is slow; the bootstrapped $Q^\pi$ reuses off-policy data and improves iteratively, which is what lets the online phase keep climbing rather than stalling near the offline policy. The advantage the actor needs is $A(s,a) = Q(s,a) - V(s)$, and since $V(s) = \mathbb{E}_{a\sim\pi}[Q(s,a)]$ I estimate it by evaluating the min of the twin critics at an action *sampled from the current policy*. This $V$ estimate does sample $\pi$, but only to compute a baseline subtraction *inside the weight* — it never feeds an optimization that pulls the policy toward a high-$Q$ invented action, so it does not reopen the SPOT failure; the improvement direction stays pure weighted-MLE on logged actions.

In this harness the AWAC fill is specific. The actor is a $3\times256$ Gaussian with a *state-independent* `log_std` (an `nn.Parameter`, clamped to $[-20, 2]$), using a plain `Normal` with a hard action clamp — not the template's `TanhTransform` Gaussian. The critics are $3\times256$ and return $(\text{batch}, 1)$ *unsqueezed* (the template squeezes), each with its own Adam optimizer. The advantage weight is $\texttt{clamp\_max}(\exp(\text{adv}/\lambda),\,100)$ with $\texttt{awac\_lambda} = 0.1$ — a sharper temperature than IQL's $\beta$ — and the actor loss is $(-\log\text{-prob}\cdot\text{weight}).\text{mean}()$ on the *dataset* actions; the targets get standard Polyak `soft_update` each step. And the decisive structural point: `on_online_start` is a *no-op*. The same advantage-weighted update runs unchanged offline and online — the buffer simply grows with online data — so there is no transition schedule to mis-tune, none of the $\lambda$-cooling and optimizer resets SPOT had to engineer. AWAC removes the transition-handling machinery entirely, which is its own kind of robustness.

```python
class Actor(nn.Module):
    """AWAC GaussianPolicy — 3x256 MLP, state-independent log_std, Normal + clamp."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 min_log_std: float = -20.0, max_log_std: float = 2.0):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
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
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._mlp(torch.cat([state, action], dim=-1))


class OfflineOnlineAlgorithm:
    """AWAC — Advantage Weighted Actor-Critic for offline-to-online RL."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4, device="cuda"):
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

    def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            next_actions, _ = self.actor(next_states)
            q_next = torch.min(
                self.target_critic_1(next_states, next_actions),
                self.target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self.discount * (1.0 - dones) * q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        critic_loss = q1_loss + q2_loss
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        with torch.no_grad():
            pi_action, _ = self.actor(states)
            v = torch.min(
                self.critic_1(states, pi_action),
                self.critic_2(states, pi_action),
            )
            q = torch.min(
                self.critic_1(states, actions),
                self.critic_2(states, actions),
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self.awac_lambda), self.exp_adv_max
            )

        action_log_prob = self.actor.log_prob(states, actions)
        actor_loss = (-action_log_prob * weights).mean()
        log_dict["actor_loss"] = actor_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

        return log_dict

    def select_action(self, state: np.ndarray) -> np.ndarray:
        return self.actor.act(state, self.device)

    def on_online_start(self):
        # AWAC needs no special handling at transition
        pass
```
