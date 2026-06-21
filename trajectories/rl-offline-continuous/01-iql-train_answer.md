The scaffold I am handed has a `train` that does nothing, and the disease it is silent about is the one that decides everything else. I have a fixed dataset $D$ of transitions $(s, a, r, s', a')$, no environment to query, and I want a policy better than whoever collected $D$. The standard tool — minimize the TD error toward $r + \gamma \max_{a'} Q(s', a')$ and act greedily — is poison offline, for a single precise reason. The bootstrap target contains a maximization (or, in continuous control, an actor that ascends $Q$) over the next action $a'$, and that ranges over actions that never appear in $D$ at $s'$. On those out-of-distribution actions the critic has no data; it extrapolates, and because the max actively hunts for whichever action the network happens to over-value, the extrapolation is almost always *upward*. The inflated target backs up through the Bellman recursion, the policy steers toward the over-valued actions, and with no fresh transitions to correct the mistake, the error compounds into divergence. Every fix in the literature — generative-model constraints (BCQ, BEAR; Fujimoto et al. 2019, Kumar et al. 2019), conservative value penalties (CQL; Kumar et al. 2020), behavior-regularized actor-critics (BRAC, AWAC; Wu et al. 2019, Nair et al. 2020) — still, somewhere in its loop, queries a learned $Q$ at an action that was not in the data: BCQ's generative model can emit OOD actions, CQL must sample OOD actions precisely to push them down. I want a *floor* whose value learning never evaluates the critic at a single action outside the dataset, so that any later rung's gains can be credited to mechanisms beyond "don't touch unseen actions."

I propose **IQL — implicit Q-learning**. The starting observation is that the safe in-sample target, the SARSA target $r + \gamma Q(s', a')$ with the *dataset's own* next action $a'$ (which the scaffold even hands me as `next_actions`), never touches an OOD action — but MSE fits $Q$ to the *mean* of its targets, so its fixed point is $Q^{\pi_\beta}$, the value of the behavior policy. That is pure evaluation, with no improvement and no iterated dynamic programming, so it cannot *stitch* — cannot carry the value of a good downstream state backward across transitions belonging to different dataset trajectories. What I actually want in the backup is neither the mean over $a'$ nor the unrestricted max, but the value of the *best in-support action*: a max restricted to actions the behavior policy could plausibly produce at $s'$. That restriction is the whole game — it improves (so iterating it does real DP and can stitch) yet never reaches an OOD action (so it stays safe) — but I cannot compute it by enumerating or sampling actions and querying $Q$ at each, because that is exactly the OOD query I am refusing.

The reframe that makes it computable is to treat $Q(s, a')$, as $a'$ ranges over the behavior distribution at a fixed $s$, as a *random variable* whose randomness comes from the action. SARSA's MSE estimates its mean; what I want is the upper edge of its support. The statistic that climbs into the upper tail of a random variable, estimable by a one-line reweighting of MSE on in-sample data, is the **expectile**. The $\tau$-expectile minimizes the asymmetric squared loss

$$L_2^\tau(u) = \big|\,\tau - \mathbf{1}(u < 0)\,\big|\cdot u^2,$$

which weights a positive residual (a sample above the estimate) by $\tau$ and a negative one by $1-\tau$. At $\tau = 0.5$ both weights are $\tfrac12$ and this is plain MSE, so the $0.5$-expectile is the mean; for $\tau > 0.5$ samples *above* the estimate count for more, pulling it up, and as $\tau \to 1$ it approaches the supremum of the support. That supremum is precisely the in-support max I wanted, now expressed as a regression I can run with SGD on dataset actions only.

The load-bearing subtlety is that I must take this expectile over *actions only*. The naive move — putting the asymmetric loss directly on the TD residual $r + \gamma Q(s', a') - Q(s, a)$ — is wrong, because that target carries two sources of randomness: the action $a'$, which I *want* to be optimistic about (the best $a'$ is the improvement signal), and the stochastic transition $s'$, which I emphatically do *not* (optimism over dynamics rewards a target that is high merely because the dice landed on a lucky next state, and that optimism compounds into a wildly over-valued function). So I split the labor across two networks. A value network $V(s)$ takes the upper expectile over actions with the transition fixed: regress $V(s)$ asymmetrically against the target critic's $Q(s, a)$ over dataset $(s, a)$, with advantage $A = Q(s,a) - V(s)$ and loss $L_2^\tau(A)$. Then $Q$ is backed up onto $r + \gamma V(s')$ by an *ordinary* MSE that averages honestly over the transition. $V$ does the optimistic in-support action selection; $Q$ does the honest mean over dynamics; both losses touch only dataset $(s, a, s')$, and no policy and no OOD action appear anywhere in value training. This is provably a spectrum — at $\tau = 0.5$ it is SARSA policy evaluation of $\pi_\beta$, and as $\tau \to 1$ it is Q-learning restricted to in-support actions — so $\tau$ is the dial between safety and improvement. I set $\tau = 0.7$ rather than slamming it to $1$, because a large $\tau$ leans on extreme upper residuals and is a harder, higher-variance fit; $0.7$ sits high enough to improve over $\pi_\beta$ on locomotion without the variance blow-up.

The per-step update order is dictated by the dependencies. First I snapshot $\texttt{next\_v} = V(s')$ under no-grad, since the $Q$ update needs a stable $V$ at the next state. Then the $V$ update, with `target_q` the *min* over the two target critics of $Q(s, a)$ — clipped double-Q, so the $V$ regression cannot chase an inflated single critic — and $V$ stepped on $L_2^{0.7}(\texttt{target\_q} - V(s))$. Then the $Q$ update: targets $r + (1-\text{done})\,\gamma\,\texttt{next\_v}$, both online critics regressed to it by MSE, followed by a Polyak update of both target critics at $\tau_{\text{Polyak}} = 5\times10^{-3}$. The honest MSE here is correct *because* $V(s')$ has already done the optimistic action selection; what remains is to average $\gamma V(s')$ over the transition, which is exactly what a mean does.

The policy must obey the same commandment — never query $Q$ at an unseen action — so I cannot do argmax or DDPG-style $\nabla_a Q$ ascent, both of which evaluate $Q$ at the policy's possibly-OOD action. I extract by advantage-weighted regression. The KL-constrained improvement problem has the closed form $\pi^*(a|s) \propto \pi_\beta(a|s)\exp(A(s,a)/\lambda)$, and projecting it onto the parametric policy by weighted maximum likelihood gives the loss $\exp\!\big(\beta\,(Q - V)\big)\cdot\big(-\log\pi(a|s)\big)$ averaged over dataset $(s, a)$ — a pure reweighting of *observed* actions by how advantaged they are, with an implicit stay-near-$\pi_\beta$ constraint baked in and no OOD query anywhere. The inverse temperature $\beta = 3.0$ interpolates between behavior cloning ($\beta \to 0$) and greedy concentration ($\beta \to \infty$); since a handful of transitions can carry huge advantages whose $\exp$ weights would swamp the loss, I clamp $\exp(\beta A)$ at $\texttt{EXP\_ADV\_MAX} = 100$. I cosine-anneal the actor learning rate over the full $10^6$ steps so the weighting can be aggressive early and settle late, and the actor is a *state-independent* log-std Gaussian — a Tanh-mean MLP plus a single learned log-std vector — because advantage-weighted maximum likelihood needs only clean log-probabilities of dataset actions, not a full reparameterized squashing. At evaluation I roll out the distribution mean. Critic and value nets are two-hidden-layer 256-unit MLPs, and I leave rewards exactly as the dataset gives them — no D4RL reward rescaling — so this floor is the unadorned IQL.

```python
# EDITABLE region of custom.py — step 1: IQL
from torch.optim.lr_scheduler import CosineAnnealingLR

EXP_ADV_MAX = 100.0

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Actor(nn.Module):
    """GaussianPolicy for IQL — state-independent log_std, forward returns Normal."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 orthogonal_init: bool = False):
        super().__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def forward(self, state: torch.Tensor) -> Normal:
        mean = self.net(state)
        std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class Critic(nn.Module):
    """Q-function Q(s, a). 2 x 256 MLP (IQL reference architecture)."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class ValueFunction(nn.Module):
    """State value function V(s). 2 x 256 MLP (IQL reference architecture)."""

    def __init__(self, state_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class OfflineAlgorithm:
    """IQL — Implicit Q-Learning with expectile regression and advantage-weighted actor."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        replay_buffer=None,
        discount: float = 0.99,
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        orthogonal_init: bool = True,
        device: str = "cuda",
    ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        # IQL hyperparameters
        self.beta = 3.0
        self.iql_tau = 0.7

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, int(1e6))

        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_1_target = deepcopy(self.critic_1).requires_grad_(False).to(device)
        self.critic_2_target = deepcopy(self.critic_2).requires_grad_(False).to(device)
        self.q_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=critic_lr,
        )

        self.vf = ValueFunction(state_dim, orthogonal_init).to(device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=critic_lr)

    def _update_v(self, observations, actions, log_dict):
        with torch.no_grad():
            target_q = torch.min(
                self.critic_1_target(observations, actions),
                self.critic_2_target(observations, actions),
            )
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(self, next_v, observations, actions, rewards, dones, log_dict):
        targets = rewards + (1.0 - dones.float()) * self.discount * next_v.detach()
        q1 = self.critic_1(observations, actions)
        q2 = self.critic_2(observations, actions)
        q_loss = (F.mse_loss(q1, targets) + F.mse_loss(q2, targets)) / 2.0
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)

    def _update_policy(self, adv, observations, actions, log_dict):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        observations, actions, rewards, next_observations, dones, *_ = batch
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)

        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict
```
