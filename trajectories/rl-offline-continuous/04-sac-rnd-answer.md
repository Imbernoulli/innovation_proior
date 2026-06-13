**Problem (from step 3).** ReBRAC tops the ladder (hc 63.35 / maze 93.95, std 56.8 -> 18.4 / walker
87.54), but its conservatism is a *fixed* L2 behavior-cloning pull `(π − a)²` — a blunt, per-dataset
hand-tuned proxy for "how OOD is this action," pulling every proposed action toward the single logged
action regardless of whether it is well-supported. What is missing is a *learned per-`(s,a)` OOD
measure*: penalize sharply where the action leaves the data, barely at all where it stays in support.

**Key idea (SAC-RND — anti-exploration by random network distillation).** Replace the fixed BC proxy
with a learned uncertainty penalty — the mirror of an online novelty bonus. A frozen random RND
*target* `g(s,a)` and a trained *predictor* `ĝ(s,a)` give `b(s,a) = ‖ĝ − g‖²` (normalized by a running
std): small on in-data actions, large on OOD ones. Naive `[s,a]`-concat RND is *escapable* (the actor
drives `b` down on OOD actions without returning to data); the fix is **FiLM conditioning of the prior
on the action** — the action emits per-unit `(γ, β)` scaling/shifting a hidden layer, making `b` a
*sharp* function of the action, minimizable only by staying in support. Subtract `β·b` from **both** the
actor objective and the critic bootstrap target.

**Why it beats a fixed BC.** L2 says "be near this one action"; the RND penalty says "be anywhere the
data supports, flee only where it does not" — strictly more expressive, so the actor can roam among
in-support actions to find higher value. It gets ensemble-quality pessimism from one small network,
admissible under the ≈1.05× parameter cap that forbids the SAC-N/EDAC ensembles.

**Base / hyperparameters.** SAC: Tanh-Gaussian actor, twin LayerNorm critics with `min`, auto-tuned
entropy `α` to target entropy `−dim(A)`, Polyak `τ = 5e-3`, `γ = 0.99`. RND embedding 32, FiLM MLP at
locked width 256, predictor trained on dataset `(s,a)` only, target frozen, bonus running-std
normalized. Per-dataset penalty `β` from the method's sweep: halfcheetah-medium `0.3`, walker2d-medium
`8.0`, maze2d-medium `1.0`, read from the env name.

**The bar to beat.** No leaderboard SAC-RND row exists; the strongest baseline is ReBRAC
(hc 63.35 / maze 93.95 / walker 87.54). Falsifiable: clear ReBRAC on the two locomotion datasets where
its BC was bluntest (reference reports ~66.6 hc, ~91.6 walker), and *hold near 90* on Maze2d (the
honest open question for an entropy-regularized stochastic actor on a stitching task).

```python
# EDITABLE region of custom.py — finale: SAC-RND
CONFIG_OVERRIDES: Dict[str, Any] = {"normalize_reward": False}

# Per-environment anti-exploration penalty coefficient beta (from the method's
# per-dataset sweep). Detected from --env; falls back to a safe mid value.
import sys as _sys

def _detect_env() -> str:
    for i, arg in enumerate(_sys.argv):
        if arg == "--env" and i + 1 < len(_sys.argv):
            return _sys.argv[i + 1]
        if arg.startswith("--env="):
            return arg.split("=", 1)[1]
    return ""

_SACRND_BETA = {
    "halfcheetah-medium-v2": 0.3,
    "walker2d-medium-v2": 8.0,
    "hopper-medium-v2": 5.0,
    "maze2d-medium-v1": 1.0,
}
_BETA = _SACRND_BETA.get(_detect_env(), 1.0)


class FiLMMLP(nn.Module):
    """RND sub-network: an MLP over `feature`, FiLM-modulated by `context`.
    The context produces per-unit (gamma, beta) that scale/shift the first
    hidden layer -- the conditioning that makes the RND prior discriminative.
    All hidden widths are 256 (locked)."""

    def __init__(self, feature_dim: int, context_dim: int, out_dim: int):
        super().__init__()
        self.film = nn.Linear(context_dim, 2 * 256)
        self.l1 = nn.Linear(feature_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, out_dim)

    def forward(self, feature: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma, beta = torch.chunk(self.film(context), 2, dim=-1)
        h = F.relu(gamma * self.l1(feature) + beta)
        h = F.relu(self.l2(h))
        return self.out(h)


class RND(nn.Module):
    """Anti-exploration Random Network Distillation over (state, action).
    A frozen random `target` and a trained `predictor`; the squared embedding
    error is large on OOD (s, a) and small on in-data (s, a). The action is the
    FiLM `feature`, the state the `context` (switch_features=True in the
    reference): the prior is conditioned on the action so the actor can drive
    the bonus down only by staying in-distribution."""

    def __init__(self, state_dim: int, action_dim: int, embedding_dim: int = 32):
        super().__init__()
        self.predictor = FiLMMLP(action_dim, state_dim, embedding_dim)
        self.target = FiLMMLP(action_dim, state_dim, embedding_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        self.register_buffer("rms_mean", torch.zeros(()))
        self.register_buffer("rms_var", torch.ones(()))
        self.register_buffer("rms_count", torch.ones(()) * 1e-4)

    def _embed(self, state, action):
        pred = self.predictor(action, state)
        with torch.no_grad():
            targ = self.target(action, state)
        return pred, targ

    def bonus(self, state, action) -> torch.Tensor:
        pred, targ = self._embed(state, action)
        raw = ((pred - targ) ** 2).sum(dim=-1)
        return raw / (torch.sqrt(self.rms_var) + 1e-8)

    def loss_and_update_rms(self, state, action) -> torch.Tensor:
        pred, targ = self._embed(state, action)
        raw = ((pred - targ) ** 2).sum(dim=-1)
        with torch.no_grad():
            b_mean, b_var, b_n = raw.mean(), raw.var(unbiased=False), raw.numel()
            delta = b_mean - self.rms_mean
            tot = self.rms_count + b_n
            self.rms_mean += delta * b_n / tot
            m_a = self.rms_var * self.rms_count
            m_b = b_var * b_n
            self.rms_var.copy_((m_a + m_b + delta ** 2 * self.rms_count * b_n / tot) / tot)
            self.rms_count.copy_(tot)
        return raw.mean()


class TanhGaussianActor(nn.Module):
    """SAC stochastic policy: 3 x 256 trunk -> (mu, log_sigma), tanh-squashed.
    Returns squashed action and its log-prob with the tanh correction."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_sigma = nn.Linear(256, action_dim)

    def _dist(self, state):
        h = self.trunk(state)
        mu = self.mu(h)
        log_sigma = self.log_sigma(h).clamp(-5.0, 2.0)
        return Normal(mu, torch.exp(log_sigma))

    def sample(self, state):
        dist = self._dist(state)
        raw = dist.rsample()
        log_prob = dist.log_prob(raw).sum(-1)
        action = torch.tanh(raw)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        return self.max_action * action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        h = self.trunk(state)
        action = self.max_action * torch.tanh(self.mu(h))
        return action.cpu().data.numpy().flatten()


class Critic(nn.Module):
    """Q(s, a). 3 x 256 MLP with LayerNorm (as in the SAC-RND reference)."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class OfflineAlgorithm:
    """SAC-RND — Soft Actor-Critic with an anti-exploration RND penalty.

    The RND bonus b(s, a) (large on OOD actions) is SUBTRACTED, with coefficient
    beta, from BOTH the actor objective and the critic's bootstrap target, so the
    policy is pulled toward in-distribution actions and the value of OOD next
    actions is suppressed at the source. SAC's entropy temperature is auto-tuned.
    """

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_lr=3e-4, orthogonal_init=True, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0
        self.beta = _BETA

        self.actor = TanhGaussianActor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_1_target = deepcopy(self.critic_1).requires_grad_(False)
        self.critic_2_target = deepcopy(self.critic_2).requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr)

        self.rnd = RND(state_dim, action_dim, embedding_dim=32).to(device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=actor_lr)

        # SAC automatic entropy tuning: target entropy = -action_dim
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros((), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, _ = batch
        not_done = 1.0 - dones.squeeze(-1)
        rewards_flat = rewards.squeeze(-1)
        log_dict: Dict[str, float] = {}
        alpha = self.log_alpha.exp()

        # -- RND predictor update (distill the frozen target on dataset (s, a)) --
        rnd_loss = self.rnd.loss_and_update_rms(states, actions)
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()
        log_dict["rnd_loss"] = rnd_loss.item()

        # -- Critic update: subtract beta * RND bonus from the bootstrap target --
        with torch.no_grad():
            next_actions, next_logp = self.actor.sample(next_states)
            next_bonus = self.rnd.bonus(next_states, next_actions)
            target_q = torch.min(
                self.critic_1_target(next_states, next_actions),
                self.critic_2_target(next_states, next_actions),
            )
            target_q = target_q - alpha * next_logp - self.beta * next_bonus
            target_q = rewards_flat + not_done * self.discount * target_q

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        log_dict["critic_loss"] = critic_loss.item()

        # -- Actor update: maximize Q - alpha*logp - beta*bonus --
        pi_actions, logp = self.actor.sample(states)
        bonus = self.rnd.bonus(states, pi_actions)
        q_pi = torch.min(self.critic_1(states, pi_actions), self.critic_2(states, pi_actions))
        actor_loss = (alpha.detach() * logp + self.beta * bonus - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        log_dict["actor_loss"] = actor_loss.item()

        # -- Temperature update toward target entropy --
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        log_dict["alpha"] = alpha.item()

        # -- Polyak target update --
        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)

        return log_dict
```
