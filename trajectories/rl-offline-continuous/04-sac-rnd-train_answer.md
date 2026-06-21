ReBRAC took the top of the ladder, and its numbers tell me exactly what kind of method is still missing. It posted $63.35$ on HalfCheetah, $87.54$ on Walker2d, and — the headline — $93.95$ on Maze2d with the per-seed spread collapsed from TD3+BC's std $56.8$ down to $18.4$ ($81.7 / 85.0 / 115.1$). The LayerNorm critic and the critic-target BC killed the spurious high-value pockets that made Maze2d a coin flip, and they let the actor exploit the locomotion data harder than either predecessor. But look at what the behavior-cloning machinery actually *is*, because that is where there is still room. The BC penalty, in both the actor and the critic target, is $(\pi - a)^2$ — a *fixed, uniform* notion of "stay near the data" that pulls every proposed action toward the recorded action with the same per-dataset strength, regardless of whether the proposed action is in fact well-supported or wildly outside the data. HalfCheetah needed $\texttt{actor\_bc}$ as low as $0.001$ to let the actor move at all, Walker2d as high as $0.05$ to hold the gait — the single L2 coefficient is a blunt stand-in for "how OOD is this action," and it has to be hand-tuned per dataset precisely because L2-to-the-data is a crude proxy for OOD-ness. What ReBRAC lacks is a *per-$(s,a)$ measure of how out-of-distribution a proposed action actually is*: with that I could penalize sharply exactly where the action leaves the data and barely at all where it stays in support, instead of pulling uniformly toward one logged action.

I propose **SAC-RND — anti-exploration by random network distillation**, which replaces the fixed BC proxy with a learned uncertainty penalty. This is a genuine change of stance, the mirror of online exploration: where an online agent *adds* a novelty bonus to chase the unknown, an offline agent should *subtract* the same novelty signal to flee it. BC says "be near this specific action"; an uncertainty penalty says "be anywhere the data supports, and only flee where it does not" — strictly more expressive, because it lets the actor choose among many in-support actions to maximize value rather than gravitating to the one logged action. The strongest known uncertainty signal is a critic *ensemble*, but the $\approx1.05\times$ parameter cap forbids the $N$-critic ensemble outright — the same wall that confined ReBRAC's extra capacity to a single deeper critic. I need ensemble-quality pessimism from *one* small extra network, and that is what **Random Network Distillation** offers: keep a fixed, randomly-initialized *target* network $g(s,a)$ and train a *predictor* $\hat g(s,a)$ to match it by regression on the dataset's $(s,a)$ pairs only. On in-data actions the predictor learns to mimic the target and the error $b(s,a) = \|\hat g - g\|^2$ is small; on OOD actions it was never trained and the error is large. One frozen network, one trained network, a squared error — a per-$(s,a)$ novelty score that is exactly the signal ReBRAC's L2 was crudely approximating.

Before committing I have to confront the known failure of naive offline RND, because reproducing it would make me worse than ReBRAC. The reported problem is that plain RND is "not discriminative enough," and the mechanism is what matters. Picture the predictor taking $[s, a]$ concatenated at its input: it is a flexible MLP trained to match the target on the data, while at the same time the actor is optimized to make the penalty $b(s, \pi(s))$ small, with the actor controlling the action argument. If the action enters only as a few concatenated input dimensions, the predictor's error is a smooth, largely featureless function of the action off the data — nothing forces it to be *high* on a specific OOD action. So the actor finds action directions where the error already happens to be small, even though those actions are OOD, and slides the bonus down without returning to the data. An escapable penalty is no conservatism at all; it would let the actor walk right back into the overestimation ReBRAC's LayerNorm and BC were holding shut. The fault is the *conditioning* — how the action is injected into the prior — not RND itself.

That reframes the design precisely, and it is the crux. I do not need a new novelty principle; I need the predictor's error to be a *sharp* function of the action — genuinely small on dataset actions and genuinely large the instant the action leaves the data — so the only way the actor can drive $b$ down is to propose an in-distribution action. The lever is the architecture by which the action conditions the network. Concatenation dilutes the action's influence; I want the action to *control the computation* over the state, which is exactly what **feature-wise linear modulation (FiLM)** does. Let the state be the feature stream flowing through the MLP and let the action, through its own linear map, produce per-unit scale and shift parameters $(\gamma, \beta)$ that multiply and offset a hidden layer, $h \leftarrow \gamma \odot h + \beta$. Now a different action reshapes the function the network computes, inducing a different target embedding to match, so the predictor's error becomes genuinely action-dependent — small on the $(s,a)$ it trained on and forced to be different (untrained, hence large) on action settings the data never produced. The multiplicative gate is what denies the actor a smooth low-error escape, and it is the single architectural choice that separates a discriminative RND penalty from the escapable one. (The roles can be swapped — action as the modulated feature, state as context — which is the form I use; the substance is the multiplicative conditioning either way.)

With a discriminative $b(s,a)$ I wire it in at both places OOD actions corrupt the value — the same two-leak structure I exploited in ReBRAC, now with a learned penalty instead of L2. The actor proposes $\pi(s)$, possibly OOD, so I subtract $\beta\,b(s, \pi(s))$ from the actor objective: it still climbs $Q$ but is charged for unfamiliarity, pulled toward actions that are high-value *and* in-support. The critic target bootstraps off $Q(s', a')$ at the policy's next action, which can be OOD regardless of the actor, so I subtract $\beta\,b(s', a')$ inside the target as well:

$$y = r + \gamma(1-d)\Big[\min_i Q_i(s', a') - \alpha\,\log\pi(a'|s') - \beta\,b(s', a')\Big].$$

This is the anti-exploration mirror of adding an online bonus to both the policy reward and the value — it suppresses the over-valuation at the backup and steers the policy away from it at once, structurally what ReBRAC's actor-BC and critic-target-BC pair did, but with the conservatism now a learned OOD signal rather than a fixed pull.

I hang this on SAC rather than TD3, deliberately. SAC's actor is a stochastic Tanh-Gaussian that *samples* the actions I need to score with $b$, and its auto-tuned entropy temperature $\alpha$ — tuned to a target entropy of $-\dim(A)$ — keeps the policy from collapsing onto a single action while the RND penalty pushes it around; a stochastic policy is the natural object to anti-explore, because the penalty acts on the whole action distribution rather than one deterministic point. I keep the overestimation-control stack ReBRAC validated: twin critics with a min target and LayerNorm on each critic hidden layer (the geometric smoothing carries over unchanged), Polyak targets at $\tau = 5\times10^{-3}$, $\gamma = 0.99$. Two details bite if ignored. The RND target stays *frozen* — only the predictor trains, and only on dataset $(s,a)$, never on the actor's proposed actions, or the system games itself by making its own reference easy. And the bonus is normalized by a running standard deviation of the raw distillation error, because that error is huge early and shrinks late, so a fixed $\beta$ on the raw error would mean wildly different conservatism over a run. Finally, $\beta$ is the one knob that genuinely needs per-dataset setting — not a regression from ReBRAC but the same per-dataset tuning it already paid for its BC coefficients, one coefficient instead of two: HalfCheetah-medium, a tight single-policy dataset, wants a light penalty so the actor can exploit ($\beta = 0.3$); Walker2d-medium wants a much heavier one to hold the gait against the entropy drive ($\beta = 8.0$); Maze2d-medium takes $\beta = 1.0$. The fixed loop hands me the env name, so I read $\beta$ from it exactly as ReBRAC read its coefficients. The budget is what killed the ensemble alternative and must not kill this: the extra cost over the SAC base is the RND module — a predictor and a frozen target, each a small FiLM MLP emitting a $32$-dim embedding — and holding everything at the locked $256$ width with a two-hidden-layer FiLM RND, the total trainable parameters land just under $1.05\times$ the largest baseline on both the locomotion and the maze2d state dimensions, so this is admissible where a $10$-critic ensemble is not. The pessimism comes from one small network, paid for in math, not capacity.

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
