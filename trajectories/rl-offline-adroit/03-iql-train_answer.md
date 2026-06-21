The decoupled-penalty rung sharpened my picture of the ceiling. Pen came in at a mean of 74.6 and — as I hoped — it *tightened*: per-seed 95.5, 55.0, 73.2, a spread of forty points instead of seventy. Making the policy deterministic and putting an explicit penalty in did remove some of the sampled-$V$ variance, and the mean even crept up. But Hammer fell to 0.35 (0.34, 0.28, 0.44), *below* the previous rung's 1.05 and rock-steady at the floor, and Door collapsed to 0.016, indistinguishable from zero. The Hammer number is the tell: the per-seed values are tight and tiny, so this is not high-variance failure, it is *systematic* — the method reliably learns a policy that does almost nothing on the nailing task. The tiny actor penalty I set there was supposed to let the policy improve; instead the deterministic TD3 actor, doing one-step ascent on a critic whose target is still a single bootstrapped $\min$, has no in-support improvement signal to climb on the long contact sequence. So the explicit, decoupled, deterministic constraint fixed the *variance* on Pen but it is still fundamentally a one-step improvement off a TD3 critic, and on the tasks that need genuine multi-step value propagation it has nothing. I want to step back from "constrain the policy" entirely and attack the thing both previous rungs only worked around: the bootstrap's out-of-distribution query. Both left a learned $Q$ being evaluated, at some point, at an action the demos never contained.

I propose IQL — Implicit Q-Learning. The aim is to never query an unseen action anywhere in value training, yet still do genuine multi-step in-support dynamic programming so value can flow backward across the Hammer sequence and stitch the cloned Door data. Start from what breaks. Standard Q-learning regresses $Q(s,a)$ onto $r + \gamma\max_{a'}\bar{Q}(s',a')$, and that max ranges over *all* actions, including ones absent from the data at $s'$, where the network extrapolates upward and the policy chases it. The safe alternative is SARSA: bootstrap with the *dataset's* next action, $r + \gamma\bar{Q}(s', a')$ with $a'$ from $\mathcal{D}$. Now no off-support action is ever touched — but MSE fits $Q$ to the *mean* of those targets, so the fixed point is $Q^{\pi_\beta}$, the value of the behavior policy: pure policy evaluation, one improvement step, no iteration. That is exactly the one-step ceiling that floors the long-horizon tasks, because to get a good value at the start of the Hammer sequence I need value to flow backward across transitions. So I am caught: the max improves but queries off-support; SARSA's mean is safe but does not improve.

The resolution is to look at what SARSA's mean is missing. Read $Q(s,a)$ over the behavior actions as a per-state random variable (randomness from $a\sim\pi_\beta(\cdot|s)$); SARSA's MSE recovers its *mean*. What improvement needs is the *maximum over in-support actions* — a max restricted to actions the behavior policy could actually produce at $s$. That restriction is the whole game: it is a max (so it improves, and iterating it does real dynamic programming) but it never reaches an off-support action (so it stays safe). The obstacle is that I cannot *compute* it by sampling actions and querying $Q$ at each — the moment I sample and query, I am back to evaluating $Q$ off-support. I need the in-support max *without ever querying $Q$ at any specific $a'$*. The statistic that gives the upper tail of a random variable from its samples, without naming any point, is the expectile. The $\tau$-expectile of $X$ minimizes an *asymmetric* squared loss,

$$m_\tau = \arg\min_m \; \mathbb{E}\big[\,|\tau - \mathbb{1}(u<0)|\;u^2\,\big], \qquad u = x - m,$$

so a positive residual (a sample above my estimate) is weighted $\tau$, a negative one $1-\tau$. At $\tau = 0.5$ both weights are $\tfrac12$ and $m_{0.5}$ is the mean (SARSA). For $\tau > 0.5$ the samples *above* the estimate dominate, so the estimate is pushed up; as $\tau \to 1$ the expectile climbs to the supremum of the support. The expectiles are monotone non-decreasing in $\tau$ and bounded by the support, so the limit exists and equals the in-support max. The upper expectile of $Q$ over the behavior actions, estimated by an asymmetric-$L_2$ regression on in-sample data only, *is* the in-support max I wanted — the improvement operator, expressed as a regression I can run with SGD and zero off-support queries.

But I must be careful where I apply the expectile, and the reason is precisely the kind of optimism that could re-poison the long Hammer bootstrap. If I take the expectile of the raw TD residual $r + \gamma\bar{Q}(s',a') - Q(s,a)$, the target carries *two* sources of randomness: the action $a' \sim \pi_\beta(\cdot|s')$ (which I *want* to be optimistic over — the best in-support action is the improvement signal) and the stochastic transition $s' \sim p(\cdot|s,a)$ (which I emphatically do *not* want to be optimistic over). An upper expectile rewards high targets indiscriminately, so it would reward a target that is high merely because the dynamics happened to land in a lucky next state — conflating "there is a better action here" with "I got lucky with the dice," which compounds over the Hammer horizon into a wildly overoptimistic value. So I split the estimate. A value network $V_\psi(s)$ takes the upper expectile over actions with the transition held fixed,

$$L_V(\psi) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\big[\,L_2^\tau\big(\bar{Q}(s,a) - V_\psi(s)\big)\,\big],$$

where both $s$ and $a$ come from $\mathcal{D}$, so the only randomness for a given $s$ is the action, and $V$ becomes the $\tau$-expectile of $Q$ over dataset actions — optimistic over actions, no dynamics in it. Then back this into $Q$ with an *honest* MSE that averages over the transition,

$$L_Q(\theta) = \mathbb{E}_{(s,a,s')\sim\mathcal{D}}\big[\,\big(r + \gamma V_\psi(s') - Q_\theta(s,a)\big)^2\,\big].$$

The MSE is correct here precisely because $V_\psi(s')$ already did the optimistic action selection; what remains is to average $\gamma V_\psi(s')$ over $s'\sim p$, and a mean is the right way to average dynamics. The division of labor is the whole method: $V$ takes the upper expectile over actions, $Q$ takes the mean over transitions, both losses touch only dataset $(s,a,s')$, no policy appears in value training, and no off-support action is ever queried. And this *is* multi-step dynamic programming — the value is monotone in $\tau$, bounded by the in-support optimum, and converges to it as $\tau \to 1$, spanning SARSA ($\tau=0.5$, the floor both previous rungs effectively sat near on the hard tasks) to in-support Q-learning ($\tau \to 1$, the propagation Hammer needs). I stabilize with clipped double-$Q$ (a single twin-$Q$ module, take the $\min$) and a Polyak target critic so $V$ chases a stable $\bar{Q}$.

Now I have a near-optimal in-support $Q$ and $V$ but no policy — value training was deliberately policy-free. Extraction must obey the same commandment: never query $Q$ at an unseen action. So no $\arg\max_a Q$ (searches off-support) and no DDPG-style $\nabla_a Q$ ascent (evaluates $Q$ at the policy's possibly-off-support actions — exactly the one-step move that floored the previous rung on Hammer). What I *can* do is reweight the dataset's own actions: advantage-weighted regression. The KL-constrained improvement $\max_\pi \mathbb{E}_{a\sim\pi}[A]$ s.t. $\mathrm{KL}(\pi\|\pi_\beta)\le\varepsilon$ has the closed form $\pi^* \propto \pi_\beta\exp(A/\beta_{\text{temp}})$, and projecting it onto the parametric policy by weighted maximum likelihood gives

$$L_\pi(\phi) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\big[\,-\exp\!\big(\beta_{\text{temp}}\,(\bar{Q}(s,a) - V_\psi(s))\big)\,\log\pi_\phi(a|s)\,\big],$$

with advantage $A = Q - V$ and inverse temperature $\beta_{\text{temp}}$. This only ever evaluates dataset actions — it reweights observed $(s,a)$ by how advantaged they are — so it queries nothing unseen, inherits an implicit stay-near-$\pi_\beta$ constraint, and decouples cleanly from value training. I clip the weight $\exp(\beta_{\text{temp}} A)$ at $100$ so a few huge advantages cannot dominate.

Grounding this in the harness, the critic becomes a single twin-$Q$ module (a `Critic` holding `q1`, `q2`, with `.both()` returning both heads squeezed to scalars and `forward` returning their $\min$) plus a `ValueFunction` for $V$. The actor is where this task is specific: not the scaffold's TanhTransform Gaussian and not the previous rung's plain Normal, but a $2\times256$ MLP with **Dropout 0.1** after each hidden ReLU and a **Tanh on the mean output**, with a state-independent `log_std` and a plain `Normal` (no Tanh transform on the density). The dropout is real regularization for the AWR objective on twenty-five trajectories — it fights the actor memorizing the tiny dataset — and the Tanh-bounded mean keeps the predicted mean inside the action box where the demos live. I set $\tau_{\text{iql}} = 0.8$ (the expectile — high enough to do meaningful in-support improvement, not so high the asymmetric loss becomes unstable on thin data), $\beta_{\text{temp}} = 3.0$ (the AWR temperature), and `exp_adv_max = 100`. The actor optimizer gets a `CosineAnnealingLR` decayed over the full $10^6$ steps, which matters on this little data: it anneals the AWR updates so late training does not thrash the policy. I turn on state normalization (`CONFIG_OVERRIDES = {"normalize": True}`). Per step: update $V$ by expectile regression against the *target* critic, update $Q$ by MSE onto $r + \gamma(1-\text{done})V(s')$ and soft-update the target, then update the actor by clipped advantage-weighted log-likelihood and step the cosine schedule. I ignore $\hat a'$ in the batch — that was the previous rung's hook, not mine.

My falsifiable expectations against the previous numbers: Pen should hold high and stay tight, a mean comfortably above 74.6 with a seed spread no worse, because expectile DP plus AWR is the most reliable extractor on near-expert data. Hammer is the real test of the whole thesis: if in-support multi-step propagation is what was missing, it should finally lift clearly off the $\sim 0.35$–$1.0$ floor the two previous rungs sat at, the first sign the long sequence is being valued. And Door (cloned) should move off zero into low-but-positive territory, because the broad cloned data is exactly what stitching can exploit and neither previous rung could. If Pen rises, Hammer clears the floor, and Door turns positive across seeds, the in-support-DP hypothesis is confirmed.

```python
# EDITABLE region of custom_adroit.py — step 3: IQL
CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": True}


class Actor(nn.Module):
    """IQL GaussianPolicy — 2x256 MLP with Tanh output + Dropout(0.1),
    state-independent log_std, Normal distribution (no TanhTransform)."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 orthogonal_init: bool = False):
        super().__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = -20.0
        self._max_log_std = 2.0

    def _get_policy(self, state: torch.Tensor):
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        return Normal(mean, log_std.exp())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
        policy = self._get_policy(state)
        return policy.log_prob(action).sum(-1)

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        policy = self._get_policy(state)
        action = policy.mean if deterministic else policy.rsample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = policy.log_prob(action).sum(-1)
        return self.max_action * action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        policy = self._get_policy(state)
        action = policy.sample() if self._mlp.training else policy.mean
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action[0].cpu().numpy()


class Critic(nn.Module):
    """Twin Q-function for IQL. Two 3x256 MLPs, squeeze output to scalar."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def both(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.both(state, action)
        return torch.min(q1, q2)


class OfflineAlgorithm:
    """IQL — Implicit Q-Learning for offline RL."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_lr=3e-4, orthogonal_init=True, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        self.iql_tau = 0.8       # expectile for V loss
        self.beta = 3.0          # inverse temperature for advantage weighting
        self.exp_adv_max = 100.0

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=int(1e6))

        self.qf = Critic(state_dim, action_dim).to(device)
        self.qf_target = deepcopy(self.qf)
        self.qf_target.requires_grad_(False)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=critic_lr)

        self.vf = ValueFunction(state_dim).to(device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=critic_lr)

    def _asymmetric_l2_loss(self, u: torch.Tensor, tau: float) -> torch.Tensor:
        return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        log_dict: Dict[str, float] = {}

        # V update: expectile regression against Q_target
        with torch.no_grad():
            target_q = self.qf_target(states, actions)
        v = self.vf(states)
        adv = target_q - v
        v_loss = self._asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Q update: Bellman with V(s') as bootstrap
        with torch.no_grad():
            next_v = self.vf(next_states)
            q_target = rewards + (1.0 - dones) * self.discount * next_v
        q1, q2 = self.qf.both(states, actions)
        q_loss = (F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)) / 2.0
        log_dict["critic_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        soft_update(self.qf_target, self.qf, self.tau)

        # Actor update: advantage-weighted regression
        with torch.no_grad():
            adv_detached = target_q - self.vf(states)
            exp_adv = torch.exp(self.beta * adv_detached).clamp(max=self.exp_adv_max)
        action_log_prob = self.actor.log_prob(states, actions)
        actor_loss = torch.mean(exp_adv * (-action_log_prob))
        log_dict["actor_loss"] = actor_loss.item()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

        return log_dict
```
