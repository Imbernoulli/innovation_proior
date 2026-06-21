IQL gave me a clean read of its own ceiling. On HalfCheetah it landed $48.10$ with a standard deviation of $0.12$ across seeds — rock-steady and essentially *pinned* to the behavior policy, exactly as designed for a single-mediocre-policy dataset with no headroom. Walker2d reached $80.46$ but unevenly (std $5.9$). The damning number is Maze2d: $33.73$, the worst of the three and far from the data's reachable ceiling, with a 9-point per-seed swing. Maze2d is the stitching task — good performance needs the value of a goal-adjacent state to propagate backward across transitions from *different* trajectories — and IQL's propagation is throttled by $\tau = 0.7$: the expectile only weakly approximates the in-support max, so the value signal stays weak and the advantage-weighted extraction never pushes the policy past dataset actions. The diagnosis is sharp: IQL is *too hedged*. Its safety came from refusing to ever exploit the critic directly — no argmax, no $\nabla_a Q$ ascent — and on a task where exploitation matters most, that refusal cost the most. So I want the opposite stance: let the policy actually *ascend the critic* for real deterministic-policy-gradient improvement, and bolt on the *minimum* behavior regularization needed to keep the OOD overestimation from blowing up. IQL avoided the OOD query; I now want to *allow* it and *tame* it, because allowing it is the only way to get improvement stronger than an in-sample expectile.

I propose **TD3+BC**: stabilized TD3 plus a single behavior-cloning term in the actor loss. The exploitation engine is a deterministic actor-critic — a deterministic actor $a = \pi(s)$ ascending a learned critic by $\nabla_\phi J = \mathbb{E}[\nabla_a Q(s,a)|_{a=\pi(s)}\,\nabla_\phi \pi(s)]$ — and even online this object overestimates: the critic has function-approximation error, the actor climbs wherever the critic bulges upward, the bulges are disproportionately where the error is positive, so the actor selects for the upward errors and the value inflates. This is the same selection-for-positive-error as a discrete max, routed through the gradient, and offline it is catastrophic because no fresh transition can correct an inflated pocket. The structural cures are precisely TD3's, so I carry the full correction stack in.

First, **clipped double-Q**. A single critic's target inflates because the same estimator both selects and evaluates the next action, so it evaluates its own optimistic pick. I keep two critics and take the *min* of the two target critics at the next action wherever I need a single target value. The asymmetry is the justification: an overestimated action gets selected by the actor and propagated through every policy update — overestimation is actively chased — whereas an underestimated action is simply avoided and never amplified. So biasing toward the smaller estimate is the safe direction; the min caps the target at the worse case of a single critic. Second, **target policy smoothing**. A deterministic actor will find and sit on a narrow spurious peak in the critic and read the target off that knife-edge, so I perturb the target action with clipped Gaussian noise,

$$\tilde a' = \mathrm{clip}\big(\pi_{\text{target}}(s') + \mathrm{clip}(\epsilon,\,-c,\,c),\ -a_{\max},\ a_{\max}\big),\qquad \epsilon\sim\mathcal N(0,\sigma^2),$$

with $\sigma = 0.2\,a_{\max}$ and $c = 0.5\,a_{\max}$, which makes the target reflect the value of a small *neighborhood* and averages the spike away. Third, **delayed updates and Polyak targets**: I fit the critic for two steps per actor step ($\texttt{policy\_freq}=2$) so each policy move sees a critic that has had time to drive its error down, and soft-update all targets at $\tau = 5\times 10^{-3}$ so the bootstrap chases a stationary objective rather than its own moving estimate. These three — min target for bias, smoothing for the deterministic-peak variance, delay plus soft targets for accumulated-error variance — are one coordinated attack on function-approximation error, and they are the only reason a DPG actor is stable enough to regularize.

The offline piece is deliberately minimal, because offline I cannot validate any knob by interacting, so every extra mechanism (a generative behavior model, a divergence estimator, a conservative sampler) is tuning blind — and IQL's whole appeal was adding essentially one idea. The cheapest way to keep an ascending actor near the data is to add a behavior-cloning term straight to the policy objective: while the actor maximizes $Q(s, \pi(s))$, also penalize how far $\pi(s)$ is from the dataset action $a$ at that state. The actor objective becomes

$$\max_\pi\ \ \lambda\,Q\big(s, \pi(s)\big)\ -\ \big(\pi(s) - a\big)^2,$$

an L2 pull toward the data. I pick L2 to the dataset action over a KL or MMD divergence for the same minimality reason — no behavior model to fit, one line, deterministic policy, and no principled reason a particular divergence wins here. This directly answers the OOD overestimation: the actor can still ascend the critic, but it is anchored, so it cannot run off to the over-valued out-of-distribution actions that have no correction.

The one subtlety that makes the BC term work across datasets is the coefficient. The penalty $(\pi - a)^2$ is bounded — for actions in $[-1,1]$ it is at most about $4$ per dimension — but $Q$ scales with the reward scale, which differs across datasets, so a fixed $\lambda$ would make BC dominate on small-reward tasks and vanish on large-reward ones. I therefore make $\lambda$ a *normalizer*, not a fixed weight:

$$\lambda = \frac{\alpha}{\frac{1}{N}\sum_i \big|Q(s_i, a_i)\big|},\qquad \alpha = 2.5,$$

dividing by the mean absolute $Q$ over the minibatch so the RL/BC balance is decoupled from the reward scale and one $\alpha$ works across all three datasets. The mean $|Q|$ is *detached* so $\lambda$ is a scale, not a gradient path. That is the whole algorithm: TD3's stabilized critic plus a reward-scale-normalized L2 pull toward the dataset action. State normalization I get for free from the fixed loop (which matters because a deterministic actor is sensitive to feature scaling), and as with IQL I leave rewards unnormalized, so the only offline-specific change versus online TD3 is the BC term and its normalizer. Concretely, every step I update both critics on the min-of-target-critics smoothed target by MSE; every second step I update the actor on $-\lambda\,Q_1(s,\pi(s)) + \mathrm{MSE}(\pi(s), a)$ with $\lambda = \alpha/|Q_1|.\text{mean}().\text{detach}()$, ascending the *first* critic for the policy gradient, then Polyak the actor and both critic targets.

```python
# EDITABLE region of custom.py — step 2: TD3+BC
class DeterministicActor(nn.Module):
    """Deterministic policy pi(s) = tanh(net(s)) * max_action. 2 x 256 MLP."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    """Q-function Q(s, a). 2 x 256 MLP (TD3+BC reference architecture)."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class OfflineAlgorithm:
    """TD3+BC — Twin Delayed DDPG with Behavior Cloning regularization."""

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

        # TD3+BC hyperparameters
        self.alpha = 2.5
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2

        # Actor (deterministic) + target
        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin critics + targets
        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)

        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        not_done = 1 - dones.squeeze(-1)
        rewards_flat = rewards.squeeze(-1)
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q1 = self.critic_1_target(next_states, next_action)
            target_q2 = self.critic_2_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards_flat + not_done * self.discount * target_q

        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            q = self.critic_1(states, pi)
            lmbda = self.alpha / q.abs().mean().detach()

            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, actions)
            log_dict["actor_loss"] = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict
```
