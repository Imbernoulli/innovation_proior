TD3+BC confirmed the bet and exposed exactly the fragility I flagged. On HalfCheetah it landed $48.33$ (std $0.42$) — a dead heat with IQL's $48.10$, the expected result on a single-policy dataset with no headroom. On Walker2d it cleared IQL, $85.14$ versus $80.46$, and far tighter (std $0.26$ versus $5.9$): letting the deterministic actor ascend the critic extracted more from the walking data than the hedged expectile did, and did it stably. But Maze2d is the story: mean $50.29$ with std **$56.8$**, the per-seed scores $114.99$, $27.20$, $8.69$. One seed essentially *solved* the maze, one was mediocre, one nearly failed — the mean is a coin flip, a single lucky seed dragging up two poor ones. Maze2d's value landscape is multimodal (many goal approaches), the deterministic actor commits hard to whatever high-value pocket its critic happens to carve, and the two design choices that let TD3+BC commit — a sharp actor ascending a critic with spurious peaks, and a *single, fixed* BC coefficient $\alpha = 2.5$ normalized only by mean $|Q|$ — send different seeds into different basins and pin them there. The $8.69$ seed is a critic whose peak landed in a bad basin and an actor that BC-anchored itself there.

So I have two distinct things to fix, and they correspond to the two roles one BC knob is being asked to play. First the *variance*: the critic on Maze2d has sharp, seed-dependent peaks that the deterministic actor overfits, and TD3's target smoothing alone is clearly not enough to flatten them. Second the *coupling*: TD3+BC uses one $\alpha$ to control both how much the *actor* clones the data and, implicitly, how conservative the *critic target* is — genuinely different jobs that want different strengths on different datasets. HalfCheetah wants almost no BC pressure so the actor can squeeze the last drop of value; Walker2d wants more to hold the gait; Maze2d wants the critic target pinned hard to in-data next-actions so it stops manufacturing spurious pockets. One global $\alpha$ cannot serve all three. I propose **ReBRAC — decoupled, LayerNorm-regularized TD3+BC** — which makes exactly these two changes, minimally, inside the parameter cap.

The first change is *geometric* and changes the architecture. The critic peaks are a function-approximation pathology: an unconstrained MLP critic, fit by bootstrapped regression on a fixed dataset, develops sharp ridges in action space where it extrapolates, and those ridges are what the deterministic actor climbs into. The remedy is to make the critic *smoother as a function of its input* by adding **LayerNorm after each hidden activation**. LayerNorm normalizes the pre-activation statistics layer by layer, which bounds how fast the critic's output can change across nearby inputs — it regularizes the Lipschitz behavior of the network — so a critic that cannot spike arbitrarily across nearby $(s,a)$ cannot present the actor with a knife-edge OOD pocket to overfit. The value extrapolation that drives offline overestimation is damped at the source, by the network's geometry rather than by an explicit penalty. This is what then lets me *deepen* safely: I move actor and critic from $2\times256$ to $3\times256$, because a deeper critic has more capacity to represent the maze's value structure, and the LayerNorm is what keeps that extra capacity from turning into extra spurious peaks. The actor stays LayerNorm-free — it is the function I want sharp enough to commit, just anchored by BC; only the *critic* is what the actor overfits, so only the critic needs the smoothing. $3\times256$ with LayerNorm is the deepest twin-critic-plus-actor that stays inside the $1.05\times$-largest-baseline cap, so this is the most capacity I am allowed, spent on a geometrically-regularized critic rather than on ensembles I am not permitted.

The second change is the algorithmic core: **decouple the BC penalty into two coefficients**, each aimed at one of the two ways OOD actions leak into the value. The first leak is the one TD3+BC handles — the *actor* proposing OOD actions — so I keep an actor BC penalty $\texttt{actor\_bc}\cdot(\pi(s) - a)^2$, but now $\texttt{actor\_bc}$ is its own knob, tiny on HalfCheetah (where the actor should barely clone) and large on Walker2d (where it should stay near the gait data). The second leak is the one TD3+BC leaves unaddressed and which I think is the real Maze2d culprit: the *critic target itself*. The TD3 target evaluates $Q$ at $\pi_{\text{target}}(s') + \text{noise}$ — an action the *policy* chose at the next state, which can already be drifting OOD, and whose over-value backs up through every Bellman step regardless of what the actor does. So I add a *second* BC penalty directly in the bootstrap target,

$$y = r + \gamma(1-d)\Big[\min_i Q_i^{\text{targ}}(s', \tilde a')\ -\ \texttt{critic\_bc}\cdot\big(\tilde a' - a'_{\text{data}}\big)^2\Big],$$

where $\tilde a'$ is the smoothed policy next-action and $a'_{\text{data}}$ is the *dataset's* recorded next action at that transition, which the scaffold hands me directly as `next_actions`. This pushes the critic's bootstrap to prefer targets whose next-action stays near what the behavior policy actually did next, penalizing the value precisely when the target relies on a next-action that has drifted away from the data. It is conservatism injected at the *bootstrap*, not just at the policy — the thing that should stop Maze2d from ever forming the spurious high-value pockets one seed fell into and another could not find. Two decoupled L2 pulls, one on the actor and one on the critic target, with two coefficients: still minimal, no behavior model and no divergence estimator.

The rest of the stabilized TD3 stack is unchanged because it is still doing its job: twin critics with a min target, clipped target-policy smoothing (noise $0.2$, clip $0.5$), delayed actor updates ($\texttt{policy\_freq}=2$), Polyak targets at $\tau = 5\times10^{-3}$. The actor loss keeps a reward-scale normalization — I scale the $Q$ term by $\lambda = 1/(|Q|.\text{mean}() + \varepsilon)$, so the per-dataset BC coefficients are not also fighting the reward scale — and the actor ascends the first critic. The per-dataset hyperparameters are the price of decoupling and are read from the environment name: HalfCheetah takes $\texttt{actor\_bc}=0.001$, $\texttt{critic\_bc}=0.01$, $\text{lr}=10^{-3}$ (almost no actor cloning, light critic conservatism, fast learning on easy single-policy data); Walker2d takes $0.05$, $0.1$, $10^{-3}$ (heavier cloning to hold the gait); Maze2d-medium takes $0.003$, $0.001$, $3\times10^{-4}$ (light cloning, *very* light critic-target BC, slow careful learning — because the maze needs the actor free to commit to goal directions while the LayerNorm critic, not a heavy BC, supplies the smoothing). The learning rate is applied directly to the Adam optimizers; the batch size stays the loop default.

```python
# EDITABLE region of custom.py — step 3: ReBRAC
import sys as _sys

def _detect_env():
    """Parse --env from sys.argv to determine environment name."""
    for i, arg in enumerate(_sys.argv):
        if arg == "--env" and i + 1 < len(_sys.argv):
            return _sys.argv[i + 1]
        if arg.startswith("--env="):
            return arg.split("=", 1)[1]
    return ""

_REBRAC_ENV = _detect_env()

# Per-environment ReBRAC hyperparameters for this benchmark harness
_REBRAC_HPARAMS = {
    "halfcheetah-medium-v2": {"actor_bc_coef": 0.001, "critic_bc_coef": 0.01,  "lr": 1e-3, "batch_size": 1024},
    "walker2d-medium-v2":    {"actor_bc_coef": 0.05,  "critic_bc_coef": 0.1,   "lr": 1e-3, "batch_size": 1024},
    "hopper-medium-v2":      {"actor_bc_coef": 0.01,  "critic_bc_coef": 0.01,  "lr": 1e-3, "batch_size": 1024},
    "maze2d-large-v1":       {"actor_bc_coef": 0.003, "critic_bc_coef": 0.001, "lr": 3e-4, "batch_size": 256},
    "maze2d-medium-v1":      {"actor_bc_coef": 0.003, "critic_bc_coef": 0.001, "lr": 3e-4, "batch_size": 256},
}
_REBRAC_HP = _REBRAC_HPARAMS.get(_REBRAC_ENV, {"actor_bc_coef": 0.01, "critic_bc_coef": 0.01, "lr": 1e-3, "batch_size": 1024})

CONFIG_OVERRIDES: Dict[str, Any] = {}


class DeterministicActor(nn.Module):
    """Deterministic policy pi(s) = tanh(net(s)) * max_action.
    ReBRAC-style: 3 x 256 MLP without LayerNorm (matching CORL actor_ln=False)."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
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
    """Q-function Q(s, a). 3 x 256 MLP with LayerNorm (ReBRAC-style, critic_ln=True)."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class OfflineAlgorithm:
    """ReBRAC — Regularized Behavior Regularized Actor Critic.

    TD3-style with BC penalties on actor and critic targets.
    Per-environment BC coefficients and learning rates from CORL configs.
    """

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

        # Per-env tuned ReBRAC hyperparameters for this benchmark harness
        self.actor_bc_coef = _REBRAC_HP["actor_bc_coef"]
        self.critic_bc_coef = _REBRAC_HP["critic_bc_coef"]
        _lr = _REBRAC_HP["lr"]
        self.policy_noise = 0.2      # target policy smoothing noise
        self.noise_clip = 0.5        # clipping range for smoothing noise
        self.policy_freq = 2         # delayed actor update frequency
        self.normalize_q = True      # normalize Q in actor loss

        # Actor (deterministic, no LayerNorm) + target
        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=_lr)

        # Twin critics (with LayerNorm) + targets
        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=_lr)

        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=_lr)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, next_actions_data = batch
        not_done = 1 - dones.squeeze(-1)
        rewards_flat = rewards.squeeze(-1)
        log_dict: Dict[str, float] = {}

        # -- Critic update --
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            # BC penalty on next actions (compare policy's next actions to dataset's)
            bc_penalty = ((next_actions - next_actions_data) ** 2).sum(-1)

            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            # Subtract BC penalty from critic target (ReBRAC key idea)
            target_q = target_q - self.critic_bc_coef * bc_penalty
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

        # -- Delayed actor update --
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            q = self.critic_1(states, pi)

            # BC penalty on actor
            bc_mse = ((pi - actions) ** 2).sum(-1)

            lmbda = 1.0
            if self.normalize_q:
                lmbda = 1.0 / (torch.abs(q).mean().detach() + 1e-8)

            actor_loss = (self.actor_bc_coef * bc_mse - lmbda * q).mean()
            log_dict["actor_loss"] = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict
```
