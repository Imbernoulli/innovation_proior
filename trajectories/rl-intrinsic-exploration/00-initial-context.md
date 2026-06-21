## Research question

Sparse-reward, hard-exploration Atari: on games where positive reward can be hundreds of reward-free steps away, a policy trained only on extrinsic reward almost never sees it and therefore never learns. The design target is the **intrinsic-bonus module** — a per-transition signal added to the mostly-zero extrinsic reward so that exploration becomes directed rather than a random walk. Everything else about the agent stays fixed.

## Prior art / Background / Baselines

The fixed base learner is PPO. The policy-optimization lineage that precedes it is:

- **Deep Q-learning (DQN, Mnih et al. 2015).** Learns Q(s,a;θ) with experience replay and a target network, selecting actions by discrete argmax.
- **Vanilla policy gradients (REINFORCE, Williams 1992).** Ascends ∇_θ J = Ê_t[∇_θ log π_θ(a_t|s_t) Â_t] directly.
- **TRPO (Schulman et al. 2015b).** Maximizes a surrogate objective subject to an average-KL trust region solved with Fisher-vector products and conjugate gradients.
- **Parallel actor-critic (A3C/A2C, Mnih et al. 2016).** Uses multiple parallel actors, a single shared network with policy and value heads, n-step bootstrapped advantages, and an entropy bonus.
- **GAE (Schulman et al. 2015a).** Computes advantages as a λ-weighted average of n-step TD residuals, trading bias for variance.

## Fixed substrate / Code framework

A two-stream PPO loop is frozen and must not be touched: Atari preprocessing (grayscale 84×84, frame-skip, 4-frame stack, terminal-on-life-loss, sticky actions), a shared conv torso feeding an actor and **two** value heads — extrinsic and intrinsic — with discounts `gamma=0.999` and `int_gamma=0.99`, per-stream GAE, and the PPO update (clipped surrogate, clipped value loss, entropy bonus, Adam). The two value heads let the intrinsic stream be valued separately from the extrinsic one. Helpers available to a module: `layer_init`, `RunningMeanStd`, `RewardForwardFilter`, and `last_frame(obs)` returning the most recent frame `(B,1,84,84)`.

## Editable interface

Only one region is editable — the `IntrinsicBonusModule` class and the `mix_advantages` function in `custom_intrinsic_exploration.py`. The contract is: `compute_bonus(obs, next_obs, actions)` for the per-transition bonus, `loss(...)` for the module's training objective added to the PPO loss, `trainable_parameters()`, `normalize_rollout_rewards(...)`, `initialize(envs)` and `update_batch_stats(...)` for optional warm-up / running stats, and `mix_advantages(ext, int, args)` for combining the two advantage streams.

The starting point is the default scaffold: **no intrinsic reward**. Each method fills exactly these two definitions and nothing else.

```python
# EDITABLE region of custom_intrinsic_exploration.py — default fill (no bonus)
class IntrinsicBonusModule(nn.Module):
    """Default baseline: no intrinsic reward."""

    def __init__(self, action_dim: int, device: torch.device, args: Args):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.args = args

    def initialize(self, envs) -> None:                              # optional warm-up
        return None

    def trainable_parameters(self):                                  # params handed to the optimizer
        return []

    def update_batch_stats(self, batch_obs: torch.Tensor, batch_next_obs: torch.Tensor) -> None:
        return None

    def compute_bonus(self, obs, next_obs, actions) -> torch.Tensor: # per-transition bonus, (B,)
        return torch.zeros(obs.shape[0], device=self.device)

    def normalize_rollout_rewards(self, rollout_intrinsic) -> torch.Tensor:
        return torch.zeros_like(rollout_intrinsic)

    def loss(self, batch_obs, batch_next_obs, batch_actions) -> torch.Tensor:  # added to the PPO loss
        return torch.zeros((), device=self.device)


def mix_advantages(ext_advantages, int_advantages, args: Args) -> torch.Tensor:
    return args.ext_coef * ext_advantages                           # intrinsic stream dropped
```

## Evaluation settings

Three games span the difficulty range — **Tutankham** (medium), **Frostbite** (hard exploration), and **Private Eye** (hardest: long-horizon, deceptive, with large negative rewards for wrong moves) — each over three seeds {42, 123, 456}. Metrics, higher is better: `eval_return` (mean evaluation episodic return at the fixed budget), `auc` (area under the evaluation-return curve), and `nonzero_rate` (fraction of evaluation episodes that score anything).
