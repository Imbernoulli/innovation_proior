## Research question

Sparse-reward, hard-exploration Atari: on games where a positive reward can be hundreds of
reward-free steps away, a policy that learns only from the environment reward almost never sees one,
so it never learns. The single thing being designed is the **intrinsic-bonus module** — a
per-transition signal added on top of the (mostly zero) extrinsic reward to make exploration directed
instead of a random walk. Everything else about the agent is fixed.

## Prior art before the first rung (policy-optimization lineage)

The base learner the first rung uses — PPO — is itself the resolution of a line of policy-optimization
methods. These are the baselines that precede the ladder; the fixed substrate below is what they
converged to.

- **Deep Q-learning (DQN, Mnih et al. 2015).** Learns Q(s,a;θ) with experience replay and a target
  network, minimizing the TD error toward R + γ max_{a'} Q(s',a';θ⁻). Cracked Atari from pixels, but is
  built around a discrete argmax (awkward for continuous control) and is brittle — reward-scale or
  hyperparameter changes break it. Gap: not general across action spaces, unreliable.
- **Vanilla policy gradients (REINFORCE, Williams 1992).** Ascend ĝ = Ê_t[∇_θ log π_θ(a_t|s_t) Â_t]
  directly — general and simple, continuous or discrete. But each sample feeds one gradient step
  (data-inefficient), and the step size is treacherous: too large collapses the policy irrecoverably,
  and multi-epoch reuse of one batch is *not* justified (the surrogate is valid only at the current θ).
  Gap: cannot safely reuse data.
- **TRPO (Schulman et al. 2015b).** Maximizes Ê_t[r_t(θ) Â_t] subject to a KL trust region
  Ê_t[KL[π_old, π_θ]] ≤ δ, solved by natural-gradient steps (Fisher-vector products + conjugate
  gradient + a line search). Reliable, but heavy second-order machinery, essentially one step per batch
  (no cheap K-epoch reuse), and architecture-hostile — the KL on the output distribution is smeared by
  parameter sharing or stochastic units. Gap: reliable but heavy and sharing-hostile.
- **Parallel actor-critic (A3C/A2C, Mnih et al. 2016).** Many actors on env copies, a single network
  **sharing** a policy head and a value head, n-step bootstrapped advantages, and an **entropy bonus**
  against premature determinism. Gap: still one gradient step per batch with no trust region, so a
  shared-parameter step can be destructive.
- **GAE (Schulman et al. 2015a).** The variance-reduced advantage the base learner reuses: with
  δ_t = R_t + γV(s_{t+1}) − V(s_t), Â_t = Σ_l (γλ)^l δ_{t+l}, the λ knob trading bias (λ→0, just δ_t)
  for variance (λ→1, Monte-Carlo). λ ≈ 0.95 is the usual sweet spot.

## The fixed substrate

A two-stream PPO loop is frozen and must not be touched: Atari preprocessing (grayscale $84\times84$,
frame-skip, 4-frame stack, terminal-on-life-loss, sticky actions), a shared conv torso feeding an
actor and **two** value heads — extrinsic and intrinsic — with their own discounts (`gamma=0.999`,
`int_gamma=0.99`), per-stream GAE, and the PPO update (clipped surrogate, clipped value loss, entropy
bonus, Adam). The two value heads exist so an intrinsic stream can be valued separately from the
extrinsic one. The loop also provides helpers a module may use: `layer_init`, `RunningMeanStd`,
`RewardForwardFilter`, and `last_frame(obs)` (the most recent frame, $(B,1,84,84)$).

## The editable interface

Exactly one region is editable — the `IntrinsicBonusModule` class and the `mix_advantages` function in
`custom_intrinsic_exploration.py`. Every method on the ladder is a fill of this same contract:
`compute_bonus(obs, next_obs, actions)` (the per-transition bonus), `loss(...)` (the module's own
training objective, added to the PPO loss), `trainable_parameters()`, `normalize_rollout_rewards(...)`,
`initialize(envs)` / `update_batch_stats(...)` (optional warm-up / running stats), and
`mix_advantages(ext, int, args)` (how the two advantage streams combine).

The starting point is the scaffold default: **no intrinsic reward**. Each later method replaces exactly
these two definitions and nothing else.

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

Three games spanning the difficulty range — **Tutankham** (medium), **Frostbite** (hard exploration),
and **Private Eye** (hardest — long-horizon, deceptive, with large negative rewards for wrong moves) —
each over three seeds {42, 123, 456}. Three metrics, higher is better on all: `eval_return` (mean
evaluation episodic return at the fixed budget), `auc` (area under the evaluation-return curve over
training), and `nonzero_rate` (fraction of evaluation episodes that scored anything).
