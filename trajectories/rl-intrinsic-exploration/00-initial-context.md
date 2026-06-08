# Initial context: the intrinsic-exploration edit surface

## Research question

Sparse-reward, hard-exploration Atari: on games where a positive reward can be hundreds of
reward-free steps away, a policy that learns only from the environment reward almost never sees one,
so it never learns. The question this setup isolates is narrow and singular: **design the
intrinsic-bonus module** — a per-transition novelty signal added on top of the (mostly zero) extrinsic
reward — that makes exploration *directed* instead of a random walk. Everything else about the agent
is fixed; the bonus is the only thing being designed, and it is judged by whether it helps *across*
several games rather than buying one at another's expense.

## The fixed substrate (not editable)

The training loop is a two-stream PPO that is frozen and must not be touched:

- **Atari preprocessing**: grayscale, $84\times84$, frame-skip, 4-frame stack, terminal-on-life-loss,
  sticky actions (`repeat_action_probability=0.25`), reward clipping. Observations arrive as
  $(B,4,84,84)$ uint8 stacks.
- **Policy/value network** (`Agent`): a shared conv torso (`Conv(4,32,8,s4)`–`Conv(32,64,4,s2)`–
  `Conv(64,64,3,s1)`–FC) feeding an actor and **two** critics — `critic_ext` and `critic_int`. The
  loop maintains two value heads on purpose, so an intrinsic reward stream can be valued separately
  from the extrinsic one.
- **Two reward streams, two discounts**: the extrinsic advantage uses `gamma=0.999` (rewards are far
  apart, long horizon); the intrinsic advantage uses `int_gamma=0.99` (a transient, stepping-stone
  bonus). GAE is computed per stream; the loop then asks `mix_advantages` how to combine them.
- **Optimizer / PPO update**: clipped surrogate, clipped value loss on both heads, entropy bonus,
  Adam at `1e-4` with LR anneal. The module's own `loss(...)` is added to the PPO loss each minibatch,
  and the module's `trainable_parameters()` are optimized jointly.

The loop also provides helpers the module may use: `layer_init`, `RunningMeanStd`,
`RewardForwardFilter`, and `last_frame(obs)` (which slices the most recent frame, $(B,1,84,84)$).

## The editable interface

Exactly one region is editable — the `IntrinsicBonusModule` class and the `mix_advantages` function
inside `custom_intrinsic_exploration.py`. Every method on the ladder is a fill-in of this same
contract:

- `initialize(envs)` — optional warm-up (e.g. seed observation-normalization stats from a random
  rollout) before training.
- `trainable_parameters()` — the module parameters to hand the optimizer.
- `compute_bonus(obs, next_obs, actions) -> (B,)` — the per-transition intrinsic reward for the
  rollout (called under `no_grad`; the loop multiplies it by the not-done mask).
- `normalize_rollout_rewards(rollout_intrinsic) -> (T,B)` — rescale the rollout's intrinsic stream
  (the raw bonus scale drifts).
- `update_batch_stats(batch_obs, batch_next_obs)` — update any running statistics once per update.
- `loss(batch_obs, batch_next_obs, batch_actions) -> scalar` — the module's own training loss, added
  to the PPO loss.
- `mix_advantages(ext_advantages, int_advantages, args)` — combine the two advantage streams into the
  one the policy ascends.

The starting point is the scaffold's default: **no intrinsic reward at all**.

```python
# EDITABLE region of custom_intrinsic_exploration.py — default: no bonus
class IntrinsicBonusModule(nn.Module):
    """Default baseline: no intrinsic reward."""

    def __init__(self, action_dim: int, device: torch.device, args: Args):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.args = args

    def initialize(self, envs) -> None:
        return None

    def trainable_parameters(self):
        return []

    def update_batch_stats(self, batch_obs: torch.Tensor, batch_next_obs: torch.Tensor) -> None:
        return None

    def compute_bonus(self, obs, next_obs, actions) -> torch.Tensor:
        return torch.zeros(obs.shape[0], device=self.device)            # no bonus

    def normalize_rollout_rewards(self, rollout_intrinsic) -> torch.Tensor:
        return torch.zeros_like(rollout_intrinsic)

    def loss(self, batch_obs, batch_next_obs, batch_actions) -> torch.Tensor:
        return torch.zeros((), device=self.device)


def mix_advantages(ext_advantages, int_advantages, args: Args) -> torch.Tensor:
    return args.ext_coef * ext_advantages                              # intrinsic stream dropped
```

Each later method replaces these two definitions and nothing else: it gives `compute_bonus` a real
novelty signal, gives `loss` the objective that trains the bonus's own networks, registers those
networks in `trainable_parameters`, and (from ICM onward) turns the intrinsic stream back on in
`mix_advantages`.

## Evaluation settings

Three games spanning the difficulty range — **Tutankham** (medium), **Frostbite** (hard exploration),
and **Private Eye** (hardest — long-horizon, deceptive, with large negative rewards for wrong moves).
Each is run over three seeds {42, 123, 456}. Three metrics, higher is better on all: `eval_return`
(mean evaluation episodic return at the fixed budget), `auc` (area under the evaluation-return curve
over training — *how fast* as well as *how high*), and `nonzero_rate` (fraction of evaluation episodes
that scored anything — a blunt "did it ever find the reward").
