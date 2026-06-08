# Initial context: the intrinsic-exploration edit surface

## Research question

Sparse-reward, hard-exploration Atari: on games where a positive reward can be hundreds of
reward-free steps away, a policy that learns only from the environment reward almost never sees one,
so it never learns. The single thing being designed is the **intrinsic-bonus module** — a
per-transition signal added on top of the (mostly zero) extrinsic reward to make exploration directed
instead of a random walk. Everything else about the agent is fixed.

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

The starting point is the scaffold default: **no intrinsic reward** — `compute_bonus` returns zeros and
`mix_advantages` keeps only `ext_coef * ext_advantages`. Each later method replaces these two
definitions and nothing else.

## Evaluation settings

Three games spanning the difficulty range — **Tutankham** (medium), **Frostbite** (hard exploration),
and **Private Eye** (hardest — long-horizon, deceptive, with large negative rewards for wrong moves) —
each over three seeds {42, 123, 456}. Three metrics, higher is better on all: `eval_return` (mean
evaluation episodic return at the fixed budget), `auc` (area under the evaluation-return curve over
training), and `nonzero_rate` (fraction of evaluation episodes that scored anything).
