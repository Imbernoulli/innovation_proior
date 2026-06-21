## Research question

The diffusion policy is trained and frozen. The only design target is the **reverse-process sampler** that turns a Gaussian prior into an action at inference time: which ODE/SDE solver, and how many denoising steps (number of function evaluations, NFE). The policy network, critic, dataset, environments, seeds, and evaluation loop are fixed. Wall-clock at inference is dominated by NFE — one network pass per step — so the concrete question is: *what sampler reaches the highest D4RL return at the fewest steps?* The score rewards quality (D4RL normalized return, squashed through a sigmoid) and penalizes NFE above a floor of 10, so a sampler that holds return while cutting steps wins twice.

## Prior art / Background / Baselines

The frozen actor is a Diffusion Q-Learning policy (Wang et al., ICLR 2023): a conditional diffusion model over actions trained with a behavior-cloning denoising loss plus a Q-maximization term. At inference it predicts noise via ε_θ(a_t, t, s) and draws actions by solving the reverse process. The available samplers in the pipeline are the baselines:

- **DDPM ancestral sampling.** Reverses the forward Markov chain one noisy step at a time using the learned noise predictor. Gap: many sequential steps are required; return or sample quality drops when the step budget is cut too far.
- **DDIM.** Uses a deterministic, non-Markovian trajectory that shares the training marginals, allowing fewer steps than DDPM. Gap: quality collapses sharply once the step count falls below a moderate threshold.
- **DPM-Solver / DPM-Solver++.** High-order solvers for the diffusion ODE, including multistep variants. Gap: they are tuned for image-generation schedules and metrics; on this policy the return-per-step trade-off is still weak at very low NFE.
- **Probability-flow ODE.** The deterministic counterpart to the reverse SDE with matching marginals. Gap: discretization error still requires enough steps to keep returns high.

The actor, its training, and the inference loop are fixed; the remaining freedom is the choice of solver and step count.

## Fixed substrate / Code framework

The pipeline (`CleanDiffuser/pipelines/custom_sampling_method.py`) is frozen and must not be touched. It builds a `DQLMlp` noise-prediction network and an `IdentityCondition`, wraps them in a `DiscreteDiffusionSDE` actor trained over `diffusion_steps` levels with EMA, and uses a `DQLCritic` twin Q. Training combines the denoising loss `actor.loss(act, obs)` with a Q term. At inference the loop loads the frozen checkpoint, samples `num_candidates` actions per state via `actor.sample(...)`, and the critic re-weights them by a softmax over Q. Every call to draw actions goes through exactly one method:

```
actor.sample(prior, solver=args.solver, sample_steps=args.sampling_steps,
             condition_cfg=obs, w_cfg=1.0, use_ema=..., temperature=...)
```

`DiscreteDiffusionSDE.sample()` dispatches on the string `solver` to the reverse-process integrators (`ddpm`, `ddim`, `ode_dpmsolver++_2M`, …), each run for `sample_steps` denoising steps. The noise schedule, `diffusion_steps`, EMA, the critic, and the evaluation loop are fixed; only the solver and step count vary.

## Editable interface

Only two lines are editable: `solver` and `sampling_steps` in `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`. The pipeline reads them and passes them straight into `actor.sample(...)`. Custom in-pipeline samplers are out of scope, because they would decouple true NFE from the reported `sampling_steps`. The score's NFE term reads the same `sampling_steps` field.

The starting scaffold is the default: **DDPM ancestral sampling at 100 steps** — `solver: ddpm`, `sampling_steps: 100`. Each method replaces exactly these two lines.

```yaml
# EDITABLE region of CleanDiffuser/configs/custom/mujoco/mujoco.yaml — default fill
# (lines 15 and 17 are the only editable lines; everything else is fixed)
diffusion_steps: 100
solver: ddpm          # line 15 — reverse-process integrator passed to actor.sample(...)
predict_noise: True
sampling_steps: 100   # line 17 — NFE per inference call; also read by the score's NFE penalty
ema_rate: 0.9999
```

## Evaluation settings

Three D4RL MuJoCo environments — **hopper-medium-v2**, **walker2d-medium-v2**, **halfcheetah-medium-v2** — at seed 42, under the fixed vectorized evaluation loop (`num_envs=50`, `num_candidates=50`, `num_episodes=3`, EMA weights). Two reported quantities per env: `normalized_score` (D4RL return) and `sampling_steps` (NFE). The per-env score is `sigmoid(normalized_score) * penalty(sampling_steps)`, with `penalty = exp(-0.015·(steps−10))` for steps>10 and `1.0` at or below 10; the task score is the geometric mean across the three envs. Submitting at lower NFE is strictly preferred when quality is comparable.
