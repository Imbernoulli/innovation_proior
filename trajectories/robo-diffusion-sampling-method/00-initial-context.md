## Research question

A diffusion policy is already trained and frozen. The single thing being designed is the **reverse-process sampler** the actor uses to turn a Gaussian prior into an action at inference time: which ODE/SDE solver, and how many denoising steps (number of function evaluations, NFE). The policy network, critic, dataset, environments, seeds, and evaluation loop are all fixed. Wall-clock at inference is dominated by NFE — one network pass per step — so the question is concretely: *what sampler reaches the highest D4RL return at the fewest steps?* The score rewards quality (D4RL normalized return, squashed through a sigmoid) and penalizes NFE above a floor of 10, so a sampler that holds return while cutting steps wins twice.

## Prior art before the first rung (denoising-sampler lineage)

The diffusion actor is a Diffusion Q-Learning policy (Wang et al., ICLR 2023): a conditional diffusion model over actions, trained with a behavior-cloning denoising loss plus a Q-maximization term, with a twin critic re-weighting candidate actions at inference. What it is trained against is a *noise-prediction* network ε_θ(a_t, t, s): from an action corrupted to noise level t and conditioned on state s, predict the noise that was added. The lineage the first rung reacts to is the chain of *how you run that trained ε_θ backwards*:

- **Nonequilibrium-thermodynamics diffusion (Sohl-Dickstein et al. 2015).** Fix a forward Markov chain that grinds data to Gaussian noise over many steps; learn a parameterless-inference reverse chain. Established the skeleton but never reached high quality, and the reverse chain is as long as the forward one. Gap: many sequential steps by construction.
- **Score-matching with Langevin dynamics (Song & Ermon, NeurIPS 2019).** Learn the score ∇ log p_t at multiple noise scales and sample by annealed Langevin. Strong samples, but the step sizes and noise scales are hand-tuned and the sampler does not come from the training objective. Gap: sampler decoupled from training, slow.
- **Probability-flow / score SDE view (Song et al., ICLR 2021).** The reverse process has a deterministic ODE whose marginals match the SDE at every noise level — recasting "sample" as "solve an ODE." Gap: still needs a good solver to be fast.

The fixed substrate below is what these converged to: a trained noise-prediction diffusion actor whose *only* remaining freedom at inference is which solver runs the reverse process and over how many steps.

## The fixed substrate

The pipeline (`CleanDiffuser/pipelines/custom_sampling_method.py`) is frozen and must not be touched. It builds a `DQLMlp` noise-prediction network and an `IdentityCondition`, wraps them in a `DiscreteDiffusionSDE` actor with `diffusion_steps` training levels and EMA, and a `DQLCritic` twin Q. Training is DQL: a behavior-cloning denoising loss `actor.loss(act, obs)` plus a Q term, with the critic trained off target-Q backups. At inference the loop loads the frozen checkpoint, samples `num_candidates` actions per state via `actor.sample(...)`, and the critic re-weights them by a softmax over Q. Crucially, both training and inference call exactly one method to draw actions:

```
actor.sample(prior, solver=args.solver, sample_steps=args.sampling_steps,
             condition_cfg=obs, w_cfg=1.0, use_ema=..., temperature=...)
```

`DiscreteDiffusionSDE.sample()` dispatches on the string `solver` to a family of reverse-process integrators (`ddpm`, `ddim`, `ode_dpmsolver++_2M`, …), each run for `sample_steps` denoising steps. The noise schedule, `diffusion_steps`, EMA, the critic, and the evaluation loop are all fixed; the *only* freedom is which solver and how many steps.

## The editable interface

Exactly two lines are editable — `solver` and `sampling_steps` in `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`. Every method on the ladder is a choice of those two fields; the pipeline reads them and passes them straight into `actor.sample(...)`. Custom in-pipeline samplers are out of scope (they would decouple true NFE from the reported `sampling_steps`). The score's NFE term reads the same `sampling_steps` field.

The starting point is the scaffold default: **DDPM ancestral sampling at 100 steps** — the unmodified template (`solver: ddpm`, `sampling_steps: 100`). Each later method replaces exactly these two lines.

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
