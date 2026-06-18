## Problem

Offline RL on a fixed `medium` D4RL buffer. The floor question: how far does *cloning the behavior
policy* get me, before any critic? Naive L2 behavior cloning fits the conditional mean `E[a|s]`, a
unimodal Gaussian, and on a multimodal `medium` buffer the mean of two valid action modes is an invalid
in-between action that stalls the gait at exactly the branchy states that matter.

## Key idea

Clone the buffer with an **expressive, multimodal** actor and nothing else — no critic, no Q, no
reranking. Model the **score** of the action density by denoising: with the forward process
`a^k = sqrt(abar_k) a + sqrt(1-abar_k) eps`, the noise `eps` is the negative score up to scale, so a
network trained to predict `eps` learns `grad_a log p(a|s)` by plain MSE — energy-based-policy
expressivity with stable supervised training, no intractable normalizer `Z`. Sampling runs a stochastic
reverse chain from Gaussian noise; the random init and per-step noise draw from several modes without
averaging them. On this low-dim, single-action benchmark this is the diffusion generative core stripped
of the visuomotor add-ons (no vision encoder, no action sequence, no receding horizon): action =
denoised variable, state = condition, one action sampled per env at inference.

## Why it works

The expressive actor will not smear across action modes, so it avoids the stalling in-between actions a
Gaussian regressor emits — a clean win on balance-critical walker2d. But it is given no reward signal,
so its ceiling is the behavior policy itself: cloning a `medium` (mediocre) buffer well reproduces
mediocrity well. That is exactly why it is the weakest rung and the floor the value-based rungs must
clear — and why it is by far the cheapest (no critic training, no candidate fan-out).

## Hyperparameters

Reuse the harness backbone `DQLMlp` (emb_dim=64, positional timestep embedding) and wrapper
`DiscreteDiffusionSDE` with `diffusion_steps = sampling_steps = 5`, cosine schedule, actions in
`[-1,1]`. Actor LR `3e-4` with cosine decay; actor EMA at the wrapper default, evaluated with
`use_ema=True`. `actor.ema_update()` runs once `n_gradient_step >= 1000`. No critic, no `tau`/`eta`/
`weight_temperature`; `num_candidates` is ignored (sample-1 inference).

## Scaffold edit

```python
_FILE = "CleanDiffuser/pipelines/custom_policy.py"

# ============================================================================
# diffusion_policy baseline: diffusion BC only, no critic / no reranking
# ============================================================================

# --------------- Network Architecture -----------------
nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
nn_condition = IdentityCondition(dropout=0.0).to(args.device)

print(f"======================= Parameter Report of Diffusion Model =======================")
report_parameters(nn_diffusion)
print(f"==============================================================================")

# --------------- Diffusion Model Actor --------------------
actor = DiscreteDiffusionSDE(
    nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
    x_max=+1. * torch.ones((1, act_dim), device=args.device),
    x_min=-1. * torch.ones((1, act_dim), device=args.device),
    diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)

# ---------------------- Training ----------------------
if args.mode == "train":

    actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)

    actor.train()

    n_gradient_step = 0
    log = {"bc_loss": 0.}

    for batch in loop_dataloader(dataloader):

        obs = batch["obs"]["state"].to(args.device)
        act = batch["act"].to(args.device)

        bc_loss = actor.update(act, obs)["loss"]          # weight-free diffusion BC step
        actor_lr_scheduler.step()

        if n_gradient_step % args.ema_update_interval == 0 and n_gradient_step >= 1000:
            actor.ema_update()

        log["bc_loss"] += bc_loss

        if (n_gradient_step + 1) % args.log_interval == 0:
            log["gradient_steps"] = n_gradient_step + 1
            log["bc_loss"] /= args.log_interval
            print(f"TRAIN_METRICS gradient_steps={log['gradient_steps']} bc_loss={log['bc_loss']:.4f}")
            log = {"bc_loss": 0.}

        if (n_gradient_step + 1) % args.save_interval == 0:
            actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
            actor.save(save_path + f"diffusion_ckpt_latest.pt")

        n_gradient_step += 1
        if n_gradient_step >= args.gradient_steps:
            break

# ---------------------- Inference ----------------------
# (checkpoint load) -- critic load removed
#     actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
#     actor.eval()
#
# (eval prior) -- shrink from num_envs*num_candidates to num_envs
#     prior = torch.zeros((args.num_envs, act_dim), device=args.device)
#
# (action selection) -- single sample per env, no reranking
#     obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
#     act, _ = actor.sample(
#         prior, solver=args.solver, n_samples=args.num_envs, sample_steps=args.sampling_steps,
#         condition_cfg=obs, w_cfg=1.0, use_ema=args.use_ema, temperature=args.temperature)
#     sampled_act = act.clip(-1., 1.).cpu().numpy()
```
</content>
