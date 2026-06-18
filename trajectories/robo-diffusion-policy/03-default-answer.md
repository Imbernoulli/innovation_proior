## Problem (from step 2)

IDQL lifts the floor by reranking candidates with a value function, but its actor is pure BC — it only
samples from `mu`, so the critic can at best pick the *best of `N` samples from `mu`*. Where the good
action lives outside `mu`'s frequent support (the hopper stall at `0.62`), no reranking surfaces it. The
move is to train the actor itself toward high-`Q` actions, not just select them.

## Key idea

**Diffusion Q-Learning (DQL):** a diffusion actor with a twin-Q critic, trained by behavior cloning
*plus* a Q-maximization term, with candidate reranking by `Q` at inference.

- **Actor objective** `L_actor = L_BC + eta * L_Q`. `L_BC` is the denoising BC loss (the anchor pinning
  the actor near `mu`); `L_Q = -E[Q(s, a_new)]` with `a_new` sampled *through* the diffusion reverse
  chain with gradients on, so the actor's weights move to make the *generated* action higher-`Q`. `eta`
  trades cloning against improvement; small `eta` recovers BC, large `eta` chases `Q` and risks OOD.
- **Randomized normalized double-Q** for `L_Q`: with prob 1/2 use
  `-q1.mean()/q2.abs().mean().detach()`, else the symmetric form. The detached other-head magnitude
  normalizes the Q-gradient to unit scale (so `eta` is dataset-independent) and randomizing heads
  symmetrizes the twin critics.
- **Critic**: twin-Q, TD target `r + gamma(1-done) min(Q1_t(s',a'), Q2_t(s',a'))` with `a'` sampled
  from the actor at `s'` (the clipped-double-min underestimation fights residual overestimation; the BC
  anchor keeps `a'` near support). No expectile, no separate `V`.
- **Inference**: rerank `num_candidates` actions by `softmax(Q * weight_temperature)`; since the actor
  is already trained toward high `Q`, reranking only sharpens.

## Why it works

Pushing the `Q`-gradient into the actor's weights moves the *sampling distribution* toward high-`Q`
regions, so the candidate set itself improves — breaking IDQL's best-of-`mu` ceiling, especially on
hopper where `medium`-buffer stitching toward a clean hop pays off (expect ~1.0). The BC anchor and the
clipped twin-min keep the more aggressive (actor-sampled) bootstrap from diverging. The cost: every step
samples two full diffusion chains (critic's `next_act`, actor's `new_act`) plus a coupled critic update,
so DQL is by far the slowest rung.

## Hyperparameters

Actor backbone `DQLMlp` (emb_dim=64), wrapper `DiscreteDiffusionSDE`, `diffusion_steps =
sampling_steps = 5`. Critic `DQLCritic` twin-Q, hidden = `hidden_dim`, Adam at `critic_learning_rate`
(3e-4), Polyak `0.995`. `eta` (Q-maximization weight) and `discount = 0.99` from args; actor and critic
updated **every** step; actor EMA after 1000 steps. Inference `num_candidates = 50`,
`weight_temperature`, `use_ema=True`. This is the unmodified scaffold template (the `default` baseline:
`OPS = []`).

## Scaffold edit (unmodified template — `OPS = []`)

```python
# --------------- Network Architecture -----------------
nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
nn_condition = IdentityCondition(dropout=0.0).to(args.device)

# --------------- Diffusion Model Actor --------------------
actor = DiscreteDiffusionSDE(
    nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
    x_max=+1. * torch.ones((1, act_dim), device=args.device),
    x_min=-1. * torch.ones((1, act_dim), device=args.device),
    diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)

# ------------------ Critic ---------------------
critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
critic_target = deepcopy(critic).requires_grad_(False).eval()
critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

# ---------------------- Training ----------------------
if args.mode == "train":

    actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
    critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)
    actor.train(); critic.train()
    n_gradient_step = 0
    prior = torch.zeros((args.batch_size, act_dim), device=args.device)

    for batch in loop_dataloader(dataloader):

        obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
        act = batch["act"].to(args.device)
        rew = batch["rew"].to(args.device)
        tml = batch["tml"].to(args.device)

        # Critic Training
        current_q1, current_q2 = critic(obs, act)
        next_act, _ = actor.sample(
            prior, solver=args.solver, n_samples=args.batch_size, sample_steps=args.sampling_steps,
            use_ema=True, temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)
        target_q = torch.min(*critic_target(next_obs, next_act))
        target_q = (rew + (1 - tml) * args.discount * target_q).detach()
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        critic_optim.zero_grad(); critic_loss.backward(); critic_optim.step()

        # Policy Training: BC anchor + eta * Q-maximization on freshly sampled actions
        bc_loss = actor.loss(act, obs)
        new_act, _ = actor.sample(
            prior, solver=args.solver, n_samples=args.batch_size, sample_steps=args.sampling_steps,
            use_ema=False, temperature=1.0, condition_cfg=obs, w_cfg=1.0, requires_grad=True)
        with FreezeModules([critic, ]):
            q1_new_action, q2_new_action = critic(obs, new_act)
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        actor_loss = bc_loss + args.task.eta * q_loss
        actor.optimizer.zero_grad(); actor_loss.backward(); actor.optimizer.step()

        actor_lr_scheduler.step(); critic_lr_scheduler.step()

        # ema
        if n_gradient_step % args.ema_update_interval == 0:
            if n_gradient_step >= 1000:
                actor.ema_update()
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

        n_gradient_step += 1
        if n_gradient_step >= args.gradient_steps:
            break

# ---------------------- Inference ----------------------
# load actor + critic/critic_target; rerank num_candidates by softmax over Q:
#     obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)
#     act, _ = actor.sample(prior, ..., n_samples=args.num_envs * args.num_candidates, ...)
#     with torch.no_grad():
#         q = critic_target.q_min(obs, act).view(-1, args.num_candidates, 1)
#         w = torch.softmax(q * args.task.weight_temperature, 1)
#         act = act.view(-1, args.num_candidates, act_dim)
#         indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
#         sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()
```
</content>
