IDQL did exactly what the floor analysis predicted and the numbers draw the boundary of inference-time selection sharply. It beat the diffusion-BC floor on every environment — hopper $0.49\to0.62$, walker2d $0.66\to0.83$, halfcheetah $0.42\to0.48$ — with tight seeds, confirming that the same diffusion-BC actor plus a value function and an advantage reranking lifts the score wherever $Q$/$V$ carry useful information. But look at *where* it stalled. Walker2d, where $\mu$ has a coherent gait with exploitable spread, jumped most ($+0.17$); halfcheetah barely moved, because its `medium` buffer is a competent-but-low-ceiling runner and even perfect selection over 50 samples from $\mu$ cannot conjure actions better than the best $\mu$ would draw; and hopper, at $0.62$, is still a long way from solved. That last point is the diagnosis I must act on: IDQL's actor is *pure BC*. It only ever samples from $\mu$, and the critic merely picks the best of those samples, so the ceiling of "best of 50 samples from $\mu$" is bounded by what $\mu$ puts mass on. If the genuinely good action lives in a region $\mu$ samples rarely or never, no amount of reranking will surface it — the candidate set never contains it. The move is to stop merely *selecting* good actions and start *training the actor to prefer them*: push the gradient of $Q$ into the actor's own parameters so the distribution it samples from drifts toward high-$Q$ regions. Then the candidate set itself improves, and selection only sharpens.

I propose **DQL** — Diffusion Q-Learning — a diffusion actor with a twin-Q critic, trained by behavior cloning *plus* a $Q$-maximization term, with candidate reranking by $Q$ at inference. The danger is the same OOD wall as before, just relocated. I want the actor to maximize $\mathbb{E}_{a\sim\pi}[Q(s,a)]$ — sample an action from the diffusion actor at state $s$ and ascend $Q$ on it — but if I *only* maximize $Q$, the actor marches off the data support to wherever the critic is erroneously optimistic, and I am back to exploiting phantom optima. The fix is a behavior-regularized objective: keep the diffusion-BC term as an anchor pinning the actor near $\mu$, and add the $Q$-maximization term as the pull toward better actions,
$$L_{\text{actor}}=L_{\text{BC}}+\eta\,L_Q,\qquad L_Q=-\mathbb{E}[Q(s,a_{\text{new}})],$$
where $L_{\text{BC}}$ is the noise-prediction denoising loss (the same objective the floor and IDQL used to clone $\mu$) and $a_{\text{new}}$ is *freshly sampled from the actor's own reverse chain* with gradients enabled, so the gradient flows through the diffusion sampler back into the actor's weights — the actor's parameters move so the *generated* action has higher $Q$, not so that some external action does. The coefficient $\eta$ trades cloning against improvement: small $\eta$ stays close to $\mu$ (safe; $\eta\to0$ recovers IDQL's pure BC), large $\eta$ chases $Q$ harder but risks the OOD blow-up. This regularized policy class is what diffusion buys here — an expressive actor that can place mass on a *better* action than any single $\mu$ mode, while the BC anchor keeps it from leaving the support.

Two things need care, and both are visible in the harness's exact form. The new action is sampled with `use_ema=False` (I want gradients on the live weights, not the EMA copy) and `requires_grad=True`, then $Q$ is evaluated on it inside a `FreezeModules([critic])` block so the $Q$-gradient updates only the actor, not the critic. The subtle, harness-specific part is the *scale* of $L_Q$: it must not let one twin head dominate or blow up. The mechanism is a **randomized, normalized double-Q** trick — with probability one half,
$$q_{\text{loss}}=-\frac{\bar q_1}{\;\overline{|q_2|}\,(\text{detached})\;},\qquad\text{else}\quad q_{\text{loss}}=-\frac{\bar q_2}{\;\overline{|q_1|}\,(\text{detached})\;}.$$
The numerator is the head being maximized; the denominator is the *detached* absolute scale of the *other* head. Dividing by the other head's magnitude normalizes the $Q$-gradient to roughly unit scale, so $\eta$ has a stable, dataset-independent meaning, and randomizing which head is numerator versus denominator each step symmetrizes the two critics and prevents the actor from over-fitting to one head's idiosyncratic over-estimates. This is not the textbook $-\bar Q$; it is the literal mechanism this harness exposes, and it is what makes the BC$+Q$ balance robust across the three environments without per-environment retuning of $\eta$.

The critic side is simpler than IDQL's, and that simplification is itself the point. DQL does not need the expectile/value-net machinery, because its actor is no longer pure BC: the actor is being pushed toward high-$Q$ actions, so the natural critic target uses the actor's *own* next action rather than an in-sample expectile. The critic is a twin-Q (`DQLCritic`) with TD target
$$r+\gamma(1-\text{done})\,\min\big(Q_1^{\text{targ}}(s',a'),\,Q_2^{\text{targ}}(s',a')\big),\qquad a'\sim\text{actor}(\cdot\mid s').$$
This *does* query an action the actor proposes at $s'$, precisely the OOD query IDQL refused — but here it is tolerable for two linked reasons: the actor is BC-anchored so $a'$ stays near the support, and the twin-min `torch.min(*critic_target(next_obs, next_act))` is the standard clipped-double-Q underestimation pressure that fights the residual overestimation. So DQL trades IDQL's strict in-sample safety for a more aggressive but still-regularized bootstrap, and that aggression is the source of its higher ceiling: the critic can value actions slightly better than $\mu$'s, and the actor is trained to produce them. The critic loss is the plain `F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)`, with a Polyak target update at $0.995$ — no expectile, no separate $V$.

Against the harness this is the *unmodified template* — the default fill — so the "edit" is to change nothing, but it is worth stating how it differs from the rungs below. Versus diffusion BC, DQL adds the entire `DQLCritic` (twin-Q, target, optimizer), adds the critic TD-update reading `next_obs/rew/tml`, and replaces the actor's pure BC step with `actor.loss(act, obs) + eta * q_loss` where `q_loss` flows through a gradient-enabled actor sample. Versus IDQL, DQL swaps the actor backbone back to `DQLMlp` and the critic from `IDQLQNet + IDQLVNet` (expectile $V$, in-sample SARSA) to a single `DQLCritic` twin-Q with an actor-sampled bootstrap target; it *adds* the `eta * q_loss` actor term IDQL deliberately omits; and at inference it reranks `num_candidates` actions by a softmax over the *bare* $Q$ (`critic_target.q_min`) rather than over the *advantage* $Q-V$. That last difference is consistent with the training difference: IDQL needed the advantage because its baseline $V$ was the expectile; DQL has no $V$, so it reranks on $Q$ directly, and because the actor is already trained toward high $Q$, both the candidate set and the ranking pull the same direction — the heavy lifting has already happened in the actor's weights. The schedule also tightens: DQL updates the critic and the actor on *every* gradient step (not every other, as IDQL did), because both are now doing coupled work — the critic must keep up with an actor actively moving toward high-$Q$ regions, and a stale critic would let the actor exploit the lag. The EMA on the actor still kicks in only after 1000 steps, and the critic target is Polyak-updated every `ema_update_interval` steps. This coupling is more expensive: every step samples *two* full diffusion chains (one for the critic's `next_act`, one for the actor's `new_act`), where the floor sampled none in training and IDQL sampled none in training either — which is why DQL is by far the slowest rung. I expect it to beat IDQL on hopper especially, where training the actor toward high $Q$ breaks the best-of-$\mu$ ceiling that capped IDQL at $0.62$ and `medium`-buffer stitching toward a clean hop pays off most (approaching the expert-level $\sim1.0$), with a smaller gain on the low-ceiling halfcheetah and a hold-or-improve on the already-strong walker2d — the geometric mean coming out clearly above IDQL's, which is the claim that training-time $Q$-maximization beats inference-time selection on these datasets. The full scaffold module — the twin-Q critic, the coupled BC$+Q$ actor training with the randomized normalized double-Q `q_loss`, and the $Q$-reranking inference block — follows.

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
