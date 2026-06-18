**Problem.** TD-MPC2 must pick an action every step by optimizing a horizon-3 action sequence through a frozen latent world model, under `@torch.no_grad()` and a fixed sampling budget. Plain CEM refits a Gaussian to its top-`K` elites, but it draws *white* (temporally uncorrelated) noise: white-noise action sequences integrate to short, diffusive latent excursions that barely separate, so at a tight budget the elites are decided by jitter, not by genuinely different plans.

**Key idea (iCEM, trimmed to this slot).** Keep CEM's elite refit but make each rollout more informative. (1) **Colored noise**: replace white sampling with temporally correlated noise so a sequence persists in a direction long enough to drive the latent state somewhere the value head can distinguish. Over a 3-step horizon I implement the correlation as a one-pole exponential moving average down the horizon, `ε̃ₜ = β·ε̃ₜ₋₁ + (1−β)·εₜ` with `β=0.5` — the cheap time-domain surrogate for a `1/f^β` low-frequency tilt, which is all the spectral shaping this `no_grad` planning loop can afford. (2) **Keep-elites**: carry the top `keep_fraction=0.1` of each iteration's elites into the next iteration's pool so known-good plans are not re-discovered (only a fraction, or their tiny spread collapses the refit). (3) **Noise decay**: scale the noise by `noise_decay=0.9` per iteration — wide correlated reach early, tightening refinement late.

**Why.** Correlated actions range farther than white actions at the same draw scale, so the elites separate on plans rather than on noise; keeping elites and decaying the noise turn the fixed budget into more effective refinement. The refit stays the *unweighted* CEM mean/std — soft weighting is deliberately not used here.

**Hyperparameters.** `colored_noise_beta=0.5`, `noise_decay=0.9`, `keep_fraction=0.1`; the substrate's `horizon=3`, `num_samples=512`, `num_elites=64`, `num_pi_trajs=24`, `iterations=6`, `[min_std, max_std]`. Elites are top-`k` by predicted **value** (higher is better). Final action is the first entry of the converged mean (plus a Gaussian kick off eval).

**What to watch.** Smooth tasks (walker-walk, cartpole-swingup) should sit near ceiling — their optimal control genuinely persists. Cheetah-run is the risk: a fast high-frequency running gait, whose fast-reversing actions the `β=0.5` smoother suppresses, so the fixed correlation strength should *hurt* it. That mismatch is what forces dropping the colored-noise prior at the next rung.

```python
@torch.no_grad()
def custom_plan(agent, obs, t0=False, eval_mode=False, task=None):
    """iCEM baseline -- improved CEM with colored noise and keep-elites."""
    cfg = agent.cfg
    colored_noise_beta = 0.5  # temporal correlation strength
    noise_decay = 0.9  # per-iteration noise decay factor
    keep_fraction = 0.1  # fraction of elites kept across iterations

    # Sample policy trajectories as warm-starts
    z = agent.model.encode(obs, task)
    if cfg.num_pi_trajs > 0:
        pi_actions = torch.empty(
            cfg.horizon, cfg.num_pi_trajs, cfg.action_dim,
            device=agent.device,
        )
        _z = z.repeat(cfg.num_pi_trajs, 1)
        for t in range(cfg.horizon - 1):
            pi_actions[t], _ = agent.model.pi(_z, task)
            _z = agent.model.next(_z, pi_actions[t], task)
        pi_actions[-1], _ = agent.model.pi(_z, task)

    # Initialize sampling distribution
    z = z.repeat(cfg.num_samples, 1)
    mean = torch.zeros(cfg.horizon, cfg.action_dim, device=agent.device)
    std = torch.full(
        (cfg.horizon, cfg.action_dim), cfg.max_std,
        dtype=torch.float, device=agent.device,
    )
    if not t0:
        mean[:-1] = agent._prev_mean[1:]
    actions = torch.empty(
        cfg.horizon, cfg.num_samples, cfg.action_dim,
        device=agent.device,
    )
    if cfg.num_pi_trajs > 0:
        actions[:, :cfg.num_pi_trajs] = pi_actions

    n_keep = max(1, int(cfg.num_elites * keep_fraction))
    kept_actions = None  # elites kept from previous iteration
    noise_scale = 1.0

    # Iterate iCEM
    for iteration in range(cfg.iterations):
        n_new = cfg.num_samples - cfg.num_pi_trajs
        if kept_actions is not None:
            n_new = n_new - kept_actions.shape[1]

        # Generate temporally correlated (colored) noise
        white_noise = torch.randn(
            cfg.horizon, n_new, cfg.action_dim,
            device=agent.device,
        )
        # Apply temporal smoothing: exponential moving average along horizon
        colored_noise = torch.zeros_like(white_noise)
        colored_noise[0] = white_noise[0]
        for t in range(1, cfg.horizon):
            colored_noise[t] = (
                colored_noise_beta * colored_noise[t - 1]
                + (1 - colored_noise_beta) * white_noise[t]
            )

        # Sample actions
        actions_sample = (
            mean.unsqueeze(1)
            + noise_scale * std.unsqueeze(1) * colored_noise
        )
        actions_sample = actions_sample.clamp(-1, 1)

        # Combine: policy trajs + kept elites + new samples
        start_idx = cfg.num_pi_trajs
        if kept_actions is not None:
            actions[:, start_idx : start_idx + kept_actions.shape[1]] = kept_actions
            start_idx += kept_actions.shape[1]
        actions[:, start_idx : start_idx + n_new] = actions_sample

        if cfg.multitask:
            actions = actions * agent.model._action_masks[task]

        # Evaluate trajectories and select elites
        value = agent._estimate_value(z, actions, task).nan_to_num(0)
        elite_idxs = torch.topk(
            value.squeeze(1), cfg.num_elites, dim=0,
        ).indices
        elite_actions = actions[:, elite_idxs]

        # Keep top elites for next iteration
        kept_actions = elite_actions[:, :n_keep]

        # Update distribution (simple CEM-style)
        mean = elite_actions.mean(dim=1)
        std = elite_actions.std(dim=1).clamp(cfg.min_std, cfg.max_std)
        if cfg.multitask:
            mean = mean * agent.model._action_masks[task]
            std = std * agent.model._action_masks[task]

        # Decay noise for refinement
        noise_scale *= noise_decay

    # Select action: use the mean
    a = mean[0]
    if not eval_mode:
        a = a + std[0] * torch.randn(cfg.action_dim, device=agent.device)
    agent._prev_mean.copy_(mean)
    return a.clamp(-1, 1)
```
