**Problem.** Bare CEM lifted cheetah's good seeds (870, 907) but a contaminated elite set collapsed one seed to 722, and cartpole slipped to 867 on the same fragility. The leak is the refit: CEM's hard top-`k` average weights the single best elite the same as the marginal 64th and discards the graded value information, so a few lucky near-threshold elites can hijack the fit and the error compounds over six iterations.

**Key idea (MPPI / path-integral refit).** Replace the hard equal-weight elite average with the *soft, value-weighted* Gibbs refit. The optimal sampling target is the Gibbs distribution `q*(τ) ∝ exp(V(τ)/λ)·p(τ)` (the free-energy lower bound is tight there), so importance-sample it: `mean = Σ score·elite_actions`, `std = sqrt(Σ score·(elite_actions − mean)²)`, with `score = softmax(temperature·V)` over the elites. Shift by the max elite value before exponentiating for scale-robust numerics across the three tasks. Commit a Gibbs-drawn *evaluated* elite via `gumbel_softmax_sample`, not the never-evaluated centroid.

**Why.** The exponential pulls the mean toward the genuinely-best elites and down-weights near-threshold ones to nearly nothing, so a lucky marginal elite can no longer hijack the fit — the graded information CEM discarded is exactly what stabilizes the refit and its std. CEM is the `λ→∞` limit of this (equal weights), so MPPI generalizes rather than replaces it; one temperature, no per-task tuning. This is the TD-MPC2 default planner — the optimizer the world model was trained to plan with.

**Hyperparameters.** Substrate `horizon=3`, `num_samples=512`, `num_elites=64`, `num_pi_trajs=24`, `iterations=6`, `temperature=0.5` (= `1/λ`), `[min_std, max_std]`. Elites top-`k` by value; soft-weight *within* the truncated elite set (truncation keeps the outlier-sensitive std fit clean). Final action via Gumbel-softmax over elite scores (plus a Gaussian kick off eval).

**What to watch.** The bar to clear is CEM's: beat 833 cheetah and 867 cartpole, and do it by *closing the seed gap*. Cheetah's ceiling (870–907) is set by the world model, so the gain should come from recovering the 722 collapse seed — cheetah mean toward ~890 with a tighter spread, cartpole back above 875, walker flat at ceiling.

```python
@torch.no_grad()
def custom_plan(agent, obs, t0=False, eval_mode=False, task=None):
    """MPPI baseline -- Model Predictive Path Integral control."""
    cfg = agent.cfg

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

    # Iterate MPPI
    for _ in range(cfg.iterations):
        # Sample actions from Gaussian
        r = torch.randn(
            cfg.horizon, cfg.num_samples - cfg.num_pi_trajs,
            cfg.action_dim, device=agent.device,
        )
        actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
        actions_sample = actions_sample.clamp(-1, 1)
        actions[:, cfg.num_pi_trajs:] = actions_sample
        if cfg.multitask:
            actions = actions * agent.model._action_masks[task]

        # Evaluate trajectories and select elites
        value = agent._estimate_value(z, actions, task).nan_to_num(0)
        elite_idxs = torch.topk(
            value.squeeze(1), cfg.num_elites, dim=0,
        ).indices
        elite_value = value[elite_idxs]
        elite_actions = actions[:, elite_idxs]

        # Update sampling distribution (softmax-weighted)
        max_value = elite_value.max(0).values
        score = torch.exp(cfg.temperature * (elite_value - max_value))
        score = score / score.sum(0)
        mean = (
            (score.unsqueeze(0) * elite_actions).sum(dim=1)
            / (score.sum(0) + 1e-9)
        )
        std = (
            (
                score.unsqueeze(0)
                * (elite_actions - mean.unsqueeze(1)) ** 2
            ).sum(dim=1)
            / (score.sum(0) + 1e-9)
        ).sqrt()
        std = std.clamp(cfg.min_std, cfg.max_std)
        if cfg.multitask:
            mean = mean * agent.model._action_masks[task]
            std = std * agent.model._action_masks[task]

    # Select action from elites via Gumbel sampling
    rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
    actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
    a, std_final = actions[0], std[0]
    if not eval_mode:
        a = a + std_final * torch.randn(cfg.action_dim, device=agent.device)
    agent._prev_mean.copy_(mean)
    return a.clamp(-1, 1)
```
