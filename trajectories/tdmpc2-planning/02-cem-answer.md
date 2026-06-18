**Problem.** iCEM's fixed `β=0.5` colored noise imposed a low-frequency prior on the action sequence that one task — cheetah-run's fast, high-frequency running gait — violates, dropping it to a wide-scattered 796 mean while the smooth tasks stayed at ceiling. The keep-elites and noise-decay machinery only propagated and narrowed around that biased pool. The lever to pull is not per-task `β` tuning; it is removing the spectral prior entirely and letting the elite fit decide each task's plan shape.

**Key idea (CEM).** Strip back to bare Cross-Entropy Method: sample *white* Gaussian noise, keep the top-`K` elites by predicted value, and refit the Gaussian to them by maximum likelihood — `mean = elite mean`, `std = elite std`, per coordinate, clamped. This is the closed-form MLE fit of the closest tractable density to the variance-optimal restricted target `g*(x) ∝ I{S(x)≥γ}·f(x)`; minimizing the KL to `g*` reduces, on a finite sample, to the log-likelihood of the threshold-beating samples. No colored noise, no kept elites, no decay.

**Why.** CEM's per-coordinate elite std lets the *data* set each timestep's spread independently — tight where elites agree, wide where they disagree. On walker/cartpole the elites already carry smoothness, so CEM reconstructs it without a prior; on cheetah the elites disagree fast across the horizon, so the std stays wide on the alternating coordinates and the fast-reversing plans survive. White noise imposes no spectrum; the refit installs whatever spectrum each task's elites carry.

**Hyperparameters.** Substrate `horizon=3`, `num_samples=512`, `num_elites=64`, `num_pi_trajs=24`, `iterations=6`, `[min_std, max_std]`. Elites are top-`k` by **value** (higher is better). Refit is the **unweighted** elite mean/std — softmax weighting is deliberately left for the next rung. Final action is the first entry of the converged mean (plus a Gaussian kick off eval).

**What to watch.** Walker and cartpole should hold near 978/881 (the elites supply smoothness). Cheetah-run is the falsifiable claim: removing the high-frequency suppression should *raise* its mean above iCEM's 796 and *tighten* the seed spread. If even bare CEM leaves cheetah short of ceiling, the hard top-`k` average is discarding the graded "how much better" value information — which the next rung's soft Gibbs weighting recovers.

```python
@torch.no_grad()
def custom_plan(agent, obs, t0=False, eval_mode=False, task=None):
    """CEM baseline -- Cross-Entropy Method for trajectory optimization."""
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

    # Iterate CEM
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
        elite_actions = actions[:, elite_idxs]

        # CEM update: simple mean/std of elites (no softmax weighting)
        mean = elite_actions.mean(dim=1)
        std = elite_actions.std(dim=1).clamp(cfg.min_std, cfg.max_std)
        if cfg.multitask:
            mean = mean * agent.model._action_masks[task]
            std = std * agent.model._action_masks[task]

    # Select action: use the mean (deterministic) or sample
    a = mean[0]
    if not eval_mode:
        a = a + std[0] * torch.randn(cfg.action_dim, device=agent.device)
    agent._prev_mean.copy_(mean)
    return a.clamp(-1, 1)
```
