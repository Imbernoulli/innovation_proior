The colored-noise planner told me exactly what I had feared, and it told me in the one place I was watching. Walker-walk and cartpole-swingup came back near ceiling — $978.30$ and $881.23$ mean, tight across seeds — so on the smooth, low-frequency tasks the temporal-correlation prior cost nothing. But cheetah-run is the tell: mean $796.46$, with an enormous per-seed spread of $\{700.66, 791.38, 897.33\}$. That is not a budget problem; it is a *bias* problem, and the exact one I flagged. Cheetah-run's optimal action sequence reverses direction often, and the one-pole EMA with $\beta = 0.5$ suppresses precisely that high-frequency content — every sampled sequence inherited half of the previous step's perturbation, so the candidate pool was tilted toward slow persistent pushes and the elites were chosen from a pool that under-represented the fast-reversing plans cheetah rewards. The wide seed scatter is the signature: when the right plan lives in the frequencies my sampler damps, whether a seed stumbles into it is luck. And the keep-elites and noise-decay machinery did not rescue it — keeping a tenth of the elites only *propagates* the spectral bias forward, and decaying the noise only narrows the search around a mean already drawn from the wrong-shaped pool. The fix is not to tune $\beta$ per task — a planner should not need per-task spectral tuning — it is to remove the prior entirely and let the elite fit decide each task's plan shape.

I propose stripping back to the bare **Cross-Entropy Method**: sample white noise, keep the best elites, refit the Gaussian's mean and standard deviation to them, repeat — no colored noise, no kept elites, no decay. I want to re-derive it from the top, because the derivation tells me exactly what the refit does and does not give me. I have a black-box objective — hand it an action sequence, the value helper returns a predicted return — and I want its maximizer, with no trustworthy gradient, many local optima, expensive queries, and the sequence living in $\texttt{horizon} \times \texttt{action\_dim}$ dimensions so I cannot grid it. Reframe the problem: pick a high threshold $\gamma$ and ask not "where is the maximum?" but "how probable is it that a random draw clears $\gamma$?" — the rare-event probability $\ell(\gamma) = P(S(X) \ge \gamma)$ under a broad baseline density $f$. As I push $\gamma$ up toward the true optimum, the set $\{x : S(x) \ge \gamma\}$ shrinks down toward the maximizers and its probability under a broad density tends to zero. A density that made *that* rare event likely would sample almost exclusively near the optimum — so finding the optimizer and making the near-optimal event common are the same problem. The rare-event toolkit gives the ideal sampling density directly: importance-sample the indicator, ask which proposal makes the estimator zero-variance, and out falls

$$g^*(x) \;\propto\; \mathbb{I}\{S(x) \ge \gamma\}\, f(x),$$

the baseline density restricted to the elite event and renormalized. That is the "concentrate on the good region" distribution, and it is not a heuristic — it is the variance-minimizing target.

I cannot sample $g^*$ directly (it contains an unknown normalizer and is an arbitrary restricted shape), so I fit the closest member of a tractable family $f(\cdot;v)$ in KL divergence. Minimizing $D(g^*, f(\cdot;v))$ drops the $v$-free entropy term and reduces to maximizing $\mathbb{E}_{g^*}[\ln f(X;v)]$; substituting $g^*$ and Monte-Carlo-ing the expectation under the current law turns it into $\max_v \sum_k \mathbb{I}\{S(X_k) \ge \gamma\}\, \ln f(X_k; v)$ — the **log-likelihood of the threshold-beating samples**. So the principled update is: take the points that beat the bar and fit your sampling distribution to them by maximum likelihood; the cross-entropy minimization collapses, on a finite sample, to a maximum-likelihood fit on the elite set. A fixed high $\gamma$ would leave the elite set empty under a broad initial distribution, so I let $\gamma$ track the sample — keep the top-$\rho$ fraction every iteration — which is self-tuning and ratchets the threshold up as the distribution concentrates.

Choosing the family settles the rest. I refit every iteration, so I want a closed-form MLE and cheap sampling — the Gaussian. Its elite-restricted MLE is two lines: the log-density's $\mu$-derivative gives $\sum(x_k - \mu)/\sigma^2 = 0$, so $\mu$ is the **sample mean of the elites**; the $\sigma$-derivative gives $\sigma^2 = \frac{1}{N^e}\sum(x_k - \mu)^2$, the **MLE variance of the elites**. The whole Gaussian update is therefore: take the elite sequences, compute their per-coordinate mean and standard deviation, and those are the next iteration's sampling parameters. The mean re-centers the search where the good sequences clustered; the standard deviation re-sizes how widely to explore *each coordinate* — automatically tighter on the coordinates the elites agreed about, wider on those they did not. That second part is exactly what the colored-noise prior fought against: CEM's per-coordinate std lets the *data* decide each timestep's spread independently, so on cheetah, where good plans need large fast-alternating actions, the elite std on those coordinates simply opens up, with no $\beta$ damping it shut. The shape of the plan is *read off* the elites rather than imposed — which is why bare CEM should beat the colored version on cheetah without giving anything up on the smooth tasks, where the elites already agree on smooth persistent actions and CEM reconstructs that smoothness on its own. White noise imposes no spectrum; the refit installs whatever spectrum each task's elites carry.

In the edit surface: the substrate warm-starts `num_pi_trajs = 24` policy trajectories into the first slots of the action buffer, and I fill the remaining `num_samples - num_pi_trajs` slots with *white* Gaussian samples $\texttt{mean} + \texttt{std}\cdot\texttt{randn}$ clamped to $[-1, 1]$, the mean shifted from `_prev_mean` and std starting at `max_std`. Each of the `cfg.iterations = 6` rounds: sample, score with `agent._estimate_value` (predicted return, higher is better, so elites are the top-k by value), select the `num_elites = 64` best, and refit `mean = elite_actions.mean`, `std = elite_actions.std` clamped to $[\texttt{min\_std}, \texttt{max\_std}]$. The refit is the **unweighted** elite mean and std — every elite counts equally — which is the precise distinction from the rung above me that I am deliberately *not* yet taking: a soft, value-weighted Gibbs refit would use the magnitudes the value head reports, whereas CEM throws away all information about *how much* better the best elite is than the marginal one at the hard top-k threshold. I leave that lever for next. For the final action I take the first entry of the converged mean — the distribution's point estimate of the optimum, which in the concentrated limit *is* the optimum — with a small Gaussian kick off eval, and write the mean back to `_prev_mean`.

So the delta from the colored-noise rung is a principled subtraction. Against the measured iCEM numbers I expect walker-walk and cartpole to hold essentially where they are ($978$-ish and $881$-ish), because the smooth tasks were never hurt by the prior and the refit reconstructs the same smoothness from the elites. Cheetah-run is the falsifiable claim: removing the high-frequency suppression should *raise* its mean above iCEM's $796$ and, more tellingly, *tighten* the seed spread, since the $\{700, 791, 897\}$ scatter was the signature of luck deciding whether a seed found the fast-reversing plan my smoother damped. And if even bare CEM leaves cheetah short of what the world model can deliver, the next move is already visible — the hard top-k average discards the graded "how much better" information the value head reports, which a soft value-weighted Gibbs refit would recover.

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
