TD-MPC2 hands me one empty slot — the rule that, at each environment step, turns a population of action sequences (scored by rolling them through a frozen latent world model) into the next sampling distribution and finally into the action to execute. The whole function runs under `@torch.no_grad()` over a learned latent model, so I have no trustworthy gradient through the dynamics and the derivative-based optimizers are simply off the table; what I have is a fixed budget of roughly six iterations times five hundred twelve samples per step, and the game is to spend those samples well. The crudest thing that works is random shooting — draw a batch from a fixed Gaussian, roll each through the model, execute the first action of the best — but it never learns from its own evaluations: the distribution it samples on the last iteration is identical to the first, so it keeps paying for draws in regions it has already proven bad. The natural fix is the Cross-Entropy Method, which after each batch refits the sampling Gaussian's mean and standard deviation to its top-$K$ elites. But before I settle for CEM I want to ask whether each rollout can be made *more informative* at the same budget — and there is one specific way it can.

The leak is the **shape** of the noise CEM samples. CEM draws every timestep of the action sequence independently from a diagonal Gaussian: white, temporally uncorrelated noise along the horizon. Consider what a white-noise action sequence does to the dynamics, with the crudest possible link $dx/dt = a(t)$ for intuition — the state is the running integral of the action. If $a(t)$ has independent increments, then $x(t)$ is a Brownian random walk, and the defining fact about a Brownian walk is that it does not go anywhere: its expected squared displacement grows only like time, the increments cancel, and a fixed budget of action energy buys only a small net excursion. A white-noise rollout jitters around its starting latent state. When the value surface near the current state is flat — the cheetah is already running and a small jitter barely changes the predicted return — most of my five hundred sampled sequences come back saying nearly the same thing, and the elite set is decided by jitter rather than by genuinely different, far-ranging plans.

So I propose **iCEM** — improved CEM — trimmed to this slot, which keeps CEM's elite refit but attacks the noise shape with three additions. The first and load-bearing one is *colored noise*: replace white sampling with noise that is positively autocorrelated along the horizon, so a sampled sequence persists in a direction long enough to drive the latent state somewhere the value head can actually distinguish. The clean construction tilts the action sequence's power spectral density toward low frequencies with a power-law $1/f^\beta$ spectrum, drawn in the frequency domain by scaling random Fourier coefficients and inverse-transforming, where $\beta = 0$ is white and larger $\beta$ is smoother. But that full real-FFT sampler — with its DC-bin regularization, Nyquist realness correction, variance-restoring $\sqrt{2}$ factors and unit-variance normalizer — is more spectral machinery than a `no_grad` planning loop that runs hundreds of times per episode wants, and over a horizon of only three steps the frequency-domain construction has barely any bins to shape. So I take the cheap time-domain surrogate that carries the same intent: pass white noise through a first-order autoregressive smoother down the horizon,

$$\tilde\varepsilon_0 = \varepsilon_0, \qquad \tilde\varepsilon_t = \beta\,\tilde\varepsilon_{t-1} + (1-\beta)\,\varepsilon_t,$$

with a single correlation strength $\beta = 0.5$. This is an exponential moving average down the three planning steps — each step's perturbation inherits half of the previous step's, so a sampled sequence persists instead of flipping sign every step. It is the one-pole low-pass filter whose stationary spectrum *is* a reddened, $1/f$-like tilt: the same low-frequency bias as the power-law sampler, obtained with a recursion the loop can run in two lines. I give up the exact unit-variance single-knob interpretation but keep the load-bearing physics — correlated actions integrate to longer, smoother latent excursions than white actions at the same draw scale, so the elites separate on *plans* rather than on jitter.

The second addition closes a second CEM leak cheaply. Standard CEM uses each iteration's elite set only to refit `mean` and `std`, then throws the actual sequences away and draws a completely fresh population. But those elites were the best sequences found so far; after the refit shifts the distribution a little, much of the new population will be *worse* than the elites just deleted. So I carry the top `keep_fraction = 0.1` of this iteration's elites into the next iteration's candidate pool alongside the fresh colored samples — known-good plans survive long enough to compete again. Only a fraction, not all of them, and for a concrete variance reason: the elites have, by construction, *small* spread, so if they dominated the pool the refit would collapse `std` almost immediately and kill exploration before it begins. A tenth of the elite count retains the best plans without swamping the fresh draws that drive the fit. The third addition is annealing: the colored noise gives reach, which I want *early* while the distribution is broad and I am still locating the good basin, not *late* once the elites have concentrated. So I scale the sampled noise by a `noise_scale` that starts at one and decays by `noise_decay = 0.9` each iteration — exploration-then-exploitation written directly into the noise magnitude, complementing CEM's own `std` shrinkage.

Grounding this in the edit surface: the substrate already warm-starts `num_pi_trajs = 24` of the `num_samples` trajectories by rolling the learned policy through the latent model, and those keep the first slots of the action buffer, so each iteration's candidate pool is *policy prior + kept elites + fresh colored samples*. The mean is shifted from `_prev_mean` for temporal warm-starting and the std starts at `max_std`. Because `agent._estimate_value` returns a predicted return (higher is better), my elites are the *top-k by value*. I refit with the plain CEM rule — elite mean and elite std, clamped to $[\texttt{min\_std}, \texttt{max\_std}]$ — and I keep the refit **unweighted** on purpose: the soft Gibbs weighting is a later move, not this one. The only things I change relative to plain CEM are the noise source, the kept elites, and the decay. The final action is the first entry of the converged mean, with a small Gaussian exploration kick added off eval, and I write the mean back to `_prev_mean`.

I can already name the risk, and it is on the hardest task. Walker-walk and cartpole-swingup are smooth, low-frequency tasks whose optimal control genuinely persists in a direction, so colored sampling should land good plans efficiently and both should sit near the world model's ceiling. But cheetah-run is a fast, high-frequency running gait — its optimal action sequence reverses direction often — and my EMA smoother with $\beta = 0.5$ *suppresses* exactly that high-frequency content, biasing every sampled sequence toward slow, persistent pushes. The colored noise that helps the smooth tasks should *hurt* cheetah-run, because I have hard-coded a single correlation strength rather than letting the task choose it. If that is what happens, the diagnosis for the next rung writes itself: the spectral prior is task-mismatched, and the safer move is to drop the colored noise and the keep/decay machinery and return to plain white-noise CEM, letting the elite fit — not a hand-set $\beta$ — decide the shape of the plan.

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
