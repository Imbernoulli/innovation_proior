## Research question

TD-MPC2 plans at every environment step through a learned latent world model. The design target is the **trajectory optimizer**: the rule that, given the current observation, turns a population of sampled action sequences into the single action to execute. The world model, value/reward heads, learned policy prior, and training loop are fixed and pretrained. At a fixed planning budget of 6 iterations × 512 samples, which zeroth-order, sampling-based optimizer extracts the most return from the same world model during both data collection and evaluation?

## Prior art / Background / Baselines

The optimizer may only query the model by rolling out an action sequence and reading a scalar value; it runs under `@torch.no_grad()`. The relevant prior methods share the scaffold below.

- **Random shooting.** Draw action sequences from a fixed distribution, roll each through the model, and execute the first action of the best sequence.
- **Differential dynamic programming / iLQG.** Linearize the dynamics and quadratize the cost around a nominal trajectory, then use a backward Riccati sweep to obtain a local feedback law. It requires a differentiable model and an explicit cost.
- **Model-predictive control wrapper.** Re-optimize at every step, execute only the first action, re-observe, and warm-start the next plan from the un-executed tail. This receding-horizon wrapper is already provided by the scaffold; the inner optimizer is the design target.

## Fixed substrate / Code framework

A pretrained TD-MPC2 agent is frozen. It exposes:
- `agent.model.encode(obs, task) → z`, latent transition `next(z, a, task) → z'`, policy prior `pi(z, task) → (a, info)`, reward head, and ensemble value head `Q`;
- `agent._estimate_value(z, actions, task)` — rolls a batch of horizon-length action sequences through the model and returns each sequence's predicted discounted return;
- warm-start buffer `agent._prev_mean` of shape `(horizon, action_dim)`;
- config `agent.cfg` with planning knobs: `horizon=3`, `num_samples=512`, `num_elites=64`, `num_pi_trajs=24`, `iterations=6`, `temperature=0.5`, `min_std`, `max_std`, `action_dim`, `multitask`;
- `common.math` utilities such as `gumbel_softmax_sample` and `two_hot_inv`.

The scaffold already mixes the learned policy into planning: `num_pi_trajs=24` of the `num_samples` sequences are warm-started by unrolling the policy through the latent model.

## Editable interface

Only one region is editable: `custom_plan(agent, obs, t0, eval_mode, task)` in `custom_planner.py` (lines 15–120). It runs under `@torch.no_grad()`, must update `agent._prev_mean` for temporal warm-starting, and must return one action tensor of shape `(action_dim,)` clamped to `[-1, 1]`.

The contract is: encode the observation; generate `num_pi_trajs` policy warm-start trajectories; initialize a per-timestep Gaussian over the action sequence (mean shifted from `_prev_mean`, std at `max_std`); iterate `cfg.iterations` rounds of sample → roll out → score → select elites → refit the distribution; then commit one action.

The default fill is the **MPPI** baseline shown below. Each candidate replaces exactly this function body.

```python
# EDITABLE region of custom_planner.py — default fill (TD-MPC2 MPPI)
@torch.no_grad()
def custom_plan(agent, obs, t0=False, eval_mode=False, task=None):
    """MPPI baseline -- Model Predictive Path Integral control."""
    cfg = agent.cfg

    # Sample policy trajectories as warm-starts
    z = agent.model.encode(obs, task)
    if cfg.num_pi_trajs > 0:
        pi_actions = torch.empty(
            cfg.horizon, cfg.num_pi_trajs, cfg.action_dim, device=agent.device,
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
        (cfg.horizon, cfg.action_dim), cfg.max_std, dtype=torch.float, device=agent.device,
    )
    if not t0:
        mean[:-1] = agent._prev_mean[1:]
    actions = torch.empty(
        cfg.horizon, cfg.num_samples, cfg.action_dim, device=agent.device,
    )
    if cfg.num_pi_trajs > 0:
        actions[:, :cfg.num_pi_trajs] = pi_actions

    # Iterate MPPI
    for _ in range(cfg.iterations):
        r = torch.randn(
            cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim, device=agent.device,
        )
        actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
        actions_sample = actions_sample.clamp(-1, 1)
        actions[:, cfg.num_pi_trajs:] = actions_sample
        if cfg.multitask:
            actions = actions * agent.model._action_masks[task]

        value = agent._estimate_value(z, actions, task).nan_to_num(0)
        elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
        elite_value = value[elite_idxs]
        elite_actions = actions[:, elite_idxs]

        # softmax-weighted (Gibbs) refit of mean and std
        max_value = elite_value.max(0).values
        score = torch.exp(cfg.temperature * (elite_value - max_value))
        score = score / score.sum(0)
        mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
        std = (
            (score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)
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

## Evaluation settings

Three DMControl continuous-control tasks — **walker-walk**, **cheetah-run**, and **cartpole-swingup** — over seeds {42, 123, 456}. The same 1M-parameter TD-MPC2 agent trained for 200K steps is used throughout; only the planner changes. Metric: **episode reward**, higher is better. Planning budget stays at the default 6 iterations × 512 samples.
