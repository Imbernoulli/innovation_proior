## Research question

TD-MPC2 learns a latent world model and then *plans* at every environment step: given the current
observation it must pick one action by optimizing a short-horizon sequence of actions through the
learned dynamics. The single thing being designed is the **trajectory optimizer** — the rule that, at
each step, turns a population of sampled action sequences (scored by rolling them through the model)
into the action to execute. Everything else — the world model, its value/reward heads, the policy
prior, the training loop — is fixed and pretrained. The question is narrow: at a fixed planning budget
(comparable to 6 iterations × 512 samples), which sampling-based optimizer extracts the most return
from the same world model, both during data collection in training and during evaluation?

## Prior art before the first rung (zeroth-order trajectory optimization)

The optimizer the ladder fills is a *zeroth-order* planner: it may only query the model (roll out an
action sequence, read a scalar value), never differentiate through it (the whole function runs under
`@torch.no_grad()`). These are the methods that line precedes the ladder; the scaffold below is the
shape they all share.

- **Random shooting (Rao 2009; Nagabandi et al. 2018).** Draw a batch of action sequences from a fixed
  distribution, roll each through the model, execute the first action of the best one. Gradient-free and
  global, but it *never learns from its own evaluations* — the sampling distribution on the hundredth
  step is identical to the first, so the budget is wasted re-drawing in regions already shown to be bad.
  Gap: no adaptation of where to sample.
- **Differential dynamic programming / iLQG (Jacobson & Mayne 1970; Todorov & Li 2005).** Taylor-expand
  the dynamics and quadratize the cost along a nominal trajectory, then a backward Riccati sweep gives a
  local feedback law — fast near a good nominal. Gap: built entirely on derivatives, so it needs a
  differentiable model and an honest quadratic cost; under `no_grad` over a learned latent model it is
  simply unavailable.
- **Model-predictive control (the wrapper, García et al. 1989).** Re-optimize the horizon every step,
  execute only the first action, re-observe, re-plan; warm-start the next step from the un-executed tail.
  This receding-horizon wrapper is *given* by the scaffold (the `_prev_mean` warm-start buffer and the
  per-step call); the ladder only fills the inner optimizer.

## The fixed substrate

A pretrained TD-MPC2 agent is frozen and must not be touched. It supplies a latent world model
(`agent.model`) with `encode(obs, task) → z`, latent transition `next(z, a, task) → z'`, a policy prior
`pi(z, task) → (a, info)`, a reward head `reward`, and an ensemble value head `Q`; a value-estimation
helper `agent._estimate_value(z, actions, task)` that rolls a batch of horizon-length action sequences
through the model and returns each sequence's predicted discounted return (predicted rewards plus a
bootstrapped terminal Q at the policy action); a warm-start buffer `agent._prev_mean` of shape
`(horizon, action_dim)`; and a config `agent.cfg` carrying the planning knobs — `horizon=3`,
`num_samples=512`, `num_elites=64`, `num_pi_trajs=24`, `iterations=6`, `temperature=0.5`,
`min_std`, `max_std`, `action_dim`, `multitask`. The loop also exposes `common.math` utilities
(`gumbel_softmax_sample`, `two_hot_inv`). The substrate already mixes a **policy prior** into planning:
`num_pi_trajs=24` of the `num_samples` trajectories are warm-started by rolling the learned policy
forward through the latent model, so the optimizer always sees the policy's own guess alongside the
sampled candidates.

## The editable interface

Exactly one region is editable — the `custom_plan(agent, obs, t0, eval_mode, task)` function in
`custom_planner.py` (lines 15–120). It runs under `@torch.no_grad()`, must update `agent._prev_mean`
for temporal warm-starting, and must return one action tensor of shape `(action_dim,)` clamped to
`[-1, 1]`. Every method on the ladder is a fill of this same contract: encode the observation, warm-start
`num_pi_trajs` policy trajectories, initialize a per-timestep Gaussian over the action sequence (mean
shifted from `_prev_mean`, std at `max_std`), then iterate `cfg.iterations` rounds of *sample → roll out
→ score → select elites → refit the distribution*, and finally commit one action. The methods differ
only in the refit rule (and how the final action is drawn).

The starting point is the scaffold default: the TD-MPC2 **MPPI** fill, shown below. Each later method
replaces exactly this function body and nothing else.

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

Three DMControl continuous-control tasks spanning the difficulty range — **walker-walk** (easy, a smooth
gait), **cheetah-run** (hard, a fast high-frequency running gait) and **cartpole-swingup** (a swing-up
then balance) — each over three seeds {42, 123, 456}. The world model is the same 1M-parameter TD-MPC2
trained for 200K steps; only the planner changes. Metric: **episode reward, higher is better** on all
three. The planner affects both the quality of data collected during training and action selection at
evaluation, so a better optimizer compounds. Planning budget is held comparable to the default
(6 iterations × 512 samples).
