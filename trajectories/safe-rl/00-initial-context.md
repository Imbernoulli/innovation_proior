## Research question

Safe reinforcement learning on Safety-Gymnasium navigation: an agent must maximize task reward while keeping expected episodic cost — the count of safety violations such as entering hazards or hitting obstacles — below a fixed budget `d = 25`. The cost is a separate signal, not a term inside the reward. The design problem is the constraint-handling mechanism: how the dual state is updated from the measured cost violation, and how reward and cost advantages are combined into the single advantage PPO ascends. Everything else — rollout loop, value functions, optimizer, environment interface — is fixed.

## Prior art / Background / Baselines

- **Constrained MDPs (Altman 1999).** A standard MDP augmented with a cost function and budget `d`; maximizing reward subject to `J_c <= d` is convex in the state-action occupancy measure, so the constrained problem has zero duality gap. In the function-approximation setting one falls back to iterative primal-dual schemes that approximate the saddle point.
- **Penalty / reward-shaping baselines.** Fold safety into reward as `r - beta*c` for a fixed scalar `beta`, then run unconstrained RL. The mapping from penalty weight to resulting cost return is policy- and environment-dependent.
- **Lagrangian / primal-dual methods.** Dualize the constraint with a nonnegative multiplier `lambda`, form `L = J_r - lambda*(J_c - d)`, and alternate policy ascent with dual descent, updating `lambda` from the running violation `J_c - d`.

## Fixed substrate / Code framework

A constrained-PPO loop is frozen and must not be touched. It owns the Safety-Gymnasium environment and vectorized rollout; two value functions with separate GAE, producing a reward advantage `adv_r` and a cost advantage `adv_c` per transition; the PPO actor/critic update (clipped surrogate, value losses, Adam); and the `Metrics/EpCost` and `Metrics/EpRet` logging. The agent file is a `CustomLag` subclass of PPO. The base class supplies the fixed machinery and exposes two slots: `_update` reads the current mean episodic cost from `self._logger.get_stats('Metrics/EpCost')[0]` and calls `super()._update()` to run the PPO step; `_compute_adv_surrogate(adv_r, adv_c)` returns the single advantage PPO ascends. The budget is `self._cfgs.lagrange_cfgs.cost_limit = 25.0`; the dual step size default is `0.035`.

## Editable interface

Only one region is editable: the constraint-handling methods of `CustomLag` in `custom_lag.py` — the imports line and the methods `_init`, `_init_log`, `_update`, and `_compute_adv_surrogate`. `_init` sets up multiplier state after `super()._init()`; `_init_log` registers `Metrics/LagrangeMultiplier` after `super()._init_log()`; `_update` reads the measured mean episodic cost `Jc`, updates the multiplier, calls `super()._update()` to run the PPO step, and logs the multiplier; `_compute_adv_surrogate(adv_r, adv_c)` returns the single advantage PPO ascends.

The starting point is the default scaffold: no constraint handling. The multiplier stays at zero and `_compute_adv_surrogate` returns `adv_r` unchanged — reward-only PPO that ignores safety.

```python
# EDITABLE region of custom_lag.py -- default fill (no constraint handling)

# (no extra imports)

@registry.register
class CustomLag(PPO):
    """Custom Lagrangian-based safe RL algorithm."""

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        # Default: no multiplier update -- the mechanism is what gets designed
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Default: only use reward advantage (ignores safety constraints entirely)."""
        return adv_r
```

## Evaluation settings

Three Safety-Gymnasium navigation environments — **SafetyPointGoal1-v0**, **SafetyCarGoal1-v0**, and **SafetyPointButton1-v0** — over seeds {42, 123, 456}, each trained for the benchmark's fixed step budget. Metrics per environment: episode return (higher is better) and episode cost (lower is better, with threshold `25.0`). A method succeeds only if it earns return while holding cost at or below the budget across all three environments.
