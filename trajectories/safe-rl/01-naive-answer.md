**Problem.** Safe RL formalizes the goal as a constrained MDP: maximize reward subject to expected
episodic cost staying under a budget `d = 25`. Before designing any constraint mechanism, the floor is
the constraint-*unaware* agent — it fixes the upper bound on reward (when nothing is paid for cost)
and the uncontrolled cost (when reward is optimized freely). That floor is the denominator every later
method is judged against.

**Key idea (the fixed base learner).** The substrate is a constrained-PPO loop with two value
functions — one for reward, one for cost — each with its own GAE, so the loop produces a reward
advantage `adv_r` and a cost advantage `adv_c` per transition, and exposes how they combine. PPO's
clipped surrogate is agnostic to what the advantage means, so a constraint mechanism plugs in by (1)
updating a nonnegative multiplier `lambda` from the measured cost violation and (2) blending `adv_r`
and `adv_c` through `lambda`.

**Step-1 edit.** The naive fill disconnects safety: `lambda` stays at `0.0` (the `_update` slot only
asserts the cost statistic is not NaN, then runs the PPO step), and `_compute_adv_surrogate` returns
`adv_r`, dropping `adv_c` entirely. The cost critic still trains but never reaches the policy, so no
violation can change behavior. It is the floor by construction.

**Hyperparameters.** No mechanism hyperparameters. The base learner uses the substrate's fixed PPO
settings; `cost_limit = 25.0` is read but unused.

**What to watch.** Expect the highest returns on the ladder and a cost far above `25` on every
environment (worst on the hazard-dense SafetyPointButton1-v0). The size of that over-budget gap is the
safety debt step 2 must pay down, and the return given up to do so is the price then measured.

```python
# EDITABLE region of custom_lag.py -- step 1: naive (no constraint handling)

# (no extra imports)

@registry.register
class CustomLag(PPO):
    """Naive baseline: pure reward-maximizing PPO, safety ignored."""

    def _init(self) -> None:
        super()._init()
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        # Naive: no multiplier update, stays at 0
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Naive: ignore cost advantage entirely, optimize reward only."""
        return adv_r
```
