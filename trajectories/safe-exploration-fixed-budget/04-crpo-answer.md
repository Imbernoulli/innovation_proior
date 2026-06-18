**Problem.** PID-Lag got cost under the budget (first nonzero `budget_success_rate`: PointGoal 1.0,
CarGoal/Button 0.667) but at near-total reward collapse (0.20 / 2.0 / 0.46), and two seeds tipped just
over the line. Both failures share a root: the Lagrangian penalty is *always on* — even when feasible
the policy pays a safety tax and rides the boundary, killing reward and leaving no noise margin.

**Key idea (CRPO — constraint-rectified policy optimization).** Drop the dual variable. At the level of
what to do each epoch the reward-vs-cost trade-off is a *switch*, not a blend: if feasible, spend the
step on reward; if infeasible, spend it on reducing cost. Read the mean episode cost `Jc`; if
`Jc <= cost_limit + η` feed `adv_r` to the PPO surrogate (maximize reward); else feed `−adv_c` (minimize
cost). No λ, no dual LR, no integral/PID, no `(1+λ)` rescaling — the constraint enters only through the
`if`.

**Why.** A sharp switch carries *zero* cost penalty in the feasible region, so the policy chases reward
at full strength when it has slack (recovering what PID-Lag threw away), and reacts to a violation
immediately at full strength (no dual lag). The tolerance `η` is a deadband that absorbs noisy `Jc`
estimates — it stops chattering at the boundary and gives the reward step room — and matches the
canonical analysis's `1/√T` slack; under accurate estimates the switch decides correctly each epoch,
giving an `O(1/√T)` rate to a globally optimal feasible policy with `O(1/√T)` violation. This is the
canonical primal CRPO update line for line.

**Hyperparameters.** `cost_limit = 25.0`; tolerance `η = distance = 2.0` (canonical OmniSafe default);
PPO backbone defaults (fixed substrate). Multiplier slot logged as 1.0/0.0 = a readout of the switch.

```python
# EDITABLE region of custom_lag.py — finale: CRPO (constraint-rectified switch)
import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class CustomLag(PPO):

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._distance: float = 2.0
        self._in_violation: bool = False
        self._rew_update: int = 0
        self._cost_update: int = 0
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        self._in_violation = bool(Jc > self._cost_limit + self._distance)
        if self._in_violation:
            self._cost_update += 1
            self._lagrangian_multiplier = 1.0
        else:
            self._rew_update += 1
            self._lagrangian_multiplier = 0.0
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'CRPO: within budget+tolerance, maximize reward; otherwise descend cost only.'
        if self._in_violation:
            return -adv_c
        return adv_r
```
