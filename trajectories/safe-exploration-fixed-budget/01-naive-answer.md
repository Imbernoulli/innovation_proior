**Problem.** Safe RL under a fixed cost budget (`d = 25.0`) on three Safety-Gymnasium tasks. The
substrate is a PPO backbone with two critics — one fit to reward, one to cost — producing `adv_r` and
`adv_c`. The task is the constraint-handling rule; this rung supplies none, to fix the floor.

**Key idea (no constraint handling).** Run unconstrained PPO. The policy ascends the clipped surrogate
on the reward advantage only; the cost advantage and the Lagrange multiplier are computed/stored but
never used. There is no controller — the multiplier is pinned at `0.0` and `_compute_adv_surrogate`
returns `adv_r`.

**Why this is the floor.** With the cost stream discarded, the policy is a pure reward maximizer in a
hazard field: it learns the task with no pressure to avoid hazards, so cost is governed entirely by the
geometry of the reward-seeking path. It should post the highest reward of any rung and the worst cost,
with `budget_success_rate` zero everywhere by construction.

**Hyperparameters.** PPO backbone defaults (fixed substrate); `cost_limit = 25.0`; multiplier ≡ 0.0;
no controller state.

```python
# EDITABLE region of custom_lag.py — step 1: naive (pure PPO, ignores cost)
import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class CustomLag(PPO):

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
        # Naive: keep multiplier at zero and ignore budget pressure.
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'Naive: ignore cost advantage entirely, optimize reward only.'
        return adv_r
```
