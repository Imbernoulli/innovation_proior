**Problem.** Naive PPO ignored the cost stream and violated the budget on every run (cost 51–152 vs
limit 25, `budget_success_rate` 0 everywhere). The fix is to put the constraint into the objective the
policy ascends, with a weight that adapts to the measured violation instead of a hand-set penalty.

**Key idea (Lagrangian PPO).** Cast safe RL as the constrained MDP `max J_r s.t. J_c ≤ d` and solve its
Lagrangian saddle by primal-dual ascent. A multiplier `λ ≥ 0` is updated each epoch from the mean
episode cost by dual ascent `λ ← [λ + η(Jc − d)]_+` (over budget → λ up; under budget → λ down). The
policy ascends the combined advantage, scale-normalized so PPO's step size stays sane as λ grows:
`A = (adv_r − λ·adv_c)/(1 + λ)`.

**Why.** A constraint with a threshold saturates — get cost under `d`, then spend everything on
reward — matching the binary `budget_success_rate`. λ is the running integral of the violation, so it
self-tunes the reward/cost trade-off the fixed penalty could never set, and adapts per environment. The
`(1+λ)` divisor keeps `A = (1−u)adv_r − u·adv_c` (`u = λ/(1+λ)`) a convex blend, preserving the `1:λ`
direction while bounding the gradient scale.

**Hyperparameters.** OmniSafe `Lagrange` helper (Adam on λ); `lambda_lr = 0.035`; `cost_limit = 25.0`;
λ init from `lagrange_cfgs`; nonnegative clamp via the helper's projection.

```python
# EDITABLE region of custom_lag.py — step 2: PPO-Lag (adaptive Lagrangian multiplier)
import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.lagrange import Lagrange


@registry.register
class CustomLag(PPO):

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        self._lagrange.update_lagrange_multiplier(Jc)
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'PPO-Lag: penalize cost advantage using the learned multiplier.'
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
```
