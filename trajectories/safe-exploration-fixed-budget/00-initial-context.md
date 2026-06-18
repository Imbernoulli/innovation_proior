## Research question

A reinforcement-learning agent must navigate a Safety-Gymnasium task while keeping its accumulated
*cost* — a separate hazard signal, distinct from the task reward — under a fixed budget. The
specification is a constrained Markov decision process (CMDP): maximize the discounted reward return
`J_r(π)` subject to the discounted cost return `J_c(π) ≤ d`, with a hard limit `d = 25.0` that must
hold not only at convergence but be respected across three different environments with different
control geometries. The one transferable component this task isolates is the *constraint-handling
update rule*: given a fixed PPO backbone that already produces a reward advantage `adv_r` and a cost
advantage `adv_c` per transition, what rule for (a) updating the cost-penalty controller and (b)
combining the two advantage streams best keeps episode cost at or below 25 while retaining as much
reward as possible. The success metric `budget_success_rate` is binary per run — 1 if the final
episode cost is ≤ 25, else 0 — so a method that lowers cost on average but never crosses under the
budget scores zero on the thing the task actually measures.

## Prior art before the first rung

- **Constrained MDPs (Altman 1999).** The formal object: an MDP augmented with a cost function and a
  threshold `d`, feasible set `Π_C = {π : J_c(π) ≤ d}`, optimal constrained policy
  `π* = argmax_{π∈Π_C} J_r(π)`. It says *what* to solve but not how to solve it with a deep policy
  optimizer that is only partially converged at each iteration.
- **Policy-gradient backbones (Schulman et al. 2015, 2017).** TRPO controls policy drift with a hard
  KL trust region and a second-order step; PPO recovers the same reliability with a first-order
  clipped surrogate reused for `K` epochs. Both optimize a single scalar objective — neither carries a
  constraint, so a cost budget has to be folded into the advantage that the surrogate ascends.
- **Lagrangian duality for CMDPs (Paternain et al. 2019).** In occupancy-measure space the constrained
  problem is a linear program with zero duality gap, which is the theoretical license to convert the
  constraint into a penalty `λ·(J_c − d)` and learn `λ` by dual ascent. The gap: this clean duality
  lives in occupancy space, while neural policy optimization is nonconvex and only approximate, so the
  *controller* that drives `λ` — how fast it reacts, whether it overshoots, whether it wrecks reward —
  becomes the real algorithmic choice the task is asking about.

The weakest rung reacts to the last point by *not* reacting at all: it runs the PPO backbone and
ignores the cost signal entirely, establishing the floor that any constraint-handling rule must beat.

## The fixed substrate

The backbone is OmniSafe's on-policy PPO. The rollout loop, the two value functions (one for reward,
one for cost), the GAE estimators that produce `adv_r` and `adv_c`, the actor/critic optimizers, the
Safety-Gymnasium environments, and the evaluation script are all fixed and not editable. The training
loop (`learn`) — including the per-epoch `TRAIN_METRICS` print and the final `TEST_METRICS` print that
the harness parses for `ep_ret`, `ep_cost`, and `budget_success_rate` — is also fixed. Each
environment trains for 1M steps with `cost_limit = 25.0`.

## The editable interface

Only the `CustomLag` class in `custom_lag.py` is editable, and only two regions of it: the custom
import line, and the four methods `_init`, `_init_log`, `_update`, `_compute_adv_surrogate`. The
contract:

- `_init(self)` — must call `super()._init()` first; initialize the budget-control state (multiplier,
  any controller memory). `self._cfgs.lagrange_cfgs.cost_limit` (25.0) and
  `self._cfgs.lagrange_cfgs.lambda_lr` (0.035) are available.
- `_init_log(self)` — must call `super()._init_log()` first; register any logging keys.
- `_update(self)` — reads the current mean episode cost via
  `self._logger.get_stats('Metrics/EpCost')[0]`, updates the controller, then calls `super()._update()`
  (which runs the PPO epochs, calling `_compute_adv_surrogate` internally), then logs.
- `_compute_adv_surrogate(self, adv_r, adv_c)` — returns the per-transition advantage tensor the PPO
  surrogate ascends. The default returns `adv_r` only (ignores the constraint). This is where the
  reward and cost streams are combined.

The default scaffold fill (the weakest rung edits exactly this region):

```python
# EDITABLE region of custom_lag.py — default fill (ignores the cost budget)
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
        # Default: no multiplier update -- agent should design this
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        # Default: only use reward advantage (ignores safety constraints entirely).
        return adv_r
```

## Evaluation settings

Three Safety-Gymnasium environments, each a separate run, three seeds {42, 123, 456}:

- `SafetyPointGoal1-v0` — point robot navigating to goals through hazards.
- `SafetyCarGoal1-v0` — car robot (harder dynamics) on the same goal task.
- `SafetyPointButton1-v0` — point robot pressing goal buttons, the densest-cost geometry.

Each run: 1M environment steps, fixed `cost_limit = 25.0`. Reported per run: `ep_ret` (higher
better), `ep_cost` (lower better), `budget_success_rate` (= 1 if final `ep_cost ≤ 25`, else 0; higher
better).
