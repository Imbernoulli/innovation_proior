**Problem.** PPO-Lag's integral-only multiplier lowered cost everywhere but never crossed under the
budget — `budget_success_rate` 0 on every run — because a pure integrator reacts only to accumulated
violation and so responds too late, overshooting before λ builds. The fix is to shape the *transient*,
not just the asymptote.

**Key idea (PID Lagrangian).** Treat the training loop as a feedback control system: setpoint `d = 25`,
output `J_C`, control input λ, plant = PPO. Drive λ with a PID controller instead of integral-only.
Proportional `K_P·Δ` supplies damping (a current-error reflex); derivative `K_D·(rate of cost
increase)_+` anticipates rising cost and brakes before overshoot (rectified, one-sided, for the
one-sided constraint); integral `I = (I + K_I·Δ)_+` (anti-windup) holds the standing λ that pins
steady-state cost at the limit. `K_P = K_D = 0` recovers PPO-Lag exactly.

**Why.** Integral guarantees zero steady-state offset but is intrinsically late; proportional adds the
positive-semidefinite damping `β∇g∇gᵀ` (the penalty method's benefit without its indefinite-Hessian
baggage); derivative reads the trend and acts ahead of violation. Together they get cost under the line
*in time*. EMA smoothing (0.95) and a 10-epoch derivative delay tame the noise derivative control
amplifies. The advantage hook is unchanged: the same scale-normalized blend `(adv_r − λ·adv_c)/(1+λ)`.

**Hyperparameters.** `K_P = 0.1`, `K_I = 0.01`, `K_D = 0.01`; EMA α = 0.95; derivative delay window 10
epochs; λ clamped to `[0, 100]`; `cost_limit = 25.0`.

```python
# EDITABLE region of custom_lag.py — step 3: PID-Lag (PID-controlled multiplier)
import numpy as np
import torch

from collections import deque

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class CustomLag(PPO):

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._pid_kp: float = 0.1
        self._pid_ki: float = 0.01
        self._pid_kd: float = 0.01
        self._pid_i: float = 0.0
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._cost_ds: deque = deque(maxlen=10)
        self._cost_ds.append(0.0)
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        delta = float(Jc - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        self._delta_p = 0.95 * self._delta_p + 0.05 * delta
        self._cost_d = 0.95 * self._cost_d + 0.05 * float(Jc)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._lagrangian_multiplier = max(0.0, min(pid_o, 100.0))
        self._cost_ds.append(self._cost_d)
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'PID-Lag: combine advantages using PID-controlled multiplier.'
        penalty = self._lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)
```
