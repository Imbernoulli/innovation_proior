PID Lagrangian updates the Lagrange multiplier `lambda` in a constrained-MDP safe-RL
problem with a **PID controller** on the cost-constraint error, instead of the traditional
integral-only subgradient update. It reinterprets the safe-RL training loop as a feedback control
system — cost limit `d` is the setpoint, measured episodic cost `J_C` is the output, `lambda` is
the control input, and the policy-optimization step is the (uncertain, nonlinear) plant — and
applies proportional + integral + derivative control to make constraint satisfaction *responsive*
during training, not just at convergence.

## Problem it solves

Constrained MDP: `max_pi J(pi) s.t. J_C(pi) <= d`. The traditional Lagrangian update `lambda <-
(lambda + K_I (J_C - d))_+` is pure integral control: it accumulates the running violation, reacts
late, and makes the cost oscillate about and overshoot the limit throughout training (the cost and
`lambda` curves show a 90-degree phase lag, the signature of an ill-tuned integrator). Its single
gain `K_I` forces a bad trade — larger `K_I` damps oscillation but degrades reward. The goal is to
hold cost near/below `d` at *every* iteration while preserving reward, staying as simple as the
Lagrangian baseline.

## Key idea

Expand the multiplier update from one term to three:

- **Proportional** `K_P (J_C - d)` — instantaneous reaction to the current violation. In the
  continuous Platt-Barr dynamics it enters `lambda_dot` as `beta g_dot`, adding `beta grad g grad^T g`
  (a positive-semidefinite outer product) to the damping matrix → damps oscillations. Same damping
  the quadratic penalty method gives, but without modifying the primal objective/gradient and
  without the penalty's extra possibly-indefinite `c g grad^2 g` term.
- **Integral** `K_I integral(J_C - d)` — the original term; kept because only it supplies the
  nonzero steady-state `lambda` that eliminates residual violation at convergence (proportional and
  derivative both vanish when the error and its trend go to zero).
- **Derivative** `K_D d/dt(J_C)` — predictive; raises `lambda` when cost is *rising*, before it
  crosses `d`. In the continuous dynamics it enters `lambda_dot` as `gamma g_ddot`, which gives
  `x_ddot + B^{-1} A x_dot + (alpha g + gamma x_dot^T grad^2 g x_dot) B^{-1} grad g = 0`,
  with `B = I + gamma grad g grad^T g`. The new velocity-quadratic term is modulated by constraint
  curvature and the acceleration equation carries it with a negative sign, so positive curvature
  along the motion pushes in the `-B^{-1} grad g` direction. Rectified `(.)_+` so it brakes cost
  increases but does not impede decreases.

Setting `K_P = K_D = 0` recovers the traditional Lagrangian method exactly — a strict generalization.

## Final algorithm (PID-controlled multiplier)

```
Choose gains K_P, K_I, K_D >= 0;  I <- 0;  J_C_prev <- 0
repeat each iteration k:
    receive J_C                              # estimated mean episodic cost
    Delta   <- J_C - d                       # proportional error (constraint violation)
    partial <- ( J_C - J_C_prev )_+          # derivative term, rectified
    I       <- ( I + Delta )_+               # integral, anti-windup clamp
    lambda  <- ( K_P Delta + K_I I + K_D partial )_+   # PID output, lambda >= 0
    J_C_prev <- J_C
    return lambda
```

Step-size consistency (used in the policy loop, for the PID method and the baselines): rescale the
objective by `1/(1+lambda)`, since `arg max (J - lambda J_C) = arg max (J - lambda J_C)/(1+lambda)`.
This keeps the policy-gradient magnitude bounded as `lambda` grows, giving the advantage blend

```
adv = ( adv_r - lambda * adv_c ) / ( 1 + lambda ).
```

Use **separate** reward and cost critics (`V_R`, `V_C`), not a single critic on `r + lambda c`,
because `lambda` now changes rapidly and a folded target would be stale.

## Practical estimation (noisy minibatch costs)

`J_C` is a sample average over finished trajectories, so smooth the controller inputs:

- EMA the proportional input and the cost for the derivative (factors near 1).
- Compute the derivative as a rectified difference of the smoothed cost against a **delayed** copy
  from `d_delay` iterations back (a clean finite difference over a window, not a one-step diff).
- Leave the integral un-smoothed (it is already a low-pass accumulation).
- In the concrete constrained-PPO implementation, `diff_norm` clamps both the integral accumulator
  and the output penalty to `[0, 1]`; without `diff_norm` or `sum_norm`, the output is capped by
  `penalty_max`.

## Reward-scale invariance (complementary technique)

Scaling reward by `rho` forces `lambda*` to scale by `rho`, breaking gain transfer across
environments/scales. Weight the cost gradient by `beta = || grad J || / || grad J_C ||` so that
`lambda = 1` always means reward and cost contribute equal-magnitude gradients; then `lambda*` is
encouraged toward 1 regardless of reward scale and `lambda_0 = 1` is the natural init. A KL-ratio
policy-space variant can balance update sizes instead of raw gradient norms, but it requires extra
policy-update comparisons.

## Working code

Filling the multiplier-update and advantage-blend slots of a constrained-PPO harness. EMA on the
proportional and derivative inputs, derivative as a rectified delayed difference, integral with
anti-windup, output projected nonnegative; advantage blend rescaled by `1/(1+lambda)`.

```python
from __future__ import annotations
from collections import deque

import numpy as np
import torch


class PIDLagrangian:
    """PID controller on the Lagrange multiplier for an inequality safety constraint.
    K_P = K_D = 0 recovers the traditional integral-only Lagrangian update."""

    def __init__(
        self,
        pid_kp: float,
        pid_ki: float,
        pid_kd: float,
        pid_d_delay: int,
        pid_delta_p_ema_alpha: float,
        pid_delta_d_ema_alpha: float,
        sum_norm: bool,
        diff_norm: bool,
        penalty_max: int,
        lagrangian_multiplier_init: float,
        cost_limit: float,
    ) -> None:
        self._pid_kp = pid_kp
        self._pid_ki = pid_ki
        self._pid_kd = pid_kd
        self._pid_d_delay = pid_d_delay
        self._pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
        self._sum_norm = sum_norm
        self._diff_norm = diff_norm
        self._penalty_max = penalty_max
        self._pid_i = lagrangian_multiplier_init   # integral accumulator (gain folded in)
        self._cost_ds: deque[float] = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p = 0.0                          # EMA of the error (proportional input)
        self._cost_d = 0.0                           # EMA of the cost (for the derivative)
        self._cost_limit = cost_limit
        self._cost_penalty = 0.0                     # the multiplier (controller output)

    @property
    def lagrangian_multiplier(self) -> float:
        return self._cost_penalty

    def pid_update(self, ep_cost_avg: float) -> None:
        delta = float(ep_cost_avg - self._cost_limit)              # Delta = J_C - d
        # integral with gain folded in, anti-windup clamp at 0
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        # smoothed proportional input
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        # smoothed cost for a clean derivative
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        # rectified delayed-difference derivative (brakes cost increases only)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._cost_penalty = max(0.0, pid_o)                       # lambda >= 0
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)                         # advance the delay queue


def compute_adv_surrogate(
    adv_r: torch.Tensor, adv_c: torch.Tensor, lam: float
) -> torch.Tensor:
    """Step-size-consistent reward/cost advantage blend."""
    return (adv_r - lam * adv_c) / (1.0 + lam)


# Wiring into the fixed constrained-PPO outer step:
class ConstrainedPPO:
    def _init_controller(self) -> None:
        self._pid = PIDLagrangian(**self.cfg.lagrange_cfgs)

    def _update_multiplier(self, ep_cost: float) -> None:
        ep_cost = float(ep_cost)
        assert not np.isnan(ep_cost), 'cost is nan'
        self._pid.pid_update(ep_cost)
        self._lambda = self._pid.lagrangian_multiplier

    def _compute_adv_surrogate(
        self, adv_r: torch.Tensor, adv_c: torch.Tensor
    ) -> torch.Tensor:
        return compute_adv_surrogate(adv_r, adv_c, self._lambda)
```
