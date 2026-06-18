**Problem.** The Adam dual loop (PPOLag) pulled cost down from naive but left it nearly twice over budget
on every environment and seed (PointGoal 45.6, CarGoal 46.7, PointButton 56.2, against `d = 25`), while
already paying heavily in return. It gave up the reward of a safe method without delivering the safety:
the integral-only multiplier reacts only to *accumulated* violation, so it crawls toward feasibility too
slowly to get there inside the training budget, and cranking `lambda_lr` only makes a delayed plant
oscillate harder.

**Key idea.** Read the safe-RL loop as a feedback control system — cost limit `d` is the setpoint,
measured episodic cost `J_c` is the output, `lambda` is the control input, the PPO step is the plant — and
note the dual update `lambda <- [lambda + K_I*(J_c - d)]_+` is *integral-only* control. Replace it with a
full PID controller on the cost-constraint error: proportional + integral + derivative.

**Why it works.** Through Platt & Barr's second-order collapse, the proportional term injects a
positive-semidefinite `beta*(grad_g)(grad_g)^T` into the damping matrix — the quadratic-penalty method's
damping, but touching only the lambda update, not the policy objective, and without the penalty's bundled
indefinite Hessian term — so it damps the oscillation and reacts to the *current* gap immediately. The
derivative term, decoupled through `B = I + gamma*(grad_g)(grad_g)^T`, creates a velocity-quadratic,
curvature-modulated force that acts *ahead* of violations (rectified to brake cost increases only). The
integral term stays because only it supplies the standing lambda that gives zero steady-state violation.
Setting `K_P = K_D = 0` recovers the integral-only dual update exactly — this is a strict generalization.

**This task's implementation.** The CPPOPID configuration: gains `kp = 0.1`, `ki = 0.01`, `kd = 0.01`;
integral with anti-windup `I <- max(0, I + ki*delta)`; EMA-smoothed proportional input and smoothed cost,
both with coefficient 0.95 (`x <- 0.95*x + 0.05*new`); rectified delayed-difference derivative over a
queue of 10 epochs `pid_d = max(0, cost_d - cost_ds[0])`; output `max(0, kp*delta_p + I + kd*pid_d)`. The
harness exposes none of the general controller's sum-norm / diff-norm / penalty-cap branches, fixes the
EMA at 0.95 and the delay at 10 inline, and seeds the integral at 0.0 (lambda starts at zero). The
advantage blend is unchanged — `(adv_r - lambda*adv_c)/(1 + lambda)` — so the only difference from PPOLag
is the controller on lambda, making the comparison clean.

**Hyperparameters.** `kp = 0.1, ki = 0.01, kd = 0.01`, derivative delay = 10 epochs, EMA alpha = 0.95,
`cost_limit = 25.0`, integral/lambda init = 0.0.

**What to watch (the bar).** Cost should land at or below 25 on all three environments — PointButton from
56 into the mid-20s or below, the others from the mid-40s under 25 — which is the thing neither naive nor
the integral-only loop achieved. Return is the honest price: it need not recover toward naive and may drop
below the PPOLag 15/19/4, because actually holding cost at the budget forces detours the dual loop skipped.
Success on the actual objective is *controlled cost* across all environments; that is the bar.

```python
# EDITABLE region of custom_lag.py -- step 3: PID Lagrangian (CPPOPID)

from collections import deque

@registry.register
class CustomLag(PPO):
    """PID Lagrangian: a PID controller on the dual multiplier (CPPOPID defaults)."""

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        # PID controller gains (CPPOPID defaults)
        self._pid_kp: float = 0.1
        self._pid_ki: float = 0.01
        self._pid_kd: float = 0.01
        # PID state
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
        # PID update
        delta = float(Jc - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        self._delta_p = 0.95 * self._delta_p + 0.05 * delta
        self._cost_d = 0.95 * self._cost_d + 0.05 * float(Jc)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._lagrangian_multiplier = max(0.0, pid_o)
        self._cost_ds.append(self._cost_d)
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """PID Lagrangian: combine advantages using the PID-controlled multiplier."""
        penalty = self._lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)
```
