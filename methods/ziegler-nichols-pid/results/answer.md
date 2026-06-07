# PID feedback control and the Ziegler–Nichols tuning rules

## Problem

A single-loop controller must hold a measured variable at a setpoint despite load changes, where the process is a black box of unknown internal lags. Proportional-only control fails two ways at once: it leaves a steady-state **offset** after any load change (it can hold only one valve position at zero error), and raising its gain to shrink that offset drives the loop into **oscillation** — there is an ultimate gain at which the loop sustains a steady cycle. One knob cannot deliver both zero offset and good damping. We need a richer feedback law *and* a model-free way to set its constants in the field.

## Key idea

Use three control effects on the error e = setpoint − measurement:
- **Proportional (P)** — basic restoring action, ∝ e.
- **Integral / reset (I)** — ∝ ∫e dt; moves the valve at a velocity set by the error, so it only stops when e = 0, killing the offset.
- **Derivative / pre-act (D)** — ∝ de/dt; anticipates and damps the oscillation, and lets P and I be set more aggressively.

Then **tune model-free** by letting the process reveal its own gain and timescale, two ways:
- **Ultimate-cycle (closed-loop):** with I and D off, raise the proportional gain until the loop holds a sustained oscillation. Record the ultimate gain Kᵤ and its period Tᵤ.
- **Reaction-curve (open-loop):** open the loop, step the valve, record the S-shaped response; read its maximum slope (reaction rate R, normalized to a unit step as R₁ = R/ΔF) and its **lag** L (dead time, where the inflection tangent meets the initial level).

Set every constant as a fixed fraction of those measured numbers, chosen to give **quarter-amplitude decay** (each overshoot ¼ of the last). The two methods are one recipe, linked by Kᵤ ≈ 2/(R₁L) and Tᵤ ≈ 4L.

## Algorithm

Parallel-form law (with anti-windup and derivative on measurement):

    u(t) = Kp·e + Ki·∫e dt − Kd·d(measurement)/dt,   Ki = Kp/Ti,  Kd = Kp·Td

where Ti is the integral (reset) time, reset rate = 1/Ti, and Td is the derivative (pre-act) time.

**Ultimate-cycle tuning** (from Kᵤ, Tᵤ):

| Controller | Kp | Ti | Td |
|---|---|---|---|
| P | 0.5 Kᵤ | — | — |
| PI | 0.45 Kᵤ | Tᵤ/1.2 ≈ 0.83 Tᵤ | — |
| PID | 0.6 Kᵤ | Tᵤ/2 | Tᵤ/8 |

(reset rate: PI = 1.2/Tᵤ, PID = 2/Tᵤ; pre-act: PID = Tᵤ/8.)

**Reaction-curve tuning** (from unit reaction rate R₁ and lag L):

| Controller | Kp | Ti | Td |
|---|---|---|---|
| P | 1/(R₁L) | — | — |
| PI | 0.9/(R₁L) | L/0.3 ≈ 3.33 L | — |
| PID | 1.2/(R₁L) | 2L | 0.5 L |

(reset rate: PI = 0.3/L, PID = 0.5/L; pre-act: PID = 0.5 L.) In first-order-plus-dead-time terms R₁ = K/T (process gain over time constant), so PID Kp = 1.2·T/(K·L).

Procedure: measure (Kᵤ, Tᵤ) or (R₁, L); look up Kp, Ti, Td; form Ki = Kp/Ti, Kd = Kp·Td; run the controller; if the response is too lively, scale all three down slightly. For cascaded loops (e.g. a quadrotor), tune the innermost loop first, close it, then tune outward.

## Code

```python
from math import inf


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class PID:
    """Single-loop PID, parallel form, e = setpoint - measurement:
        u = Kp*e + Ki*∫e dt - Kd*d(measurement)/dt
    with conditional-integration anti-windup and derivative-on-measurement
    (no derivative kick on setpoint steps), plus a 1st-order derivative filter."""

    def __init__(self, Kp, Ki, Kd, dt, out_min=-inf, out_max=+inf, tau=0.0):
        self.Kp, self.Ki, self.Kd, self.dt = Kp, Ki, Kd, dt
        self.out_min, self.out_max, self.tau = out_min, out_max, tau
        self.integ = 0.0
        self.prev_meas = None
        self.deriv = 0.0

    def update(self, setpoint, meas):
        e = setpoint - meas
        P = self.Kp * e
        if self.prev_meas is None:
            self.prev_meas = meas
        d_meas = (meas - self.prev_meas) / self.dt        # derivative on measurement
        if self.tau > 0.0:
            a = self.dt / (self.tau + self.dt)
            self.deriv = (1.0 - a) * self.deriv + a * d_meas
        else:
            self.deriv = d_meas
        D = -self.Kd * self.deriv

        integ_try = self.integ + self.Ki * e * self.dt
        u_raw = P + integ_try + D
        u = clamp(u_raw, self.out_min, self.out_max)
        # anti-windup: keep the integral only if not winding into saturation
        if u_raw == u or (e > 0.0) != (u_raw > u):
            self.integ = integ_try
        self.prev_meas = meas
        return u


def tune_ultimate_cycle(Ku, Tu, kind="PID"):
    """Ziegler-Nichols closed-loop rule from ultimate gain Ku and period Tu."""
    if kind == "P":
        Kp, Ti, Td = 0.5 * Ku, inf, 0.0
    elif kind == "PI":
        Kp, Ti, Td = 0.45 * Ku, Tu / 1.2, 0.0          # reset rate 1.2/Tu
    elif kind == "PID":
        Kp, Ti, Td = 0.6 * Ku, Tu / 2.0, Tu / 8.0      # reset 2/Tu, pre-act Tu/8
    else:
        raise ValueError(kind)
    Ki = Kp / Ti if Ti != inf else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd


def tune_reaction_curve(R1, L, kind="PID"):
    """Ziegler-Nichols open-loop rule from unit reaction rate R1 and lag L.
    Equivalent to the closed-loop rule via Ku=2/(R1*L), Tu=4*L."""
    if kind == "P":
        Kp, Ti, Td = 1.0 / (R1 * L), inf, 0.0
    elif kind == "PI":
        Kp, Ti, Td = 0.9 / (R1 * L), L / 0.3, 0.0      # reset rate 0.3/L
    elif kind == "PID":
        Kp, Ti, Td = 1.2 / (R1 * L), 2.0 * L, 0.5 * L  # reset 0.5/L, pre-act L/2
    else:
        raise ValueError(kind)
    Ki = Kp / Ti if Ti != inf else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd


# Cascaded quadrotor axis: inner attitude loop nested in outer position loop.
# Tune the inner loop to its own ultimate cycle, close it, tune the outer loop.
inner = PID(*tune_ultimate_cycle(Ku_rate, Tu_rate, "PID"), dt=0.002,
            out_min=-1.0, out_max=1.0)     # motor-differential command
outer = PID(*tune_ultimate_cycle(Ku_pos, Tu_pos, "PI"), dt=0.01,
            out_min=-0.3, out_max=0.3)     # commanded tilt angle


def step(pos_sp, pos, ang):
    tilt_cmd = outer.update(pos_sp, pos)   # position error -> desired tilt
    return inner.update(tilt_cmd, ang)     # tilt error -> motor differential
```
