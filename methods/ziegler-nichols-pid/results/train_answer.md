A controller sits between a measured variable — temperature, pressure, level, flow — and a valve, and its single job is to hold that variable at a setpoint despite load changes, the disturbances that demand a sustained shift in valve position. The simplest and near-universal corrective law makes valve action proportional to the pen's deviation from setpoint, with one knob, the sensitivity, and it only half-works. It fails in two ways at once that are visible on any chart record. First, it leaves a standing offset: a proportional controller commands exactly one valve position when the pen sits at setpoint, so when a load change forces the process to need a *different* sustained valve position, the only way the law can produce it is for the pen to park away from setpoint, far enough that sensitivity times deviation equals the new demand. That residual error is inversely proportional to sensitivity and directly proportional to the load change, and it is baked into the structure. Second, the obvious cure — crank sensitivity up so the offset shrinks — runs straight into instability: there is a definite ultimate sensitivity $S_u$ above which any oscillation grows and below which it decays, and exactly at $S_u$ the loop holds a steady cycle whose amplitude ratio is $1$. The one knob is trapped between two evils, offset and oscillation, that pull it in opposite directions. No single proportional gain can deliver both zero offset and good damping. And the deeper trouble is that even with a richer law, a real process is a black box of unknown lags whose differential equation is a forbidding mess of exponentials no field engineer will solve per installation. We need both a better feedback law and a quick, model-free way to set its constants on the actual plant.

I propose the three-term proportional–integral–derivative controller together with the Ziegler–Nichols tuning rules that set its constants from a measured response. The structure follows term for term from Minorsky's 1922 ship-steering analysis, where a good helmsman steers not only on present heading error but on how long the ship has been off course and how fast its heading is swinging: the rudder angle becomes a sum of a term in the deviation, a term in the time-integral of the deviation, and a term in its derivative. Mapping that onto the process loop, with the error written as $e = \text{setpoint} - \text{measurement}$, the law is the parallel sum $u = K_p\,e + K_i\!\int\! e\,dt - K_d\,\dfrac{d(\text{measurement})}{dt}$. The proportional term is the basic restoring push. The reset (integral) term is the cure for the standing offset: it moves the valve at a *velocity* proportional to the present deviation, so the valve keeps creeping in the correcting direction as long as any deviation remains and only stops when $e = 0$ — at steady state the valve is stationary, which forces zero offset whatever the load. Its natural measure is a reset rate in per-minute units, equal to $1/T_i$ where $T_i$ is the reset time. But reset is not free: it responds to the *history* of the error, so its correction arrives late, and late corrections feed oscillation, lengthening the period and eroding the stability margin. That is what the pre-act (derivative) term buys back. Reacting to the rate of pen movement, it leads the other two — it starts correcting on velocity before the deviation itself has grown — so it anticipates and damps the loop's natural ring, which in turn lets the proportional gain and reset run more aggressively than they otherwise could. Its measure is the pre-act time $T_d$, with units of minutes, and the gains the code carries follow directly: $K_i = K_p/T_i$ and $K_d = K_p\,T_d$.

The structure is the easy half; the load-bearing contribution is choosing $K_p$, $T_i$, $T_d$ on a process I cannot model. The move is to let the process reveal its own gain and timescale and to express every constant as a fixed fraction of what I measure. The two evils were two faces of one underlying quantity — how hard I can push before instability and how fast it then rings — so I measure that boundary directly. Run the loop on proportional action alone, reset and pre-act off, and raise sensitivity until the pen holds a sustained oscillation that neither grows nor decays: that sensitivity is the ultimate gain $K_u$ and the cycle's period is $T_u$, two numbers read off a chart with no model. $K_u$ sets the gain scale, $T_u$ the time scale. I do not want to sit at $K_u$ — amplitude ratio $1$ is useless — but at the well-damped compromise everyone settles on, quarter-amplitude decay, where each overshoot is one-quarter of the last. On the roughly universal amplitude-ratio-versus-percent-of-$K_u$ curve, that target lands at very nearly half the ultimate, so proportional-only takes $K_p = 0.5\,K_u$. Adding reset erodes stability, so I pay for it by backing the proportional gain down a notch to restore the damping: $K_p = 0.45\,K_u$, with a reset rate of $1.2/T_u$, i.e. $T_i = T_u/1.2 \approx 0.83\,T_u$, an integral time a touch under one ultimate period. For the full controller, pre-act is stabilizing, so I can push the proportional gain back up past the proportional-only value to $K_p = 0.6\,K_u$, run the reset twice as fast at $2/T_u$ so $T_i = T_u/2$, and set the pre-act to $T_d = T_u/8$. The internal ratio is telling: $T_d$ is a quarter of $T_i$, the derivative anticipating the near future about as far as the integral cleans up the recent past, both keyed to the single timescale $T_u$ the loop handed me.

The closed-loop test has one real drawback: to find $K_u$ I must deliberately drive a running process to the edge of instability, which an operator will not thank me for on a touchy or slow loop. So I give a second route that never destabilizes the loop. Break it open, apply a sudden sustained step of size $\Delta F$ to the valve by hand, and record the S-shaped reaction curve the pen traces. Two features carry the information: the maximum slope at the inflection point, the reaction rate $R$ in pen-inches per minute, normalized to a unit step as $R_1 = R/\Delta F$, which measures the loop gain; and the lag $L$, the dead time found by projecting the inflection tangent back to where it meets the pen's initial level, which is pure delay and precisely what destabilizes feedback. These open-loop features predict the same closed-loop boundary, through the dictionary $$S_u \approx \frac{2}{R_1 L}, \qquad T_u \approx 4L,$$ since a dead-time-dominated process oscillates with a period set by its delay and tolerates gain inversely proportional to $R_1 L$. Substituting into the recipe already built reproduces the constants without any new tuning numbers: for full PID, $K_p = 0.6\,S_u = 1.2/(R_1 L)$, reset rate $= 2/T_u = 0.5/L$ so $T_i = 2L$, and pre-act $= T_u/8 = 0.5\,L$. Again $T_d$ is a quarter of $T_i$ — the two methods are one recipe seen from the closed-loop edge and the open-loop bump, and that consistency confirms no arithmetic slipped. In first-order-plus-dead-time terms the slope of an S-curve from a process with steady gain $K$ and dominant time constant $T$ is $R_1 \approx K/T$, so the PID proportional gain becomes $1.2\,T/(K\,L)$.

Two corrections make the law survive real hardware, and neither is about tuning. A real valve has travel limits, so a large persistent error lets the integral keep accumulating while the valve is already pinned at its stop, storing an enormous value that the controller must unwind before the valve even leaves the limit — integral windup, and a huge overshoot when the error reverses. The cure is conditional integration: compute the tentative output, clamp it to the valve limits, and commit the new integral only when the unclamped output is within limits or when the error would pull the output back out of saturation, so the integrator never winds deeper into a saturated actuator. The second correction concerns the derivative. Differentiating the error means a setpoint step injects a step into $e$, whose derivative is an impulse that spikes the valve — derivative kick, pure artifact, since the process did not move, only the setpoint did. Because the derivative's physical job is to damp the motion of the process variable, it should differentiate the measurement, not the error: at constant setpoint $\tfrac{d}{dt}(\text{setpoint}-\text{measurement}) = -\tfrac{d(\text{measurement})}{dt}$, so the damping is identical, but a setpoint change leaves the smooth measurement untouched and there is no kick. A small first-order filter with time constant a fraction of $T_d$ keeps a noisy measurement's derivative from chattering. The same machinery handles a multivariable plant like a quadrotor without ever writing its dynamics: each axis is a single loop, nested as an inner attitude loop inside an outer position loop, since horizontal position is steered through tilt exactly as a ship's heading is steered through the rudder — tune the innermost loop to its own ultimate cycle, close it, then tune outward.

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
inner = PID(*tune_ultimate_cycle(Ku_ang, Tu_ang, "PID"), dt=0.002,
            out_min=-1.0, out_max=1.0)     # motor-differential command
outer = PID(*tune_ultimate_cycle(Ku_pos, Tu_pos, "PI"), dt=0.01,
            out_min=-0.3, out_max=0.3)     # commanded tilt angle


def step(pos_sp, pos, ang):
    tilt_cmd = outer.update(pos_sp, pos)   # position error -> desired tilt
    return inner.update(tilt_cmd, ang)     # tilt error -> motor differential
```
