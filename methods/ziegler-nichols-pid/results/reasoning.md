Let me start from the thing that actually goes wrong on the floor, because the corrective law should be built to fix it, not chosen because it's elegant. I have a process — a heater, a tank, a header — and a valve, and a pen that records how far my measured variable sits from where I want it. I get to decide how the valve moves in response to that pen. The simplest honest choice, and the one nearly every controller already uses, is to make the valve move in proportion to the pen's deviation: push the valve open in proportion to how far below setpoint I am. Call the constant of proportionality the sensitivity. One knob. And it half-works. But it has two failures I can see plainly on the chart records, and the whole design problem is really about curing both at once.

The first failure is offset. A purely proportional controller can hold exactly one valve position when the pen is sitting at setpoint — because at zero deviation the proportional term commands one specific, fixed valve output. Now suppose a load change comes through: the process now *needs* a different sustained valve position to stay at setpoint. The only way my proportional law can produce that different valve position is for the pen to sit *away* from setpoint, far enough that sensitivity × deviation equals the new valve demand. So the pen parks at a standing error. That residual offset is inversely proportional to my sensitivity setting and directly proportional to the size of the load change — it's baked into the structure of proportional-only action. The obvious reflex is to crank sensitivity way up so the offset shrinks toward nothing.

And that reflex runs straight into the second failure. As I raise sensitivity, the loop gets jumpier, and it's common knowledge that with infinite sensitivity — on/off control — the loop just oscillates forever. There's a definite, repeatable point: a particular sensitivity above which any little oscillation grows until it hits some maximum amplitude, and below which any oscillation dies back to a straight line. Right at that point the loop holds a steady oscillation that neither grows nor decays — each wave the same size as the one before it. I'll call that the ultimate sensitivity, Su, and the amplitude-ratio language makes it precise: amplitude ratio is the size of one wave divided by the size of the previous wave, and at Su the amplitude ratio is exactly 1. Below Su it's less than 1 (decaying), above it's greater than 1 (growing). So my one knob is trapped between two evils: turn it down to kill the ringing and the offset balloons; turn it up to kill the offset and the loop rings, then sustains, then runs away. A single proportional gain provably cannot give me both "no offset" and "well damped" — those pull the knob in opposite directions. I need more than one effect.

What more? I should look at how this exact problem was solved once before, on a physically different but structurally identical system: steering a ship. Minorsky, working on autopilots for the New Mexico back in 1922, didn't guess a controller — he watched a good helmsman and he wrote down the ship's equation of motion. Moment of inertia times angular acceleration, minus the frictional resistance to turning, equals the turning moment from the rudder. The question is what to make the rudder angle a function of. A helmsman, he noticed, doesn't just steer on the present heading error; he steers on how long the ship has been off course, and on how fast the heading is swinging. So Minorsky let the rudder angle be a sum of three terms: one proportional to the heading deviation, one proportional to the time-integral of the deviation, and one proportional to its derivative. And when you put that rudder law back into the equation of motion and ask what each term does, the roles are clean. The proportional term is the basic restoring push. The integral term is the cure for a *standing* disturbance — a steady crosswind would force a proportional helm to hold a permanent offset to fight it, but an integral term keeps adding rudder as long as any deviation persists, so it grinds the standing error to zero. The derivative term anticipates: it reacts to the rate the heading is changing, so it leads the proportional action and damps the ship's natural hunting oscillation before it builds.

That maps onto my process problem term for term. My offset is exactly Minorsky's standing-crosswind error: a proportional controller leaves it; an integrating action will erase it. So I want a second effect whose output is proportional not to the deviation but to the *accumulated* deviation — equivalently, an effect that moves the valve at a *velocity* proportional to the present deviation, so the valve keeps creeping in the correcting direction for as long as any deviation is left, and only stops when the deviation reaches zero. At steady state the valve has stopped moving, which means the deviation must be zero — offset gone, automatically, whatever the load. That's the reset action, and its natural measure is a reset *rate*: how many times per minute the reset action reproduces the proportional correction. A rate, per-minute, because it's a velocity-per-deviation.

But Minorsky's integral term, and every honest account of feedback, warns me that this isn't free. Integrating action erodes stability. Intuitively it must: the integral lags the error — it responds to the *history* of the deviation, so its correction arrives late, and late corrections are exactly what feed an oscillation rather than damp it. So I expect that switching on reset will make the loop ring more for the same proportional setting, and lengthen the period of the oscillation (the extra lag slows the whole loop down). I should hold that expectation — it's going to force me to back off the proportional setting when I add reset.

And the third effect, the derivative — pre-act, the anticipatory one. The trouble with proportional-plus-reset is that both effects react to where the pen *is* (now or accumulated); neither reacts to where it's *going*. If the pen is screaming toward setpoint, a proportional controller keeps pushing hard until the instant it arrives, and then overshoots. What I want is to ease off — or even push back — when the pen is moving fast, in proportion to that speed. So a third term proportional to the rate of pen movement, which by construction leads the other two: it starts correcting based on velocity before the deviation itself has grown. That's the derivative, Minorsky's hunting-damper. Its effect should be the mirror image of reset's: it *adds* damping, *shortens* the period, and — because it's stabilizing — it should let me run higher sensitivity and faster reset than I otherwise could, getting back the aggressiveness that reset cost me. Its measure is a *time*: output change per rate-of-pen-movement is (valve units)/(valve units per minute) = minutes. The pre-act time is, loosely, how far into the future the velocity term lets the controller "see."

So the corrective law I've reasoned my way to is the three-term sum: valve output equals a proportional term in the deviation, plus a reset term in the integral of the deviation, plus a pre-act term in the derivative of the deviation. Writing the deviation (error) as e = setpoint − measurement, and using Kp for sensitivity, this is u = Kp·e + (reset)·∫e dt + (pre-act)·de/dt. Good. That's the *structure*. But the structure is the easy half, and Minorsky's autopilot is a warning here: he had to set his three constants by trial and error on the actual ship. On a process I don't even have his luxury of a clean equation of motion — a real process is a chain of lags whose individual values I don't know and whose combination is a forbidding mess of exponentials. I am not going to solve that differential equation per installation. So the real problem, the one with no good answer yet, is: *how do I pick Kp, the reset rate, and the pre-act time on a black-box process, in the field, quickly?*

Here's the move. I can't model the process, but I can *interrogate* it and let it tell me its own timescale and gain, and then express the three constants in terms of whatever I measure. The two evils — offset and oscillation — were two faces of one underlying quantity: how hard I can push before the loop goes unstable, and how fast it rings when it does. If I could measure *that boundary*, I'd have a natural reference for every setting. And I can measure it directly. Run the loop on proportional action only — reset off, pre-act off — and slowly turn the sensitivity up until the pen holds a steady, sustained oscillation that neither grows nor decays. That sensitivity is, by definition, the ultimate sensitivity Su, and the period of that oscillation, call it Pu, is the natural period of the loop at the stability boundary. Two numbers, read off a chart, no model required. Su tells me the gain scale of the loop; Pu tells me its time scale. Everything else should be expressed as a fraction of these.

Now, what fraction? Start with proportional alone. I don't want to sit *at* Su — that's the sustained-oscillation boundary, amplitude ratio 1, useless. I want a comfortably damped response. The amplitude-ratio-versus-sensitivity relationship, expressed as percent of Su, is roughly the same shape on every application — which is the whole reason Su is a good common reference. Reading that relationship, the well-damped compromise everyone settles on is quarter-amplitude decay: each overshoot one-quarter the size of the last, amplitude ratio 0.25. That's slow enough to look stable, fast enough not to crawl. And on that universal curve, amplitude ratio 0.25 lands at very nearly half of Su. So for a proportional controller: set the sensitivity to half the ultimate. Kp = 0.5 Su. Clean, model-free, and it's just "find the oscillation point and cut it in half."

Now add reset. I argued reset erodes stability — so if I keep sensitivity at 0.5 Su and switch reset on, the amplitude ratio climbs back above 0.25 and I've lost my compromise. So I have to pay for reset by backing the proportional setting down a notch, to restore the damping. The amount is small — dropping from 0.5 Su to 0.45 Su is enough to absorb reset's destabilizing tendency and get the amplitude ratio back to about a quarter. So Kp = 0.45 Su for proportional-plus-reset. And the reset rate itself: too slow and the offset crawls away over many periods; too fast and the extra lag dominates, the loop rings and the period stretches out badly. The right setting must scale with the loop's own timescale — a faster loop (small Pu) can tolerate, and needs, faster reset. So reset rate ∝ 1/Pu. Pinning the constant from the response that gives quarter-amplitude recovery without the period blowing up: reset rate = 1.2/Pu per minute. Equivalently, since reset rate is 1/Ti where Ti is the reset (integral) time, Ti = Pu/1.2 ≈ 0.83 Pu — the integral time is a bit under one ultimate period. That feels right: the integral should act over roughly the loop's natural period, not much faster (it'd destabilize) or much slower (offset would linger).

Now the full three-term controller, adding pre-act. Pre-act is stabilizing — it's the mirror of reset — so now I can afford to push the proportional setting back *up*, past where I had it, and even past the proportional-only value, because the added damping makes the higher gain safe. So Kp climbs to 0.6 Su for the full PID — the derivative's damping buys back the aggressiveness that reset cost, and then some. With more damping in hand I can also run the reset faster, working the offset off more quickly: reset rate = 2/Pu per minute, i.e. Ti = Pu/2 = 0.5 Pu — the integral time drops to half an ultimate period, twice as fast as the reset-only case, exactly because pre-act is holding the stability margin. And the pre-act time itself: it's a derivative time, a fraction of the loop's natural period — enough to anticipate the oscillation but not so much that the derivative starts amplifying noise and, past its own optimum, *reduces* stability again. Set it to one-eighth of the period: pre-act time Td = Pu/8 = 0.125 Pu. Notice the internal consistency of the recipe: pre-act (Pu/8) is a quarter of the reset time (Pu/2). The derivative looks ahead about a quarter as far as the integral looks back — derivative anticipating the near future, integral cleaning up the recent past, both keyed to the one timescale Pu the loop handed me.

So the closed-loop recipe, end to end: drive the proportional-only loop to sustained oscillation, read Su and Pu, then —
P: Kp = 0.5 Su.
PI: Kp = 0.45 Su, reset rate = 1.2/Pu.
PID: Kp = 0.6 Su, reset rate = 2/Pu, pre-act time = Pu/8.
Every number is a fraction of two things I can read straight off a chart. That's the field method I wanted.

But there's a real objection to this method, and I should take it seriously rather than wave it away. To find Su I have to deliberately push the *running* process to the edge of instability — drive it into sustained oscillation. On a touchy or expensive process, an operator won't thank me for cycling his column on purpose, and on a slow loop it can take a long time to coax out a clean sustained oscillation. I'd like a second route to the same constants that never destabilizes the loop. So instead of probing the closed loop at its stability boundary, let me probe the *open* loop with a single bump and infer the loop's gain and timescale from the transient.

Break the loop — disconnect the controller from the valve — and apply a sudden, sustained step of size ΔF to the valve, by hand. The pen will trace a curve. For a process that's a chain of lags, that curve is S-shaped: nothing much happens at first (the lags swallow the step), then the pen accelerates to some maximum rate, then it eases off as the process approaches its new equilibrium. This reaction curve *is* the process's fingerprint. Two features of it carry the information I need. The maximum slope — the steepest rate the pen reaches, at the inflection point — call it the reaction rate R, in pen-inches per minute; it measures how *fast and how strongly* the process responds, i.e. the loop gain. And if I draw the tangent at that inflection point and project it backward to where it crosses the pen's *initial* level, the time from the step to that crossing is an effective dead time — call it the lag L. L measures how long the process dithers before it really gets going; it's the part of the dynamics that's pure delay as far as the controller is concerned, and delay is precisely what destabilizes feedback. Since R scales with the step size, I normalize: the unit reaction rate R₁ = R/ΔF is the slope per unit valve step.

Now I need to connect (R₁, L) back to settings. The bridge is that these open-loop features predict the very same closed-loop boundary I measured before. A process dominated by dead time L is exactly one that, in closed loop, will oscillate, and its period at the stability boundary should scale with L — the delay sets the timescale of the ring. Working out what proportional gain just sustains oscillation for a process whose response rises at slope R₁ after a dead time L: the loop goes unstable when the gain is large enough that, over one effective delay, the correction overshoots and reinforces. That ultimate sensitivity comes out inversely proportional to the product R₁L — fast-responding (large R₁) or sluggish-to-start (large L) processes tolerate less gain — with Su ≈ 2/(R₁L). And the period at that boundary scales with the delay: Pu ≈ 4L. Those two relations are the dictionary between the open-loop fingerprint and the closed-loop quantities.

With that dictionary I don't need to derive new tuning numbers — I just substitute Su = 2/(R₁L) and Pu = 4L into the recipe I already built. Proportional: Kp = 0.5 Su = 0.5·2/(R₁L) = 1/(R₁L). For PI: Kp = 0.45 Su = 0.45·2/(R₁L) = 0.9/(R₁L), and reset rate = 1.2/Pu = 1.2/(4L) = 0.3/L. For full PID: Kp = 0.6 Su = 0.6·2/(R₁L) = 1.2/(R₁L), reset rate = 2/Pu = 2/(4L) = 0.5/L, and pre-act time = Pu/8 = 4L/8 = 0.5 L. Let me read those back in plainer terms. The proportional gain is 1.2/(R₁L) — inversely proportional to both how fast the process responds and how long it delays, which is exactly the intuition: a snappy or laggy process must be driven gently. The reset rate 0.5/L means the integral time Ti = 1/reset = 2L — integrate over twice the dead time. And the pre-act time is half the dead time, Td = 0.5L. So Ti = 2L and Td = 0.5L means the derivative time is a quarter of the integral time again — the same internal ratio as the closed-loop recipe, which it has to be, since one was derived from the other through Pu = 4L. That consistency is a good sign I haven't made an arithmetic slip: the two methods are the same recipe seen from two sides, the closed-loop edge and the open-loop bump.

In modern transfer-function language the reaction-curve numbers are even more transparent. The slope R₁ of an S-curve from a process with steady gain K and dominant time constant T is R₁ ≈ K/T (it climbs toward a total change of K with characteristic rate set by T). So 1/(R₁L) = T/(K L), and the PID proportional gain 1.2/(R₁L) becomes 1.2·T/(K L), with Ti = 2L and Td = 0.5L — the familiar first-order-plus-dead-time form, gain scaling like (time constant)/(gain × delay).

Now I have the two recipes, but a controller that just computes Kp·e + Ki·∫e + Kd·de/dt and writes it to the valve will misbehave in two ways that have nothing to do with tuning and everything to do with the fact that real valves and real setpoints aren't ideal. I have to handle both before this is usable.

First problem: the valve has limits. It can't open past fully-open. Suppose a big load change drives a large persistent error; the integral term keeps accumulating — that's its job — but the valve is already pinned at its maximum, so all that extra accumulated integral does *nothing* but build up a huge stored value. When the process finally catches up and the error reverses, the controller has to *unwind* that enormous integral before the valve even leaves the stop, so the loop sails right past setpoint and overshoots badly. Integral windup. The cure is to stop the integral from accumulating into a saturated actuator: when the computed output is past the valve's limit, don't commit the new integral term — only integrate when doing so won't push further into saturation. Conditional integration. So I'll compute a tentative output, clamp it to the valve limits, and only keep the integral update if the unclamped output is within limits, or if the error would drive it back out of saturation. That keeps the integrator honest.

Second problem: the derivative on a setpoint step. If I take the derivative of the *error* e = setpoint − measurement, and an operator slams the setpoint to a new value, the error has a step in it, and the derivative of a step is an impulse — the pre-act term spikes the valve violently for one instant. Derivative kick, and it's pure artifact: the *process* didn't suddenly move, the setpoint did. The fix follows from what the derivative term is physically for — it damps the motion of the process variable, the hunting. So it should differentiate the measurement, not the error. At constant setpoint, d/dt(setpoint − measurement) = −d(measurement)/dt anyway, so the damping behavior is identical; but on a setpoint change the measurement is smooth, so there's no kick. Derivative on measurement: the pre-act term becomes −Kd·d(measurement)/dt. And since differentiating a noisy measurement amplifies the noise, I'll put a small first-order filter on that derivative — a time constant a fraction of Td — so the pre-act term doesn't chatter.

Let me convert the reset/pre-act measures into the gains the code carries. The reset rate is per-minute and equals 1/Ti, so the integral gain Ki = Kp·(reset rate) = Kp/Ti. The pre-act time *is* Td, so the derivative gain Kd = Kp·Td. With those, the law is u = Kp·e + Ki·∫e dt − Kd·d(measurement)/dt, with the integral conditionally clamped to the valve limits. That's the whole controller.

```python
from math import inf


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class PID:
    """Single-loop PID in parallel form. e = setpoint - measurement.
        u = Kp*e + Ki*∫e dt - Kd*d(measurement)/dt
    with conditional-integration anti-windup and derivative-on-measurement."""

    def __init__(self, Kp, Ki, Kd, dt, out_min=-inf, out_max=+inf, tau=0.0):
        self.Kp, self.Ki, self.Kd, self.dt = Kp, Ki, Kd, dt
        self.out_min, self.out_max, self.tau = out_min, out_max, tau
        self.integ = 0.0            # accumulated ∫e dt scaled into output units
        self.prev_meas = None
        self.deriv = 0.0            # filtered derivative of the measurement

    def update(self, setpoint, meas):
        e = setpoint - meas
        P = self.Kp * e                                   # present error
        if self.prev_meas is None:
            self.prev_meas = meas
        # derivative on MEASUREMENT, not error -> no kick when setpoint steps
        d_meas = (meas - self.prev_meas) / self.dt
        if self.tau > 0.0:                                # 1st-order noise filter
            a = self.dt / (self.tau + self.dt)
            self.deriv = (1.0 - a) * self.deriv + a * d_meas
        else:
            self.deriv = d_meas
        D = -self.Kd * self.deriv                         # anticipatory damping

        integ_try = self.integ + self.Ki * e * self.dt    # tentative reset
        u_raw = P + integ_try + D
        u = clamp(u_raw, self.out_min, self.out_max)
        # anti-windup: commit the integral only if we're not winding deeper into
        # a saturated valve (unsaturated, or the error would pull us back in).
        if u_raw == u or (e > 0.0) != (u_raw > u):
            self.integ = integ_try
        self.prev_meas = meas
        return u
```

And the two tuning recipes, each turning an observed response into (Kp, Ki, Kd). The reset rate is 1/Ti so Ki = Kp/Ti; the pre-act time is Td so Kd = Kp·Td.

```python
def tune_ultimate_cycle(Ku, Tu, kind="PID"):
    """From a sustained proportional-only oscillation: ultimate sensitivity Ku
    and its period Tu. Settings as fractions of those two read-off-the-chart
    numbers, set for ~quarter-amplitude decay."""
    if kind == "P":
        Kp, Ti, Td = 0.5 * Ku, inf, 0.0
    elif kind == "PI":
        Kp, Ti, Td = 0.45 * Ku, Tu / 1.2, 0.0      # reset rate 1.2/Tu -> Ti=Tu/1.2
    elif kind == "PID":
        Kp, Ti, Td = 0.6 * Ku, Tu / 2.0, Tu / 8.0  # reset 2/Tu, pre-act Tu/8
    else:
        raise ValueError(kind)
    Ki = Kp / Ti if Ti != inf else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd


def tune_reaction_curve(R1, L, kind="PID"):
    """From an open-loop step: unit reaction rate R1 (max slope per unit valve
    step) and lag L (dead time). Same recipe via Su=2/(R1 L), Pu=4L."""
    if kind == "P":
        Kp, Ti, Td = 1.0 / (R1 * L), inf, 0.0
    elif kind == "PI":
        Kp, Ti, Td = 0.9 / (R1 * L), L / 0.3, 0.0   # reset 0.3/L -> Ti=L/0.3
    elif kind == "PID":
        Kp, Ti, Td = 1.2 / (R1 * L), 2.0 * L, 0.5 * L  # reset 0.5/L, pre-act L/2
    else:
        raise ValueError(kind)
    Ki = Kp / Ti if Ti != inf else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd
```

For the quadrotor this is the natural fit precisely because I never wrote down its dynamics: each axis is its own single loop, and I nest them — an inner attitude (angle-rate, then angle) loop tuned first, and an outer position loop that hands the inner loop an angle setpoint, since horizontal position is steered through tilt exactly the way a ship's heading is steered through the rudder. I tune the innermost loop to its own ultimate cycle (raise its proportional gain until the rate oscillates, read Ku and Tu, apply 0.6/0.5/0.125), then close it and tune the loop outside it, working outward.

```python
# cascade: outer position loop feeds an attitude setpoint to the inner rate loop
inner = PID(*tune_ultimate_cycle(Ku_rate, Tu_rate, "PID"), dt=0.002,
            out_min=-1.0, out_max=1.0)     # motor-differential command, clamped
outer = PID(*tune_ultimate_cycle(Ku_pos, Tu_pos, "PI"), dt=0.01,
            out_min=-0.3, out_max=0.3)     # commanded tilt angle, clamped

def step(pos_sp, pos, ang, ang_rate):
    tilt_cmd = outer.update(pos_sp, pos)        # outer: position -> desired tilt
    return inner.update(tilt_cmd, ang)          # inner: tilt -> motor differential
```

The whole chain, in one breath: proportional action alone leaves an offset and, pushed for accuracy, rings — two evils one knob can't reconcile — so I add a reset term that moves the valve at a velocity set by the deviation to grind the offset to zero, and a pre-act term keyed to the pen's velocity to anticipate and damp the ring and buy back the gain reset cost; then, since I can't model the process, I let it tell me its own gain and timescale — either as the ultimate sensitivity Su and period Pu where the proportional loop just sustains oscillation, or as the slope R₁ and dead time L of its open-loop reaction curve — and I set every constant as a fixed fraction of those (0.6 Su, reset 2/Pu, pre-act Pu/8; equivalently 1.2/(R₁L), reset 0.5/L, pre-act L/2) chosen to land on quarter-amplitude decay; and I make it survive reality with conditional-integration anti-windup and derivative-on-measurement.
