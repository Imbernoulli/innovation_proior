# Context: holding a process at setpoint with a model-free feedback controller

## Research question

A controller sits between a measured variable (temperature, pressure, level, flow) and a valve. Its job is to keep the variable at a desired setpoint in spite of load changes — disturbances that demand a sustained change in valve position. The precise problem has two layers. First, *what corrective law* should the controller apply: how should valve action depend on the pen's behavior (its displacement from setpoint, and the way that displacement evolves) so that the loop returns cleanly to setpoint after a disturbance, with neither a permanent residual error nor a persistent ringing? Second — and this is the part that has no good answer — *how should the adjustable constants of that law be chosen on a real installation*, where the process is a black box of unknown lags? A purely mathematical attack on a multi-lag process drowns in exponentials and trigonometric functions that the working engineer has no time to solve. What is wanted is a feedback law with a small number of physically meaningful knobs, plus a quick, repeatable field recipe that produces good settings for those knobs from the observed response of the actual process, without ever writing down its differential equation.

## Background

The state of practice is the three-effect industrial controller. Across pneumatic, hydraulic, and electric instruments, essentially all controllers combine one, two, or at most three simple effects, each of which converts pen behavior (the pen records the measured variable's deviation from setpoint) into valve behavior.

**Proportional response** gives valve movement proportional to pen movement: a 2-degree pen deflection commands twice the valve change of a 1-degree deflection. Its magnitude is the **sensitivity** (valve output change per inch of pen travel; its reciprocal is the *throttling range* or proportional band). Sensitivity ranges from zero (manual) to infinite (on–off). Two empirical facts about proportional-only control are load-bearing and well established from chart records:

- *On–off / very high sensitivity always oscillates.* It is common knowledge that infinitely high proportional response is unstable, oscillating continuously. As sensitivity is lowered there is a definite, easily found point — the **ultimate sensitivity** Su — above which any oscillation grows to some maximum amplitude, and below which an oscillation of any size decays to straight-line control. Exactly at Su the loop sustains a steady oscillation whose **amplitude ratio** (the amplitude of each wave relative to the wave before it) equals 1: each wave is the same size as the last. Below Su the amplitude ratio is < 1; above, > 1. The relationship between amplitude ratio and sensitivity-as-percent-of-Su is roughly invariant across applications, which makes Su a natural common reference point. A quarter-amplitude-decay response — amplitude ratio 0.25, each overshoot one-quarter of the last — is the usual good compromise between speed and stability, and it occurs at roughly half the ultimate sensitivity.

- *Proportional-only control leaves a steady-state offset.* A proportional controller can hold exactly one valve position when the pen is at setpoint. Any load change that requires a *different* sustained valve position can be produced only by the pen sitting away from setpoint far enough to command that valve change. This residual **offset** varies inversely with the sensitivity setting and directly with the size of the load change. Raising sensitivity to shrink the offset drives the loop toward the oscillation boundary — so offset and amplitude ratio are two evils the single proportional knob must trade off against each other.

**Automatic reset** gives valve *velocity* proportional to the pen's displacement from setpoint: as long as any deviation persists, the valve keeps moving in the correcting direction, so it drives the steady offset to zero. Its measure is the **reset rate** in units of per-minute — the number of times per minute that reset duplicates the proportional correction. Two side effects are observed: increasing reset rate speeds the offset's removal but *reduces* stability (raises the amplitude ratio) and *lengthens* the period of oscillation.

**Pre-act** gives an additional valve movement proportional to the *rate* of pen movement; it acts only with proportional response and ceases when the pen is stationary. Its measure is the **pre-act time** in minutes (output change per rate of pen movement, psi ÷ psi/min). It is anticipatory: by responding to how fast the pen is moving it leads the proportional action. Observed effects are the mirror image of reset — pre-act *increases* stability and *shortens* the period, and it lets larger settings of the other two effects be used; but above an optimum pre-act setting stability falls again.

The conceptual ancestor of this three-term structure is the automatic ship-steering analysis of **Minorsky (1922)**, "Directional Stability of Automatically Steered Bodies." Designing an autopilot for the USS New Mexico, Minorsky observed a skilled helmsman steering not only on the present course error but on how long the ship had been off course and on how fast the heading was changing. He wrote the ship's rotational equation of motion — moment of inertia, frictional resistance to turning, and the moment from the rudder — and, solving for the rudder angle, obtained three terms proportional respectively to the heading deviation, to the time-integral of the deviation, and to its derivative. The proportional term turns the rudder toward the desired course; the integral term works off a persistent bias such as a steady crosswind that a proportional helm would leave as a standing offset; the derivative term anticipates and damps the heading's natural hunting oscillation. The control constants were fixed by trial and error and the autopilot performed well in sea trials. This is the load-bearing precedent: it establishes *that* the three terms — present error, accumulated error, rate of error — are the right ingredients and *why* each is needed, but it leaves the constants to be found by trial, with no systematic procedure for setting them on an arbitrary process whose dynamics are unknown.

A second background fact, also from Minorsky's line and from the on–off observation above, is that **stability is not automatic**: raising the gain of a feedback loop eventually destabilizes it, and adding the integral term — desirable for killing offset — itself erodes stability margin. So any recipe for the constants must respect a stability boundary, not merely chase zero error.

## Baselines

**Proportional-only control (the dominant practice).** Core idea: valve change ∝ pen deviation, one knob (sensitivity). Behavior: a damped oscillation settling to straight-line control, with a residual offset after any load change. Gap: cannot remove offset (only one valve position is held at setpoint), and shrinking offset by raising sensitivity pushes the loop toward sustained oscillation at the ultimate sensitivity. A single knob cannot satisfy "no offset" and "well damped" at once.

**Minorsky's three-term steering law (1922).** Core idea: rudder angle = a·(deviation) + b·∫(deviation)dt + c·d(deviation)/dt, derived from the ship's equation of motion by recognizing that a good helmsman uses present, accumulated, and rate-of-change information. Math/behavior: the integral term annihilates a constant disturbance (the standing offset), the derivative term adds anticipatory damping to the hunting mode, and the proportional term sets the basic restoring action. Gap: the three constants a, b, c were obtained by trial and error on one specific vessel; there is no general, model-free procedure to choose them for an arbitrary process, and computing them from the plant's dynamics requires a tractable equation of motion that an industrial process with a chain of unknown lags does not provide.

**Direct mathematical (transfer-function) controller design.** Core idea: write the process's differential equation or transfer function, solve for the controller that gives a desired loop response. Gap: a real process is a series of lags whose individual values are unknown and whose combination is a forbidding assortment of exponential and trigonometric terms; the working engineer cannot afford to derive and solve it per installation. There is no quick field method here — only a lengthy analysis that, in practice, is not carried out.

## Evaluation settings

The natural testbeds are single-loop process-control applications — temperature, pressure, liquid-level, and flow loops — each a process of unknown internal lags driven by a valve and observed through a recording pen. Two kinds of experiment are available on any such loop without knowing the plant model. *Closed-loop*: with the loop running on proportional action alone, raise the sensitivity until the pen sustains a steady oscillation, and read off the ultimate sensitivity Su and the period Pu of that oscillation. *Open-loop*: break the loop, apply a sudden sustained step of size ΔF to the valve, and record the S-shaped "reaction curve" the pen traces; from it read the maximum slope (the reaction rate R, in pen-inches per minute, normalized to a unit valve step as R₁ = R/ΔF) and the lag L (the time at which the tangent at the inflection point, projected back, meets the pen's initial level). The yardsticks for judging a setting are all read off the same chart records: the amplitude ratio (decay of successive overshoots, with quarter-amplitude decay the target), the period of oscillation, the maximum deviation after a load change, the time to settle, and the residual offset. These experiments and metrics — the ultimate-cycle test, the reaction-curve step test, and the amplitude-ratio / period / offset readings — exist independently of any particular rule for setting the constants. A canonical multivariable target for the same feedback idea is a quadrotor: each axis (roll, pitch, yaw, altitude, and the horizontal positions steered through tilt) is a loop to be regulated, naturally arranged as an inner attitude loop nested inside an outer position loop, exercising several single-loop controllers at once.

## Code framework

The primitives that already exist: a clock with a fixed sample interval Δt, a way to read the measured variable and write a valve command, the ability to compute an error from a setpoint, an actuator with known travel limits, and a simulator that can integrate a process model forward in time so that step responses and sustained oscillations can be reproduced. What is missing is the corrective law itself — how the command depends on the error and its history — and the procedure that turns an observed response into the law's constants.

```python
from math import inf


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class Controller:
    """A single-loop feedback controller: each tick it reads the setpoint and
    the measured variable and returns a bounded valve command. The corrective
    law tying command to the error and its history is the slot to be filled."""

    def __init__(self, dt, out_min=-inf, out_max=+inf):
        self.dt = dt
        self.out_min, self.out_max = out_min, out_max
        # TODO: whatever internal memory the corrective law needs

    def update(self, setpoint, meas):
        # TODO: form the error and produce a bounded command from it and its
        # history; the structure of this law is the whole question.
        pass


def tune_from_sustained_oscillation(...):
    """Turn an observed closed-loop sustained oscillation into the controller's
    constants. The mapping from the observed oscillation to the constants is the
    recipe to be derived."""
    # TODO
    pass


def tune_from_step_response(...):
    """Turn an observed open-loop step (reaction-curve) response into the
    controller's constants. The mapping is the recipe to be derived."""
    # TODO
    pass
```
