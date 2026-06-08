# Context: the electrodynamics of moving bodies at the turn of the century

## Research question

By the turn of the century two enormously successful theories sit side by side and quietly contradict each other. Newtonian mechanics, with its Galilean principle that the laws of motion are identical in every uniformly-moving frame and that velocities simply add, governs everything that moves. Maxwell's electrodynamics governs light and the electromagnetic field, and from its own equations it fixes a definite propagation speed `c = 1/√(ε₀μ₀)`. The trouble is the word *definite*: a speed is a speed *with respect to something*, and Maxwell's equations, read in the usual way, do not say with respect to which frame. The standard answer — an all-pervading "luminiferous ether" that is the rest frame of light — should make the speed of light depend on one's motion through it, exactly the way the speed of sound depends on the wind. Yet every attempt to detect that dependence has failed.

The precise problem: **find a single consistent kinematics in which Maxwell's electrodynamics holds in every inertial frame, the speed of light comes out the same in every inertial frame, and the unobservability of the ether is explained rather than patched.** A solution must reproduce all the first-order optical results, explain the second-order null result, and remove the spurious asymmetries that the moving-body electrodynamics of the day generates. It must do this without privileging any frame, since no experiment has ever revealed one.

## Background

**Maxwell's field theory.** Between 1861 and 1873 Maxwell unified electricity, magnetism and light into one field theory. In empty space his equations make the electric and magnetic fields obey a wave equation whose propagation speed is the constant `c` built from `ε₀` and `μ₀`. Light *is* this wave. The field, not action at a distance, is the primary object. The theory's success is total — Hertz's 1887 production and detection of electromagnetic waves confirmed it spectacularly — and that very success makes its silence about the frame of `c` impossible to ignore.

**The magnet-and-conductor asymmetry.** Maxwell's electrodynamics, applied to moving bodies, draws a sharp distinction between two cases that are physically identical. Move a magnet past a stationary conductor: a changing magnetic field creates an electric field, which drives a current. Move the conductor past a stationary magnet: there is no electric field near the magnet, but the conductor's charges, moving through the field, feel a magnetic "electromotive force" that drives — the *same* current, of the same path and intensity. The observable effect depends only on the relative motion, yet the theory tells two different stories with two different intermediate causes (an `E`-field versus an EMF). The asymmetry is in the description, not in the phenomena. This is a clue that the framework over-describes the world — that it carries a notion of absolute rest the phenomena don't.

**Galilean relativity and the addition of velocities.** Classical kinematics says: pick any inertial frame; the laws of mechanics are the same; coordinates transform as `x' = x − vt`, `y' = y`, `z' = z`, and crucially `t' = t` — time is absolute, shared by all frames. Velocities add: a thing moving at `u` in one frame moves at `u − v` in a frame moving at `v`. Applied to light, this says light's measured speed must change from frame to frame. So a frame-independent `c` and Galilean kinematics cannot both be right.

**The ether and the ether-drift experiments.** If light is a wave it needs a medium, the ether, at rest in some privileged frame. The Earth, orbiting the Sun at about 30 km/s, must move through it, so a sufficiently sensitive optical experiment should reveal an "ether wind." Attempts to find one — first-order-in-`v/c` interference experiments, and the decisive second-order experiment of Michelson and Morley (1881, refined 1887) — return null. Michelson and Morley split a light beam down two perpendicular arms and recombine it; rotating the apparatus should shift the interference fringes by a few tenths of a fringe as the arms exchange their orientation relative to the supposed wind. The predicted shift does not appear. The result is a persistent, carefully-established anomaly: either Galilean kinematics is not universally valid for optics, or the ether interpretation of Maxwell's equations must be modified. Both pillars had been confirmed in their own domains, so abandoning either is painful.

**The riding-a-light-beam puzzle.** A purely conceptual probe sharpens the same tension. Imagine chasing a light beam at speed `c`. By Galilean addition you should overtake it enough to see a *static*, spatially-periodic electromagnetic field standing still in space. But no such frozen light is known, in experience or in Maxwell's equations — Maxwell's equations have no static spatially-oscillating field solution of that kind in vacuum. So either the equations fail for the fast observer, or the Galilean picture of "catching up to light" is wrong. If the relativity principle is to survive, the fast observer's laws must be the same as the rest observer's — otherwise he could detect his own uniform motion by the fact that light misbehaves for him, which uniform motion is never supposed to let you do.

## Baselines

**Galilean / Newtonian kinematics (the default).** `x' = x − vt`, `t' = t`. Universal absolute time; velocities add linearly. *Gap:* makes `c` frame-dependent; cannot accommodate a measured speed of light that is the same in every frame, which the ether-drift nulls increasingly demand.

**Lorentz's electron theory and the theorem of corresponding states (1895).** Lorentz keeps the stationary ether and Maxwell's equations in the ether frame, then asks how the equations look for a system moving uniformly through the ether at `v`. To first order in `v/c` he proves the *theorem of corresponding states*: introduce shifted space `x' = x − vt`, `y'=y`, `z'=z`, and a new time variable
```
t' = t − (v/c²) x        ("local time")
```
and the source-free Maxwell equations in the moving system, written in the primed variables, take the same form as in a system at rest. Hence a moving observer using local time makes the same optical observations as a rest observer — which explains why first-order ether-drift experiments come out null. *Gap:* `t'` is an **auxiliary mathematical time**; Lorentz retains a "true" ether time `t` and a privileged frame. Local time is a calculational device, not the actual time of the moving frame.

**The FitzGerald–Lorentz contraction hypothesis (1889–1895).** The first-order theorem does not cover the second-order Michelson–Morley null. To kill that, Lorentz (independently of FitzGerald) assumes that bodies moving through the ether physically *contract* along the direction of motion by the factor `√(1 − v²/c²)`, shortening the relevant interferometer arm just enough to cancel the expected shift. *Gap:* this is a separate **dynamical hypothesis** about how the electromagnetic forces binding matter respond to motion through the ether — bolted on specifically to save the phenomena, with no kinematic necessity.

**Lorentz's 1904 transformation.** Pushing beyond first order, Lorentz writes the full transformation, in modern notation
```
x' = l(v) γ (x − vt),   y' = l(v) y,   z' = l(v) z,   t' = l(v) γ (t − vx/c²),
γ = 1/√(1 − v²/c²),
```
with `γ` now structurally central and an undetermined overall scale `l(v)`. Moving electromagnetic oscillators are shown to have retarded periods, giving local time a physical foothold. *Gap:* the ether frame and the true-time/local-time distinction survive; contraction is still dynamical; the scale `l(v)` is left open; and the result is not promoted to a universal kinematics governing *all* clocks and rods regardless of their constitution.

**Poincaré's reformulation (1900–1905).** Poincaré presses the *principle of relativity* as a general postulate, criticizes the conceptual standing of an immobile ether, reads Lorentz's local time as the time a moving observer would obtain by synchronizing clocks with light signals, and fixes the scale `l(v) = 1` by requiring the transformations to form a group. *Gap:* the synchronization reading and the relativity principle are present, but the decisive step — making the *relativity of simultaneity itself* the operational definition of time and deriving everything from two postulates without an ether — is not consummated.

## Evaluation settings

The natural yardsticks are the existing experiments and the existing theoretical demands, all of which predate any new theory:

- **First-order ether-drift experiments** (optical interference, aberration, the Fizeau moving-water drag experiment giving the dragging coefficient `1 − 1/n²`) — any acceptable kinematics must reproduce their (mostly null, or first-order) results.
- **The Michelson–Morley second-order interferometer** (1887): perpendicular-arm interferometer, fringe-shift on rotation as the metric of an ether wind; the benchmark a successful theory must explain by predicting *no* shift.
- **Stellar aberration and the Doppler effect** of light from moving sources/observers — classical formulas exist; a new kinematics must contain them as limits and correct them at order `v²/c²`.
- **Maxwell's equations themselves** as a consistency target: the transformation laws for `E` and `B` must leave the form of the source-free and convection-current Maxwell equations invariant between frames.
- **Moving-electron dynamics**: the velocity-dependence of an electron's inertia (the "longitudinal" and "transverse" mass measured in deflection experiments with crossed electric and magnetic fields) — a candidate kinematics implies definite predictions here.

## Code framework

The object is analytic — a coordinate transformation and its consequences — so the scaffold is just enough numerics to check any proposed transformation. It starts with two inertial frames in relative motion along a shared axis, the operational fact that clocks are set by light signals, and the empty slot where the law connecting the two frames' space-time coordinates will go.

```python
import numpy as np

C = 1.0  # work in units where the speed of light is 1

class InertialFrame:
    """A frame K' moving at speed v along the shared x-axis of frame K."""
    def __init__(self, v):
        assert abs(v) < C
        self.v = v

def galilean(x, t, v):
    """The default kinematics: absolute time, linear velocity addition."""
    xp = x - v * t
    tp = t                      # time is the same in every frame (to be questioned)
    return xp, tp

def synchronize_clock(t_emit, t_return):
    """Operational time-setting by a light signal sent to a remote clock and reflected back.
    Definition of synchrony: the remote event is dated at the MIDPOINT of emission and return."""
    # TODO: this midpoint rule is the only handle we have on 'simultaneous at a distance';
    # whether it yields the same simultaneity in two frames is exactly the open question.
    return 0.5 * (t_emit + t_return)

def coordinate_transform(x, t, v):
    """The law connecting (x, t) in K to (x', t') in a frame moving at v.
    UNKNOWN. Must be consistent with: (P1) the laws are identical in K and K',
    and (P2) a light signal has speed c in BOTH frames."""
    # TODO: derive this from the two postulates + the synchronization definition above.
    pass

def length_of_moving_object(rest_length, v):
    """Length of a moving rigid body as measured in the frame it moves through."""
    # TODO: read off from coordinate_transform once it exists.
    pass

def rate_of_moving_clock(v):
    """Rate of a moving clock relative to the frame it moves through."""
    # TODO: read off from coordinate_transform once it exists.
    pass

def add_velocities(w, v):
    """Composition of a velocity w (in K') with the frame velocity v, as seen in K."""
    # TODO: derive from coordinate_transform; must never exceed c for sub-c inputs.
    pass
```
