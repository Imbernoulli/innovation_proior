# Context: generating tunable quadruped gaits

## Research question

How does one make a four-legged robot walk, trot, and bound — and switch fluidly between those
gaits and across speeds — without hand-authoring and re-tuning a separate set of joint
trajectories for every gait at every speed?

A quadruped gait is a remarkably constrained object. All four legs trace essentially the same
periodic stride (a swing phase, where the foot lifts and swings forward, alternating with a
stance phase, where the foot is planted and pushes the body forward), but each leg is held at a
fixed *phase offset* relative to the others. The gait is nothing but that pattern of relative
phases. Different gaits are different offset patterns, and different speeds call for different
gaits: at low speed an animal uses a statically stable walk; at high speed a dynamically stable
trot or bound. A controller therefore has to (i) produce a clean periodic per-leg motion whose
amplitude and frequency can be dialed smoothly, (ii) lock the four legs into a chosen relative
phasing, (iii) let that phasing be changed to retrieve a different gait, and (iv) do all this
robustly, so that a perturbation, a parameter change, or rough terrain does not destroy the
coordination. A solution that meets this bar would reduce a high-dimensional, twelve-or-more
joint trajectory problem to a handful of interpretable parameters.

## Background

**Quadruped gaits and their phase relationships.** Label the legs front-right, front-left,
rear-right, rear-left. Each leg's motion over one stride can be assigned a phase in [0,1).
Measured gaits sit at characteristic inter-leg phase offsets. In a **trot** the diagonal pairs
(FR+RL, FL+RR) move together and the two diagonals are half a stride apart. In a **pace** the
ipsilateral pairs (both left, both right) move together, half a stride apart. In a **bound** the
front pair moves together, the hind pair moves together, the two pairs half a stride apart. A
**walk** is a four-beat gait in which the legs follow one another roughly a quarter stride apart.
A **gallop** is an asymmetric fast gait with front and hind pairs nearly in phase but slightly
broken. The **duty factor** — the fraction of the stride a foot spends on the ground — is above
0.5 for walking gaits (so that support is continuous) and drops below 0.5 for running gaits;
slow animals run high duty factors (≈0.8) with phases near 40–50%, typical mammals run duty
factors ≈0.6–0.75 with phases near 25%, and the duty factor that minimizes mechanical work
shifts the favored footfall phasing. There is, in mammals, a roughly linear relation between the
*inverse of stance duration* and locomotion speed, while swing duration stays nearly constant
across speeds — speed is regulated mainly by shortening stance, not swing.

**Biological central pattern generators.** In vertebrates, the spinal cord contains distributed
neural circuits that produce coordinated rhythmic motor output from simple tonic (non-rhythmic)
descending drive, and can do so even when isolated from the brain and from sensory feedback.
The canonical building block is the *half-centre*: two neuron populations that cannot oscillate
alone but, coupled by mutual inhibition, settle into anti-phase oscillation — exactly the
alternation a flexor/extensor joint needs. Sensory feedback (notably foot-contact and load
sensing) is known to strongly modulate the timing of the swing→stance and stance→swing
transitions in animals; load under the foot can delay lift-off, and ground contact can trigger
the switch to stance. This biological picture suggests that a locomotion controller need not
explicitly command every joint angle: a low-dimensional rhythmic source, modulated by a few
parameters and coupled to the body through feedback, can produce the full high-dimensional
coordinated movement.

**Limit cycles and phase oscillators.** A stable limit cycle is an isolated closed trajectory of
a dynamical system that nearby trajectories converge to; transient perturbations are forgotten as
the state returns to the cycle. The Andronov–Hopf normal form is the canonical generator of one:
in polar coordinates ṙ = α(μ − r²)r, θ̇ = ω, the radius converges to r = √μ and the phase
advances uniformly at ω. The amplitude (μ) and frequency (ω) are explicit, separable parameters,
and the cycle is harmonic and structurally stable — its shape does not change as ω is varied.
Other oscillators (Van der Pol, Rayleigh, matsuoka half-centre models) also produce limit cycles
but their waveform shape changes with their parameters.

**Coupled phase oscillators (synchrony).** The Kuramoto model studies how a population of phase
oscillators synchronizes under coupling: φ̇_i = ω_i + Σ_j K_ij sin(φ_j − φ_i). The sinusoidal
coupling pulls phase differences toward stable fixed points; with a phase-bias term
sin(φ_j − φ_i − Φ_ij) the locked state sits at φ_j − φ_i = Φ_ij. Such models do not explain how
rhythm arises; they take oscillators as given and characterize what their *couplings* do — which
is precisely the question of how to set inter-leg phasing. Relatedly, the theory of *symmetric
coupled cell networks* (Golubitsky and Stewart; Golubitsky, Stewart, Buono and Collins's modular
network for legged locomotion, 1998) shows that the symmetry of a network of identical coupled
cells forces the existence of symmetric periodic solutions — and that the standard quadruped
gaits correspond to the symmetry subgroups of a four-cell ring. The H/K theorem characterizes
which spatio-temporal symmetry patterns (hence which gaits) a coupled network can support and
makes the stable one selectable by the coupling, independent of the individual cell's internal
dynamics.

**The brittleness of open-loop scripted gaits.** A direct alternative is to author each joint's
trajectory as an explicit clocked function of time (e.g. sinusoids or splines played back from a
phase clock) and hand-tune it per gait per speed. This is fragile in specific ways. The replayed
trajectory has no internal notion of its own phase as a dynamical state, so it cannot restore
phase after a disturbance — a leg knocked off schedule simply resumes its clock at the wrong
place, and there is no force pulling the four legs back into their relative phasing. Re-tuning the
frequency mid-stride, or splicing one gait into another, injects discontinuities. And every
(gait, speed) pair must be re-authored by hand. These are the pain points a rhythmic dynamical
controller is meant to remove.

## Baselines

- **Hand-scripted / clocked open-loop trajectories.** Joint angles are explicit periodic
  functions of a time clock, tuned by hand. Core method: pick a waveform and a phase schedule
  per leg; replay. Gap: no restoring dynamics (phase is not a state), so perturbations and
  parameter changes break coordination; brittle re-tuning; one design per gait per speed.

- **Half-centre / neural-network CPG models.** Biologically faithful networks of (spiking or
  leaky-integrator) neurons whose mutual inhibition yields anti-phase rhythms; coupled
  half-centres drive flexor/extensor pairs. Core math: reciprocal inhibition between neuron
  populations producing oscillation. Gap for robotics: detailed spiking models are
  computationally heavy for real-time control, and the mapping from neuron parameters to gait
  parameters (frequency, amplitude, phase offsets) is indirect and hard to design or tune.

- **Single-oscillator-per-joint with fixed sinusoidal coupling (Kuramoto-style).** Treat each
  actuated joint as a phase oscillator with constant natural frequency, coupled sinusoidally with
  fixed weights and phase biases. Core math: φ̇_i = ω_i + Σ_j K_ij sin(φ_j − φ_i − Φ_ij), output
  a sinusoid of the phase. Strength: robust phase-locking and synchrony, re-locks after
  perturbation. Gap left open: a plain phase oscillator has a *single* frequency, so the swing
  and stance portions of a stride share one duration — yet biology decouples them (speed sits in
  stance, swing stays constant); and a bare phase variable still needs an amplitude/limit-cycle
  layer and a principled way to *select* a specific gait's symmetric solution as the stable one.

- **Symmetric coupled-cell network gait design.** Use network symmetry (Golubitsky–Stewart) to
  guarantee that walk/trot/pace/bound exist as symmetric periodic solutions of a four-cell ring
  and to make one of them stable by choosing the coupling. Core result: the H/K theorem ties
  admissible spatio-temporal symmetry patterns to subgroup pairs. Gap left open on its own: it
  specifies the coupling architecture and which gait is stable, but not the per-leg oscillator
  whose swing/stance shape and durations actually drive the joints, nor how feedback enters.

## Evaluation settings

The natural testbed is a physics simulator of an articulated quadruped with revolute leg joints
and contact, where leg joints are driven toward targets by PD control. Representative simulators
available at the time are rigid-body engines such as ODE (and front-ends like Webots), and more
recently PyBullet, exercising small quadrupeds with one to three actuated joints per leg in the
sagittal plane. The robot is placed on flat ground and, separately, on sloped or uneven terrain
and stairs to probe robustness. The relevant primitives are: a CPG/oscillator state integrated at
a fixed time step; an inverse-kinematics map from a desired foot position in the leg frame to
joint angles; a per-joint PD law and optionally a Cartesian-space PD law via the leg Jacobian;
and per-foot contact/force sensing. The natural performance metric is forward locomotion speed
(optionally normalized by body length), with cost of transport and gait regularity (per-leg
contact timing, duty cycle) as secondary readouts; and the natural design knobs are the
oscillator frequency/amplitude/phase-offset parameters, which form a low-dimensional search space
to optimize.

## Code framework

A fixed-time-step rhythm generator feeds a robot through inverse kinematics and PD control, with
the rhythm dynamics and gait-coupling left as empty slots.

```python
import numpy as np

class RhythmGenerator:
    """One oscillatory unit per leg; produces a periodic signal per leg with
    controllable amplitude, frequency, and inter-leg phasing. The internal
    dynamics and the inter-leg coupling are the slots to be designed."""

    def __init__(self, n_legs=4, dt=0.001, **params):
        self.n = n_legs
        self.dt = dt
        # per-leg oscillator state (e.g. amplitude/phase or x/y); TODO: choose representation
        self.state = np.zeros((2, n_legs))
        self.params = params
        self._set_gait_coupling(params.get("gait", "TROT"))

    def _set_gait_coupling(self, gait):
        # TODO: the coupling that pins the inter-leg phase offsets defining each gait
        self.coupling = None

    def _integrate(self):
        # TODO: per-leg limit-cycle dynamics + inter-leg coupling, one Euler step
        pass

    def update(self):
        """Advance the oscillators and emit desired foot positions (x, z) per leg."""
        self._integrate()
        x = np.zeros(self.n)   # TODO: map oscillator state -> fore/aft foot position
        z = np.zeros(self.n)   # TODO: map oscillator state -> foot height (lift in swing)
        return x, z


def foot_target_to_torques(env, x, z, foot_y, side_sign, kp, kd):
    """Existing robot primitives: IK + joint PD (and optional Cartesian PD)."""
    action = np.zeros(3 * 4)
    q  = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()
    for i in range(4):
        leg_xyz = np.array([x[i], side_sign[i] * foot_y, z[i]])
        leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)
        tau = kp * (leg_q - q[3*i:3*i+3]) + kd * (0 - dq[3*i:3*i+3])
        # optional: Cartesian PD via leg Jacobian -> tau += J^T (kpC (x_des - x) + kdC (-v))
        action[3*i:3*i+3] = tau
    return action


def run(env, gen, n_steps):
    for _ in range(n_steps):
        x, z = gen.update()
        action = foot_target_to_torques(env, x, z, foot_y=0.0838,
                                        side_sign=np.array([-1, 1, -1, 1]),
                                        kp=np.array([150., 70., 70.]),
                                        kd=np.array([2., 0.5, 0.5]))
        env.step(action)
        v = env.robot.GetBaseLinearVelocity()   # forward speed = objective to maximize
```
