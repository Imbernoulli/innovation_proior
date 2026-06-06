# Central Pattern Generators for quadruped gaits

## Problem

Produce parameterized, tunable quadruped gaits — walk, trot, pace, bound — and switch between
them and across speeds, without hand-authoring a separate joint trajectory for each gait and
speed. A gait is one periodic stride per leg held at fixed inter-leg phase offsets; the
controller must generate that rhythm, lock the legs at the right relative phases, let the phasing
be changed to pick a gait, and restore both shape and phasing after perturbations — all while
reducing the high-dimensional joint problem to a few interpretable knobs.

## Key idea

Drive each leg with a **coupled nonlinear oscillator** (a Central Pattern Generator). Each
oscillator is a **Hopf limit cycle** whose amplitude and frequency are explicit, separable
parameters and whose shape is invariant to frequency. Oscillators are coupled **Kuramoto-style**,
with a sinusoidal phase-difference term whose stable fixed point sits at a prescribed phase
offset — so a **phase-offset matrix Φ is literally the gait**. Switching gait swaps Φ and nothing
else. Because the gait is encoded by a handful of parameters (frequency, amplitude, phase
offsets, swing/stance split), forward speed can be maximized by a **low-dimensional black-box /
evolutionary search** in simulation.

## The dynamics

**Per-leg oscillator (polar Hopf), legs ordered FR, FL, RR, RL.** Amplitude `r_i`, phase `θ_i`:

    ṙ_i = α (μ − r_i²) r_i                                  # r_i → √μ : stable limit cycle
    θ̇_i = ω_i + Σ_j r_j K sin(θ_j − θ_i − Φ_ij)            # Kuramoto coupling locks gait
    ω_i = ω_swing if sin θ_i > 0 else ω_stance              # swing/stance split (sets speed, duty)

The coupling term is zero and restoring at `θ_j − θ_i = Φ_ij`, so the legs lock into the offsets
in Φ; α sets how fast perturbations to the cycle decay; `r_j` scales each leg's influence by how
"alive" its oscillator is.

**Cartesian variant with independent swing/stance frequencies and feedback** (the form used to
decouple the two half-durations and to inject touch-sensor feedback). With `r² = x² + y²`:

    ẋ_i = α (μ − r_i²) x_i − ω_i y_i
    ẏ_i = β (μ − r_i²) y_i + ω_i x_i + Σ_j k_ij y_j + u_i
    ω_i = ω_stance / (e^{−b y_i} + 1) + ω_swing / (e^{b y_i} + 1)     # smooth sigmoid blend

`y`'s sign is the swing/stance indicator; using β ≫ α converges fast on `y` (crisp indicator)
while keeping `x` (the joint command) smooth. The coupling matrix `k_ij` is designed by symmetric
coupled-cell theory (H/K theorem) so exactly the desired gait is the stable symmetric solution.
Foot-contact feedback enters `y` only:

    u_i = −sign(y_i) F                    # fast transition: drive y → 0 (delay ≈ y/F)
    u_i = −ω_i x_i − Σ_j k_ij y_j         # stop & wait at boundary: ẋ=ẏ=0 at y=0, x=±√μ
    u_i = 0                               # otherwise

gated by the oscillator phase and the foot pressure sensor, so a loaded foot waits and a
triggered foot snaps — coupling the rhythm to the actual contact events.

**Gait = the offset matrix.** Trot = diagonal pairs in phase, the two diagonals π apart; bound =
front pair together, hind pair together, π apart; pace = ipsilateral pairs together; walk =
four-beat, legs ~π/2 apart. Speed is set mainly by `ω_stance` (shorter stance → faster), matching
the mammalian inverse-stance↔speed relation, while `ω_swing` is held roughly constant. Duty
factor follows the ratio of the two frequencies.

**Output map (oscillator → foot, leg frame).** Fore/aft from `r cosθ`, lift gated by `sin θ`:

    x_i = − d · r_i · cos θ_i
    z_i = − h + clearance · sin θ_i      if sin θ_i > 0   (swing: foot lifts)
    z_i = − h + penetration · sin θ_i    otherwise        (stance: slight ground press)

with step length `d`, nominal height `h`. Inverse kinematics turns `(x_i, z_i)` into joint
targets, then joint PD (and optional Cartesian PD via the leg Jacobian) produces torques.

## Code

```python
import numpy as np

class HopfNetwork:
    """One Hopf oscillator per leg (polar form), Kuramoto-coupled to set the gait.
    State X: row 0 = amplitudes r_i, row 1 = phases theta_i. Legs ordered FR, FL, RR, RL."""

    def __init__(self,
                 mu=1.0,                  # amplitude^2: r -> sqrt(mu)  (stroke amplitude)
                 omega_swing=5*2*np.pi,   # angular speed in swing  (sin theta > 0)
                 omega_stance=2*2*np.pi,  # angular speed in stance (sets forward speed / duty)
                 gait="TROT",             # selected purely by the phase-offset matrix PHI
                 alpha=50.0,              # limit-cycle convergence rate
                 coupling_strength=1.0,   # Kuramoto coupling K
                 couple=True,
                 dt=0.001,
                 ground_clearance=0.07,   # foot lift in swing
                 ground_penetration=0.01, # foot press in stance
                 robot_height=0.30,       # nominal standing height h
                 des_step_len=0.05):      # fore/aft stroke length d
        self.X     = np.zeros((2, 4))
        self.X_dot = np.zeros((2, 4))
        self._mu, self._alpha = mu, alpha
        self._omega_swing, self._omega_stance = omega_swing, omega_stance
        self._couple, self._K, self._dt = couple, coupling_strength, dt
        self._gc, self._gp = ground_clearance, ground_penetration
        self._h, self._d = robot_height, des_step_len
        self._set_gait(gait)
        self.X[0, :] = np.random.rand(4) * 0.1   # start near origin in r
        self.X[1, :] = self.PHI[0, :]            # seed phases at the gait offsets

    def _set_gait(self, gait):
        # PHI[i, j] = target offset (theta_j - theta_i) the coupling locks to. The matrix IS the gait.
        PI = np.pi
        trot  = np.array([[0, PI, PI, 0],   [PI, 0, 0, PI],   [PI, 0, 0, PI],   [0, PI, PI, 0]])
        walk  = np.array([[0, PI, 3*PI/2, PI/2],
                          [-PI, 0, PI/2, -PI/2],
                          [-3*PI/2, -PI/2, 0, -PI],
                          [-PI/2, PI/2, PI, 0]])
        bound = np.array([[0, 0, PI, PI],   [0, 0, PI, PI],   [PI, PI, 0, 0],   [PI, PI, 0, 0]])
        pace  = np.array([[0, PI, 0, PI],   [PI, 0, PI, 0],   [0, PI, 0, PI],   [PI, 0, PI, 0]])
        self.PHI = {"TROT": trot, "WALK": walk, "BOUND": bound, "PACE": pace}[gait]

    def _integrate(self):
        X = self.X.copy()
        X_dot = np.zeros((2, 4))
        r, theta = X[0, :], X[1, :]
        for i in range(4):
            r_dot = self._alpha * (self._mu - r[i]**2) * r[i]                 # amplitude -> sqrt(mu)
            omega = self._omega_swing if np.sin(theta[i]) > 0 else self._omega_stance
            theta_dot = omega                                                # swing/stance split
            if self._couple:
                for j in range(4):
                    theta_dot += r[j] * self._K * np.sin(theta[j] - theta[i] - self.PHI[i, j])
            X_dot[:, i] = (r_dot, theta_dot)
        self.X = X + X_dot * self._dt          # Euler step
        self.X_dot = X_dot
        self.X[1, :] %= 2 * np.pi              # phases in [0, 2pi)

    def update(self):
        """Advance oscillators; return desired foot (x, z) per leg in the leg frame."""
        self._integrate()
        r, theta = self.X[0, :], self.X[1, :]
        x = -self._d * r * np.cos(theta)
        z = np.where(np.sin(theta) > 0,
                     -self._h + self._gc * np.sin(theta),
                     -self._h + self._gp * np.sin(theta))
        return x, z


def run(env, cpg, n_steps, foot_y=0.0838,
        kp=np.array([150., 70., 70.]), kd=np.array([2., 0.5, 0.5]),
        kpC=np.diag([2500.]*3), kdC=np.diag([40.]*3), add_cartesian_pd=True):
    """Close the loop on a quadruped in a physics sim; return mean forward speed (the objective)."""
    side = np.array([-1, 1, -1, 1])                  # body-right hip sign
    speeds = []
    for _ in range(n_steps):
        xs, zs = cpg.update()
        q, dq = env.robot.GetMotorAngles(), env.robot.GetMotorVelocities()
        action = np.zeros(12)
        for i in range(4):
            foot = np.array([xs[i], side[i]*foot_y, zs[i]])      # desired foot pos, leg frame
            q_des = env.robot.ComputeInverseKinematics(i, foot)  # IK -> joint targets
            tau = kp*(q_des - q[3*i:3*i+3]) + kd*(0 - dq[3*i:3*i+3])    # joint PD
            if add_cartesian_pd:
                J, pos = env.robot.ComputeJacobianAndPosition(i)
                v = J @ dq[3*i:3*i+3]
                tau += J.T @ (kpC @ (foot - pos) + kdC @ (-v))         # Cartesian PD
            action[3*i:3*i+3] = tau
        env.step(action)
        speeds.append(env.robot.GetBaseLinearVelocity()[0])
    return np.mean(speeds[-1000:])
```

## Why it works

- **Limit cycle, not a clocked script.** A Hopf oscillator's periodic motion is *attracting*, so
  perturbations decay; its phase is a genuine state, so events and re-tuning don't break
  continuity. Hopf specifically because amplitude (μ) and frequency (ω) are separate knobs and
  the cycle shape is invariant to ω — retune speed without re-authoring the waveform.
- **Kuramoto coupling = gait selection.** `sin(θ_j − θ_i − Φ_ij)` has a stable zero at the
  prescribed offset, so the legs lock into Φ and re-lock after a kick; the gait is the matrix Φ.
- **Phase-dependent ω.** Independent swing/stance durations match the biology (speed in stance,
  swing ≈ constant) and set the duty factor; the sigmoid blend keeps ω(y) smooth at the switch.
- **Symmetry-based coupling design.** The H/K theorem makes exactly the desired gait the stable
  symmetric solution, independent of the swing/stance durations and of the cell dynamics.
- **Feedback on `y`.** Touch sensors gate the swing↔stance transitions (wait-when-loaded,
  snap-when-triggered) without disturbing the joint command on `x`, coupling rhythm to contact.
- **Few parameters → optimizable.** The behavior reduces to ~8 interpretable knobs (frequencies,
  amplitude, phase offsets/gait, step length, clearance), a low-dimensional space in which a
  black-box / evolutionary optimizer can maximize forward speed in simulation.
