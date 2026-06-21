A quadruped gait is, once you strip away the apparent complexity of twelve-plus coordinated joints, a remarkably small object: every leg traces one and the same periodic stride — lift, swing forward, plant, push back — and the only thing that distinguishes a walk from a trot from a bound is the *pattern of relative phase offsets* at which the four legs are held against that shared stride. A trot keeps the diagonal pairs together with the two diagonals half a stride apart; a bound keeps the front pair together and the hind pair together, half a stride apart; a pace keeps the ipsilateral pairs together; a walk is a four-beat gait with the legs following each other about a quarter stride apart. And which gait you want is set by speed — a statically stable walk when slow, a dynamically stable trot or bound when fast — with the biological clue that in mammals speed is regulated almost entirely by *shortening the stance phase* (speed scales roughly with the inverse of stance duration) while swing duration stays nearly constant. So the controller I need must do four things: produce a clean periodic per-leg signal whose amplitude and frequency I can dial smoothly, lock the four legs at chosen relative phases, let me *change* those phases to retrieve a different gait, and survive perturbations and rough terrain without losing coordination.

The obvious approach — script each joint as an explicit function of a time clock, $A\sin(2\pi f t + \psi_i)$ with a per-leg offset $\psi_i$, and replay it — fails precisely on three of those four requirements, and for one root reason. The phase here lives only in the global clock $t$; it is not a state of the system. So when a leg is knocked off schedule, the commanded trajectory keeps evaluating against $t$ as if nothing happened — there is no restoring force pulling the legs back into formation. Retuning $f$ mid-stride makes the argument $2\pi f t$ jump, spiking joint velocities. And every gait–speed pair is a separate hand-authoring job. The neural-network CPG models from biology fix the robustness but are heavy for real-time control and expose no clean map from neuron parameters to frequency, amplitude, and phase. Plain Kuramoto-coupled phase oscillators give robust phase-locking but carry only a bare phase with no amplitude of their own, and their coupling weights are hand-set rather than tied to a gait. Symmetric coupled-cell theory says which gaits a four-cell network *can* support but leaves the cell dynamics, the joint drive, and the world-in-the-loop unspecified. None of these alone is the controller.

I propose a network of coupled nonlinear oscillators — a Central Pattern Generator — that I call the Hopf network. The move is to make the trajectory the *output of an attracting dynamical system* rather than a replayed script, so that "forget the perturbation" is just the system returning to its limit cycle and "lock the legs" is just coupling their states. Each leg is one Hopf (Andronov–Hopf normal form) oscillator. In polar form the per-leg dynamics is

$$\dot r_i = \alpha\,(\mu - r_i^2)\,r_i, \qquad \dot\theta_i = \omega_i + \sum_j r_j\,K\,\sin(\theta_j - \theta_i - \Phi_{ij}).$$

The amplitude equation alone earns its place: for $r_i^2 < \mu$ we have $\dot r_i > 0$ and for $r_i^2 > \mu$ we have $\dot r_i < 0$, so $r_i$ is pulled to the stable circle $r_i = \sqrt{\mu}$. The amplitude $\sqrt{\mu}$ and the frequency $\omega$ are therefore *separate, non-interfering knobs*, and — this is why Hopf specifically rather than Van der Pol or a Matsuoka half-centre, whose cycle shapes warp with their parameters — the harmonic cycle's shape is invariant to $\omega$, so I can retune speed without re-authoring the waveform. The rate $\alpha$ is a third independent knob: how fast the cycle forgets a kick.

The coupling term is the gait. $\sin(\theta_j - \theta_i - \Phi_{ij})$ is zero exactly at $\theta_j - \theta_i = \Phi_{ij}$, and its sign restores: a lagging leg is sped up, a leading one slowed, so the prescribed offset is a *stable* fixed point of the phase difference. The four legs lock into whatever offset pattern I write into the matrix $\Phi$, and re-lock after a disturbance because that fixed point is stable. The payoff is that the gait *is* the matrix $\Phi$ — to switch from trot to bound to pace to walk I change nothing but $\Phi$. The weight $r_j$ in front of each coupling term is a deliberate touch: a leg whose oscillator amplitude has collapsed contributes little, so its influence scales with how "alive" its source oscillator is. The output map turns each $(r_i,\theta_i)$ into a foot position in the leg frame — fore/aft from $r\cos\theta$, lift gated by the sign of $\sin\theta$:

$$x_i = -d\,r_i\cos\theta_i, \qquad z_i = \begin{cases} -h + \text{clearance}\cdot\sin\theta_i & \sin\theta_i > 0 \ \text{(swing: foot lifts)}\\[2pt] -h + \text{penetration}\cdot\sin\theta_i & \text{otherwise (stance: slight press)}\end{cases}$$

with step length $d$ and nominal standing height $h$; inverse kinematics and PD control then turn the foot target into joint torques.

Two refinements complete the method. First, the biology says speed comes from stance, not swing, but a single $\omega$ forces both half-periods to be equal fractions of the stride. So I make the angular speed *phase-dependent*. Writing the same Hopf cycle in Cartesian form (with the convention $y>0$ for stance, $y<0$ for swing, $r^2 = x^2 + y^2$),

$$\dot x_i = \alpha\,(\mu - r_i^2)\,x_i - \omega_i y_i, \qquad \dot y_i = \beta\,(\mu - r_i^2)\,y_i + \omega_i x_i + \sum_j k_{ij}\,y_j + u_i,$$

I blend two frequencies through a sigmoid on $y$,

$$\omega_i = \frac{\omega_{\text{stance}}}{e^{-b y_i} + 1} + \frac{\omega_{\text{swing}}}{e^{b y_i} + 1},$$

which hands off to $\omega_{\text{stance}}$ for large positive $y$ and to $\omega_{\text{swing}}$ for large negative $y$, smoothly and with no discontinuity at the $y=0$ boundary. Now stance duty is $(1/\omega_{\text{stance}})/(1/\omega_{\text{stance}} + 1/\omega_{\text{swing}})$, and shortening stance — raising $\omega_{\text{stance}}$ — sets speed while $\omega_{\text{swing}}$ holds the swing constant, exactly matching the mammalian relation. I use two different convergence rates, $\beta \gg \alpha$ (e.g. $\beta = 50$, $\alpha = 5$): $x$ is the variable I read out as the joint command, so I converge it *gently* to keep its derivative smooth, while $y$ is only the swing/stance indicator, so I snap it onto the cycle *hard* to keep the indicator crisp. The linear coupling $\sum_j k_{ij} y_j$ now sits in the $\dot y$ equation, and I design the matrices $k_{ij}$ by symmetric coupled-cell theory: a gait is a spatio-temporal symmetry (a trot is invariant under "swap the diagonal pairs and shift by half a period"), and the H/K theorem guarantees that for the right subgroup pair a symmetric periodic solution with that symmetry exists and can be made asymptotically stable when the cells carry at least two-dimensional internal dynamics — which $(x,y)$ does. Choosing $k_{ij}$ so that exactly the desired gait is the stable solution makes the gait-selection algebraic and independent of the swing/stance durations, so the matrix picks the gait and the frequencies pick the speed, orthogonally.

Second, the input $u_i$ closes the loop on the world via foot-contact sensing, injected into $\dot y$ only so the joint command $x$ stays clean. I need two behaviors. To *stop and wait* at a transition when a foot is still loaded at the end of stance or has not touched down at the end of swing, set $u_i = -\omega_i x_i - \sum_j k_{ij} y_j$; at $y=0$ this cancels the rotation and coupling terms, giving $\dot y = 0$ and $\dot x = \alpha(\mu - x^2)x$, which vanishes at $x = \pm\sqrt{\mu}$ — the system halts exactly at the transition point and cannot cross it (the Jacobian there is $[[-2\alpha\mu, -\omega_i],[0,0]]$, eigenvalues $-2\alpha\mu$ and $0$, the stable direction tangent to the $y=0$ axis). To *force a fast transition* when the sensor says switch now, set $u_i = -\,\text{sign}(y_i)\,F$ with $F$ large; then $\dot y \approx -\text{sign}(y_i)F$ dominates and $y$ reaches $0$ in about $|y_i|/F$ seconds — a bounded, controllable delay rather than a hard reset. Otherwise $u_i = 0$. So a loaded foot waits and a triggered foot snaps, negotiating the rhythm's timing with the actual contact events and letting the controller ride out terrain it never saw. The whole controller collapses to a handful of interpretable knobs — $\mu$, $\omega_{\text{swing}}$, $\omega_{\text{stance}}$, $\alpha$, $K$ and the gait matrix, plus $d$, clearance, $h$ — a low-dimensional space in which a black-box or evolutionary optimizer can maximize forward speed in simulation, something hopeless over raw joint splines.

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
