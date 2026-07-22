OK, let me think this through from scratch. I want a four-legged robot to walk, and then to trot, and then to bound, and to slide between those and across speeds without me sitting there re-authoring trajectories by hand. So first: what *is* a gait, concretely, before I commit to any controller?

Stare at a trotting dog. Each leg does the same thing over and over — lift, swing forward, plant, push back, lift again. One periodic stride. A swing phase where the foot is in the air going forward, a stance phase where it's planted and driving the body forward. So per leg, the motion is periodic, full stop. The interesting part isn't the single leg; it's how the four legs are *arranged in time*. In a trot the diagonal pairs go together — front-right with rear-left, front-left with rear-right — and the two diagonals are half a stride apart. In a bound the front pair goes together, the hind pair goes together, and front and hind are half a stride apart. In a pace it's the two left legs together, the two right together. A walk is a four-beat thing, the legs following each other about a quarter stride apart. So a gait is *literally just a set of inter-leg phase offsets on top of one shared periodic stride.* That's the whole content. And which gait you want depends on speed — slow you want a statically stable walk, where enough feet are always down; fast you want a dynamically stable trot or bound. There's even a clean biological hint here: in mammals, speed is set almost entirely by *shortening the stance* — there's roughly a linear relation between speed and the inverse of stance duration — while the swing duration stays nearly constant. Hold onto that; it'll matter.

So the problem decomposes: I need (1) a clean periodic per-leg signal whose amplitude and frequency I can dial, (2) a way to lock the four legs at chosen relative phases, (3) the ability to *change* those relative phases to get a different gait, and (4) robustness — a kick or a bump or a parameter tweak shouldn't wreck the coordination.

The obvious first thing to try is the dumb thing: just script it. Write each joint angle as an explicit function of a time clock — a sinusoid or a spline — give each leg its phase offset, and replay. Let me actually push on this, because I want to feel *why* it's not enough rather than just assert it. So leg i's hip angle is, say, A·sin(2π f t + ψ_i), ψ_i the per-leg offset, knees a function of that. Fine. It moves. Now perturb it: the robot trips, a leg gets shoved. What happens? The clock t keeps ticking; my sinusoid keeps evaluating sin(2π f t + ψ_i) as if nothing happened. The leg's *commanded* trajectory has no memory that the leg is now in the wrong place — there's no state that says "I am currently at phase p and I should be at phase q." Phase here is not a variable of the system; it's buried in the global clock t. So there is no restoring force. Nothing pulls the legs back into their relative phasing after a disturbance. Worse: suppose I want to speed up mid-stride by raising f. The argument 2π f t has a t in it, so changing f makes the phase 2π f t jump discontinuously — a velocity spike at the joints. And splicing walk into trot means hand-stitching two scripts. Every (gait, speed) is a separate authoring job. Three of my four requirements — retunable, switchable, robust — are exactly the ones scripting fails at, and it fails at them for one root reason: *the phase is not a dynamical state with its own restoring dynamics.* It's a lookup against an external clock.

So flip it. I don't want to *replay* a trajectory; I want the trajectory to be the *output of a dynamical system* whose stable behavior is the periodic stride, and whose phase is a genuine state variable. Then "forget the perturbation and return to the cycle" is just the system being attracting, and "lock the legs" is just coupling the states.

What kind of dynamical system has a periodic motion as its *stable* behavior — an attracting one, so transients die out? A stable limit cycle. An isolated closed orbit that nearby trajectories spiral onto. That's exactly the "transient perturbations are rapidly forgotten" property I want. Now, biology agrees with this picture from the other direction: the spinal cord has these distributed circuits — central pattern generators — that put out coordinated rhythmic motor commands from nothing but a constant tonic drive, even cut off from the brain and from sensory feedback. The smallest unit is a half-centre: two neuron populations that can't oscillate alone but, wired with mutual inhibition, settle into anti-phase oscillation — which is exactly the flexor/extensor alternation a joint needs. So nature also says: don't command every angle; make a low-dimensional rhythmic source and let the body fill in the rest. Good. But I don't want to simulate spiking neurons — detailed neuron models are expensive for real-time control and, more to the point, the map from "synaptic conductance" to "frequency, amplitude, phase offset" is indirect and a nightmare to design and tune. I want the *abstraction* — a stable limit cycle whose amplitude and frequency are knobs I can grab directly.

So what's the cleanest oscillator whose amplitude and frequency are literally separate parameters? Write it in polar coordinates and ask for radius and angle to be controllable independently. The Andronov–Hopf normal form does exactly this. In polar form:

  ṙ = α(μ − r²) r
  θ̇ = ω

Look at the radius equation alone: r² < μ gives ṙ > 0, r² > μ gives ṙ < 0, so r is pulled to r = √μ, a stable circle. The phase just winds at rate ω. So the amplitude is √μ, set by μ; the frequency is ω; and they don't interfere — I can change one without touching the other. And crucially the cycle is *harmonic and structurally stable*: its shape (a circle in the (r cosθ, r sinθ) plane) does not change as I vary ω. That last property is the one that kills the scripting headache — I can retune the speed by changing ω and the *waveform shape stays the same*, no re-authoring. (I could use a Van der Pol or a Matsuoka half-centre oscillator instead — they also limit-cycle — but their cycle *shape* warps as you change parameters, so I'd lose the "retune frequency, keep shape" guarantee. Hopf it is.) And the convergence rate to the cycle is α, a third independent knob — how fast it forgets a perturbation.

Now, one oscillator per leg. How does (r_i, θ_i) become a foot trajectory? The foot needs to go fore-and-aft and to lift in swing. The fore/aft part is naturally x_i = −d · r_i · cos θ_i: as θ winds, the foot sweeps forward and back with stroke length set by d (and modulated by the amplitude r_i). For the height, I want the foot to *lift* only during swing and stay down (even press in slightly) during stance. Where's the swing/stance boundary in θ? Take sin θ_i as the indicator: when sin θ_i > 0 the foot is in the lift-half (swing), when sin θ_i < 0 it's in the planted-half (stance). So

  z_i = −h + clearance · sin θ_i   when sin θ_i > 0   (swing, foot rises by up to "clearance")
  z_i = −h + penetration · sin θ_i when sin θ_i ≤ 0   (stance, slight ground press)

with h the nominal standing height. Clean: one oscillator drives one foot's (x, z); the knee/elbow follow by inverse kinematics. So per leg I have a tiny limit-cycle clock, and the whole controller's per-leg shape is three numbers: amplitude μ, frequency ω, and convergence α, plus the geometric d, clearance, h.

Now requirement (2) and (3): lock the four legs at chosen relative phases, and be able to change those phases to switch gaits. I have four phase variables θ_1..θ_4. I want θ_j − θ_i pinned to some target offset Φ_ij. What's the canonical way to make phase oscillators settle at prescribed phase differences? This is the Kuramoto question — how a population of phase oscillators synchronizes under coupling. The Kuramoto coupling is sinusoidal in the phase difference. So add to each θ̇_i a sum over the others:

  θ̇_i = ω + Σ_j r_j K sin(θ_j − θ_i − Φ_ij)

Why does this lock at Φ_ij? Look at one pair. The term sin(θ_j − θ_i − Φ_ij) is zero when θ_j − θ_i = Φ_ij, and its sign restores: if θ_i lags too far, the term pushes θ̇_i up to catch up; if it leads, it slows. So θ_j − θ_i = Φ_ij is a *stable* fixed point of the phase difference. The four phases lock into exactly the offset pattern I wrote into the matrix Φ. And — this is the payoff — *the gait is literally the matrix Φ.* Want a trot? Put the diagonal pairs at zero offset and the diagonals π apart: with legs ordered FR, FL, RR, RL, the trot pattern has Φ such that FR and RL share a phase, FL and RR share the opposite phase, half a cycle away — the {0, π, π, 0} structure across the row. Want a bound? Front pair together, hind pair together, π between them. Want a pace? Left pair, right pair. To change gait I change *nothing but Φ*; the oscillators, the output map, everything else is untouched. That's requirement (3) for free.

And notice the coupling re-locks after a perturbation, because Φ_ij is a *stable* fixed point — kick a leg and the sine terms drag it back into formation. That's requirement (4) on the *coordination* side. The K is the coupling strength; the r_j weight is a nice touch — it means a leg whose oscillator amplitude has collapsed (r_j small) doesn't yank the others around, the influence scales with how "alive" the source oscillator is.

Let me pause and check I've actually solved the original problem. Periodic per-leg motion with dial-able amplitude/frequency — yes, Hopf. Lock at chosen relative phases — yes, Kuramoto coupling. Switch gait — yes, swap Φ. Robust to kicks — yes, both the limit cycle (amplitude/shape restores) and the coupling fixed point (phasing restores). And the whole controller is now a *handful of interpretable numbers*: μ (amplitude), ω (frequency), α (convergence), K (coupling strength), the Φ matrix (gait), plus d, clearance, h for the foot geometry. Twelve-plus joint trajectories collapsed to a tiny parameter vector. That low dimensionality is itself a feature: I can *search* over (ω, μ, step length, …) — grid search, or an evolutionary / black-box optimizer in simulation — to maximize forward speed, which would be hopeless over raw joint splines.

Now I want to go back to that biological hint I parked: speed is set by stance duration, swing stays roughly constant. My Hopf oscillator has a *single* ω — so swing and stance take the same fraction of the period, always. That's wrong for what I want. If I want to go faster the right way, I should shorten stance while leaving swing alone — independent control of the two half-periods. With one ω I can't. So the plain Hopf needs surgery: make the angular speed *phase-dependent* — fast through one half of the cycle, slow through the other.

How do I make ω depend on where I am in the cycle, smoothly? I need a switch that is "ωswing while in the swing half, ωstance while in the stance half," but smooth so there's no jerk at the boundary. A sigmoid on the variable whose sign marks the half does it. Let me write the feedback-friendly Cartesian form with the convention y > 0 for stance and y < 0 for swing. The polar foot-map code later uses sin θ > 0 for swing; that is just a phase/sign convention, not a different mechanism.

  ẋ = α(μ − r²) x − ω y
  ẏ = β(μ − r²) y + ω x,    r = √(x² + y²)

(this is the same Hopf limit cycle, r → √μ, rotating at ω). Now y > 0 vs y < 0 marks the two halves. Set

  ω = ωstance / (e^{−b y} + 1) + ωswing / (e^{b y} + 1).

Check the limits: when y is large and positive, e^{−b y} → 0 so the first term → ωstance and e^{b y} → ∞ so the second → 0; ω → ωstance. When y is large negative, it flips: ω → ωswing. In between, the two sigmoids hand off smoothly, so ω(y) is a continuous blend with no discontinuity at y = 0. Now ωswing and ωstance are *separate knobs*: I tune stance frequency to set speed (matching the biology), and leave swing frequency alone for stability. On the limit cycle the stance half takes about π/ωstance and the swing half about π/ωswing, so the stance duty factor is (1/ωstance)/(1/ωstance + 1/ωswing); shortening stance means increasing ωstance.

One more design choice in that pair of equations: why two different convergence rates α and β, and why faster on y (e.g. α = 5, β = 50)? The x variable is the one I actually read out as the control policy — it drives the joint. I want x to be *smooth*, its time-derivative not too violent. y, on the other hand, I just need to snap onto the limit cycle quickly so the swing/stance indicator is crisp. So converge hard on y (big β) and gently on x (small α): the cycle locks fast in the indicator direction while the actuated signal stays smooth. If I made both fast, x would have a large derivative and the joints would jerk; both slow and the cycle is sloppy. The asymmetry is deliberate.

Now the coupling, in this richer oscillator. I want the four-cell network to support walk, trot, pace, bound, and I want to *guarantee* that the gait I asked for is the *stable* periodic solution and the others aren't accidentally also stable. Just hand-wiring sine couplings worked for phase-locking, but can I design the network *principledly* so that the symmetry of the desired gait is forced? Yes — this is exactly what the theory of symmetric coupled cell networks gives me (Golubitsky and Stewart; and Golubitsky, Stewart, Buono and Collins's modular network for legged locomotion). The idea: if a network of identical cells has a symmetry — a permutation of the cells that leaves the coupling unchanged — then the network is forced to have periodic solutions with that same symmetry. A gait *is* a spatio-temporal symmetry: a trot, for instance, is invariant under "swap the two diagonal pairs and shift by half a period." The H/K theorem makes this precise: for a pair of subgroups K ⊂ H ⊂ Γ of the symmetry group Γ (with cells whose internal dynamics is at least two-dimensional — mine are, I have (x,y)), there exist periodic solutions with spatial symmetries K and spatio-temporal symmetries H exactly when H/K is cyclic and K is an isotropy subgroup, and the solution can be made asymptotically stable. So: take the symmetry group of the gait I want, build the coupling matrix so the network is invariant under it, and the H/K theorem hands me the gait as a periodic solution; then I pick the coupling coefficients so that *that* solution is the stable one and the conjugate gaits (which generically co-exist — e.g. pace and bound and trot all live in the same 4-cell symmetry, as subgroups {I,(12)(34)}, {I,(13)(24)}, {I,(14)(23)}) are not. Concretely I add a linear coupling Σ_j k_ij y_j into the ẏ equation:

  ẋ_i = α(μ − r_i²) x_i − ω_i y_i
  ẏ_i = β(μ − r_i²) y_i + ω_i x_i + Σ_j k_ij y_j

and choose the coupling matrices k_ij (one per gait) so the desired symmetric solution is stable — verified numerically. The beauty is that the network design is *algebraic*, independent of the cell's internal dynamics, so it's scalable (hexapods, etc.) and the gait type is set purely by the matrix and is *independent of the swing/stance durations* I dialed in earlier. Two orthogonal knob sets: the matrix picks the gait, the frequencies pick the speed/duty.

Now the last piece, robustness against the *world*, not just against kicks: sensory feedback. Open-loop, even this beautiful CPG can mistime its foot placement on rough ground — it'll try to lift a foot that's still loaded, or hold a foot up when the ground came early. Biology fixes this by letting foot-contact and load sensing modulate the swing↔stance transition timing: load under the foot delays lift-off, ground contact triggers stance. I want the same: let touch sensors gate the phase transitions. The clean way is to add a control input u_i and inject it into the ẏ equation (not the ẋ equation — y is the swing/stance indicator, so modulating y reshapes the timing; and since x is my actual joint command, keeping the input off x guarantees x stays smooth):

  ẋ_i = α(μ − r_i²) x_i − ω_i y_i
  ẏ_i = β(μ − r_i²) y_i + ω_i x_i + Σ_j k_ij y_j + u_i

Two behaviors I need. First, *stop before transition*: if a foot is still loaded at the end of stance, or hasn't touched down at the end of swing, I want the oscillator to *wait* at the boundary. The boundary is y = 0 (the swing/stance crossing). To freeze there I need ẋ = ẏ = 0 at y = 0. Set u_i = −ω_i x_i − Σ_j k_ij y_j. Then at y = 0: ẏ = β(μ − x²)·0 + ω x + Σ k y − ω x − Σ k y = 0 (the input cancels the rotation and coupling terms), and ẋ = α(μ − x²) x − ω·0 = α(μ − x²)x, which is zero when x = ±√μ. So the system halts at x = ±√μ, y = 0 — exactly the transition point — and waits. Linearize there: the Jacobian is [[−2αμ, −ω_i], [0, 0]], with eigenvalues −2αμ and 0. The stable eigenvector is (1, 0), tangent to the y = 0 axis, while the zero-eigenvalue direction is tilted, (−ω_i/(2αμ), 1). So the local flow is pulled onto the transition axis and cannot pass through it while the stop input is active; it converges to −√μ when approached from y > 0, the stance side, and to +√μ from y < 0, the swing side. Second, *fast transition*: when the sensor says "switch *now*" (weight dropped off in stance; foot hit ground early in swing), I want to drive through the boundary quickly. Set u_i = −sign(y_i)·F with F large. If y_i > 0, the input is −F and y decreases; if y_i < 0, the input is +F and y increases. During the transition x ≈ const and |F| dominates everything else, so ẏ ≈ −sign(y_i)F and y reaches 0 in about |y_i(t_transition)|/F seconds — a controllable, bounded delay rather than a hard reset. So the full feedback law is: u_i = −sign(y_i)F for a forced fast transition, u_i = −ω_i x_i − Σ k_ij y_j to stop and wait at the boundary, and u_i = 0 otherwise, the choice gated by the oscillator's phase and the foot's pressure sensor. This couples the CPG *to the body*: the controller's timing is now negotiated with the actual contact events, which is what makes it ride out terrain it never saw.

Let me take stock of the causal chain, because I want to make sure each piece earned its place. A gait is one periodic stride per leg held at fixed inter-leg phase offsets → scripting it fails because phase isn't a state, so nothing restores it → make each leg a stable *limit cycle* (Hopf, because its amplitude and frequency are separate knobs and its shape is invariant to frequency) → read the foot's fore/aft off r cosθ and its lift off the sign of sinθ → lock the legs with Kuramoto sinusoidal coupling, whose stable fixed point is exactly the prescribed offset, so the *gait is the offset matrix* → since biology sets speed by stance and keeps swing constant, make the angular speed *phase-dependent* via a sigmoid blend of ωswing and ωstance so the two half-durations are independent → converge faster on the indicator y than on the policy x so the joint command stays smooth → design the coupling matrix by symmetry (H/K theorem) so exactly the desired gait is the stable solution, independent of the durations → gate the swing/stance transitions with foot-contact feedback injected into y, to wait-when-loaded and snap-when-triggered → and because the whole thing is now a handful of interpretable parameters (ω_swing, ω_stance, μ, coupling/Φ, step length, clearance), I can hand the speed objective to a black-box / evolutionary search over those few knobs in simulation.

Now let me write the controller. I'll use the polar form for the core network — amplitude r and phase θ directly, with ω switching on the sign of sin θ for the swing/stance split and Kuramoto coupling carrying the phase-offset matrix Φ — and integrate with Euler. Then map (r, θ) to foot (x, z), run inverse kinematics, and close joint and Cartesian PD loops to produce torques in a quadruped simulator; the base forward velocity is the objective I'd optimize the few parameters against.

```python
import numpy as np

class HopfNetwork:
    """One Hopf oscillator per leg (polar form), Kuramoto-coupled to set the gait.
    State: row 0 = amplitudes r_i, row 1 = phases theta_i. Legs ordered FR, FL, RR, RL."""

    def __init__(self,
                 mu=1.0,                  # amplitude^2: r converges to sqrt(mu)  -> stroke amplitude
                 omega_swing=5*2*np.pi,   # angular speed during swing  (sin theta > 0)
                 omega_stance=2*2*np.pi,  # angular speed during stance (sin theta < 0) -> sets speed
                 gait="TROT",             # selected by the phase-offset matrix PHI
                 alpha=50.0,              # limit-cycle convergence rate (forgets perturbations)
                 coupling_strength=1.0,   # Kuramoto coupling weight K
                 couple=True,
                 dt=0.001,
                 ground_clearance=0.07,   # foot lift in swing
                 ground_penetration=0.01, # foot press in stance
                 robot_height=0.30,       # nominal standing height h
                 des_step_len=0.05):      # fore/aft stroke length d
        self.X     = np.zeros((2, 4))     # [r; theta]
        self.X_dot = np.zeros((2, 4))
        self._mu, self._alpha = mu, alpha
        self._omega_swing, self._omega_stance = omega_swing, omega_stance
        self._couple, self._K, self._dt = couple, coupling_strength, dt
        self._gc, self._gp = ground_clearance, ground_penetration
        self._h, self._d = robot_height, des_step_len
        self._set_gait(gait)
        # start near the origin in r, phases seeded at the gait offsets
        self.X[0, :] = np.random.rand(4) * 0.1
        self.X[1, :] = self.PHI[0, :]

    def _set_gait(self, gait):
        # PHI[i, j] = target phase offset (theta_j - theta_i) that the coupling locks to.
        # The matrix IS the gait; nothing else changes between gaits.
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
            # amplitude: stable limit cycle, r -> sqrt(mu)
            r_dot = self._alpha * (self._mu - r[i]**2) * r[i]
            # phase: angular speed switches on the swing/stance half (sign of sin theta)
            omega = self._omega_swing if np.sin(theta[i]) > 0 else self._omega_stance
            theta_dot = omega
            # Kuramoto coupling: stable fixed point at theta_j - theta_i = PHI[i, j] -> locks the gait
            if self._couple:
                for j in range(4):
                    theta_dot += r[j] * self._K * np.sin(theta[j] - theta[i] - self.PHI[i, j])
            X_dot[:, i] = (r_dot, theta_dot)
        self.X = X + X_dot * self._dt          # Euler step
        self.X_dot = X_dot
        self.X[1, :] %= 2 * np.pi              # keep phases in [0, 2pi)

    def update(self):
        """Advance oscillators; return desired foot (x, z) per leg in the leg frame."""
        self._integrate()
        r, theta = self.X[0, :], self.X[1, :]
        x = -self._d * r * np.cos(theta)                       # fore/aft stroke
        z = np.where(np.sin(theta) > 0,
                     -self._h + self._gc * np.sin(theta),      # swing: lift
                     -self._h + self._gp * np.sin(theta))      # stance: slight press
        return x, z


# ---- closing the loop on a quadruped in simulation -------------------------------
# Few interpretable knobs (omega_stance, omega_swing, mu, des_step_len, gait) -> low-dim
# search to maximize forward base velocity.

def run(env, cpg, n_steps, foot_y=0.0838,
        kp=np.array([150., 70., 70.]), kd=np.array([2., 0.5, 0.5]),
        kpC=np.diag([2500.]*3), kdC=np.diag([40.]*3), add_cartesian_pd=True):
    side = np.array([-1, 1, -1, 1])                 # body-right hip sign
    speeds = []
    for _ in range(n_steps):
        xs, zs = cpg.update()
        q  = env.robot.GetMotorAngles()
        dq = env.robot.GetMotorVelocities()
        action = np.zeros(12)
        for i in range(4):
            foot = np.array([xs[i], side[i]*foot_y, zs[i]])     # desired foot pos, leg frame
            q_des = env.robot.ComputeInverseKinematics(i, foot) # IK -> joint targets
            tau = kp*(q_des - q[3*i:3*i+3]) + kd*(0 - dq[3*i:3*i+3])   # joint PD
            if add_cartesian_pd:                                # optional Cartesian PD via Jacobian
                J, pos = env.robot.ComputeJacobianAndPosition(i)
                v = J @ dq[3*i:3*i+3]
                tau += J.T @ (kpC @ (foot - pos) + kdC @ (-v))
            action[3*i:3*i+3] = tau
        env.step(action)
        speeds.append(env.robot.GetBaseLinearVelocity()[0])     # forward speed = objective
    return np.mean(speeds[-1000:])
```
