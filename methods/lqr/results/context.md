# Context: optimal state-feedback control of a linear plant with a quadratic cost

## Research question

Given a linear dynamical plant of *arbitrary* order — many states, possibly many coupled inputs and outputs — how do you design a feedback control law `u = k(x)` that drives the state to zero with a good transient, in a way that is (a) **optimal** against a precise, tunable measure of "bad transient," (b) **constructive** (there is an actual algorithm that produces the gains for an n-th-order multi-input plant, not a hand-tuning procedure), and (c) **guaranteed to stabilize** the closed loop?

The pain is concrete. Servo and regulator design in the late 1950s is dominated by frequency-domain shaping — Bode plots, Nyquist diagrams, root locus, lead-lag compensators tuned to hit phase and gain margins. This works one loop at a time and is essentially single-input single-output (SISO): the designer reads off a gain/phase curve and tunes a compensator by hand. For a plant with several inputs and outputs that interact (a multivariable, MIMO plant), there is no principled way to choose all the gains *jointly*. You tune loop after loop and fight the cross-coupling by trial and error; there is no notion of a *jointly optimal* set of feedback gains, no algorithm that ingests the plant matrices and returns the controller, and no a-priori promise that the tuned controller is even stable. The goal is a theory that takes the plant `ẋ = Fx + Gu` and a cost functional and *computes* the optimal feedback law, for any order, for MIMO, with a stability guarantee.

## Background

The state-space description of a linear plant is the starting object:

    ẋ = F(t)x + G(t)u,   y = H(t)x,

with `x ∈ ℝⁿ` the state, `u ∈ ℝᵐ` the control, `y` the output, and `Φ(t, t₀)` the state-transition matrix of the homogeneous part. Time-invariant plants have constant `F, G, H` and `Φ(t,t₀) = e^{F(t−t₀)}`.

The idea of choosing a controller to **minimize the integral of the squared error** is older than state space: it goes back to Wiener's work on optimal filtering/prediction and to Hall's analysis of linear servomechanisms in the 1940s, and was developed at length in the influential Newton–Gould–Kaiser treatment of analytical servo design. The integral-squared-error (ISE) criterion is the right *idea* — it turns "good transient" into a single number you can minimize — but the formulations of the time stayed unsatisfactory mathematically and, more practically, the algorithms could only be carried out for rather low-order systems. There was no clean state-space theory and no constructive procedure for a general plant.

Two newer bodies of theory are arriving precisely at this moment and are the load-bearing tools:

**Calculus of variations.** The classical machinery (Euler–Lagrange equations, the Weierstrass condition, the second-variation test) characterizes the curve that extremizes an integral functional `∫L dt`. Carathéodory's "royal road" embeds a single extremal in a whole *field* of extremals and converts the problem into a first-order partial differential equation — the Hamilton–Jacobi equation. The strengthened Legendre / second-variation condition for a genuine minimum is that the Hessian of the integrand in the control be positive (semi)definite, `∂²L/∂u² ≥ 0`. A matrix Riccati differential equation is already known to *emerge* in the study of these second variations. The limitation that matters here: the variational solution is *local* and *open-loop* — it produces an extremal trajectory and an adjoint, i.e. a `u*(t)` tied to one initial condition, not a feedback law `u = k(x)` you can evaluate online from whatever state you are in.

**The Hamiltonian / costate viewpoint.** Pontryagin's maximum principle (late 1950s) sharpens the variational necessary conditions: introduce a costate (adjoint) vector `ξ`, form the Hamiltonian `H = L + ⟨ξ, Fx + Gu⟩`, and the optimal control extremizes `H` pointwise while state and costate satisfy a coupled two-point boundary-value problem. Same open-loop character: solving the boundary-value problem yields the optimal input for a *given* boundary condition; re-posing it for a new state means solving it again.

**Dynamic programming and the value function.** Bellman's principle of optimality (mid-1950s) takes the complementary, closed-loop view. Define the *cost-to-go* `V°(x, t)` — the minimum remaining cost starting from state `x` at time `t`. The principle of optimality says an optimal trajectory's tail is itself optimal, which yields the Hamilton–Jacobi–Bellman partial differential equation

    V°_t + min_u [ L(x,u,t) + V°_x · (Fx + Gu) ] = 0.

The minimizing `u` here is automatically a function of the current state `x` (through `V°_x`) — this is exactly the feedback structure the frequency-domain methods and the open-loop variational methods could not deliver. The catch: the HJB PDE is generally intractable; for it to become a usable algorithm you need a special structure that collapses the PDE to something finite-dimensional.

**Lyapunov's second method.** Stability of `ẋ = f(x)` can be certified without solving the ODE by exhibiting a function `V(x) > 0` whose time-derivative along trajectories is negative; then trajectories run downhill to the equilibrium. This is the tool that will let one *prove* the closed loop is stable rather than assume it. A relevant cautionary fact about the design space: it is commonly assumed, tacitly and incorrectly, that a controller which minimizes a cost is automatically stabilizing — minimizing a finite cost does **not** by itself force trajectories to decay, so stability is a separate property that must be established, not taken for granted.

**Estimation as the structural mirror.** The optimal-estimation problem (recover the state from noisy outputs) and the optimal-control problem turn out to be duals of one another under the time-reversal `t ↦ −t` together with the swap `F ↔ F'`, `G ↔ H'`. Wiener's filtering theory is the estimation sister of the control problem; the same quadratic/Riccati structure appears on both sides.

## Baselines

**Frequency-domain SISO design (Bode / Nyquist / root locus; Hall).** Core idea: represent the loop by its transfer function, shape the open-loop gain and phase across frequency with lead/lag compensators, and read robustness off the phase and gain margins. Math/algorithm: graphical loop-shaping and pole placement on the root locus, tuned by hand to margin specifications. Gap: SISO and hand-tuned. No principled extension to coupled multi-input multi-output plants, no joint optimality across loops, no algorithmic synthesis from `(F, G)`, and no stability guarantee beyond the margins the designer happens to achieve.

**Integral-squared-error analytical design (Wiener; Newton–Gould–Kaiser).** Core idea: pick the controller minimizing `∫ e²(t) dt` (or a weighted variant), turning transient quality into one scalar objective — the correct conceptual move. Math/algorithm: spectral-factorization / Wiener–Hopf style solutions for the optimal transfer function. Gap: the formulation is mathematically unsatisfactory and the resulting algorithms are only workable for low-order systems; it is not a state-space theory, and it does not produce a constructive feedback gain for a general n-th-order MIMO plant.

**Calculus-of-variations / maximum-principle optimal control.** Core idea: directly minimize `∫L dt` subject to the dynamics, via Euler–Lagrange / the Hamiltonian and costate. Math/algorithm: Hamiltonian `H = L + ⟨ξ, Fx + Gu⟩`, stationarity in `u`, the canonical state/costate equations, and a two-point boundary-value problem; the second-variation condition `∂²L/∂u² ≥ 0` for a minimum. Gap: *open-loop and local* — it returns an extremal `u*(t)` for one boundary condition, requiring a fresh boundary-value solve for every initial state, rather than a closed-form feedback law `u = k(x)` computable from the present state.

**Dynamic programming / HJB.** Core idea: the cost-to-go `V°(x,t)` and the principle of optimality give a PDE whose pointwise minimizer is a state-feedback law. Math/algorithm: `V°_t + min_u[L + V°_x·(Fx+Gu)] = 0`. Gap: the PDE is generically intractable (the curse of dimensionality); it is a feedback *characterization*, not yet an algorithm, until a structural assumption makes it solvable in closed form.

## Evaluation settings

The natural objects to test such a controller on are linear state-space plants `ẋ = Fx + Gu`, both time-invariant and time-varying, of varying order, single- and multi-input. A canonical multivariable testbed is a rigid body to be regulated/steered — e.g. a quadrotor linearized about hover, whose small-signal dynamics split into translational and attitude channels coupled by gravity (a commanded tilt accelerates horizontal position) — exercising the MIMO case the SISO methods cannot handle. The relevant yardsticks are the closed-loop eigenvalues / decay rate (is the controlled plant asymptotically stable, and how fast), the transient response of states and inputs to an initial offset or a reference command, and the trade-off between state regulation and control effort as the cost weights are varied. The cost weights `(Q, R)` are the experimental knobs: heavier state weight buys faster regulation at the price of larger control inputs, and vice versa. These plants, the eigenvalue/transient metrics, and the weight-sweep protocol are all available independently of any particular synthesis method.

## Code framework

The primitives that already exist: dense linear algebra (matrix products, inverse/solve, eigen- and Schur decompositions), an ODE integrator for simulating `ẋ = Fx + Gu`, and the ability to assemble the plant matrices `(A, B)` and symmetric weight matrices `(Q, R)` from a physical model. The slot to be filled is the synthesis routine that turns `(A, B, Q, R)` into a feedback gain, plus the matrix equation it rests on.

```python
import numpy as np
import scipy.linalg
from scipy.integrate import odeint


def solve_cost_to_go_matrix(A, B, Q, R):
    """Return the symmetric matrix S that characterizes the optimal
    quadratic cost-to-go for  dx/dt = A x + B u  with running cost
    x'Q x + u'R u  (Q = Q' >= 0, R = R' > 0).

    The matrix is fixed by a still-to-be-derived matrix equation in
    (A, B, Q, R); solving that equation is the whole problem.
    """
    # TODO: derive the matrix equation S must satisfy and solve it.
    pass


def feedback_gain(A, B, Q, R):
    """Return the constant state-feedback gain K, the cost matrix S, and
    the closed-loop poles for the infinite-horizon quadratic cost."""
    # TODO: S = solve_cost_to_go_matrix(A, B, Q, R)
    # TODO: K = (some closed form in R, B, S)
    pass


def closed_loop_poles(A, B, K):
    """Eigenvalues of the controlled plant A - B K, used as a stability check."""
    return scipy.linalg.eig(A - B @ K)[0]


def simulate(A, B, K, x0, t):
    """Integrate the closed loop  dx/dt = (A - B K) x  from x0."""
    def rhs(x, _t):
        u = -K @ x
        return A @ x + B @ u
    return odeint(rhs, x0, t)
```
