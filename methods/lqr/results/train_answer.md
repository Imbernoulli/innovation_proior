The problem I want to solve is the design of feedback for a linear plant $\dot{x} = Ax + Bu$ of arbitrary order, where the state $x$ collects everything I care to regulate and $u$ is the bank of controls I get to choose. The goal is a control law $u = k(x)$ that drives the state back to zero with a clean transient, and I want three things at once that no existing method delivers together: it should be *optimal* against a precise, tunable measure of transient quality; it should be *constructive*, an algorithm that ingests the plant matrices and returns the gains rather than a hand-tuning ritual; and it should come with a *stability guarantee*. The way servo and regulator design is done now is frequency-domain loop-shaping — draw the Bode or Nyquist plot of one loop, add a lead-lag compensator, and tune by hand until the phase and gain margins look acceptable. That is fine for a single input and a single output. But the moment the plant is genuinely multivariable, with several inputs and outputs that talk to each other, loop-shaping collapses: I am stuck tuning loop after loop and fighting the cross-coupling by trial and error, with no way to choose *all* the gains jointly so they are best together, no algorithm that consumes $(A,B)$ and hands back the controller, and no a-priori promise that the thing I tuned is even stable. The integral-squared-error tradition of Wiener, Hall, and Newton–Gould–Kaiser had the right instinct — minimize $\int e^2\,dt$, turning "the transient looks good" into one scalar to minimize — but its machinery was a spectral/transfer-function affair that only worked for low-order systems and never became a clean state-space theory or a constructive feedback law. And the optimal-control routes available, the calculus of variations and the maximum principle, return the wrong kind of object: solving the Hamiltonian two-point boundary-value problem yields an open-loop optimal *signal* $u^*(t)$ tied to one initial condition, so a bump to a new state forces a fresh solve — that is not the feedback law $u=k(x)$ that disturbance rejection and the very notion of a controller demand.

The method I propose is the Linear-Quadratic Regulator. I keep the ISE instinct and re-pose it in state space by writing "good transient with bounded effort" as the quadratic cost $$J = \int_0^\infty \left( x^\top Q x + u^\top R u \right) dt, \qquad Q = Q^\top \succeq 0, \quad R = R^\top \succ 0,$$ where $x^\top Q x$ penalizes how far the state is from zero — $Q$ saying which combinations of states I care about and how much — and $u^\top R u$ pays for control, without which the minimizer would be degenerate and demand infinite control to crush the state instantly. The pair $(Q,R)$ is exactly the trade-off knob the old methods lacked: turn $R$ down and buy aggressive regulation at the price of big inputs, turn it up and get gentle control. The choice of a quadratic rather than $|x|$ or $x^4$ is not just heritage; it is what makes the optimization close in feedback form. Insisting on $R \succ 0$ strictly (not merely semidefinite) is load-bearing: it is the second-variation / Legendre condition $\partial^2 L/\partial u^2 \succ 0$ that makes the inner minimization over $u$ strictly convex and uniquely solved, and it guarantees $R^{-1}$ exists — if $R$ were singular some control direction would be free and the problem ill-posed. $Q$ may stay merely $\succeq 0$, since I need not penalize every state.

To get a feedback law rather than an open-loop signal I take the dynamic-programming view. Define the cost-to-go $V^*(x)$, the minimum remaining cost from state $x$; Bellman's principle of optimality, applied over a vanishing time step, yields the Hamilton–Jacobi–Bellman equation $$0 = \min_u \left[ x^\top Q x + u^\top R u + V^*_x \cdot (Ax + Bu) \right].$$ The minimizing $u$ here depends on $V^*_x$ evaluated at the current $x$, so whatever falls out of the inner minimization is *automatically* a feedback law — the closed-loop object the variational route could not give. The price is that the HJB is in general an intractable PDE in $n$ state variables; I need a structural assumption to collapse it. Here the linear-quadratic structure earns its keep. Because scaling the initial state by $c$ scales every reachable trajectory's state and control by $c$ and hence the cost by $c^2$, the value function ought to be a quadratic form, so I guess the ansatz $V^*(x) = x^\top S x$ with $S = S^\top$. This is the move that turns the PDE finite-dimensional, since a quadratic form is fully described by the matrix $S$. Then $V^*_x = 2Sx$, and the inner minimization sees only the $u$-dependent pieces $u^\top R u + 2 x^\top S B u$, an unconstrained quadratic that, because $R \succ 0$, is strictly convex; setting its gradient to zero, $2Ru + 2 B^\top S x = 0$, gives $$u^* = -R^{-1} B^\top S\, x = -Kx, \qquad K = R^{-1} B^\top S.$$ The feedback is *linear in the state* — I did not assume linear feedback, I assumed a quadratic value function and linear feedback fell out. The minus sign is not cosmetic: it came straight from $2Ru = -2B^\top S x$, it is negative feedback, and it is the sign that has any chance of stabilizing the loop. Substituting $u^*$ back, the $u$-dependent terms collapse to $-x^\top S B R^{-1} B^\top S x$, and after symmetrizing the cross term $2x^\top S A x = x^\top(SA + A^\top S)x$, requiring the HJB to hold for *all* $x$ forces the symmetric middle matrix to vanish, giving the continuous-time algebraic Riccati equation $$A^\top S + S A - S B R^{-1} B^\top S + Q = 0,$$ which is quadratic in $S$ through the $-SBR^{-1}B^\top S$ term. Solving this one matrix equation yields the constant, time-invariant optimal gain — constructive, MIMO, any order. (The finite-horizon case is the same algebra with a time-varying $S(t)$ solving the Riccati differential equation $-\dot S = A^\top S + SA - SBR^{-1}B^\top S + Q$ backward from the terminal weight $S(t_f)=Q_f$; the infinite-horizon $\dot S \to 0$ is what makes the gain a fixed matrix rather than clock-dependent.)

Two things in "set $\dot S = 0$" are wishes that need proof. First, that the infinite-horizon problem is well-posed and the Riccati limit exists: for the cost-to-go $x^\top S x$ to be finite at all, *some* control must drive every state to the origin with finite energy. The clean sufficient condition is complete controllability — captured by the controllability Gramian $W(t_0,t_1) = \int_{t_0}^{t_1} \Phi(t_0,t) B B^\top \Phi^\top(t_0,t)\,dt$ being positive definite, or in the time-invariant case the rank test $\operatorname{rank}[\,B,\,AB,\,A^2B,\,\dots,\,A^{n-1}B\,] = n$. Then any state can be steered to zero with finite energy, the optimal cost over longer horizons is bounded and monotone, hence converges, and the limit solves the CARE. The weakest sufficient condition is stabilizability — only the nondecaying modes need be controllable — and that is the solver-level minimum. Second, stability, which is the trap: it is widely and wrongly assumed that any cost-minimizing controller is stabilizing. It is not — minimizing a *finite* cost does not by itself force the state to decay, since a mode the cost does not see (absent from $Q$) could grow while the cost-to-go stays finite. So stability must be proved, and the value function is itself the Lyapunov candidate. Along the optimal closed loop the HJB gives $$\frac{d}{dt}\, V^*(x(t)) = V^*_x \cdot \dot x = -\left( x^\top Q x + u^{*\top} R u^* \right) \le 0,$$ so the cost-to-go decreases at exactly the instantaneous running-cost rate. That gives Lyapunov stability; to upgrade to asymptotic stability I must rule out a nonzero closed-loop trajectory on which the running cost stays zero, which is precisely complete observability of $(A, Q^{1/2})$ — the dual of controllability, with its own rank test — so that no nonzero motion hides from the state-cost output forever and the only invariant set with $\dot V^* = 0$ is the origin, making $A - BK$ Hurwitz. The weakest form is detectability. The closed loop is $\dot x = (A - BR^{-1}B^\top S)x$, and the same Lyapunov identity confirms the $-K$ sign is the stabilizing one — had I taken $+K$, the identity would not hold. Numerically, since the CARE is quadratic I cannot just invert something; the costate route hands the answer through the $2n\times 2n$ Hamiltonian matrix $Z = \begin{bmatrix} A & -BR^{-1}B^\top \\ -Q & -A^\top \end{bmatrix}$, whose eigenvalues pair as $\lambda$ and $-\overline{\lambda}$. The stabilizing $S$ is the graph of the stable invariant subspace: ordered-Schur-decompose $Z$ to put the negative-real-part eigenvalues first, take the leading $n$ columns $[U_1; U_2]$, and form $S = U_2 U_1^{-1}$ when $U_1$ is nonsingular. The library solver implements the same idea in a numerically safer extended-Hamiltonian-pencil form with balancing, deflation, and QZ ordering. The whole synthesis is then: solve the CARE for $S$, set $K = R^{-1}B^\top S$, apply $u = -Kx$.

```python
import numpy as np
import scipy.linalg
from scipy.integrate import odeint


def solve_cost_to_go_matrix(A, B, Q, R):
    """Continuous-time infinite-horizon cost matrix.
    Plant   dx/dt = A x + B u
    Cost    J = ∫ ( x'Q x + u'R u ) dt,   Q = Q' >= 0,  R = R' > 0.
    """
    # Algebraic Riccati equation: A'S + SA - S B R^{-1} B'S + Q = 0.
    return scipy.linalg.solve_continuous_are(A, B, Q, R)


def feedback_gain(A, B, Q, R):
    """Return gain K (u = -K x), cost-to-go matrix S, closed-loop poles."""
    S = solve_cost_to_go_matrix(A, B, Q, R)
    # Gain from the HJB inner-min: u* = -R^{-1} B'S x.
    K = np.linalg.solve(R, B.T @ S)
    return K, S, closed_loop_poles(A, B, K)


def closed_loop_poles(A, B, K):
    """Eigenvalues of A - B K."""
    return scipy.linalg.eig(A - B @ K)[0]


def simulate(A, B, K, x0, t):
    """Integrate the closed loop dx/dt = (A - B K) x from x0."""
    def rhs(x, _t):
        return A @ x + B @ (-K @ x)
    return odeint(rhs, x0, t)
```

On a genuinely multivariable plant — a quadrotor linearized about hover — the structure the SISO methods could never handle shows up directly. Hover decouples the small-signal dynamics into channels coupled by gravity: a pitch tilt $\theta$ accelerates horizontal position by $g\cdot\theta$, so position is steered *through* attitude. That $g$-coupling is exactly the cross-channel interaction hand-tuning would have to fight; here it is just an entry in $A$, and the Riccati solver chooses all the gains in a channel jointly.

```python
g, m, Ix, Iy, Iz = 9.81, 1.0, 0.1, 0.1, 0.2

# x-channel: state [x, xdot, pitch, pitchdot]
Ax = np.array([[0, 1, 0, 0],
               [0, 0, g, 0],      # xddot = g * pitch   (gravity coupling)
               [0, 0, 0, 1],
               [0, 0, 0, 0]])
Bx = np.array([[0], [0], [0], [1 / Ix]])   # pitch-torque input

Q = np.eye(4); Q[0, 0] = 10.0    # weight position error; R is the effort knob
R = np.array([[1.0]])

K, S, poles = feedback_gain(Ax, Bx, Q, R)
# control law tracks a reference: u = -K (x - x_ref)
```
