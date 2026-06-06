# The Linear-Quadratic Regulator (LQR) and the algebraic Riccati equation

## Problem

For a linear plant `ẋ = Ax + Bu` of arbitrary order, with possibly many coupled inputs and outputs, find a **state-feedback** control law `u = k(x)` that regulates the state to zero optimally against a tunable measure of transient quality and control effort — constructively (an algorithm, not hand-tuning), for the MIMO case, with a stability guarantee. The cost to minimize is the quadratic functional

    J = ∫₀^∞ ( xᵀQx + uᵀRu ) dt,   Q = Qᵀ ⪰ 0,  R = Rᵀ ≻ 0.

`Q` weights state error, `R` weights control effort; together they are the regulation-vs-effort trade-off knob.

## Key idea

Take the dynamic-programming / cost-to-go view. The Hamilton–Jacobi–Bellman equation for the optimal cost-to-go `V*(x)` is

    0 = min_u [ xᵀQx + uᵀRu + V*ₓ·(Ax + Bu) ].

Because the dynamics are linear and the cost quadratic, guess a **quadratic value function** `V*(x) = xᵀSx` with `S = Sᵀ`. Then `V*ₓ = 2Sx`, the inner minimization over `u` becomes an unconstrained strictly-convex quadratic (`R ≻ 0`), and setting its gradient to zero gives **linear state feedback** — it is derived, not assumed:

    u* = −R⁻¹Bᵀ S x = −K x,   K = R⁻¹Bᵀ S.

Substituting `u*` back and requiring the HJB to hold for all `x` forces `S` to satisfy the **continuous-time algebraic Riccati equation (CARE)**:

    AᵀS + SA − S B R⁻¹ Bᵀ S + Q = 0.

Solving this one matrix equation yields the constant, time-invariant optimal gain.

**Why the hypotheses.** If `(A, B)` is **controllable** (`rank[B, AB, …, Aⁿ⁻¹B] = n`, equivalently the controllability Gramian is positive definite), every state can be driven to the origin with finite energy, so the infinite-horizon cost is finite and the stabilizing CARE solution `S ⪰ 0` exists. If in addition `(A, Q^{1/2})` is **observable**, the cost "sees" every mode, so along the closed loop `d/dt(xᵀSx) = −(xᵀQx + u*ᵀRu*) ≺ 0`: `V* = xᵀSx` is a strict Lyapunov function and `A − BK` is Hurwitz. Minimizing a finite cost is **not** by itself stabilizing — observability is what makes it so.

## Variants

- **Finite horizon** (`t_f < ∞`): the same algebra with a time-varying `S(t)` solving the *Riccati differential equation* (RDE), integrated backward from the terminal weight:

      −Ṡ = AᵀS + SA − S B R⁻¹ Bᵀ S + Q,   S(t_f) = Q_f,

  giving a time-varying gain `K(t) = R⁻¹BᵀS(t)`.

- **Discrete time** (`x[k+1] = Ax[k] + Bu[k]`): the discrete algebraic Riccati equation (DARE)

      S = AᵀSA − AᵀSB(R + BᵀSB)⁻¹BᵀSA + Q,   K = (R + BᵀSB)⁻¹BᵀSA,  u = −Kx.

  The extra `(R + BᵀSB)⁻¹` appears because the one-step minimization over `u[k]` sees `S` through `BᵀSB`.

**Solving the CARE.** It is quadratic in `S`, so it is solved via the `2n×2n` Hamiltonian matrix `Z = [[A, −BR⁻¹Bᵀ], [−Q, −Aᵀ]]`, whose eigenvalues are symmetric about the imaginary axis. The stabilizing `S` is the graph of the stable invariant subspace: ordered-Schur-decompose `Z`, take the leading `n` columns `[U₁; U₂]`, and `S = U₂U₁⁻¹`.

## Code

Faithful to `scipy.linalg.solve_continuous_are` (which forms exactly that Hamiltonian pencil) and the standard `K = R⁻¹BᵀS`, `u = −Kx` pattern.

```python
import numpy as np
import scipy.linalg
from scipy.integrate import odeint


def lqr(A, B, Q, R):
    """Continuous-time infinite-horizon LQR.
    Plant   dx/dt = A x + B u
    Cost    J = ∫ ( x'Q x + u'R u ) dt,   Q = Q' >= 0,  R = R' > 0.
    Returns gain K (u = -K x), cost-to-go matrix S, closed-loop poles.
    """
    # Algebraic Riccati equation: A'S + SA - S B R^{-1} B'S + Q = 0.
    S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    # Gain from the HJB inner-min: u* = -R^{-1} B'S x.
    K = np.linalg.solve(R, B.T @ S)
    cl_poles = scipy.linalg.eig(A - B @ K)[0]
    return K, S, cl_poles


def dlqr(A, B, Q, R):
    """Discrete-time infinite-horizon LQR for x[k+1] = A x[k] + B u[k]."""
    S = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ S @ B, B.T @ S @ A)
    cl_poles = scipy.linalg.eig(A - B @ K)[0]
    return K, S, cl_poles


def simulate(A, B, K, x0, t):
    """Integrate the closed loop dx/dt = (A - B K) x from x0."""
    def rhs(x, _t):
        return A @ x + B @ (-K @ x)
    return odeint(rhs, x0, t)
```

### Worked example: quadrotor about hover (MIMO)

Linearizing a quadrotor about hover decouples the small-signal dynamics into translational and attitude channels coupled by gravity: a pitch tilt `θ` accelerates horizontal position by `g·θ`, so position is steered *through* attitude. LQR chooses all the gains in a channel jointly — exactly the cross-coupling that defeats one-loop-at-a-time tuning.

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

K, S, poles = lqr(Ax, Bx, Q, R)
# control law tracks a reference: u = -K (x - x_ref)
```

The other channels (`y` with `−g` coupling, `z`, yaw) are built the same way; each solves its own CARE and yields a constant gain, and `eig(A − BK)` confirms the closed-loop poles sit in the open left half-plane.
