# Model Predictive Control (MPC): constrained receding-horizon control

## Problem

Regulate (or track a setpoint with) a plant `x_{k+1} = A x_k + B u_k` while respecting **hard constraints** that the unconstrained optimal controller cannot: actuator saturation `u in U`, actuator slew-rate limits on `Δu = u_k - u_{k-1}`, and state/output limits `x in X`. LQR/LQG produce an optimal *unconstrained* law `u = -Kx`; when an input saturates, the applied input is no longer `-Kx` and the optimality/stability guarantees (which assume `-Kx` is applied exactly) break. MPC keeps the quadratic objective but folds the constraints in directly, and crucially does so in a way that *anticipates* limits before they are hit.

## Key idea

At each sampling instant, with the current measured state `x(t)`, solve a **finite-horizon constrained optimal control problem** over `N` steps for the whole input sequence, then **apply only the first input**, re-measure, and re-solve (the *receding horizon*):

    min_{u_0..u_{N-1}}  sum_{k=0}^{N-1} (x_k'Q x_k + u_k'R u_k) + x_N'P x_N
    s.t.  x_{k+1} = A x_k + B u_k,   x_0 = x(t),
          u_k in U,  x_k in X,  k=0..N-1,   x_N in X_f.

This is a convex **quadratic program** (quadratic cost, linear equality dynamics, linear inequality boxes). Re-solving from the measured state each step is what closes the loop in **feedback** (disturbance rejection, robustness to model error); enforcing `u in U`, `x in X` over the whole horizon is what keeps the applied input feasible and anticipates future state-limit violations. In the region where no constraint is active, the QP solution's first input *reproduces* the LQR gain `-Kx` — MPC is LQR with the saturations folded in.

**Two QP constructions.**
- *Condensed (dense):* eliminate the states by substitution, `X = S^x x(0) + S^u U_0` with `S^x=[I;A;...;A^N]` and `S^u` block-lower-triangular: for state block `i=0..N` and input block `j=0..N-1`, the `(i,j)` block is `A^{i-1-j}B` when `i>j` and zero otherwise. Then `J = U_0'H U_0 + 2x'F U_0 + x'Y x` with `H = S^u' Qbar S^u + Rbar > 0`, `F = S^x' Qbar S^u`, `Y = S^x' Qbar S^x`, `Qbar = blockdiag(Q,..,Q,P)`, `Rbar = blockdiag(R,..,R)`. Decision variable `U_0` only (dim `mN`); constraints `G_0 U_0 <= w_0 + E_0 x(0)`. Unconstrained minimum `U_0* = -H^{-1}F'x` whose first block is the LQR gain.
- *Sparse (banded):* keep `(x_0..x_N, u_0..u_{N-1})` as decision variables, dynamics as equality constraints, boxes as bounds. Bigger but sparse — the form a sparse QP solver (OSQP) prefers.

## Why the terminal pieces (feasibility and stability)

A finite horizon, re-solved repeatedly, is **not** automatically feasible or stabilizing the way the infinite-horizon LQR is.

- **Persistent feasibility.** Feasible now does not imply feasible next step: a short horizon can drive a plant with integrating or unstable modes into a region where, given the bounded input, no admissible sequence keeps the state in the box. Fix: a **terminal constraint** `x_N in X_f`, with `X_f subset X`, and `X_f` control invariant under admissible inputs. Then a one-step shift of the optimal sequence, appended with one more admissible terminal move, is feasible for the next problem, so feasibility propagates. With slew-rate limits, the previous input is included in the state or the appended terminal move is required to satisfy the last rate bound.

- **Stability via terminal cost.** Minimizing a finite-horizon cost need not force decay. Let `J*(x)` be the QP optimal value. Shifting the optimal sequence and appending the terminal control `v` on `X_f` gives the bound
  `J*(x^+) <= J*(x) - q(x,u_0*) - p(x_N) + q(x_N,v) + p(Ax_N+Bv)`.
  If the terminal pieces satisfy the **control-Lyapunov condition** `-p(x) + q(x,v) + p(Ax+Bv) <= 0` on `X_f`, then `J*(x^+) <= J*(x) - q(x,u_0*)`. With a positive-definite regulated stage cost, or the usual detectability condition, this makes `J*` a Lyapunov function and gives asymptotic stability on the feasible set.

- **The LQR connection.** Choose the terminal cost `p(x)=x'Px` with `P = S`, the discrete-LQR value matrix (DARE solution `S = A'SA - A'SB(R+B'SB)^{-1}B'SA + Q`), and the terminal controller `v = -Kx`. Then the control-Lyapunov inequality holds **with equality**, because it reduces to `(A-BK)'S(A-BK) - S + Q + K'RK = 0`, the closed-loop Lyapunov identity implied by the DARE for `K=(R+B'SB)^{-1}B'SA`. The terminal cost then equals the infinite-horizon LQR tail cost `sum_{k>=N}(x_k'Qx_k+u_k'Ru_k)`: MPC = a constrained head spliced onto an LQR tail.

## Practical knobs

- **Move/rate penalty `Q_Δu` on `Δu = u_k - u_{k-1}`:** damps aggressiveness, adds robustness to model error, conditions the QP, and lets the actuator **slew-rate** limit `Δu in [Δu_min, Δu_max]` enter as a linear constraint. Requires carrying the last applied input `u_{-1}` as a parameter.
- **Hard inputs, soft states:** input limits are physical (hard); state/output limits can be softened with signed slack variables in the state-bound rows and a large quadratic penalty, as in the full sparse implementation, so a disturbance can't make the QP infeasible — it returns the least-violating solution while inputs stay strictly within their box.
- **Warm start:** consecutive QPs differ only in the right-hand side (`x(t)`, `u_{-1}`); set the solver up once and warm-start each re-solve so it finishes within one sampling interval.

## Code

The code uses the OSQP sparse/banded formulation: decision vector `(x_0..x_N, u_0..u_{N-1})`, dynamics as equalities, box and slew-rate constraints as bounds, and only the first input applied. This is the no-slack core; the deployment variant adds the signed slack block described above.

```python
import numpy as np
import scipy.sparse as sparse
import scipy.linalg
import osqp


class MPCController:
    """Linear constrained receding-horizon controller (sparse OSQP form).
    OSQP solves 1/2 z'Pz + q'z; constants are omitted, so P=Q and q=-Q xref
    encode the half-scaled tracking objective. Each step: solve the
    finite-horizon QP from the measured state, apply only the first input,
    re-measure, re-solve.
    QxN can be set to the LQR terminal cost; optional xfmin/xfmax tighten
    x_N to a terminal box.
    """
    def __init__(self, Ad, Bd, Np=20, x0=None, xref=None, uref=None, uminus1=None,
                 Qx=None, QxN=None, Qu=None, QDu=None,
                 xmin=None, xmax=None, umin=None, umax=None, Dumin=None, Dumax=None,
                 xfmin=None, xfmax=None):
        self.Ad, self.Bd = sparse.csc_matrix(Ad), sparse.csc_matrix(Bd)
        self.nx, self.nu = self.Bd.shape
        self.Np = Np
        self.x0 = np.asarray(x0 if x0 is not None else np.zeros(self.nx)).reshape(-1)
        self.uminus1 = np.asarray(uminus1 if uminus1 is not None else np.zeros(self.nu)).reshape(-1)
        self.xref = np.asarray(xref if xref is not None else np.zeros(self.nx)).reshape(-1)
        self.uref = np.asarray(uref if uref is not None else np.zeros(self.nu)).reshape(-1)
        self.Qx = sparse.csc_matrix(Qx if Qx is not None else np.zeros((self.nx, self.nx)))
        self.QxN = sparse.csc_matrix(QxN if QxN is not None else self.Qx)
        self.Qu = sparse.csc_matrix(Qu if Qu is not None else np.zeros((self.nu, self.nu)))
        self.QDu = sparse.csc_matrix(QDu if QDu is not None else np.zeros((self.nu, self.nu)))
        self.xmin = np.asarray(xmin if xmin is not None else -np.inf*np.ones(self.nx)).reshape(-1)
        self.xmax = np.asarray(xmax if xmax is not None else  np.inf*np.ones(self.nx)).reshape(-1)
        self.umin = np.asarray(umin if umin is not None else -np.inf*np.ones(self.nu)).reshape(-1)
        self.umax = np.asarray(umax if umax is not None else  np.inf*np.ones(self.nu)).reshape(-1)
        self.Dumin = np.asarray(Dumin if Dumin is not None else -np.inf*np.ones(self.nu)).reshape(-1)
        self.Dumax = np.asarray(Dumax if Dumax is not None else  np.inf*np.ones(self.nu)).reshape(-1)
        self.xfmin = np.asarray(xfmin if xfmin is not None else self.xmin).reshape(-1)
        self.xfmax = np.asarray(xfmax if xfmax is not None else self.xmax).reshape(-1)
        self.prob = osqp.OSQP()

    @staticmethod
    def _mv(M, v):
        return np.asarray(M @ v).reshape(-1)

    def _build(self):
        Np, nx, nu = self.Np, self.nx, self.nu
        Ad, Bd = self.Ad, self.Bd
        # Cost Hessian over (x_0..x_N, u_0..u_{N-1}).
        P_x = sparse.block_diag(
            [sparse.kron(sparse.eye(Np, format='csc'), self.Qx), self.QxN],
            format='csc')
        D = sparse.eye(Np, format='csc') - sparse.eye(Np, k=-1, format='csc')
        P_u = sparse.kron(sparse.eye(Np, format='csc'), self.Qu) \
            + sparse.kron((D.T @ D).tocsc(), self.QDu)
        P = sparse.block_diag([P_x, P_u], format='csc')
        q_x = np.hstack([np.tile(-self._mv(self.Qx, self.xref), Np),
                         -self._mv(self.QxN, self.xref)])
        q_u = np.tile(-self._mv(self.Qu, self.uref), Np)
        q_u[:nu] += -self._mv(self.QDu, self.uminus1)
        q = np.hstack([q_x, q_u])
        # Dynamics equality Ad x_k + Bd u_k - x_{k+1} = 0 and x_0 = x(t).
        Ax = sparse.kron(sparse.eye(Np+1), -sparse.eye(nx)) \
             + sparse.kron(sparse.eye(Np+1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, Np)), sparse.eye(Np)]), Bd)
        Aeq = sparse.hstack([Ax, Bu]); leq = np.hstack([-self.x0, np.zeros(Np*nx)]); ueq = leq
        # Box constraints on every x_k and u_k; optional terminal box on x_N.
        Aineq = sparse.eye((Np+1)*nx + Np*nu)
        xmin_stack = np.tile(self.xmin, Np+1); xmin_stack[-nx:] = self.xfmin
        xmax_stack = np.tile(self.xmax, Np+1); xmax_stack[-nx:] = self.xfmax
        lineq = np.hstack([xmin_stack, np.tile(self.umin, Np)])
        uineq = np.hstack([xmax_stack, np.tile(self.umax, Np)])
        # Slew-rate constraints Dumin <= u_k - u_{k-1} <= Dumax.
        Adu = sparse.hstack([
            sparse.csc_matrix((Np*nu, (Np+1)*nx)),
            sparse.kron(D, sparse.eye(nu, format='csc'))])
        ldu = np.tile(self.Dumin, Np); ldu[:nu] += self.uminus1
        udu = np.tile(self.Dumax, Np); udu[:nu] += self.uminus1
        A = sparse.vstack([Aeq, Aineq, Adu]).tocsc()
        l = np.hstack([leq, lineq, ldu]); u = np.hstack([ueq, uineq, udu])
        self.P, self.q, self.A, self.l, self.u = P, q, A, l, u
        self._rate0 = (Np+1)*nx + (Np+1)*nx + Np*nu   # index of first rate-bound row

    def setup(self):
        self._build()
        self.prob.setup(self.P, self.q, self.A, self.l, self.u,
                        warm_start=True, verbose=False)

    def step(self):
        res = self.prob.solve()
        if res.info.status != 'solved':
            raise ValueError('QP infeasible / not solved: ' + res.info.status)
        base = (self.Np+1)*self.nx
        u0 = res.x[base:base+self.nu].copy()   # apply only the first input
        self.uminus1 = u0
        return u0

    def update(self, x_meas, u_prev=None):
        self.x0 = np.asarray(x_meas).reshape(-1)
        if u_prev is not None:
            self.uminus1 = np.asarray(u_prev).reshape(-1)
        self.l[:self.nx] = -self.x0; self.u[:self.nx] = -self.x0     # new x_0 = x(t)
        r = self._rate0
        self.l[r:r+self.nu] = self.Dumin + self.uminus1             # u_0 - u_{-1} bound
        self.u[r:r+self.nu] = self.Dumax + self.uminus1
        base = (self.Np+1)*self.nx
        self.q[base:base+self.nu] = -self._mv(self.Qu, self.uref) - self._mv(self.QDu, self.uminus1)
        self.prob.update(l=self.l, u=self.u, q=self.q)


def lqr_terminal(Ad, Bd, Q, R):
    """Terminal cost option: discrete-LQR value matrix S (and gain K). Splicing
    x_N'S x_N onto the horizon makes the finite-horizon cost stand in for the
    infinite-horizon cost, so the receding-horizon loop is stabilizing."""
    S = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad)
    return K, S


def closed_loop(Ad, Bd, ctrl, x0, nsim):
    """Receding horizon: solve, apply first input, step plant, re-measure, repeat."""
    x = x0.copy(); xs, us = [], []
    for _ in range(nsim):
        u = ctrl.step()
        x = Ad @ x + Bd @ u
        ctrl.update(x)
        xs.append(x.copy()); us.append(u.copy())
    return np.array(xs), np.array(us)
```

### Worked example: point mass to a setpoint, with input/state/rate limits

```python
Ts, M, b = 0.2, 2.0, 0.3
Ad = np.array([[1.0, Ts], [0, 1.0 - b/M*Ts]])
Bd = np.array([[0.0], [Ts/M]])
Qx = sparse.diags([0.5, 0.1]); Qu = 2.0*sparse.eye(1); QDu = 10.0*sparse.eye(1)

K, S = lqr_terminal(Ad, Bd, np.array([[0.5, 0], [0, 0.1]]), np.array([[2.0]]))
QxN = sparse.csc_matrix(S)                 # terminal cost = LQR value matrix

ctrl = MPCController(
    Ad, Bd, Np=20, x0=np.array([0.1, 0.2]),
    xref=np.array([7.0, 0.0]), uref=np.array([0.0]), uminus1=np.array([0.0]),
    Qx=Qx, QxN=QxN, Qu=Qu, QDu=QDu,
    xmin=np.array([-10, -10.0]), xmax=np.array([7.0, 10.0]),   # position capped at 7
    umin=np.array([-1.2]), umax=np.array([1.2]),               # bounded force
    Dumin=np.array([-0.2]), Dumax=np.array([0.2]))             # slew-rate limit
ctrl.setup()
xs, us = closed_loop(Ad, Bd, ctrl, np.array([0.1, 0.2]), 100)
```
