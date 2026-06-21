I am going to describe the control method I would use when the plant has hard limits that an ordinary linear-quadratic regulator cannot respect. The canonical name is Model Predictive Control, or MPC, and the version I have in mind is the linear constrained receding-horizon controller. It keeps the quadratic cost that makes LQR elegant, but replaces the closed-form gain with a per-step quadratic program that enforces actuator saturation, slew-rate limits, and state or output bounds directly.

The starting point is the same discrete-time linear model that LQR assumes: x_{k+1} = A x_k + B u_k. For the unconstrained regulation problem with cost sum_k (x_k' Q x_k + u_k' R u_k), dynamic programming gives the familiar linear law u = -K x, where K comes from the discrete algebraic Riccati equation and the value function x' S x is a Lyapunov function for the closed loop. That law is optimal and stabilizing, but every step of its proof assumes the input -K x is applied exactly. In a real plant the actuator clips anything outside its range, the applied input is no longer -K x, and the optimality and stability guarantees vanish. Worse, LQR has no notion of deliberately riding a constraint; it cannot plan to keep a controlled variable just below a safety limit, which is exactly where many process plants make money.

My response is to stop asking for a global closed-form policy and instead solve, from the current measured state, a finite-horizon constrained optimal control problem. At time t I choose a whole sequence of future inputs u_0, ..., u_{N-1} to minimize the cost over the next N steps subject to the model and the hard limits. Even though I only ever apply the first input, the later inputs matter because they price delayed consequences and let me satisfy future state constraints before I reach them. After applying u_0 I measure the new true state and solve again. This receding-horizon mechanism is what closes the loop: re-solving from the measured state makes the applied input a function of the current state, which gives feedback, disturbance rejection, and robustness to model error.

The optimization is a convex quadratic program. The cost is quadratic in the decision variables, the dynamics are linear equalities, and the input and state boxes are linear inequalities. With R positive definite the Hessian is strictly positive definite, so the QP has a unique minimizer and can be solved reliably by standard codes. There are two equivalent ways to write it. The condensed form eliminates the states using the prediction equation X = S^x x(t) + S^u U_0, leaving a dense QP in the input sequence alone. The sparse banded form keeps the states and inputs as decision variables and writes the dynamics as equality constraints; it is larger but sparser and matches the structure that solvers like OSQP exploit. In the region where no constraint is active, the first input of the QP solution reproduces the LQR gain. So MPC is not a rejection of LQR; it is LQR with saturations folded in.

Truncating the horizon is not free. A finite horizon, re-solved repeatedly, is not automatically feasible or stabilizing the way infinite-horizon LQR is. First, feasibility now does not imply feasibility forever. A short-sighted optimizer can let the state drift into a region where, given bounded inputs, no admissible sequence keeps it inside the box, and the next QP becomes infeasible. The fix is a terminal constraint x_N in X_f with X_f control invariant under admissible inputs. A one-step shift of the previous optimal sequence, appended with an admissible terminal move, then produces a feasible candidate for the next problem, so feasibility propagates forward. Second, stability requires that the finite-horizon cost behave like the infinite-horizon cost. The right terminal cost is the LQR value function x_N' S x_N, where S solves the discrete algebraic Riccati equation. With this choice the terminal inequality -x' S x + x' (Q + K' R K) x + x' (A - B K)' S (A - B K) x = 0 holds with equality, so the optimal QP value J* is a Lyapunov function and the closed loop is asymptotically stable on the feasible set. Intuitively, MPC becomes a constrained head spliced onto an unconstrained LQR tail.

In practice I also penalize input increments Delta u_k = u_k - u_{k-1} with a weight Q_Du. This damps aggressiveness, improves robustness to model error, conditions the QP numerically, and lets slew-rate limits enter as linear constraints. The last applied input u_{-1} is carried as an extra parameter, exactly like the current state. Input limits stay hard because they are physical, while state or output limits can be softened with signed slack variables and a large penalty so a disturbance cannot kill the controller at the worst moment. Consecutive QPs differ only in the right-hand side through the new x(t) and u_{-1}, so the same solver setup is reused with a warm start.

The code block below implements the sparse OSQP formulation for a constrained point-mass plant. It builds the decision vector (x_0, ..., x_N, u_0, ..., u_{N-1}), writes the dynamics as equality constraints, the state and input boxes as bound constraints, the slew-rate limits as difference constraints, and applies only the first input of the solution before re-solving from the new measured state. The terminal cost is set to the discrete LQR value matrix, making the finite-horizon objective stand in for the infinite-horizon one.

```python
import numpy as np
import scipy.sparse as sparse
import scipy.linalg
import osqp


class MPCController:
    """Linear constrained receding-horizon controller (sparse OSQP form).

    Plant: x_{k+1} = Ad x_k + Bd u_k.
    Each step: solve the finite-horizon QP from the measured state,
    apply only the first input, re-measure, re-solve.
    """
    def __init__(self, Ad, Bd, Np=20, x0=None, xref=None, uref=None, uminus1=None,
                 Qx=None, QxN=None, Qu=None, QDu=None,
                 xmin=None, xmax=None, umin=None, umax=None,
                 Dumin=None, Dumax=None, xfmin=None, xfmax=None):
        self.Ad = sparse.csc_matrix(Ad)
        self.Bd = sparse.csc_matrix(Bd)
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
        self.xmin = np.asarray(xmin if xmin is not None else -np.inf * np.ones(self.nx)).reshape(-1)
        self.xmax = np.asarray(xmax if xmax is not None else np.inf * np.ones(self.nx)).reshape(-1)
        self.umin = np.asarray(umin if umin is not None else -np.inf * np.ones(self.nu)).reshape(-1)
        self.umax = np.asarray(umax if umax is not None else np.inf * np.ones(self.nu)).reshape(-1)
        self.Dumin = np.asarray(Dumin if Dumin is not None else -np.inf * np.ones(self.nu)).reshape(-1)
        self.Dumax = np.asarray(Dumax if Dumax is not None else np.inf * np.ones(self.nu)).reshape(-1)
        self.xfmin = np.asarray(xfmin if xfmin is not None else self.xmin).reshape(-1)
        self.xfmax = np.asarray(xfmax if xfmax is not None else self.xmax).reshape(-1)
        self.prob = osqp.OSQP()

    @staticmethod
    def _mv(M, v):
        return np.asarray(M @ v).reshape(-1)

    def _build(self):
        Np, nx, nu = self.Np, self.nx, self.nu
        Ad, Bd = self.Ad, self.Bd
        # Hessian over (x_0..x_N, u_0..u_{N-1}).
        P_x = sparse.block_diag(
            [sparse.kron(sparse.eye(Np, format='csc'), self.Qx), self.QxN],
            format='csc')
        D = sparse.eye(Np, format='csc') - sparse.eye(Np, k=-1, format='csc')
        P_u = (sparse.kron(sparse.eye(Np, format='csc'), self.Qu)
               + sparse.kron((D.T @ D).tocsc(), self.QDu))
        P = sparse.block_diag([P_x, P_u], format='csc')
        q_x = np.hstack([np.tile(-self._mv(self.Qx, self.xref), Np),
                         -self._mv(self.QxN, self.xref)])
        q_u = np.tile(-self._mv(self.Qu, self.uref), Np)
        q_u[:nu] += -self._mv(self.QDu, self.uminus1)
        q = np.hstack([q_x, q_u])
        # Dynamics equality and initial condition.
        Ax = (sparse.kron(sparse.eye(Np + 1), -sparse.eye(nx))
              + sparse.kron(sparse.eye(Np + 1, k=-1), Ad))
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, Np)), sparse.eye(Np)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-self.x0, np.zeros(Np * nx)])
        ueq = leq
        # Box constraints on states and inputs; terminal box on x_N.
        Aineq = sparse.eye((Np + 1) * nx + Np * nu)
        xmin_stack = np.tile(self.xmin, Np + 1)
        xmin_stack[-nx:] = self.xfmin
        xmax_stack = np.tile(self.xmax, Np + 1)
        xmax_stack[-nx:] = self.xfmax
        lineq = np.hstack([xmin_stack, np.tile(self.umin, Np)])
        uineq = np.hstack([xmax_stack, np.tile(self.umax, Np)])
        # Slew-rate constraints.
        Adu = sparse.hstack([
            sparse.csc_matrix((Np * nu, (Np + 1) * nx)),
            sparse.kron(D, sparse.eye(nu, format='csc'))])
        ldu = np.tile(self.Dumin, Np)
        udu = np.tile(self.Dumax, Np)
        ldu[:nu] += self.uminus1
        udu[:nu] += self.uminus1
        self.P, self.q = P, q
        self.A = sparse.vstack([Aeq, Aineq, Adu]).tocsc()
        self.l = np.hstack([leq, lineq, ldu])
        self.u = np.hstack([ueq, uineq, udu])
        self._rate0 = (Np + 1) * nx + (Np + 1) * nx + Np * nu

    def setup(self):
        self._build()
        self.prob.setup(self.P, self.q, self.A, self.l, self.u,
                        warm_start=True, verbose=False)

    def step(self):
        res = self.prob.solve()
        if res.info.status != 'solved':
            raise ValueError('QP not solved: ' + res.info.status)
        base = (self.Np + 1) * self.nx
        u0 = res.x[base:base + self.nu].copy()
        self.uminus1 = u0
        return u0

    def update(self, x_meas, u_prev=None):
        self.x0 = np.asarray(x_meas).reshape(-1)
        if u_prev is not None:
            self.uminus1 = np.asarray(u_prev).reshape(-1)
        self.l[:self.nx] = -self.x0
        self.u[:self.nx] = -self.x0
        r = self._rate0
        self.l[r:r + self.nu] = self.Dumin + self.uminus1
        self.u[r:r + self.nu] = self.Dumax + self.uminus1
        base = (self.Np + 1) * self.nx
        self.q[base:base + self.nu] = (-self._mv(self.Qu, self.uref)
                                        - self._mv(self.QDu, self.uminus1))
        self.prob.update(l=self.l, u=self.u, q=self.q)


def lqr_terminal(Ad, Bd, Q, R):
    """Return the discrete LQR value matrix S and gain K."""
    S = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad)
    return K, S


def closed_loop(Ad, Bd, ctrl, x0, nsim):
    """Receding-horizon simulation: solve, apply first input, step, repeat."""
    x = x0.copy()
    xs, us = [], []
    for _ in range(nsim):
        u = ctrl.step()
        x = Ad @ x + Bd @ u
        ctrl.update(x)
        xs.append(x.copy())
        us.append(u.copy())
    return np.array(xs), np.array(us)


# Demonstration: point mass with bounded force, position cap, and slew-rate limit.
if __name__ == '__main__':
    Ts, M, b = 0.2, 2.0, 0.3
    Ad = np.array([[1.0, Ts], [0.0, 1.0 - b / M * Ts]])
    Bd = np.array([[0.0], [Ts / M]])
    Qx = sparse.diags([0.5, 0.1])
    Qu = 2.0 * sparse.eye(1)
    QDu = 10.0 * sparse.eye(1)
    K, S = lqr_terminal(Ad, Bd, np.array([[0.5, 0.0], [0.0, 0.1]]),
                        np.array([[2.0]]))
    QxN = sparse.csc_matrix(S)
    ctrl = MPCController(
        Ad, Bd, Np=20, x0=np.array([0.1, 0.2]),
        xref=np.array([7.0, 0.0]), uref=np.array([0.0]), uminus1=np.array([0.0]),
        Qx=Qx, QxN=QxN, Qu=Qu, QDu=QDu,
        xmin=np.array([-10.0, -10.0]), xmax=np.array([7.0, 10.0]),
        umin=np.array([-1.2]), umax=np.array([1.2]),
        Dumin=np.array([-0.2]), Dumax=np.array([0.2]))
    ctrl.setup()
    xs, us = closed_loop(Ad, Bd, ctrl, np.array([0.1, 0.2]), 100)
    print('final state:', xs[-1])
    print('max position:', xs[:, 0].max())
    print('max |input|:', np.abs(us).max())
```
