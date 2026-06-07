# Differential Dynamic Programming (DDP) and its Gauss–Newton variant iLQR

## Problem

Find the control sequence minimizing a finite-horizon cost over a *nonlinear* discrete-time plant,

    min_{u_0..u_{N-1}}  J = Σ_{i=0}^{N-1} ℓ(x_i, u_i) + ℓ_f(x_N),   s.t.  x_{i+1} = f(x_i, u_i),  x_0 given,

and emit not just an open-loop sequence but a **local feedback law** `K_i` at every step. The linear-quadratic regulator solves this exactly when `f` is linear and `ℓ` quadratic — one backward Riccati sweep gives time-varying feedback gains — but that recursion stops closing the moment `f` is nonlinear, because the value function is no longer quadratic.

## Key idea

Work in deviations `(δx, δu)` around a nominal trajectory `(x̄, ū)`, and locally quadratize the Bellman value function. Define the change in cost-to-go under a perturbation,

    Q(δx, δu) = ℓ(x̄+δx, ū+δu) − ℓ(x̄, ū) + V(f(x̄+δx, ū+δu)) − V(f(x̄, ū)),

and expand to second order using a quadratic model `(V'_x, V'_xx)` of the value at the next step (primes = next step). The coefficients are

    Q_x  = ℓ_x + f_xᵀ V'_x
    Q_u  = ℓ_u + f_uᵀ V'_x
    Q_xx = ℓ_xx + f_xᵀ V'_xx f_x   (+ V'_x · f_xx)
    Q_uu = ℓ_uu + f_uᵀ V'_xx f_u   (+ V'_x · f_uu)
    Q_ux = ℓ_ux + f_uᵀ V'_xx f_x   (+ V'_x · f_ux).

The per-step Bellman minimization over `δu` of this quadratic (positive-definite `Q_uu`) is closed-form:

    δu*(δx) = k + K δx,    k = −Q_uu⁻¹ Q_u   (feedforward),    K = −Q_uu⁻¹ Q_ux   (feedback).

Substituting back propagates the value model one step backward:

    V_x  = Q_x + Kᵀ Q_uu k + Kᵀ Q_u + Q_uxᵀ k
    V_xx = Q_xx + Kᵀ Q_uu K + Kᵀ Q_ux + Q_uxᵀ K     (symmetrized)
    ΔV   = ½ kᵀ Q_uu k + kᵀ Q_u.

This is the LQR backward Riccati pass generalized to time-varying linearizations `f_x, f_u`; LQR is the special case where `f` is linear (so `f_xx = f_ux = f_uu = 0`) and a single pass is exact.

**iLQR vs. full DDP.** The bracketed terms `V'_x · f_xx`, `V'_x · f_uu`, `V'_x · f_ux` are the value gradient contracted with the *dynamics curvature* (rank-three tensors). Keeping them gives a true Newton step on the trajectory — **full DDP**, quadratic local convergence. Dropping them gives the **Gauss–Newton** approximation — **iLQR**: only the Jacobians `f_x, f_u` are needed (much cheaper), `Q_uu = ℓ_uu + f_uᵀ V'_xx f_u` stays positive (semi)definite by construction when `ℓ_uu ≻ 0`, and the step is nearly the full Newton step near the solution.

**Regularization (Levenberg–Marquardt).** Far from the optimum `Q_uu` can lose positive-definiteness. Adding `μI` to the *next-step value Hessian* before pulling it back,

    Q̃_uu = ℓ_uu + f_uᵀ (V'_xx + μI) f_u,    Q̃_ux = ℓ_ux + f_uᵀ (V'_xx + μI) f_x,

penalizes the induced *state* deviation; unlike the simpler control-based `Q_uu + μI`, the feedback gain `K` does not vanish as `μ → ∞` — it instead pulls the new trajectory toward the trusted nominal. `μ` grows fast when the backward pass hits an indefinite `Q̃_uu` and decays toward 0 (snapping to 0 below `μ_min`) on success.

**Forward pass with line search.** A backward pass yields `(k_i, K_i)`; the forward pass re-rolls the *true* nonlinear dynamics under the closed-loop corrected control, with a backtracking step size `0 < α ≤ 1` on the feedforward only:

    x̂_0 = x̄_0,    û_i = ū_i + α k_i + K_i (x̂_i − x̄_i),    x̂_{i+1} = f(x̂_i, û_i).

The step is accepted when the realized reduction matches the model's predicted reduction `ΔJ(α) = α Σ kᵢᵀQ_u + (α²/2) Σ kᵢᵀQ_uu kᵢ`, via `z = [J(ū) − J(û)] / ΔJ(α) > c_1`; otherwise `α` shrinks (and, if even the smallest fails, `μ` grows and the backward pass repeats).

## Algorithm

```
Given nominal (x̄, ū), repeat until cost stops improving:
  1. Forward rollout: integrate f, collect f_x, f_u (and f_xx,f_ux,f_uu if DDP)
     and ℓ_x, ℓ_u, ℓ_xx, ℓ_ux, ℓ_uu along the trajectory.
  2. Backward pass (i = N-1 .. 0): start V_x,V_xx at the terminal cost; form
     the Q-expansion with μ-regularized V'_xx; set k=-Q̃_uu⁻¹Q_u, K=-Q̃_uu⁻¹Q̃_ux;
     update V_x, V_xx, ΔV. If Q̃_uu not PD: increase μ, restart backward pass.
  3. Forward pass: line-search α in (0,1]; û = ū + αk + K(x̂-x̄), re-roll true f;
     accept first α with sufficient realized cost decrease, else shrink α.
  4. On acceptance: decrease μ, set (x̄,ū) ← (x̂,û).
Return the optimized trajectory and the feedback gains {K_i}.
```

## Code

Faithful to the canonical `anassinator/ilqr` controller (Tassa-style state-based regularization, the `hessians` flag toggling iLQR↔DDP, backtracking `α`).

```python
import numpy as np


class iLQR:
    """Finite-horizon iterative LQR / DDP.

    x_{i+1} = f(x_i, u_i); cost = sum_i l(x_i, u_i) + l_f(x_N).
    use_hessians=False -> iLQR (Gauss-Newton, drop dynamics Hessians);
    use_hessians=True  -> full DDP (keep V'_x . f_xx / f_ux / f_uu).
    """

    def __init__(self, dynamics, cost, N, use_hessians=False, max_reg=1e10):
        self.dynamics, self.cost, self.N = dynamics, cost, N
        self.use_hessians = use_hessians and getattr(dynamics, "has_hessians", False)
        # Levenberg-Marquardt schedule.
        self.mu, self.mu_min, self.mu_max = 1.0, 1e-6, max_reg
        self.delta_0, self.delta = 2.0, 2.0

    def fit(self, x0, us, n_iterations=100, tol=1e-6):
        self.mu, self.delta = 1.0, self.delta_0
        alphas = 1.1 ** (-np.arange(10) ** 2)        # backtracking 1 -> ~0
        us = us.copy()
        k = np.zeros((self.N, self.dynamics.action_size))
        K = np.zeros((self.N, self.dynamics.action_size, self.dynamics.state_size))
        changed, converged = True, False

        for _ in range(n_iterations):
            if changed:                              # re-linearize around nominal
                (xs, fx, fu, L, lx, lu, lxx, lux, luu,
                 fxx, fux, fuu) = self._forward_rollout(x0, us)
                J = L.sum()
                changed = False

            accepted = False
            try:
                k, K = self._backward_pass(fx, fu, lx, lu, lxx, lux, luu,
                                           fxx, fux, fuu)
                for alpha in alphas:                 # line search on feedforward
                    xs_new, us_new = self._control(xs, us, k, K, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)
                    if J_new < J:
                        converged = abs((J - J_new) / J) < tol
                        J, xs, us = J_new, xs_new, us_new
                        changed = accepted = True
                        self.delta = min(1.0, self.delta) / self.delta_0
                        self.mu *= self.delta
                        if self.mu <= self.mu_min:
                            self.mu = 0.0
                        break
            except np.linalg.LinAlgError:
                pass                                 # Quu not PD -> failure

            if not accepted:                         # damp harder and retry
                self.delta = max(1.0, self.delta) * self.delta_0
                self.mu = max(self.mu_min, self.mu * self.delta)
                if self.mu >= self.mu_max:
                    break
            if converged:
                break

        self._k, self._K = k, K
        return xs, us

    def _Q(self, fx, fu, lx, lu, lxx, lux, luu, Vx, Vxx,
           fxx=None, fux=None, fuu=None):
        Q_x = lx + fx.T @ Vx                          # l_x + f_x' V'_x
        Q_u = lu + fu.T @ Vx                          # l_u + f_u' V'_x
        Q_xx = lxx + fx.T @ Vxx @ fx                  # l_xx + f_x' V'_xx f_x
        reg = self.mu * np.eye(Vxx.shape[0])          # damp V'_xx (state-based)
        Q_ux = lux + fu.T @ (Vxx + reg) @ fx          # l_ux + f_u'(V'_xx+muI) f_x
        Q_uu = luu + fu.T @ (Vxx + reg) @ fu          # l_uu + f_u'(V'_xx+muI) f_u
        if self.use_hessians:                         # full DDP: dynamics curvature
            Q_xx += np.tensordot(Vx, fxx, axes=1)     # + V'_x . f_xx
            Q_ux += np.tensordot(Vx, fux, axes=1)     # + V'_x . f_ux
            Q_uu += np.tensordot(Vx, fuu, axes=1)     # + V'_x . f_uu
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def _backward_pass(self, fx, fu, lx, lu, lxx, lux, luu,
                       fxx=None, fux=None, fuu=None):
        Vx, Vxx = lx[-1], lxx[-1]                     # terminal value = final cost
        k = np.empty((self.N, lu.shape[1]))
        K = np.empty((self.N, lu.shape[1], lx.shape[1]))
        for i in range(self.N - 1, -1, -1):
            extra = (fxx[i], fux[i], fuu[i]) if self.use_hessians else ()
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(
                fx[i], fu[i], lx[i], lu[i], lxx[i], lux[i], luu[i], Vx, Vxx, *extra)
            k[i] = -np.linalg.solve(Q_uu, Q_u)        # feedforward -Q_uu^{-1} Q_u
            K[i] = -np.linalg.solve(Q_uu, Q_ux)       # feedback    -Q_uu^{-1} Q_ux
            Vx = Q_x + K[i].T @ Q_uu @ k[i] + K[i].T @ Q_u + Q_ux.T @ k[i]
            Vxx = Q_xx + K[i].T @ Q_uu @ K[i] + K[i].T @ Q_ux + Q_ux.T @ K[i]
            Vxx = 0.5 * (Vxx + Vxx.T)                 # keep symmetric
        return k, K

    def _control(self, xs, us, k, K, alpha):
        # Forward pass: re-roll TRUE dynamics under closed-loop corrected control.
        xs_new = np.zeros_like(xs); xs_new[0] = xs[0]
        us_new = np.zeros_like(us)
        for i in range(self.N):
            us_new[i] = us[i] + alpha * k[i] + K[i] @ (xs_new[i] - xs[i])
            xs_new[i + 1] = self.dynamics.f(xs_new[i], us_new[i], i)
        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        c = sum(self.cost.l(xs[i], us[i], i) for i in range(self.N))
        return c + self.cost.l(xs[-1], None, self.N, terminal=True)

    def _forward_rollout(self, x0, us):
        N, ns, na = self.N, self.dynamics.state_size, self.dynamics.action_size
        xs = np.empty((N + 1, ns)); xs[0] = x0
        fx = np.empty((N, ns, ns)); fu = np.empty((N, ns, na))
        L = np.empty(N + 1)
        lx = np.empty((N + 1, ns)); lu = np.empty((N, na))
        lxx = np.empty((N + 1, ns, ns)); lux = np.empty((N, na, ns)); luu = np.empty((N, na, na))
        if self.use_hessians:
            fxx = np.empty((N, ns, ns, ns)); fux = np.empty((N, ns, na, ns)); fuu = np.empty((N, ns, na, na))
        else:
            fxx = fux = fuu = None
        for i in range(N):
            x, u = xs[i], us[i]
            xs[i + 1] = self.dynamics.f(x, u, i)
            fx[i], fu[i] = self.dynamics.f_x(x, u, i), self.dynamics.f_u(x, u, i)
            L[i] = self.cost.l(x, u, i)
            lx[i], lu[i] = self.cost.l_x(x, u, i), self.cost.l_u(x, u, i)
            lxx[i], lux[i], luu[i] = self.cost.l_xx(x, u, i), self.cost.l_ux(x, u, i), self.cost.l_uu(x, u, i)
            if self.use_hessians:
                fxx[i], fux[i], fuu[i] = self.dynamics.f_xx(x, u, i), self.dynamics.f_ux(x, u, i), self.dynamics.f_uu(x, u, i)
        x = xs[-1]
        L[-1] = self.cost.l(x, None, N, terminal=True)
        lx[-1] = self.cost.l_x(x, None, N, terminal=True)
        lxx[-1] = self.cost.l_xx(x, None, N, terminal=True)
        return xs, fx, fu, L, lx, lu, lxx, lux, luu, fxx, fux, fuu
```

For online / model-predictive use, the solver is wrapped in a receding-horizon loop: solve from the current state, apply the first control `û_0`, re-measure, warm-start from the shifted previous solution, and re-solve — turning the trajectory optimizer into a feedback policy.
