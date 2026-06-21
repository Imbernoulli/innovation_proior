We want to compute the control sequence $u_0,\dots,u_{N-1}$ that minimizes a finite-horizon cost $J = \sum_{i=0}^{N-1}\ell(x_i,u_i) + \ell_f(x_N)$ subject to a discrete-time *nonlinear* plant $x_{i+1} = f(x_i,u_i)$ with $x_0$ given — and not just an open-loop sequence, but a control with a local feedback gain $K_i$ attached at every step, so the plant can be steered back onto the optimized trajectory after it is bumped off. For the linear plant with quadratic cost this is settled and beautiful: the value function $V(x,i)=x'S_i x$ stays quadratic, Bellman's principle of optimality collapses the minimization over the whole sequence into a backward sweep of single-control minimizations, each minimization is a strictly convex unconstrained quadratic in $u$ (because $R\succ 0$), and setting its gradient to zero gives linear feedback $u_i^* = -K_i x_i$ together with the backward Riccati recursion for $S_i$ — one sweep from $S_N=Q_f$ yields every gain, exactly, with feedback baked in. The property that makes it work is *quadratic-in, quadratic-out*: the value model survives the backward step in finite parametric form.

The moment $f$ is nonlinear, that closure dies. $V(f(x,u),i+1)$ is some arbitrary nonlinear function of $(x,u)$, the bracket inside Bellman's min is no longer a quadratic I can minimize in closed form, and there is no finite matrix $S_i$ to propagate; the machine seizes. The brute-force escape — stack the whole control sequence into one vector, eliminate the states by rolling out $f$, and hand $J$ to a generic nonlinear-programming solver — does find a local minimum, but it throws away the two things I most wanted to keep: it discards the stage-wise, backward-in-time Markov decomposition that made the linear case a single sweep, and it returns an *open-loop* sequence with no $K_i$, so the output is a trajectory, not a controller. The dynamic-programming view handed me $K_i$ for free as the by-product of the per-step minimization, and I refuse to lose it. The costate/indirect-shooting route is no better: linearizing the Hamiltonian's necessary conditions gives an open-loop correction tied to one boundary condition, and a feedback form only reappears after positing an affine costate ansatz $\delta\lambda_i = S_i\,\delta x_i + v_i$ and grinding the matrix-inversion lemma — and because it linearizes $f$, it is only first-order in the dynamics. The real task is to keep the backward-recursion machinery of the linear-quadratic regulator and make it survive a nonlinear $f$.

I propose Differential Dynamic Programming and its Gauss–Newton variant iterative LQR (iLQR). The lever is that I do not need a *global* quadratic value function — I only need a local one. I almost always have, or can cheaply roll out, a nominal trajectory $(\bar x,\bar u)$, even a bad one. So work in deviations $\delta x_i = x_i - \bar x_i$, $\delta u_i = u_i - \bar u_i$, and ask only for the best *correction*. The object the Bellman step actually minimizes is the bracket, so I expand *that*, measured as a deviation from its nominal value:
$$Q(\delta x,\delta u) = \ell(\bar x+\delta x,\bar u+\delta u) - \ell(\bar x,\bar u) + V(f(\bar x+\delta x,\bar u+\delta u),i{+}1) - V(f(\bar x,\bar u),i{+}1).$$
Given a quadratic model $(V'_x, V'_{xx})$ of the value at the next step (primes denote next step), Taylor-expand to second order. The cost part is immediate. For the value term, let $\delta f = f(\bar x+\delta x,\bar u+\delta u)-f(\bar x,\bar u)$; then $V(\bar f+\delta f)-V(\bar f)\approx V'_x\cdot\delta f + \tfrac12\delta f' V'_{xx}\delta f$, with $\delta f \approx f_x\,\delta x + f_u\,\delta u + \tfrac12(\delta x,\delta u)\cdot\partial^2 f\cdot(\delta x,\delta u)$. The first-order-in-$\delta f$ term contributes $V'_x\cdot(f_x\delta x + f_u\delta u)$ at first order and $\tfrac12 V'_x\cdot(\partial^2 f)$ — the value gradient dotted into the dynamics curvature — at second order; the $\tfrac12\delta f'V'_{xx}\delta f$ term contributes $\tfrac12(f_x\delta x+f_u\delta u)'V'_{xx}(f_x\delta x+f_u\delta u)$. Matching coefficients against the block form of $Q$ reads off every coefficient:
$$Q_x = \ell_x + f_x' V'_x,\qquad Q_u = \ell_u + f_u' V'_x,$$
$$Q_{xx} = \ell_{xx} + f_x' V'_{xx} f_x \;(+\, V'_x\!\cdot f_{xx}),\quad Q_{uu} = \ell_{uu} + f_u' V'_{xx} f_u \;(+\, V'_x\!\cdot f_{uu}),\quad Q_{ux} = \ell_{ux} + f_u' V'_{xx} f_x \;(+\, V'_x\!\cdot f_{ux}).$$
Each piece is the chain rule made literal: $Q_x$ is the immediate cost change plus where the nudge lands ($f_x\delta x$) costed by $V'_x$; $f_x'V'_{xx}f_x$ is the value's curvature pulled back through the linearized dynamics; and the cross term $f_u'V'_{xx}f_x$ in $Q_{ux}$ is exactly what couples the optimal control correction to the state deviation — this is where feedback comes from.

Now the Bellman step: minimize $Q(\delta x,\delta u)$ over $\delta u$ for given $\delta x$. With $Q_{uu}$ positive definite this is the clean convex quadratic of the LQR case; setting $\partial Q/\partial\delta u = Q_u + Q_{uu}\delta u + Q_{ux}\delta x = 0$ gives
$$\delta u^*(\delta x) = k + K\,\delta x,\qquad k = -Q_{uu}^{-1}Q_u,\qquad K = -Q_{uu}^{-1}Q_{ux}.$$
The feedforward $k$ is the Newton-step-in-the-control on the nominal — it improves the trajectory itself and is independent of $\delta x$; the feedback gain $K$ says how to adjust the control in response to a state deviation. I did not insert feedback by hand: it dropped out of the cross term $Q_{ux}$, exactly as $K_i$ dropped out in the Riccati derivation, and the minus sign is forced (negative feedback, descending the quadratic). Substituting $\delta u^* = k + K\delta x$ back into $Q$ propagates the value model one step backward, staying quadratic-in/quadratic-out so the recursion closes:
$$V_x = Q_x + K'Q_{uu}k + K'Q_u + Q_{ux}'k,\qquad V_{xx} = Q_{xx} + K'Q_{uu}K + K'Q_{ux} + Q_{ux}'K \text{ (symmetrized)},$$
with the running predicted decrease $\Delta V = \tfrac12 k'Q_{uu}k + k'Q_u$ and the sweep initialized at the terminal value $V_x(N)=\ell_{f,x}$, $V_{xx}(N)=\ell_{f,xx}$. When $f$ is linear ($f_x=A$, $f_u=B$, all $\partial^2 f=0$) and $\ell$ quadratic, this reduces exactly to the discrete Riccati pass and, the model then being exact, converges in a single iteration — LQR is the special case, and the nonlinear problem has become a sequence of time-varying LQR subproblems re-linearized as the trajectory moves.

The bracketed curvature terms $V'_x\cdot f_{xx}$, $V'_x\cdot f_{uu}$, $V'_x\cdot f_{ux}$ are the crux. Keeping them makes the update a genuine second-order Newton step on the trajectory with quadratic local convergence — this is full DDP. But two problems argue against keeping them by default. First, cost: $f_{xx}$ is an $n\times n\times n$ tensor per step (likewise $f_{ux}$, $f_{uu}$), and for the high-DOF plants I care about — a multi-link arm, a humanoid — forming and contracting these rank-three tensors blows up the dominant cost of computing dynamics derivatives. Second, sign: $V'_x\cdot f_{uu}$ has no definite sign and can destroy the positive-definiteness of $Q_{uu}$ far from the optimum where $V'_x$ is large, turning $-Q_{uu}^{-1}Q_u$ into an ascent toward a saddle. So the iLQR trade is to drop all three tensor terms:
$$Q_{xx} = \ell_{xx} + f_x'V'_{xx}f_x,\qquad Q_{uu} = \ell_{uu} + f_u'V'_{xx}f_u,\qquad Q_{ux} = \ell_{ux} + f_u'V'_{xx}f_x.$$
This is precisely the *Gauss–Newton* approximation to the Hessian — keep the first-derivative outer products, discard the residual-times-second-derivative term — and it buys three things at once: it needs only the Jacobians $f_x, f_u$; it keeps $Q_{uu}$ positive (semi)definite by construction whenever $\ell_{uu}\succ 0$ and $V'_{xx}\succeq 0$ (the congruence $f_u'V'_{xx}f_u$ is PSD only then), so the inverse exists and $\delta u^*$ is a real minimizer; and near a small-residual solution the dropped term is small, so the step is nearly the full Newton step and convergence stays fast. Full DDP remains a switchable option for problems where the dynamics curvature genuinely matters and stays benign.

Two failure modes remain, and I import the standard second-order-optimization fixes rather than invent anything. First, far from the optimum $Q_{uu}$ can still lose positive-definiteness. The textbook move is Levenberg–Marquardt damping, but the naive choice $Q_{uu}+\mu I$ is the wrong currency: it penalizes $\delta u$ uniformly even though a unit control change has very different effect depending on how strongly $f_u$ enters the dynamics, and worse, as $\mu\to\infty$ the gain $K=-(Q_{uu}+\mu I)^{-1}Q_{ux}\to 0$ — the controller stops correcting state deviations exactly when it is being most cautious, throwing away the feedback. What I actually want to penalize is *state* deviation from the trusted nominal, so I add $\mu I$ to the next-step value Hessian *before* pulling it back through the dynamics:
$$\tilde Q_{uu} = \ell_{uu} + f_u'(V'_{xx}+\mu I)f_u\;(+\,V'_x\!\cdot f_{uu}),\qquad \tilde Q_{ux} = \ell_{ux} + f_u'(V'_{xx}+\mu I)f_x\;(+\,V'_x\!\cdot f_{ux}),$$
with $k = -\tilde Q_{uu}^{-1}Q_u$, $K = -\tilde Q_{uu}^{-1}\tilde Q_{ux}$. This is a penalty $\mu\,\lVert f_x\delta x + f_u\delta u\rVert^2$ in disguise — on the induced state deviation, in the natural metric the dynamics impose — and as $\mu\to\infty$, $K$ does *not* vanish; instead the step is forced to keep the new trajectory close to the old one in state space while retaining feedback. Because I perturb $V'_{xx}\to V'_{xx}+\mu I$ inside the $Q$ terms, the value update must be computed from the $k$/$K$ form above rather than the compact $V_x = Q_x - Q_{xu}Q_{uu}^{-1}Q_u$, whose cancellations no longer hold once $Q_{uu}$ is perturbed. The schedule grows $\mu$ fast on an indefinite $\tilde Q_{uu}$ and decays it toward zero on success (snapping to $0$ below a floor $\mu_{\min}$), using a growing geometric factor $\delta$.

The second failure mode: the backward pass succeeds but the *full* step overshoots, leaving the linearization's region of validity so the true cost rises or the rollout diverges. This is the line-search situation. I introduce a backtracking step size $0<\alpha\le 1$ on the *feedforward only*, and re-roll the *true* nonlinear dynamics under the closed-loop corrected control:
$$\hat x_0 = \bar x_0,\qquad \hat u_i = \bar u_i + \alpha\,k_i + K_i(\hat x_i - \bar x_i),\qquad \hat x_{i+1} = f(\hat x_i,\hat u_i).$$
Scaling only $\alpha k_i$ means $\alpha=0$ leaves the trajectory unchanged while $\alpha=1$ is the full step, and keeping $K$ at full strength lets the feedback re-aim the control at the states actually being visited even when the feedforward is shortened — which is precisely why convergence is fast and why rolling through the *true* $f$ (not the linear model) gives the gains their meaning. Acceptance compares realized to predicted reduction: the model's signed cost change is $\Delta J(\alpha) = \alpha\sum_i k_i'Q_u(i) + (\alpha^2/2)\sum_i k_i'Q_{uu}(i)k_i$, negative on a descent step (since $k=-Q_{uu}^{-1}Q_u$ makes $k'Q_u<0$ dominate), so the predicted reduction is $-\Delta J(\alpha)>0$; I form the ratio $z = [J(\bar u)-J(\hat u)]/(-\Delta J(\alpha))$ and accept only if $z$ exceeds a small constant $c_1$ (Armijo sufficient decrease). If $z$ fails or the rollout diverged, shrink $\alpha$; if even the smallest $\alpha$ fails, the backward model was untrustworthy, so bump $\mu$ and redo the backward pass. The full loop: from a nominal, roll forward gathering the per-step Jacobians (and dynamics Hessians only for DDP) and cost derivatives; run the backward pass for $k_i, K_i$ and predicted $\Delta J$, restarting with larger $\mu$ if $\tilde Q_{uu}$ goes indefinite; run the forward pass with line search, accepting the first $\alpha$ that passes; on acceptance, decrease $\mu$ and take the new trajectory as nominal; repeat until the cost stops improving, re-linearizing each iteration so the sequence of LQR subproblems chases the moving nonlinear optimum. For online use, the solver is wrapped in a receding-horizon loop: solve from the current state, apply $\hat u_0$, re-measure, warm-start from the shifted previous solution, and re-solve.

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
