A motion planner hands me a fixed geometric path for a robot arm — a curve $q(s)$, $s\in[0, s_{\mathrm{end}}]$, through configuration space that is already collision-free and respects the workspace — and the only thing left to decide is the *timing*: the increasing scalar function $s(t)$ that says where along the curve the arm sits at each instant, with the realized motion being $q(s(t))$. For a 7-DOF arm running the same welding seam or pick-and-place loop thousands of times a day, that timing *is* the productivity, so I want the minimum-time time-scaling: traverse the given path as fast as physically possible without ever saturating an actuator. The motors impose hard limits — each joint has a torque ceiling $\tau_{\min}\le\tau\le\tau_{\max}$, often joined by direct joint-velocity and joint-acceleration bounds — and a good solution must find the fastest $s(t)$ that respects them everywhere, correctly report when a path is not dynamically traversable at all, and be fast and reliable enough to run online inside a kinodynamic planner that may call it thousands of times.

The landscape offers two unsatisfying poles. The founding numerical-integration phase-plane method projects the dynamics onto the path, derives the maximum- and minimum-acceleration fields $\beta$ and $\alpha$, and builds the optimal velocity profile by integrating forward along $\beta$ and backward along $\alpha$ while hunting for the switch points where the profile kisses the Maximum Velocity Curve from below. It is fast — it computes the bang-bang controls rather than searching for them — but it is fragile: at *zero-inertia* points, where some constraint's $s̈$-coefficient $a_i(s)$ vanishes, both $\alpha_i$ and $\beta_i$ carry $a_i$ in the denominator and the fields *diverge*, so the acceleration to use becomes a direction-dependent $0/0$ ambiguity, and picking it slightly wrong makes the integrated profile oscillate, cross the MVC, and spuriously declare failure. These dynamic singularities are common in real arm motions and are, by every account, the number-one cause of failure in such implementations; the careful machinery that fixes them runs to thousands of lines and still has edge cases. The opposite pole sets $x:=ṡ^2$, discretizes, and writes the whole problem as one large convex program over all grid variables — robust, simple, 100% success — but with $O(N)$ variables and $O(mN)$ inequalities it costs about $O(KmN^3)$, an order of magnitude too slow for an online subroutine. I want both fast and robust, which neither delivers.

I propose Time-Optimal Path Parameterization based on Reachability Analysis, TOPP-RA. The starting point is structural: fixing the geometry collapses everything onto the single scalar $s(t)$, and the chain rule

$$q̇ = q'(s)\,ṡ,\qquad q̈ = q'(s)\,s̈ + q''(s)\,ṡ^2$$

substituted into the manipulator dynamics $M(q)q̈ + q̇^\top C(q)q̇ + g(q) = \tau$ makes the torque *linear in $s̈$ and linear in $ṡ^2$*:

$$\tau = a(s)\,s̈ + b(s)\,ṡ^2 + c(s),\qquad a = M q',\quad b = M q'' + q'^\top C q',\quad c = g.$$

Every nonlinearity — the inertia matrix, the Coriolis tensor — is absorbed into the path-only coefficient functions, so each actuator limit becomes a per-position linear inequality $a_i(s)\,s̈ + b_i(s)\,ṡ^2 + c_i(s)\le 0$ (velocity and acceleration bounds drop into the same template), and the two-sided torque box stacks into one-sided rows. Because the traversal time $T=\int_0^{s_{\mathrm{end}}} ds/ṡ$ has $1/ṡ$ decreasing in $ṡ$ with *no coupling across positions*, the minimum-time problem reduces to a purely pointwise one: drive $ṡ$ as high as the constraints allow at every $s$. The optimal profile therefore rides as high as feasible under the Maximum Velocity Curve — the locus where the admissible-$s̈$ interval $[\alpha,\beta]$ pinches shut as $ṡ$ grows — with the textbook bang-bang shape of a min-time problem whose control enters linearly: accelerate maximally, decelerate maximally, switching on the MVC.

What makes TOPP-RA fast *and* robust is refusing to realize that bang-bang structure by integrating divergent fields. Instead I look at the *discretized* problem. Discretize $s$ into $N$ segments; on segment $[s_i,s_{i+1}]$ hold $u_i=s̈$ constant and let $x_i=ṡ_i^2$ be the squared velocity at grid point $i$. From $d(ṡ^2)/ds = 2s̈$ with constant $s̈$, $ṡ^2$ is exactly linear in $s$ across the segment, giving

$$x_{i+1} = x_i + 2\Delta_i u_i,\qquad \Delta_i = s_{i+1}-s_i,$$

an *exactly* linear state update — no approximation in the relation itself — while the per-stage constraint $a_i u_i + b_i x_i + c_i \in C_i$ (the torque polytope $Fw\le g$) is linear in $(u_i, x_i)$. So the discretized TOPP is a discrete-time *linear* system with *linear* control–state constraints, and a scalar state. That is exactly the setting of set-membership control — the reachability/controllability theory from model-predictive control — where, for a linear system with polytopic constraints, the reachable and controllable sets are themselves polytopes; and because here the state is one-dimensional, those polytopes are mere **intervals**, two numbers apiece, computable by two tiny optimizations. This is the bridge between fast and robust: I borrow reachability, but each per-step computation is microscopic because the state is 1D.

Concretely, the $i$-stage *controllable set* $K_i$ is the set of squared velocities at stage $i$ from which an admissible control sequence can steer the system to the prescribed goal velocity at stage $N$. I build it backward: $K_N=\{ṡ_N^2\}$, and to step from $K_{i+1}$ back to $K_i$ I take the one-step set $Q_i(I)=\{x\in X_i:\exists\,u\in U_i(x),\ x+2\Delta_i u\in I\}$. Since the admissible polygon $\Omega_i=\{(u,x):a_i u+b_i x+c_i\in C_i\}$ is a polytope, $I$ is an interval, and $(u,x)\mapsto x+2\Delta_i u$ is linear, $Q_i(I)$ is again an interval, so I only need its two endpoints, each a two-variable LP:

$$x^{+} = \max\, x,\qquad x^{-} = \min\, x\quad \text{s.t. } a_i u + b_i x + c_i\in C_i\ \text{ and }\ x + 2\Delta_i u\in I.$$

Two variables, $m+2$ inequalities — a simplex or active-set solver cracks that in microseconds — and the backward sweep is $2N$ such LPs, $O(mN)$ total. If any $K_i$ comes out empty the path is not parameterizable from there; if $K_0$ is empty or $ṡ_0^2\notin K_0$ the instance is infeasible, and I get that diagnosis for free with no trap-point analysis. The MVC's switch points are now *implicit* in how the controllable-set endpoints bend; I never locate them, never evaluate a field at a singularity. Then the forward pass extracts the profile greedily, which is exactly what the pointwise cost demands: from $x^*_0=ṡ_0^2$, at each stage take the *highest* control that is admissible and keeps the next state inside the next controllable set,

$$u^*_i = \max\, u\quad \text{s.t. } (u, x^*_i)\in\Omega_i\ \text{ and }\ x^*_i + 2\Delta_i u\in K_{i+1},\qquad x^*_{i+1} = x^*_i + 2\Delta_i u^*_i,$$

another $N$ tiny LPs forward. Membership in $K_{i+1}$ is what guarantees the greed never traps me into a corner from which the goal becomes unreachable.

Two things make me trust this. Correctness: by backward induction every admissible profile satisfies $x_i\in K_i$ — if $x_i\in K_i$ and $x_i = x_{i-1}+2\Delta_{i-1}u_{i-1}$ with $(u_{i-1},x_{i-1})\in\Omega_{i-1}$, the one-step-set definition forces $x_{i-1}\in K_{i-1}$ — so failure is reported only when the instance is truly infeasible, and a forward induction shows any returned sequence is admissible. Optimality: the greedy profile dominates every admissible profile pointwise, $x^*_i\ge x_i$, and since $T=\sum_i\Delta_i/\sqrt{x_i}$ decreases as the $x$'s rise, that pointwise dominance means minimum time. The induction hinges on the maximal-transition function $T^\beta_i(x)=x+2\Delta\,\beta_i(x)$ being *non-decreasing* in $x$: then $x^*_i\ge x_i$ gives $T^\beta_i(x^*_i)\ge T^\beta_i(x_i)\ge x_{i+1}$, the last step because $x_{i+1}$ is *some* admissible successor and hence no larger than the maximal one. That monotonicity can only fail at zero-inertia points — but here the singularity is demoted from a *robustness* problem that crashed the integrator to a benign *discretization-accuracy* problem: it merely opens a sub-optimality gap that shrinks to zero as $\Delta\to 0$. So the method is exactly optimal away from singularities and asymptotically optimal through them, while never being numerically fragile at them. The mirror construction — propagate the *reachable* set forward and sweep backward picking the lowest control — is a valid dual that also answers the admissible-velocity-propagation query a planner wants, on the same LP machinery.

One practical point on assembling $a,b,c$ without ever forming $M$ or $C$: recursive Newton–Euler inverse dynamics gives only the products $\mathrm{inv\_dyn}(q,q̇,q̈)=M(q)q̈+q̇^\top C(q)q̇+g(q)$, and three calls isolate the coefficients — $\mathrm{inv\_dyn}(q,0,0)=g=c$ (zero velocity kills Coriolis, zero acceleration kills inertia); $\mathrm{inv\_dyn}(q,0,q')=Mq'+g$, so $a=\mathrm{inv\_dyn}(q,0,q')-c$; and $\mathrm{inv\_dyn}(q,q',q'')=Mq''+q'^\top C q'+g$, so $b=\mathrm{inv\_dyn}(q,q',q'')-c$. The torque box becomes $F=[I;-I]$, $g=[\tau_{\max};-\tau_{\min}]$. The whole pipeline — project the dynamics, sweep controllable sets backward, sweep greedily forward — is $O(mN)$, faster than numerical integration's $O(m^2N)$ and far faster than the convex program's $O(KmN^3)$, with a 100% success rate and a few hundred lines of code.

```python
import numpy as np

# --- Project the dynamics onto the path: a(s) sddot + b(s) sdot^2 + c(s) = w (= torque). ---
# Three inverse-dynamics evaluations isolate c, a, b without forming M(q), C(q).
def torque_constraint_params(path, gridpoints, inv_dyn, tau_min, tau_max):
    dof  = path.dof
    p    = path.eval(gridpoints)     # q(s_i)
    ps   = path.evald(gridpoints)    # q'(s_i)
    pss  = path.evaldd(gridpoints)   # q''(s_i)
    zero = np.zeros(dof)
    c = np.array([inv_dyn(p_, zero, zero)           for p_           in p])                  # g(q)
    a = np.array([inv_dyn(p_, zero, ps_) for p_, ps_ in zip(p, ps)]) - c                      # M q'
    b = np.array([inv_dyn(p_, ps_, pss_) for p_, ps_, pss_ in zip(p, ps, pss)]) - c           # M q'' + q'^T C q'
    F = np.vstack([np.eye(dof), -np.eye(dof)])      # torque box as polytope F w <= g
    g = np.concatenate([tau_max, -tau_min])
    return a, b, c, F, g

# --- Stagewise LP over y = (u, x) = (sddot, sdot^2). ---
def solve_stage_lp(i, obj, a, b, c, F, g, delta, x_pin, xnext_lo, xnext_hi, solve_lp):
    # dynamics rows: [F a_i, F b_i] . (u, x) <= g - F c_i
    A  = [np.column_stack([F.dot(a[i]), F.dot(b[i])])]
    bb = [g - F.dot(c[i])]
    # controllable-set membership: xnext_lo <= x + 2 delta u <= xnext_hi
    if xnext_hi is not None:
        A.append(np.array([[2 * delta,  1.0]]));  bb.append(np.array([ xnext_hi]))
    if xnext_lo is not None:
        A.append(np.array([[-2 * delta, -1.0]])); bb.append(np.array([-xnext_lo]))
    A  = np.vstack(A); bb = np.concatenate(bb)
    lb = np.array([-np.inf, 0.0 if x_pin is None else x_pin])
    ub = np.array([ np.inf, np.inf if x_pin is None else x_pin])
    return solve_lp(obj, A, bb, lb, ub)             # y = (u, x); NaNs if infeasible

# --- Backward pass: controllable sets K_i as intervals (two LPs per stage). ---
def compute_controllable_sets(N, params, deltas, sd_end, solve_lp):
    a, b, c, F, g = params
    K = np.zeros((N + 1, 2))
    K[N] = [sd_end**2, sd_end**2]
    obj_max_x = np.array([0.0, -1.0])               # maximize x  ==  minimize -x
    for i in range(N - 1, -1, -1):
        x_hi = solve_stage_lp(i,  obj_max_x, a, b, c, F, g, deltas[i],
                              None, K[i+1, 0], K[i+1, 1], solve_lp)[1]
        x_lo = solve_stage_lp(i, -obj_max_x, a, b, c, F, g, deltas[i],
                              None, K[i+1, 0], K[i+1, 1], solve_lp)[1]
        K[i] = [max(x_lo, 0.0), x_hi]
        if np.isnan(K[i]).any():                    # one-step set empty
            return K
    return K

# --- Forward pass: greedily take the highest controllable control. ---
def compute_parameterization(N, params, deltas, sd_start, sd_end, solve_lp):
    a, b, c, F, g = params
    K = compute_controllable_sets(N, params, deltas, sd_end, solve_lp)
    x0 = sd_start**2
    if np.isnan(K).any() or not (K[0, 0] - 1e-9 <= x0 <= K[0, 1] + 1e-9):
        return None                                 # infeasible / start not controllable
    xs = np.zeros(N + 1); us = np.zeros(N); xs[0] = x0
    for i in range(N):
        obj = np.array([-2 * deltas[i], -1.0])      # maximize x + 2 delta u  ==  maximize speed
        y = solve_stage_lp(i, obj, a, b, c, F, g, deltas[i],
                           xs[i], K[i+1, 0], K[i+1, 1], solve_lp)  # pin x = xs[i]
        us[i] = y[0]
        x_next = xs[i] + 2 * deltas[i] * us[i]
        xs[i+1] = min(K[i+1, 1], max(K[i+1, 0], x_next))          # clamp into controllable set
    return us, np.sqrt(xs)                           # (sddot, sdot) over the grid


# ---------------------------- usage ----------------------------
# import toppra as ta, toppra.constraint as constraint, toppra.algorithm as algo
# path   = ta.SplineInterpolator(ss, way_pts)             # geometric path q(s)
# pc_vel = constraint.JointVelocityConstraint(vlims)
# pc_acc = constraint.JointAccelerationConstraint(alims)
# instance = algo.TOPPRA([pc_vel, pc_acc], path)          # reachability-based TOPP
# jnt_traj = instance.compute_trajectory()                # rest-to-rest, minimum time
# t = np.linspace(0, jnt_traj.duration, 100); q = jnt_traj(t)
```
