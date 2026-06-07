# Time-Optimal Path Parameterization (TOPP)

## Problem

Given a fixed geometric path q(s), s∈[0, s_end], in a robot's configuration space (already collision-free, supplied by a planner) and actuator limits (joint torque bounds, optionally joint velocity/acceleration bounds), find the **minimum-time** time-scaling s(t) so that the executed motion q(s(t)) traverses the path as fast as possible without violating any limit — or report that the path is not dynamically traversable.

## Key idea

Fixing the geometry leaves a single scalar degree of freedom, the timing s(t). The chain rule

  q̇ = q′(s) ṡ,   q̈ = q′(s) s̈ + q″(s) ṡ²

substituted into the manipulator dynamics M(q)q̈ + q̇ᵀC(q)q̇ + g(q) = τ makes the torque **linear in (s̈, ṡ²)**:

  τ = a(s) s̈ + b(s) ṡ² + c(s),   a = M q′,  b = M q″ + q′ᵀC q′,  c = g.

So actuator limits become, at each s, linear inequalities a_i(s) s̈ + b_i(s) ṡ² + c_i(s) ≤ 0. Because the traversal time T = ∫ ds/ṡ is minimized by raising the velocity profile ṡ(s) **pointwise**, the optimal solution rides as high as feasible under the Maximum Velocity Curve (where the admissible-s̈ interval pinches shut), with a bang-bang structure (accelerate maximally, decelerate maximally, switching on the MVC).

The robust, fast realization (TOPP-RA) works in the discretized phase plane with the squared velocity as state. Set x = ṡ², u = s̈ constant per segment; then

  x_{i+1} = x_i + 2 Δ_i u_i

is an **exactly linear** discrete-time system, and the per-stage constraints a_i u + b_i x + c_i ∈ C_i (torque polytope) are linear in (u, x). This is a linear system with linear control–state constraints, so reachability machinery applies and, since the state is scalar, all the relevant sets are **intervals** computed by tiny 2-variable LPs:

- **Controllable set** K_i = squared velocities at stage i from which the goal velocity ṡ_N is reachable. Backward recursion K_N = {ṡ_N²}, K_i = one-step-set(K_{i+1}); each interval endpoint is one LP.
- **Forward greedy pass**: from x*_0 = ṡ_0², at each stage take the maximal control u keeping x*_{i+1} ∈ K_{i+1}.

Switch points are implicit in the controllable sets (no explicit search), and the zero-inertia "dynamic singularities" that wreck phase-plane numerical integration become a benign discretization-accuracy issue rather than a failure mode. Complexity O(mN) (m constraint rows, N grid points): faster than numerical integration's O(m²N) and far faster than the convex-program O(KmN³), with a 100% success rate.

## Algorithm

```
Input: path q(s), constraints, grid s_0..s_N, start/goal velocities sdot_0, sdot_N
1. Project: for each i, compute a_i, b_i, c_i and polytope (F, g) from q(s_i), q'(s_i), q''(s_i).
2. Backward pass (controllable sets, intervals K_i):
     K_N = { sdot_N^2 }
     for i = N-1 .. 0:
       x^+ = max x  s.t. a_i u + b_i x + c_i in C_i  and  x + 2 Δ_i u in K_{i+1}
       x^- = min x  (same constraints)
       K_i = [max(x^-, 0), x^+]              # empty -> path not parameterizable
     if K_0 empty or sdot_0^2 not in K_0:  return Infeasible
3. Forward pass (greedy):
     x*_0 = sdot_0^2
     for i = 0 .. N-1:
       u*_i = max u  s.t. (u, x*_i) in Ω_i  and  x*_i + 2 Δ_i u in K_{i+1}
       x*_{i+1} = x*_i + 2 Δ_i u*_i
4. Return profile sdot(s_i) = sqrt(x*_i), accelerations sddot = u*_i;
   integrate dt = ds / sdot to recover q(t).
```

Correctness: backward induction shows any admissible profile has x_i ∈ K_i, so failure is reported only when truly infeasible; forward induction shows the returned sequence is admissible. Optimality: the greedy profile dominates every admissible profile pointwise (x*_i ≥ x_i) whenever the maximal transition function is non-decreasing, hence minimizes T; with zero-inertia points the sub-optimality gap vanishes as Δ → 0.

## Code

Faithful to the `toppra` library (reachability-based TOPP). Core: project the dynamics with three inverse-dynamics calls, sweep controllable sets backward, then greedily forward.

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
