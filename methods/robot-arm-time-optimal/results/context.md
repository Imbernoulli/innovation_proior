# Context: Driving a fixed robot path at the actuator limit

## Research question

A motion planner hands us a *geometric path* for a robot arm: a curve q(s), s∈[0, s_end], through the n-dimensional configuration space that is already collision-free and respects the workspace. What it does **not** give us is timing — how fast to move along that curve. For a 7-DOF industrial arm running a welding seam, a pick-and-place loop, or a machining pass thousands of times a day, the timing is the productivity: every tenth of a second per traversal is money. The geometry is fixed; the only freedom left is the *time-scaling*, the increasing scalar function s(t) that says where along the path the robot is at each instant. The recovered motion is q(s(t)).

The goal is the **minimum-time** time-scaling: traverse the given path as fast as physically possible without ever exceeding the actuators. The actuators impose hard limits — each joint motor has a torque ceiling τ_min ≤ τ_i ≤ τ_max, and often there are direct joint-velocity and joint-acceleration bounds as well. Push the timing faster and somewhere the required torque saturates a motor or a speed limit is breached; the path is then no longer dynamically traversable. A solution must (i) find the fastest s(t) that keeps every joint inside its limits everywhere along the path, (ii) detect when a path is *not* traversable at all, and (iii) be fast and reliable enough to run online, as an inner loop inside a kinodynamic planner that may call it thousands of times.

What makes this both tractable and subtle is that fixing the geometry collapses a high-dimensional trajectory-optimization problem (optimize a curve in ℝⁿ over time) into a problem with **one** scalar degree of freedom, s(t). The question is how to exploit that collapse to compute the optimal timing exactly and robustly.

## Background

**Rigid-body manipulator dynamics.** An n-DOF arm with configuration q∈ℝⁿ obeys

  M(q) q̈ + q̇ᵀ C(q) q̇ + g(q) = τ,

where M(q) is the (symmetric positive-definite) inertia matrix, the term q̇ᵀC(q)q̇ collects Coriolis/centrifugal forces (quadratic in velocity), g(q) is gravity, and τ is the vector of joint torques. Given (q, q̇, q̈), the torque τ is computed by **inverse dynamics**; the recursive Newton–Euler algorithm does this in O(n) without ever forming the full M and C explicitly, returning only the products that matter. The actuator constraint is the box τ_min ≤ τ ≤ τ_max.

**Path/timing separation.** When the path q(s) is fixed, q is a function of the single scalar s, and s is in turn a function of time. Differentiating by the chain rule, with ′ = d/ds and ˙ = d/dt:

  q̇ = q′(s) ṡ,  q̈ = q′(s) s̈ + q″(s) ṡ².

So the joint velocities and accelerations are determined by (s, ṡ, s̈) — the path position, path velocity, and path acceleration — together with the fixed geometric derivatives q′(s), q″(s). The 2D **phase plane** (s, ṡ) becomes the natural state space: a velocity *profile* ṡ(s) over s∈[0, s_end] encodes a candidate timing, and the traversal time is T = ∫ dt = ∫₀^{s_end} ds/ṡ.

**The objective is pointwise.** Because T = ∫ ds/ṡ and 1/ṡ is decreasing in ṡ, the traversal time shrinks monotonically as the velocity profile is raised. There is no trade-off across path positions: to go fastest, push ṡ as high as the constraints permit at *every* s. This is the structural fact that makes the problem special — minimizing a path integral reduces to maximizing the velocity profile pointwise, subject only to the dynamic feasibility limits.

**The Maximum Velocity Curve (MVC).** For each fixed s, the admissible path accelerations s̈ form an interval whose width shrinks as ṡ grows; at a critical velocity the interval collapses to a single point and beyond it there is no feasible s̈ at all. The locus of these critical velocities over s is the Maximum Velocity Curve. It follows directly from the actuator limits becoming bounds on s̈, and any valid velocity profile must lie on or below it.

**Bang-bang optimal control.** Pontryagin's Maximum Principle applied to the minimum-time problem with the pointwise objective above yields a *bang-bang* solution: the optimal path acceleration s̈ sits on a constraint boundary almost everywhere, alternating between maximum acceleration and maximum deceleration, with the profile riding under the MVC and switching between the two modes at isolated *switch points*. This is the classical characterization of the time-optimal velocity profile.

**Dynamic singularities.** At certain path positions a constraint loses its dependence on s̈ (a "zero-inertia point": the s̈ coefficient of some constraint vanishes). Near such points the maximum/minimum-acceleration vector fields *diverge*, and the acceleration to use there is not naturally defined. Handling velocity bounds introduces a separate "direct" MVC with its own trap points.

**Set-membership control / reachability (from control theory and MPC).** For a discrete-time *linear* system with *linear* state–control inequality constraints, there is a mature toolkit: the **reachable set** (states attainable from a starting set under admissible controls) and the **controllable set** (states from which the goal set can be reached). When the state is scalar and the constraints are polytopic, these sets are intervals whose endpoints solve small linear programs. This machinery is standard in the model-predictive-control literature.

## Baselines

**Numerical-Integration (NI) phase-plane method (Bobrow, Dubowsky & Gibson, 1985; Shin & McKay, 1985).** The founding approach. Project the dynamics onto the path: substituting q̇ = q′ṡ, q̈ = q′s̈ + q″ṡ² into M q̈ + q̇ᵀC q̇ + g = τ gives, row by row,

  a_i(s) s̈ + b_i(s) ṡ² + c_i(s) ≤ 0,

with a = M q′ (stacked ± for the two torque bounds), b = M q″ + q′ᵀC q′, c = g − τ-limits. Each row is linear in s̈ and in ṡ²; solving for s̈ gives an upper bound β_i = (−c_i − b_i ṡ²)/a_i when a_i>0 and a lower bound α_i when a_i<0. Taking the tightest over all rows, β(s,ṡ)=min_q β_q and α(s,ṡ)=max_p α_p define the maximum- and minimum-acceleration fields; feasibility is α ≤ s̈ ≤ β. The MVC is where α=β. The optimal profile is built by integrating these fields: integrate forward along β (max accel) and backward along α (decel) from the boundary velocities, locate the α→β switch points on the MVC, and stitch the β- and α-arcs together into the time-optimal profile. Complexity O(m²N) (m constraint rows, N grid points), dominated by building the MVC via O(m²) quadratics per grid point.

**Robust NI with singularity handling (Pham, 2014).** A complete treatment of the dynamic singularities the original NI method encounters: characterize exactly which zero-inertia points are genuine (non-differentiable) singularities, and derive the correct singular acceleration to initiate integration there (a slope λ chosen so the α-profile's tangent points at the singular point on the MVC), plus a clean treatment of the direct-velocity MVC and its trap points. It exploits the explicit bang-bang structure, so controls are computed, not searched.

**Convex-Optimization (CO) method (Verscheure et al., 2008; Hauser, 2014).** Observe that with x := ṡ² the path-projected constraints become *linear* in the unknowns (path accelerations and squared velocities at the grid points), and the traversal time is a convex function of them. So formulate the whole TOPP as one large convex program over all grid variables at once, and hand it to an off-the-shelf solver. Simple to implement, robust (100% success), and flexible (other convex objectives, redundant actuation, contact constraints). The single program has O(N) variables and O(mN) inequalities; solving it (e.g. as a sequence of LPs) costs about O(KmN³).

## Evaluation settings

The natural test bed is a robot arm (e.g. a 6- or 7-DOF manipulator such as a Barrett WAM, or a generic redundant arm) tracking spline paths through random waypoints in configuration space, subject to joint velocity, joint acceleration, and joint torque limits. Harder settings stress the constraint count m (stacking many joint and Cartesian constraints) and the discretization grid size N, and include redundantly-actuated arms and legged robots under contact-stability (linearized friction-cone) constraints, where the per-position admissible set is a polytope projection. The quantities of interest are: the computed minimum traversal time and its convergence as the grid is refined; the wall-clock solve time and how it scales with m and N; the success rate across many random instances (does the method return a valid parameterization whenever one exists, and correctly report infeasibility otherwise); and the constraint-satisfaction error introduced by discretization (collocation O(Δ) vs. interpolation O(Δ²) schemes). Reference points for comparison are the NI and CO baselines run on the same instances.

## Code framework

The pieces below already exist before the method: a way to represent the geometric path and evaluate its derivatives, an inverse-dynamics routine, a small LP/QP solver, and a grid. The one empty slot is the engine that turns the path-projected, discretized constraints into the optimal velocity profile.

```python
import numpy as np

# --- Geometric path: already given by a planner; we only evaluate it. ---
class GeometricPath:
    """A fixed curve q(s), s in [0, s_end], with derivatives."""
    def eval(self, s):    ...   # q(s)      shape (n,)
    def evald(self, s):   ...   # q'(s)     shape (n,)
    def evaldd(self, s):  ...   # q''(s)    shape (n,)
    @property
    def path_interval(self): ...  # (0, s_end)
    dof = None

# --- Inverse dynamics: recursive Newton-Euler; products only, no full M, C. ---
def inv_dyn(q, qd, qdd):
    """Return tau = M(q) qdd + qd^T C(q) qd + g(q)."""
    ...

# --- Actuator limits as a polytope on the constrained quantity w (here w = tau). ---
def torque_polytope(tau_min, tau_max):
    """Return F, g with F w <= g encoding tau_min <= w <= tau_max."""
    ...

# --- A small linear-program solver over a few variables. ---
def solve_lp(objective_c, A_ub, b_ub, lb, ub):
    """min c.y  s.t.  A_ub y <= b_ub,  lb <= y <= ub.  y is low-dimensional."""
    ...

# --- Project the second-order constraints onto the path at each grid point. ---
def constraint_params(path, gridpoints, tau_lim):
    """For each s_i return a_i, b_i, c_i and the polytope (F, g) such that
       a_i * sddot + b_i * sdot^2 + c_i  is the constrained quantity w,
       feasible iff F w <= g.  (Built from inv_dyn and the path derivatives.)"""
    ...  # TODO: chain-rule projection q' , q''  ->  a, b, c

# --- Discretize the path into N segments. ---
def make_grid(path, N):
    return np.linspace(*path.path_interval, N + 1)

# === The slot the method will fill: produce the fastest feasible timing. ===
def parameterize(path, gridpoints, params, sd_start, sd_end):
    """Given the path-projected discretized constraints, return the
       minimum-time path-velocity profile sdot(s_i) (and accelerations),
       or report that the path is not traversable.

       Returns sdd (shape N,), sd (shape N+1,)  -- or None if infeasible."""
    # TODO: turn the per-stage path-projected constraints into the optimal profile.
    pass

# --- Recover q(t) from the velocity profile by integrating dt = ds / sdot. ---
def retime(path, gridpoints, sd):
    """Integrate the profile back to a time-stamped trajectory q(t)."""
    ...
```
