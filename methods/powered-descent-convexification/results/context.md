# Context: minimum-fuel powered-descent (soft landing) guidance

## Research question

A spacecraft on its final powered-descent phase — a Mars lander, a lunar descent stage, a vertical-takeoff-vertical-landing booster — must fly from a known position and velocity to a **soft landing** at (or as close as possible to) a target site: terminal velocity zero, terminal altitude zero, upright. It does this by throttling and steering a single rocket engine (or a fixed cluster) whose exhaust both decelerates the vehicle and **consumes the only mass it has to spend**. The objective that dominates everything else is **fuel**: every kilogram of propellant burned on descent is a kilogram of payload (science instruments, sample-return canister, crew consumables) that could not be carried. So the guidance problem is: among all descent trajectories that respect the vehicle's physics and its safety constraints, find the one that **burns the least fuel** — equivalently, the one that **lands with the most mass**.

The trajectory must obey hard constraints that make the problem nontrivial: the engine is **throttleable only within a band** — it cannot run below a minimum thrust ρ_min (deep throttling causes combustion instability) and cannot exceed a maximum ρ_max; the vehicle must stay above the terrain on approach (a **glide-slope cone**); the thrust vector cannot tilt arbitrarily far from vertical (a **pointing / tilt** limit); speed may be capped; and the mass strictly decreases as fuel burns. The boundary conditions (initial state, zero terminal velocity, zero terminal altitude) are fixed.

What a usable solution must additionally achieve, beyond just being optimal on paper: it must be computable **onboard, autonomously, within a bounded and predictable amount of time**, and it must return a **globally** optimal (or certified-infeasible) answer — not a local minimum that depends on a lucky initial guess. A lander firing its descent engine has seconds of margin; a guidance algorithm that might stall in a local minimum, or whose runtime is unbounded, cannot be trusted to fire the engine.

## Background

**The minimum-fuel soft-landing problem is a classical optimal control problem.** The vehicle is a point mass under constant gravity g (plus, in a rotating planet frame, Coriolis and centripetal terms through the planet's angular rate ω). Writing position r and velocity ṙ, and letting T_c be the commanded thrust vector and m the mass,

  r̈ = g + T_c/m − 2 S(ω) ṙ − S(ω)² r,  ṁ = −α ‖T_c‖,

where S(ω) is the skew-symmetric cross-product matrix of ω and α = 1/(I_sp g₀) is the mass-flow-per-Newton constant from the rocket equation (propellant consumed in proportion to thrust magnitude). Minimum fuel is

  minimize ∫₀^{tf} ‖T_c(t)‖ dt  ⇔  maximize m(tf),

because integrating ṁ = −α‖T_c‖ gives m(tf) = m₀ − α∫‖T_c‖.

**Pontryagin's maximum principle** characterizes the optimizer. Form the Hamiltonian with costates (adjoint variables) on r, ṙ, m; the necessary conditions say the optimal thrust at each instant maximizes the Hamiltonian over the admissible thrust set. Because the running cost ‖T_c‖ and the mass costate enter linearly in the thrust magnitude, the magnitude part of the problem is **bang-bang**: the optimal throttle sits at ρ_max or ρ_min and switches between them, governed by the sign of a scalar switching function built from the costates. The direction of optimal thrust aligns with the velocity costate (the "primer vector"): T_c* points along λ_v, with magnitude pinned to a bound. This bang-bang structure of minimum-fuel rocket control is the classical result the whole problem inherits — the engine is "on hard, off (to minimum), on hard."

**The structural obstacle: the thrust magnitude lives in an annulus.** Combustion stability forbids running the engine below ρ_min; performance caps it at ρ_max. So at every instant

  ρ_min ≤ ‖T_c(t)‖ ≤ ρ_max,  with 0 < ρ_min < ρ_max.

The upper bound ‖T_c‖ ≤ ρ_max is a ball — a convex set. The lower bound ‖T_c‖ ≥ ρ_min is the **complement of a ball** — a nonconvex set. The admissible thrust set is therefore a nonconvex **annulus** (a shell). This is a known, load-bearing pathology: it is exactly the lower bound that makes the feasible control set nonconvex, and it is not removable by ignoring it — a trajectory that coasts at ‖T_c‖ < ρ_min is one the physical engine cannot fly.

**Two further nonconvexities.** (i) The dynamics are **nonlinear**: T_c/m has the control divided by the (varying) mass state. (ii) A thrust **pointing** constraint that limits the tilt of T_c from a reference direction n̂, written n̂ᵀT_c ≥ ‖T_c‖cos θ_p, is nonconvex when θ_p > 90° (an obtuse "pie slice").

**Convex optimization and second-order cone programming (SOCP).** A convex program has a feasible set that is a convex set and a convex objective; for such problems every local minimum is global, and **interior-point methods** solve them to a specified accuracy in a deterministic, bounded number of iterations — the property a flight algorithm needs. A second-order cone program is the convex problem whose constraints are linear together with **second-order cone** constraints ‖A x + b‖₂ ≤ cᵀx + d (a vector's Euclidean norm bounded by an affine function). Norm-bounds on thrust, cone constraints like glide slope, and norm caps on velocity are all naturally second-order cone constraints; SOCP is therefore the right target class, and mature interior-point SOCP solvers (e.g. SeDuMi, ECOS, Clarabel) exist.

**Established empirical facts that frame the design.** It is well documented that minimum-fuel rocket thrust profiles are bang-bang (max–min–max). It is also a known property of this landing problem that the **minimum-fuel cost is unimodal in the time-of-flight** tf — a single-variable line search over tf finds the best flight time. And for the planetary-landing geometry, the glide-slope cone is observed to be touched by the optimal trajectory only at isolated instants (typically once mid-flight and at touchdown), never sustained over an interval — a fact that will matter for any guarantee involving the state constraints.

## Baselines

**Direct nonlinear-programming / nonlinear optimal-control collocation.** Discretize the trajectory and hand the full nonconvex problem — annulus thrust bound, nonlinear dynamics and all — to a general nonlinear programming solver (SQP / interior-point on a nonconvex NLP), or solve the maximum-principle two-point boundary-value problem by shooting. Core idea: parameterize states/controls on a grid, enforce the dynamics as equality constraints, minimize fuel. Gap: these methods can converge to **local** minima, depend heavily on the initial guess, and carry **no bound on the number of iterations** to a solution — there is no certificate that the answer is globally optimal or that it will arrive in time. For an autonomous engine firing, "might find a good trajectory, eventually, if warm-started well" is not acceptable.

**Polynomial / analytic guidance laws (Apollo-style, gravity-turn, explicit guidance).** Historically, descent guidance used closed-form or low-order polynomial acceleration profiles chosen to meet the boundary conditions. Core idea: pick a parameterized acceleration history (e.g. quadratic in time) and solve for coefficients hitting the terminal state. Gap: these are **not fuel-optimal** and cannot honor the full constraint set (annulus thrust band, glide slope, pointing, divert maneuvers) — they trade optimality and constraint fidelity for closed-form simplicity.

**Convexify by simply dropping the lower thrust bound.** One could relax ρ_min ≤ ‖T_c‖ away entirely, leaving only ‖T_c‖ ≤ ρ_max (convex), and solve the resulting convex problem. Core idea: enlarge the feasible set to a ball so the problem becomes convex. Gap: the solution will, in general, **command thrust below ρ_min** (or zero) where the unconstrained optimum wants to coast — physically unflyable; the relaxation is not faithful, so its optimum is not a solution to the real problem.

## Evaluation settings

The natural test bed is **Mars / planetary powered descent** to a target site. A scenario is defined by: gravity magnitude and direction; planet rotation rate ω; engine parameters ρ_min, ρ_max and specific impulse I_sp (hence α); wet (initial) and dry (minimum allowable) mass; the glide-slope cone half-angle; the thrust pointing/tilt limit; a maximum-velocity cap; and the initial position/velocity offset from the target (including large-divert cases where the lander must fly far cross-range to reach a safe site). The decision metrics are **propellant consumed / landed mass**, **landing miss distance** to the target, and — for onboard use — **solve time and iteration count**. The yardstick a new method is held to is whether it returns the global fuel optimum (or a certificate of infeasibility), honors every constraint, and does so in bounded time. Free-final-time scenarios add a one-dimensional search over the flight time tf.

## Code framework

The pre-method scaffold is a discretized trajectory-optimization harness: a parameter block, a discrete dynamics/constraint builder, and a convex solve. The contribution to be filled in is *how the nonconvex thrust band and nonlinear mass dynamics are encoded so the builder emits a convex program* — that is the single empty slot.

```python
import numpy as np
import cvxpy as cp

class DescentParameters:
    """Vehicle, environment, and grid parameters known before any method."""
    def __init__(self):
        self.g = np.array([-3.71, 0.0, 0.0])     # gravity vector [m/s^2]
        self.omega = np.array([2.53e-5, 0.0, 6.62e-5])  # planet rate [rad/s]
        self.m_wet = 2000.0                       # initial mass [kg]
        self.m_dry = ...                          # minimum allowable mass [kg]
        self.rho_min = ...                        # lower thrust bound [N]
        self.rho_max = ...                        # upper thrust bound [N]
        self.alpha = ...                          # mass flow per Newton [s/m]
        self.theta_p = ...                        # thrust tilt limit
        self.gamma_gs = ...                       # glide-slope angle
        self.v_max = ...                          # speed cap
        self.r0 = ...; self.v0 = ...              # initial state
        self.N = ...; self.dt = ...               # grid

    def Smatrix(self):
        """Skew matrix of omega for Coriolis/centripetal terms."""
        ...

def build_dynamics(p):
    """Linear part of the translational dynamics (A, B) on [r; v]."""
    # Known: r_ddot = g + (thrust term) - 2 S v - S^2 r
    ...
    return A, B

def encode_thrust_band(p, control_vars):
    """
    TODO — THE CONTRIBUTION.
    The thrust magnitude must satisfy the nonconvex annulus
        rho_min <= ||T_c|| <= rho_max,
    and the mass dynamics m_dot = -alpha ||T_c|| are nonlinear (T_c / m).
    Produce, from these, constraints a convex (SOCP) solver can accept,
    WITHOUT giving up the true optimum. How to encode this is the open slot.
    """
    pass  # TODO

def build_problem(p):
    """Assemble the discretized trajectory-optimization problem."""
    # state, control, (and whatever auxiliary variables the contribution needs)
    x = cp.Variable((6, p.N))      # [r; v]
    # ... mass / log-mass variable, control variable(s) ...
    cons = []
    A, B = build_dynamics(p)
    for k in range(p.N - 1):
        # discrete dynamics for [r; v]
        pass  # TODO: x_{k+1} = x_k + (A x_k + B (...)) dt
        # mass depletion update
        pass  # TODO
    cons += encode_thrust_band(p, None)   # TODO: the convex thrust encoding
    # glide slope, pointing, velocity cap, boundary conditions
    pass  # TODO
    objective = cp.Minimize(...)          # TODO: minimum-fuel objective
    return cp.Problem(objective, cons)

def solve_descent(p):
    prob = build_problem(p)
    prob.solve(solver=cp.CLARABEL)
    return prob

def time_of_flight_search(p):
    """Outer 1-D search over the free final time tf (cost unimodal in tf)."""
    # golden-section over tf, calling solve_descent on a fixed grid each time
    pass  # TODO
```
