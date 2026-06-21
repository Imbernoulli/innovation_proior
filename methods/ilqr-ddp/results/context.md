# Context: locally optimal feedback control of a nonlinear plant over a finite horizon

## Research question

Given a discrete-time *nonlinear* plant `x_{i+1} = f(x_i, u_i)` and a cost `J = Σ ℓ(x_i, u_i) + ℓ_f(x_N)` over a finite horizon, how do you compute the control sequence `u_0, …, u_{N-1}` that minimizes `J` — and obtain not just an open-loop sequence but a control with a **feedback law** attached, so the plant can be steered back onto the optimized trajectory when it is bumped off? The setting calls for handling a general smooth nonlinearity in `f` and in `ℓ`, working with the temporal structure of the problem, and emitting at each step of the optimized trajectory a local feedback gain `K_i` so the result is a usable controller.

For the *linear* plant with a *quadratic* cost the answer is settled: dynamic programming gives a quadratic value function, the per-step minimization over the control is an unconstrained convex quadratic, and a single backward Riccati recursion produces the time-varying optimal feedback gains in closed form — one sweep, exact, with feedback included. With nonlinear `f`, the value function is no longer quadratic and the per-step minimization is no longer a clean quadratic. One alternative is a generic nonlinear program over the whole stacked decision vector `(u_0, …, u_{N-1})`. The question is how to compute a locally optimal control with feedback gains for a nonlinear plant over a finite horizon.

## Background

**Bellman's principle of optimality and the value function.** The cost-to-go from state `x` at time `i` under the optimal remaining control is the value function `V(x, i) = min_{u_i,…,u_{N-1}} Σ_{j≥i} ℓ(x_j, u_j) + ℓ_f(x_N)`. The principle of optimality says the tail of an optimal trajectory is itself optimal, collapsing the minimization over an entire control sequence into a *sequence of single-control minimizations* run backward in time:

    V(x, i) = min_u [ ℓ(x, u) + V(f(x, u), i+1) ],   V(x, N) = ℓ_f(x).

Tabulating `V` over the whole state space scales exponentially in the state dimension `n`; the recursion is practical when `V` has a finite parametric form that the backward step preserves.

**The discrete-time LQR backward Riccati recursion.** For the linear plant `x_{i+1} = A x_i + B u_i` and quadratic cost `Σ (x_i'Q x_i + u_i'R u_i) + x_N'Q_f x_N` (with `Q ⪰ 0`, `R ≻ 0`), the value function is quadratic at every step, `V(x, i) = x'S_i x`. Substituting into the Bellman recursion, the bracket is quadratic in `u`; because `R ≻ 0` the inner minimization is a strictly convex unconstrained quadratic, solved by setting the gradient to zero:

    u_i* = −(R + B'S_{i+1}B)^{-1} B'S_{i+1}A x_i = −K_i x_i,

and back-substitution gives the **discrete algebraic/difference Riccati recursion**

    S_i = A'S_{i+1}A − A'S_{i+1}B (R + B'S_{i+1}B)^{-1} B'S_{i+1}A + Q,   S_N = Q_f.

A single backward sweep from `S_N` produces every gain `K_i`. The value function is quadratic in and quadratic out, and the feedback structure (`u_i*` depends on `x_i`, i.e. on `V`'s gradient) is automatic.

**The calculus-of-variations / costate route.** The other classical handle on optimal control is the Pontryagin/Hamiltonian formulation: adjoin the dynamics with a costate (multiplier) `λ`, form the Hamiltonian `H = ℓ + λ'f`, and impose stationarity in `u` together with the costate recursion `λ_i = ∂H/∂x`. For a trajectory-tracking problem this can be linearized around a nominal `(x̄, ū)`: write `δx_{i+1} = A_i δx_i + B_i δu_i` with `A_i = ∂f/∂x`, `B_i = ∂f/∂u`, and solve the resulting linear two-point boundary-value problem for the optimal *deviation* `δu_i`. Taken literally, the costate recursion produces an open-loop correction tied to one boundary condition; a feedback form is recovered by positing an affine ansatz `δλ_i = S_i δx_i + v_i` and applying the matrix-inversion lemma, at which point a Riccati-like recursion for `S_i` reappears, driven by an auxiliary affine term `v_i` carrying the nominal's first-order residual.

**Newton vs. Gauss–Newton for the cost.** Minimizing a sum of stage costs subject to the dynamics is, abstractly, a smooth optimization in the stacked controls. A full Newton step needs the exact Hessian, which for a dynamics-constrained problem inherits the *second* derivatives of `f` — a rank-three tensor `∂²f` at every step (an `n × n × n` tensor per step). For nonlinear-least-squares-shaped objectives, the Gauss–Newton method drops the term proportional to the residual times `∂²f` and keeps only the first-derivative outer products; the resulting approximate Hessian is positive semidefinite by construction, the step is cheaper, and near a small-residual solution it is nearly identical to the Newton step.

**Local validity of a linearization, and trust.** Linearizing `f` around a nominal trajectory is accurate in a neighborhood. A full optimization step computed from the linearized model can leave that neighborhood, where the model's predicted cost decrease departs from the true cost — the setting that motivates Levenberg–Marquardt damping and line search in unconstrained optimization. Two standard fixes from second-order optimization apply: (i) when a local curvature (Hessian) approximation fails to be positive definite — which it can, far from a minimum, or whenever a curvature term is indefinite — the step is damped; and (ii) when the model is locally valid but the full step overshoots, a backtracking step-size restores descent.

## Baselines

**Discrete-time LQR (backward Riccati).** Quadratic value function, convex per-step minimization, one backward Riccati sweep yielding time-varying feedback gains. Exact for a linear plant with a quadratic cost; for nonlinear `f` it serves as the local model at a single operating point.

**Single-operating-point linearization + LQR.** Linearize `f` once about an equilibrium, design one LQR gain `K` from a single Riccati solve, and use it as a fixed feedback law near the linearization point.

**Generic nonlinear programming over the stacked controls.** Treat `min J` over `(u_0, …, u_{N-1})` (or over states-and-controls with the dynamics as equality constraints) as one large NLP and hand it to an SQP/interior-point solver — a general-purpose optimizer with the dynamics stapled on as constraints, returning an open-loop optimal sequence.

**Costate / indirect shooting around a nominal.** Linearize the Hamiltonian's necessary conditions about a nominal trajectory and solve the resulting linear two-point boundary-value problem for the deviation `δu`; indirect (Pontryagin) optimization, iterated. A feedback form follows from the affine costate ansatz and the matrix-inversion lemma, and the route linearizes `f` (first-order in the dynamics).

## Evaluation settings

The natural testbeds are nonlinear plants with smooth dynamics where a large-amplitude, finite-horizon maneuver is wanted: a multi-link arm (a 2-link planar arm, or a musculo-skeletal arm with ~10 state dimensions and several muscle controls) performing an energy-optimal reaching movement to a target; canonical underactuated benchmarks such as the inverted pendulum / cart-pole and the acrobot swing-up; a planar swimmer or one-legged hopper; and, at the high end, a simulated humanoid getting up from an arbitrary pose or recovering from a large disturbance. The solver-independent yardsticks are: the achieved total cost `J` for a fixed horizon and target; the number of iterations / convergence rate to a local minimum (first-order vs. second-order behavior); robustness of the resulting closed-loop controller to state perturbations (does the feedback gain pull a bumped trajectory back?) and to modeling error (controls optimized on one model, applied to another); and, for online/receding-horizon use, the wall-clock per-step solve time relative to the control rate. The horizon `N`, the cost weights, and the target state are the experimental knobs.

## Code framework

The primitives that already exist: dense linear algebra (matrix products, symmetric solves, an eigen/Cholesky check for positive-definiteness), a routine that integrates the plant `x_{i+1} = f(x_i, u_i)` forward from `x_0` under a control sequence, and a way to obtain along a trajectory the first derivatives of the dynamics `f` (and optionally its second derivatives) together with the first and second derivatives of the stage cost `ℓ` — by analytic formulas, finite differences, or automatic differentiation. The slots to be filled are the backward recursion that turns those per-step derivatives into a control correction, and the forward step that applies the correction to produce an improved trajectory.

```python
import numpy as np


def forward_rollout(f, cost, x0, us):
    """Integrate x_{i+1} = f(x_i, u_i) from x0 under control path us, and
    collect along the trajectory the per-step derivatives of the dynamics
    (f_x, f_u, and optionally the second-order f_xx, f_ux, f_uu) and of the
    stage cost (l_x, l_u, l_xx, l_ux, l_uu), plus the terminal cost derivatives.
    Returns the state path xs, the running cost, and all of these arrays."""
    # TODO: roll out the nominal trajectory and gather Jacobians/Hessians.
    pass


def backward_pass(derivatives):
    """Given the per-step dynamics/cost derivatives along the nominal
    trajectory and the terminal value derivatives, sweep backward in time and
    return a correction to the control sequence, one per step.

    This is the slot where the still-to-be-derived backward recursion lives:
    how to propagate a finite description of the cost-to-go from step i+1 to
    step i through the nominal dynamics, and read a control correction off it.
    """
    # TODO: derive the per-step recursion and the form of the correction.
    pass


def forward_update(f, cost, xs, us, correction, alpha):
    """Apply the correction to the nominal (xs, us) to produce a new trajectory
    by re-integrating the true nonlinear dynamics f, with a step-size alpha in
    (0, 1] controlling how much of the correction is taken."""
    # TODO: re-roll the true dynamics under the corrected control.
    pass


def solve(f, cost, x0, us_init, n_iterations):
    """Iterate forward_rollout -> backward_pass -> forward_update until the
    cost stops improving, damping the backward step and backtracking alpha
    when a step fails to reduce the true cost."""
    # TODO: the outer loop, with the damping/line-search bookkeeping.
    pass
```
