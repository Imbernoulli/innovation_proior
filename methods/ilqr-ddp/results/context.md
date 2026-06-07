# Context: locally optimal feedback control of a nonlinear plant over a finite horizon

## Research question

Given a discrete-time *nonlinear* plant `x_{i+1} = f(x_i, u_i)` and a cost `J = Σ ℓ(x_i, u_i) + ℓ_f(x_N)` over a finite horizon, how do you actually *compute* the control sequence `u_0, …, u_{N-1}` that minimizes `J` — and not just an open-loop sequence, but a control with a **feedback law** attached, so the plant can be steered back onto the optimized trajectory when it is bumped off? The method has to (a) handle a general smooth nonlinearity in `f` and in `ℓ`, (b) exploit the temporal structure of the problem rather than treat it as one giant unstructured optimization, (c) converge quickly — at the rate of a second-order method, not a first-order one — and (d) emit, at every step of the optimized trajectory, a local feedback gain `K_i` so the result is a usable controller, not a single trajectory.

The pain is sharp and specific. For the *linear* plant with a *quadratic* cost the answer is settled and beautiful: dynamic programming gives a quadratic value function, the per-step minimization over the control is an unconstrained convex quadratic, and a single backward Riccati recursion produces the time-varying optimal feedback gains in closed form — one sweep, exact, with feedback included. But the moment `f` is nonlinear, that backward recursion no longer closes: the value function is no longer quadratic, the per-step minimization is no longer a clean quadratic, and there is no finite object to propagate backward. One can fall back on a generic nonlinear program over the whole stacked decision vector `(u_0, …, u_{N-1})`, but that throws away the very thing that made the linear case tractable — the stage-wise, backward-in-time recursion — and a generic optimizer returns an open-loop sequence with no feedback gains. The goal is to keep the backward-recursion machinery of the linear-quadratic regulator and make it work for a nonlinear plant.

## Background

**Bellman's principle of optimality and the value function.** The cost-to-go from state `x` at time `i` under the optimal remaining control is the value function `V(x, i) = min_{u_i,…,u_{N-1}} Σ_{j≥i} ℓ(x_j, u_j) + ℓ_f(x_N)`. The principle of optimality says the tail of an optimal trajectory is itself optimal, which collapses the minimization over an entire control sequence into a *sequence of single-control minimizations* run backward in time:

    V(x, i) = min_u [ ℓ(x, u) + V(f(x, u), i+1) ],   V(x, N) = ℓ_f(x).

This is the structural fact everything rests on. The catch is the curse of dimensionality: tabulating `V` over the whole state space is hopeless for any non-trivial `n`, so the recursion is only useful when `V` has a finite parametric form that the backward step preserves.

**The discrete-time LQR backward Riccati recursion.** For the linear plant `x_{i+1} = A x_i + B u_i` and quadratic cost `Σ (x_i'Q x_i + u_i'R u_i) + x_N'Q_f x_N` (with `Q ⪰ 0`, `R ≻ 0`), the value function *is* quadratic at every step, `V(x, i) = x'S_i x`. Substituting into the Bellman recursion, the bracket is quadratic in `u`; because `R ≻ 0` the inner minimization is a strictly convex unconstrained quadratic, solved by setting the gradient to zero:

    u_i* = −(R + B'S_{i+1}B)^{-1} B'S_{i+1}A x_i = −K_i x_i,

and back-substitution gives the **discrete algebraic/difference Riccati recursion**

    S_i = A'S_{i+1}A − A'S_{i+1}B (R + B'S_{i+1}B)^{-1} B'S_{i+1}A + Q,   S_N = Q_f.

A single backward sweep from `S_N` produces every gain `K_i`. Quadratic-in / quadratic-out is exactly the property that lets the recursion close, and the feedback structure (`u_i*` depends on `x_i`, i.e. on `V`'s gradient) is automatic. This is the machine to be salvaged.

**The calculus-of-variations / costate route.** The other classical handle on optimal control is the Pontryagin/Hamiltonian formulation: adjoin the dynamics with a costate (multiplier) `λ`, form the Hamiltonian `H = ℓ + λ'f`, and impose stationarity in `u` together with the costate recursion `λ_i = ∂H/∂x`. For a trajectory-tracking problem this can be linearized around a nominal `(x̄, ū)`: write `δx_{i+1} = A_i δx_i + B_i δu_i` with `A_i = ∂f/∂x`, `B_i = ∂f/∂u`, and solve the resulting linear two-point boundary-value problem for the optimal *deviation* `δu_i`. The structural limitation is that the costate recursion, taken literally, produces an *open-loop* correction tied to one boundary condition; recovering a feedback form requires positing an affine ansatz `δλ_i = S_i δx_i + v_i` and grinding through the matrix-inversion lemma — at which point a Riccati-like recursion for `S_i` reappears, now driven by an auxiliary affine term `v_i` carrying the nominal's first-order residual.

**Newton vs. Gauss–Newton for the cost.** Minimizing a sum of stage costs subject to the dynamics is, abstractly, a smooth optimization in the stacked controls. A full Newton step needs the exact Hessian, which for a dynamics-constrained problem inherits the *second* derivatives of `f` — a rank-three tensor `∂²f` at every step. For nonlinear-least-squares-shaped objectives, the Gauss–Newton method drops the term proportional to the residual times `∂²f` and keeps only the first-derivative outer products; the resulting approximate Hessian is positive semidefinite by construction, the step is far cheaper, and near a small-residual solution it is nearly identical to the Newton step. The same Newton/Gauss–Newton choice is available for the trajectory problem and turns on whether the second derivatives of the *dynamics* are retained.

**Local validity of a linearization, and trust.** Linearizing `f` around a nominal trajectory is only accurate in a neighborhood. A full optimization step computed from the linearized model can leap outside that neighborhood, where the model's predicted cost decrease has nothing to do with the true cost — the same failure that motivates Levenberg–Marquardt damping and line search in unconstrained optimization. Two well-known phenomena set up the method: (i) when the per-step control-Hessian of the local quadratic model fails to be positive definite (which it can, far from a minimum, or whenever a curvature term is indefinite), the unconstrained minimizer is meaningless and the step must be damped; and (ii) when the model is locally valid but the full step still overshoots, a backtracking step-size restores descent. These are not new observations about this method — they are the standard pathologies of second-order optimization, imported wholesale.

## Baselines

**Discrete-time LQR (backward Riccati).** Core idea and math as above: quadratic value function, convex per-step minimization, one backward Riccati sweep yielding time-varying feedback gains. Gap: it is *exact only for a linear plant with a quadratic cost*. For nonlinear `f` the value function is not quadratic, the recursion does not close, and LQR simply does not apply — except as the local model at a single operating point, which is valid only in a small neighborhood and carries no notion of improving a whole nonlinear trajectory.

**Single-operating-point linearization + LQR.** Linearize `f` once about an equilibrium, design one LQR gain, and use it as a fixed feedback law. Core idea: a constant gain `K` from one Riccati solve. Gap: valid only near the linearization point; for a large-amplitude maneuver (a reaching movement, a getting-up motion, an acrobatic recovery) the plant travels far from any single operating point, so a single linear model is wrong over most of the trajectory and there is no mechanism to optimize the *trajectory itself*.

**Generic nonlinear programming over the stacked controls.** Treat `min J` over `(u_0, …, u_{N-1})` (or over states-and-controls with the dynamics as equality constraints) as one large NLP and hand it to an SQP/interior-point solver. Core idea: a general-purpose optimizer with the dynamics stapled on as constraints. Gap: it discards the stage-wise backward recursion that makes the problem special, so it does not naturally exploit the temporal/Markov structure; and a generic NLP returns an *open-loop* optimal sequence with no per-step feedback gain `K_i`, so the output is a trajectory, not a controller — exactly the feedback the dynamic-programming view hands you for free.

**Costate / indirect shooting around a nominal.** Linearize the Hamiltonian's necessary conditions about a nominal trajectory and solve the resulting linear two-point boundary-value problem for the deviation `δu`. Core idea: indirect (Pontryagin) optimization, iterated. Gap: literal forward/backward sweeps of the costate give an open-loop correction; a feedback form only emerges after positing the affine costate ansatz and applying the matrix-inversion lemma, and even then the route is first-order in the dynamics (it linearizes `f`), so it does not by itself recover the second-order convergence that retaining dynamics curvature would give.

## Evaluation settings

The natural testbeds are nonlinear plants with smooth dynamics where a large-amplitude, finite-horizon maneuver is wanted: a multi-link arm (a 2-link planar arm, or a musculo-skeletal arm with ~10 state dimensions and several muscle controls) performing an energy-optimal reaching movement to a target; canonical underactuated benchmarks such as the inverted pendulum / cart-pole and the acrobot swing-up; a planar swimmer or one-legged hopper; and, at the high end, a simulated humanoid getting up from an arbitrary pose or recovering from a large disturbance. The yardsticks that exist independently of any particular solver are: the achieved total cost `J` for a fixed horizon and target; the number of iterations / convergence rate to a local minimum (first-order vs. second-order behavior); robustness of the resulting closed-loop controller to state perturbations (does the feedback gain pull a bumped trajectory back?) and to modeling error (controls optimized on one model, applied to another); and, for the online/receding-horizon use, the wall-clock per-step solve time relative to the control rate. The horizon `N`, the cost weights, and the target state are the experimental knobs.

## Code framework

The primitives that already exist: dense linear algebra (matrix products, symmetric solves, an eigen/Cholesky check for positive-definiteness), a routine that integrates the plant `x_{i+1} = f(x_i, u_i)` forward from `x_0` under a control sequence, and a way to obtain along a trajectory the first derivatives of the dynamics `f` (and optionally its second derivatives) together with the first and second derivatives of the stage cost `ℓ` — by analytic formulas, finite differences, or automatic differentiation. The slots to be filled are the backward recursion that turns those per-step derivatives into a correction with feedback, and the forward step that applies the correction to produce an improved trajectory.

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
    return a correction to the control sequence: a feedforward part and a
    state-dependent feedback part, one pair per step.

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
