## Research question

Given a plant with a known discrete-time linear model `x_{k+1} = A x_k + B u_k`, find a feedback control law that regulates the state to a setpoint *while respecting hard limits the plant physically imposes*: actuators that saturate (a valve is between fully closed and fully open; a motor delivers bounded torque; a power converter bounded voltage), actuator slew-rate limits (the input can only change so fast per step), and state/output limits (a tank level, a temperature, a pressure, a battery state-of-charge must stay inside a safe band). The control must anticipate these limits before they are hit, not react after a violation, and it must do so for multivariable plants where inputs and outputs are coupled.

Why it matters: in process plants the economically optimal operating point typically lies *at the intersection of constraints* — you make money by pushing a controlled variable as close to its limit as possible without crossing it (Prett & Gillette 1980).

## Background

The state of the art for optimal linear regulation is the linear-quadratic regulator and its stochastic companion, the linear-quadratic-Gaussian (LQG) controller, developed from Kalman's early-1960s work. For a linear plant and a quadratic cost `J = sum_k (x_k' Q x_k + u_k' R u_k)`, dynamic programming yields a value function that is exactly quadratic, `V*(x) = x' S x`, and the optimal control is *linear state feedback* `u = -K x`, where `K = (R + B'SB)^{-1} B'SA` and `S` solves the discrete algebraic Riccati equation

    S = A'SA - A'SB (R + B'SB)^{-1} B'SA + Q.

The infinite prediction horizon endows this law with strong stabilizing properties: for a stabilizable, detectable plant with `Q >= 0`, `R > 0`, the closed loop `A - BK` is stable, and `x'Sx` is a Lyapunov function whose one-step decrease equals the running cost. The Kalman filter supplies the state estimate when only outputs are measured. This is constructive (an algorithm eating `(A,B,Q,R)`), MIMO, and optimal — and it became standard in aerospace, where accurate first-principles models are available.

The reasons LQG had limited impact on the process industries, despite its theoretical strength, are recorded as: constraints; process nonlinearities; model uncertainty; unique performance criteria; and cultural factors (Kwakernaak & Sivan 1972 added output, offset-free, and target extensions).

Two older ideas are available. First, the moving-horizon idea: Propoi (1963) described a moving-horizon controller, and Lee & Markus (1967) discussed open-loop optimal control computed over a finite look-ahead from the current state. Second, quadratic programming: minimizing a convex quadratic subject to linear inequality constraints is among the simplest optimization problems, solvable reliably by standard codes, with a unique minimizer when the Hessian is positive definite.

Two empirical facts about existing plants set up the problem:
- Real actuators saturate and have rate limits; real states/outputs have hard safety bands.
- For a constrained plant with integrating or unstable modes, horizon length matters: a finite-horizon optimization that is too short may not maintain state-constraint satisfaction over time.

## Baselines

- **LQR / LQG (Kalman 1960; Kwakernaak & Sivan 1972).** Quadratic cost, linear dynamics; DP gives `u = -Kx` with `K` from the Riccati equation. Optimal, MIMO, constructive, provably stabilizing on the infinite horizon.

- **IDCOM / model predictive heuristic control (Richalet, Rault, Testud & Papon 1978).** The first reported industrial predictive controller. Finite-impulse-response plant model `y_{k+j} = sum_i h_i u_{k+j-i}`; quadratic objective over a finite prediction horizon; the predicted output is driven toward a *reference trajectory*, a first-order path from the current output to the setpoint whose time constant tunes the closed-loop aggressiveness and gives robustness to model error; input and output constraints are checked inside an iterative ("heuristic") solver, justified as the dual of identification.

- **DMC — dynamic matrix control (Cutler & Ramaker 1980; Prett & Gillette 1980).** Linear *step-response* model `y_{k+j} = sum_i s_i Δu_{k+j-i} + s_N u_{k+j-N}`; the predicted future output change is a linear function of the future input moves through the "dynamic matrix"; the optimal move vector solves an unconstrained least-squares problem with a move-suppression penalty on `Δu` (which also conditions the numerics). Multivariable by superposition; the controller gain is precomputed off-line, only the first move is applied.

- **QDMC — quadratic DMC (Cutler, Morshedi & Haydel 1983; García & Morshedi 1986).** Recognizes that the DMC objective, with future outputs related to the input-move vector through the dynamic matrix, can be written as a *standard quadratic program* in which input and output constraints appear explicitly as linear inequalities. The Hessian is positive definite for a linear plant, so the QP is convex and solvable by standard codes.

## Evaluation settings

The natural test plants are constrained linear (or linearized) systems where the limits are the whole point. Representative settings:
- **Constrained double integrator** `x_{k+1} = [[1,1],[0,1]]x_k + [[0],[1]]u_k`, with `-1 <= u <= 1` and a state box `-5 <= x <= 5` (or `-10 <= x <= 10`): the canonical small example for studying feasibility and stability of a finite-horizon receding controller.
- **A small unstable plant**, e.g. `x_{k+1} = [[2,1],[0,0.5]]x_k + [[1],[0]]u_k` with input/state boxes: exposes loss of feasibility and divergence for short horizons.
- **A point mass / second-order mechanical plant** `[[1,Ts],[0,1-bTs/M]]`, with bounded force `|u| <= u_max`, a position box, and a slew-rate bound on the force — a clean closed-loop regulation/setpoint task.
- **An inverted pendulum on a cart** linearized about the upright, with bounded actuator force and a track-length state limit.

Metrics/protocol: closed-loop simulation from a grid of initial states; whether the state and input constraints are respected for all time; whether the closed-loop converges to the setpoint; the region of initial states from which the loop stays feasible and converges; per-step solve time within one sampling interval; and, as a tuning study, how horizon length `N` and the weights `(Q, R)` and rate penalty change the behavior.

## Code framework

Available primitives: a discrete-time LTI model, a convex QP solver, and a Riccati/LQR solver for unconstrained quadratic regulation.

```python
import numpy as np
import scipy.sparse as sparse
import scipy.linalg
import osqp


def lqr_terminal(Ad, Bd, Q, R):
    """Discrete LQR: returns gain K (u=-Kx) and value matrix S (x'Sx),
    from the discrete algebraic Riccati equation."""
    S = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad)
    return K, S

class Controller:
    def __init__(self, Ad, Bd, N, **kw):
        self.Ad, self.Bd, self.N = Ad, Bd, N
        # TODO: store weights, references, constraint boxes
        self.prob = osqp.OSQP()

    def _build(self):
        # TODO: choose the decision variables, write the quadratic cost,
        #       encode the model and hard limits, and store
        #       (P_qp, q_qp, A_qp, l_qp, u_qp) for the QP solver.
        pass

    def setup(self):
        # TODO: build the optimization for the current state and hand it to the solver
        pass

    def step(self):
        # TODO: solve the optimization; return the FIRST input only
        pass

    def update(self, x_meas, u_prev=None):
        # TODO: re-insert the new measured state (and last input); re-assemble
        pass


def closed_loop(Ad, Bd, ctrl, x0, nsim):
    x = x0
    for t in range(nsim):
        u = ctrl.step()              # solve optimization, take first input
        x = Ad @ x + Bd @ u          # plant step (or integrate true dynamics)
        ctrl.update(x)               # re-measure, re-assemble for next step
    return x
```
