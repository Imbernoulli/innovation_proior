Let me start from the thing that's actually broken, because the controller I already trust has a hole in it. I have a plant I can write as a discrete linear model `x_{k+1} = A x_k + B u_k`, and for the quadratic cost `sum_k (x_k'Q x_k + u_k'R u_k)` I know the answer cold: dynamic programming gives a quadratic value function `x'Sx`, the inner minimization over `u` is unconstrained and strictly convex because `R>0`, so it solves in closed form and the optimal control is linear feedback `u = -Kx` with `K=(R+B'SB)^{-1}B'SA` and `S` from the Riccati equation. That law is optimal, it's MIMO, it's one matrix equation, and on the infinite horizon `x'Sx` is a Lyapunov function so the closed loop is stable. This is the settled baseline I am trying not to lose.

Every step of that derivation assumed I get to apply `-Kx` *exactly*. But my actuator can't deliver an arbitrary number. A valve goes from fully shut to fully open and no further; a motor has a maximum torque; a power converter a maximum voltage; and the input can only *change* so fast per step. So `u` lives in a box `u in U`, and on top of that the state has its own hard limits — a tank level, a pressure, a temperature, a battery's state of charge that must stay in a safe band, so `x in X`. When `-Kx` asks for an input outside `U`, the real plant just clips it. And the moment it clips, the input applied is not `-Kx` anymore, so the optimality argument is gone and — worse — the Lyapunov argument is gone too, because the discrete decrease `V(Ax-BKx)-V(x) = -(x'Qx+x'K'RKx)` was computed *for the exact optimal input*. Clipping can destabilize a loop that was provably stable. And there's no version of `-Kx` that says "ride the constraint": in a process plant the money is made by sitting a controlled variable as close to its limit as you dare without crossing, and a linear gain has no concept of "approach but don't violate."

So the real problem is: keep the quadratic objective I like, but *add hard constraints* `u in U`, `x in X`, and get a feedback law out the other end.

First instinct: just redo the dynamic programming with the constraints stapled on. The HJB inner step becomes `min over u in U of [x'Qx + u'Ru + (cost-to-go of Ax+Bu)]` with the further requirement that `Ax+Bu` land in `X`. Try it for one step and stare at what `u*(x)` looks like. Without constraints the minimizer was `-Kx`, linear, and the value function stayed a clean quadratic, which is exactly what let the recursion close. With the box, the minimizer is the *projection* of that unconstrained solution onto the feasible region — so in the interior it's `-Kx`, but near a face of the box it saturates, and the boundary between "interior" and "saturated" is itself a function of `x`. The control law is piecewise — affine on polyhedral pieces of state space — and the value function is piecewise-quadratic, no longer a single `x'Sx`. So the magic that made the Riccati recursion close (quadratic in, quadratic out) is dead. There's no one-shot gain anymore. Wall.

I don't actually need a closed-form law over all of state space, though. I need, *right now, at the state I'm sitting at*, the best input. So drop the demand for a global feedback formula and just solve the optimization numerically, from the current measured state, over a finite window. Pose this: I'm at `x(t)`, look `N` steps ahead, and choose a whole sequence of inputs `u_0,...,u_{N-1}` to minimize the cost over that window subject to the model and the limits:

    min   sum_{k=0}^{N-1} (x_k'Q x_k + u_k'R u_k) + x_N'P x_N
    s.t.  x_{k+1} = A x_k + B u_k,   x_0 = x(t),
          u_k in U,  x_k in X,  k = 0..N-1.

The decision variable is the entire input sequence, not just the next input. Even if the controller eventually applies only `u_0`, the later moves are how the optimization prices delayed consequences and future state limits. The terminal term `x_N'P x_N` is the only place the finite window can remember the cost beyond step `N`; for the moment `P` is the unknown tail-cost weight I still have to pin down.

Now, what kind of optimization is this? The cost is a sum of quadratic forms, so quadratic in the decision variables. The dynamics are linear equalities. The box limits `u in U`, `x in X` are linear inequalities. A quadratic objective with linear equality and inequality constraints is a quadratic program; with the input penalty making the Hessian positive definite, it is convex with a unique global minimizer and a reliable solve with standard codes. That's the trade I'm making: I gave up the closed-form Riccati gain, and in exchange I got a problem that's *one of the simplest things an optimizer can chew on*, and it eats the constraints natively.

Let me actually build the QP, because the structure matters and there are two ways to do it. The honest way to see it is to eliminate the states. The model says `x_0 = x(t)`, `x_1 = Ax(t)+Bu_0`, `x_2 = A^2x(t)+ABu_0+Bu_1`, and so on — every future state is the current state propagated plus a convolution of the inputs so far. Stack `X = [x_0;x_1;...;x_N]` and `U_0 = [u_0;...;u_{N-1}]`. Then

    X = S^x x(t) + S^u U_0,

with `S^x = [I; A; A^2; ...; A^N]` (the free response) and `S^u` the lower-triangular block-banded matrix whose `(i,j)` block is `A^{i-1-j}B` for `i>j` and zero otherwise (the forced response — input `u_j` reaches state `x_i` through `i-1-j` steps of the dynamics). Now the stacked cost is `X'Qbar X + U_0'Rbar U_0` with `Qbar = blockdiag(Q,...,Q,P)` (the last block is the terminal weight `P`) and `Rbar = blockdiag(R,...,R)`. Substitute the prediction:

    J = (S^x x + S^u U_0)' Qbar (S^x x + S^u U_0) + U_0'Rbar U_0
      = U_0'(S^u'Qbar S^u + Rbar)U_0 + 2x'(S^x'Qbar S^u)U_0 + x'(S^x'Qbar S^x)x.

So in terms of `U_0` alone this is `U_0'H U_0 + 2x'F U_0 + x'Y x` with

    H = S^u'Qbar S^u + Rbar,   F = S^x'Qbar S^u,   Y = S^x'Qbar S^x.

`Rbar > 0` because `R>0`, and `S^u'Qbar S^u` is at least positive semidefinite, so `H > 0` — strictly convex, unique minimizer, exactly the well-posedness I wanted. This is the *condensed* form: the only decision variable left is `U_0`, dimension `mN`, and the constraints `u_k in U`, `x_k in X` all become linear inequalities `G_0 U_0 <= w_0 + E_0 x(t)` once I substitute `X = S^x x + S^u U_0` into the state box. A small dense QP parameterized by the current state `x(t)` through the linear term `2F'x` and the right-hand side `E_0 x`.

Let me sanity-check the condensed form against the thing I trust. Strip the constraints for a second — then `min_U U'HU + 2x'FU` is unconstrained, gradient zero gives `U_0* = -H^{-1}F'x`. The *first block* of that is some gain times `x`, and if I worked it through it would reproduce the finite-horizon LQR feedback — and as `N` and the terminal weight are taken consistently, the very `-Kx` I started from. Good: the constrained QP, in the region where no constraint is active, *is* LQR. The QP is LQR with the saturations folded in. That's the unification I was hoping for — I haven't thrown LQR away, I've embedded it.

There's a second way to build the QP that I'll actually prefer in code, and the reason is sparsity. Instead of eliminating the states, keep them as decision variables too: let the vector be `(x_0,...,x_N,u_0,...,u_{N-1})`, write the dynamics `x_{k+1}-Ax_k-Bu_k=0` as *equality constraints*, and the boxes as plain bound constraints on the variables. The Hessian is now block-diagonal `blockdiag(Q,...,Q,P,R,...,R)` and the constraint matrix is banded — much bigger than the condensed `H`, but extremely sparse, and a sparse QP solver exploits that. The condensed form pays `O(N^2)`-ish dense structure to shrink the variable count; the sparse form keeps everything banded. For a solver like OSQP that loves sparsity, the banded form is the better target. Either way it's the *same* optimization; I'll write the banded one.

I've computed an optimal *sequence* `u_0*,...,u_{N-1}*` from `x(t)`. The naive thing would be to just play that whole sequence out open-loop. Don't. The sequence was computed from a model, and the model is wrong in detail and the plant gets disturbed; running it blind, errors accumulate and there's no correction. So: apply only `u_0*`, throw the rest away. At the next instant the plant has moved to some `x(t+1)` — *measure it* (don't trust the predicted `x_1`), and re-solve the whole `N`-step problem from this new, true state. Apply that new first input. Repeat. The horizon slides forward one step each time; it *recedes*.

Why this is the right thing and not a hack: re-solving from the measured state is what injects feedback. The open-loop sequence is a function of `x(t)` fixed at `t`; re-measuring and re-optimizing makes the *applied* input a function of the *current* state `x(t+1)`, which is the definition of feedback and the source of disturbance rejection and robustness to model error. It's the same instinct as measuring the state before each LQR multiply — except here the "law" is an optimization rather than a matrix. And the payoff for paying that per-step solve cost is precisely the constraints: because I re-impose `u in U` and `x in X` over the whole look-ahead window every time, the input I actually apply *never* violates a limit, and a state limit `N` steps out is *anticipated now* and steered around before it's reached. That anticipation is exactly what a clipped `-Kx` could never do. This is, recognizably, the old moving-horizon idea — Lee & Markus wrote it down in 1967, "measure the current state, compute the open-loop control, use the first portion, then measure again and repeat," and Propoi sketched it earlier — but now it's practical because the per-step problem is a convex QP a standard solver dispatches inside one sampling interval.

The whole sequence earns its keep in two ways, even though I only ever apply `u_0`. The cost over the horizon plus the terminal term is a *proxy for the infinite-horizon cost* — it's how the controller knows that an input which looks cheap now leads somewhere expensive later. And the future inputs are what let me satisfy a *future* state constraint: to promise that `x` won't blow through a limit three steps out, I have to plan the inputs that keep it inside, even though I'll re-plan them next step. Drop the look-ahead and you can't anticipate; that's the whole game.

Before I trust this, I have to worry about something, because I just replaced an *infinite*-horizon optimal control (LQR), which was provably stable and always feasible, with a *finite*-horizon one solved repeatedly. Truncating the horizon is not free. Two things can break.

Take feasibility first. At `x(t)` the QP has a solution — fine. Does that guarantee the *next* QP, from `x(t+1)`, also has one? Not in general. Picture even the constrained double integrator `x_{k+1} = [[1,1],[0,1]]x_k + [[0],[1]]u_k`, with a bounded input `|u|<=1` and a state box. With a short horizon, the optimizer is shortsighted: it minimizes cost over `N` steps and happily lets the state drift toward a region where, given the bounded input, there's *no* sequence that can keep it inside the box anymore. The current QP is feasible, applies its first input, the state moves — and now the next QP has no feasible point at all. The controller has driven itself into a corner it can't get out of. So "feasible now" does not imply "feasible forever"; that's a property I have to *engineer*, call it persistent feasibility.

How do I guarantee it? The clean fix is to add a *terminal constraint*: require `x_N in X_f`, with `X_f subset X`, and choose `X_f` to be control invariant under admissible inputs — from any point in `X_f` there is a `v in U` such that `Ax+Bv` is still in `X_f`. The argument is a one-step shift. Suppose the QP at `x(t)` is feasible with sequence `u_0*,...,u_{N-1}*` and predicted states ending in `x_N in X_f`. At `t+1` the true state is `x_1 = Ax(t)+Bu_0*`. Build a candidate for the new problem by shifting: `u_1*,...,u_{N-1}*` carries the prediction through the old `x_N`, and then invariance gives one more admissible input `v` with `Ax_N+Bv in X_f` to fill the last slot. All the old state and input constraints are inherited by the shifted prediction, the terminal state is again in `X_f`, and the appended input is admissible by construction. If slew-rate limits are part of the constraint set, I either augment the state with the previously applied input or require the terminal choice to satisfy the last planned rate bound `Dumin <= v-u_{N-1}* <= Dumax`; the shift proof is the same after that bookkeeping. So feasibility at `t` produces feasibility at `t+1`: persistent feasibility holds whenever the terminal set is invariant for the full admissible constraint set. (If I don't want an explicit `X_f`, the alternative is to make `N` long enough that the look-ahead reaches into the controllable region on its own — but the terminal-set version is the guarantee I can actually certify.)

Stability is subtler. Even with feasibility nailed down, minimizing a *finite*-horizon cost does not force the closed loop to decay to the setpoint. Same kind of failure mode as in the infinite-horizon theory, where I needed observability so the cost "saw" every mode — but here the horizon is the issue. On a short horizon the optimizer can leave a slowly-diverging tail "for later," beyond the window, and pay nothing for it, so the closed loop wanders or even diverges while every individual QP is solved happily. I need to make the finite-horizon cost behave like the infinite one.

The lever is that terminal term `x_N'P x_N` I've been carrying with a placeholder `P`. The idea: the head of the horizon, steps `0..N-1`, handles the constrained transient; the terminal cost should account for *everything after step `N`* — the cost of the tail from `x_N` onward. If `P` is chosen so that `x_N'P x_N` equals the true cost-to-go from `x_N` under some stabilizing controller that keeps the tail inside the constraints, then the finite-horizon objective is no longer shortsighted: it's an exact (or upper-bounding) stand-in for the infinite-horizon cost, and minimizing it inherits the infinite horizon's stability.

What's the cost-to-go of the tail? If, once the state reaches the terminal region `X_f`, the constraints are no longer active and the *unconstrained* optimal controller takes over — and that controller is exactly LQR, with value function `x'Sx`, `S` the Riccati solution — then the cost from `x_N` to infinity under LQR is precisely `x_N' S x_N`. So set `P = S`, the LQR value matrix. The terminal cost *is the LQR value function*. The picture is: run the constrained MPC until the state enters `X_f`, where the unconstrained LQR is admissible, and from there `x'Sx` accounts exactly for the rest. MPC = a constrained head splice onto an LQR tail.

Let me verify this actually makes the closed loop a Lyapunov system, because "it should" isn't a proof. Let `J*(x)` be the optimal value of the QP started at `x`. I want `J*` to strictly decrease along the closed loop. At `x` the optimizer is `u_0*,...,u_{N-1}*` with states `x_0=x,...,x_N in X_f`, value `J*(x)`. Apply `u_0*`; the plant moves to `x^+ = Ax + Bu_0*`. Now bound `J*(x^+)` by exhibiting one *feasible* (not necessarily optimal) sequence for the new problem: the shift-and-append from before, `u_1*,...,u_{N-1}*, v`, where `v` is the LQR move at the terminal state. Its cost is the old cost, minus the stage cost `q(x,u_0*)` I peeled off the front, minus the old terminal cost `p(x_N)`, plus the new last stage cost `q(x_N,v)` plus the new terminal cost `p(Ax_N+Bv)`:

    J*(x^+) <= J*(x) - q(x,u_0*) - p(x_N) + q(x_N,v) + p(Ax_N+Bv).

If the terminal pieces collapse — if `-p(x_N) + q(x_N,v) + p(Ax_N+Bv) <= 0` for the terminal input `v` on `X_f` — then

    J*(x^+) <= J*(x) - q(x,u_0*),

and, when the stage cost is positive definite in the regulated state (or under the usual detectability condition), that decrease is strict away from the origin. With `J* >= 0` bounded below, that's a Lyapunov function and the origin is asymptotically stable with domain of attraction the feasible set. So the whole stability requirement boils down to one condition on the terminal cost: there must exist a terminal feedback `v` on `X_f` with

    -p(x) + q(x,v) + p(Ax+Bv) <= 0   for all x in X_f,

i.e. `p` is a control-Lyapunov function and `X_f` is invariant under that terminal feedback. (`p` of this kind is exactly a control-Lyapunov function — it certifies a one-step decrease.)

And now check that `p(x)=x'Sx` with the LQR terminal controller `v=-Kx` satisfies it. Plug `q(x,v)=x'Qx+v'Rv`, `v=-Kx`, into the inequality and ask when it's an equality:

    -x'Sx + x'(Q + K'RK)x + x'(A-BK)'S(A-BK)x = 0   for all x,

i.e. `(A-BK)'S(A-BK) - S + Q + K'RK = 0`. That's the discrete Lyapunov equation for the LQR closed loop, and it holds *with equality* precisely when `S` solves the discrete algebraic Riccati equation with `K=(R+B'SB)^{-1}B'SA`. So choosing `P=S` makes the terminal inequality tight, and the terminal cost equals the genuine infinite-horizon LQR tail cost `sum_{k>=N}(x_k'Qx_k+u_k'Ru_k)`. The stability recipe is not a heuristic; it's the LQR value function dropped in as the terminal cost, with `X_f` a small invariant neighborhood where `-Kx` stays inside the box. Without that terminal cost a short horizon can diverge; with it, the finite-horizon controller is certifiably stabilizing.

The input itself is not the only actuator quantity that matters. Real valves and motors also cannot jump instantaneously, and a controller that exploits large alternating moves is usually exploiting the model more than the plant. So the cost should see the input increment `Δu_k = u_k - u_{k-1}` with weight `Q_Δu`: the running cost gets a `Δu_k'Q_Δu Δu_k` term. That damps an over-aggressive controller, buys robustness to model error because small moves cannot exploit a wrong model as hard, and lets the actuator's slew-rate limit `Δu in [Δu_min, Δu_max]` enter as another linear constraint. It also conditions the QP numerically. To carry `Δu` I have to remember the last applied input `u_{-1}` as an extra parameter, exactly the way I remember `x(t)`.

I also cannot treat every constraint the same way. Input limits are physical — the actuator simply can't exceed them, so they stay *hard*. But a *state/output* limit can become infeasible: a big disturbance can shove the state to where no admissible input keeps it inside the box, and then the QP has no solution and the controller is dead at the worst moment. So in practice I soften the state/output constraints — introduce signed slack variables in the state-bound rows and add a large quadratic penalty on them. A positive slack relaxes one side of a two-sided box, a negative slack relaxes the other, and the penalty makes either violation expensive. The QP then returns the least-violating solution while inputs stay strictly within their hard box. The bare controller without slack is the clean core; the slack is the deployment wrapper.

From one step to the next, the QP barely changes — only the right-hand side moves through the new `x(t)` and new `u_{-1}`. The Hessian, the constraint matrix, and the cost shape stay fixed. So I set the solver up once and on each step just update the parameter vectors and re-solve from the previous solution as a warm start. That's what makes the per-step solve fast enough to run online.

In code this means a banded sparse QP handed to OSQP. The decision vector is `z = (x_0,...,x_N, u_0,...,u_{N-1})`; dynamics are equality constraints, boxes are bound constraints, the rate rows apply the block difference `u_k-u_{k-1}`, and the first input of the solution is what I apply.

```python
import numpy as np
import scipy.sparse as sparse
import scipy.linalg
import osqp


class MPCController:
    """Linear constrained receding-horizon controller.

    Plant:  x_{k+1} = Ad x_k + Bd u_k   (discrete LTI)
    OSQP solves 1/2 z'Pz + q'z; constants are omitted, so P=Q and q=-Q xref
    encode the half-scaled tracking objective. Each step solve, from the
    measured state, the finite-horizon QP
        min  1/2 * (sum_{k=0}^{N-1} ((x_k-xref)'Qx(x_k-xref)
                                     + (u_k-uref)'Qu(u_k-uref)
                                     + Du_k'QDu Du_k)
                    + (x_N-xref)'QxN(x_N-xref))
        s.t. x_{k+1} = Ad x_k + Bd u_k,  x_0 = x(t)
             xmin <= x_k <= xmax,  umin <= u_k <= umax
             Dumin <= u_k - u_{k-1} <= Dumax
    apply only u_0, re-measure, re-solve (receding horizon).
    QxN can be set to the LQR terminal cost; optional xfmin/xfmax tighten
    x_N to a terminal box.
    """

    def __init__(self, Ad, Bd, Np=20, x0=None, xref=None, uref=None, uminus1=None,
                 Qx=None, QxN=None, Qu=None, QDu=None,
                 xmin=None, xmax=None, umin=None, umax=None, Dumin=None, Dumax=None,
                 xfmin=None, xfmax=None):
        self.Ad, self.Bd = sparse.csc_matrix(Ad), sparse.csc_matrix(Bd)
        self.nx, self.nu = self.Bd.shape
        self.Np = Np
        # current state and last input: the two parameters that change each step
        self.x0 = np.asarray(x0 if x0 is not None else np.zeros(self.nx)).reshape(-1)
        self.uminus1 = np.asarray(uminus1 if uminus1 is not None else np.zeros(self.nu)).reshape(-1)
        self.xref = np.asarray(xref if xref is not None else np.zeros(self.nx)).reshape(-1)
        self.uref = np.asarray(uref if uref is not None else np.zeros(self.nu)).reshape(-1)
        # weights: stage state/input, terminal, and the rate (move-suppression) penalty
        self.Qx = sparse.csc_matrix(Qx if Qx is not None else np.zeros((self.nx, self.nx)))
        self.QxN = sparse.csc_matrix(QxN if QxN is not None else self.Qx)
        self.Qu = sparse.csc_matrix(Qu if Qu is not None else np.zeros((self.nu, self.nu)))
        self.QDu = sparse.csc_matrix(QDu if QDu is not None else np.zeros((self.nu, self.nu)))
        # hard boxes; default to +-inf where unset
        self.xmin = np.asarray(xmin if xmin is not None else -np.inf*np.ones(self.nx)).reshape(-1)
        self.xmax = np.asarray(xmax if xmax is not None else  np.inf*np.ones(self.nx)).reshape(-1)
        self.umin = np.asarray(umin if umin is not None else -np.inf*np.ones(self.nu)).reshape(-1)
        self.umax = np.asarray(umax if umax is not None else  np.inf*np.ones(self.nu)).reshape(-1)
        self.Dumin = np.asarray(Dumin if Dumin is not None else -np.inf*np.ones(self.nu)).reshape(-1)
        self.Dumax = np.asarray(Dumax if Dumax is not None else  np.inf*np.ones(self.nu)).reshape(-1)
        self.xfmin = np.asarray(xfmin if xfmin is not None else self.xmin).reshape(-1)
        self.xfmax = np.asarray(xfmax if xfmax is not None else self.xmax).reshape(-1)
        self.prob = osqp.OSQP()

    @staticmethod
    def _mv(M, v):
        return np.asarray(M @ v).reshape(-1)

    def _build(self):
        Np, nx, nu = self.Np, self.nx, self.nu
        Ad, Bd = self.Ad, self.Bd

        # --- cost: block-diagonal Hessian over (x_0..x_N, u_0..u_{N-1}) ---
        # state part: Qx on x_0..x_{N-1}, terminal QxN on x_N
        P_x = sparse.block_diag(
            [sparse.kron(sparse.eye(Np, format='csc'), self.Qx), self.QxN],
            format='csc')
        # D U = [u_0, u_1-u_0, ..., u_{N-1}-u_{N-2}]; D'D gives the
        # tridiagonal rate-penalty Hessian.
        D = sparse.eye(Np, format='csc') - sparse.eye(Np, k=-1, format='csc')
        P_u = sparse.kron(sparse.eye(Np, format='csc'), self.Qu) \
            + sparse.kron((D.T @ D).tocsc(), self.QDu)
        P = sparse.block_diag([P_x, P_u], format='csc')
        # linear term pulls states/inputs toward the references; the Du term also
        # injects -QDu u_{-1} into the first input (couples to last applied input)
        q_x = np.hstack([np.tile(-self._mv(self.Qx, self.xref), Np),
                         -self._mv(self.QxN, self.xref)])
        q_u = np.tile(-self._mv(self.Qu, self.uref), Np)
        q_u[:nu] += -self._mv(self.QDu, self.uminus1)
        q = np.hstack([q_x, q_u])

        # --- equality constraints: dynamics  Ad x_k + Bd u_k - x_{k+1} = 0 ---
        # and the initial condition x_0 = x(t), all as l == u rows
        Ax = sparse.kron(sparse.eye(Np+1), -sparse.eye(nx)) \
             + sparse.kron(sparse.eye(Np+1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, Np)),
                                        sparse.eye(Np)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-self.x0, np.zeros(Np*nx)]); ueq = leq

        # --- inequality (box) constraints on every x_k and u_k ---
        Aineq = sparse.eye((Np+1)*nx + Np*nu)
        xmin_stack = np.tile(self.xmin, Np+1); xmin_stack[-nx:] = self.xfmin
        xmax_stack = np.tile(self.xmax, Np+1); xmax_stack[-nx:] = self.xfmax
        lineq = np.hstack([xmin_stack, np.tile(self.umin, Np)])
        uineq = np.hstack([xmax_stack, np.tile(self.umax, Np)])

        # --- rate (slew-rate) constraints  Dumin <= u_k - u_{k-1} <= Dumax ---
        # first block row is u_0 alone, with u_{-1} carried in the bound; the
        # remaining block rows are u_k - u_{k-1}.
        Adu = sparse.hstack([
            sparse.csc_matrix((Np*nu, (Np+1)*nx)),
            sparse.kron(D, sparse.eye(nu, format='csc'))
        ])
        ldu = np.tile(self.Dumin, Np)
        udu = np.tile(self.Dumax, Np)
        ldu[:nu] += self.uminus1
        udu[:nu] += self.uminus1

        A = sparse.vstack([Aeq, Aineq, Adu]).tocsc()
        l = np.hstack([leq, lineq, ldu]); u = np.hstack([ueq, uineq, udu])
        self.P, self.q, self.A, self.l, self.u = P, q, A, l, u
        # row index where the rate-bound block starts: dynamics-eq rows
        # (Np+1)*nx, then box rows (Np+1)*nx + Np*nu
        self._rate0 = (Np+1)*nx + (Np+1)*nx + Np*nu

    def setup(self):
        self._build()
        self.prob.setup(self.P, self.q, self.A, self.l, self.u,
                        warm_start=True, verbose=False)

    def step(self):
        res = self.prob.solve()
        if res.info.status != 'solved':
            raise ValueError('QP infeasible / not solved')
        # the decision vector is (x_0..x_N, u_0..u_{N-1}); take the FIRST input
        base = (self.Np+1)*self.nx
        u0 = res.x[base:base+self.nu].copy()
        self.uminus1 = u0
        return u0

    def update(self, x_meas, u_prev=None):
        # re-insert the new measured state into the x_0 = x(t) equality rows,
        # refresh the first rate bound (u_0 - u_{-1}) and the Du linear term with
        # the last input, then warm-start re-solve. Only RHS / q change.
        self.x0 = np.asarray(x_meas).reshape(-1)
        if u_prev is not None:
            self.uminus1 = np.asarray(u_prev).reshape(-1)
        self.l[:self.nx] = -self.x0
        self.u[:self.nx] = -self.x0
        r = self._rate0
        self.l[r:r+self.nu] = self.Dumin + self.uminus1
        self.u[r:r+self.nu] = self.Dumax + self.uminus1
        # the Du-on-first-input contribution sits in the u_0 slot of q
        base = (self.Np+1)*self.nx
        self.q[base:base+self.nu] = -self._mv(self.Qu, self.uref) - self._mv(self.QDu, self.uminus1)
        self.prob.update(l=self.l, u=self.u, q=self.q)
```

The terminal weight `QxN` is where the LQR connection lives — set it to the discrete Riccati solution when the terminal region is chosen so the LQR move is admissible:

```python
def lqr_terminal(Ad, Bd, Q, R):
    """Terminal cost option: LQR infinite-horizon value matrix S (and gain K).
    Splicing x_N'S x_N onto the horizon makes the finite-horizon cost stand in
    for the infinite-horizon cost, so the receding-horizon loop is stabilizing."""
    S = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad)
    return K, S
```

And the closed loop is the receding-horizon loop itself: solve, apply the first input, step the real plant, re-measure, re-solve.

```python
def closed_loop(Ad, Bd, ctrl, x0, nsim):
    x = x0
    xs, us = [], []
    for t in range(nsim):
        u = ctrl.step()              # solve the QP from the current state
        x = Ad @ x + Bd @ u          # real plant advances under the applied input
        ctrl.update(x)               # re-measure, re-assemble (feedback)
        xs.append(x.copy()); us.append(u.copy())
    return np.array(xs), np.array(us)
```
