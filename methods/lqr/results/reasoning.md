Let me start from what's actually painful, because the math should be in service of fixing it. I have a plant — some physical thing, a vehicle, a servo — and I can write its small-signal behavior in state-space form, `ẋ = Fx + Gu`, with `x` the state vector and `u` the controls I get to choose. I want a controller that pushes the state back to zero with a clean transient. The way everyone does this right now is loop-shaping: draw the Bode or Nyquist plot of one loop, slap a lead-lag compensator on it, tune by hand until the phase and gain margins look acceptable. And that's fine when there's one input and one output. But the moment the plant has several inputs and several outputs that talk to each other — a genuinely multivariable plant — I'm stuck tuning loop after loop and fighting the cross-coupling by trial and error. There's no notion of choosing *all* the gains together so they're jointly best. There's no algorithm that eats the matrices `(F, G)` and hands me back the controller. And there's nothing telling me the thing I tuned is even stable. I want all three: optimal, constructive, for arbitrary order and MIMO, with a stability guarantee.

So the first move is to stop tuning and write down what "good transient" actually means as a number I can minimize. That idea isn't new — Wiener and Hall, and then Newton, Gould and Kaiser, already said: minimize the integral of the squared error, `∫ e² dt`. That's the right instinct; it converts "the transient looks good" into one scalar. But their machinery was a spectral/transfer-function affair that only really worked for low-order systems and never became a clean state-space theory or a constructive feedback law. I want to keep their objective and re-pose it in state space.

What should the integrand be? I'm penalizing the transient, so I want a measure of how far the state is from zero, integrated over time. The cheapest honest choice is a quadratic form in the state, `x'Qx`, with `Q` symmetric and positive semidefinite — `Q` says which combinations of states I care about and how much. But if that's the *only* term, the minimizer is degenerate: nothing stops me from demanding infinite control to crush the state instantly, and real actuators saturate and real control costs energy. So I have to pay for control too. Add `u'Ru`, `R` symmetric. So

    J = ∫₀^∞ ( x'Qx + u'Ru ) dt,

and `(Q, R)` is exactly the knob the old methods lacked — turn `R` down and I buy aggressive regulation at the price of big inputs; turn it up and I get gentle control. That's the trade-off made explicit.

Why quadratic, though, and not, say, `|x|` or `x⁴`? Partly heritage — it's the ISE idea. But there's a sharper reason I can feel coming: with linear dynamics, a quadratic cost is the one combination where the optimization is going to *close in feedback form*. The running cost is a sum of two quadratic forms; its Hessian in `u` is `2R`. If I insist `R` is strictly positive definite (not merely semidefinite), then the cost is *strictly* convex in `u` and `R⁻¹` exists — which is precisely the second-variation / Legendre condition for a genuine minimum, `∂²L/∂u² ≻ 0`. So `R ≻ 0` isn't decoration; it's what makes the inner minimization over `u` well-posed and uniquely solved. (If `R` were singular, some control direction would be free and the problem would be ill-posed — you'd want to apply infinite control in that direction.) `Q` can stay merely `≥ 0`: I don't have to penalize every state, only enough of them.

Now, how do I actually find the minimizing controller. Two routes are on the table; the one I instinctively reach for first is the calculus of variations / maximum-principle machinery. Treat this as minimizing `∫L dt` subject to `ẋ = Fx + Gu`. Adjoin the dynamics with a costate (multiplier) `ξ` and form the Hamiltonian

    H = L + ⟨ξ, Fx + Gu⟩ = x'Qx + u'Ru + ξ'(Fx + Gu).

Stationarity in `u`: `∂H/∂u = 2Ru + G'ξ = 0`, so `u = −½R⁻¹G'ξ`. The costate evolves by `ξ̇ = −∂H/∂x` and the state by `ẋ = ∂H/∂ξ`, giving a coupled linear system in `(x, ξ)` — the canonical equations — with `x` pinned at the start and `ξ` pinned at the end. That's a two-point boundary-value problem. I can solve it. But look at what I get: an optimal *input signal* `u*(t)` tied to one specific initial state. If the plant gets bumped to a new state, the boundary-value problem changes and I have to solve it all over again. That's open-loop. It is *not* a feedback law `u = k(x)` I can evaluate from whatever state I'm currently in. And feedback is the entire point — it's what gives disturbance rejection and what a controller actually *is*. So this route, taken literally, gives me the wrong kind of object. Wall.

The fix is to change viewpoint to the one that's natively closed-loop: dynamic programming. Instead of asking for the optimal trajectory from one start, define the *cost-to-go*

    V°(x, t) = the minimum remaining cost if I'm at state x at time t.

Bellman's principle of optimality says the tail of an optimal trajectory is itself optimal, and that gives, in the limit of small time steps, a partial differential equation. Take a tiny step `dt`: the cost-to-go now equals the running cost over `dt` plus the cost-to-go from where I land, minimized over the control I pick now:

    V°(x,t) = min_u [ (x'Qx + u'Ru) dt + V°(x + (Fx+Gu)dt, t+dt) ].

Expand the second term to first order, `V° + V°_t dt + V°_x·(Fx+Gu) dt`, cancel `V°`, divide by `dt`:

    0 = V°_t + min_u [ x'Qx + u'Ru + V°_x · (Fx + Gu) ].

That's the Hamilton–Jacobi–Bellman equation. The minimizing `u` inside is automatically a function of the current state, because it depends on `V°_x` evaluated at `x`. So whatever pops out of that inner `min` *is* a feedback law. This is the closed-loop object the variational route couldn't give me. The price is that the HJB PDE is, in general, hopeless — a PDE in `n` state variables. I need a structural assumption that collapses it.

Here's where the linear-quadratic structure earns its keep. The running cost is quadratic in `x` and the dynamics are linear. By symmetry, the cost-to-go starting from `x` ought to be quadratic in `x` too — scaling the initial state by `c` scales every reachable trajectory's state-and-control by `c` (the dynamics are linear) and hence scales the cost by `c²`. So I'll *guess* the value function is a quadratic form:

    V°(x, t) = x' S(t) x,   S symmetric.

That's an ansatz, and it's the move that turns the PDE into something finite-dimensional, because a quadratic form is fully described by the matrix `S`. Then `V°_x = 2 S x` and `V°_t = x' Ṡ x`.

Substitute into the HJB and stare at the inner minimization:

    min_u [ x'Qx + u'Ru + 2 x'S (Fx + Gu) ].

The only `u`-dependent pieces are `u'Ru + 2 x'S G u`. That's an unconstrained quadratic in `u`, and because `R ≻ 0` it's strictly convex, so I just set its gradient to zero:

    ∂/∂u ( u'Ru + 2 x'SG u ) = 2Ru + 2 G'S x = 0   ⟹   u* = −R⁻¹ G' S x.

There it is. The optimal control is `−R⁻¹G'Sx` — *linear in the state*. I didn't assume linear feedback; I assumed a quadratic value function and the linear feedback fell out of the inner minimization. Write it as

    u* = −K x,   K = R⁻¹ G' S.

The minus sign matters and I should hold onto why it's there: it came straight from `2Ru = −2G'Sx`. It's *negative* feedback. Whether it actually stabilizes is something I still have to prove — but the sign that has any chance of stabilizing is this one, and I'll confirm it places the closed-loop dynamics in the left half-plane below.

Now push `u*` back into the HJB to find the equation `S` must satisfy. Plug `u* = −R⁻¹G'Sx` into `u'Ru + 2x'SGu`:

    u*'Ru* = x'S G R⁻¹ R R⁻¹ G'S x = x'S G R⁻¹ G'S x,
    2 x'S G u* = −2 x'S G R⁻¹ G'S x.

Their sum is `−x'S G R⁻¹ G'S x`. So the HJB becomes, collecting every term as a quadratic form in `x`:

    x' Ṡ x + x'Qx + 2 x'S F x − x'S G R⁻¹ G'S x = 0.

The cross term `2x'SFx` I should symmetrize, since `S` is symmetric: `2x'SFx = x'(SF + F'S)x`. So the bracket-free statement, holding for *all* `x`, forces the symmetric matrix in the middle to vanish:

    Ṡ + Q + S F + F'S − S G R⁻¹ G'S = 0,

i.e.

    −Ṡ = F'S + S F − S G R⁻¹ G'S + Q,   S(t_f) = Q_f.

That's a matrix differential equation of *Riccati* type — quadratic in the unknown `S` through the `−SGR⁻¹G'S` term. It runs *backward* in time from the terminal weight `S(t_f) = Q_f` (the cost-to-go at the final time is just the terminal penalty). For the finite-horizon problem, integrating this backward gives `S(t)` and hence a *time-varying* optimal gain `K(t) = R⁻¹G'S(t)`. I notice this Riccati object is the same kind of equation that was already known to show up in the second-variation analysis of the calculus of variations — but here it's not a side-condition, it's *the algorithm* that produces the feedback gain.

Now the case I actually want is the infinite horizon, `t_f → ∞`, which is what makes the controller a fixed gain rather than a clock-dependent one. Intuitively, far from the terminal time the backward solution should forget the terminal condition and settle to a constant: `Ṡ → 0`. Setting `Ṡ = 0` in the Riccati ODE gives the *algebraic* Riccati equation

    F'S + S F − S G R⁻¹ G'S + Q = 0,

and the optimal control is the *constant*, *time-invariant* state feedback

    u* = −K x,   K = R⁻¹ G' S.

This is the payoff: a single matrix equation in `S`, and once I solve it the gain is just a multiply. Constructive, MIMO, any order. But I've been sloppy in two places and I need to pin them down, because "set `Ṡ = 0`" is a wish, not a proof. First: does the backward solution actually *converge* to a constant `S`, and is that constant the right (stabilizing) one? Second: even granting a constant `S`, is the closed loop `F − GK` actually stable? Both of these turn out to need hypotheses on the plant.

Take convergence first. The infinite-horizon cost-to-go `x'Sx` is the *minimum* total cost from `x`. For that to be finite at all, there has to exist *some* control that drives `x` to zero with finite cost — otherwise the cost-to-go is infinite and the whole problem is vacuous. So I need: from any state, I can reach the origin in finite time with finite control energy. That's a property of `(F, G)` alone, and it deserves a name — call the plant **completely controllable** if every state can be steered to zero by some admissible control on a finite interval. I can make this concrete. The reachability of the origin from `x` is governed by the symmetric matrix

    W(t₀, t₁) = ∫_{t₀}^{t₁} Φ(t₀, t) G G' Φ'(t₀, t) dt,

the controllability Gramian. The plant is completely controllable exactly when `W` is positive definite for some `t₁ > t₀`: if `W ≻ 0` then `u(t) = −G'Φ'(t₀,t)W⁻¹x` is an explicit control that sends `x` to the origin, and the energy of that control is `x'W⁻¹x` — finite. Conversely if `W` is only semidefinite, there's a direction `x ≠ 0` annihilated by it that you provably cannot steer. So `W ≻ 0` is the clean characterization. For a *time-invariant* plant I can grind `W ≻ 0` down further: expand `Φ = e^{Ft}`, and `W` is singular iff some `x ≠ 0` has `x'e^{Ft}G = 0` for all `t`; differentiating repeatedly at `t = 0` gives `x'G = x'FG = ... = 0`, and by Cayley–Hamilton (every higher power of `F` is a combination of `I,...,Fⁿ⁻¹`) those are *all* the conditions. So complete controllability ⇔

    rank [ G, FG, F²G, …, Fⁿ⁻¹G ] = n.

That's the controllability rank test, and the point of it here is that controllability is what makes the infinite-horizon cost finite, which is what makes the backward Riccati solution `Π(t; 0, t₁)` converge to a constant `S̄` as `t₁ → ∞`. The argument is clean: for any `x`, pick a control that drives it to zero by some fixed time and is zero afterward — that control's cost is an upper bound on the optimal cost-to-go, finite and independent of `t₁`, so `Π` is bounded above; and extending the horizon can only add nonnegative cost, so `Π` is nondecreasing in `t₁`. Bounded and monotone ⟹ the limit exists, and by continuity it solves the algebraic Riccati equation. Good — that licenses "set `Ṡ = 0`."

Now stability, and this is the trap. It is widely assumed — tacitly, and it's wrong — that any controller minimizing a cost must be stabilizing. It isn't automatic: minimizing a *finite* cost does not by itself force the state to decay to zero. Picture a mode of the plant that the cost simply doesn't see — it doesn't appear in `Q` — and that costs essentially nothing to leave alone. The cost-to-go can be finite while that mode quietly runs off to infinity. So stability is a genuinely separate property and I have to *prove* it, not assume it. Lyapunov's second method is the right tool, and beautifully, the value function itself is the natural candidate. Take `V°(x) = x'Sx`. Along the optimal closed-loop motion, what is its time-derivative? Differentiate the HJB story: the HJB at the optimum says `V°_x·(Fx + Gu*) + (x'Qx + u*'Ru*) = 0` (the `V°_t` term is zero in steady state), i.e.

    d/dt V°(x(t)) = V°_x · ẋ = −( x'Qx + u*'Ru* ).

The cost-to-go *decreases along trajectories at exactly the instantaneous running cost rate.* So `V̇° ≤ 0` always, and it's strictly negative as long as `x'Qx + u*'Ru* > 0`. With `V° = x'Sx > 0` for `x ≠ 0` and `V° → ∞` as `‖x‖ → ∞`, that's a Lyapunov function and the closed loop is asymptotically stable — *provided* the running cost can't stay zero along a nonzero trajectory. That's the missing hypothesis: I need the cost to actually "see" every mode that the control can't kill on its own. If `Q = H'(·)H` weights an output, the condition is that the plant be **completely observable** through that output — the dual notion to controllability (it's literally controllability of the time-reversed dual plant `F↦F'`, `G↦H'`), with its own Gramian and, in the time-invariant case, its own rank test `rank[H'; F'H'; …; (F')ⁿ⁻¹H'] = n`. Observability is what rules out the cheap-but-unobserved runaway mode: if every mode is visible in the cost, `x'Qx + u'Ru` can't vanish along a nonzero motion, so `V̇° ≺ 0` strictly and `F − GK` is Hurwitz. (To make these uniform statements for time-varying plants I'd also bound `Q, R` above and below by positive constants, so the Lyapunov function is sandwiched by increasing functions of `‖x‖` on both sides — but the structure is exactly this.) So: controllability ⟹ finite cost ⟹ the algebraic Riccati solution exists; observability ⟹ the optimal closed loop is actually stable. Both conditions earn their place; neither is decoration.

Two loose ends before code. First, the sign sanity check I promised: the closed loop is `ẋ = (F − GK)x = (F − G R⁻¹ G'S)x`. Is `F − GR⁻¹G'S` stable? That's exactly what the Lyapunov argument just established — `x'Sx` decreases along it — so yes, the `−K` (negative-feedback) sign is the stabilizing one. Good; if I'd taken `+K` the same `V°` would *increase* and I'd have the wrong sign. Second: how do I actually solve the algebraic Riccati equation numerically? It's quadratic in `S`, so I can't just invert something. But the costate route from before secretly hands me the answer. The canonical state/costate equations of the linear-quadratic problem are linear in `(x, ξ)` with the block matrix

    Z = [ F      −G R⁻¹ G' ;
          −Q     −F'        ].

This `2n×2n` Hamiltonian matrix has a spectral symmetry: its eigenvalues come in `±` pairs (if `λ` is an eigenvalue so is `−λ`). The stabilizing solution `S` is the one whose graph `{(x, Sx)}` is the *stable* invariant subspace of `Z` — the span of the eigenvectors with negative real part. Concretely: compute an ordered Schur (or eigen-) decomposition of `Z` that puts the stable eigenvalues first, take the leading `n` columns `[U₁; U₂]` of the transformation, and `S = U₂ U₁⁻¹`. That's the modern Riccati solver — and it's the direct descendant of the old transition-matrix formula `P(t) = [Θ₂₁ + Θ₂₂P(t₁)][Θ₁₁ + Θ₁₂P(t₁)]⁻¹` from propagating the canonical equations, just stated in terms of the stable subspace instead of a terminal condition. In code I'll lean on the library that forms this Hamiltonian pencil and does the ordered decomposition for me; I just have to hand it `(F, G, Q, R)`.

Let me land it. I'll rename to the textbook letters `ẋ = Ax + Bu`, cost `∫(x'Qx + u'Ru)dt`. The whole synthesis is: solve the algebraic Riccati equation for `S`, then `K = R⁻¹B'S`, then `u = −Kx`.

```python
import numpy as np
import scipy.linalg
from scipy.integrate import odeint


def lqr(A, B, Q, R):
    """Continuous-time infinite-horizon optimal state feedback.
    Plant   dx/dt = A x + B u
    Cost    J = ∫ ( x'Q x + u'R u ) dt,   Q = Q' >= 0,  R = R' > 0.
    """
    # Solve the algebraic Riccati equation  A'S + SA - S B R^{-1} B'S + Q = 0
    # for the symmetric cost-to-go matrix S (the value function is x'Sx).
    # The solver forms the 2n×2n Hamiltonian pencil and takes its stable
    # invariant subspace — the descendant of propagating the canonical
    # state/costate equations — to pick the stabilizing root.
    S = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # Read the feedback gain off the value function: the inner min over u in
    # the HJB gave u* = -R^{-1} B'S x, i.e. K = R^{-1} B'S.
    K = np.linalg.solve(R, B.T @ S)

    # Closed-loop poles A - B K should all sit in the open left half-plane;
    # that the Lyapunov function x'Sx decays along motions is exactly why.
    cl_poles = scipy.linalg.eig(A - B @ K)[0]
    return K, S, cl_poles
```

And exercising it on a genuinely multivariable plant — a quadrotor linearized about hover — shows the structure the SISO methods could never handle. Hover decouples the small-signal dynamics into four channels, and the load-bearing physics is that horizontal position is steered through *attitude*: tilting the pitch by `θ` accelerates `x` by `g·θ`, so the `x`-channel state is `[x, ẋ, θ, θ̇]` with gravity coupling position to tilt, and the input is a pitch torque. That `g`-coupling is exactly the cross-channel interaction hand-tuning would have to fight; here it's just an entry in `A` and the Riccati solver handles all the gains jointly.

```python
g, m, Ix, Iy, Iz = 9.81, 1.0, 0.1, 0.1, 0.2

# x-channel: state [x, xdot, pitch, pitchdot]; tilt accelerates x by g.
Ax = np.array([[0, 1, 0, 0],
               [0, 0, g, 0],     # ẍ = g · pitch  (gravity coupling)
               [0, 0, 0, 1],
               [0, 0, 0, 0]])
Bx = np.array([[0], [0], [0], [1 / Ix]])   # input is a pitch torque

Q = np.eye(4); Q[0, 0] = 10.0   # weight position error most; R is the effort knob
R = np.array([[1.0]])

K, S, poles = lqr(Ax, Bx, Q, R)   # u = -K (x - x_ref): one gain, all states jointly
```

The whole chain, in one breath: I wanted feedback, not a hand-tuned loop — so I wrote "good transient + bounded effort" as `J = ∫(x'Qx + u'Ru)dt` with `(Q,R)` the trade-off knob; the open-loop variational route gave only `u*(t)`, so I switched to the cost-to-go `V°(x)` and the HJB; guessing `V° = x'Sx` collapsed the PDE, and the inner min over `u` dropped out *linear* feedback `u = −R⁻¹B'Sx` with `S` solving the algebraic Riccati equation `A'S + SA − SBR⁻¹B'S + Q = 0`; controllability makes the infinite-horizon cost finite so that `S` exists, observability makes `x'Sx` a strict Lyapunov function so the closed loop is actually stable; and the gain is computed by taking the stable invariant subspace of the Hamiltonian matrix — `solve_continuous_are`, `K = R⁻¹B'S`, `u = −Kx`.
