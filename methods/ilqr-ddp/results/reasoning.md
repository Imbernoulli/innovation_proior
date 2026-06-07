Let me start from the one thing I actually trust and figure out exactly where it breaks. For a linear plant `x_{i+1} = A x_i + B u_i` and a quadratic cost `Σ (x_i'Q x_i + u_i'R u_i) + x_N'Q_f x_N`, I have the complete answer and I can write it from memory: the value function `V(x,i)` is quadratic, `x'S_i x`; Bellman's principle of optimality turns the minimization over the whole control sequence into a backward sweep of single-step minimizations; each single-step minimization is an unconstrained convex quadratic in `u` because `R ≻ 0`, so I set the gradient to zero and get linear feedback `u_i = −K_i x_i` with `K_i = (R + B'S_{i+1}B)^{-1}B'S_{i+1}A`, and `S_i` falls out of the backward Riccati recursion `S_i = A'S_{i+1}A − A'S_{i+1}B(R+B'S_{i+1}B)^{-1}B'S_{i+1}A + Q`. One sweep from `S_N = Q_f` and I have every gain, exactly, with feedback baked in. The property that makes all of this work is that the value function stays *quadratic* under the backward step — quadratic in, quadratic out — and that the per-step minimization is therefore a clean convex quadratic.

Now the plant I actually care about is `x_{i+1} = f(x_i, u_i)` with `f` a genuine nonlinearity — a multi-link arm doing a big reaching movement, a pendulum swinging all the way up, a hopper. And `ℓ` may be a general smooth cost, not just `x'Qx`. The moment `f` is nonlinear, the value function is no longer quadratic. `V(f(x,u), i+1)` is the composition of a quadratic-ish thing with a nonlinear `f`, which is some arbitrary nonlinear function of `(x,u)`; the bracket inside Bellman's min is no longer a quadratic I can minimize in closed form; there is no finite matrix `S_i` to propagate. The backward recursion doesn't close. The whole machine seizes.

The brute-force escape is to stop using the recursion: stack the entire control sequence `(u_0,…,u_{N-1})` into one big vector, write `J` as a function of it (with the states eliminated by rolling out `f`), and throw it at a general nonlinear-programming solver. That works in the sense that an optimizer will find a local minimum. But it throws away the two things I most wanted to keep. First, it discards the stage-wise, backward-in-time structure — the Markov decomposition that made the linear case a single sweep instead of a giant solve. Second, and worse for control, a generic NLP hands me back an *open-loop* sequence `u_0*,…,u_{N-1}*` and nothing else. There is no `K_i`. If the plant gets bumped off this trajectory, I have no feedback law to pull it back. The dynamic-programming view gave me `K_i` for *free*, as the by-product of the per-step minimization. I don't want to lose that. So the real task is: keep the backward-recursion machinery of the linear-quadratic regulator, and make it survive a nonlinear `f`.

Here's the obvious lever. I don't have a *global* quadratic value function anymore, but I don't need one. I almost always have, or can cheaply get, a *nominal* trajectory `(x̄_0,…,x̄_N)` with its controls `(ū_0,…,ū_{N-1})` — even a bad one, even `ū ≡ 0` rolled forward through `f`. What I really want is the best *correction* to it. So let me work in deviations: `δx_i = x_i − x̄_i`, `δu_i = u_i − ū_i`, and ask only for a *local* model of the value function around the nominal — accurate near `(x̄, ū)`, where I'm actually going to move. Locally, a smooth `V` looks quadratic. That's the crack to pry open: don't approximate `V` over all of state space; approximate it to second order around the nominal, at each step, and let the backward recursion propagate that local quadratic the way it propagated the exact quadratic in the linear case.

Which object do I expand? The temptation is to Taylor-expand `V(x,i)` directly. But the thing the Bellman step actually minimizes is the bracket, `ℓ(x,u) + V(f(x,u), i+1)`, as a function of the control. Let me name that bracket, measured as a deviation from its nominal value, and expand *it*. Define

    Q(δx, δu) = ℓ(x̄+δx, ū+δu) − ℓ(x̄, ū) + V(f(x̄+δx, ū+δu), i+1) − V(f(x̄, ū), i+1).

This `Q` is the local change in cost-to-go if, sitting near the nominal at step `i`, I perturb the state by `δx` and the control by `δu`. If I have a quadratic model of `V` at step `i+1` — a gradient `V'_x` and a Hessian `V'_xx`, where I'll write the prime to mean "next step" — then I can expand `Q` to second order in `(δx, δu)` and, crucially, *minimize over `δu`*, which is the Bellman min. Let me actually do the expansion, carefully, because the whole method lives or dies on getting these coefficients right.

The cost part is easy: `ℓ(x̄+δx, ū+δu) − ℓ(x̄,ū) ≈ ℓ_x'δx + ℓ_u'δu + ½[δx;δu]'[[ℓ_xx, ℓ_xu],[ℓ_ux, ℓ_uu]][δx;δu]`. The interesting part is the value term `V(f(x̄+δx, ū+δu)) − V(f(x̄,ū))`, because `f` sits inside `V`. Let `δf = f(x̄+δx, ū+δu) − f(x̄,ū)`. To second order, `V(f̄ + δf) − V(f̄) ≈ V'_x · δf + ½ δf' V'_xx δf`. Now I need `δf` itself to second order in `(δx, δu)`:

    δf ≈ f_x δx + f_u δu + ½(δx,δu)·∂²f·(δx,δu),

where `f_x = ∂f/∂x`, `f_u = ∂f/∂u` are the Jacobians along the nominal and `∂²f` is the rank-three tensor of second derivatives. Plug this into `V'_x·δf + ½ δf'V'_xx δf` and collect by order. The first-order-in-`δf` term `V'_x·δf` contributes `V'_x·(f_x δx + f_u δu)` at first order, and at second order it contributes `½ V'_x·(∂²f)(·,·)` — the gradient of `V` dotted into the *curvature of the dynamics*. The `½ δf'V'_xx δf` term, to leading (second) order, contributes `½(f_x δx + f_u δu)' V'_xx (f_x δx + f_u δu)`. So, matching the linear and quadratic coefficients of `Q(δx,δu)` against the symmetric block form

    Q ≈ Q_x'δx + Q_u'δu + ½[δx;δu]'[[Q_xx, Q_xu],[Q_ux, Q_uu]][δx;δu],

I can read off every coefficient:

    Q_x  = ℓ_x + f_x' V'_x
    Q_u  = ℓ_u + f_u' V'_x
    Q_xx = ℓ_xx + f_x' V'_xx f_x + V'_x · f_xx
    Q_uu = ℓ_uu + f_u' V'_xx f_u + V'_x · f_uu
    Q_ux = ℓ_ux + f_u' V'_xx f_x + V'_x · f_ux.

Let me make sure I believe each piece. `Q_x = ℓ_x + f_x'V'_x` is the chain rule: a state nudge changes the immediate cost (`ℓ_x`) and changes where I land (`f_x δx`), which costs `V'_x` per unit landing-displacement, transposed back through `f_x`. Same logic for `Q_u`. The `f_x'V'_xx f_x` in `Q_xx` is the value's curvature *pulled back* through the linearized dynamics — a nudge `δx` propagates to `f_x δx` in next-state space, where it feels curvature `V'_xx`. The `f_u'V'_xx f_x` in `Q_ux` is the same pull-back but mixing a control nudge into a state nudge through the dynamics — this cross term is exactly what will couple the optimal control correction to the state deviation, i.e. it is where feedback comes from. Good, the structure is the LQR Riccati pull-back generalized to time-varying linearizations `f_x, f_u`.

And those last terms, `V'_x · f_xx`, `V'_x · f_uu`, `V'_x · f_ux` — the value gradient contracted with the *second derivative of the dynamics* — are the genuinely new objects, the ones the linear case never had because for a linear plant `f_xx = 0`. I'll come back to them; they're going to be both the source of the method's fast convergence and a source of trouble. For now keep them.

Now the Bellman step: minimize `Q(δx, δu)` over `δu` for a given `δx`. With `Q` quadratic and `Q_uu` positive definite, this is exactly the clean convex quadratic minimization I had in LQR. Set the gradient in `δu` to zero:

    ∂Q/∂δu = Q_u + Q_uu δu + Q_ux δx = 0   ⟹   δu*(δx) = −Q_uu^{-1}(Q_u + Q_ux δx).

Split it into the two pieces that fall out naturally:

    δu*(δx) = k + K δx,   k = −Q_uu^{-1} Q_u,   K = −Q_uu^{-1} Q_ux.

There's the payoff, and notice what each half is. `k = −Q_uu^{-1}Q_u` is a *feedforward* correction: it's the Newton-step-in-the-control on the nominal, the part that improves the trajectory itself, and it's independent of `δx`. `K = −Q_uu^{-1}Q_ux` is a *feedback* gain: it says how to adjust the control in response to a state deviation `δx`. I did not put feedback in by hand — it dropped out of the cross term `Q_ux` in the per-step minimization, exactly as `K_i` dropped out of the cross term in the LQR Riccati derivation. The sign is forced: `Q_uu δu = −(Q_u + Q_ux δx)`, so the inverse comes with a minus, negative feedback. If I'd flipped it I'd be climbing the quadratic, not descending it.

Now close the recursion: substitute `δu*(δx) = k + Kδx` back into `Q(δx, δu)` to get the local quadratic model of the value at step `i`, as a function of `δx` alone. This is the step that propagates the quadratic backward and must stay quadratic-in-quadratic-out, or the recursion doesn't close. Write `Q` in block form and substitute `δu = k + Kδx`. Collecting the constant, linear, and quadratic-in-`δx` terms gives the new value model. Let me grind the algebra rather than wave at it. With

    Q = const + Q_x'δx + Q_u'δu + ½ δx'Q_xx δx + δu'Q_ux δx + ½ δu'Q_uu δu,

substitute `δu = k + Kδx`. The terms:
- `Q_u'δu = Q_u'k + Q_u'Kδx`.
- `½ δu'Q_uu δu = ½(k+Kδx)'Q_uu(k+Kδx) = ½k'Q_uu k + k'Q_uu Kδx + ½δx'K'Q_uu Kδx`.
- `δu'Q_ux δx = (k+Kδx)'Q_ux δx = k'Q_ux δx + δx'K'Q_ux δx`.

Now gather by order in `δx`. The constant term (the value decrease `ΔV` from taking this step) is `Q_u'k + ½k'Q_uu k`. The coefficient of `δx` (the new `V_x`) is

    V_x = Q_x + Q_u'K  (as a row) + K'Q_uu k + Q_ux'k,

which I'll write columnwise as `V_x = Q_x + K'Q_u + K'Q_uu k + Q_ux'k`. The quadratic coefficient (the new `V_xx`) is

    V_xx = Q_xx + K'Q_uu K + K'Q_ux + Q_ux'K,

symmetrized. So the backward recursion is closed: given `(V'_x, V'_xx)` at step `i+1`, I form the five `Q` coefficients, get `k` and `K`, and produce `(V_x, V_xx)` at step `i`, plus a running tally of the predicted total cost decrease `ΔV = Σ (k'Q_u + ½k'Q_uu k)`. Initialize the sweep with the terminal value, which is just the final cost: `V_x(N) = ℓ_{f,x}`, `V_xx(N) = ℓ_{f,xx}`.

Let me sanity-check against the case I trust. Make `f` linear: `f_x = A`, `f_u = B`, and all the `f_xx`-type tensors vanish. Make `ℓ` the quadratic `x'Qx + u'Ru`, so `ℓ_xx = 2Q`, `ℓ_uu = 2R`, `ℓ_ux = 0`. Then `Q_uu = 2R + B'V'_xx B`, `Q_ux = B'V'_xx A`, `Q_xx = 2Q + A'V'_xx A`, and `K = −Q_uu^{-1}Q_ux = −(2R + B'V'_xx B)^{-1}B'V'_xx A`. That is precisely the LQR gain (up to the factor-of-2 convention from writing the cost without the ½), and the `V_xx` recursion reduces to the discrete Riccati recursion. Better still: because the model is then *exact* (a quadratic cost over linear dynamics has no higher-order error), one backward pass plus one correction lands on the exact optimum — convergence in a single iteration, exactly as LQR promises. So LQR is the special case where `f` is already linear, and the new thing is just running this same pull-back with the trajectory's *time-varying* linearizations and re-linearizing as the trajectory moves. The nonlinear problem became a sequence of time-varying LQR subproblems.

Before I trust this I should pin down a competing route I could have taken, because seeing why it's worse sharpens what I've got. Instead of Bellman, take the costate/Hamiltonian view: linearize the dynamics about the nominal, `δx_{i+1} = A_i δx_i + B_i δu_i`, form the Hamiltonian `H_i = ½(x̄+δx)'Q(x̄+δx) + ½(ū+δu)'R(ū+δu) + δλ_{i+1}'(A_iδx_i + B_iδu_i)`, and impose stationarity `∂H/∂δu = 0` and the costate recursion `δλ_i = ∂H/∂δx`. Stationarity gives `0 = R(ū+δu) + B'δλ_{i+1}`, i.e. `δu = −R^{-1}B'δλ_{i+1} − ū`. The trouble is that this is *open-loop*: `δu` is expressed through the costate `δλ_{i+1}`, not through the current state deviation `δx`. To turn it into feedback I'd have to *guess* an affine relation `δλ_i = S_i δx_i + v_i`, substitute it into the coupled state/costate equations, and crank the matrix-inversion lemma; out the other end comes a Riccati recursion `S_i = A'S_{i+1}(A − B K) + Q` with `K = (B'S_{i+1}B + R)^{-1}B'S_{i+1}A`, plus an *auxiliary affine sequence* `v_i = (A − BK)'v_{i+1} − K'Rū + Qx̄` that carries the nominal's first-order residual, and `δu` ends up as feedback `−Kδx` plus correction terms in `v_{i+1}` and `ū`. It gets to the same place — a feedback gain from a Riccati recursion plus a feedforward term — but only after positing the ansatz and doing the inversion-lemma gymnastics, and because it linearized `f`, it's first-order in the dynamics: it never had the `V'_x·f_xx` curvature terms. The Bellman/`Q`-expansion route gave me the same feedback `K` and feedforward `k` *directly* from one quadratic minimization, and it gives me the option of keeping the dynamics curvature. So the `Q`-expansion is the cleaner instrument. The `v_i` of the costate route is, recognizably, just my feedforward `k` propagated; they're two views of one object.

Now back to those curvature terms, `V'_x·f_xx` and friends, because they're the crux. If I keep them, my `Q_xx, Q_uu, Q_ux` carry the second derivative of the dynamics, and then the whole update is a genuine second-order (Newton) step on the trajectory — it should converge quadratically near the optimum, which is the dream. So why not always keep them? Two problems, and I have to feel both. First, cost: `f_xx` is an `n × n × n` tensor (one `n×n` Hessian per output coordinate of `f`), `f_ux` and `f_uu` likewise. For a plant with a dozen states that's already a lot of second derivatives to form and contract at every step of every iteration, and for the complex systems I care about — a high-DOF arm, a humanoid — computing the dynamics derivatives is the dominant cost, and second derivatives blow it up. Second, and more insidious: `V'_x·f_xx` has no definite sign. The Gauss–Newton-flavored part `f_u'V'_xx f_u` is at least positive semidefinite when `V'_xx ⪰ 0` (it's a congruence of a PSD matrix) and `ℓ_uu ≻ 0` makes `Q_uu` positive definite — which is exactly the condition I need to *invert* `Q_uu` and to know `δu*` is a minimizer, not a saddle. But adding `V'_x·f_uu` can destroy that positive definiteness whenever the value gradient and the dynamics curvature conspire, especially far from the optimum where `V'_x` is large. An indefinite `Q_uu` means `−Q_uu^{-1}Q_u` is not a descent direction at all — it's the Newton pathology, climbing toward a saddle.

So here's the trade I'll make. Drop the three dynamics-curvature tensor terms entirely:

    Q_xx = ℓ_xx + f_x' V'_xx f_x
    Q_uu = ℓ_uu + f_u' V'_xx f_u
    Q_ux = ℓ_ux + f_u' V'_xx f_x.

What I'm left with is exactly the *Gauss–Newton* approximation to the Hessian: I've kept the first-derivative-of-`f` outer products and discarded the residual-times-second-derivative term, just as Gauss–Newton does for nonlinear least squares. This buys three things at once. It's much cheaper — only the Jacobians `f_x, f_u` are needed, no rank-three tensors. It keeps `Q_uu` positive (semi)definite by construction whenever `ℓ_uu ≻ 0` *and* `V'_xx ⪰ 0` — the congruence `f_u'V'_xx f_u` is PSD only when the pulled-back `V'_xx` is — so in that regime the inverse exists and `δu*` is a real minimizer. And near a small-residual solution the dropped term is small, so the step is nearly the full Newton step and convergence is still fast (superlinear in practice), even if I've formally given up the exact quadratic rate. That's the iterative LQR: the dynamics enter only through their linearization, and the per-step problem is a time-varying LQR. The full second-order version — keeping `V'_x·f_xx` etc. — is differential dynamic programming proper; I'll keep it as a switchable option for problems where the curvature genuinely matters and stays benign, but the default is the Gauss–Newton drop.

Two failure modes remain, and both are familiar from second-order optimization, so I'll import the familiar fixes rather than invent anything. First failure: even with the Gauss–Newton form, far from the optimum `Q_uu` can come out non-positive-definite — `V'_xx` itself can drift indefinite through the recursion, or `ℓ_uu` may be only semidefinite — and then `−Q_uu^{-1}Q_u` is garbage. This is precisely when Levenberg–Marquardt damps a Newton step. The textbook move is to add a multiple of the identity to the control-Hessian, `Q̃_uu = Q_uu + μ I`, with `μ` a Levenberg–Marquardt parameter: as `μ → 0` it's the full second-order step, as `μ → ∞` it's a tiny gradient step `−(1/μ)Q_u`, and somewhere in between is a positive-definite-enough matrix I can safely invert. This is just adding a quadratic penalty on the *control* deviation around the nominal, which makes the step more conservative.

But let me stare at that for a second, because there's a defect. The control-based penalty `Q_uu + μI` penalizes `δu` directly, and the *same* `δu` has different effects at different times depending on how strongly it enters the dynamics through `f_u`. A unit control change where `f_u` is large moves the state a lot; where `f_u` is small, barely at all. So a uniform penalty on `δu` is the wrong currency. And worse, as `μ → ∞` with the `Q_uu + μI` scheme, the feedback gain `K = −(Q_uu+μI)^{-1}Q_ux → 0`: the controller stops correcting state deviations exactly when it's being most cautious, which throws away the feedback I worked to get. What I actually want to penalize is *state* deviation from the trusted nominal. So put the damping on `V'_xx` instead — add `μ I` to the next-step value Hessian *before* pulling it back through the dynamics:

    Q̃_uu = ℓ_uu + f_u'(V'_xx + μ I) f_u   (+ V'_x·f_uu, if DDP)
    Q̃_ux = ℓ_ux + f_u'(V'_xx + μ I) f_x   (+ V'_x·f_ux, if DDP)
    k = −Q̃_uu^{-1} Q_u,   K = −Q̃_uu^{-1} Q̃_ux.

Now the penalty is `μ·‖f_x δx + f_u δu‖²` in disguise — a penalty on the *state deviation* the step induces, in the natural metric the dynamics impose. As `μ → ∞`, `K` does *not* vanish; instead the step is forced to keep the new trajectory close to the old one in state space, which is exactly the conservative behavior I want — stay in the region where the linearization is trustworthy — while *retaining* feedback. That's the better regularization. (Note: because I'm modifying `V'_xx → V'_xx + μI` inside the `Q` terms, I should compute the value-function update `V_x, V_xx` from the `k`/`K` form I derived above — `V_x = Q_x + K'Q_u + K'Q_uu k + Q_ux'k` and the matching `V_xx` — rather than from the algebraically-equivalent-but-only-when-unregularized compact form `V_x = Q_x − Q_xu Q_uu^{-1}Q_u`, because once I've perturbed `Q_uu` the cancellations in the compact form no longer hold and would inject an error.)

For the schedule: when the backward pass hits a non-positive-definite `Q̃_uu`, I increase `μ` fast and restart the backward pass — the minimal `μ` that restores definiteness is often large, so I want geometric growth, `μ ← μ·Δ` with `Δ` itself growing, `Δ ← max(Δ_0, Δ·Δ_0)` from some base `Δ_0` (say 2). When a step succeeds I decrease `μ` to chase fast convergence, and snap it to zero once it drops below a floor `μ_min` (say `1e-6`). Quadratic-ish growth up, quadratic-ish decay down.

Second failure mode: suppose the backward pass succeeded and `k, K` are fine, but the *full* step still overshoots — the new trajectory strays so far that the linearization is no longer valid and the true cost actually goes *up*, or the rollout diverges. This is the line-search situation. Introduce a backtracking step size `0 < α ≤ 1` on the feedforward part:

    û_i = ū_i + α k_i + K_i (x̂_i − x̄_i),   x̂_{i+1} = f(x̂_i, û_i),   x̂_0 = x̄_0.

Two things to be careful about here. I scale only the feedforward `α k_i`, not the feedback: at `α = 0` the trajectory is unchanged (the feedforward is killed and `x̂_i ≡ x̄_i` so the feedback term is zero too), and as `α` grows from 0 to 1 I interpolate from "no change" to "full Newton/Gauss–Newton step." Keeping `K` at full strength means the feedback still corrects whatever divergence the smaller feedforward causes during the rollout. And the rollout uses the *true* nonlinear `f`, not the linear model — this is the step that makes `(x̂, û)` an actual feasible trajectory and gives the feedback gains their meaning. Computing the new controls by rolling forward through `f` while applying `K_i(x̂_i − x̄_i)` is precisely why the method converges fast: the feedback re-aims the control at the states actually being visited, so even when the feedforward is shortened the trajectory tracks sensibly.

How do I decide whether to accept a given `α`? Compare the *actual* cost reduction to the reduction the quadratic model *predicted*. The model predicts, for step size `α`, a signed cost change

    ΔJ(α) = α Σ_i k_i'Q_u(i) + (α²/2) Σ_i k_i'Q_uu(i) k_i

— the first term linear in `α` from the feedforward gradient, the second quadratic in `α` from the feedforward curvature; this is just `Σ (α k'Q_u + ½α²k'Q_uu k)`, the `ΔV` tally evaluated at the scaled step. This quantity is *negative* on a good step: `k = −Q_uu^{-1}Q_u`, so `k'Q_u = −Q_u'Q_uu^{-1}Q_u < 0` dominates the linear term and `ΔJ(α) < 0` is a predicted *decrease*. So the predicted reduction proper is `−ΔJ(α) > 0`. Then form the ratio of realized to predicted reduction, `z = [J(ū) − J(û)] / (−ΔJ(α))`, and accept only if `z` exceeds a small positive constant `c_1` (an Armijo-style sufficient-decrease test: the true cost has to come down by at least a fixed fraction of what the model promised). If `z` fails, or the rollout diverged, shrink `α` and re-roll. If even the smallest `α` fails, that's a signal the backward model was untrustworthy — bump `μ` and redo the backward pass.

Putting the whole loop together: start from a nominal `(x̄, ū)`; roll forward through `f` collecting the per-step Jacobians `f_x, f_u` (and the cost derivatives `ℓ_x, ℓ_u, ℓ_xx, ℓ_ux, ℓ_uu`, and the dynamics Hessians only if doing full DDP); run the backward pass to get `k_i, K_i` and the predicted `ΔJ`, bumping `μ` and restarting if `Q̃_uu` goes indefinite; run the forward pass with line search on `α`, accepting the first `α` whose realized reduction passes the test; on acceptance, decrease `μ` and take the new trajectory as the nominal; repeat until the cost stops improving. Each accepted iteration re-linearizes around the improved trajectory, so the sequence of LQR subproblems chases the moving nonlinear optimum.

Let me write it. The structure mirrors what I derived: a forward rollout that gathers derivatives, a backward pass implementing the `Q`-expansion and the `k`/`K`/value recursion with the state-based regularization, a forward update that re-rolls the true dynamics under the corrected closed-loop control with step size `α`, and an outer loop carrying the Levenberg–Marquardt schedule and the accept test. A boolean toggles whether the dynamics Hessians are included (DDP) or dropped (iLQR / Gauss–Newton).

```python
import numpy as np


class iLQR:
    """Finite-horizon iterative LQR / DDP.

    Dynamics x_{i+1} = f(x_i, u_i); cost sum l(x_i,u_i) + l_f(x_N).
    use_hessians=False -> iLQR (Gauss-Newton, drop f_xx/f_ux/f_uu);
    use_hessians=True  -> full DDP (keep the dynamics-curvature tensors).
    """

    def __init__(self, dynamics, cost, N, use_hessians=False, max_reg=1e10):
        self.dynamics = dynamics      # f, f_x, f_u, (f_xx, f_ux, f_uu)
        self.cost = cost              # l, l_x, l_u, l_xx, l_ux, l_uu (+ terminal)
        self.N = N
        # Only keep the dynamics Hessians if the model can supply them.
        self.use_hessians = use_hessians and getattr(dynamics, "has_hessians", False)
        # Levenberg-Marquardt schedule: mu damps V_xx; grows fast on failure,
        # decays toward 0 on success (snap to 0 below mu_min).
        self.mu, self.mu_min, self.mu_max = 1.0, 1e-6, max_reg
        self.delta_0, self.delta = 2.0, 2.0

    def fit(self, x0, us, n_iterations=100, tol=1e-6):
        self.mu, self.delta = 1.0, self.delta_0
        alphas = 1.1 ** (-np.arange(10) ** 2)   # backtracking 1 -> ~0
        us = us.copy()
        na, ns = self.dynamics.action_size, self.dynamics.state_size
        k = np.zeros((self.N, na))               # last accepted gains, kept warm
        K = np.zeros((self.N, na, ns))
        changed, converged = True, False

        for _ in range(n_iterations):
            accepted = False
            if changed:                          # re-linearize around nominal
                xs, fx, fu, L, lx, lu, lxx, lux, luu, fxx, fux, fuu = \
                    self._forward_rollout(x0, us)
                J = L.sum()
                changed = False

            try:
                k, K = self._backward_pass(fx, fu, lx, lu, lxx, lux, luu,
                                           fxx, fux, fuu)
                for alpha in alphas:             # line search on feedforward
                    xs_new, us_new = self._control(xs, us, k, K, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)
                    if J_new < J:                # accept first improving step
                        converged = abs((J - J_new) / J) < tol
                        J, xs, us, changed, accepted = J_new, xs_new, us_new, True, True
                        # success -> relax regularization
                        self.delta = min(1.0, self.delta) / self.delta_0
                        self.mu *= self.delta
                        if self.mu <= self.mu_min:
                            self.mu = 0.0
                        break
            except np.linalg.LinAlgError:
                pass                             # Quu not PD -> treat as failure

            if not accepted:                     # no alpha worked -> damp more
                self.delta = max(1.0, self.delta) * self.delta_0
                self.mu = max(self.mu_min, self.mu * self.delta)
                if self.mu >= self.mu_max:
                    break
            if converged:
                break

        self._k, self._K = k, K                  # store the feedback policy
        return xs, us

    def _Q(self, fx, fu, lx, lu, lxx, lux, luu, Vx, Vxx,
           fxx=None, fux=None, fuu=None):
        # Q-function second-order expansion around the nominal (primes = next).
        Q_x = lx + fx.T @ Vx                      # l_x + f_x' V'_x
        Q_u = lu + fu.T @ Vx                      # l_u + f_u' V'_x
        Q_xx = lxx + fx.T @ Vxx @ fx              # l_xx + f_x' V'_xx f_x
        # State-based (Tassa) regularization: damp V'_xx before pulling back,
        # so the penalty is on induced state deviation and feedback survives.
        reg = self.mu * np.eye(Vxx.shape[0])
        Q_ux = lux + fu.T @ (Vxx + reg) @ fx      # l_ux + f_u'(V'_xx+muI) f_x
        Q_uu = luu + fu.T @ (Vxx + reg) @ fu      # l_uu + f_u'(V'_xx+muI) f_u
        if self.use_hessians:                     # full DDP only: dynamics curvature
            Q_xx += np.tensordot(Vx, fxx, axes=1) # + V'_x . f_xx
            Q_ux += np.tensordot(Vx, fux, axes=1) # + V'_x . f_ux
            Q_uu += np.tensordot(Vx, fuu, axes=1) # + V'_x . f_uu
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def _backward_pass(self, fx, fu, lx, lu, lxx, lux, luu,
                       fxx=None, fux=None, fuu=None):
        Vx, Vxx = lx[-1], lxx[-1]                 # terminal value = final cost
        k = np.empty((self.N, lu.shape[1]))
        K = np.empty((self.N, lu.shape[1], lx.shape[1]))
        for i in range(self.N - 1, -1, -1):
            args = (fxx[i], fux[i], fuu[i]) if self.use_hessians else ()
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(
                fx[i], fu[i], lx[i], lu[i], lxx[i], lux[i], luu[i], Vx, Vxx, *args)
            # Minimizer of the quadratic in du: du = k + K dx.
            k[i] = -np.linalg.solve(Q_uu, Q_u)    # feedforward  -Q_uu^{-1} Q_u
            K[i] = -np.linalg.solve(Q_uu, Q_ux)   # feedback     -Q_uu^{-1} Q_ux
            # Value update from the k/K form (robust under regularization).
            Vx = Q_x + K[i].T @ Q_uu @ k[i] + K[i].T @ Q_u + Q_ux.T @ k[i]
            Vxx = Q_xx + K[i].T @ Q_uu @ K[i] + K[i].T @ Q_ux + Q_ux.T @ K[i]
            Vxx = 0.5 * (Vxx + Vxx.T)             # keep it symmetric
        return k, K

    def _control(self, xs, us, k, K, alpha):
        # Forward pass: re-roll TRUE dynamics under closed-loop corrected control.
        xs_new = np.zeros_like(xs); xs_new[0] = xs[0]
        us_new = np.zeros_like(us)
        for i in range(self.N):
            # u + alpha*feedforward + feedback*(state deviation)
            us_new[i] = us[i] + alpha * k[i] + K[i] @ (xs_new[i] - xs[i])
            xs_new[i + 1] = self.dynamics.f(xs_new[i], us_new[i], i)
        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        c = sum(self.cost.l(xs[i], us[i], i) for i in range(self.N))
        return c + self.cost.l(xs[-1], None, self.N, terminal=True)

    def _forward_rollout(self, x0, us):
        # Roll nominal forward and gather per-step Jacobians/Hessians of f and l.
        ... # f, f_x, f_u (and f_xx,f_ux,f_uu if use_hessians); l and its derivs
```

The chain in one breath: LQR solves the linear problem exactly by a backward Riccati sweep because the value function stays quadratic, but a nonlinear `f` kills that closure — so I work in deviations around a nominal trajectory and quadratize the *bracket* `Q(δx,δu) = ℓ + V'∘f` to second order, which gives `Q_x = ℓ_x+f_x'V'_x`, `Q_u = ℓ_u+f_u'V'_x`, `Q_xx = ℓ_xx+f_x'V'_xx f_x(+V'_x·f_xx)`, `Q_uu = ℓ_uu+f_u'V'_xx f_u(+V'_x·f_uu)`, `Q_ux = ℓ_ux+f_u'V'_xx f_x(+V'_x·f_ux)`; minimizing over `δu` drops out a feedforward `k = −Q_uu^{-1}Q_u` and a feedback gain `K = −Q_uu^{-1}Q_ux`, and substituting back propagates `V_x, V_xx` backward — exactly the LQR Riccati pass with time-varying linearizations, with LQR as the linear special case that converges in one pass. Keeping the dynamics-curvature tensors `V'_x·f_xx` is full DDP (Newton); dropping them is iLQR (Gauss–Newton) — cheaper, keeps `Q_uu` positive definite, nearly the full step. Then a Levenberg–Marquardt `μ` on `V'_xx` damps an indefinite `Q_uu` while preserving feedback, a backtracking `α` on the feedforward with an Armijo accept-test keeps the forward rollout inside the linearization's validity, and the forward pass re-rolls the *true* dynamics under `û = ū + αk + K(x̂−x̄)` so the feedback re-aims the control at the states actually visited — iterate, re-linearizing each time, until the cost stops dropping.
