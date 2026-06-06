Let me start from where the pain actually is. I have a structural design, variables `x = (x_1, ..., x_n)` — cross-sectional areas, thicknesses, per-element densities, whatever — and I want

```
minimize    f_0(x)
subject to  f_i(x) <= 0,   i = 1, ..., m
            x_min <= x <= x_max.
```

The objective is something like weight or compliance, the constraints are stresses, displacements, eigenvalues. And the brutal fact dominating everything: to know `f_i(x)` and `∂f_i/∂x_j` at *one* point `x`, I have to run a finite element analysis plus an adjoint sensitivity solve. That one evaluation is the entire cost. Everything else — whatever arithmetic I do to decide the next point — is free by comparison. `n` is large, one variable per element; `m`, the number of behavioral constraints, is small. The responses are nonlinear and not even monotone in `x`.

So the shape of any sane method is forced on me before I design anything. I cannot do a line search that probes `f_0` at five trial points — that is five FE solves to take one step. I cannot ask for a Hessian — that is `n^2` second derivatives I will never compute. What I *can* afford, once per design point, is: the values `f_i(x^k)`, the gradients `∂f_i/∂x_j` at `x^k`, and then an unlimited amount of *cheap* arithmetic on those numbers. So the method has to be: at `x^k`, from value-and-gradient, build an *explicit* approximate problem — call it a subproblem — that is cheap to solve completely without touching the FE solver again; solve it; that gives `x^{k+1}`; analyze there; repeat. The whole game is the quality of that subproblem. If it is a faithful, safe local model I converge in a handful of FE calls. If it is a bad model I either overshoot into garbage or take timid little steps and burn analyses.

What do people do now. The most naive thing is linearize `f_i` in the direct variables `x_j` and solve the linear program. But a first-order Taylor model `f_i(x^k) + Σ_j (∂f_i/∂x_j)(x_j - x_j^k)` has no curvature at all. The LP it generates is unbounded — the minimizer wants to run to the corner of the box. So I am forced to bolt on artificial move limits `|x_j - x_j^k| <= δ_j` to keep the step sane, and now the whole behavior of the method is governed by those `δ_j`. Too big and the linear model, which is only good in a tiny neighborhood, sends me somewhere it has no business predicting; too small and I crawl, one cautious step per expensive FE call. And the right `δ_j` is different for every problem and every iteration. This is exactly the fragility I want gone. The diagnosis is: linear is too crude *because it carries no curvature*, and patching that with move limits just moves the fragility into the limits.

The incumbent in density and sizing work is the optimality-criteria update. It is clever: write the KKT stationarity for a single monotone resource constraint — material volume, `Σ_e v_e x_e <= V` — get `λ`, the multiplier, and update each variable multiplicatively, `x_e ← x_e · ((-∂f_0/∂x_e)/(λ v_e))^η`, with `λ` chosen by bisection so the volume binds and `η` a damping exponent. It is dirt cheap and it scales. But look at what it leans on: *one* constraint, and a *monotone* one, so the multiplicative form has a definite sign and the bisection on a single `λ` works. The moment I have several general behavioral constraints — three stresses and two displacements, with sensitivities of mixed sign — there is no clean multiplicative rule, no single `λ` to bisect, and the exponent `η` is hand-tuned per problem with no theory behind it. OC is a special-purpose heuristic, not a general optimizer. I want the generality of mathematical programming with the cheapness of OC.

Now the reciprocal trick, which is the real seed. People noticed long ago that linearizing in `1/x_j` instead of `x_j` works far better for structural responses. And there is a hard reason, not just empirics: for a statically determinate structure a stress or a displacement is *exactly* a linear function of the reciprocal areas — force over area. So `1/x_j` is the natural coordinate; a first-order expansion in it is exact for the determinate case and very good otherwise. Write it out: expanding `f` to first order in the variable `1/x_j`,

```
f(x) ≈ f(x^k) + Σ_j (∂f/∂x_j) · (x_j^k)^2 · (1/x_j^k − 1/x_j).
```

(The `(x_j^k)^2` is the chain-rule factor `∂x_j/∂(1/x_j)` at `x^k`, with a sign.) This *does* carry curvature — `1/x_j` is convex — so it is bounded, no move limits needed there. But stare at the curvature sign. The term `(∂f/∂x_j)(x_j^k)^2/x_j` is convex in `x_j` only when its coefficient is positive, i.e. when `∂f/∂x_j > 0`. When `∂f/∂x_j < 0` that same reciprocal term is *concave*, and a concave piece in a minimization subproblem is poison: it pulls the minimizer to a boundary in the wrong way and destroys convexity. So pure reciprocal is convex only half the time, depending on the sign of each sensitivity.

The fix is staring at me from the two cases. If the reciprocal term is convex exactly when `∂f/∂x_j > 0`... wait, let me recompute, because for a *minimization* surrogate I want convexity and I want the reciprocal where it helps. The reciprocal `1/x_j` is itself convex and *decreasing* in `x_j`. Its coefficient in the expansion above is `(∂f/∂x_j)(x_j^k)^2`, sign of `∂f/∂x_j`. A convex term needs a positive coefficient on `1/x_j`, i.e. `∂f/∂x_j > 0`. Hmm, but physically the reciprocal helps for *resource-like* responses where increasing `x` decreases the response, i.e. `∂f/∂x_j < 0`. Let me not argue from physics and just argue from convexity, term by term. I have a derivative `∂f_i/∂x_j` to match at `x^k`, and I get to choose, *per variable, per function*, whether to spend it on a direct (linear) term or a reciprocal term — and I will choose whichever keeps the surrogate convex. A linear term `c·x_j` is convex for either sign of `c`. A reciprocal term `c/x_j` is convex only for `c > 0`. So: if `∂f_i/∂x_j > 0` I can use either; if `∂f_i/∂x_j < 0` the reciprocal would be concave, so I must use the direct (linear) term there, or I need a *different* reciprocal — one that is decreasing the other way.

This is the convex-linearization idea: pick, for each term, the direct variable when `∂f_i/∂x_j > 0` and the reciprocal when `∂f_i/∂x_j < 0`,

```
f_i(x) ≈ f_i(x^k) + Σ_{∂f_i/∂x_j > 0} (∂f_i/∂x_j)(x_j − x_j^k)
                  + Σ_{∂f_i/∂x_j < 0} (∂f_i/∂x_j)(x_j^k)^2 (1/x_j − 1/x_j^k).
```

Now every retained term is convex (a line, or a positively-weighted `1/x_j`), so the whole surrogate is convex *and separable* — no cross terms, each `x_j` appears alone. And there is a bonus I should not waste: among all the ways to mix direct and reciprocal variables matching the same gradient, this sign-based choice is the *most conservative* — it bounds the true function from above near `x^k` most tightly (Starnes and Haftka noticed this on buckling constraints). Conservative means: the subproblem minimizer, when I evaluate the true `f_i` there, will not be worse than the model predicted — so I can take the full subproblem step without a line search. That is precisely what I need given that a line search costs FE solves.

So convex linearization gives me convex, separable, conservative subproblems, cheap to solve. Why is that not the end? Because of the reciprocal's *fixed singularity*. The reciprocal term `1/x_j` blows up at `x_j = 0` — its asymptote is pinned at the origin. Once I fix the iterate `x^k`, the curvature of every term is *determined*: the second derivative of `(x_j^k)^2/x_j` is `2(x_j^k)^2/x_j^3`, a fixed number at `x^k`. I have no dial. If that curvature is too gentle the model is nearly linear and I overshoot; if too sharp the model is over-conservative and I crawl. Worse, the asymptote at 0 is far from `x^k` when `x_j^k` is large, so the model is barely curved there exactly when I might want it tighter. And if a sensitivity `∂f_i/∂x_j` flips sign from one iteration to the next, the term jumps from direct to reciprocal — a discontinuous change in the model — and the design can oscillate. The only cure within convex linearization is, again, external move limits. I am back to the fragility.

So here is the thing I keep circling. The reciprocal `1/x_j = 1/(x_j − 0)` has its asymptote nailed at 0. What if I *unnail* it? Replace `1/(x_j − 0)` with `1/(x_j − L_j)`, where `L_j` is a *lower asymptote* I get to place below `x_j^k`. As `L_j` moves up toward `x_j^k`, the term `1/(x_j − L_j)` gets sharply curved near the iterate — strongly convex, very conservative, small steps. As `L_j` recedes toward `−∞`, the term flattens out and approaches a linear term — barely curved, aggressive, big steps. So the *position of the asymptote is a continuous conservativeness knob*. That is exactly the dial convex linearization was missing. And it is per-variable: I can make some coordinates cautious and others bold.

Let me make this symmetric, because I have two signs of derivative to handle. For the direct/increasing case I want a term that is convex, increasing, and whose curvature I can also tune. The mirror image of `1/(x_j − L_j)` is `1/(U_j − x_j)` with an *upper asymptote* `U_j > x_j^k`: it is convex, increasing in `x_j`, blows up as `x_j → U_j`, and flattens to linear as `U_j → +∞`. So give each variable a pair of moving asymptotes `L_j < x_j^k < U_j` and build the approximation out of two terms:

```
f_i(x) ≈ r_i + Σ_j [ p_ij/(U_j − x_j) + q_ij/(x_j − L_j) ],
```

with `p_ij, q_ij >= 0` so that each term is convex (and the whole thing separable). Both basis functions live between the asymptotes; the iterate sits strictly inside `(L_j, U_j)`.

Now I have to set `p_ij`, `q_ij`, `r_i` so this matches the real function to first order at `x^k`, and I want it to stay convex, which means keeping `p_ij, q_ij >= 0`. Convexity I can guarantee by construction with one clean choice: at each `(i, j)` put the entire first-order weight on *one* of the two terms — the one whose slope has the right sign — and set the other coefficient to zero. The derivative of the surrogate term at `x_j` is

```
d/dx_j [ p_ij/(U_j − x_j) + q_ij/(x_j − L_j) ] = p_ij/(U_j − x_j)^2 − q_ij/(x_j − L_j)^2.
```

The `p`-term contributes a *positive* slope, the `q`-term a *negative* slope. So if `∂f_i/∂x_j > 0`, I match it with the `p`-term alone (`q_ij = 0`); if `∂f_i/∂x_j < 0`, I match it with the `q`-term alone (`p_ij = 0`). Matching at `x = x^k`:

```
∂f_i/∂x_j > 0:   p_ij/(U_j − x_j^k)^2 = ∂f_i/∂x_j   ⇒   p_ij = (U_j − x_j^k)^2 · ∂f_i/∂x_j,   q_ij = 0,
∂f_i/∂x_j < 0:   −q_ij/(x_j^k − L_j)^2 = ∂f_i/∂x_j  ⇒   q_ij = −(x_j^k − L_j)^2 · ∂f_i/∂x_j,  p_ij = 0,
```

so both `p_ij` and `q_ij` come out `>= 0` automatically — convexity for free — and the `(U_j − x_j^k)^2` / `(x_j^k − L_j)^2` scaling is not arbitrary, it is exactly the factor that cancels the `(U_j − x)^{-2}` / `(x − L)^{-2}` from differentiating the basis function so that the surrogate's slope equals the true slope at `x^k`. Then `r_i` is fixed by matching the value:

```
r_i = f_i(x^k) − Σ_j [ p_ij/(U_j − x_j^k) + q_ij/(x_j^k − L_j) ].
```

This is the construction. Let me double-check it really is the missing dial. The surrogate's second derivative in `x_j` is

```
d^2/dx_j^2 = 2 p_ij/(U_j − x_j)^3 + 2 q_ij/(x_j − L_j)^3,
```

nonnegative for `L_j < x_j < U_j` and `p_ij, q_ij >= 0` — convex, confirmed, and separable. At the matching point, say for the `q`-term case, the curvature is `2 q_ij/(x_j^k − L_j)^3 = −2(∂f_i/∂x_j)/(x_j^k − L_j)`. As `L_j → x_j^k` this diverges — arbitrarily conservative, tiny steps; as `L_j → −∞` it goes to zero — the term tends to the linear Taylor term, aggressive. So yes: the gradient match is *independent* of where I put the asymptotes (the scaling absorbs it), but the *curvature*, hence the conservativeness and the implied step, is set entirely by how close the asymptotes sit to `x^k`. I have decoupled "match the gradient" from "choose the step," and the second is now a smooth geometric choice instead of a move limit.

And convex linearization falls out as a special case to sanity-check against: send `U_j → +∞` and put `L_j = 0`. Then the `p`-term `p_ij/(U_j − x_j)` with `p_ij = (U_j − x_j^k)^2 ∂f_i/∂x_j` tends to the linear term `(∂f_i/∂x_j)(x_j − x_j^k)` (expand `1/(U − x) ≈ 1/U + x/U^2`, multiply by `U^2 ∂f`), and the `q`-term `q_ij/(x_j − 0)` is exactly the reciprocal term. So fixed asymptotes at `{0, ∞}` reproduce direct/reciprocal mixing; moving them is the generalization. Good — I have not lost anything, I have added a control.

Now, who sets `L_j` and `U_j`, and how. I do not get to see the true function between FE calls, so I must drive the asymptotes from the *history* of iterates — the cheap information I already have. The intuition: if a variable is marching steadily in one direction over the last few iterations, the model is being too cautious there and I should relax — push the relevant asymptote *away* from `x_j` to flatten the term and let it take a bigger step. If a variable is *zig-zagging* — it went up, now it wants to come back down — the model overshot, it is not conservative enough, and I should tighten — pull the asymptote *in* toward `x_j` to add curvature and damp the oscillation. That is a trust-region adaptation, but done by moving a singularity instead of by a line search or an explicit radius.

I can detect the two cases from the sign of the product of the last two steps. Let `s_j = (x_j^k − x_j^{k−1})(x_j^{k−1} − x_j^{k−2})`. If `s_j > 0` the last two moves had the same sign — steady progress, relax. If `s_j < 0` the moves reversed — oscillation, tighten. So pick a factor `γ_j`: `γ_j = 1.2` when `s_j > 0` (push asymptotes out), `γ_j = 0.7` when `s_j < 0` (pull them in), `γ_j = 1` when `s_j = 0`. The asymmetry `1.2` up / `0.7` down is deliberate — relax gently when things go well, but contract more decisively when I catch an oscillation, so damping wins over acceleration when they conflict. Update by scaling the *previous distance* from the asymptote:

```
L_j = x_j^k − γ_j (x_j^{k−1} − L_j^{k−1}),
U_j = x_j^k + γ_j (U_j^{k−1} − x_j^{k−1}).
```

For the first two iterations there is no history, so just place them at a default fraction of the variable range, `L_j = x_j − 0.5(x_max − x_min)`, `U_j = x_j + 0.5(x_max − x_min)` — moderate conservativeness to start. And I should clamp the asymptote distance into a band so a runaway product never drives `L_j` on top of `x_j` (curvature → ∞, dead) or off to infinity (curvature → 0, linear, unbounded): keep `(x_j − L_j)` and `(U_j − x_j)` between `0.01` and `10` times the range. The numbers `0.5`, `0.7`, `1.2`, `0.01`, `10` are just defaults of this oscillation heuristic; a user who wants a more conservative method shrinks the initial fraction and the growth factor.

One more guard on the subproblem itself: I must not let the optimizer push `x_j` all the way onto an asymptote, where the term is singular. So I optimize over a shrunk box `α_j <= x_j <= β_j` that stays a safe fraction off the asymptotes and also respects a move bound. Concretely `α_j = max{ x_min, L_j + 0.1(x_j^k − L_j), x_j^k − 0.5(x_max − x_min) }` and symmetrically for `β_j`: the `0.1` (call it `albefa`) keeps me at least ten percent of the way off the asymptote so `1/(x − L)` stays finite, and the `0.5` (`move`) caps the step at half the range. These three together just say: stay in the real box, stay off the asymptotes, don't move too far at once.

So the subproblem at `x^k` is

```
minimize    r_0 + Σ_j [ p_0j/(U_j − x_j) + q_0j/(x_j − L_j) ]
subject to  r_i + Σ_j [ p_ij/(U_j − x_j) + q_ij/(x_j − L_j) ] <= 0,   i = 1, ..., m
            α_j <= x_j <= β_j.
```

Convex, separable, with `p, q >= 0`, and the asymptotes/box absorbing all the conservativeness and step control. Now: how do I actually *solve* it, cheaply, every iteration?

Here is where separability pays. I have `n` primal variables but only `m` constraints, and `m` is small. The natural move is to go to the *dual*. Form the Lagrangian with multipliers `λ_i >= 0` on the constraints. Because the objective and every constraint are sums of the same per-variable basis functions, the Lagrangian's `x`-part is again separable: define `p_j(λ) = p_0j + Σ_i λ_i p_ij`, `q_j(λ) = q_0j + Σ_i λ_i q_ij`, and the `x`-dependent part of the Lagrangian is

```
ψ(x, λ) = Σ_j [ p_j(λ)/(U_j − x_j) + q_j(λ)/(x_j − L_j) ].
```

For fixed `λ`, minimizing this over `x` splits into `n` independent one-dimensional problems. Set the derivative in `x_j` to zero:

```
p_j(λ)/(U_j − x_j)^2 − q_j(λ)/(x_j − L_j)^2 = 0
⇒  √p_j(λ) · (x_j − L_j) = √q_j(λ) · (U_j − x_j)
⇒  x_j(λ) = ( L_j √p_j(λ) + U_j √q_j(λ) ) / ( √p_j(λ) + √q_j(λ) ),
```

a *closed-form* per-variable minimizer, then clamped to `[α_j, β_j]` if it falls outside (the constraint there being active). No iteration, `O(n)` arithmetic, zero FE calls. This is the whole reason the convex-separable form was worth chasing: the inner primal solve is analytic.

The dual function is `W(λ) = ψ(x(λ), λ) − Σ_i λ_i b_i` (writing `b_i = −r_i`), the minimized Lagrangian as a function of the multipliers, which I maximize over `λ >= 0`. It is concave (a pointwise minimum over `x` of functions affine in `λ`), and it lives in `m` dimensions — small. Its gradient is the cleanest thing in the whole method: by the envelope theorem the explicit `λ`-dependence dominates, so

```
∂W/∂λ_i = g_i(x(λ)) − b_i = (value of constraint i's approximation at x(λ)),
```

the dual gradient is literally the primal constraint residuals. And the dual Hessian is available in closed form too, from `dx_j/dλ_i` differentiated through `x_j(λ)`, with the sum restricted to the *free* variables (the ones not clamped at `α_j` or `β_j`) — the clamped ones contribute zero, which is what makes the dual Hessian piecewise and the active set matter. So I maximize the `m`-dimensional concave `W(λ)` by a Newton/SQP step in the space of currently-active multipliers, recover `x* = x(λ*)`, and that is `x^{k+1}`. The expensive object — `n` large — never enters the solve as an `n`-dimensional system; it collapses to `m`.

Let me also make the formulation robust to the case where the user's problem is infeasible or badly scaled, because in structural work a constraint can be violated badly at the start and a hard subproblem can then be infeasible — and an infeasible subproblem stalls the whole outer loop. I will embed the standard NLP into a slightly larger always-feasible problem with artificial variables `y = (y_1, ..., y_m) >= 0` and a scalar `z >= 0`:

```
minimize    f_0(x) + a_0 z + Σ_i ( c_i y_i + ½ d_i y_i^2 )
subject to  f_i(x) − a_i z − y_i <= 0,   i = 1, ..., m
            x ∈ X, y >= 0, z >= 0.
```

Each `y_i` can relax constraint `i`, but at a cost `c_i y_i` in the objective. Choose `a_0 = 1`, `a_i = 0`, `d_i = 1`, and `c_i` "large": then `z = 0` and `y_i = 0` at the optimum whenever the original problem is feasible, so I recover exactly `min f_0 s.t. f_i <= 0`; but if it is infeasible, some `y_i > 0` lets the method keep moving instead of dying. (The same template, with other `a_i, c_i, d_i`, also expresses min–max and least-squares problems for free — worth keeping general.) The `c_i` should be "reasonably large," not `10^10`, for numerical sanity; if it turns out some `y_i > 0` at the solution, bump `c_i` and re-solve. The subproblem is built the same way — the `f_i` get replaced by their MMA approximations, the `y, z` terms carry through unchanged.

Now I want to actually nail the subproblem solve in code, and rather than write a bespoke dual maximizer with its active-set bookkeeping for the clamped variables (which is where the dual Hessian goes discontinuous and the SQP needs care), I will solve the subproblem's KKT system directly with a *primal–dual interior-point* method. It is the same optimality conditions, just handled with a barrier so I never have to branch on which variables are clamped — the interior-point iterates stay strictly inside the box and the active set emerges in the limit. Let me write the KKT conditions for the subproblem. With multipliers `λ` on the constraints, `ξ, η` on the lower/upper box bounds `α_j <= x_j <= β_j`, `μ` on `y >= 0`, `ζ` on `z >= 0`:

```
∂ψ/∂x_j − ξ_j + η_j = 0,                       j = 1, ..., n
c_i + d_i y_i − λ_i − μ_i = 0,                   i = 1, ..., m
a_0 − ζ − Σ_i a_i λ_i = 0
g_i(x) − a_i z − y_i − b_i <= 0,    λ_i ( · ) = 0
ξ_j (x_j − α_j) = 0,  η_j (β_j − x_j) = 0,  μ_i y_i = 0,  ζ z = 0
all primal feasibilities and λ, ξ, η, μ, ζ >= 0,
```

where `∂ψ/∂x_j = p_j(λ)/(U_j − x_j)^2 − q_j(λ)/(x_j − L_j)^2`. The subproblem is a regular convex program, so these KKT conditions are necessary and sufficient — solving them *is* solving the subproblem.

The interior-point relaxation: introduce slacks `s_i` for the constraints and replace each complementarity zero by a small `ε > 0`,

```
ξ_j (x_j − α_j) = ε,  η_j (β_j − x_j) = ε,  μ_i y_i = ε,  ζ z = ε,  λ_i s_i = ε,
```

with all the `(x_j − α_j), (β_j − x_j), y_i, z, s_i, ξ_j, η_j, μ_i, ζ, λ_i` kept strictly positive. For each fixed `ε` this perturbed system has a unique solution — it is exactly the KKT system of the strictly convex *log-barrier* problem

```
minimize g_0(x) + a_0 z + Σ_i(c_i y_i + ½ d_i y_i^2)
         − ε Σ_j log(x_j − α_j) − ε Σ_j log(β_j − x_j)
         − ε Σ_i log y_i − ε Σ_i log s_i − ε log z
subject to g_i(x) − a_i z − y_i + s_i = b_i,
```

whose strict-positivity constraints are enforced automatically by the logarithms. I solve the relaxed system by Newton's method, then shrink `ε` (multiply by `0.1`) and repeat, tracing the central path down to the true KKT point as `ε → 0`.

The Newton step is the load-bearing piece, so let me actually take it. Linearizing the relaxed conditions at the current point gives a big system in `(∆x, ∆y, ∆z, ∆λ, ∆ξ, ∆η, ∆µ, ∆ζ, ∆s)`. The diagonal blocks let me eliminate `∆ξ, ∆η, ∆µ, ∆ζ, ∆s` immediately. From `ξ_j(x_j − α_j) = ε`, differentiating, `(x_j − α_j)∆ξ_j + ξ_j ∆x_j = ε − ξ_j(x_j − α_j)`, so

```
∆ξ = −(x − α)^{−1} ⟨ξ⟩ ∆x − ξ + ε (x − α)^{−1},
```

and symmetrically

```
∆η =  (β − x)^{−1} ⟨η⟩ ∆x − η + ε (β − x)^{−1},
∆µ = −y^{−1} ⟨µ⟩ ∆y − µ + ε y^{−1},
∆ζ = −(ζ/z) ∆z − ζ + ε/z,
∆s = −λ^{−1} ⟨s⟩ ∆λ − s + ε λ^{−1}.
```

Substituting these back collapses the system to one in `(∆x, ∆y, ∆z, ∆λ)` only. The curvature of the objective enters through the diagonal matrix `Ψ` with entries

```
Ψ_jj = ∂^2ψ/∂x_j^2 = 2 p_j(λ)/(U_j − x_j)^3 + 2 q_j(λ)/(x_j − L_j)^3,
```

the second derivatives of the moving-asymptote terms (positive — the convexity I built in), and the constraint gradients through the `m × n` matrix `G` with

```
G_ij = ∂g_i/∂x_j = p_ij/(U_j − x_j)^2 − q_ij/(x_j − L_j)^2.
```

After eliminating `∆ξ, ∆η`, the `x`-block becomes a diagonal `D_x = Ψ + ⟨ξ/(x−α)⟩ + ⟨η/(β−x)⟩`; after eliminating `∆µ`, the `y`-block is `D_y = d + ⟨µ/y⟩`; the `λ`-block gets `D_λ = ⟨s/λ⟩ + ⟨1/D_y⟩` from folding `∆y, ∆s` together. That leaves

```
[ D_x        G^T  ] [∆x]   [−δ̃_x]
[       ζ/z  −a^T ] [∆z] = [−δ̃_z]
[ G    −a   −D_λ  ] [∆λ]   [−δ̃_λ]
```

(with the `∆y` row already merged). Now the key efficiency choice: `D_x` is `n × n` diagonal, `D_λ` is `m × m`. If `m < n` I eliminate `∆x = −D_x^{−1}(δ̃_x + G^T ∆λ)` and reduce to an `(m+1) × (m+1)` system in `(∆λ, ∆z)` — cheap, because `m` is small. If instead `m >= n` I eliminate `∆λ` and reduce to an `(n+1) × (n+1)` system in `(∆x, ∆z)`. Either way I factor a small dense matrix, never the full thing. This is the same `m`-vs-`n` collapse the dual gave me, realized inside the Newton solve. Then back-substitute for the eliminated increments, take a fraction-to-the-boundary step (scale so no positive variable crosses zero, the `1.01` safety factor), backtrack until the residual norm decreases, and loop; shrink `ε`; repeat until `ε` is tiny. Recover `x^{k+1} = x`.

Let me assemble the whole thing as one outer iteration. Analyze at `x^k` (the one FE call). Update the asymptotes `L_j, U_j` from the last three iterates by the oscillation rule. Build `p_ij, q_ij, r_i` (with `b_i = −r_i`) by the sign-split gradient match, and the shrunk box `α_j, β_j`. Solve the convex separable subproblem by the interior-point method just derived to get `x^{k+1}` and the multipliers. Test the KKT residual of the *original* problem for stopping. Shift history and repeat. Each outer step costs exactly one analysis; all the asymptote arithmetic and the subproblem solve are free.

Here is the core, grounded in a standard implementation. First the subproblem builder — asymptote update, sign-split coefficients, box:

```python
import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve

def mmasub(m, n, k, xval, xmin, xmax, xold1, xold2,
           f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d,
           move=0.5, asyinit=0.5, asydecr=0.7, asyincr=1.2,
           asymin=0.01, asymax=10, raa0=1e-5, albefa=0.1):
    een = np.ones((n, 1)); eem = np.ones((m, 1))
    xmami = np.maximum(xmax - xmin, 1e-5 * een)

    # Moving asymptotes L=low, U=upp. First two iters: a default fraction
    # of the range. Later: scale the previous distance by the oscillation
    # factor (1.2 if the last two steps agreed in sign -> relax; 0.7 if they
    # reversed -> tighten), then clamp the distance to [asymin, asymax]*range.
    if k <= 2:
        low = xval - asyinit * xmami
        upp = xval + asyinit * xmami
    else:
        s = (xval - xold1) * (xold1 - xold2)
        factor = een.copy()
        factor[s > 0] = asyincr
        factor[s < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        low = np.maximum(low, xval - asymax * xmami)
        low = np.minimum(low, xval - asymin * xmami)
        upp = np.minimum(upp, xval + asymax * xmami)
        upp = np.maximum(upp, xval + asymin * xmami)

    # Trust box [alfa, beta]: stay off the asymptotes (albefa) and bound the
    # step (move), within the real box.
    alfa = np.maximum(np.maximum(low + albefa * (xval - low),
                                 xval - move * xmami), xmin)
    beta = np.minimum(np.minimum(upp - albefa * (upp - xval),
                                 xval + move * xmami), xmax)

    # Sign-split coefficients: put the gradient on the U-term where it is
    # positive (p), on the L-term where negative (q), scaled by (U-x)^2 /
    # (x-L)^2 so the surrogate's slope matches df at xval. p,q>=0 => convex.
    ux1 = upp - xval; xl1 = xval - low
    ux2, xl2 = ux1 * ux1, xl1 * xl1
    p0 = np.maximum(df0dx, 0); q0 = np.maximum(-df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 / xmami       # tiny strict-convexity floor
    p0 = (p0 + pq0) * ux2; q0 = (q0 + pq0) * xl2
    P = np.maximum(dfdx, 0); Q = np.maximum(-dfdx, 0)
    PQ = 0.001 * (P + Q) + raa0 * (eem @ (1 / xmami).T)
    P = (diags(ux2.flatten()).dot((P + PQ).T)).T
    Q = (diags(xl2.flatten()).dot((Q + PQ).T)).T
    b = P @ (1 / ux1) + Q @ (1 / xl1) - fval     # b_i = -r_i

    # Solve the convex separable subproblem by interior point.
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, 1e-7, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp
```

Then the interior-point subproblem solver — the central-path loop, the Newton system with the `m<n` / `m>=n` elimination, fraction-to-boundary step, residual backtracking, `ε`-shrink:

```python
def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    een = np.ones((n, 1)); eem = np.ones((m, 1))
    epsi = 1.0
    # Strictly-interior start.
    x = 0.5 * (alfa + beta); y = eem.copy(); z = np.array([[1.0]])
    lam = eem.copy(); s = eem.copy(); zet = np.array([[1.0]])
    xsi = np.maximum(een / (x - alfa), een); eta = np.maximum(een / (beta - x), een)
    mu = np.maximum(eem, 0.5 * c)

    while epsi > epsimin:                       # trace the central path down
        ux1 = upp - x; xl1 = x - low
        plam = p0 + P.T @ lam; qlam = q0 + Q.T @ lam
        dpsidx = plam / ux1**2 - qlam / xl1**2  # d psi / dx
        # residuals of the eps-relaxed KKT system
        residu = stack_residuals(...)           # (omitted: as in the conditions above)
        while residumax > 0.9 * epsi and inner < 200:
            ux1 = upp - x; xl1 = x - low
            plam = p0 + P.T @ lam; qlam = q0 + Q.T @ lam
            gvec = P @ (1/ux1) + Q @ (1/xl1)
            GG = (diags((1/ux1**2).flatten()).dot(P.T)).T \
               - (diags((1/xl1**2).flatten()).dot(Q.T)).T          # G_ij
            dpsidx = plam / ux1**2 - qlam / xl1**2
            # eliminated-out RHS pieces (delx, dely, delz, dellam) ...
            diagx = 2 * (plam / ux1**3 + qlam / xl1**3) \
                  + xsi / (x - alfa) + eta / (beta - x)            # D_x = Psi + ...
            diagy = d + mu / y
            diaglamyi = s / lam + eem / diagy                       # D_lambda
            if m < n:                            # eliminate dx -> (m+1) system
                blam = dellam + dely/diagy - GG @ (delx/diagx)
                Alam = diags(diaglamyi.flatten()) \
                     + (diags((1/diagx).flatten()).dot(GG.T).T) @ GG.T
                AA = block([[Alam, a], [a.T, -zet/z]])
                sol = solve(AA, np.vstack([blam, delz]))
                dlam = sol[:m]; dz = sol[m:m+1]
                dx = -delx/diagx - (GG.T @ dlam)/diagx
            else:                                # eliminate dlam -> (n+1) system
                Axx = diags(diagx.flatten()) \
                    + (diags((1/diaglamyi).flatten()).dot(GG).T) @ GG
                # ... symmetric assembly, solve for dx, dz, then dlam
            # recover dy, dxsi, deta, dmu, dzet, ds from the elimination formulas
            # fraction-to-boundary step (1.01) + backtrack until residual drops
            steg = step_to_boundary(...)
            backtrack_until_residual_decreases(...)
        epsi *= 0.1                              # shrink the barrier
    return x, y, z, lam, xsi, eta, mu, zet, s
```

And the outer driver — one FE call per iteration, asymptote/subproblem in between:

```python
xval = x0.copy(); xold1 = x0.copy(); xold2 = x0.copy()
low = xmin.copy(); upp = xmax.copy()
a0, a, c, d = 1.0, np.zeros((m,1)), 1000*np.ones((m,1)), np.ones((m,1))
for k in range(1, maxit + 1):
    f0val, df0dx, fval, dfdx = fe_analysis(xval)          # the one expensive call
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
        m, n, k, xval, xmin, xmax, xold1, xold2,
        f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d)
    _, kktnorm, _ = kktcheck(m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s,
                             xmin, xmax, df0dx, fval, dfdx, a0, a, c, d)
    if kktnorm < tol:
        break
    xold2, xold1, xval = xold1, xval, xmma.copy()
```

The causal chain, start to finish: FE responses are the only expensive thing, so I must replace the true problem by a cheap explicit subproblem from value-and-gradient — a linear model carries no curvature and needs fragile move limits, so reciprocal variables add curvature, but their `1/x_j` is convex only for one sign of the derivative and its singularity is nailed at the origin, giving no control; freeing that singularity into a pair of moving asymptotes `L_j < x_j < U_j` and matching the gradient with a sign-split pair of terms `p_ij/(U_j−x_j) + q_ij/(x_j−L_j)` makes the surrogate convex and separable by construction, with the asymptote *positions* acting as a continuous, per-variable conservativeness dial that replaces move limits; the dial is driven from the iterate history by an oscillation heuristic (relax on steady progress, tighten on zig-zag); the resulting convex separable subproblem is solved through its low-dimensional dual — analytically minimizing the Lagrangian variable-by-variable and maximizing the `m`-dimensional concave dual whose gradient is just the constraint residuals — realized robustly as a primal–dual interior-point Newton method that collapses to a small `min(m, n)`-sized linear solve per step; wrapping the constraints with cheap artificial relaxation variables keeps every subproblem feasible. One analysis per outer iteration, everything else free.
