OK, let me think this through from scratch. I want to solve `min cᵀx` subject to `Ax = b, x ≥ 0`, and I want two things that, in 1983, nobody gets at once: a worst-case bound that is polynomial in `n` and in the bit-size `L` of the data, and an algorithm that actually runs fast on the big sparse problems people care about. Right now I have to pick a pole. Dantzig's simplex method is the practical champion — maintain a vertex, pivot along an edge to a better adjacent vertex, repeat; each pivot is cheap, sparsity is preserved, and empirically it finishes in something like a linear number of pivots. But Klee and Minty built a squashed cube where Dantzig's rule marches through all `2ⁿ` vertices, so the worst case is exponential. The other pole is Khachiyan's ellipsoid method: genuinely polynomial, `O(n⁶L²)`, the first proof that LP is in P. But it's useless in practice — gigantic constants, it always behaves like its worst case, it ignores sparsity, and the round-off errors *accumulate*, so the precision you need grows with the iteration count. I have a fast-but-exponential method and a polynomial-but-unusable one. Neither is what I want.

Let me stare at *why* simplex is fragile. Its whole life is on the boundary — it lives at vertices and crawls along edges. The exponential blowup is a property of the polytope's combinatorics: there are exponentially many vertices and the path can be forced to visit a huge fraction of them. So the boundary is where the trouble lives. What if I never touch a vertex — keep a point `x > 0` strictly inside the feasible region and cut across the interior toward the optimum? Then there is no combinatorial vertex graph to get lost in. The interior is a smooth convex set; on it `cᵀx` is just a linear function and I can bring gradients, Hessians, Newton's method to bear, none of which the vertex picture lets me use. That at least seems worth trying: go through the interior rather than around the boundary.

But there's an immediate obstacle. The optimum I'm chasing is *on* the boundary — it's a vertex, or at least a face. If I want to stay strictly interior I can never actually arrive; I can only approach. And as I push toward the boundary the constraint `x ≥ 0` is right there, threatening to be violated the moment a coordinate hits zero. I need a way to feel the boundary repelling me before I crash into it, smoothly, so I can use calculus.

The natural device is a penalty that is mild deep inside and explodes at the wall. For the constraint `xⱼ ≥ 0` I want a function of `xⱼ` that is finite for `xⱼ > 0` and `→ +∞` as `xⱼ → 0⁺`. The cleanest such function is `−ln xⱼ`. Sum over coordinates:

```
φ(x) = − Σⱼ ln xⱼ.
```

This is convex, smooth on the open orthant, and a wall of infinite height sits on every face `xⱼ = 0`. So instead of minimizing `cᵀx` over a region with a hard boundary, minimize `cᵀx` plus this barrier — but I have to weight them. If the barrier dominates, I sit near the analytic center and ignore the objective; if the objective dominates, I run into the wall. Put a knob `t > 0` on the objective:

```
F_t(x) = t·cᵀx − Σⱼ ln xⱼ,    minimized over  {x : Ax = b, x > 0}.
```

For small `t` the barrier wins and the minimizer is some bland central point; for large `t` the objective wins and the minimizer is pushed toward the true optimum, but always held strictly inside by the barrier. This is exactly Frisch's 1955 idea, and Fiacco and McCormick's SUMT in the 1960s: solve a sequence of these for `t → ∞`. So why didn't this already settle the question? Because the classical analysis treated each `F_t` as a generic smooth function to be minimized, and concluded that as `t → ∞` the subproblems become *ill-conditioned* — near the boundary the Hessian of the barrier blows up, the minimization slows, round-off grows — and the whole approach lost favor in the late 1960s with no polynomial bound ever attached. So the barrier is suggestive but it carries a reputation for being numerically treacherous at the very limit I need. Rather than trust or distrust that reputation, let me derive what the minimizers actually look like and decide for myself.

Fix `t`. Minimize `F_t(x) = t cᵀx − Σ ln xⱼ` subject to `Ax = b`. Lagrangian with multiplier `y` for the equalities:

```
∇F_t(x) − Aᵀy = 0  ⇒  t·c − (1/x₁, …, 1/xₙ)ᵀ = Aᵀy,
```

writing `1/x` coordinatewise. Let me rescale the multiplier carefully. Set `s := (1/x)/t`, so `sⱼ = 1/(t xⱼ)`, equivalently `t xⱼ sⱼ = 1`. The stationarity condition `t c − (1/x) = Aᵀỹ`, divided by `t`, becomes `c − (1/(t x)) = Aᵀ(ỹ/t)`, i.e. with `y := ỹ/t`,

```
Aᵀy + s = c,    where  sⱼ = 1/(t xⱼ),  i.e.  xⱼ sⱼ = 1/t  for every j.
```

Now look at what I've got. The minimizer `x*(t)` of the barrier problem, together with `y` and `s`, satisfies:

```
Ax = b,    x > 0              (primal feasibility, strict)
Aᵀy + s = c,    s > 0         (dual feasibility, strict)
xⱼ sⱼ = 1/t   for all j       (perturbed complementarity)
```

Stare at that last line. The actual optimality (KKT) conditions for the LP are the same three lines *except* the last would be `xⱼ sⱼ = 0` — complementary slackness, each pair has one of them zero. The barrier hasn't given me a different problem; it has given me a smooth deformation of the optimality conditions, with the hard combinatorial complementarity `xⱼsⱼ = 0` softened to `xⱼsⱼ = 1/t`. As `t → ∞`, `1/t → 0` and I recover true optimality. The minimizers `x*(t)` trace a smooth curve through the interior — call it the central path — and it ends at the optimum.

And here's a payoff I didn't go looking for. `s` is *automatically dual feasible*: `Aᵀy + s = c` with `s > 0` is exactly feasibility for the LP dual `max bᵀy s.t. Aᵀy + s = c, s ≥ 0`. So every point on the central path hands me, for free, a feasible dual point and therefore a *lower bound* on the optimum. What's the gap? For any primal-dual feasible pair, `cᵀx = (Aᵀy+s)ᵀx = yᵀ(Ax) + sᵀx = bᵀy + sᵀx`, so the duality gap is `cᵀx − bᵀy = xᵀs`. Now on the central path `xⱼsⱼ = 1/t` for each of the `n` coordinates, so

```
gap = xᵀs = Σⱼ xⱼ sⱼ = n/t.
```

That is clean and it's the whole game: if I can produce a central point with parameter `t`, I *know* I am within `n/t` of optimal, measured by a genuine dual certificate, not a heuristic. To get `ε`-optimal I need `t ≈ n/ε`. So the barrier is not just a penalty trick — its stationary points carry their own optimality certificate, and the parameter `t` is exactly the inverse gap. That reframes whether the late-1960s pessimism was about the right thing; I'll come back to it.

So the algorithm wants to be: follow the central path, increasing `t`, until `n/t < ε`, then round to a vertex. The two questions are *how to move along it* and *how aggressively to raise `t`*. For the move: at a given `t`, the central point is the unconstrained (on the affine set `Ax=b`) minimizer of a smooth convex function `F_t`. The tool for minimizing a smooth convex function fast is Newton's method. Write the central-path conditions as a system and Newton it. With `Ax=b` always maintained and the barrier Hessian `∇²φ = diag(1/x²) =: X⁻²` (here `X = diag(x)`), one Newton step for `min t cᵀx − Σ ln xⱼ s.t. Ax = b` solves

```
[ X⁻²   Aᵀ ] [ Δx ]   [ −t c + X⁻¹ 1 ]
[  A    0  ] [ Δy ] = [      0       ].
```

Eliminate `Δx` from the top block: `Δx = X²(−tc + X⁻¹1 − AᵀΔy)`, substitute into `AΔx = 0` to get the normal equations

```
A X² Aᵀ Δy = −t A X² c + A X 1 = −t A X² c + b
```

(using `Ax = b`). The matrix `A X² Aᵀ` is positive definite when `A` has full row rank, and if `A` is sparse so is `A X² Aᵀ`, so a Cholesky factorization handles it. So each step costs essentially one solve with a matrix of the form `A D² Aᵀ`, `D` diagonal positive — and `D` changes only a little between steps. That's promising for practice, and it's the same kind of kernel Dikin's affine scaling needed.

But now I hit exactly the wall the SUMT people hit. Newton on `F_t` converges quadratically only *near* the minimizer of `F_t`, and the textbook description of "near" depends on the condition number of the Hessian at the minimizer and the Lipschitz constant of `∇²f` — and as `t → ∞` and the iterate nears the boundary, the barrier Hessian `X⁻²` becomes wildly ill-conditioned. So if I jump `t` way up and start Newton fresh, I'm outside the good region and convergence is slow and dirty. This is the historical objection, and it's real *if I take the classical convergence statement at face value.* So before I accept it, let me look hard at that statement.

Something about it does smell wrong. Newton's method is *affine-invariant*: if I change coordinates `x = M x̃` the Newton iterates map over exactly, the method doesn't care about the linear frame. Yet the *theory* of where it converges is stated in terms of the condition number of `∇²f(x*)` and the Lipschitz constant of `∇²f` — both of which are *frame-dependent*, they change when I change `M`. So the theory is describing the method using a structure the method itself is blind to. That's an ad hoc Euclidean ruler smuggled in. If there is a frame-*independent* way to say when Newton works, the ill-conditioning fear might be an artifact: ill-conditioning is itself frame-dependent, it's a number you read off a fixed Euclidean ruler. Let me try to find the frame-independent statement and see whether the fear survives it.

A strongly convex `f` defines, at every point `x`, *its own* local inner product: `⟨u,v⟩_x = uᵀ∇²f(x) v`, with the local norm `‖h‖_x = √(hᵀ∇²f(x)h)`. In *this* ruler the Hessian at `x` is, by construction, the identity — perfectly conditioned, no matter what coordinates I'm in. So measure everything in the function's own local norm and the frame-dependence is gone. The only thing left to quantify is how fast the Hessian *changes* — its Lipschitz behavior — but measured in this same local ruler. So I want a bound on the third derivative of `f` controlled by the second derivative, self-referentially. What form can such a bound take? Under `h → λh`, the third derivative `D³f(x)[h,h,h]` scales as `λ³`, and a single factor `D²f(x)[h,h]` scales as `λ²`; to match `λ³` on the right I need `(D²f(x)[h,h])^{3/2}`. So the powers are forced:

```
|D³f(x)[h,h,h]| ≤ κ · (D²f(x)[h,h])^{3/2}
```

for some constant `κ` — and `3/2` is not a free choice, homogeneity in `h` fixes it.

What should `κ` be? Under `f → αf` the two sides scale differently (left by `α`, right by `α^{3/2}`), so `κ` can be normalized by rescaling `f` — meaning a *canonical* value of `κ` should be pinned by the most important function in sight. The barrier is built out of `−ln x` on the positive reals, so let me just compute its third-derivative ratio and let that fix `κ`. For `f(x) = −ln x`: `f'(x) = −1/x`, `f''(x) = 1/x²`, `f'''(x) = −2/x³`. Then `|f'''(x)| = 2/x³` and `(f''(x))^{3/2} = (1/x²)^{3/2} = 1/x³`, so

```
|f'''(x)| = 2 · (f''(x))^{3/2},   exactly, for all x > 0.
```

an identity for every `x > 0`, not just a coincidence at one point. So with `κ = 2` the inequality holds with *equality* for `−ln x`, with no rescaling. That's the natural normalization: `κ = 2`, and `−ln x` is the boundary case of the class. The property is preserved under sums (third derivatives and Hessians add diagonally) and under adding a linear term (`t cᵀx` has zero third derivative), so `φ = −Σ ln xⱼ` and the whole family `F_t` inherit it. Call a function obeying `|D³f[h,h,h]| ≤ 2(D²f[h,h])^{3/2}` self-concordant. `F_t` is self-concordant for every `t`, automatically.

Now does the convergence story actually come out frame-independent? Measure progress by the Newton decrement

```
λ(x,f) = ‖∇f(x)‖*_x = √( ∇f(x)ᵀ [∇²f(x)]⁻¹ ∇f(x) ),
```

the length of the gradient in the local dual norm — equivalently the local-norm length of the Newton step itself, which vanishes exactly at the minimizer. The self-concordance inequality, integrated along the step, gives for the damped Newton step `x⁺ = x − [1/(1+λ)]·[∇²f(x)]⁻¹∇f(x)`:

```
λ(x⁺,f) ≤ 2 λ(x,f)².
```

This is the statement I was hoping existed. Read off its consequences: once `λ < 1/2` the map `λ ↦ 2λ²` contracts (since `2·(1/2)² = 1/2`, and below `1/2` it strictly shrinks), so the decrement squares each step — pure quadratic convergence — and the region of quadratic convergence is `{x : λ(x,f) ≤ 1/4}` (there `2λ² ≤ 2·(1/16) = 1/8 < λ`), a statement with *no* condition number and *no* Lipschitz constant in it. So the late-1960s pessimism was measuring the wrong thing: the ill-conditioning it feared is a reading off the fixed Euclidean ruler, and in the function's own ruler the barrier is as well-behaved as anything. That's what lets me hope for a polynomial bound where SUMT got none.

Now I can design the path-following carefully instead of jumping `t`. Keep the current iterate `x` *close to* the central point `x*(t)` in the sense `λ(x, F_t) ≤ 0.1`. Then nudge `t` up by just enough that `x` is still inside the quadratic-convergence region of the *new* center `x*(t⁺)`, take one Newton step, and I'm back to `λ ≤ 0.1` for the new `t`. How big a nudge can I afford? The amount the center moves when I bump `t` is controlled by a second property the barrier has beyond plain self-concordance: it is a `ϑ`-self-concordant *barrier*, meaning the gradient is also controlled in the local norm,

```
|Df(x)[h]| ≤ √ϑ · √(D²f(x)[h,h]),
```

with `ϑ = n` for `φ = −Σ ln xⱼ` (each `−ln xⱼ` contributes `1`, and they add). This `ϑ` is exactly the gap-vs-`t` constant: the central path satisfies `cᵀx*(t) − min cᵀx ≤ ϑ/t`, which for the orthant barrier is the `n/t` I already verified. The half-power `√ϑ` in the gradient bound is what lets me raise `t` multiplicatively by a factor `(1 + 0.1/√ϑ)` per step while keeping `x` within range of the next center:

```
t ← (1 + 0.1/√n) · t,   then one Newton step on F_t.
```

Each step keeps `λ ≤ 0.1` and multiplies `t` by `1 + 0.1/√n`. To drive the gap `n/t` below `ε` from a starting `t₀`, I need `t` up by a factor `~ n/(ε t₀)`, which at rate `1 + 0.1/√n` per step takes `O(√n · log(n/(t₀ε)))` steps. That `√n`, not `n`, is the prize — and it comes directly from the `√ϑ` in the barrier's gradient bound. Each step is one Cholesky solve with `A D² Aᵀ`. And because each step *re-derives* the certificate from the current point, round-off does not accumulate the way it does in the ellipsoid method — an error in one step is just a slightly-off interior point that the next step corrects. The method is self-correcting.

So I have a provably polynomial, practically-flavored interior algorithm: follow the central path with short Newton steps. Good. But there's a second, quite different route to the same interior idea, one that doesn't start from "trust the barrier" at all — it starts from "make a gradient step honest by rescaling," and it produces a strikingly clean potential-reduction argument of its own.

Reduce the LP, by a preliminary transformation, to a canonical form: minimize `cᵀx` over `x` in the simplex `Δ = {x ≥ 0, Σxⱼ = 1}` intersected with a subspace `Ω = {Ax = 0}` (homogeneous), with the target minimum value known to be `0` and the simplex center `a₀ = e/n` a strictly feasible start. (I'll justify "known minimum value `0`" by combining the primal and dual into one problem in a moment.) Why the simplex? Because at the *center* of the simplex the geometry is as round as it gets: the largest inscribed ball `B(a₀,r)` and the smallest circumscribing ball `B(a₀,R)` satisfy `R/r = n−1`, a controlled ratio. If I'm at the center, I can take a real step.

Here's the bound that makes "round body" pay off. Suppose at the center `a₀` I optimize the objective not over the whole polytope `P` but over an inscribed ellipsoid `E ⊆ P`, getting a point `a′`. How much progress? Inflate `E` by a factor `v` to `E′ = vE ⊇ P`. With `f_E, f_P, f_{E′}` the minima of the linear `f` over the three sets, linearity gives `f(a₀) − f_{E′} = v[f(a₀) − f_E]`, and since `E ⊆ P ⊆ E′`,

```
f(a₀) − f_P ≤ f(a₀) − f_E    and    f(a₀) − f_P ≥ f(a₀) − f_{E′} = v[f(a₀) − f_E],
```

so `[f(a₀) − f_{a′}] / [f(a₀) − f_P] ≥ 1/v`. Optimizing over the inscribed ball drops the gap-to-optimum by at least a factor `1 − 1/v` per step — and `v` is exactly the inscribed-to-circumscribed ratio `R/r`. At the simplex center `v = R/r = n−1`. So one step buys a `(1 − 1/(n−1))` factor; `O(n)` steps buy a constant-factor reduction. That's the engine. But it only works *at the center*, where `R/r` is small.

The catch: after one step I'm no longer at the center. To take another good step I must bring my current point back to the center — and not by translation (that would warp the polytope), but by a transformation that fixes the simplex and carries my point to `a₀`. Which transformation? An affine map can't fix a simplex and move an arbitrary interior point to the center while keeping the structure I need. But a projective map can. Take

```
T(a, a₀): x ↦ x′,   x′ⱼ = (xⱼ/aⱼ) / Σ_k (x_k/a_k),
```

i.e. divide each coordinate by the current point's coordinate `aⱼ` and renormalize to the simplex. In matrix form with `D = diag(a)`, `x′ = D⁻¹x / (eᵀD⁻¹x)`. Check the three properties this needs. A simplex vertex `e_j` has `x_k/a_k = 0` for `k ≠ j`, so it maps to `e_j` — vertices fixed. The facet `xⱼ = 0` maps to `x′ⱼ = 0` — facets fixed. The current point `a` has `aⱼ/aⱼ = 1` for all `j`, so `x′ = (1,…,1)/n = a₀` — the current point goes to the center. And under it the homogeneous constraint `Σ Aᵢxᵢ = 0` becomes `Σ Aᵢaᵢ x′ᵢ = 0`, i.e. the columns rescale to `A′ᵢ = aᵢAᵢ` — still homogeneous. All three properties hold, so in the transformed space I'm at the center, I take my inscribed-ball step there, and then apply `T⁻¹` to come back. Repeat. This is the projective rescaling: instead of distorting the body to re-center, I re-center *by* a body-preserving projective change of coordinates. The affine-scaling instinct (Dikin) is the same instinct — rescale so the current point is central — but projective rather than affine, which is what keeps the simplex and its facets fixed and the constraints homogeneous.

Now the snag that the path-following route also hit, in a new guise: I want to claim "a constant-factor reduction in the objective per `O(n)` steps," but the *objective* `cᵀx` is a linear function, and linear functions are *not* invariant under the projective map `T`. After transforming, `cᵀx` is no longer linear in `x′`. So I can't just track `cᵀx` across steps; I need a quantity that *is* well-behaved under projective maps, so that "reduce it by a constant per step" composes across steps. Ratios of linear functions *are* sent to ratios of linear functions by `T`. So build a potential out of logarithms of such ratios. The objective should appear in the numerator (so reducing it helps) and the barrier in the denominator (so I'm pushed off the boundary). Take

```
f(x) = n · ln(cᵀx) − Σⱼ ln xⱼ.
```

This is `ϑ ln(cᵀx) + φ(x)` with `ϑ = n` — the log of the objective scaled by `n`, plus the log barrier. Two things make it the right object. First, under `T(a,a₀)` it maps to a function of the *same form* (each `ln xⱼ` picks up an additive `−ln aⱼ` constant, and `ln(cᵀx)` becomes log-of-ratio terms that are again logs of linear ratios) — so "reduce `f` by `δ`" is a statement I can make in the transformed frame at the center and then it's *invariant* back in the original frame. Second, reducing `f` forces the objective down: since `Σ ln xⱼ ≥ −n ln n` is bounded on the simplex, driving `f → −∞` forces `n ln(cᵀx) → −∞`, i.e. `cᵀx → 0`, the known target. Concretely `f(x⁽ᵏ⁾)` decreasing by a constant `δ` per step gives, after the bounded barrier terms are accounted for, `ln(cᵀx⁽ᵏ⁾/cᵀa₀) ≤ ln n − kδ/n`, so `cᵀx⁽ᵏ⁾ / cᵀa₀ ≤ 2^{−q}` after `O(n(q + log n))` steps — geometric reduction of the objective.

So one step is: project to center via `T`, take a step inside the inscribed ball `B(a₀, αr)` that reduces the (transformed) potential, project back via `T⁻¹`. The remaining question is how to take that central step, and here the linear-approximation trick closes the loop with the round-body bound. At the center, optimizing the *potential* `f′` over the inscribed ball can be approximated by optimizing the *linear* function `c′ᵀx` (with `c′ = Dc`) over the ball, because over a small enough ball the log-of-ratio potential is, to first order, that linear function. So the central step is just: project `c′` orthogonally onto the constraint null space, normalize, and step a length `αr` against it —

```
c_p = [I − Bᵀ(BBᵀ)⁻¹B] Dc,    ĉ = c_p/‖c_p‖,    b′ = a₀ − α r ĉ,
```

where `B` is `AD` with a row of ones appended (to keep `Σx=1`), and `r = 1/√(n(n−1))` is the inscribed-ball radius.

Why optimize over a ball of radius `αr` with `α = 1/4` rather than the full inscribed ball `r`? The linear approximation of the potential is only accurate over a small ball, and `α < 1` keeps it tight; and a margin below the boundary lets me do the arithmetic approximately and absorb round-off without leaving the simplex. But `α` is a real knob with a real cost — too small and the per-step potential reduction `δ` shrinks toward zero and I lose the geometric rate. So I should not just declare `α = 1/4` good; I should compute the `δ` it actually delivers. The per-step reduction works out to

```
δ = ln(1 + α) − β²/(2(1−β)),     β = α·√(n/(n−1)),
```

where the first term is the drop along the segment from `a₀` to the true linear minimizer `x*` clipped to the ball (using `Π(1+Pⱼ) ≥ 1 + ΣPⱼ`), and the subtracted term is the second-order slack between the *linear* minimizer and the *potential* minimizer, bounded via `|Σ ln(xⱼ/a₀ⱼ)| ≤ β²/(2(1−β))` on the ball. Let me actually evaluate this rather than wave at "`δ ≈ 1/8`." The worst case for `δ` is small `n` (`β` largest); plugging in:

```
n = 3:    β = 0.25·√(3/2) = 0.30619,  δ = ln 1.25 − 0.30619²/(2·0.69381) = 0.22314 − 0.06756 = 0.15558
n = 4:    β = 0.28868,                δ = 0.22314 − 0.05857 = 0.16457
n = 10:   β = 0.26352,                δ = 0.22314 − 0.04714 = 0.17600
n = 100:  β = 0.25126,                δ = 0.22314 − 0.04215 = 0.18099
n → ∞:    β = 0.25,                   δ = 0.22314 − 0.04167 = 0.18148
```

So across all `n ≥ 3`, `δ` ranges from about `0.156` (worst, at `n=3`) up to `0.181` (the `n→∞` limit) — every value comfortably above `1/8 = 0.125`. Good: I can safely state `δ ≥ 1/8` as a clean lower bound, and it's a *conservative* one (the truth is closer to `0.15–0.18`), so `α = 1/4` is a sound choice and the `O(n(q + log n))` step count holds. If I'd picked `α` much smaller the `ln(1+α)` term would have collapsed and `δ` could have gone negative; the computation is what tells me `1/4` is in the safe zone rather than my having guessed it.

I should pin down the "minimum value known to be `0`" and the starting point, because the whole canonical form leaned on them. The trick is to *combine primal and dual into one problem*. Stack the primal `Ax ≥ b, x ≥ 0`, the dual `Aᵀu ≤ c, u ≥ 0`, and the no-gap condition `cᵀx − bᵀu = 0`. By duality, this combined system is feasible *iff* the original LP has a finite optimum, and then any feasible point of the combined system is primal-and-dual optimal — so the optimal objective of the feasibility problem is `0`, exactly the target the potential drives toward. Introduce slacks (`Ax − y = b`, `Aᵀu + v = c`) and one artificial variable `λ` (with column `b − Ax₀ + y₀` etc.) so that an obvious strictly interior point `(x₀,y₀,u₀,v₀,λ=1)` is feasible and minimizing `λ` to `0` solves the combined system. Then the projective map of the positive orthant into the simplex puts it in canonical form. So the "known target `0`" isn't an assumption — it's manufactured by folding optimality into feasibility.

One more practical point on cost, because it's what separates this from the ellipsoid method. Each step's work is dominated by computing `c_p`, which needs the inverse of `BBᵀ` — really of `A D² Aᵀ` where `D` is diagonal. But `D` changes only slightly from step to step (only the current point's coordinates move, and only a little). So instead of refactoring from scratch, maintain a *working approximation* `D′` to `D` and update `(A D′² Aᵀ)⁻¹` by rank-one corrections whenever an entry of `D′` drifts too far (keeping each ratio `D′ⱼ/Dⱼ ∈ [1/2, 2]`). Each rank-one update is `O(n²)` by the Sherman–Morrison formula `(M+uvᵀ)⁻¹ = M⁻¹ − (M⁻¹u)(M⁻¹v)ᵀ/(1+vᵀM⁻¹u)`. A counting argument over `m` steps shows the total number of updates is `O(m√n)`, so the *average* work per step is `O(n^{2.5})` instead of `O(n³)`. Replacing the exact `D` by the approximate `D′` corresponds to solving the inscribed-ball problem with a weighted ellipsoid instead of the round ball; since the entries of the error matrix stay in `[1/2,2]`, the ball is sandwiched and the per-step potential-reduction guarantee survives with a slightly redefined constant. Net worst case `O(n^{3.5} L)` arithmetic on `O(L)`-bit numbers — better than the ellipsoid's `O(n⁶L²)` by a factor of `n^{2.5}`.

Now, the projective potential-reduction method and the central-path follower are the same animal seen from two sides. The potential `f = n ln(cᵀx) − Σ ln xⱼ` is `ϑ ln(cᵀx) + φ(x)` with `ϑ = n`; under the projective map `p(x) = x/cᵀx` it equals the restriction of the barrier `φ` to a slice, and "reduce the potential" is "run to infinity along a ray," which is "`cᵀx → 0`," which is "slide down the central path." The barrier route gives the better `√n` iteration count (from the `√ϑ` gradient bound and short steps near the path); the potential route gives a single monotone merit function and a step that can be lengthened by line search far beyond the conservative theoretical `δ`. Either way the per-step kernel is one solve with `A D² Aᵀ`.

Which leaves the question of what to actually *implement*, because the version that wins in practice keeps both the primal and the dual in play at once instead of following only the primal path. Go back to the deformed-KKT view. The thing I'm really hunting is a point satisfying

```
Ax = b,  x > 0      (primal feasibility)
Aᵀy + s = c,  s > 0 (dual feasibility)
xⱼ sⱼ = 0           (complementarity)
```

and the central path is the smoothing `xⱼ sⱼ = μ` with `μ = 1/t` shrinking to `0`. So treat the whole thing as a single nonlinear system in `(x,y,s)`:

```
F(x,y,s) = [ Ax − b ;  Aᵀy + s − c ;  X S e − μ e ] = 0,
```

`X = diag(x)`, `S = diag(s)`, `e = (1,…,1)ᵀ`, and apply Newton's method to it, *targeting* `μ` a bit smaller than the current average `μ_now = xᵀs/n` each iteration, while keeping `x>0, s>0`. The Newton system is

```
[ A   0   0 ] [ Δx ]   [ −(Ax−b)        ]
[ 0   Aᵀ  I ] [ Δy ] = [ −(Aᵀy+s−c)     ]
[ S   0   X ] [ Δs ]   [ −(XSe − σμ e)  ],
```

with a centering parameter `σ ∈ [0,1]`: `σ=1` aims straight at the central point (pure centering), `σ=0` aims straight at optimality (the affine-scaling / predictor direction). Eliminating `Δs = −X⁻¹(SΔx + (XSe − σμe))` and then `Δx` reduces this, exactly as before, to one solve with a matrix `A (S⁻¹X) Aᵀ` — same `A D² Aᵀ` kernel, `D² = S⁻¹X`.

The practical scheme that emerges (Mehrotra's predictor–corrector) does this in two solves sharing one factorization. First a *predictor*: solve with `σ=0` (the affine direction) and ask how much the complementarity `μ` would drop if I took that step — the ratio `μ_aff/μ` measures how good the affine direction is. Set the centering adaptively, `σ = (μ_aff/μ)³`: if the affine step is great (`μ_aff` tiny) `σ→0` and I barrel toward optimality; if it's poor, `σ→1` and I re-center. Then a *corrector* reuses the same factorization to add the second-order term: the affine step left a residual `Δx_aff ∘ Δs_aff` in the bilinear complementarity equation, so fold it into the right-hand side, `XSe → XSe + Δx_aff∘Δs_aff − σμe`, and solve again. Add the two directions, and finally choose step lengths by the *fraction-to-boundary* rule — take `η = 0.9995` of the largest step that keeps `x>0` and `s>0` — separately in the primal and dual. Stop when the relative residual `‖(Ax−b, Aᵀy+s−c, X S e)‖/(1+max(‖b‖,‖c‖))` is below tolerance, which is just the duality gap plus feasibility, the certificate I built at the very start.

Let me write it — `solve_lp` is exactly the piece just derived: one Cholesky factorization of `A diag(x/s) Aᵀ` per iteration (with a `1e-10` diagonal nudge before the factorization, so a near-singular late-stage `D` doesn't kill the Cholesky), reused for the affine predictor and for the corrector that folds in `Δx_aff∘Δs_aff − σμe`, then a fraction-to-boundary step at `η = 0.9995`, stopping on the KKT residual `‖(Ax−b, Aᵀy+s−c, x∘s)‖`. The one piece not yet pinned down is the starting point. Take the least-norm primal solution `x = Aᵀ(AAᵀ)⁻¹b` and the least-squares dual `y` minimizing `‖Aᵀy − c‖` (so `s = c − Aᵀy`), then shift whichever of `x, s` has a non-positive entry into the interior by `1.5·|min|`, and push them further apart from `0` by half their inner product divided by the other's coordinate sum, so `x∘s` doesn't start too close to the boundary. Reasonable enough — though it's exactly this last heuristic, not the Newton iteration, that turns out to be the fragile part, as a later random-LP check exposes.

Let me trace it on a problem I can solve by hand: `max 2x₁+3x₂` s.t. `x₁+x₂ ≤ 4, x₁−x₂ ≤ 1`. The corner `(0,4)` gives `12`; `(2.5,1.5)` gives `9.5`; `(0,0)` gives `0` — so the optimum is `12` at `(0,4)`. In standard form, negate to minimize and add slacks `x₃,x₄ ≥ 0`:

```
A = [[1, 1, 1, 0],   b = [4, 1],   c = [-2, -3, 0, 0].
     [1,-1, 0, 1]]
```

Running `solve_lp(A,b,c)` and printing `μ` and `cᵀx` each iteration:

```
iter 0: mu=3.25e+00  res=1.76e+00  cx=-9.200980
iter 1: mu=3.28e-01  res=2.22e-01  cx=-9.559168
iter 2: mu=7.55e-01  res=3.94e-01  cx=-11.313722
iter 3: mu=3.16e-02  res=2.17e-02  cx=-11.887617
iter 4: mu=1.66e-05  res=1.09e-05  cx=-11.999944
iter 5: mu=8.30e-09  res=5.46e-09  cx=-12.000000
```

It returns `x = (0, 4, 0, 5)` with `cᵀx = −12`, i.e. the maximum `12` at `(x₁,x₂) = (0,4)` — the hand answer, in 5 iterations. The `μ` trace is informative: it isn't monotone (it ticks up at iter 2), which is the adaptive `σ` re-centering after a too-aggressive affine step, and then it collapses by orders of magnitude per step (`3e-2 → 2e-5 → 8e-9`) once it locks onto the path — the quadratic tail I argued for. And `(0,4)` is a *vertex*: the interior method approached the boundary and the fraction-to-boundary rule let the right coordinates go to zero in the limit. So the whole arc — interior start, central-path approach, boundary optimum recovered — actually happens.

Now a harder check: a random standard-form LP, `m=8, n=20`, against `scipy.optimize.linprog` (HiGHS). My first attempt at this exposed something I should record. I generated `A` standard-normal, `b = A·x_feas` for a random positive `x_feas`, and `c` *fully random* — and `solve_lp` overflowed to NaN on the second iteration. The cause is the starting point, not the iteration: with a fully random `c`, the least-norm dual `s = c − Aᵀ(AAᵀ)⁻¹Ac` lands with many large-magnitude negative entries, the `1.5·|min|` shift inflates the start enormously, `d = x/s` spans a huge dynamic range, and the first normal-equations solve produces a step that flies out of the cone. The fix is to generate a *dual-feasible* `c` (`c = Aᵀy₀ + s₀` with `s₀ > 0`), which is the regime the simple Mehrotra start is built for. With that, across eight seeds the method matches HiGHS:

```
seed 0:  ipm=-1.758738  scipy=-1.758738  |diff|=1.3e-08  iters=6  feas=2.0e-13
seed 1:  ipm= 2.507460  scipy= 2.507460  |diff|=1.6e-10  iters=6  feas=1.3e-14
seed 4:  ipm=12.447122  scipy=12.447122  |diff|=1.2e-10  iters=6  feas=1.8e-14
seed 5:  ipm=-10.99777  scipy=-10.99777  |diff|=1.7e-08  iters=6  feas=7.4e-14
...   (all eight: |diff| < 2e-8, feas < 1e-12, 6 iterations each)
```

Six iterations to match a production solver to `~1e-8`, feasibility at machine level — and a sharp reminder that the `1e-10` diagonal shift and the `1.5·|min|` start are load-bearing heuristics, not decoration: the method's *convergence* is robust (self-correcting, as argued), but its *starting heuristic* assumes a reasonably scaled, dual-feasible-ish problem, and a casually-built instance can break the start before the self-correction ever gets a chance. That's a real limitation of this implementation, distinct from the theory, and worth keeping in mind.
