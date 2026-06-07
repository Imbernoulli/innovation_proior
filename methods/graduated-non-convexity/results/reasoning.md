Let me start from the thing that actually breaks in practice: I fit a model to data by least squares, and a few bad measurements ruin the fit. Concretely, I have points `(t_i, y_i)` that should lie near a line `y = a t + b`, and I solve `min_{a,b} Σ_i (a t_i + b − y_i)²`. If every `y_i` is the true line plus a little Gaussian noise, this is perfect — it's the maximum-likelihood estimate. But suppose a fraction of the points are *garbage*: a mismatched correspondence, a sensor that returned nonsense, a depth reading that fell between two surfaces. Each such point has a huge residual `x_i = a t_i + b − y_i`, and least squares charges me `x_i²` for it. The optimizer, trying to reduce that enormous squared term, tilts the whole line toward the garbage. A single outlier at distance `D` contributes `D²`, which can dwarf the contribution of all the good points combined, so the fit is whatever minimizes the pull of the outliers, not the bulk. With many outliers the line is meaningless.

I want to see *why* least squares is so fragile, because the precise diagnosis points at the fix. The quantity that matters is how much a single measurement can move the solution — its *influence*. For an error norm `ρ(x)` summed over residuals, the influence of a residual is proportional to the derivative `ψ(x) = ρ'(x)`: it's the "force" that residual exerts on the fit. For least squares `ρ(x) = x²`, so `ψ(x) = 2x` — the force grows *linearly and without bound* in the residual. That's the whole disease in one line. A point a thousand units off the line pulls a thousand times harder than a point one unit off. No amount of good data can outvote one sufficiently distant outlier, because the outlier's vote grows with its distance.

So the cure has to be a `ρ` whose influence does *not* grow without bound — ideally one whose influence *falls back toward zero* for large residuals, so that a point far enough off the line is effectively ignored, contributes essentially nothing, casts no vote. This is the redescending idea from robust statistics. If `ψ(x) → 0` as `|x| → ∞`, then beyond some scale a residual stops mattering: the estimator looks only at the bulk and discards the gross errors. What does such a `ρ` look like? Near zero I still want it quadratic — for the good points, behaving like least squares is exactly right, that's the right noise model. Far away I want it *flat* — once a point is clearly an outlier, increasing its residual further shouldn't change the cost, because I've already decided to ignore it; a flat region means zero influence. So `ρ` should be a bowl near the origin that *saturates* to a constant far out.

The cleanest such function is the truncated quadratic: `ρ(x) = min(x², c̄²)` — quadratic until the residual squared hits a threshold `c̄²`, then capped. Below the cap the point is an inlier and behaves normally; at the cap it's declared an outlier and pays a fixed price `c̄²` no matter how far out it is. And I recognize this object from a completely different direction, which is reassuring. In piecewise-smooth *reconstruction* — recover a surface `u` from noisy data `d`, smooth almost everywhere but with discontinuities at boundaries — people regularize with `Σ_s (u_s − d_s)² + λ Σ_{s~t} (u_s − u_t)²`. The smoothness term `(u_s − u_t)²` is a quadratic that blurs every edge, because a steep gradient is penalized without bound. Geman and Geman's (1984) fix was a binary *line process* `l_{st} ∈ {0,1}`: write the bond as `(u_s − u_t)²(1 − l_{st}) + α l_{st}`, so flipping `l_{st}=1` *cuts* the bond — replace the smoothness penalty by a fixed cost `α` and let the surface jump freely there. Now if there's no spatial coupling among the `l`'s, I can minimize over each `l_{st}` independently in closed form: `min_{l ∈ {0,1}} [ (u_s−u_t)²(1−l) + α l ] = min((u_s−u_t)², α)`. Out drops the truncated quadratic again. So "reject outliers of the data" and "allow discontinuities in the reconstruction" are the *same* mathematical move — quadratic near zero, capped far away. The cap is a discontinuity; the cap is an outlier. Good, one idea covers both.

Here's the wall, and it's a hard one. The truncated quadratic is *non-convex*. The bowl-then-flat shape has a kink and a flat shoulder; summed over many residuals, each of which is a non-convex function of the parameters, the total energy is riddled with local minima. If I just run gradient descent or coordinate descent on `Σ_i min(x_i(a,b)², c̄²)`, I land in whichever local basin my starting guess happens to sit in. And which basin that is corresponds to which subset of points the descent decided were "inliers" — once a point's residual crosses the cap during the iteration, it's frozen as an outlier and stops pulling, but that freezing depends entirely on where I started. Different initializations give different inlier sets give different fits. The global minimum — the *correct* fit, where the true inliers are below the cap and the true outliers above it — is one basin among combinatorially many, and descent has no way to find it without already roughly knowing the answer. This is exactly the trap that makes redescending estimators "powerful but fragile": the power is the non-convexity, and so is the fragility.

I could throw a stochastic global optimizer at it — simulated annealing, the way Geman and Geman minimized their line-process energy. That works in principle: cool slowly and the random walk concentrates on the global minimum. But it's expensive and slow, and it feels like overkill. The non-convex energy I have isn't arbitrary; it's a *specific*, structured function, quadratic-near-zero-flat-far-out. Surely I can exploit that structure deterministically instead of paying for stochastic search. Let me think about what's actually making it hard.

The hardness is the *flatness* — the cap. If the cost were globally convex, descent would find the unique global minimum from any start, no init needed, no search. The truncated quadratic is convex on the bowl and convex (trivially) on the flat shoulder, but the *transition* between them — the kink, where the function stops curving up and goes flat — is where convexity dies. Negative curvature lives in that transition region. So the obstacle is concentrated in the shoulder where inlier becomes outlier.

What if I don't attack the true cost directly, but a *softened* version of it — one where the transition is gentle enough that the whole thing is convex? Picture replacing the sharp kink with a smooth bridge: quadratic bowl, then a smooth concave segment that bends the curve over, then the flat shoulder. The concave bridge has negative curvature, that's unavoidable if the function is going to level off. But here's the lever: how *negative* that curvature is, is something I can control by how *gradually* the bridge bends. A very gentle, very wide bridge has only mildly negative curvature. And the total energy is the data term plus this smoothness/robust term. The data term `Σ (a t_i + b − y_i)²` is quadratic — it has *positive* curvature, a definite amount of it. So if I make the robust term's most-negative curvature small enough that the data term's positive curvature dominates everywhere, the *sum* is convex. There's a threshold: spread the bridge wide enough — make the concave region's curvature shallow enough — and the convexity of the data term wins, and the whole energy becomes convex with a unique global minimum.

That's the move. Build a *family* of cost functions `ρ_μ`, parameterized by a control `μ` that sets how sharp the bowl-to-flat transition is. At one extreme of `μ` the transition is so gradual that the energy is convex — call this the convex surrogate. At the other extreme `ρ_μ` sharpens back into the true truncated quadratic with its hard cap. Then: start at the convex end, where I can find the global minimum from any starting point with no init at all, because there's only one minimum. Then move `μ` a little toward the sharp end, deforming the cost slightly, and re-minimize *starting from the previous solution* — the minimum I just found is an excellent initialization for the slightly-deformed cost, because I only nudged the landscape. Keep going: gradually sharpen `μ` back to the true cost, re-minimizing at each step, tracking the minimizer as it migrates from the easy convex landscape into the hard non-convex one. If each deformation is small enough that the tracked minimum doesn't jump basins, I arrive at the true cost sitting in (or very near) its *global* minimum — the correct robust fit — without ever having done a global search. Graduated non-convexity: graduate the non-convexity in, don't swallow it whole.

Why graduate slowly instead of jumping straight to the true cost? Because jumping to the true cost from the convex solution is just "minimize the non-convex cost starting from one particular point" — and that point, the convex-surrogate minimizer, is some compromise fit that doesn't yet know which points are outliers; descend from there into the truncated quadratic and I fall into whatever basin is nearest, which is back to the original trap. The slowness is the whole method: each small step keeps the tracked minimizer in the global basin as that basin slowly deforms, so I never have to *find* the global basin — I'm *born* in it (at the convex end) and I never leave it. It's the deterministic analogue of cooling slowly in annealing: there, slow cooling keeps a stochastic walk near equilibrium; here, slow deformation keeps a deterministic minimizer near the global optimum. Same principle, no randomness.

Now I need the actual family. Let me take the truncated quadratic and write down a one-parameter surrogate. I'll work with the Geman–McClure form first because it's the cleanest — it's already smooth, no piecewise cases. Geman–McClure is `ρ(x) = x²/(1 + x²)` (with a scale `c̄`, `ρ(x) = c̄² x²/(c̄² + x²)`): quadratic-like `≈ x²` near zero, saturating to `c̄²` as `|x| → ∞`. It's redescending and non-convex. To make a surrogate, introduce `μ` so that
`ρ_μ(x) = μ c̄² x² / (μ c̄² + x²)`.
Look at the limits. As `μ → ∞`, the denominator `μ c̄² + x² → μ c̄²`, so `ρ_μ(x) → x²` — pure quadratic, convex. As `μ → 1`, `ρ_μ` is exactly the original Geman–McClure cost. So `μ` sweeps from a convex quadratic (`μ` large) down to the true non-convex cost (`μ = 1`). For Geman–McClure the continuation runs `μ` *downward*. Good — that's the convex-to-true family I wanted, and the control sits inside the function smoothly.

So the algorithm shape is: outer loop over `μ` from large to 1; inner, minimize `Σ_i ρ_μ(x_i(a,b))`. But the inner minimization is still a nonlinear least-squares-ish problem in `(a,b)` for each fixed `μ`. I'd like each inner step to be *cheap and global* — ideally reduce it to ordinary weighted least squares, which I can solve in closed form. Can I? This is where I want a way to turn `min Σ ρ_μ(x)` into a reweighted-least-squares iteration with an *explicit* weight I can write in closed form.

Reweighted least squares is the classic trick: to minimize `Σ ρ(x_i)`, alternate — fix the weights, solve weighted least squares; fix the fit, update the weights `z_i = ρ'(x_i)/(2 x_i)`. That weight down-weights large residuals automatically. But in plain IRLS the weight is *slaved* to the residual, `z = ρ'(x)/(2x)`, a mere function of `x`; there's no explicit outlier variable, and worse, IRLS just descends on the same non-convex `ρ`, so it inherits the local-minimum trap. I want something better: an explicit formulation where the weight is its *own* variable, ranging over `[0,1]`, that I can constrain and reason about — and where the equivalence to the robust cost is exact, not heuristic.

Let me try to manufacture exactly that. Introduce, for each residual `x`, a variable `z ∈ [0,1]` — think of it as "how much I trust this measurement," `z ≈ 1` an inlier, `z ≈ 0` an outlier — and write a *joint* energy
`E(x, z) = x² z + Ψ(z)`
where `Ψ(z)` is a penalty I get to design. The first term: if I trust the point (`z=1`) I pay its full squared residual `x²`; if I distrust it (`z=0`) I pay nothing for the residual. The second term `Ψ(z)` is the price of distrust — it must *penalize* setting `z` small, otherwise the trivial solution `z=0` everywhere (ignore all data) would win. Now the question: can I choose `Ψ` so that minimizing `E(x,z)` over `z` reproduces my robust cost, i.e. `min_{z ∈ [0,1]} (x² z + Ψ(z)) = ρ(x)`? If so, the robust estimation problem `min_x Σ ρ(x_i)` becomes the joint problem `min_{x, z_i} Σ [x_i² z_i + Ψ(z_i)]`, and *that* I can solve by alternating: fix `z`, solve weighted least squares in `x` (closed form, global); fix `x`, solve for `z` (which, as I'll see, is closed form per point). The weights `z_i` are now first-class variables.

To find `Ψ`, demand that the two formulations have the same minimizer in `x`. At the optimum of `E` over `z`, set `∂E/∂z = 0`: `x² + Ψ'(z) = 0`. And I want the `x`-gradient of `E` (after eliminating `z`) to match that of `ρ`: differentiating `E = x² z + Ψ(z)` total in `x` with `z` at its optimum, the `∂/∂z` part vanishes by stationarity, leaving `dρ/dx = 2x z`, so `z = ρ'(x)/(2x)`. There's my weight — and it's exactly the IRLS weight, which tells me I'm on the right track: the explicit-`z` formulation *contains* IRLS, but now `z` is a variable I introduced deliberately, not a definition forced on me.

Now solve for `Ψ`. To make the algebra clean, change variable to `w = x²` and define `φ(x²) ≜ ρ(x)`, so `φ` is `ρ` as a function of the squared residual. Then `φ'(x²) = ρ'(x)/(2x)` by the chain rule (`dφ/d(x²) = (dρ/dx)/(d(x²)/dx) = ρ'(x)/(2x)`), so the weight is simply `z = φ'(x²)`. From the stationarity condition `x² + Ψ'(z) = 0` and `z = φ'(x²)`, I get `−x² = Ψ'(φ'(x²))`. To recover `Ψ`, integrate. Multiply both sides by `2x φ''(x²)` (assume `φ'' ≠ 0` for now) and integrate in `x`: the left side `∫ Ψ'(φ'(x²)) φ''(x²) d(x²) = Ψ(φ'(x²))`, and the right side `∫ −x² φ''(x²) d(x²) = −x² φ'(x²) + φ(x²)` by parts. So
`Ψ(φ'(x²)) = −x² φ'(x²) + φ(x²)`.
Substitute `z = φ'(x²)`, hence `x² = (φ')⁻¹(z)`, to get `Ψ` as a function of `z` alone:
`Ψ(z) = φ((φ')⁻¹(z)) − z (φ')⁻¹(z)`.
That's a clean construction — `Ψ` is (minus) the Legendre/convex-conjugate transform of `φ`. Given any robust cost, compute `φ`, invert `φ'`, and read off the penalty.

When does this work — when is `z = φ'(x²)` actually a *minimum* of `E` over `z`, and does `z` land in `[0,1]`? For it to be a minimum I need `∂²E/∂z² = Ψ''(φ'(x²)) > 0`. Differentiating `−x² = Ψ'(φ'(x²))` in `x` gives `Ψ''(φ'(x²)) = −1/φ''(x²)`, so `Ψ'' > 0` exactly when `φ'' < 0` — when `φ` is *concave*. So the condition for a robust cost to have this explicit-outlier-weight form is that `φ(w) = ρ(√w)` be concave. And for `z` to be a legitimate "trust" variable in `[0,1]`, I need it to range over `[0,1]` as the residual goes from `0` to `∞`: `lim_{w→0} φ'(w) = 1` (zero residual ⇒ full trust) and `lim_{w→∞} φ'(w) = 0` (huge residual ⇒ no trust). Three conditions: `φ'(0)=1`, `φ'(∞)=0`, `φ'' < 0`. Every redescending robust norm I care about satisfies them.

Let me verify on Geman–McClure, both because it's the case I'll use and to make sure the construction is right. `ρ(x) = x²/(1+x²)`, so `φ(w) = w/(1+w)`. Then `φ'(w) = 1/(1+w)²`. Check the conditions: `φ'(0) = 1` ✓, `φ'(∞) = 0` ✓, and `φ''(w) = −2/(1+w)³ < 0` ✓ concave. The weight is `z = φ'(x²) = 1/(1+x²)²` — small for large residual, exactly the soft outlier rejection I want. Invert: `z = 1/(1+w)²` gives `1+w = z^{−1/2}`, so `(φ')⁻¹(z) = −1 + 1/√z`. Then
`Ψ(z) = φ((φ')⁻¹(z)) − z(φ')⁻¹(z)`.
With `w = (φ')⁻¹(z) = −1 + 1/√z`, I have `φ(w) = w/(1+w) = (−1+1/√z)/(1/√z) = 1 − √z`, and `z·w = z(−1+1/√z) = −z + √z`, so `Ψ(z) = (1 − √z) − (−z + √z) = 1 − 2√z + z = (1 − √z)² = (√z − 1)²`. Clean. So Geman–McClure's outlier process is `Ψ(z) = (√z − 1)²`, and
`E(x,z) = x² z + (√z − 1)²`.
Let me sanity-check that `min_z E` really gives back `ρ`. At `x = 1`: minimize `z + (√z−1)²` over `z∈[0,1]`. The weight should be `z = 1/(1+1)² = 1/4`; plug in, `E = 1·(1/4) + (1/2 − 1)² = 0.25 + 0.25 = 0.5`, and `ρ(1) = 1/2`. ✓. At `x = 2.5`: `z = 1/(1+6.25)² ≈ 0.019`, `E ≈ 6.25·0.019 + (0.138−1)² ≈ 0.119 + 0.743 ≈ 0.862`, and `ρ(2.5) = 6.25/7.25 ≈ 0.862`. ✓. The duality holds exactly.

Now stitch duality to the continuation. I don't apply the duality to the *true* `ρ`; I apply it to each *surrogate* `ρ_μ`. The construction goes through with `μ` carried along: for the Geman–McClure surrogate `ρ_μ(x) = μ c̄² x²/(μ c̄² + x²)`, the same recipe yields the penalty `Φ_{ρ_μ}(z) = μ c̄² (√z − 1)²` and, at fixed `x` (residual `r̂²`), the weight that minimizes `z r̂² + Φ_{ρ_μ}(z)` in closed form. Differentiate `z r̂² + μ c̄²(√z − 1)²` in `z`: `r̂² + μ c̄²(1 − 1/√z) = 0` ⟹ `1/√z = 1 + r̂²/(μ c̄²) = (μ c̄² + r̂²)/(μ c̄²)` ⟹ `√z = μ c̄²/(μ c̄² + r̂²)` ⟹
`z = (μ c̄² / (μ c̄² + r̂²))²`.
A residual `r̂` small compared to `√(μ c̄²)` gets weight near 1; large gets weight near 0; and as `μ → ∞` the weight `→ 1` for every finite residual — which is exactly why the surrogate is convex there (everyone's an inlier, it's plain least squares). As `μ → 1` the weights sharpen to the true Geman–McClure rejection. The continuation in `μ` *is* a gradual hardening of the outlier weights.

So each outer iteration, at fixed `μ`, I alternate two closed-form steps: **variable update** — fix weights `z_i`, solve `min_{a,b} Σ_i z_i (a t_i + b − y_i)²`, ordinary weighted least squares, `x = (AᵀZA)⁻¹ AᵀZy`, global and cheap; **weight update** — fix the fit, set each `z_i = (μ c̄²/(r̂_i² + μ c̄²))²` from the residuals. Then step `μ` toward the true cost and repeat. The inner problem is never non-convex — it's weighted least squares — and all the non-convexity is handled by the slow march of `μ`. That's the payoff of routing through the duality: the surrogate's hard inner minimization is replaced by alternating two convex/closed-form pieces.

Let me also work the truncated-quadratic surrogate directly, because that's the cost I most want (a clean cap, a true outlier/inlier dichotomy) and its continuation runs in the *opposite* direction, which is worth pinning down so I don't get the schedule backward. The truncated least squares cost is `ρ(x) = x²` for `x² ≤ c̄²`, else `c̄²`. I need a smooth surrogate that interpolates a quadratic bowl and the flat shoulder with a controllable concave bridge. Take
`ρ_μ(x) = x²` for `x² ∈ [0, μ/(μ+1) c̄²]`; `= 2 c̄|x|√(μ(μ+1)) − μ(c̄² + x²)` for `x² ∈ [μ/(μ+1) c̄², (μ+1)/μ c̄²]`; `= c̄²` for `x² ≥ (μ+1)/μ c̄²`.
The middle piece is the concave bridge — its second derivative in `x` is `ρ_μ'' = −2μ`, a *negative* curvature whose magnitude is `2μ`. So when `μ → 0`, the bridge's negative curvature `→ 0`: the surrogate becomes convex (the bridge flattens out into a straight tangent line, no negative curvature left for the data term's positive curvature to fight). And as `μ → ∞`, the two breakpoints `μ/(μ+1) c̄²` and `(μ+1)/μ c̄²` both `→ c̄²`, the bridge collapses to a point, and `ρ_μ` recovers the hard truncated quadratic. So for truncated least squares the continuation runs `μ` *upward*, from `0` (convex) to `∞` (true cost) — mirror image of Geman–McClure. The invariant across both is "increase the non-convexity gradually"; the direction of `μ` is just a convention of each parameterization.

I should check the surrogate is continuous at the breakpoints, or the whole thing is ill-defined. At `x² = μ/(μ+1) c̄²`: the quadratic gives `x² = μ/(μ+1) c̄²`. The bridge gives `2c̄|x|√(μ(μ+1)) − μ(c̄² + x²)`; with `|x| = √(μ/(μ+1)) c̄`, the first term is `2c̄ · √(μ/(μ+1)) c̄ · √(μ(μ+1)) = 2 c̄² μ`, and `−μ(c̄² + μ/(μ+1) c̄²) = −μ c̄²(1 + μ/(μ+1)) = −μ c̄² (2μ+1)/(μ+1)`. Sum: `μ c̄²(2 − (2μ+1)/(μ+1)) = μ c̄² ( (2μ+2 − 2μ−1)/(μ+1) ) = μ c̄²/(μ+1)`, which equals the quadratic value `μ/(μ+1) c̄²`. ✓ Continuous. At `x² = (μ+1)/μ c̄²`: bridge with `|x| = √((μ+1)/μ) c̄` gives first term `2c̄·√((μ+1)/μ)c̄·√(μ(μ+1)) = 2c̄²(μ+1)`, minus `μ(c̄² + (μ+1)/μ c̄²) = μ c̄² + (μ+1)c̄² = c̄²(2μ+1)`, sum `c̄²(2μ+2 − 2μ−1) = c̄²` — equals the flat value. ✓.

Run the duality on this and the weight update comes out as a thresholded version of the Geman–McClure one. The penalty is `Φ_{ρ_μ}(z) = μ(1−z)/(μ+z) c̄²`, and minimizing `z r̂² + Φ_{ρ_μ}(z)` over `z ∈ [0,1]` gives a clamped weight: if the squared residual exceeds the upper threshold `(μ+1)/μ c̄²`, the point is decisively an outlier and `z = 0`; if it's below the lower threshold `μ/(μ+1) c̄²`, it's decisively an inlier and `z = 1`; in between, the smooth bridge gives `z = √(c̄² μ(μ+1)/r̂²) − μ`. Let me confirm the in-between formula joins the clamps continuously. At `r̂² = μ/(μ+1) c̄²` (lower threshold): `z = √(c̄² μ(μ+1)·(μ+1)/(μ c̄²)) − μ = √((μ+1)²) − μ = (μ+1) − μ = 1` ✓. At `r̂² = (μ+1)/μ c̄²` (upper): `z = √(c̄² μ(μ+1)·μ/((μ+1) c̄²)) − μ = √(μ²) − μ = μ − μ = 0` ✓. So the truncated-least-squares weight slides smoothly from 1 down to 0 across the bridge — a *soft* inlier/outlier decision that hardens as `μ` grows and the two thresholds squeeze together toward `c̄²`, at which point it becomes a hard 0/1 cut. That's the binary line process re-emerging in the `μ → ∞` limit, which is exactly right: the true truncated quadratic *is* the eliminated binary line process.

Now the schedule. I need a starting `μ` and a stepping rule. For Geman–McClure I want to start convex over the *actual* residual range — no point making it convex out at residuals I'll never see. After the first weighted-least-squares fit (weights all 1), look at the largest squared residual `r²_max`. A reasonable convexifying start is `μ = 2 r²_max / c̄²` — large enough that even the worst residual sits where the surrogate is still quadratic — then decrease `μ ← μ/1.4` each outer iteration, stopping when `μ` drops below 1 (the true cost). For truncated least squares, mirror it: start `μ = c̄²/(2 r²_max − c̄²)` (small, convex) and *increase* `μ ← 1.4 μ`, stopping when the weighted residual sum `Σ_i z_i r̂_i²` converges or the weights have all gone binary. Why a geometric factor like 1.4 rather than additive steps? Because it's scale-free — it doesn't presume an absolute scale for `μ`, and it automatically takes comparable *fractional* steps throughout, so the deformation per step stays roughly uniform in the relevant log-scale. Slow enough that the tracked minimizer follows the moving global basin; fast enough that the run terminates in a handful of outer iterations. And `c̄` itself is set from the inlier noise: it's the largest residual I'd expect from a genuine inlier, e.g. a chi-squared quantile of the measurement-noise variance, so that "above `c̄`" reliably means "not an inlier."

One more thing the explicit weights buy me, beyond cheap inner steps. Because `z_i` is an honest variable now — not a slaved function of the residual — I can put *prior structure* on it. In reconstruction, outliers/discontinuities aren't placed independently: edges form contours, they hysterese, they don't cluster spuriously. With `z` explicit I can add interaction terms among neighboring `z`'s (reward connected edges, penalize isolated ones) and *still* run the same continuation, because the penalty `Φ` retains the control parameter. Plain IRLS, with `z` slaved to `x`, has nowhere to attach such a prior. The duality isn't only computational convenience; it re-exposes the line process I integrated out, now in analog `[0,1]` form, ready to be constrained.

Let me write it, mirroring a clean implementation: alternate weighted-least-squares variable update and closed-form weight update, with an outer continuation loop on `μ`. I'll do truncated-least-squares as the main routine (the cleanest outlier/inlier semantics) and Geman–McClure as the smooth-weight variant, and demonstrate on a line fit drowning in outliers, where plain least squares is hopeless.

```python
import numpy as np


def weighted_least_squares(A, y, w):
    # Variable update: argmin_x sum_i w_i (A_i x - y_i)^2, closed form.
    # This is the only place the model enters; for a nonlinear model it
    # would be a (globally solvable) weighted problem in its place.
    W = np.sqrt(w)[:, None]
    Aw, yw = A * W, y * np.sqrt(w)
    x, *_ = np.linalg.lstsq(Aw.T @ Aw, Aw.T @ yw, rcond=None)
    return x


def gnc_tls(A, y, barc2, factor=1.4, max_iter=1000, tol=1e-6):
    # Graduated non-convexity with the truncated-least-squares cost.
    # Continuation runs mu from ~0 (convex surrogate) upward toward the
    # true truncated quadratic. barc2 = c_bar^2 = inlier noise bound.
    N = A.shape[0]
    w = np.ones(N)                              # first fit trusts everyone
    x = weighted_least_squares(A, y, w)
    r2 = (A @ x - y) ** 2
    mu = barc2 / (2.0 * r2.max() - barc2)       # start where surrogate is convex
    prev = np.inf
    for _ in range(max_iter):
        x = weighted_least_squares(A, y, w)     # variable update (global)
        r2 = (A @ x - y) ** 2
        th1 = (mu + 1) / mu * barc2             # above -> certain outlier (w=0)
        th2 = mu / (mu + 1) * barc2             # below -> certain inlier  (w=1)
        # weight update = closed-form minimizer of  w r2 + Phi_{rho_mu}(w):
        w = np.where(r2 >= th1, 0.0,
             np.where(r2 <= th2, 1.0,
                      np.sqrt(barc2 * mu * (mu + 1) / r2) - mu))  # soft bridge
        cost = float(w @ r2)                    # weighted residual sum
        if abs(cost - prev) < tol or np.all((w < 1e-10) | (w > 1 - 1e-10)):
            break                               # converged / weights gone binary
        prev = cost
        mu *= factor                            # sharpen the non-convexity
    return x, w


def gnc_gm(A, y, barc2, factor=1.4, max_iter=1000, tol=1e-6):
    # Same skeleton, Geman-McClure cost: smooth weights, mu runs DOWN from
    # large (convex/quadratic) toward 1 (true Geman-McClure).
    N = A.shape[0]
    w = np.ones(N)
    x = weighted_least_squares(A, y, w)
    r2 = (A @ x - y) ** 2
    mu = 2.0 * r2.max() / barc2                 # start convex over the data
    prev = np.inf
    for _ in range(max_iter):
        x = weighted_least_squares(A, y, w)
        r2 = (A @ x - y) ** 2
        w = (mu * barc2 / (r2 + mu * barc2)) ** 2   # closed-form GM weight
        cost = float(w @ r2)
        if mu <= 1.0 or abs(cost - prev) < tol:
            break
        prev = cost
        mu /= factor                            # sharpen toward the true cost
    return x, w


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, frac = 100, 0.8                          # 80% of points are gross outliers
    a_gt, b_gt = 2.0, 1.0
    t = rng.uniform(-5, 5, N)
    A = np.column_stack([t, np.ones(N)])        # linear model y = a t + b
    y = a_gt * t + b_gt + rng.normal(0, 0.1, N)
    out = rng.choice(N, int(frac * N), replace=False)
    y[out] += rng.uniform(-50, 50, len(out))    # corrupt with large errors

    barc2 = (3 * 0.1) ** 2                       # inlier bound ~ (3 sigma)^2
    x_ls, *_ = np.linalg.lstsq(A, y, rcond=None)   # plain least squares: trapped
    x_tls, w = gnc_tls(A, y, barc2)
    x_gm, _ = gnc_gm(A, y, barc2)
    print("ground truth :", [a_gt, b_gt])
    print("plain LS     :", x_ls.round(3))      # dragged off by the outliers
    print("GNC-TLS      :", x_tls.round(3))      # recovers the line
    print("GNC-GM       :", x_gm.round(3))
```

To recap the causal chain: least squares is wrecked by outliers because its influence `ψ = 2x` grows without bound, so the cure is a redescending robust cost whose influence falls back to zero — quadratic near the origin, flat far out (the truncated quadratic, which is also the discontinuity-preserving smoothness term with the binary line process eliminated). That cost is non-convex, so descent gets trapped in a basin chosen by initialization; rather than search stochastically, I build a one-parameter family of surrogates from a convex member to the true cost, find the unique global minimum of the convex one with no init, and slowly graduate the non-convexity in, tracking the minimizer into the global basin of the true cost. To make each inner minimization cheap, I introduce explicit per-point trust weights `z ∈ [0,1]` with a penalty `Ψ` constructed as the conjugate of `φ(x²) = ρ(x)` — provided `φ` is concave with `φ'(0)=1, φ'(∞)=0` — which makes `min_z (x²z + Ψ(z)) = ρ(x)` exact and turns each step into weighted least squares (global, closed form) plus a closed-form weight that *is* the soft inlier/outlier decision; the surrogate's control parameter survives into `Ψ`, so the same continuation runs on the weights, hardening them from "trust everyone" to a clean inlier/outlier cut as the non-convexity is graduated in.
