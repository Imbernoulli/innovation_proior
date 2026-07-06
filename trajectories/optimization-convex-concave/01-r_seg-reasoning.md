I am starting from the scaffold's default fill, which is already a real method: plain stochastic
extragradient. That default exists because the obvious thing — simultaneous descent-ascent
`z_{t+1} = z_t − τ F(z_t)` — does not even survive the easiest convex-concave instance. On `f=xy`
the operator is `F(z)=[y,−x]=Jz`, a 90° rotation, and the step is exactly orthogonal to the vector
pointing at the saddle, so `‖z_{t+1}‖² = ‖z_t‖² − 2τ⟨z_t,F(z_t)⟩ + τ²‖F(z_t)‖² = (1+τ²)‖z_t‖²`:
strictly outward for every step size, with no `τ` that rescues it because the cross term
`⟨z_t,Jz_t⟩=z_tᵀJz_t` is identically zero (`J` is skew). The extragradient look-ahead fixes that —
take a forward half-step to `w`, evaluate the field *there*, then step from the original `z_t` — and
on the same field that anchor manufactures a `−τ²I` inward term, turning the modulus `√(1+τ²)>1` into
`√(1−τ²(1−τ²))<1`. So the default scaffold contracts the rotation. The question for the first rung of
this ladder is not whether extragradient works; it is what to add to extragradient when the thing I am
graded on is the *gradient norm under noise*, and that is where I want to think carefully before I
commit, because the cheapest-looking fix is exactly the one I am going to try first and it has a
built-in cost I want to have measured rather than guessed.

Let me make the two competing forces concrete on this field before I generalize, because the numbers
set the scale of everything that follows. Simultaneous descent-ascent multiplies `‖z‖²` by `(1+τ²)`
every step, so over the 900 bilinear iterations at `τ=0.1` it inflates the norm by
`(1.01)^{450} = e^{450·ln 1.01} = e^{4.478} ≈ 88`, taking `‖z0‖ = √200 ≈ 14.14` out to about `1245` —
a clean, catastrophic blow-up that is why the scaffold cannot be the naive step. The extragradient
look-ahead, by contrast, has per-step modulus `√(1−τ²(1−τ²))`; at `τ=0.1` that is
`√(1−0.01·0.99) = √0.9901 = 0.99504 < 1`. So on this particular field the bare look-ahead already
contracts. The trouble is that this contraction is a *special property of the pure rotation* — the
`−τ²I` term exists because `J²=−I`, a coincidence of the bilinear operator — and it is not something I
can rely on when the field is merely monotone and *flat*, like the clipped `(δ,ν)` component, where
there is no analogous curvature for the look-ahead to grab. What I want going into this ladder is a
contraction I can trust on both instances, including the flat one, and that is a stronger requirement
than "extragradient happens to spiral inward on `xy`."

Let me write down what stochastic extragradient can actually promise, because the shape of its
guarantee tells me where to push. The oracle here is the exact operator `F(z)` plus a fixed-scale
additive Gaussian update perturbation, so each half-step is an exact evaluation followed by a noise
draw. Suppose for a moment the operator were `λ`-strongly monotone with unique zero `z*`. Running one
SEG step with `η<1/(4L)` and tracking `‖z_{t+1}−z*‖²`, I can grind the bookkeeping out rather than
quote it. Write the two half-steps as `z_{1/2}=z_t−ηF(z_t)` (predictor) and `z_{t+1}=z_t−ηF(z_{1/2})`
(corrector), so the update identities are `ηF(z_t)=z_t−z_{1/2}` and `ηF(z_{1/2})=z_t−z_{t+1}`. Expand
`‖z_{t+1}−z*‖² = ‖z_t−z*‖² − 2⟨z_t−z_{t+1}, z_t−z*⟩ + ‖z_t−z_{t+1}‖²`, and substitute
`z_t−z_{t+1}=ηF(z_{1/2})` into the inner product; then the standard trick is to split
`z_t−z* = (z_{1/2}−z*) + (z_t−z_{1/2})` so the progress term becomes
`−2η⟨F(z_{1/2}), z_{1/2}−z*⟩ − 2η⟨F(z_{1/2}), z_t−z_{1/2}⟩`. The first of these is `≤ −2ηλ‖z_{1/2}−z*‖²`
by strong monotonicity with `F(z*)=0`; the second recombines with the leftover `‖z_t−z_{t+1}‖²` and
the predictor identity, and the Lipschitz bound `‖F(z_{1/2})−F(z_t)‖ ≤ L‖z_{1/2}−z_t‖` lets the whole
remainder be absorbed as long as `η ≤ 1/(4L)`, because the discretization error carries a factor
`(η²L²−1)` that is negative there. Taking expectations over the two additive noise draws — each an
independent `σ`-scale perturbation entering `z_{1/2}` and `z_{t+1}` — contributes a `+cη²σ²` term from
the squared-norm expansions. The result is a per-step inequality of the form

  `λ E‖z_{1/2}−z*‖² ≤ (1/η) E[ ‖z_t−z*‖² − ‖z_{t+1}−z*‖² ] + c·η σ²`.

Two things in that inequality decide everything. The distance contracts geometrically toward `z*`,
which is the good part; but every useful term carries `λ`, and the noise enters as a fixed floor that
does *not* vanish with more iterations. To see the floor, drop the nonnegative left side and read the
inequality as `E‖z_{t+1}−z*‖² ≤ (1−ηλ)E‖z_t−z*‖² + cη²σ²` — the strong-monotonicity margin turns the
telescoping form into an honest geometric contraction with rate `(1−ηλ)`. Unrolling the recursion,
the fixed point of `r ↦ (1−ηλ)r + cη²σ²` is `r_∞ = cη²σ²/(ηλ) = cησ²/λ`. So the squared distance to
`z*` marches down geometrically until it hits `‖z_t−z*‖² ≈ cησ²/λ` and stops there: a ball whose
radius grows with `η` and `σ` and *shrinks with `λ`*. To make the ball smaller I must shrink `η`, but
`η` also sets the contraction rate `(1−ηλ)` and hence the `1/(ηT)` speed at which I reach the ball —
shrinking it cripples the optimization term. That is the SEG wall, and it is sharp: for a fixed
budget `T` the best constant step trades ball-radius against reach-time, and neither can be driven to
zero. The one free lever that shrinks the ball *without* touching the reach-time is `λ` itself — a
larger strong-monotonicity margin — and that is exactly the quantity my problem does not have.

But there is a worse problem hiding in that inequality for *this* task: my operators are merely
convex-concave, so `λ=0`. Every term in the contraction inequality is multiplied by `λ`, so at `λ=0`
the inequality says nothing — there is no contraction at all, only the merely-monotone `O(1/k)`
best-iterate gap, and no last-iterate gradient-norm guarantee. The bilinear field is monotone but
with a strictly zero strong-monotonicity margin (pure rotation); the `(δ,ν)` field's clipped
component is monotone but flat, again no margin. So the honest situation entering this ladder is: the
default SEG converges, but slowly and to a noise-limited neighborhood, and the property that would
make it fast — strong monotonicity — is exactly the property I do not have.

Before I reach for anything structural, I want to rule out the cheaper knobs, because if one of them
worked I would rather turn a dial than change the operator. There are three within reach. The first is
to keep bare SEG and simply tune the step size to sit in a smaller neighborhood. But at `λ=0` there is
no radius to tune down: the floor `cησ²/λ` is a statement about a *contracting* method, and with no
contraction the noise merely diffuses the iterate around a flat merely-monotone level set — shrinking
`η` shrinks the diffusion per step but there is nothing pulling the iterate back, so it buys nothing
in the last-iterate norm and costs me the `1/(ηT)` progress. This is not a subtlety I can wave off, because the ball-radius formula `cησ²/λ` I just derived
degenerates precisely here: at `λ=0` it reads `cησ²/0 = ∞`, meaning there is no finite neighborhood
the iterate is pulled into, so "shrink `η` to shrink the ball" is not even a well-posed move — there
is no ball. The second is the implicit/proximal step
`z_{t+1}=(I+τF)^{-1}(z_t)`, which is firmly nonexpansive for monotone `F` and contracts
unconditionally — the geometric ideal I would love to have. But it needs `F` evaluated at the unknown
next point, a nonlinear solve buried inside every iteration, and on a generic field that is not
something the fixed single-loop step lets me run. The third is to note that both the divergence I
started with and the flatness I am now stuck at have the same root — the operator has no restoring
force toward its zero — and to simply *install* that restoring force. None of the cheap dials
manufacture the missing contraction; the only lever that does is to put strong monotonicity into the
operator myself, and that is suspiciously cheap, so cheap is worth checking before anything elaborate.

Take `F` and add `λ` times a pull toward a fixed anchor point `a`:

  `G(z) = F(z) + λ(z − a)`.

For any `z,z'`, `⟨G(z)−G(z'), z−z'⟩ = ⟨F(z)−F(z'), z−z'⟩ + λ‖z−z'‖² ≥ λ‖z−z'‖²`, so `G` is
`λ`-strongly monotone *by construction*, regardless of `F` being merely monotone, and it is still
`(L+λ)`-Lipschitz, essentially as smooth as `F`. `G` is the gradient operator of the Tikhonov-
regularized saddle objective `f(x,y) + (λ/2)‖x−a_x‖² − (λ/2)‖y−a_y‖²` — a strongly-convex penalty on
`x`, a strongly-concave one on `y`. If I run the extragradient step I already trust on `G` instead of
`F`, I get the SEG contraction the bare problem could never give me: the iterate contracts toward
`G`'s zero, and the noise floor is `η σ²/λ` with a genuine `λ>0`.

The catch is the one I have to face squarely, because it is the entire cost of this first rung:
running on `G` drives me toward `G`'s zero `w*`, which is *not* `z*`, the zero of `F`. I have solved a
different problem, and the metric is `‖F‖`, not `‖G‖`. So the question that decides whether the trick
is legitimate is how far `w*` is from being a near-stationary point of the original `F`. From the
definition, `F(w*) = G(w*) − λ(w*−a) = −λ(w*−a)`, so even the *exact* solution of the regularized
problem has residual `‖F(w*)‖ = λ‖w*−a‖`. That is an irreducible bias: no matter how perfectly I
solve `G`, the gradient norm I report cannot fall below `λ` times the distance from the anchor to the
regularized solution. I want a clean transfer bound. For any candidate `z̃`,

  `‖F(z̃)‖ ≤ ‖G(z̃)‖ + λ‖z̃−a‖ ≤ ‖G(z̃)‖ + λ‖z̃−w*‖ + λ‖w*−a‖`,

and strong monotonicity of `G` (with `G(w*)=0`) gives `λ‖z̃−w*‖ ≤ ‖G(z̃)‖`, so
`‖F(z̃)‖ ≤ 2‖G(z̃)‖ + λ‖w*−a‖`. The first term I can crush by running SEG on `G`; the second,
`λ‖w*−a‖`, is the price of regularizing.

Where to put the anchor `a`? The only special point I have at the start is `z0` itself; I cannot
anchor at `z*` because I do not know it. So anchor at the initial point, `a=z0`, fixed forever — this
is the restarted/regularized SEG rung, "R-SEG." I should check the anchored solution does not run off
to infinity. With `a=z0`, strong monotonicity of `G` between `w*` and `z*` plus `F(z*)=0` gives
`λ‖w*−z*‖² ≤ G(z*)ᵀ(z*−w*) = λ(z*−z0)ᵀ(z*−w*)`; dividing by `λ` and polarizing yields *both*
`‖w*−z0‖ ≤ ‖z*−z0‖` and `‖w*−z*‖ ≤ ‖z*−z0‖`. The regularized solution is no farther from the anchor
than the true solution is — anchoring at `z0` is geometrically safe — and the transfer bound
collapses to `‖F(z̃)‖ ≤ 2‖G(z̃)‖ + λ‖z0−z*‖`. The lever is now explicit: drive `G`'s gradient norm
to zero and the only thing left is the irreducible `λ‖z0−z*‖`.

There is a genuine tension in `λ` sitting inside that transfer bound, and it is what fixes the
constant. Large `λ` makes `G` strongly monotone, so the SEG contraction is fast and the noise floor
`η σ²/λ` small — but the bias `λ‖z0−z*‖` is large; small `λ` kills the bias but barely regularizes,
so the conditioning `L/λ` blows up and the noise floor explodes. The right `λ` balances bias against
conditioning, `λ ~ ε/D` with `D` a bound on `‖z0−z*‖`. The harness fixes the constants for me:
`τ=0.1, λ=0.1` on bilinear and `τ=1.0, λ=0.01` on `(δ,ν)`. The implementation is one extragradient
step on `G`: each half-step adds a fixed pull `τλ(z0−z)` toward the anchor and the oracle noise, the
predictor pulling from `z` and the corrector re-evaluating the operator at the look-ahead `w` while
still pulling toward `z0`. The anchor is `z0` and never moves — the transfer bound uses only the
single distance `‖z0−z*‖` and needs no extra anchor state. Two operator evaluations, two noise draws,
the fixed `z0` anchor; the full scaffold module is in the answer.

I do not want to leave the bias as a symbol, because on `bilinear` it is the whole story and I can
compute it exactly rather than lean on the transfer-bound upper estimate. On `f=xy`, `G(z)=Jz+λ(z−z0)`
with `J=[[0,1],[−1,0]]`, and its zero solves `(J+λI)w* = λ z0`, i.e. `w* = λ(J+λI)^{-1} z0`. The
matrix `J+λI = [[λ,1],[−1,λ]]` has determinant `λ²+1` and inverse `(1/(λ²+1))[[λ,−1],[1,λ]]`. With
`z0=[10,10]ᵀ`, `[[λ,−1],[1,λ]][10,10]ᵀ = [10λ−10, 10+10λ]ᵀ = 10[λ−1, λ+1]ᵀ`, so
`w* = (10λ/(λ²+1))·[λ−1, λ+1]ᵀ`. On the rotation field `‖F(z)‖=‖Jz‖=‖z‖`, so the residual at the exact
anchored solution is `‖F(w*)‖ = ‖w*‖ = (10λ/(λ²+1))√((λ−1)²+(λ+1)²) = (10λ/(λ²+1))√(2λ²+2)
= 10√2·λ/√(λ²+1)`. At `λ=0.1` that is `10·1.41421·0.1/√1.01 = 1.41421/1.00499 = 1.40720`. So the best
`bilinear` gradient norm this rung can possibly reach, no matter how many of the 900 iterations I
spend, is about `1.407` — the anchored solution itself sits that far from the origin, and the pull
toward `z0=[10,10]` is what parks it there.

This exact `w*` also lets me check the non-expansiveness geometry I leaned on, and on the rotation it
turns out to be exactly tight, which is reassuring. The inequality `‖w*−z*‖² ≤ (z*−z0)ᵀ(z*−w*)`
rearranges to `⟨w*−z*, w*−z0⟩ ≤ 0` — the vectors from `w*` back to the true solution and back to the
anchor make an obtuse angle, so by Thales' theorem `w*` sits inside the sphere whose diameter is the
segment `[z0, z*]`, hence within `‖z0−z*‖` of *both* endpoints. Plugging the numbers in with `z*=0`,
`⟨w*, w*−z0⟩ = (−0.891)(−10.891) + (1.089)(−8.911) = 9.704 − 9.704 = 0` exactly: on the pure rotation
`w*` lands *on* the sphere, at the right-angle vertex, because `F` carries zero strong-monotonicity
margin so `G`'s excess is purely the `λ` I injected. The right angle means Pythagoras holds,
`‖w*−z0‖² + ‖w*−z*‖² = ‖z0−z*‖²`, and indeed `14.071² + 1.407² = 197.99 + 1.98 = 199.97 ≈ 200 =
‖z0−z*‖²`. So the anchored solution is exactly as far from `z0` as it needs to be and no farther, and
the bias `λ‖w*−z0‖ = 0.1·14.071 = 1.407` is the leg of that right triangle — a fully consistent
picture, not a hand-wave.

Two limits confirm the formula is doing the right thing and expose the lever cleanly. As `λ→0`,
`w* → 0 = z*`: no regularization, the anchored solution collapses onto the true saddle and the
residual vanishes. As `λ→∞`, `10λ/(λ²+1) → 10/λ` and `[λ−1,λ+1] → [λ,λ]`, so `w* → [10,10] = z0` and
the residual `→ ‖z0‖ = √200 ≈ 14.14`: infinite regularization pins the iterate at the anchor, the
worst point on the field. So `10√2·λ/√(λ²+1)` climbs monotonically from `0` to `‖z0‖` as `λ` runs
`0→∞`, which is the bias-versus-`λ` lever made fully explicit. And for small `λ` it linearizes to
`10√2·λ = λ‖z0‖ = λ‖z0−z*‖`, exactly the transfer-bound term; at `λ=0.1` the bound gives `1.41421`
against the exact `1.40720`, so the bound is only `0.5%` loose — tight to first order. The value
`λ=0.1` the harness chose sits right where the bias is `≈1.4`, and the harness's 900 iterations cannot
touch it.

It is worth asking why the harness picked `λ=0.1` on bilinear rather than something much smaller,
since the bias `10√2·λ/√(λ²+1)` falls linearly as I shrink `λ` and I might be tempted to drive it to,
say, `λ=0.01` and cut the bilinear residual by a factor of ten to `≈0.14`. The reason I cannot just do
that is the other side of the transfer bound, `2‖G(z̃)‖`, which I have been treating as crushable but
is only crushable at a *rate* set by the conditioning `κ = (L+λ)/λ`. The anchored operator on the
rotation contracts with a modulus whose distance below one scales like `λ` (the strong-monotonicity
margin the anchor installs), so the number of steps to drive `‖G‖` down to a target scales like
`κ·log(1/target) ~ (1/λ)·log(1/target)`. At `λ=0.1` that budget is comfortably inside 900 iterations;
at `λ=0.01` the contraction is ten times slower and the anchored solve would not finish inside the
fixed budget, so `‖G‖` would still be sizeable and the transfer bound would be dominated by the
unconverged `2‖G‖` term rather than by the smaller bias I was chasing. So there is a genuine interior
optimum: too-large `λ` is all bias, too-small `λ` is all unconverged optimization error and an exploded
noise floor `ησ²/λ`, and `λ ~ ε/D` with `D ~ ‖z0−z*‖` sits between them. On bilinear that lands
`λ` near `0.1`, and the price is a bias I have now computed exactly to be `≈1.4`. I note this so the
record is honest: `λ=0.1` is not a bad choice given the fixed budget — it is the balanced one — and
the `1.4` is the *structural* cost of anchoring far from the solution, not a tuning mistake I could dial
away without breaking the convergence of `G`.

The `(δ,ν)` case is the opposite extreme, and I want a rough figure there too even though I cannot
compute it exactly. There `z0 ~ N(0,I)` in the `2d=200`-dimensional joint space, so `‖z0‖ ≈ √200 ≈
14.1` again, but `λ=0.01` is tiny, so even the worst-case bias `λ‖z0−z*‖ ≤ 0.01·14 ≈ 0.14`, and the
true residual `λ‖w*−z0‖` is smaller still because `w*` lies between `z0` and `z*`. On the noise side,
with `τ=1` and `σ=0.02` the SEG-on-`G` neighborhood radius scales like `√(τσ²/λ) = √(0.0004/0.01)
= √0.04 = 0.2`, and `‖F‖` near `w*` is at most about `(L+λ)` times that radius plus the bias. So I
expect `delta_nu` somewhere around `0.1`, an order of magnitude below the bilinear `1.4`, dominated by
neither the bias nor the noise catastrophically — mostly the mild regularization stabilizing a start
that is already near the solution. The exact constant on the noise radius I cannot pin down on paper;
the `delta_nu` column will tell me.

So the falsifiable expectation for this first rung, against which the next rung will be judged. The
`bilinear` final gradient norm should be pinned by the anchor bias at about `1.4` — the exact `w*`
residual `1.407` plus a hair of finite-iteration and `σ=0.001` noise slack — catastrophically worse
than what a no-anchor step would give, because the fixed pull toward `[10,10]ᵀ` is literally dragging
the iterate back toward the worst possible point and holding it there. The `delta_nu` norm should be
small, around `0.1` or below. The mean `final_gradient_norm` is the average of the two, so it will be
dragged up almost entirely by the bilinear half — I expect a mean near `0.7–0.8`, the worst of any
rung on this ladder. There is a sharper second signature worth stating, because it distinguishes a
bias-limited method from a variance-limited one: since the bilinear score is deterministic anchor
bias, not variance, turning the noise up should barely move it — the high-noise `bilinear` column
should sit within a percent or two of the default. If instead the bilinear number swings with the
noise level, my whole diagnosis is wrong. The diagnosis I want in the record is structural: the
fixed-`z0` Tikhonov anchor buys strong monotonicity and noise robustness but pays an irreducible
`λ‖z0−z*‖` bias, and on the bilinear instance the start is so far from the solution that the bias *is*
the score. The lever to pull next is to keep extragradient's own anti-rotation contraction — the
`−τ²I` term that needs no external anchor at all — while removing the `λ(z−z0)` pull that is dragging
the iterate toward `[10,10]`, and to see how far the bare, unbiased look-ahead step gets on its own.
