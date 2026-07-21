I am starting from the scaffold's default fill, which is already a real method: plain stochastic
extragradient. That default exists because the obvious thing — simultaneous descent-ascent
`z_{t+1} = z_t − τ F(z_t)` — does not even survive the easiest convex-concave instance. On `f=xy`
the operator is `F(z)=[y,−x]=Jz`, a 90° rotation, and the step is exactly orthogonal to the vector
pointing at the saddle, so `‖z_{t+1}‖² = ‖z_t‖² − 2τ⟨z_t,F(z_t)⟩ + τ²‖F(z_t)‖² = (1+τ²)‖z_t‖²`:
strictly outward for every step size, with no `τ` that rescues it because the cross term
`⟨z_t,Jz_t⟩=z_tᵀJz_t` is identically zero (`J` is skew). The extragradient look-ahead fixes that —
take a forward half-step to `w`, evaluate the field *there*, then step from the original `z_t` — and
on the same field that anchor manufactures a `−τ²I` inward term, turning the modulus `√(1+τ²)>1` into
`√(1−τ²(1−τ²))<1`. So the default step contracts the rotation. The question is not whether
extragradient works; it is what to add to it when the graded quantity is the *gradient norm under
noise*. The cheapest-looking fix is the one I will try first, and it has a built-in cost I want
measured rather than guessed.

Making the two competing forces concrete on this field sets the scale of everything that follows.
Simultaneous descent-ascent multiplies `‖z‖²` by `(1+τ²)`
every step, so over the 900 bilinear iterations at `τ=0.1` it inflates the norm by
`(1.01)^{450} = e^{450·ln 1.01} = e^{4.478} ≈ 88`, taking `‖z0‖ = √200 ≈ 14.14` out to about `1245` —
a clean, catastrophic blow-up that is why the scaffold cannot be the naive step. The extragradient
look-ahead, by contrast, has per-step modulus `√(1−τ²(1−τ²))`; at `τ=0.1` that is
`√(1−0.01·0.99) = √0.9901 = 0.99504 < 1`. So on this particular field the bare look-ahead already
contracts. The trouble is that this contraction is a *special property of the pure rotation* — the
`−τ²I` term exists because `J²=−I`, a coincidence of the bilinear operator — and it is not something I
can rely on when the field is merely monotone and *flat*, like the clipped `(δ,ν)` component, where
there is no analogous curvature for the look-ahead to grab. What I want is a contraction I can trust
on both instances, including the flat one — a stronger requirement than "extragradient happens to
spiral inward on `xy`."

The shape of stochastic extragradient's guarantee tells me where to push. The oracle here is the
exact operator `F(z)` plus a fixed-scale additive Gaussian update perturbation, so each half-step is
an exact evaluation followed by a noise draw. Suppose for a moment the operator were `λ`-strongly
monotone with unique zero `z*`. Running one SEG step with `η<1/(4L)` and tracking `‖z_{t+1}−z*‖²`:
write the two half-steps as `z_{1/2}=z_t−ηF(z_t)` (predictor) and `z_{t+1}=z_t−ηF(z_{1/2})`
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
component is monotone but flat, again no margin. So the honest situation here is: the
default SEG converges, but slowly and to a noise-limited neighborhood, and the property that would
make it fast — strong monotonicity — is exactly the property I do not have.

Before I reach for anything structural, I want to rule out the cheaper knobs. There are three within
reach. The first is to keep bare SEG and simply tune the step size to sit in a smaller neighborhood.
But at `λ=0` the floor `cησ²/λ` reads `cησ²/0 = ∞`: there is no ball to tune down, only diffusion of
the iterate around a flat merely-monotone level set with nothing pulling it back, so shrinking `η`
buys nothing in the last-iterate norm and costs me the `1/(ηT)` progress. The second is the
implicit/proximal step `z_{t+1}=(I+τF)^{-1}(z_t)`, which is firmly nonexpansive for monotone `F` and contracts
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

The catch is the one I have to face squarely, because it is the entire cost of this move:
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
is the restarted/regularized SEG method, "R-SEG." I should check the anchored solution does not run off
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

On `bilinear` the bias is the whole story, and I can compute it exactly rather than lean on the
transfer-bound upper estimate. On `f=xy`, `G(z)=Jz+λ(z−z0)`
with `J=[[0,1],[−1,0]]`, and its zero solves `(J+λI)w* = λ z0`, i.e. `w* = λ(J+λI)^{-1} z0`. The
matrix `J+λI = [[λ,1],[−1,λ]]` has determinant `λ²+1` and inverse `(1/(λ²+1))[[λ,−1],[1,λ]]`. With
`z0=[10,10]ᵀ`, `[[λ,−1],[1,λ]][10,10]ᵀ = [10λ−10, 10+10λ]ᵀ = 10[λ−1, λ+1]ᵀ`, so
`w* = (10λ/(λ²+1))·[λ−1, λ+1]ᵀ`. On the rotation field `‖F(z)‖=‖Jz‖=‖z‖`, so the residual at the exact
anchored solution is `‖F(w*)‖ = ‖w*‖ = (10λ/(λ²+1))√((λ−1)²+(λ+1)²) = (10λ/(λ²+1))√(2λ²+2)
= 10√2·λ/√(λ²+1)`. At `λ=0.1` that is `10·1.41421·0.1/√1.01 = 1.41421/1.00499 = 1.40720`. So the best
`bilinear` gradient norm this method can possibly reach, no matter how many of the 900 iterations I
spend, is about `1.407` — the anchored solution itself sits that far from the origin, and the pull
toward `z0=[10,10]` is what parks it there. The small-`λ` linearization `10√2·λ = λ‖z0‖ = λ‖z0−z*‖`
recovers the transfer-bound term exactly; at `λ=0.1` the bound gives `1.41421` against the exact
`1.40720`.

Could I not shrink `λ` to cut this? The bias `10√2·λ` falls linearly, so `λ=0.01` would drop the
bilinear residual to `≈0.14`. But the other side of the transfer bound, `2‖G(z̃)‖`, is only crushable
at a rate set by the conditioning `κ=(L+λ)/λ`: the anchored rotation contracts with a margin scaling
like `λ`, so driving `‖G‖` to a target costs `~(1/λ)·log(1/target)` steps. At `λ=0.1` that fits inside
900 iterations; at `λ=0.01` the solve is ten times slower, does not finish, and the transfer bound is
then dominated by the unconverged `2‖G‖` term rather than the smaller bias I was chasing — while the
noise floor `ησ²/λ` explodes by the same factor. So there is a genuine interior optimum, `λ ~ ε/D`
with `D ~ ‖z0−z*‖`; on bilinear it lands `λ` near `0.1` and the `≈1.4` bias is the structural cost of
anchoring far from the solution, not a tuning mistake.

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

So the expectations. The `bilinear` final gradient norm should be pinned by the anchor bias at about
`1.4` — the exact `w*` residual `1.407` plus a hair of finite-iteration and `σ=0.001` slack — while
`delta_nu` should be small, around `0.1` or below. The mean is the average of the two, dragged up
almost entirely by the bilinear half, so I expect near `0.7–0.8`. A distinguishing signature: since
the bilinear score is deterministic anchor bias, not variance, turning the noise up should barely move
it — the high-noise `bilinear` column within a percent or two of default. If it swings with the noise
level instead, the diagnosis is wrong. The lever to pull next is to keep extragradient's own
anti-rotation contraction — the `−τ²I` term that needs no external anchor — while removing the
`λ(z−z0)` pull dragging the iterate toward `[10,10]`, and see how far the bare look-ahead step gets.
