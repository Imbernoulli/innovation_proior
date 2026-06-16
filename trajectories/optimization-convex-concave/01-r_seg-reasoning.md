I am starting from the scaffold's default fill, which is already a real method: plain stochastic
extragradient. That default exists because the obvious thing — simultaneous descent-ascent
`z_{t+1} = z_t − τ F(z_t)` — does not even survive the easiest convex-concave instance. On `f=xy`
the operator is `F(z)=[y,−x]=Jz`, a 90° rotation, and the step is exactly orthogonal to the vector
pointing at the saddle, so `‖z_{t+1}‖² = ‖z_t‖² − 2τ⟨z_t,F(z_t)⟩ + τ²‖F(z_t)‖² = (1+τ²)‖z_t‖²`:
strictly outward for every step size, with no `τ` that rescues it because the cross term is
identically zero. The extragradient look-ahead fixes that — take a forward half-step to `w`,
evaluate the field *there*, and step from the original `z_t` — and on the same field that anchor
manufactures a `−τ²I` inward term, turning the modulus `√(1+τ²)>1` into `√(1−τ²(1−τ²))<1`. So the
default scaffold contracts the rotation. The question for the first rung of this ladder is not
whether extragradient works; it is what to add to extragradient when the thing I am graded on is the
*gradient norm under noise*, and that is where I want to think carefully before I commit, because the
cheapest-looking fix is exactly the one I am going to try first and it has a built-in cost I want to
have measured rather than guessed.

Let me write down what stochastic extragradient can actually promise, because the shape of its
guarantee tells me where to push. The oracle here is the exact operator `F(z)` plus a fixed-scale
additive Gaussian update perturbation, so each half-step is an exact evaluation followed by a noise
draw. Suppose for a moment the operator were `λ`-strongly monotone with unique zero `z*`. Running one
SEG step with `η<1/(4L)` and tracking `‖z_{t+1}−z*‖²`, the standard predictor-corrector bookkeeping
— substitute the update identities `ηF(z_{1/2}) = z_t−z_{t+1}` and `ηF(z_t)=z_t−z_{1/2}` so every
gradient becomes a difference of iterates, polarize, let the intermediate `‖z_t−z_{t+1}‖²` cancel
between the predictor and corrector identities, and absorb the Lipschitz cross term with `η≤1/(4L)`
— lands on a per-step inequality of the form

  `λ E‖z_{1/2}−z*‖² ≤ (1/η) E[ ‖z_t−z*‖² − ‖z_{t+1}−z*‖² ] + c·η σ²`.

Two things in that inequality decide everything. The distance contracts geometrically toward `z*`,
which is the good part; but every useful term carries `λ`, and the noise enters as a fixed floor
`c η σ²/λ` that does *not* vanish with more iterations. So SEG with a constant step marches to a ball
of radius governed by `η σ²/λ` and stops there — to shrink the ball I must shrink `η`, which cripples
the `1/(η T)` optimization term. That is the SEG wall, and it is sharp.

But there is a worse problem hiding in that inequality for *this* task: my operators are merely
convex-concave, so `λ=0`. Every term in the contraction inequality is multiplied by `λ`, so at `λ=0`
the inequality says nothing — there is no contraction at all, only the merely-monotone `O(1/k)`
best-iterate gap, and no last-iterate gradient-norm guarantee. The bilinear field is monotone but
with a strictly zero strong-monotonicity margin (pure rotation); the `(δ,ν)` field's clipped
component is monotone but flat, again no margin. So the honest situation entering this ladder is: the
default SEG converges, but slowly and to a noise-limited neighborhood, and the property that would
make it fast — strong monotonicity — is exactly the property I do not have.

So the first move I will make is the one that *manufactures* the missing strong monotonicity, because
it is suspiciously cheap and cheap is worth checking before anything elaborate. Take `F` and add `λ`
times a pull toward a fixed anchor point `a`:

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

That irreducible term is exactly where I expect this rung to hurt on *these* instances, and I want to
be precise about it because it is the falsifiable prediction. There is a genuine tension in `λ`:
large `λ` makes `G` strongly monotone, so the SEG contraction is fast and the noise floor
`η σ²/λ` small — but the bias `λ‖z0−z*‖` is large; small `λ` kills the bias but barely regularizes,
so the conditioning `L/λ` blows up and the noise floor explodes. The right `λ` balances bias against
conditioning, `λ ~ ε/D` with `D` a bound on `‖z0−z*‖`. The harness fixes the constants for me:
`τ=0.1, λ=0.1` on bilinear and `τ=1.0, λ=0.01` on `(δ,ν)`. The implementation is one extragradient
step on `G`: each half-step adds a fixed pull `τλ(z0−z)` toward the anchor and the oracle noise, the
predictor pulling from `z` and the corrector re-evaluating the operator at the look-ahead `w` while
still pulling toward `z0`. The anchor is `z0` and never moves — the transfer bound uses only the
single distance `‖z0−z*‖` and needs no extra anchor state. Two operator evaluations, two noise
draws, the fixed `z0` anchor; the full scaffold module is in the answer.

Now let me reason about what this will actually score on these two instances, because the irreducible
bias is large precisely where the start is far from the solution — and on `bilinear`, `z0=[10,10]ᵀ`
is *very* far from the saddle at the origin. Take the bilinear case concretely. The true solution is
`z*=0`, so `‖z0−z*‖ = ‖[10,10]‖ = √200 ≈ 14.14`. With `λ=0.1`, the irreducible residual floor is
`λ‖z0−z*‖ ≈ 0.1 × 14.14 ≈ 1.41`. That is enormous — it means the *best* bilinear gradient norm this
rung can possibly reach is about `1.4`, no matter how many of the 900 iterations I spend, because the
anchored solution `w*` itself sits a distance `‖w*‖ ≈ ‖F(w*)‖/λ` from the origin and `‖F(z)‖=‖z‖` on
this field. The fixed pull toward `z0=[10,10]` is literally dragging the iterate back toward the
worst possible point. So I should expect the bilinear final gradient norm to be pinned near `1.4`,
catastrophically worse than what plain SEG (no anchor bias) would give. The `(δ,ν)` case is the
opposite extreme: there `z0 ~ N(0,I)` is already near the origin-ish solution and `λ=0.01` is tiny,
so the bias `λ‖z0−z*‖` is small and the mild regularization mostly just adds a little stability —
I expect a small `delta_nu` gradient norm there, maybe an order or two below the bilinear value.

So the falsifiable expectation for this first rung, against which the next rung will be judged: the
`bilinear` final gradient norm should be dominated by the anchor bias and land around `1.4`, while
the `delta_nu` norm should be small (well under `0.1`). The mean `final_gradient_norm` is the average
of the two, so it will be dragged up almost entirely by the bilinear half — I expect a mean near
`0.7–0.8`, the worst of any rung on this ladder. The diagnosis is structural and I want it in the
record: the fixed-`z0` Tikhonov anchor buys strong monotonicity and noise robustness but pays an
irreducible `λ‖z0−z*‖` bias, and on the bilinear instance the start is so far from the solution that
the bias *is* the score. The next rung's job is to keep extragradient's anti-rotation contraction
without paying that anchor bias at all — i.e. to drop the `λ(z−z0)` pull and see how far the bare
look-ahead step goes when it is not being dragged toward `[10,10]`.
