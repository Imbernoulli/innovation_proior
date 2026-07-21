The strongest baseline cleared its bar and told me the one piece it left unused. Zero-init — CFG++'s
manifold-safe renoise with the first `K = 2` highest-noise steps made inert — came in at FID 22.76 /
23.31 / 25.49, under both priors on every variant: below standard CFG's 23.65 / 24.29 / 25.74 and
plain CFG++'s 23.99 / 24.89 / 25.88. The zero-init hypothesis held cleanly: keeping the clean `noise_uc`
renoise while deleting the under-committed, unreliable high-noise prefix beat the hard guided renoise
*and* the no-skip clean renoise, without paying standard CFG's off-manifold cost. Now lay out the
board, because the shape of the three trajectories is what points at the next move. SD v1.5 ran 23.99
-> 23.65 -> 22.76, a total drop of `1.235`; SD v2-base 24.89 -> 24.29 -> 23.31, `1.582`; SDXL 25.88
-> 25.74 -> 25.49, only `0.384`. Zero-init's gain over standard CFG is `0.89` / `0.98` / `0.25`.
Every way I slice it, SDXL improved between a third and a quarter as much as the other two, and each
individual step on SDXL was tiny — `-0.14` then `-0.25`.

SDXL is the signal. Its FID barely budges as I swap renoise rules and skip steps: three different
structural interventions moved it a total of `0.384` while the smaller models moved four times as
far. That near-invariance says the remaining error on SDXL is not in *when* I guide (the prefix
barely touched it) or in *which noise renoises* (the swap moved it `0.14`), but in *how the two
predictions are mixed* at the steps I do guide. Every lever I have pulled acts on timing or the
renoise; none has touched the interior of the mix, and SDXL is telling me that is where its untapped
room is.

That is precisely the lever I set aside when filling zero-init, which took the inert prefix but kept
the *plain* CFG++ mix `noise_uc + w(noise_c - noise_uc)`. The other half is a per-sample correction to
*how the unconditional prediction enters the mix*. Every fill so far subtracts a full unit of
`noise_uc` inside the difference `noise_c - noise_uc`, treating the coefficient on `noise_uc` as fixed
at 1 — but there is no reason the best baseline to subtract is exactly one times the raw unconditional
prediction. If `noise_uc` is, at this particular `(z_t, t)`, too large or partly mis-aligned with the
conditional one, subtracting a full unit pollutes the guidance direction with a component the model
did not intend. The fix is to choose, per sample and per step, *how much* of `noise_uc` to subtract.

A per-sample correction is the right *kind* of thing for SDXL's near-invariance. Re-tuning the global
scale `w` cannot fix a mis-scaling that varies image to image and step to step — one number applied
identically to every sample just trades one fixed compromise for another. Making `w` time-varying
still applies the same scale to every sample at a given `t`, correcting a per-*step* pattern while
leaving the per-*sample* geometry untouched. The only lever that adapts to the actual `(noise_c,
noise_uc)` geometry of *this* image at *this* step is a per-sample, per-step scalar computed from the
two predictions — a different object from `w` in exactly the dimension SDXL is complaining about.

Derive what "how much" should be. I want guidance to amplify the part of the conditional prediction
the unconditional one does not already explain — the conditional *residual* — instead of the raw
conditional vector against a possibly mis-scaled baseline. Introduce a scalar `s` on the
unconditional prediction: `v_s = s noise_uc + w (noise_c - s noise_uc) = w noise_c - (w-1) s
noise_uc`, the affine combination `(1-w) s noise_uc + w noise_c`; at `s = 1` this is exactly the mix
every prior fill used, so `s` is a genuine generalization. To pick `s` I would minimize `||v_s -
v^*||^2` against the true guided noise `v^*` — but `v^*` is invisible, I have only the two
predictions.

The way through is to bound the unavailable loss and see which part depends on `s`. Write `delta = w
- 1` so `v_s = v_c + delta(v_c - s v_uc)`; then `v_s - v^* = (v_c - v^*) + delta(v_c - s v_uc)`, and
for any positive `lambda`, Young's inequality gives

  `||v_s - v^*||^2 <= (1 + lambda) ||v_c - v^*||^2 + (1 + 1/lambda) delta^2 ||v_c - s v_uc||^2.`

The first term hides the truth `v^*` but does *not* depend on `s`; the only `s`-dependent part is a
positive constant times `||v_c - s v_uc||^2`. And `lambda` never has to be chosen — it enters only
through the positive factor multiplying the `s`-dependent term, and scaling by a positive constant
does not move an argmin, so `argmin_s` is the same for every `lambda > 0`. The free parameter drops
out, the tell that the projection is intrinsic to the problem and not an artifact of the bound. So
minimizing the bound replaces "match the invisible truth" with a solvable projection: differentiate
`||noise_c - s noise_uc||^2 = ||noise_c||^2 - 2 s noise_c^T noise_uc + s^2 ||noise_uc||^2`, set the
derivative to zero,

  `s* = (noise_c^T noise_uc) / ||noise_uc||^2.`

The second derivative is `2 ||noise_uc||^2 > 0`, a genuine minimizer whenever `noise_uc` is nonzero.
Geometrically `s* noise_uc` is the orthogonal projection of `noise_c` onto the line spanned by
`noise_uc`, and `noise_c - s* noise_uc` is the residual the unconditional prediction cannot explain —
exactly the direction I want guidance to push. So the optimized-scale mix is `s* noise_uc + w
(noise_c - s* noise_uc)`: rescale the unconditional baseline to its best least-squares match, then
amplify only the leftover residual. Per-sample, one dot product and one squared norm per step, no
extra network evaluation, and it reduces to the prior mix exactly when the residual vanishes.

A 2D toy makes concrete what the projection removes. Take `noise_uc = (1, 0)` and a strongly-aligned
`noise_c = (0.9, 0.3)`. The plain difference the prior fills amplify is `noise_c - noise_uc = (-0.1,
0.3)` — the `-0.1` component *along* `noise_uc` is a leftover of the mis-scaled baseline with nothing
to do with the conditional-specific direction, and guidance at `w = 7.5` blows it up to `-0.75`,
injecting `0.75` units of raw unconditional prediction (sign-flipped) for no semantic reason. The
optimized scale is `s* = 0.9/1 = 0.9`, and the residual `noise_c - s* noise_uc = (0, 0.3)` is purely
orthogonal to `noise_uc` — exactly the conditional-specific component. Amplifying `(0, 0.3)` pushes
only where the conditional prediction adds, never along the shared baseline.

In the implicit-classifier language of the standard-CFG fill, the guided direction was `noise_c -
noise_uc`, the score with a *unit* coefficient on the unconditional term; the optimized scale says
that unit coefficient was an unexamined default. `w` sets *how hard* to push along the residual, `s*`
sets *what the residual is* by fixing the baseline it is measured against — a sharper classifier,
because the difference now isolates only the conditional-specific component and not a mis-scaled chunk
of the shared image prior both predictions carry.

The limits are sensible. Collinear predictions `noise_c = gamma noise_uc` give `s* = gamma`, residual
`0`, and the mix collapses to `s* noise_uc = noise_c` — the zero-init baseline's behavior exactly (and
`s* = 1` in the equal case, recovering the prior unit coefficient). At the orthogonal extreme
`noise_c^T noise_uc = 0`, `s* = 0`, and guidance pushes along the full conditional vector with no
baseline subtracted, since an unconditional prediction sharing no direction with the conditional one
explains none of it.

How big is the correction in practice? Decompose `s* = (||noise_c||/||noise_uc||) cos(theta)`. Both
are noise predictions of the same latent under conditionings that share almost everything except the
prompt, so I expect them strongly aligned (`cos theta` well below 1 but far from 0) and of comparable
norm, putting `s*` somewhat below 1. That matters because the change to the guided direction relative
to the baseline is `w (1 - s*) noise_uc`: even a modest `1 - s* ~ 0.1` times `w = 7.5` is an order-`0.75
noise_uc` shift — an order-one change, not a rounding correction. So I expect the optimized scale to
do real, measurable work; whether it lowers FID, and whether `s*` really sits below 1, only the FID
and CLIP tables settle.

The finale is the full method the `zeroinit` variant half-implemented, three pieces each fixing a
distinct failure diagnosed on the way up: the `noise_uc` renoise removes off-manifold drift (CFG++),
the inert prefix drops the unreliable high-noise steps (zero-init), and the optimized scale corrects the
per-step mis-scaling of the unconditional baseline every prior fill fixed at 1. The first two are
already in the strongest baseline, so the edit is surgical: keep the inert prefix (`if step < K:
continue`, `K = 2`), keep the `noise_uc` renoise, and replace *only* the mix — `noise_pred =
noise_uc * s* + cfg_guidance (noise_c - noise_uc * s*)`. The three actions are orthogonal — the prefix
decides *whether* to guide, the `noise_uc` renoise decides *how to lift*, `s*` decides *what to
amplify* in the denoise — so the renoise line is untouched and the CFG++ manifold and invertibility
argument carries over verbatim; the optimized scale changes only which point `z0t` I denoise to, and
away from collinearity it denoises to a cleaner residual, if anything more faithful.

A few implementation cautions, all shape and dtype rather than math. The dot product and norm are
per sample over the flattened non-batch dimensions: for SD the latent is `(B, 4, 64, 64)`, flatten to
`(B, 16384)`, sum `noise_c * noise_uc` and `noise_uc^2` along dim 1 with `keepdim` for a `(B, 1)`
scalar; SDXL is `(B, 4, 128, 128)` flattened to `(B, 65536)`. Reshape to `(B, 1, 1, 1)` so it
broadcasts over channel and spatial axes — a single scalar per image, not a global scalar across the
batch (which would tie unrelated prompts together) nor a per-element one (a much higher-dimensional
object than the scalar the derivation produced). The denominator needs a `+1e-8` floor so a near-zero
unconditional prediction does not blow up `s*`, and the scale is cast back to the prediction's dtype
since the SD path runs under fp16 autocast. It is computed from the *detached* predictions inside the
existing `no_grad` block — a sampling-time correction, nothing trained, the `requires_grad_()` fossil
inert as at the CFG++ fill. SDXL is identical in `reverse_process()`: same guard, same per-sample
projection, same `noise_uc` renoise, indexing `alphas_cumprod[t]`. The full scaffold module for both
variants is in the answer.

The optimized scale adds only one dot product and one squared norm per step, no network call, and the
inert prefix *removes* two `predict_noise` calls per image — so the full three-piece method stays
strictly inside the two-NFE budget, in fact below it.

The bar is zero-init's 22.76 / 23.31 / 25.49. Since the optimized scale only changes the mix on
already-guided steps and leaves the renoise and prefix exactly as the strongest baseline has them, it
should never do worse than zero-init when the predictions are collinear (residual vanishes, mix
recovers the baseline) and should help wherever the raw unconditional baseline is mis-scaled. The
sharpest expectation is on SDXL, whose FID barely moved across the three prior fills (`0.384` total):
by elimination its residual error sits in the one place the timing and renoise levers never touched,
the mix, and `s*` is the only lever aimed there — so a drop below 25.49 would be the cleanest evidence
the projection is doing real work. I have no first-principles reason SDXL *should* carry the most
per-sample mis-scaling (its larger `65536`-dim latent gives more room to differ in norm and alignment,
but that is a hypothesis), only the empirical near-invariance pointing there. On SD v1.5 and SD
v2-base I expect smaller gains on the already-strong 22.76 / 23.31, largely fixed by the
renoise-and-prefix combination. The validation, seed and prompt set pinned: all three FID at or below
zero-init's, largest drop on SDXL, and CLIP not falling — since the optimized scale amplifies the
conditional *residual*, a CLIP drop would mean it traded alignment for distribution-matching.
