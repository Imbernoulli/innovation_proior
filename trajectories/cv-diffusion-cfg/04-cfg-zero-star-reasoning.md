The strongest baseline cleared the bar I set for it, and in doing so it told me exactly the one piece it
left unused. Zero-init — CFG++'s manifold-safe renoise with the first `K = 2` highest-noise steps made
inert — came in at FID 22.76 / 23.31 / 25.49 on SD v1.5 / SD v2-base / SDXL, under both priors on every
variant: below standard CFG's 23.65 / 24.29 / 25.74 and below plain CFG++'s 23.99 / 24.89 / 25.88. So the
rung-3 hypothesis held cleanly: keeping CFG++'s clean `noise_uc` renoise while deleting its
under-committed, unreliable high-noise prefix beat the hard guided renoise *and* the no-skip clean
renoise, without paying standard CFG's off-manifold cost. Now let me lay out the whole board and read it,
because the shape of the three trajectories, not any single number, is what points at the next move. On
SD v1.5 the sequence is 23.99 -> 23.65 -> 22.76, a total drop of `1.235` from the CFG++ floor. On SD
v2-base it is 24.89 -> 24.29 -> 23.31, a total of `1.582`. On SDXL it is 25.88 -> 25.74 -> 25.49, a total
of `0.384`. Read against the best prior at each rung the story is the same: zero-init's gain over standard
CFG is `0.89` on SD v1.5, `0.98` on SD v2-base, and `0.25` on SDXL. Every way I slice it, SDXL improved
between a third and a quarter as much as the other two variants, and each individual step on SDXL was
tiny — `-0.14` then `-0.25`.

SDXL is the signal. It is a high-resolution model whose FID barely budges as I swap renoise rules and
skip steps: three different structural interventions — CFG++'s clean renoise, standard CFG's guided
renoise, and now the inert prefix — and its FID moved a total of `0.384` while the smaller models moved
four times as far. That near-invariance says the remaining error on SDXL is not in *when* I guide (the
prefix barely touched it) or in *which noise renoises* (swapping `eps_uc` for `eps_g` moved it `0.14`),
but in *how the two predictions are mixed* at the steps I do guide. Every lever I have pulled so far acts
on the timing or the renoise; none has touched the interior of the mix itself, and SDXL is telling me
that is where its untapped room is.

And that is precisely the lever I flagged and set aside at rung 3. The zero-init baseline only implemented
half of the method it is named after: it took the inert prefix but kept the *plain* CFG++ mix, `noise_pred
= noise_uc + cfg_guidance (noise_c - noise_uc)`. The other half — the half the harness's fill omits — is a
per-sample correction to *how the unconditional prediction enters the mix*. That is the natural next move,
and I can motivate it from a pattern that runs through every rung so far: all of them have treated the
coefficient on `noise_uc` as fixed at 1. CFG++, standard CFG, and zero-init all subtract a full unit of
`noise_uc` inside the difference `noise_c - noise_uc`. But there is no reason the best baseline to subtract
is exactly one times the raw unconditional prediction. If the unconditional prediction is, at this
particular `(z_t, t)`, too large or too small or partly mis-aligned with the conditional one, subtracting a
full unit of it pollutes the guidance direction with a component the model did not intend. The fix is to
choose, per sample and per step, *how much* of `noise_uc` to subtract.

Before I derive the right amount, let me be sure a per-sample correction is the *kind* of thing SDXL's
near-invariance actually calls for, by walking the alternatives that also live "inside the mix." I could
re-tune the global scale `w` for SDXL — but `w` is one number applied identically to every sample and
every step, and a single global scalar cannot fix a mis-scaling that varies from image to image and step
to step; it would just trade one fixed compromise for another, and I already argued against sweeping it.
I could make `w` time-varying, `w(t)` — but that still applies the same scale to every sample at a given
`t`, so it corrects a per-*step* pattern while leaving the per-*sample* geometry untouched. The only lever
that adapts to the actual `(noise_c, noise_uc)` geometry of *this* image at *this* step is a per-sample,
per-step scalar computed from the two predictions themselves. That is a different object from `w` in
exactly the dimension SDXL is complaining about — image-to-image variation the global knobs average over —
which is why it is the right shape of fix and not just another scalar to tune.

Let me derive what "how much" should be, because the right scalar falls out of a projection rather than a
sweep. I want guidance to amplify the part of the conditional prediction the unconditional prediction does
not already explain — the conditional *residual* — instead of amplifying the raw conditional vector
against a possibly mis-scaled unconditional baseline. So introduce a scalar `s` on the unconditional
prediction and write the guided noise as `v_s = s noise_uc + w (noise_c - s noise_uc)`. Expand it: `v_s = s
noise_uc + w noise_c - w s noise_uc = w noise_c - (w-1) s noise_uc`, which is the affine combination `(1-w)
s noise_uc + w noise_c`; at `s = 1` this is exactly `noise_uc + w(noise_c - noise_uc)`, the mix every prior
rung used, so `s` is a genuine generalization with the old rule at `s = 1`. Now how to pick `s`? If I could
see the true guided noise `v^*` I would just minimize `||v_s - v^*||^2` over `s`. But `v^*` is invisible —
I have only the two predictions — and that is the wall.

The way through is to bound the unavailable loss and see which part of the bound depends on `s`. Write
`delta = w - 1` so that `v_s = v_c + delta(v_c - s v_uc)` — check: `v_c + delta v_c - delta s v_uc =
w v_c - (w-1) s v_uc`, matching the expansion above. Then `v_s - v^* = (v_c - v^*) + delta(v_c - s v_uc)`, and
the unavailable loss is `||(v_c - v^*) + delta(v_c - s v_uc)||^2`. For any positive `lambda`, Young's
inequality gives `||a + b||^2 <= (1 + lambda)||a||^2 + (1 + 1/lambda)||b||^2`, so

  `||v_s - v^*||^2 <= (1 + lambda) ||v_c - v^*||^2 + (1 + 1/lambda) delta^2 ||v_c - s v_uc||^2.`

The first term still hides the truth `v^*`, but it does *not* depend on `s`; the only `s`-dependent part of
the bound is a positive constant times `||v_c - s v_uc||^2`. And I never have to choose `lambda`: it enters
only through the positive factor `(1 + 1/lambda) delta^2` multiplying the `s`-dependent term, and scaling a
function by a positive constant does not move its argmin, so `argmin_s` of the bound is the same for every
`lambda > 0`. The free parameter Young's inequality introduced drops out of the answer, which is the tell
that the projection is intrinsic to the problem and not an artifact of how I bounded it. Concretely the
only `s`-dependent part is `||v_c - s v_uc||^2` — equivalently, in noise units, `||noise_c - s
noise_uc||^2`. That is a quantity built entirely from the two predictions I already have. So minimizing the
bound replaces the impossible "match the invisible truth" with a solvable projection. It is a one-line
least squares: differentiate `||noise_c - s noise_uc||^2 = ||noise_c||^2 - 2 s noise_c^T noise_uc + s^2
||noise_uc||^2` with respect to `s`, set `-2 noise_uc^T (noise_c - s noise_uc) = 0`, and solve,

  `s* = (noise_c^T noise_uc) / ||noise_uc||^2.`

The second derivative is `2 ||noise_uc||^2 > 0`, so this is a genuine minimizer whenever `noise_uc` is
nonzero, not a saddle or a maximum. Geometrically `s* noise_uc` is the orthogonal projection of `noise_c`
onto the line spanned by `noise_uc`, and `noise_c - s* noise_uc` is the residual the unconditional
prediction cannot explain — exactly the direction I want guidance to push, instead of the whole conditional
vector measured against a raw, possibly mis-scaled baseline. So the optimized-scale mix is `s* noise_uc + w
(noise_c - s* noise_uc)`: rescale the unconditional baseline to its best least-squares match, then amplify
only the leftover conditional residual. It is per-sample (the projection is computed over each sample's
flattened prediction), it adds one dot product and one squared norm per step — no extra network evaluation,
well inside the fixed NFE budget — and it reduces to the prior mix exactly when the residual vanishes.

A two-dimensional toy makes concrete what the projection actually removes. Take `noise_uc = (1, 0)` and a
strongly-aligned `noise_c = (0.9, 0.3)`. The plain difference the prior rungs amplify is `noise_c -
noise_uc = (-0.1, 0.3)` — note the `-0.1` component *along* `noise_uc`, a leftover of the mis-scaled
baseline that has nothing to do with the conditional-specific direction. Guidance at `w = 7.5` blows that
spurious component up to `-0.75` along `noise_uc`, injecting `0.75` units of raw unconditional prediction
(sign-flipped) into the guided noise for no semantic reason. Now the optimized scale: `s* = <noise_c,
noise_uc>/||noise_uc||^2 = 0.9/1 = 0.9`, and the residual `noise_c - s* noise_uc = (0.9, 0.3) - 0.9(1,0) =
(0, 0.3)` is purely orthogonal to `noise_uc` — exactly the conditional-specific component, with the
along-baseline contamination removed. Amplifying `(0, 0.3)` by `w` pushes only in the direction the
conditional prediction adds, never along the shared baseline. That `-0.1 -> -0.75` spurious term the plain
mix would have amplified, and the optimized mix zeroes, is the per-step mis-scaling made visible.

I should check this is genuinely a new degree of freedom and not just a relabeled guidance scale, because
if `s*` only ever rescaled what `w` already controls it would buy nothing. It does not. `w` is a single
global scalar fixed for the whole run, applied identically to every sample and every step; `s*` is computed
per sample and per step from the actual geometry of that step's two predictions. The two act on different
objects — `w` sets *how hard* to push along the conditional residual, `s*` sets *what the residual is* by
fixing the baseline the residual is measured against. There is a clean reading in the implicit-classifier
language I used at rung 2: there, the guided direction was the difference `noise_c - noise_uc`, the
implicit-classifier score with a *unit* coefficient on the unconditional term. The optimized scale says
that unit coefficient was an unexamined default — the implicit classifier is sharper when the unconditional
baseline is first projected onto the conditional prediction, so that the difference isolates only the
conditional-specific component and not a mis-scaled chunk of the shared image prior that both predictions
carry.

Now the limit checks, which are the real test of whether the projection is the *right* scalar rather than a
tuned one. Suppose the two predictions are collinear, `noise_c = gamma noise_uc`. Then `s* = gamma
noise_uc^T noise_uc / ||noise_uc||^2 = gamma`, and the residual `noise_c - s* noise_uc = gamma noise_uc -
gamma noise_uc = 0` — guidance has nothing to amplify, and the mix collapses to `s* noise_uc = noise_c`,
the strongest baseline's behavior exactly. In the special collinear-and-equal case `noise_c = noise_uc` we
get `s* = 1`, recovering the prior mix's unit coefficient. So wherever the two predictions agree up to
scale, the finale is identical to zero-init; it can only differ where there is a genuine off-axis
component to correct. At the opposite extreme, if `noise_uc` were orthogonal to `noise_c` then `noise_c^T
noise_uc = 0`, `s* = 0`, and guidance pushes along the full conditional vector `w noise_c` with no
unconditional baseline subtracted at all — also sensible, since an unconditional prediction that shares no
direction with the conditional one explains none of it and should be discarded. Both limits are exactly
what "amplify only the unexplained residual" should produce, which is the reassurance I wanted that `s*` is
derived, not fitted.

How big is the correction likely to be in practice? Decompose `s* = (noise_c^T noise_uc)/||noise_uc||^2 =
(||noise_c||/||noise_uc||) cos(theta)`, where `theta` is the angle between the two predictions. Both are
noise predictions of the same latent under two conditionings that share almost everything except the
prompt, so I expect them to be strongly aligned — `cos(theta)` well below 1 but far from 0 — and of
comparable norm, which puts `s*` somewhat below 1 rather than at it. That matters because the change to the
guided direction relative to the baseline is `w (1 - s*) noise_uc`: even a modest `1 - s* ~ 0.1` gets
multiplied by `w = 7.5`, giving an order-`0.75 noise_uc` shift in the guided noise — an order-one change,
not a rounding correction. So I expect the optimized scale to be doing real, measurable work rather than a
negligible nudge; whether that work lowers FID, and whether `s*` really sits below 1 as I am guessing, is
something only the FID and CLIP tables can confirm. I am not certain of the sign of the effect, only that
its magnitude is not negligible.

This composes with everything the strongest baseline already does, which is the whole point. The finale is
the full method the `zeroinit` baseline only half-implemented, and it is three pieces, each addressing a
distinct failure I diagnosed on the way up. The `noise_uc` renoise fixes the off-manifold drift — rung
one's lesson. The inert prefix removes the unreliable high-noise steps — rung three's lesson. And the
optimized scale corrects the per-step mis-scaling of the unconditional baseline that every prior rung left
at a fixed coefficient of 1 — the new ingredient the baseline omitted. The first two are already
in the strongest baseline; only the third is new, so the edit is surgical: keep the inert prefix (`if step
< K: continue`, `K = 2`), keep CFG++'s manifold-safe renoise with `noise_uc`, and replace *only* the mix —
swap `noise_pred = noise_uc + cfg_guidance (noise_c - noise_uc)` for `noise_pred = noise_uc * s* +
cfg_guidance (noise_c - noise_uc * s*)`. Nothing else in the step moves.

Let me verify the composition does not quietly undo an earlier fix, because stacking three interventions is
where a subtle regression would hide. The renoise line is untouched — it is still `zt = at_prev.sqrt() *
z0t + (1-at_prev).sqrt() * noise_uc` — so rung one's entire manifold and invertibility argument carries
over verbatim: the lift to the next latent is still along the clean unconditional direction, and the
optimized scale changes only which point `z0t` I denoise to, never how I renoise from it. Does the new
`z0t` stay sane? It is `(z_t - sqrt(1-a_t) noise_pred)/sqrt(a_t)` with the optimized `noise_pred`, and I
can check its limits through the same lens as before: when the predictions are collinear, `noise_pred =
noise_c` and `z0t = x_hat_c`, the full conditional posterior mean — the same well-behaved endpoint the
strongest baseline lands on. Away from collinearity, the optimized mix removes the along-baseline
contamination from the guided noise, which makes the denoised estimate depend on a *cleaner* residual, so
if anything the denoise is more faithful, not less. And the prefix is a pure loop guard that gates all of
this — it does not interact with the mix at all, it just decides whether the step runs. So the three
pieces are genuinely orthogonal in their action: prefix decides *whether* to guide, `noise_uc` renoise
decides *how to lift*, and `s*` decides *what to amplify* in the denoise. None steps on another, which is
the structural reassurance that I can add the third without disturbing the two that already work.

A few implementation cautions so the projection is faithful, and they are all about shape and dtype rather
than math. The dot product and norm must be taken per sample over the flattened non-batch dimensions: for
SD the latent is `(B, 4, 64, 64)`, so I flatten to `(B, 16384)`, sum `noise_c * noise_uc` and `noise_uc^2`
along dim 1 with `keepdim`, and get a `(B, 1)` scalar per image; for SDXL the latent is `(B, 4, 128, 128)`,
flattening to `(B, 65536)`, same reduction. I then reshape that `(B, 1)` to `(B, 1, 1, 1)` so it broadcasts
back over the channel and spatial axes — a single scalar per image, not a global scalar across the batch
(which would tie unrelated prompts together) and not a per-element one (which would be a different, much
higher-dimensional object than the scalar the derivation produced). The denominator needs a small floor
(`+1e-8`) so a near-zero unconditional prediction does not blow up `s*`. The scale must be cast back to the
prediction's dtype, since the SD path runs under fp16 autocast and the reduction may promote it. And the
projection is computed from the *detached* predictions inside the existing `no_grad` block — it is a
sampling-time correction, nothing is trained, and the `requires_grad_()` fossil on the latent stays inert
exactly as it was at rung one. None of this touches the renoise, which stays `noise_uc`; the optimized
scale lives entirely inside the construction of the denoise mix. For SDXL the fill is identical in
`reverse_process()`: same `if step < K: continue` guard, same per-sample projection over the flattened
predictions, same `noise_uc` renoise, indexing `alphas_cumprod[t]`. The full scaffold module for both
variants is in the answer.

It is worth totalling the cost of the finished method against the budget, because the accounting is
quietly favorable. The optimized scale is one dot product and one squared norm per step — vector
operations on tensors I already hold, no network call. The `noise_uc` renoise is free; it was always one
of the two predictions. And the inert prefix does not merely add nothing, it *removes* work: skipping the
first `K = 2` steps means two fewer `predict_noise` calls per image than a no-skip sampler. So the full
three-piece method runs at strictly *fewer* network evaluations than plain CFG on the same trajectory,
while being the strongest rule on the ladder — the two saved calls at the useless high-noise start pay for
nothing lost, since the prefix argument said those steps were net harmful anyway. The contract's "exactly
two predictions per step, no extra NFE" ceiling is not just respected; the finale sits under it.

The bar this has to clear is the strongest baseline's measured FID — 22.76 / 23.31 / 25.49 — and the
expectation is specific because of where the projection can and cannot act. Since the optimized scale only
*changes the mix on the steps that are already guided*, and leaves the renoise and the prefix exactly as
the strongest baseline has them, it should never do worse than zero-init when `noise_c` and `noise_uc`
happen to be collinear (the residual vanishes and the mix recovers the baseline) and should help wherever
they are not — i.e. wherever the raw unconditional baseline is mis-scaled, which by the SDXL near-invariance
is most plausibly on the model that resisted every other lever. So the sharpest prediction is on SDXL: the
variant whose FID barely moved across rungs 1-3 (25.88 -> 25.74 -> 25.49, a total of `0.384`) is exactly
where a per-step correction to the mix, rather than to the renoise or the timing, has the most untapped
room, and a drop below 25.49 there would be the cleanest evidence that the optimized scale is doing real
work. I should be honest that I do not have a first-principles reason SDXL *should* carry the most
per-sample mis-scaling — I can wave at its larger `65536`-dimensional latent giving the two predictions
more room to differ in norm and alignment, but that is a hypothesis, not a derivation. What I actually have
is the empirical near-invariance across rungs 1-3: the timing and renoise levers moved SDXL only `0.384`
total, so by elimination its residual error is most likely in the one place none of them touched, the mix
interior, and `s*` is the only lever aimed there. So the SDXL prediction is a bet grounded in what is left
after eliminating the other two levers, and the FID table is what settles it. On SD v1.5 and SD v2-base I expect smaller gains on top of an already-strong 22.76 / 23.31, since
those variants were largely fixed by the renoise-and-prefix combination and have less residual mis-scaling
left to correct. What I would validate, with the seed and prompt set pinned exactly as the baselines ran:
that all three FID numbers come in at or below zero-init's, with the largest improvement on SDXL, and that
CLIP score does not fall — because the optimized scale amplifies the conditional *residual*, so if it were
merely trading alignment for distribution-matching, CLIP would drop while FID improved, and the method
would then have to be judged on whether it preserved the prompt-following that guidance is for.
