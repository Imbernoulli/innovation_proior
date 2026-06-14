The strongest baseline cleared the bar I set for it, and in doing so it told me exactly the one piece
it left unused. Zero-init — CFG++'s manifold-safe renoise with the first `K = 2` highest-noise steps
made inert — came in at FID 22.76 / 23.31 / 25.49 on SD v1.5 / SD v2-base / SDXL, under both priors
on every variant: below standard CFG's 23.65 / 24.29 / 25.74 and below plain CFG++'s 23.99 / 24.89 /
25.88. So the rung-3 hypothesis held cleanly: keeping CFG++'s clean `noise_uc` renoise while deleting
its under-committed, unreliable high-noise prefix beat the hard guided renoise *and* the no-skip
clean renoise, and it did so without paying standard CFG's off-manifold cost. The gains were largest
where there was the most to fix — 1.23 on SD v1.5, 0.98 on SD v2-base — and smallest on SDXL (0.25),
which is the variant that has stubbornly resisted every move so far (cfgpp 25.88, cfg 25.74, zeroinit
25.49). SDXL is the signal: a high-resolution model whose FID barely budges as I swap renoise rules
and skip steps, which says the remaining error there is not in *when* I guide or *which noise
renoises*, but in *how the two predictions are mixed* at the steps I do guide.

And that is precisely the lever I flagged and set aside at rung 3. The zero-init baseline only
implemented half of the method it is named after: it took the inert prefix but kept the *plain* CFG++
mix, `noise_pred = noise_uc + cfg_guidance (noise_c - noise_uc)`. The other half — the half the
harness's fill omits — is a per-sample correction to *how the unconditional prediction enters the
mix*. That is the natural next move, because every rung so far has treated the coefficient on
`noise_uc` as fixed at 1: CFG++, standard CFG, and zero-init all subtract a full unit of `noise_uc`
inside the difference `noise_c - noise_uc`. But there is no reason the best baseline to subtract is
exactly one times the raw unconditional prediction. If the unconditional prediction is, at this
particular `(z_t, t)`, too large or too small or partly mis-aligned with the conditional one,
subtracting a full unit of it pollutes the guidance direction with a component the model did not
intend. The fix is to choose, per sample and per step, *how much* of `noise_uc` to subtract.

Let me derive what "how much" should be, because the right scalar falls out of a projection. I want
guidance to amplify the part of the conditional prediction that the unconditional prediction does not
already explain — the conditional *residual* — rather than amplifying the raw conditional vector
against a possibly mis-scaled unconditional baseline. So introduce a scalar `s` on the unconditional
prediction and write the guided noise as `s * noise_uc + w (noise_c - s * noise_uc)`, which collapses
to `(1 - w) s noise_uc + w noise_c`; at `s = 1` this is exactly the mix every prior rung used. Now
how to pick `s`? If I could see the true guided noise I would minimize `||v_s - v^*||^2` over `s`,
but `v^*` is invisible — that is the wall. The way through is to write `delta = w - 1` so the guided
prediction is `v_s = v_c + delta(v_c - s v_uc)`, then the unavailable loss is
`||(v_c - v^*) + delta(v_c - s v_uc)||^2`. For any positive `lambda`, Young's inequality bounds this
above by `(1 + lambda)||v_c - v^*||^2 + (1 + 1/lambda) delta^2 ||v_c - s v_uc||^2`. The first term
still hides the truth, but it does *not* depend on `s`; the only `s`-dependent part of the bound is a
positive constant times `||v_c - s v_uc||^2` — equivalently, in noise units, `||noise_c - s
noise_uc||^2`. That is a quantity built entirely from the two predictions I already have, so
minimizing the bound replaces the impossible "match the invisible truth" with a solvable projection.
It is a one-line least squares: set the derivative `-2 noise_uc^T (noise_c - s noise_uc)` to zero and
solve,

  `s* = (noise_c^T noise_uc) / ||noise_uc||^2`.

The second derivative `2 ||noise_uc||^2` is positive, so this is the minimizer whenever `noise_uc` is
nonzero. Geometrically `s* noise_uc` is the orthogonal projection of `noise_c` onto the line spanned
by `noise_uc`, and `noise_c - s* noise_uc` is the residual the unconditional prediction cannot
explain — exactly the direction I want guidance to push, instead of the whole conditional vector
measured against a raw, possibly mis-scaled baseline. So the optimized-scale mix is `s* noise_uc + w
(noise_c - s* noise_uc)`: rescale the unconditional baseline to its best least-squares match, then
amplify only the leftover conditional residual. It is per-sample (the projection is computed over
each sample's flattened prediction), it adds one dot product and one squared norm per step — no extra
network evaluation, well inside the fixed NFE budget — and it reduces to the prior mix exactly when
`noise_c` and `noise_uc` are already collinear (then the residual `noise_c - s* noise_uc` vanishes, and
`s* = 1` in the equal-predictions case).

I should check this is genuinely a new degree of freedom and not just a relabeled guidance scale,
because if `s*` only ever rescaled what `w` already controls it would buy nothing. It does not: `w`
is a single global scalar fixed for the whole run, applied identically to every sample and every step,
whereas `s*` is computed per sample and per step from the actual geometry of that step's two
predictions. The two act on different objects — `w` sets *how hard* to push along the conditional
residual, `s*` sets *what the residual is* by fixing the baseline the residual is measured against.
There is also a clean reading in the implicit-classifier language I used at rung 2: there, the guided
direction was the difference `noise_c - noise_uc`, the implicit-classifier score with a *unit*
coefficient on the unconditional term. The optimized scale says that unit coefficient was an
unexamined default — the implicit classifier is sharper when the unconditional baseline is first
projected onto the conditional prediction, so that the difference isolates only the conditional-specific
component and not a mis-scaled chunk of the shared image prior. And a sanity check on the limits: if
the two predictions are collinear, the residual `noise_c - s* noise_uc` is zero and I recover the
strongest baseline's mix exactly (with `s* = 1` when they are equal), so the finale can only differ
where there is a real off-axis component to correct; if `noise_uc` were orthogonal to `noise_c`,
`s* = 0` and guidance would push along the full conditional vector. Both
limits are sensible, which is the reassurance that the projection is the right scalar rather than a
tuned one.

This composes with everything the strongest baseline already does, which is the whole point. The
finale is the full method the `zeroinit` baseline only half-implemented: keep the inert prefix (`if
step < K: continue`, `K = 2`), keep CFG++'s manifold-safe renoise with `noise_uc`, and replace *only*
the mix — swap `noise_pred = noise_uc + cfg_guidance (noise_c - noise_uc)` for the optimized-scale
form `noise_pred = noise_uc * s* + cfg_guidance (noise_c - noise_uc * s*)`. Three pieces, each
addressing a distinct failure I diagnosed on the way up: the `noise_uc` renoise fixes the off-manifold
drift (rung 1's lesson), the inert prefix removes the unreliable high-noise steps (rung 3's lesson),
and the optimized scale corrects the per-step mis-scaling of the unconditional baseline that every
prior rung left at a fixed coefficient of 1. The first two are already in the strongest baseline; the
third is the genuinely new, published ingredient it omitted.

A few implementation cautions so the projection is faithful. The dot product and norm must be taken
per sample over the flattened non-batch dimensions, then reshaped to broadcast back over the latent's
channel and spatial axes — a single scalar per image, not a global scalar across the batch and not a
per-element one. The denominator needs a small floor (`+1e-8`) so a near-zero unconditional
prediction does not blow up `s*`. The scale must be cast back to the prediction's dtype (the SD path
runs under fp16 autocast). And the projection is computed from the *detached* predictions inside the
existing `no_grad` block — it is a sampling-time correction, nothing is trained. None of this touches
the renoise, which stays `noise_uc`; the optimized scale lives entirely inside the construction of the
denoise mix. For SDXL the fill is identical in `reverse_process()`: same `if step < K: continue`
guard, same per-sample projection over the flattened predictions, same `noise_uc` renoise, indexing
`alphas_cumprod[t]`. The full scaffold module for both variants is in the answer.

The bar this has to clear is the strongest baseline's measured FID — 22.76 / 23.31 / 25.49 — and the
expectation is specific. Because the optimized scale only *changes the mix on the steps that are
already guided*, and leaves the renoise and the prefix exactly as the strongest baseline has them, it
should never do worse than zero-init when `noise_c` and `noise_uc` happen to be collinear (the residual
vanishes and the mix recovers the baseline) and should help wherever they are not — i.e. wherever the raw unconditional
baseline is mis-scaled, which is most likely on the model that has resisted every other lever. So the
sharpest prediction is on SDXL: the variant whose FID barely moved across rungs 1-3 (25.88 -> 25.74
-> 25.49) is exactly where a per-step correction to the mix, rather than to the renoise or the
timing, has the most untapped room, and a drop below 25.49 there would be the cleanest evidence that
the optimized scale is doing real work. On SD v1.5 and SD v2-base I expect smaller gains on top of an
already-strong 22.76 / 23.31, since those variants were largely fixed by the renoise-and-prefix
combination. What I would validate, with the seed and prompt set pinned exactly as the baselines ran:
that all three FID numbers come in at or below zero-init's, with the largest improvement on SDXL, and
that CLIP score does not fall — the optimized scale amplifies the conditional *residual*, so if it
were merely trading alignment for distribution-matching, CLIP would drop while FID improved, and the
method would have to be judged on whether it preserved the prompt-following that guidance is for.
