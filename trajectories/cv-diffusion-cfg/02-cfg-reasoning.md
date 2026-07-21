CFG++'s numbers came back worse than I would have guessed, which sharpens what to try next. On the
three frozen variants the on-manifold sampler scored FID 23.99 on SD v1.5, 24.89 on SD v2-base, and
25.88 on SDXL. Read as data before I react: all three are the *highest* FID on the board — there is
no variant where the manifold-safe renoise is even middling — and they climb monotonically, `24.89 -
23.99 = 0.90` and `25.88 - 24.89 = 0.99`, a near-constant `~0.9-1.0` step per variant tier. That
ordering is not noise: the higher-capacity, higher-resolution models have sharper reference
distributions, and an under-committed sampler drifts farther from a sharper target, so the same
conservatism costs progressively more up the tier. I have only seed 42 — the harness pins it — so I
read *levels* and *cross-variant structure*, not spreads, and I should not over-read a tenth of a
point. But the loudest signal survives that caution: `25.88` on SDXL is both the highest absolute FID
and the top of the climb, the model where the unconditional renoise has the most room to
under-commit. And since the cross-variant steps are `~0.9-1.0` of a point, FID differences here are
tenths, not units — whatever I try next should be predicted in tenths: closing under-commitment
plausibly recovers a fraction of a point per variant, most where the sample sits farthest from a
sharp reference, not several points.

This is the risk I flagged when I started at CFG++: the task does not score stability or
invertibility, it scores FID against the COCO reference distribution, and on that single axis the
gentle interpolation plus unconditional renoise sits farther from the reference than I hoped. The
mechanism is concrete. CFG++'s step forms `eps_cfgpp = eps_uc + lambda (eps_c - eps_uc)`,
Tweedie-denoises with the mix, then renoises with the *unconditional* noise `eps_uc`. The conditional
signal enters a step only through the denoise term, weighted by `lambda`; the renoise, which
contributes the larger `sqrt(1-a_{t-1})` share of the transition at high noise, carries *none* of it,
so per step the prompt-ward push is a fraction of what it could be, and whatever conditional
displacement the denoise introduced is partly overwritten when the next latent is reconstituted along
a purely unconditional direction. Add up many gentle, unconditionally-renoised steps and the
distribution is clean but under-committed: it does not sharpen onto the prompt-consistent modes as
aggressively as the FID reference — itself real, sharp images — rewards. The diagnosis is not "CFG++
is broken"; it is "CFG++ trades distribution sharpness for manifold safety, and on this benchmark the
trade went the wrong way."

If under-commitment is the disease, I have three moves: raise the interpolation scale toward `1`;
blend the two renoise directions, `beta eps_uc + (1-beta) eps_g`; or put the *guided* noise fully
back into the renoise, standard CFG. Raising `lambda` does not touch the leak: even at `lambda = 1`
the denoise becomes the full conditional posterior-mean but the renoise is *still* `eps_uc`, so the
per-step conditional injection into the renoise is `sqrt(1-a_{t-1})` times a renoise coefficient on
`(eps_c - eps_uc)` that is `0` regardless of `lambda`. The compounding sum stays exactly `0` no
matter how hard I turn the denoise; the disease is the unconditional renoise, not the size of the
nudge. The partial blend does address the leak but introduces a fresh continuous hyperparameter and
smears together the two clean rules I am trying to *contrast* — I want to know whether the guided
renoise beats the manifold-safe one before interpolating between them. So the disciplined next step
is the extreme point `beta = 0`: standard classifier-free guidance, filled cleanly so its number is
directly comparable to CFG++'s.

Standard CFG treats guidance not as a manifold-constrained nudge but as sampling from a *sharpened*
posterior `p^w(x|c) ∝ p(x) p(c|x)^w`. Take `grad_x log`: `grad_x log p^w(x|c) = grad_x log p(x) + w
grad_x log p(c|x)`. The implicit-classifier identity does the work: by Bayes `p(c|x) ∝ p(x|c)/p(x)`,
so `grad_x log p(c|x) = grad_x log p(x|c) - grad_x log p(x)` — the class signal is the *difference*
of the conditional and unconditional scores, no separate classifier required. Converting scores to
noise (`grad_x log p_t(x) = -eps_uc/sqrt(1-a_t)`, conditional likewise), the `1/sqrt(1-a_t)` is a
global scalar on both terms and cancels, leaving the clean linear mix `eps_g = eps_uc + w (eps_c -
eps_uc)` with nothing classifier-shaped left. The entire class signal — the thing classifier guidance
trained a whole network to compute the gradient of — is just the gap `eps_c - eps_uc` between two
predictions I already have. Amplifying it by `w` raises `p(c|x)` to the `w`-th power, an
inverse-temperature sharpening: at `w = 7.5`, if one candidate is twice as prompt-consistent as
another the sharpened density weights it `2^7.5 ≈ 181x` more, a `3:1` ratio becomes `3^7.5 ≈
3790:1`. That enormous concentration cuts both ways — it is exactly what can pull the sample
distribution onto the sharp reference modes and lower FID, and exactly what, on a mis-estimated
classifier direction, collapses diversity and over-saturates.

The structural fact is what happens to `eps_g` inside the DDIM step: standard CFG substitutes it into
*both* halves — the line that differs from the CFG++ fill is the renoise, `z_{t-1} = sqrt(a_{t-1}) x_hat_g
+ sqrt(1-a_{t-1}) eps_g`. The guided noise drives the lift, not the unconditional noise. Why would
that lower FID when I just argued the guided renoise is off-manifold? Because "off-manifold" and
"closer to the reference distribution" are not the same axis. At the noisy steps that dominate most
of a DDIM trajectory `a_{t-1}` is small, so `sqrt(1-a_{t-1}) ~ 0.9` and the renoise carries most of
the *magnitude* of the next latent — which is why swapping that direction from `eps_uc` to `eps_g` is
a large structural change. Standard CFG's renoise injects `sqrt(1-a_{t-1}) w (eps_c - eps_uc)`, about
`6.8 (eps_c - eps_uc)` of prompt-amplified signal into `z_{t-1}` every step; CFG++'s injects exactly
`0`. So standard CFG accumulates a per-step conditional injection CFG++ structurally never receives,
and that accumulation is the sharpening — the `p(c|x)^w` concentration onto the prompt-consistent
modes where the FID reference lives. The price is over-saturation and broken inversion, but the task
measures neither; a sharper, more prompt-committed distribution can be *closer* to the reference
statistics than CFG++'s cautious one. The harder push buys FID on exactly the axis the metric
rewards, the opposite of what the CFG++ fill optimized for.

Now the scaffold-specific subtlety, the single most important difference from the CFG++ fill. There I
kept the mix and let the harness supply the small CFG++ scale through `cfg_guidance`. But standard
CFG needs a *moderately high* scale to do its sharpening — the `w in [5, 30]` band — and the harness
by default passes the *small* CFG++ value. If I swap the renoise to `eps_g` but leave a scale around
`0.6` in place, `eps_g = eps_uc + 0.6 (eps_c - eps_uc)` is only `60%` of the way from `eps_uc` toward
`eps_c`, and renoising with that is barely distinguishable from renoising with `eps_uc`: the
sharpening never fires, and I get a strictly worse CFG++. So the fill must make *two* load-bearing
changes: override the incoming scale, `cfg_guidance = 7.5` at the top of `sample()`, ignoring
whatever the harness passed; and swap the renoise from `noise_uc` back to `noise_pred`. The guided
renoise is pointless at a tiny scale and the high scale is what makes the sharpening strong enough to
move FID — neither works without the other. `w = 7.5` is the canonical high-quality CFG scale, and I
take it as given rather than sweeping it: the harness pins the seed and prompt set, I am comparing
rules, and a swept `w` would confound "guided renoise beats manifold renoise" with "I found a good
`w`." I am not pretending the denoise is on-manifold now — `x_hat_g = -6.5 x_hat_uc + 7.5 x_hat_c` is
the same extrapolation I warned against when deriving CFG++ — I am *knowingly* accepting it, because the
metric rewards distribution sharpness rather than manifold-safety.

The rest of the classifier-free story — training one network with the condition randomly nulled,
getting the unconditional branch for free — is training-time machinery, none of it mine here: the
weights are frozen, `predict_noise` already returns both predictions, and the only live slot is the
inference-time rule: mix at a high scale, denoise with the mix, renoise with the mix.

Reading the result as an accounting changes what the number means. By swapping the renoise back to
`eps_g` I re-incur both costs I removed at the CFG++ fill: the `O(w)` inversion error and the
off-manifold drift. So the measured FID is the *net* of a benefit (concentration onto reference
modes) and a cost (off-manifold distortion) on the same trajectory, and the size of any win is
itself informative — a large drop below CFG++ would say the sharpening dominates cleanly, a small one
that the two effects nearly cancel. The *margin* is the reading, not just the sign.

For SDXL the contract is identical in `reverse_process()`: override `cfg_guidance = 7.5`, get
`noise_uc, noise_c` from `predict_noise` with the dual embeddings, denoise with `noise_pred`, renoise
with `noise_pred`. The full scaffold module for both variants is in the answer.

Stated against CFG++'s numbers: if under-commitment is why CFG++ landed at the top of the FID
range, putting the guided noise back into the renoise at `w = 7.5` should *lower* FID below CFG++'s
23.99 / 24.89 / 25.88 on all three variants, largest where CFG++ was weakest and had the most room —
SDXL at 25.88. The mirror-image risk is over-sharpening: if `7.5` distorts rather than concentrates
the distribution, the tell-tale signature is CLIP climbing (the sample follows the prompt harder)
while FID stalls or regresses. If the guided renoise wins, the next question is whether I can keep its
harder push while clawing back some of CFG++'s on-manifold cleanliness — the renoise choice and the
high-noise early steps being the two obvious levers.
