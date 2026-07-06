CFG++'s numbers came back and they are worse than I would have guessed, which sharpens exactly what to
try next. On the three frozen variants the on-manifold sampler scored FID 23.99 on SD v1.5, 24.89 on SD
v2-base, and 25.88 on SDXL. Let me read those three numbers as data before I react. First, all three
are the *highest* FID I have on the board — there is no variant on which the manifold-safe renoise is
even middling. Second, they climb monotonically across the variants: `24.89 - 23.99 = 0.90` from SD v1.5
to SD v2-base, and `25.88 - 24.89 = 0.99` from SD v2-base to SDXL, a near-constant `~0.9-1.0` step up
per variant tier. That ordering is not random noise: the higher-capacity, higher-resolution models have
sharper reference distributions, and an under-committed sampler drifts farther from a sharper target, so
the same conservatism costs progressively more as I move up the model tier. I only have one seed (42) —
the harness pins it — so I cannot read a variance or a confidence interval off this table; I have to
read *levels* and *cross-variant structure*, not spreads, and I should be careful not to over-interpret
a tenth of a point. But the loudest signal survives that caution: `25.88` on SDXL is both the highest
absolute FID and the top of the monotone climb, the high-resolution model where the unconditional renoise
has the most room to under-commit and drift the sample statistics away from the reference set.

One more thing to read off the table before I move, so I calibrate what a "win" even looks like. The
three numbers live in the `24-26` band and the cross-variant steps are `~0.9-1.0` of a FID point. FID
differences at this benchmark are tenths of a point, not units — the gap between a good and a mediocre
guidance rule here is small in absolute terms. So whatever I try next, I should predict its effect in
*tenths*, and I should not expect the guided renoise to slash several points off; if under-commitment is
the disease, closing it plausibly recovers a fraction of a point per variant, most where the sample sits
farthest from a sharp reference. Setting that expectation now keeps me from over-reading a modest
improvement as a triumph or a modest one as a failure.

I built CFG++ to be the principled floor: it removes the off-manifold extrapolation in the denoise and
the guided noise in the renoise, so it should be stable, artifact-free, and invertible, and I believe it
is all of those things. But the task does not score stability or invertibility; it scores FID against the
COCO reference distribution, and on that single axis the gentle interpolation plus unconditional renoise
sits farther from the reference statistics than I hoped. That is precisely the risk I flagged when I
started here: the very caution that fixes saturation and broken inversion can cost FID against a sampler
that pushes harder.

So let me read the failure precisely before I move, because the diagnosis dictates the fix. CFG++'s
reverse step is: form the mixed noise `eps_cfgpp = eps_uc + lambda (eps_c - eps_uc)`, Tweedie-denoise
with that mix, then renoise with the *unconditional* noise `eps_uc`. The renoise choice is the whole
character of the method. By renoising with `eps_uc` I lift the denoised estimate to the next noisy
manifold along the *clean, unconditional* direction — which is exactly why it stays on-manifold, but
also exactly why the sample's per-step motion toward the prompt is limited to the small `lambda` nudge
inside the denoise. Here is the mechanism made concrete: the conditional signal enters a CFG++ step only
through the denoise term, weighted by `lambda`; the renoise, which contributes the larger `sqrt(1-a_{t-1})`
share of the transition at high noise, carries *none* of it. So per step the prompt-ward push is a
fraction of what it could be, and worse, whatever conditional displacement the denoise did introduce is
partly overwritten when the next latent is reconstituted along a purely unconditional noise direction.
The harness runs CFG++ at a small interpolation scale (the value it passes into `cfg_guidance`), so the
conditional push per step is deliberately modest to begin with. Add up many gentle, unconditionally-renoised
steps and the generated distribution is clean but under-committed: it does not sharpen onto the
prompt-consistent modes as aggressively as the FID reference — itself a set of real, sharp images —
rewards. The diagnosis is not "CFG++ is broken"; it is "CFG++ trades distribution sharpness for manifold
safety, and on this FID-scored benchmark the trade went the wrong way."

If under-commitment is the disease, what are my actual moves? I can see three. I could keep the CFG++
renoise but raise the interpolation scale toward `1` — push `lambda` up so the denoise commits harder.
I could blend the two renoise directions, `beta eps_uc + (1-beta) eps_g`, a continuous dial between
manifold-safe and guided. Or I could put the *guided* noise fully back into the renoise, which is
standard CFG. Let me walk the first two far enough to reject them on their own terms rather than by
taste. Raising `lambda` in CFG++ does not touch the actual leak: even at `lambda = 1` the denoise
becomes the full conditional posterior-mean `x_hat_c`, but the renoise is *still* `eps_uc`, so the
transition to the next latent still discards the conditional direction and the per-step signal still
cannot compound across steps. The disease is the unconditional renoise, not the size of the denoise
nudge, so turning the denoise knob to its maximum leaves the mechanism of under-commitment intact — it
just approaches conditional-only sampling, which the task already tells me is weak. I can put a number
on this: the per-step conditional injection into the renoise is `sqrt(1-a_{t-1}) * (renoise coefficient
on (eps_c - eps_uc))`, and for *any* CFG++ variant that renoise coefficient is `0` regardless of
`lambda`, so the compounding sum stays exactly `0` no matter how hard I turn the denoise. That is the
same quantity I will compute below as roughly `~7 (eps_c - eps_uc)` per step for the guided renoise; the
gap between `0` and `~7` is the entire thing the denoise knob cannot close. The partial-renoise
blend `beta eps_uc + (1-beta) eps_g` does address the leak, but it introduces a fresh continuous
hyperparameter I would have to set, and it deliberately smears together the two clean rules I am trying
to *contrast* — I want to know whether the guided renoise beats the manifold-safe one on this metric
before I start interpolating between them. So the disciplined next rung is the extreme point of that
blend, `beta = 0`: standard classifier-free guidance, the older, harder-pushing rule, filled cleanly so
its number is directly comparable to CFG++'s. The entire derivation I care about is *why renoising with
the guided noise pushes the sample's statistics toward the reference even though it abandons CFG++'s
manifold guarantee.* Let me build it from the ground the scaffold gives me.

Standard CFG starts from the same two predictions per step, `eps_uc` and `eps_c`, but it does not treat
guidance as a manifold-constrained nudge. It treats it as sampling from a *sharpened* posterior
`p^w(x|c) ∝ p(x) p(c|x)^w`. Take `grad_x log` of that sharpened density: `grad_x log p^w(x|c) = grad_x
log p(x) + w grad_x log p(c|x)`, the unconditional score plus `w` times the classifier gradient. Now
the implicit-classifier identity does the work. By Bayes, `p(c|x) ∝ p(x|c)/p(x)`, so `grad_x log p(c|x)
= grad_x log p(x|c) - grad_x log p(x)` — the class signal is the *difference* of the conditional and
unconditional scores, no separate classifier required. Convert scores to noise predictions: the score
of the noised marginal is `grad_x log p_t(x) = -eps_uc/sqrt(1-a_t)`, and the conditional score is
`-eps_c/sqrt(1-a_t)`. Substitute everything back:

  `grad_x log p^w(x|c) = -eps_uc/sqrt(1-a_t) + w ( -eps_c/sqrt(1-a_t) + eps_uc/sqrt(1-a_t) )
                       = -(1/sqrt(1-a_t)) ( eps_uc + w (eps_c - eps_uc) ).`

The `1/sqrt(1-a_t)` factor from converting score to epsilon is a global scalar on both terms, so it
cancels when I read the guided noise back off, and out drops the clean linear mix `eps_g = eps_uc + w
(eps_c - eps_uc)` with nothing classifier-shaped left in it. Worth pausing on why that difference is the
right object: the entire class signal — the thing classifier guidance trained a whole separate network
to compute the gradient of — is just the gap `eps_c - eps_uc` between two predictions I already have.
Amplifying it by `w` is what raises `p(c|x)` to the `w`-th power, i.e. an inverse-temperature sharpening
of the classifier: `w > 1` dumps probability mass onto the prompt-consistent modes. It is worth feeling
how violent `w = 7.5` is as an inverse temperature. If under the implicit classifier one candidate image
is twice as prompt-consistent as another, the sharpened density weights it not `2x` but `2^7.5 ≈ 181x`
more; a `3:1` ratio becomes `3^7.5 ≈ 3790:1`. That is enormous concentration, and it cuts both ways —
it is exactly the mechanism that could pull the sample distribution onto the sharp reference modes and
lower FID, and it is exactly the mechanism that, pushed too far, collapses diversity and over-saturates.
The same `2^7.5` that concentrates onto the right modes will, on a mis-estimated classifier direction,
concentrate just as hard onto the wrong ones. So standard CFG is
not "add a classifier gradient"; it is "amplify the difference of two predictions I am already
computing," and the only remaining question is whether that amplified difference should drive the lift
to the next latent or be discarded in favor of `eps_uc`.

The crucial structural fact is what happens to that `eps_g` inside the DDIM step: standard CFG
substitutes it for the noise in *both* halves. Tweedie-denoise with `eps_g`, and — the line that differs
from rung one — renoise with `eps_g` as well: `z_{t-1} = sqrt(a_{t-1}) x_hat_g + sqrt(1-a_{t-1}) eps_g`.
The guided noise drives the lift to the next noisy manifold, not the unconditional noise.

Why would that lower FID when I just argued at rung one that the guided renoise is off-manifold? Because
"off-manifold" and "closer to the reference distribution" are not the same axis, and I want to be precise
about the difference rather than wave at it. Renoising with `eps_g` means every step both denoises toward
*and* re-injects noise along the prompt-amplified direction, so the conditional signal compounds across
the trajectory instead of leaking out through an unconditional renoise. Contrast the two steps
term-by-term: CFG++ puts the conditional signal into the denoise (weight `lambda`) and nothing into the
renoise; standard CFG puts the `w`-amplified signal into the denoise *and* the same amplified signal into
the renoise, so the conditional direction is present in both the `sqrt(a_{t-1})` and the `sqrt(1-a_{t-1})`
shares of the next latent. Nothing cancels it out when the latent is reconstituted, so it accumulates step
over step. At a high guidance scale this is exactly the sharpening the inverse-temperature reading
predicts — `p(c|x)^w` for `w > 1` concentrating mass onto the prompt-consistent modes — and the FID
reference lives on those modes. The price is the over-saturation and the broken inversion I cared about
at rung one, but the task measures neither; it measures distance to the reference image statistics, and a
sharper, more prompt-committed sample distribution can be *closer* to that reference than CFG++'s cautious
one. So the prediction is that the harder push buys FID precisely on the axis the metric rewards, which
is the opposite of what rung one optimized for.

Let me quantify where in the step that compounding actually happens, because it tells me why the renoise
— not the denoise — is the lever. In the transition `z_{t-1} = sqrt(a_{t-1}) x_hat + sqrt(1-a_{t-1})
eps`, the denoise contributes the `sqrt(a_{t-1})` signal share and the renoise contributes the
`sqrt(1-a_{t-1})` noise share. At the noisy steps that dominate most of a DDIM trajectory `a_{t-1}` is
small, so `sqrt(1-a_{t-1})` is close to `1` and `sqrt(a_{t-1})` is small: the renoise term carries most
of the *magnitude* of the next latent. So the direction I renoise along is precisely where the bulk of
the transition's mass goes at high noise — which is exactly why swapping that direction from `eps_uc` to
`eps_g` is a large structural change and not a cosmetic one. Trace the conditional component across a
single step under each rule. Standard CFG's renoise injects `sqrt(1-a_{t-1}) w (eps_c - eps_uc)` of the
prompt-amplified direction directly into `z_{t-1}`; at `w = 7.5` and `sqrt(1-a_{t-1}) ~ 0.9` that is
about `6.8 (eps_c - eps_uc)` of conditional signal deposited into the next latent every step. CFG++'s
renoise injects exactly `0` of it — the `eps_uc` lift is orthogonal to the guidance question. Over the
whole trajectory standard CFG is therefore accumulating a sum of per-step conditional injections that
CFG++ structurally never receives, and that accumulation is the sharpening. This is the concrete content
of "the conditional signal compounds": not a metaphor but roughly `~7 (eps_c - eps_uc)` per step of extra
prompt-ward drift that the unconditional renoise throws away.

Now the scaffold-specific subtlety, which is the single most important implementation difference between
this rung and the last, and the thing I must get exactly right. At rung one I kept the mix expression and
let the harness supply the small CFG++ scale through `cfg_guidance`. But standard CFG needs a *moderately
high* scale to do its sharpening — the `w` in roughly `[5, 30]` band — and the harness, by default,
passes in the *small* CFG++ value. This is not a cosmetic detail; let me check what happens if I get it
wrong. Suppose I swap the renoise to `eps_g` but leave the harness's small scale in place, say some
`cfg_guidance` around `0.6`. Then `eps_g = eps_uc + 0.6 (eps_c - eps_uc)` is a vector only `60%` of the
way from `eps_uc` toward `eps_c`, and renoising with *that* is barely distinguishable from renoising with
`eps_uc` — the sharpening I derived never fires, because the thing I am compounding is a whisper. I would
get neither method: a barely-guided sampler that renoises with a barely-guided noise, a strictly worse
CFG++. So the standard-CFG fill must make *two* changes from rung one, and both are load-bearing:
override the incoming scale and fix it at the value CFG actually needs, *and* swap the renoise noise from
`noise_uc` back to `noise_pred`. Concretely, at the top of `sample()` it sets `cfg_guidance = 7.5`,
ignoring whatever the harness passed, then forms `noise_pred = noise_uc + cfg_guidance * (noise_c -
noise_uc)` at that high scale, Tweedie-denoises with `noise_pred`, and renoises with `noise_pred`. The
guided renoise is pointless at a tiny scale, and the high scale is what makes the sharpening strong
enough to move FID; neither change works without the other.

Two limit checks on the mix so I know the `7.5` is doing what I think. At `w = 1`, `eps_g = eps_c` and
the step is plain conditional sampling with no guidance at all. At `w = 0`, `eps_g = eps_uc` and it is
the unconditional sampler. So `w` interpolates through `w = 1` (conditional) and extrapolates beyond it
for `w > 1`, which is where the sharpening lives; `7.5` sits squarely in the useful band. And I should
be honest with myself about what `7.5` does to the denoise, because it is exactly the thing I warned
against at rung one: `x_hat_g = (1-w) x_hat_uc + w x_hat_c = -6.5 x_hat_uc + 7.5 x_hat_c`, the estimate
thrown `6.5` segment-lengths past the conditional endpoint, off `M`. I am not pretending this is
on-manifold. I am *knowingly* accepting the extrapolation this time, because the metric rewards
distribution sharpness rather than manifold-safety, and the whole bet of this rung is that on FID the
sharpening is worth the off-manifold distortion.

I should be explicit about the accounting, because it changes how I will read the result. By swapping the
renoise back to `eps_g` I am re-incurring both costs I paid to remove at rung one: the `O(w)` DDIM
inversion error and the off-manifold drift of the guided noise. So the measured FID of standard CFG is
not "the sharpening benefit"; it is the *net* of a benefit (concentration onto reference modes) and a
cost (off-manifold distortion) applied to the same trajectory. That means the size of any win is itself
informative. A large drop below CFG++ would say the sharpening dominates cleanly and the off-manifold
cost is cheap on this metric. A small drop — a tenth or two, consistent with the tenths-scale I
calibrated — would say the two effects are nearly cancelling, that the guided renoise is buying sharpness
and paying it back in distortion in almost equal measure. I do not know yet which it is, but I am
deciding in advance to read the *margin*, not just the sign: whether the guided renoise wins is one
question, and how cleanly it wins is a second one that will tell me whether there is a better rule that
keeps the sharpening while shedding the distortion.

I want to be careful not to over-tell this rung. The classifier-free derivation has a whole story about
training one network with the condition randomly nulled, about the implicit classifier not being the
gradient of any real classifier so the adversarial-attack worry dissolves, about getting the
unconditional branch for free. None of that training-time machinery is mine to touch here: the weights
are frozen, `predict_noise` already returns both `eps_uc` and `eps_c` from one batched call, and the
only slot I fill is the per-step combination and which noise drives each half of the DDIM step. So the
part of standard CFG that is *live* in this scaffold is exactly the inference-time rule: mix at a high
scale, denoise with the mix, renoise with the mix. The `7.5` is the canonical high-quality CFG scale and
I take it as given rather than sweeping it, because the harness pins the seed and the prompt set and I am
comparing rules, not tuning a single rule's scalar; a swept `w` would confound "guided renoise beats
manifold renoise" with "I found a good `w`," and I want the clean comparison.

For the SDXL variant the contract is identical in `reverse_process()`: index `at = alphas_cumprod[t]`,
`at_next = alphas_cumprod[t - skip]`, override `cfg_guidance = 7.5`, get `noise_uc, noise_c` from
`predict_noise` with the dual prompt embeddings, denoise with `noise_pred`, and renoise with
`noise_pred`. Same two changes from rung one, same two predictions per step, same fixed NFE budget. The
full scaffold module for both variants is in the answer.

The falsifiable expectation, stated against rung one's measured numbers. If under-commitment is really
why CFG++ landed at the top of the FID range, then putting the guided noise back into the renoise at the
high CFG scale should *lower* FID on all three variants below CFG++'s 23.99 / 24.89 / 25.88 — the harder
push pulling the generated distribution onto the prompt-consistent modes the reference set occupies. I
expect the largest gain exactly where CFG++ was weakest, SDXL at 25.88, because that is where the
unconditional renoise had the most room to under-commit and the monotone climb put the most distance
between the sample and a sharp reference. The risk is the mirror image of rung one: if `7.5` over-sharpens
— collapsing modes and over-saturating hard enough to *distort* the distribution rather than concentrate
it — FID could fail to improve or even regress, and I would expect the tell-tale signature to be CLIP
climbing (the sample follows the prompt harder) while FID stalls (the distribution has been pushed off
the reference in a different direction). So the clean test is simply whether standard CFG's three FID
numbers come in under CFG++'s three. If they do, the lesson is that on this benchmark the guided renoise
beats the manifold-safe one, and the next question becomes whether I can keep standard CFG's harder push
while clawing back *some* of CFG++'s on-manifold cleanliness — the renoise choice and the high-noise
early steps being the two obvious levers to revisit.
