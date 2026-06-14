CFG++'s numbers came back and they are worse than I would have guessed, which sharpens exactly what
to try next. On the three frozen variants the on-manifold sampler scored FID 23.99 on SD v1.5,
24.89 on SD v2-base, and 25.88 on SDXL — and every one of those is the *highest* FID I will see on
this ladder. I built CFG++ to be the principled floor: it removes the off-manifold extrapolation in
the denoise and the guided noise in the renoise, so it should be stable, artifact-free, and
invertible. And I believe it is all of those things. But the task does not score stability or
invertibility; it scores FID against the COCO reference distribution, and on that single axis the
gentle interpolation plus unconditional renoise sits *farther* from the reference statistics than I
hoped. That is precisely the risk I flagged when I started here: the very caution that fixes
saturation and broken inversion can cost FID against a sampler that pushes harder. The 25.88 on SDXL
is the loudest signal — the high-resolution model, where the unconditional renoise has the most room
to under-commit to the prompt and drift the sample's statistics away from the reference set.

So let me read the failure precisely before I move, because the diagnosis dictates the fix. CFG++'s
reverse step is: form the mixed noise `eps_cfgpp = eps_uc + lambda (eps_c - eps_uc)`, Tweedie-denoise
with that mix, then renoise with the *unconditional* noise `eps_uc`. The renoise choice is the whole
character of the method. By renoising with `eps_uc` I lift the denoised estimate to the next noisy
manifold along the *clean, unconditional* direction — which is exactly why it stays on-manifold, but
also exactly why the sample's per-step motion toward the prompt is limited to the small `lambda`
nudge inside the denoise. The harness runs CFG++ at a small interpolation scale (the value it passes
into `cfg_guidance`), so the conditional push per step is deliberately modest. Add up many gentle,
unconditionally-renoised steps and the generated distribution is clean but under-committed: it does
not sharpen onto the prompt-consistent modes as aggressively as the FID reference (itself a set of
real, sharp images) rewards. The diagnosis is not "CFG++ is broken"; it is "CFG++ trades distribution
sharpness for manifold safety, and on this FID-scored benchmark the trade went the wrong way."

If under-commitment is the disease, the obvious remedy is to put the *guided* noise back into the
renoise — which is precisely what standard CFG does. So the next rung is not a more elaborate method;
it is the older, harder-pushing one, standard classifier-free guidance, and the entire derivation I
care about is *why renoising with the guided noise pushes the sample's statistics toward the
reference even though it abandons CFG++'s manifold guarantee*. Let me build it from the ground the
scaffold gives me.

Standard CFG starts from the same two predictions per step, `eps_uc` and `eps_c`, but it does not
treat guidance as a manifold-constrained nudge. It treats it as sampling from a *sharpened*
posterior `p^w(x|c) ∝ p(x) p(c|x)^w`. Take `grad_x log` of that sharpened density and parameterize
the score with the noise predictor: the gradient of `log p(x)` is the unconditional score, the
gradient of `log p(c|x)` is — by the implicit-classifier identity `p(c|x) ∝ p(x|c)/p(x)` — the
*difference* of the conditional and unconditional scores, and out drops the linear mix `eps_g =
eps_uc + w (eps_c - eps_uc)`. Worth pausing on why that difference is the right object. A classifier
is hiding inside any pair of conditional-and-unconditional generative models: by Bayes,
`p(c|x) ∝ p(x|c)/p(x)`, so `grad_x log p(c|x) = grad_x log p(x|c) - grad_x log p(x)`, and in
noise-prediction units that is `-(1/sigma)(eps_c - eps_uc)`. The entire class signal — the thing
classifier guidance trained a whole separate network to compute the gradient of — is just the gap
between the conditional and unconditional predictions I already have. Amplifying it by `w` is what
raises `p(c|x)` to the `w`-th power, and the `sigma` factor from converting score to epsilon cancels
the `1/sigma` from converting back, leaving the clean linear mix with nothing classifier-shaped left
in it. So standard CFG is not "add a classifier gradient"; it is "amplify the difference of two
predictions I am already computing," and the renoise question is then simply whether that amplified
difference should drive the lift to the next latent or be discarded in favor of `eps_uc`. The crucial structural fact is what happens to that `eps_g` inside the
DDIM step: standard CFG substitutes it for the noise in *both* halves. Tweedie-denoise with `eps_g`,
and — the line that differs from rung one — renoise with `eps_g` as well: `z_{t-1} = sqrt(a_{t-1})
x_hat_g + sqrt(1-a_{t-1}) eps_g`. The guided noise drives the lift to the next noisy manifold, not
the unconditional noise.

Why would that lower FID when I just argued the guided renoise is "off-manifold"? Because
"off-manifold" and "closer to the reference distribution" are not the same axis. Renoising with
`eps_g` means every step both denoises toward and *re-injects noise along* the prompt-amplified
direction, so the conditional signal compounds across the trajectory instead of leaking out through
an unconditional renoise. At a high guidance scale this is exactly the sharpening that maps onto the
inverse-temperature reading: `p(c|x)^w` for `w > 1` dumps probability mass onto the
prompt-consistent modes, and the FID reference — real, prompt-relevant, sharp images — lives on
those modes. The price is the over-saturation and the broken inversion I cared about at rung one, but
the task does not measure either; it measures distance to the reference image statistics, and a
sharper, more prompt-committed sample distribution can be *closer* to that reference than CFG++'s
cautious one. So the prediction is that the harder push buys FID precisely on the axis the metric
rewards, which is the opposite of what rung one optimized for.

Now the scaffold-specific subtlety, which is the single most important implementation difference
between this rung and the last, and the thing I must get exactly right. At rung one I kept the mix
expression and let the harness supply the small CFG++ scale through `cfg_guidance`. But standard CFG
needs a *moderately high* scale to do its sharpening — the `w` in roughly `[5, 30]` band — and the
harness, by default, passes in the *small* CFG++ value. If I simply renoised with `eps_g` while
still using the harness's small scale, I would get neither method: a barely-guided sampler that
renoises with a barely-guided noise. So the standard-CFG fill must *override* the incoming scale and
fix it at the value CFG actually needs. The scaffold edit makes exactly this choice: at the top of
`sample()` it sets `cfg_guidance = 7.5`, ignoring whatever the harness passed, and then forms
`noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)` at that high scale, Tweedie-denoises
with `noise_pred`, and **renoises with `noise_pred`** (the guided noise) rather than `noise_uc`.
That is the complete delta from rung one: two changes, not one — hardcode the scale to 7.5, and swap
the renoise noise from `noise_uc` back to `noise_pred`. Both are necessary; the guided renoise is
pointless at a tiny scale, and the high scale is what makes the sharpening strong enough to move FID.

I want to be careful not to over-tell this rung. The classifier-free derivation has a whole story
about training one network with the condition randomly nulled, about the implicit classifier not
being the gradient of any real classifier so the adversarial-attack worry dissolves, about getting
the unconditional branch for free. None of that training-time machinery is mine to touch here: the
weights are frozen, `predict_noise` already returns both `eps_uc` and `eps_c` from one batched call,
and the only slot I fill is the per-step combination and which noise drives each half of the DDIM
step. So the part of standard CFG that is *live* in this scaffold is exactly the inference-time rule:
mix at a high scale, denoise with the mix, renoise with the mix. The 7.5 is the canonical
high-quality CFG scale and I take it as given rather than sweeping it, because the harness pins the
seed and the prompt set and I am comparing rules, not tuning a single rule's scalar.

For the SDXL variant the contract is identical in `reverse_process()`: index `at =
alphas_cumprod[t]`, `at_next = alphas_cumprod[t - skip]`, override `cfg_guidance = 7.5`, get
`noise_uc, noise_c` from `predict_noise` with the dual prompt embeddings, denoise with
`noise_pred`, and renoise with `noise_pred`. Same two changes from rung one, same two predictions per
step, same fixed NFE budget. The full scaffold module for both variants is in the answer.

The falsifiable expectation, stated against rung one's measured numbers. If under-commitment is
really why CFG++ landed at the top of the FID range, then putting the guided noise back into the
renoise at the high CFG scale should *lower* FID on all three variants below CFG++'s 23.99 / 24.89 /
25.88 — the harder push pulling the generated distribution onto the prompt-consistent modes the
reference set occupies. I expect the largest gain exactly where CFG++ was weakest, SDXL at 25.88,
because that is where the unconditional renoise had the most room to under-commit. The risk is the
mirror image of rung one: if 7.5 over-sharpens — collapsing modes and over-saturating hard enough to
*distort* the distribution rather than concentrate it — FID could fail to improve or even regress,
and CLIP would climb while FID stalls. So the clean test is simply whether standard CFG's three FID
numbers come in under CFG++'s three. If they do, the lesson is that on this benchmark the guided
renoise beats the manifold-safe one, and the next question becomes whether I can keep standard CFG's
harder push while clawing back *some* of CFG++'s on-manifold cleanliness — the renoise choice and
the high-noise early steps being the two obvious levers to revisit.
