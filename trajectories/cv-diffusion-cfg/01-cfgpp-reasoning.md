I start the ladder at the rung the scaffold itself points to. The editable class is literally named
`BaseDDIMCFGpp` and its comment says the baseline renoises with the unconditional noise to keep the
trajectory on the data manifold — so the floor of this task is not standard CFG but its
manifold-constrained cousin, CFG++. Before I fill the stub I want to be sure I understand *why* this
is the right place to begin and exactly which one line of the DDIM step it changes, because
everything later on the ladder is a perturbation of this fill and I do not want to build on a floor I
have not actually derived.

What are my actual options for filling this one stub? The contract lets me change three things and only
these three: how the two predictions combine into the denoised estimate, which prediction drives the
renoise, and how those choices vary with time. So the honest menu at rung one is short. I could fill
plain conditional sampling — ignore `eps_uc`, denoise and renoise with `eps_c` alone — but that is the
weakly-aligned sampler the whole task exists to improve on, so it is a non-starter as a floor. I could
fill standard CFG — form `eps_g` and drive both halves with it — which is the older, harder-pushing
rule and the obvious thing to try. Or I could fill the manifold-constrained version the class is named
for, which drives the denoise with a mixed noise but the renoise with `eps_uc`. I deliberately do *not*
start with plain CFG, even though it is simpler to justify, for two reasons that are about sequencing
rather than taste: the scaffold class is literally `BaseDDIMCFGpp` with a comment saying the renoise
uses the unconditional noise to stay on the manifold, so the intended floor is the constrained version;
and starting from the principled sampler forces me to derive *what exactly goes wrong* with the guided
renoise before I decide whether its harder push is worth the damage. If I began at CFG I would have a
number but no mechanism. So I derive CFG++ first, understand the two off-manifold sources it removes, and
keep standard CFG in reserve as the explicit next rung if the on-manifold caution turns out to cost me on
the metric.

Let me restate the failure that motivates starting above plain CFG, and make it quantitative rather
than folkloric. I have one frozen U-Net I can call two ways per step, `eps_uc = eps_theta(z_t, empty)`
and `eps_c = eps_theta(z_t, c)`, and standard CFG forms the guided noise `eps_g = eps_uc + w (eps_c -
eps_uc)` and plugs it into the DDIM step. A DDIM step is "denoise then renoise." Writing `a_t` for the
cumulative signal rate `bar_alpha_t`, the forward relation is `z_t = sqrt(a_t) x + sqrt(1-a_t) eps`, so
Tweedie's formula inverts it: the posterior-mean estimate of the clean signal is `x_hat = (z_t -
sqrt(1-a_t) eps) / sqrt(a_t)`, a point that lands on the clean data manifold `M`. Then the renoise
lifts it back to the next, slightly-less-noisy level: `z_{t-1} = sqrt(a_{t-1}) x_hat + sqrt(1-a_{t-1})
eps`. CFG substitutes `eps_g` for the noise in *both* of those lines, and that double substitution is
where the damage is. I want to see the size of the damage before I fix it.

Take the denoise first. With `eps_g = eps_uc + w (eps_c - eps_uc)` plugged into Tweedie, linearity
gives `x_hat_g = x_hat_uc - w * sqrt(1-a_t)/sqrt(a_t) * (eps_c - eps_uc)`. The conditional denoised
estimate alone is `x_hat_c = x_hat_uc - sqrt(1-a_t)/sqrt(a_t) (eps_c - eps_uc)`, so the coefficient on
the difference is exactly `x_hat_uc - x_hat_c`, and substituting collapses the guided denoised estimate
to an affine combination of the two endpoints: `x_hat_g = (1-w) x_hat_uc + w x_hat_c`. Stare at the
weights. If `w` were in `[0,1]` this would be an *interpolation* — a point on the segment between
`x_hat_uc` and `x_hat_c`, and because `M` is locally piecewise-linear, that segment stays on `M`. But
the useful CFG regime is `w` in `[5,30]`. At the canonical `w = 7.5` the coefficients are literally
`(1-w, w) = (-6.5, 7.5)`: the estimate sits `6.5` segment-lengths *past* `x_hat_c`, off the end of the
linear piece, off `M`. That is not a subtle drift; it is an extrapolation more than six times the
length of the segment I was supposed to stay on.

And it is worst exactly where the model is least able to absorb it. The gap between the two endpoints
is `x_hat_uc - x_hat_c = (sqrt(1-a_t)/sqrt(a_t))(eps_c - eps_uc)`, and that prefactor `sqrt(1-a_t)/
sqrt(a_t)` explodes at high noise: at `a_t = 0.5` it is `1`, at `a_t = 0.1` it is `3`, and at the very
first DDIM step where `a_t` can be around `0.005` it is roughly `14.1`. So near the start of sampling
the two denoised endpoints are already `~14x ||eps_c - eps_uc||` apart, and CFG then throws the estimate
`6.5` of *those* lengths beyond `x_hat_c`. The off-manifold overshoot in `x_hat` space scales like
`(w-1) * sqrt(1-a_t)/sqrt(a_t) * ||eps_c - eps_uc||`, largest at the noisy source end. If `M` has local
curvature `kappa`, the distance from `M` of a point pushed a tangential distance `d` past the segment
grows like `kappa d^2` — quadratic in the overshoot, hence quadratic in `w`. I cannot measure `kappa`,
but the scaling is the point: the extrapolation penalty is not linear in the guidance I dial up, it is
roughly quadratic, which is why cranking `w` produces the sudden early-step color shift and the
saturated, posterized look people report. Source one of the trouble: `w > 1` turns interpolation into
extrapolation, and the extrapolation is geometrically worst at the high-noise start.

Now the renoise line, which is a second, independent insult. CFG renoises with `eps_g` too: `z_{t-1} =
sqrt(a_{t-1}) x_hat_g + sqrt(1-a_{t-1}) eps_g`. The first term already carried an off-manifold
`x_hat_g`, but even setting that aside, the noise added back to lift the point to the next noisy
manifold is itself the *guided, extrapolated* direction `eps_g`, not the plain unconditional `eps_uc`.
The transition is being driven by an off-manifold noise vector. Two distinct off-manifold sources, then:
the `w>1` extrapolation in the denoise, and the guided noise in the renoise. The whole point of CFG++
is to fix both with a single change to the renoise, and to make the scale interpretable in `[0,1]`
instead of a magnification factor I crank to 12.

Before I invent anything, is there a textbook route that sidesteps the extrapolation? The two classical
guidance recipes are classifier guidance and classifier-free guidance. Classifier guidance steers the
score with the gradient of a separately trained, noise-aware classifier `p(c|z_t)`, `score <- score + s
grad_z log p(c|z_t)`. But the substrate here is a frozen latent-diffusion stack with exactly the U-Net
and VAE weights and the tokenizer — there is no noise-aware classifier in it, and even if I trained one,
evaluating its gradient at each step is one more forward-and-backward pass, an extra network evaluation
the two-NFE budget forbids. So classifier guidance is doubly unavailable: no classifier, and no budget
for one. Classifier-free guidance is available — it is exactly the `eps_c`/`eps_uc` pair I already have
— but it *is* the rule whose extrapolation I just diagnosed. So neither textbook recipe gives me an
on-manifold sampler for free; I have to build the constraint into the step myself.

How do I fix this from first principles rather than by hand-tuning `w`? The fact that keeps nagging me
is that the *unconditional* DDIM step is clean and well-behaved — its renoise uses `eps_uc`, it stays
on the manifold, and it inverts cleanly. So instead of thinking of guidance as "amplify the conditional
score," which is exactly what produced the `w>1` extrapolation, I reframe it as: run the well-behaved
unconditional sampler, but at each step nudge the *denoised estimate* toward satisfying the text. That
is an optimization on the manifold, `min_{x in M} l(x)`, for a loss `l` that measures "this clean image
does not match the prompt." That shape is exactly a diffusion inverse-problem solver, where a
data-consistency loss is minimized while the diffusion process keeps the iterate on the right manifold;
I will borrow that machinery.

The natural `l` for "match the text" is the text-conditioned score-matching loss the network was
trained on — noise `x` to level `t`, ask the conditional network to predict the noise, penalize the
residual: `l(x) = || eps_theta(sqrt(a_t) x + sqrt(1-a_t) eps, c) - eps ||^2`. Minimizing this over clean
`x` pushes `x` toward images the conditional model finds typical for the prompt, a direct target for
alignment unlike CFG's indirect sharpened-posterior route. Now I have to be honest about how expensive
its gradient is, because that cost decides the whole method. The honest gradient of a diffusion loss
carries the U-Net Jacobian `d eps_theta / d x`, a `D x D` operator where `D` is the latent dimension.
For SD v1.5 / v2 the latent is `4 x 64 x 64`, so `D = 16384`; for SDXL it is `4 x 128 x 128`, so `D =
65536`. Even a single Jacobian-vector product means one backward pass through the frozen U-Net, which
is one extra network evaluation per step. The contract is explicit that "exactly two predictions are
available per step from one batched network call" and "no extra neural function evaluations are
allowed." So the honest gradient is simply off-budget — it would double my NFE. That constraint, not a
matter of taste, forces the Jacobian-free route: the decomposed-diffusion / score-distillation
observation says I can drop the Jacobian and take the gradient *with respect to the denoised estimate
itself*, `grad_{x_hat} l(x_hat)`, a one-step projected gradient on the tangent space at `x_hat`, with
no backprop through the network. So one reverse step becomes

  `z_{t-1} = sqrt(a_{t-1}) ( x_hat_uc - gamma_t grad_{x_hat} l(x_hat_uc) ) + sqrt(1-a_{t-1}) eps_uc`.

Notice the renoise term in this template is `eps_uc`, not anything guided — because the template is
solving the *unconditional* PF-ODE and the conditioning enters *only* through the data-consistency
gradient on the denoised estimate. The text never touches the renoise. That is not a heuristic swap I
am choosing; it is what the optimization-on-the-manifold view forces, and it is precisely the fix for
source two.

One design point I have to settle inside this template is how many gradient steps per reverse step. A
true minimizer of `l(x)` over `M` would run an inner loop — several gradient steps, re-evaluating the
loss each time. But `l` is built from `eps_c`, and every fresh evaluation of the gradient at a moved
iterate needs a new `eps_c = eps_theta(., c)` call, which is another network evaluation. With the budget
pinned at two predictions per step I get exactly *one* gradient step per reverse step, and I let the
outer diffusion iteration — the sequence of denoise/renoise transitions across `t` — do the annealing
an inner loop would otherwise do. This is the decomposed-diffusion discipline: one cheap
projected-gradient step on the denoised estimate, then let the renoise carry me to the next noise level
where I get a fresh prediction. So the single step size `gamma_t` is the only free knob the optimization
introduces, and I am about to watch it fold into the interpolation scalar and disappear.

Now compute the gradient step and watch the denoise half too, because I need the loss in closed form to
know what `grad_{x_hat} l` actually is. Reduce `l`. From the forward relation `z_t = sqrt(a_t) x +
sqrt(1-a_t) eps` and the conditional Tweedie estimate `x_hat_c = (z_t - sqrt(1-a_t) eps_c)/sqrt(a_t)`, I
can eliminate `z_t`: rearranging the Tweedie line gives `z_t = sqrt(a_t) x_hat_c + sqrt(1-a_t) eps_c`,
and equating with the forward relation, `sqrt(a_t)(x - x_hat_c) + sqrt(1-a_t)(eps - eps_c) = 0`, so
`eps_c - eps = (sqrt(a_t)/sqrt(1-a_t))(x - x_hat_c)`. Substituting into the residual, `l(x) = ||eps_c -
eps||^2 = (a_t/(1-a_t)) || x - x_hat_c ||^2`, a clean quadratic in `x`. Its gradient is `grad_x l =
(2 a_t/(1-a_t)) (x - x_hat_c)`, and evaluated at the unconditional estimate `x = x_hat_uc` it is
`(2 a_t/(1-a_t)) (x_hat_uc - x_hat_c)`. Plug it into the denoised half: `x_hat_uc - gamma_t (2 a_t/
(1-a_t)) (x_hat_uc - x_hat_c) = x_hat_uc - lambda (x_hat_uc - x_hat_c) = (1-lambda) x_hat_uc + lambda
x_hat_c`, collapsing all the prefactors into a single scalar `lambda := 2 a_t gamma_t / (1-a_t)`. The
denoised estimate is now `x_hat_cfgpp = (1-lambda) x_hat_uc + lambda x_hat_c`. Same affine *shape* as
CFG, but the coefficient is `lambda`, a step size I keep in `[0,1]`. With `lambda` in `[0,1]` this is an
honest interpolation — a convex combination of two on-manifold endpoints, on the segment, hence on `M`.
Source one is fixed: I never extrapolate past the conditional endpoint, so the denoised estimate cannot
shoot off the manifold, and `lambda` now has an interpretable meaning (`0` = unconditional, `1` =
conditional, in between a blend) instead of being a magnification factor.

I want to verify that the noise-mixing rewrite of this really is identical, not just similar, because
the whole implementation hinges on it. Define `eps_cfgpp = eps_uc + lambda (eps_c - eps_uc)` — identical
in form to CFG but with `lambda` in place of `w`. Claim: the Tweedie estimate of this mixed noise equals
the interpolated `x_hat_cfgpp`. Compute it directly: `(z_t - sqrt(1-a_t) eps_cfgpp)/sqrt(a_t) =
x_hat_uc - lambda (sqrt(1-a_t)/sqrt(a_t)) (eps_c - eps_uc)`. And I already have `x_hat_uc - x_hat_c =
(sqrt(1-a_t)/sqrt(a_t))(eps_c - eps_uc)`, so `(sqrt(1-a_t)/sqrt(a_t))(eps_c - eps_uc) = x_hat_uc - x_hat_c`, giving
`x_hat_uc - lambda (x_hat_uc - x_hat_c) = (1-lambda) x_hat_uc + lambda x_hat_c`. That is exactly
`x_hat_cfgpp`. The two views coincide, verified line by line. So the step is: denoise with the *mixed*
noise, renoise with the *unconditional* noise. Side by side with CFG: CFG forms `eps_g`,
Tweedie-denoises with `eps_g`, renoises with `eps_g`; CFG++ forms `eps_cfgpp`, Tweedie-denoises with
`eps_cfgpp`, renoises with `eps_uc`. **The single line that changed is the renoise noise**, `eps_g ->
eps_uc` (and the scale moved from `w > 1` to `lambda` in `[0,1]`). It costs nothing extra — I already
compute `eps_uc` and `eps_c` every step for the mix, exactly two network evaluations, the same fixed NFE
budget the harness allows — yet it removes both off-manifold sources at once.

Two limit checks to make sure the rule degenerates sensibly. At `lambda = 0` the denoise is `x_hat_uc`
and the renoise is `eps_uc`: the step is literally the unconditional DDIM sampler, the clean baseline I
built the whole reframe around — good, guidance off means no guidance. At `lambda = 1` the denoise is
`x_hat_c`, the full conditional posterior-mean, while the renoise stays `eps_uc`: I denoise as if fully
conditioned but still lift along the clean unconditional direction, so even at maximum interpolation I
never inject guided noise into the transition. Both endpoints are exactly what the derivation says they
should be, which reassures me the collapsed scalar `lambda` is carrying the right meaning rather than
hiding a sign error.

Let me put one concrete step side by side to feel the size of the difference I am buying. Take an early
step with `a_t = 0.1`, so `sqrt(1-a_t)/sqrt(a_t) = 3`, and write `u := ||eps_c - eps_uc||` for the
magnitude of the conditional signal at that step. CFG++ at `lambda = 0.6` displaces the denoised
estimate from `x_hat_uc` by `lambda * 3 * u = 1.8 u`, landing at a point `0.6` of the way along the
segment from `x_hat_uc` to `x_hat_c` — safely inside. Standard CFG at `w = 7.5` displaces it by
`w * 3 * u = 22.5 u`, a factor of `12.5` further, and `6.5` of those segment-lengths lie *past* `x_hat_c`,
off the linear piece. Same two predictions, same step: one rule nudges the estimate a fraction of the
way to the conditional endpoint, the other hurls it more than six segment-lengths beyond it. That `12.5x`
is the concrete character of the trade I am making at rung one — I am giving up most of the raw per-step
motion toward the prompt in exchange for never leaving the segment. Whether that trade is the right one
for a distribution-matching metric is exactly what I cannot decide from the geometry alone.

There is a second reason to prefer the `eps_uc` renoise that the metric does not score but the geometry
does: DDIM inversion. DDIM inversion runs the forward map with the *unconditional* noise, so a reverse
step that also renoises with `eps_uc` shares that direction and the round-trip residual scales with the
small interpolation nudge — roughly `O(lambda)` per step. A reverse step that renoises with `eps_g`
mismatches the forward map by the full guided direction, `O(w)` per step. At `w = 7.5` against `lambda`
around `0.6` that is more than a tenfold larger per-step inversion residual, and it compounds over the
whole trajectory. Restoring invertibility is not the task score, but it is the same disease as the
saturation — off-manifold renoise — so fixing one fixes the other, and it tells me the `eps_uc` renoise
is structurally the right choice, not just the cautious one.

I want to be careful about one thing the harness exposes versus how the method is usually told, because
it is the one place I could accidentally write the wrong scale. The clean CFG++ story is "small
interpretable scale `lambda` in `[0,1]`," and the underlying derivation runs at `lambda` around `0.6`.
But the scaffold signature here is `sample(self, cfg_guidance=7.5, ...)` and the SD class reuses the
same `noise_uc + cfg_guidance * (noise_c - noise_uc)` mix as CFG — so the *value* of the scale at run
time is whatever the evaluation harness passes in, not the `7.5` default sitting in the signature. The
contract I am filling does not let me rename or rescale that argument; it only lets me decide which
predicted noise drives each half of the DDIM step. So my fill keeps the exact mix expression and changes
exactly the renoise: `zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc`. That is the literal
one-line edit, and it is faithful to CFG++ precisely because the small-scale behavior comes from the
harness's chosen `cfg_guidance` (it passes the small CFG++ scale), while the on-manifold geometry comes
from the `eps_uc` renoise I am writing. There is a small fossil in the scaffold worth naming so I do not
misread it: `zt = zt.requires_grad_()` is set at the top, a leftover from the projected-gradient framing
where the denoised estimate would be differentiated. Because I take the gradient of `l` analytically —
it collapses to the interpolation with no autograd graph ever built, and the whole loop sits inside
`torch.no_grad()` — that flag is inert at run time; it costs nothing and it changes nothing, which is
itself the concrete confirmation that the method is genuinely Jacobian-free. I am not importing any
extra machinery the scaffold does not have — no per-image null-text optimization, no coupled invertible
transforms, no time-varying schedule; just the renoise swap.

For the SDXL variant the contract is the same in `reverse_process()`: index `at = alphas_cumprod[t]` and
`at_next = alphas_cumprod[t - skip]` directly, get `noise_uc, noise_c` from `predict_noise` with the
dual prompt embeddings, denoise with the mixed `noise_pred`, and renoise with `noise_uc`. Same one-line
difference from CFG, same two predictions per step, same fixed NFE budget. The full scaffold module for
both variants is in the answer.

So the falsifiable expectation, stated against what comes next on the ladder. CFG++ should be a
*conservative, on-manifold* sampler: by construction it removes the off-manifold extrapolation and the
guided renoise, so I expect it to be stable and artifact-free, but at a small interpolation scale it
deliberately does *not* push as hard toward the prompt as a high-`w` guided renoise would. On the task's
FID-scored COCO setup, that conservatism is a real risk, and I can name the mechanism precisely: FID
rewards matching the reference image *distribution* — a set of real, sharp images — and a sampler that
renoises unconditionally and interpolates gently keeps the conditional push confined to the small
`lambda` nudge inside the denoise, so its generated distribution may sit slightly *farther* from the
reference statistics than a sampler that renoises with the guided noise and drives harder. The `eps_uc`
renoise, which is the whole source of the on-manifold cleanliness, is also the source of the
under-commitment: the per-step motion toward the prompt cannot compound because the lift is always along
the clean unconditional direction. I expect this to bite hardest on the highest-resolution, sharpest
variant, SDXL, where an under-committed sampler has the most room to drift the sample statistics away
from the reference. I have no measured numbers yet — the per-variant FID table will tell me whether the
unconditional renoise actually beats or trails the guided renoise. I am starting here because the
scaffold names this the baseline and because it is the principled floor; what I am watching is whether
that floor is also competitive on the one axis the metric scores. And if it trails, the immediate next
move on the ladder is obvious from the diagnosis: the disease would be the unconditional renoise's
under-commitment, and the direct remedy is to put the guided noise back into the renoise — standard CFG
— and see whether the harder push lowers FID even at the cost of the manifold guarantees I just bought.
