I start where the scaffold itself points. The editable class is literally named
`BaseDDIMCFGpp` and its comment says the baseline renoises with the unconditional noise to keep the
trajectory on the data manifold — so the floor of this task is not standard CFG but its
manifold-constrained cousin, CFG++. The contract lets me change exactly three things: how the two
predictions combine into the denoised estimate, which prediction drives the renoise, and how those
choices vary with time. Plain conditional sampling (denoise and renoise with `eps_c` alone) is the
weakly-aligned sampler the whole task exists to improve on, so it is a non-starter as a floor. That
leaves standard CFG and CFG++, the constrained version the class is named for. I derive CFG++ first:
starting from the principled sampler forces me to understand *what exactly goes wrong* with the
guided renoise before deciding whether its harder push is worth the damage, and standard CFG stays
in reserve as the fallback if the on-manifold caution costs me on the metric.

First, the failure that motivates starting above plain CFG. I have one frozen U-Net I
can call two ways per step, `eps_uc = eps_theta(z_t, empty)` and `eps_c = eps_theta(z_t, c)`, and
standard CFG forms the guided noise `eps_g = eps_uc + w (eps_c - eps_uc)` and plugs it into the DDIM
step. A DDIM step is "denoise then renoise." Writing `a_t` for the cumulative signal rate
`bar_alpha_t`, the forward relation is `z_t = sqrt(a_t) x + sqrt(1-a_t) eps`, so Tweedie's formula
inverts it: the posterior-mean estimate of the clean signal is `x_hat = (z_t - sqrt(1-a_t) eps) /
sqrt(a_t)`, a point that lands on the clean data manifold `M`. Then the renoise lifts it to the
next, slightly-less-noisy level: `z_{t-1} = sqrt(a_{t-1}) x_hat + sqrt(1-a_{t-1}) eps`. CFG
substitutes `eps_g` for the noise in *both* of those lines, and that double substitution is where
the damage is.

Take the denoise first. With `eps_g` plugged into Tweedie, linearity gives `x_hat_g = x_hat_uc - w *
sqrt(1-a_t)/sqrt(a_t) * (eps_c - eps_uc)`. The conditional denoised estimate alone is `x_hat_c =
x_hat_uc - sqrt(1-a_t)/sqrt(a_t) (eps_c - eps_uc)`, so the coefficient on the difference is exactly
`x_hat_uc - x_hat_c`, and substituting collapses the guided estimate to an affine combination of the
two endpoints: `x_hat_g = (1-w) x_hat_uc + w x_hat_c`. Stare at the weights. If `w` were in `[0,1]`
this would be an *interpolation* — a point on the segment between `x_hat_uc` and `x_hat_c`, and
because `M` is locally piecewise-linear, that segment stays on `M`. But the useful CFG regime is `w`
in `[5,30]`. At the canonical `w = 7.5` the coefficients are `(1-w, w) = (-6.5, 7.5)`: the estimate
sits `6.5` segment-lengths *past* `x_hat_c`, off the end of the linear piece, off `M`. That is an
extrapolation more than six times the length of the segment I was supposed to stay on.

And it is worst exactly where the model is least able to absorb it. The gap between the endpoints is
`x_hat_uc - x_hat_c = (sqrt(1-a_t)/sqrt(a_t))(eps_c - eps_uc)`, and that prefactor explodes at high
noise: at `a_t = 0.5` it is `1`, at `a_t = 0.1` it is `3`, and at the very first DDIM step where
`a_t` can be around `0.005` it is roughly `14.1`. So near the start of sampling the two denoised
endpoints are already `~14x ||eps_c - eps_uc||` apart, and CFG throws the estimate `6.5` of *those*
lengths beyond `x_hat_c`. If `M` has local curvature `kappa`, the distance from `M` of a point
pushed a tangential distance `d` past the segment grows like `kappa d^2` — quadratic in the
overshoot, hence roughly quadratic in `w`. I cannot measure `kappa`, but the scaling is the point:
the extrapolation penalty is not linear in the guidance I dial up, which is why cranking `w` produces
the sudden early-step color shift and the saturated, posterized look. Source one: `w > 1` turns
interpolation into extrapolation, geometrically worst at the high-noise start.

Now the renoise line, a second, independent insult. CFG renoises with `eps_g` too: even setting the
already-off-manifold `x_hat_g` aside, the noise added back to lift the point to the next noisy
manifold is itself the *guided, extrapolated* direction `eps_g`, not the plain unconditional
`eps_uc`. Two distinct off-manifold sources, then: the `w>1` extrapolation in the denoise, and the
guided noise in the renoise. CFG++ fixes both with a single change to the renoise, and makes the
scale interpretable in `[0,1]` instead of a magnification factor.

Is there a textbook route that sidesteps the extrapolation? The classical alternative to CFG is
classifier guidance, `score <- score + s grad_z log p(c|z_t)`. But the substrate here has only the
U-Net, VAE, and tokenizer — there is no noise-aware classifier, and even if I trained one, its
gradient at each step is an extra forward-and-backward pass, which the two-NFE budget forbids. So
classifier guidance is doubly unavailable. Classifier-free guidance is exactly the `eps_c`/`eps_uc`
pair I already have — but it *is* the rule whose extrapolation I just diagnosed. Neither recipe hands
me an on-manifold sampler; I have to build the constraint into the step myself.

The fact that keeps nagging me is that the *unconditional* DDIM step is clean and well-behaved: its
renoise uses `eps_uc`, it stays on the manifold, and it inverts cleanly. So instead of "amplify the
conditional score," which is what produced the `w>1` extrapolation, I reframe guidance as: run the
well-behaved unconditional sampler, but at each step nudge the *denoised estimate* toward satisfying
the text. That is an optimization on the manifold, `min_{x in M} l(x)`, for a loss `l` that measures
"this clean image does not match the prompt" — exactly the shape of a diffusion inverse-problem
solver, where a data-consistency loss is minimized while the diffusion process keeps the iterate on
the right manifold.

The natural `l` for "match the text" is the text-conditioned score-matching loss the network was
trained on: `l(x) = || eps_theta(sqrt(a_t) x + sqrt(1-a_t) eps, c) - eps ||^2`. Minimizing it over
clean `x` pushes `x` toward images the conditional model finds typical for the prompt — a direct
target for alignment, unlike CFG's indirect sharpened-posterior route. But I have to be honest about
its gradient, because that cost decides the whole method. The honest gradient of a diffusion loss
carries the U-Net Jacobian `d eps_theta / d x`, a `D x D` operator (`D = 16384` for SD v1.5/v2 at `4
x 64 x 64`, `D = 65536` for SDXL at `4 x 128 x 128`). Even one Jacobian-vector product means one
backward pass through the frozen U-Net — one extra network evaluation per step. The contract is
explicit: exactly two predictions per step, no extra NFE. So the honest gradient is off-budget, and
that constraint, not taste, forces the Jacobian-free route: drop the Jacobian and take the gradient
*with respect to the denoised estimate itself*, `grad_{x_hat} l(x_hat)`, a one-step projected
gradient on the tangent space at `x_hat`, no backprop through the network. One reverse step becomes

  `z_{t-1} = sqrt(a_{t-1}) ( x_hat_uc - gamma_t grad_{x_hat} l(x_hat_uc) ) + sqrt(1-a_{t-1}) eps_uc`.

The renoise term is `eps_uc`, not anything guided — because the template solves the *unconditional*
PF-ODE and the conditioning enters *only* through the data-consistency gradient on the denoised
estimate. That is not a heuristic swap; it is what the optimization-on-the-manifold view forces, and
it is precisely the fix for source two. And the budget also pins the inner loop to a single gradient
step per reverse step: every fresh gradient at a moved iterate would need a new `eps_c` call, so I
get exactly one, and I let the outer diffusion iteration do the annealing an inner loop would
otherwise do. The single step size `gamma_t` is the only free knob.

Reduce `l` to closed form. From the forward relation and the conditional Tweedie estimate `x_hat_c =
(z_t - sqrt(1-a_t) eps_c)/sqrt(a_t)`, eliminate `z_t`: rearranging gives `z_t = sqrt(a_t) x_hat_c +
sqrt(1-a_t) eps_c`, and equating with the forward relation, `eps_c - eps = (sqrt(a_t)/sqrt(1-a_t))(x
- x_hat_c)`. Substituting into the residual, `l(x) = (a_t/(1-a_t)) || x - x_hat_c ||^2`, a clean
quadratic. Its gradient at `x = x_hat_uc` is `(2 a_t/(1-a_t)) (x_hat_uc - x_hat_c)`. Plug into the
denoise half: `x_hat_uc - lambda (x_hat_uc - x_hat_c) = (1-lambda) x_hat_uc + lambda x_hat_c`, with
all prefactors collapsed into `lambda := 2 a_t gamma_t / (1-a_t)`. Same affine *shape* as CFG, but
the coefficient is now a step size I keep in `[0,1]`. With `lambda` in `[0,1]` this is an honest
convex combination of two on-manifold endpoints, on the segment, hence on `M`: source one fixed, and
`lambda` now means `0` = unconditional, `1` = conditional, in between a blend, instead of a
magnification factor.

The implementation is a noise mix, and the two views coincide.
Define `eps_cfgpp = eps_uc + lambda (eps_c - eps_uc)` — identical in form to CFG with `lambda` for
`w`. Its Tweedie estimate is `(z_t - sqrt(1-a_t) eps_cfgpp)/sqrt(a_t) = x_hat_uc - lambda
(sqrt(1-a_t)/sqrt(a_t)) (eps_c - eps_uc)`, and since `(sqrt(1-a_t)/sqrt(a_t))(eps_c - eps_uc) =
x_hat_uc - x_hat_c`, this is `(1-lambda) x_hat_uc + lambda x_hat_c` — exactly `x_hat_cfgpp`. So the
step is: denoise with the *mixed* noise, renoise with the *unconditional* noise. Side by side, CFG
forms `eps_g`, denoises with `eps_g`, renoises with `eps_g`; CFG++ forms `eps_cfgpp`, denoises with
`eps_cfgpp`, renoises with `eps_uc`. **The single line that changed is the renoise noise**, `eps_g
-> eps_uc` (and the scale moved from `w > 1` to `lambda` in `[0,1]`). It costs nothing extra — the
same two predictions per step — yet removes both off-manifold sources at once. The endpoints degrade
sanely: `lambda = 0` is the unconditional DDIM sampler, `lambda = 1` denoises to the full conditional
posterior-mean while still lifting along `eps_uc`.

The trade is stark. At an early step with `a_t = 0.1` (`sqrt(1-a_t)/sqrt(a_t) = 3`) and
`u := ||eps_c - eps_uc||`, CFG++ at `lambda = 0.6` displaces the denoised estimate by `1.8 u`,
safely inside the segment, while standard CFG at `w = 7.5` displaces it by `22.5 u`, with `6.5`
segment-lengths *past* `x_hat_c`. I give up most of the raw per-step motion toward the prompt in
exchange for never leaving the segment — and whether that trade is right for a distribution-matching
metric is exactly what the geometry cannot decide.

There is a second reason to prefer the `eps_uc` renoise that the metric does not score but the
geometry does: DDIM inversion runs the forward map with the *unconditional* noise, so a reverse step
that also renoises with `eps_uc` shares that direction and the round-trip residual scales with the
small interpolation nudge, `O(lambda)` per step, versus `O(w)` for an `eps_g` renoise — a more than
tenfold larger per-step residual at `w = 7.5` against `lambda ~ 0.6`, compounding over the
trajectory. Restoring invertibility is not the task score, but it is the same disease as the
saturation — off-manifold renoise — so the two failures share a fix.

One scaffold-specific point, because it is the one place I could write the wrong scale. The clean
CFG++ story runs at `lambda` around `0.6`, but the signature here is `sample(self, cfg_guidance=7.5,
...)` and the class reuses the same `noise_uc + cfg_guidance * (noise_c - noise_uc)` mix as CFG — so
the *value* of the scale at run time is whatever the evaluation harness passes in, not the `7.5`
default in the signature. The contract does not let me rename or rescale that argument; it only lets
me decide which predicted noise drives each half. So my fill keeps the exact mix and changes exactly
the renoise: `zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc`. That is faithful to CFG++
precisely because the small-scale behavior comes from the harness's chosen `cfg_guidance` while the
on-manifold geometry comes from the `eps_uc` renoise I am writing. The `zt = zt.requires_grad_()` at
the top is a fossil from the projected-gradient framing; since the gradient of `l` collapses to the
interpolation analytically, inside `torch.no_grad()`, the flag is inert. No per-image null-text
optimization, no coupled invertible transforms, no time-varying schedule; just the renoise swap.

For the SDXL variant the contract is the same in `reverse_process()`: index `at = alphas_cumprod[t]`
and `at_next = alphas_cumprod[t - skip]`, get `noise_uc, noise_c` from `predict_noise` with the dual
prompt embeddings, denoise with the mixed `noise_pred`, and renoise with `noise_uc`. The full
scaffold module for both variants is in the answer.

CFG++ is on-manifold and artifact-free by construction, but at a small interpolation scale it does
not push as hard toward the prompt as a high-`w` guided renoise would, and on the FID-scored COCO
setup that conservatism is a real risk: FID rewards matching a reference set of real, sharp images,
and a sampler whose conditional push is confined to the small `lambda` nudge inside the denoise —
never compounding, because the lift is always along the clean unconditional direction — may sit
slightly *farther* from the reference than one that renoises with the guided noise. The per-variant
FID table will tell me whether the unconditional renoise beats or trails the guided one. If it
trails, the next move is named directly: put the guided noise back into the renoise, standard CFG,
and see whether the harder push lowers FID even at the cost of the manifold guarantees I just bought.
