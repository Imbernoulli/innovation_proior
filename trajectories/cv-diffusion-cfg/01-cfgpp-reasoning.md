I start the ladder at the rung the scaffold itself points to. The editable class is literally named
`BaseDDIMCFGpp` and its comment says the baseline renoises with the unconditional noise to keep the
trajectory on the data manifold — so the floor of this task is not standard CFG but its
manifold-constrained cousin, CFG++. Before I fill the stub I want to be sure I understand *why* this
is the right place to begin and exactly what one line of the DDIM step it changes, because
everything later on the ladder is a perturbation of this fill.

Let me restate the failure that motivates starting above plain CFG. I have one frozen U-Net I can
call two ways, `eps_uc = eps_theta(z_t, empty)` and `eps_c = eps_theta(z_t, c)`, and standard CFG
forms the guided noise `eps_g = eps_uc + w (eps_c - eps_uc)` and plugs it into the DDIM step. A DDIM
step is "denoise then renoise." Using Tweedie's formula the denoised estimate of the clean signal
is `x_hat = (z_t - sqrt(1-a_t) eps) / sqrt(a_t)`, writing `a_t` for the cumulative signal rate
`bar_alpha_t`; that estimate is the posterior mean of `x_0` and lands on the clean data manifold
`M`. Then the renoise lifts it back up to the next, slightly-less-noisy level: `z_{t-1} =
sqrt(a_{t-1}) x_hat + sqrt(1-a_{t-1}) eps`. CFG substitutes `eps_g` for the noise in *both* of those
lines, and that double substitution is exactly where the damage is.

Look at the denoise first. With `eps_g = eps_uc + w (eps_c - eps_uc)` plugged into Tweedie,
linearity gives `x_hat_g = x_hat_uc - w * sqrt(1-a_t)/sqrt(a_t) * (eps_c - eps_uc)`. The conditional
denoised estimate alone is `x_hat_c = x_hat_uc - sqrt(1-a_t)/sqrt(a_t) (eps_c - eps_uc)`, so the
coefficient on the difference is exactly `x_hat_uc - x_hat_c`, and substituting collapses the
denoised estimate to an affine combination of the two endpoints: `x_hat_g = (1-w) x_hat_uc + w
x_hat_c`. Stare at the weights. If `w` were in `[0,1]` this would be an *interpolation* — a point on
the segment between `x_hat_uc` and `x_hat_c`, and because the clean manifold is locally
piecewise-linear, that segment stays on `M`. But the useful CFG regime is `w` in `[5,30]`, so
`1-w` is hugely negative and this is an *extrapolation*, shot far past `x_hat_c` off the end of the
linear piece, off `M`. That off-manifold denoised estimate is the sudden early-step shift and the
color saturation people see. Source one of the trouble: `w > 1` turns interpolation into
extrapolation.

Now the renoise line. CFG renoises with `eps_g` too: `z_{t-1} = sqrt(a_{t-1}) x_hat_g +
sqrt(1-a_{t-1}) eps_g`. The first term already carried an off-manifold `x_hat_g`, but there is a
second, separate insult: the noise added back to lift the point to the next noisy manifold is itself
the *guided, extrapolated* direction `eps_g`, not the plain unconditional `eps_uc`. Even setting
aside the bad denoised estimate, the transition is being driven by an off-manifold noise vector.
Two distinct off-manifold sources, then: the `w>1` extrapolation in the denoise, and the guided
noise in the renoise. The whole point of CFG++ is to fix both with a single change to the renoise,
and to make the scale interpretable in `[0,1]` instead of a magnification factor I crank to 12.

How do I fix this from first principles rather than by hand-tuning `w`? The fact that keeps nagging
me is that the *unconditional* DDIM step is clean and well-behaved — its renoise uses `eps_uc`, it
stays on the manifold, and it inverts cleanly. So instead of thinking of guidance as "amplify the
conditional score," which is exactly what produced the `w>1` extrapolation, I reframe it as: run the
well-behaved unconditional sampler, but at each step nudge the *denoised estimate* toward satisfying
the text. That is an optimization on the manifold, `min_{x in M} l(x)`, for a loss `l` that measures
"this clean image does not match the prompt." That shape is exactly a diffusion inverse-problem
solver, where a data-consistency loss is minimized while the diffusion process keeps the iterate on
the right manifold; I will borrow that machinery.

The natural `l` for "match the text" is the text-conditioned score-matching loss the network was
trained on — noise `x` to level `t`, ask the conditional network to predict the noise, penalize the
residual: `l(x) = || eps_theta(sqrt(a_t) x + sqrt(1-a_t) eps, c) - eps ||^2`. Minimizing this over
clean `x` pushes `x` toward images the conditional model finds typical for the prompt, a direct
target for alignment unlike CFG's indirect sharpened-posterior route. The honest gradient of such a
diffusion loss carries the U-Net Jacobian, which is expensive and badly conditioned; the
decomposed-diffusion-sampling view (and the score-distillation observation) says I can drop that
Jacobian and take the gradient *with respect to the denoised estimate itself*,
`grad_{x_hat} l(x_hat)`, a one-step projected gradient on the tangent space at `x_hat`, with no
backprop through the network. So one reverse step becomes

  `z_{t-1} = sqrt(a_{t-1}) ( x_hat_uc - gamma_t grad_{x_hat} l(x_hat_uc) ) + sqrt(1-a_{t-1}) eps_uc`.

Notice the renoise term in this template is `eps_uc`, not anything guided — because the template is
solving the *unconditional* PF-ODE and the conditioning enters *only* through the data-consistency
gradient on the denoised estimate. The text never touches the renoise. That is not a heuristic swap
I am choosing; it is what the optimization-on-the-manifold view forces, and it is precisely the fix
for source two.

Now compute the gradient step and watch the denoise half too. Reduce `l`: from the forward relation
`x_t = sqrt(a_t) x + sqrt(1-a_t) eps` and the conditional Tweedie estimate, the residual is
`eps_c - eps = sqrt(a_t)/sqrt(1-a_t) (x - x_hat_c)`, so `l(x) = (a_t/(1-a_t)) || x - x_hat_c ||^2`,
a clean quadratic. Its gradient at `x = x_hat_uc` is `(2 a_t/(1-a_t)) (x_hat_uc - x_hat_c)`. Plug it
into the denoised half: `x_hat_uc - gamma_t (2 a_t/(1-a_t)) (x_hat_uc - x_hat_c) = x_hat_uc +
lambda (x_hat_c - x_hat_uc)`, collapsing all prefactors into a single scalar `lambda := 2 a_t
gamma_t / (1-a_t)`. The denoised estimate is now `x_hat_cfgpp = (1-lambda) x_hat_uc + lambda
x_hat_c`. Same affine *shape* as CFG, but the coefficient is `lambda`, a step size I keep in
`[0,1]`. With `lambda` in `[0,1]` this is an honest interpolation — a convex combination of two
on-manifold endpoints, on the segment, hence on `M`. Source one is fixed: I never extrapolate past
the conditional endpoint, so the denoised estimate cannot shoot off the manifold, and `lambda` now
has an interpretable meaning (`0` = unconditional, `1` = conditional, in between a blend) instead of
being a magnification factor.

Re-expressing in noise-mixing notation shows how small the change really is. Define `eps_cfgpp =
eps_uc + lambda (eps_c - eps_uc)` — identical to CFG but with `lambda`. Then the interpolated
denoised estimate is exactly the Tweedie estimate of this mixed noise: `(z_t - sqrt(1-a_t)
eps_cfgpp)/sqrt(a_t)`. So the step is: denoise with the *mixed* noise, renoise with the
*unconditional* noise. Side by side with CFG: CFG forms `eps_g`, Tweedie-denoises with `eps_g`,
renoises with `eps_g`; CFG++ forms `eps_cfgpp`, Tweedie-denoises with `eps_cfgpp`, renoises with
`eps_uc`. **The single line that changed is the renoise noise**, `eps_g -> eps_uc` (and the scale
moved from `w > 1` to `lambda` in `[0,1]`). It costs nothing extra — I already compute `eps_uc` and
`eps_c` every step for the mix, exactly two network evaluations, the same fixed NFE budget the
harness allows — yet it removes both off-manifold sources at once.

I want to be careful about one thing the harness exposes versus how the method is usually told. The
clean CFG++ story is "small interpretable scale `lambda` in `[0,1]`," and the methods derivation
runs at `lambda` around 0.6. But the scaffold signature here is `sample(self, cfg_guidance=7.5,
...)` and the SD class reuses the same `noise_uc + cfg_guidance * (noise_c - noise_uc)` mix as CFG —
so the *value* of the scale at run time is whatever the evaluation harness passes in, not the 7.5
default in the signature. The contract I am filling does not let me rename or rescale that argument;
it only lets me decide which predicted noise drives each half of the DDIM step. So my fill keeps the
exact mix expression and changes exactly the renoise: `zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt()
* noise_uc`. That is the literal one-line edit, and it is faithful to CFG++ precisely because the
small-scale behavior comes from the harness's chosen `cfg_guidance` (it passes the small CFG++
scale), while the on-manifold geometry comes from the `eps_uc` renoise I am writing. I am not
importing any extra machinery the scaffold does not have — no per-image null-text optimization, no
coupled invertible transforms, no time-varying schedule; just the renoise swap.

For the SDXL variant the contract is the same in `reverse_process()`: index `at = alphas_cumprod[t]`
and `at_next = alphas_cumprod[t - skip]` directly, get `noise_uc, noise_c` from `predict_noise` with
the dual prompt embeddings, denoise with the mixed `noise_pred`, and renoise with `noise_uc`. Same
one-line difference from CFG, same two predictions per step. The full scaffold module for both
variants is in the answer.

So the falsifiable expectation, stated against what comes next on the ladder. CFG++ should be a
*conservative, on-manifold* sampler: by construction it removes the off-manifold extrapolation and
the guided renoise, so I expect it to be stable and artifact-free, but at a small interpolation
scale it deliberately does *not* push as hard toward the prompt as a high-`w` guided renoise would.
On the task's FID-scored COCO setup, that conservatism is a real risk: the metric rewards matching
the reference image *distribution*, and a sampler that renoises unconditionally and interpolates
gently may sit slightly *farther* from the reference statistics than a sampler that renoises with
the guided noise and drives harder. In other words, the very on-manifold caution that fixes
saturation and inversion might cost a little FID against a harder-pushing renoise. I am starting
here because the scaffold names this the baseline and because it is the principled floor; what I am
watching is whether the unconditional renoise, measured by per-variant FID, actually beats or
trails the guided renoise — and if it trails, the immediate next move on the ladder is obvious:
put the guided noise back into the renoise (standard CFG) and see whether the harder push lowers FID
even at the cost of the manifold guarantees.
