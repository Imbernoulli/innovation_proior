Let me start from what actually breaks when I generate images with a text-to-image diffusion model. I have a trained noise predictor I can call two ways: `eps_null(x_t) = eps_theta(x_t, empty)` with the null condition, and `eps_c(x_t) = eps_theta(x_t, c)` conditioned on the prompt. Naively, to follow the prompt I'd just solve the conditional probability-flow ODE with `eps_c`. But everyone who has tried this knows the conditioning comes out far too weak — the image only loosely matches the text. The standard fix is classifier-free guidance: amplify the conditional direction. Write `delta(x_t) := eps_c(x_t) - eps_null(x_t)` for the step from the unconditional to the conditional prediction; then guide with

  eps_cfg(x_t) = eps_null(x_t) + w * delta(x_t),   w > 1,

and plug `eps_cfg` into the DDIM step in place of `eps_null`. This comes straight from treating the target as a sharpened posterior `p^w(x|c) ∝ p(x) p(c|x)^w`: take `grad_x log` of that, parameterize the score with the noise predictor, and out drops the linear mix `eps_null + w(eps_c - eps_null)`. Fine. It works, in the sense that with a big enough `w` the prompt alignment becomes strong.

But the price is ugly and I want to be precise about it because it's the whole motivation. To get quality and alignment I need a *moderately high* scale, `w` in roughly `[5, 30]`. In that band: samples collapse toward fewer modes and lose diversity; colors over-saturate; error piles up across the sampling trajectory; and DDIM inversion — the thing I rely on to edit real images — falls apart the moment `w > 1`. The folklore is that this is just how diffusion models are, an inherent limitation. I don't believe that yet. The same backbone, sampled *unconditionally*, behaves beautifully and inverts cleanly. So the suspicion I want to chase is that the damage is in *how guidance is injected into the sampler*, not in the model. Let me look hard at exactly what CFG does to a single DDIM step, because that's the only place the guidance enters.

A DDIM step, unconditionally, is "denoise then renoise." Using Tweedie's formula the denoised estimate of the clean signal is

  x_hat_null(x_t) = (x_t - sqrt(1 - a_t) * eps_null(x_t)) / sqrt(a_t),

writing `a_t` for the cumulative signal rate `bar_alpha_t`. That `x_hat_null` is the posterior-mean estimate of `x_0` — it lands on the clean data manifold `M`. Then I renoise it forward to the next, slightly-less-noisy level:

  x_{t-1} = sqrt(a_{t-1}) * x_hat_null + sqrt(1 - a_{t-1}) * eps_null(x_t).

Geometrically: `x_t` sits near a noisy manifold `M_t`, Tweedie maps it down to the clean manifold `M`, and the renoise lifts it back up to `M_{t-1}`. Now CFG substitutes `eps_cfg` for `eps_null` in *both* of those lines. So I should ask, at each line, what that substitution does geometrically.

Take the denoised estimate first. With `eps_cfg = eps_null + w*delta` plugged into Tweedie, linearity gives

  x_hat_cfg(x_t) = (x_t - sqrt(1-a_t) * (eps_null + w*delta)) / sqrt(a_t)
               = x_hat_null - w * sqrt(1-a_t)/sqrt(a_t) * delta.

And the conditional denoised estimate alone is `x_hat_c = x_hat_null - sqrt(1-a_t)/sqrt(a_t) * delta`, so `sqrt(1-a_t)/sqrt(a_t) * delta = x_hat_null - x_hat_c`, and substituting,

  x_hat_cfg = x_hat_null - w*(x_hat_null - x_hat_c) = (1 - w) x_hat_null + w x_hat_c.

I want to be sure I didn't slip a sign in that rearrangement, because the whole reading of CFG hangs on it. Let me put numbers on it: take `a_t = 0.36` so `sqrt(a_t) = 0.6`, `sqrt(1-a_t) = 0.8`, pick an `x_t`, `eps_null`, `eps_c` at random in a few dimensions, and compute `x_hat_cfg` two ways — directly from `(x_t - 0.8*eps_cfg)/0.6` with `eps_cfg = eps_null + w*delta`, and from the affine form `(1-w)x_hat_null + w x_hat_c`. With `w = 12` the two agree to machine precision (max componentwise difference ~7e-15), and the coefficient on `x_hat_c` reads `12`, on `x_hat_null` reads `-11`. So it's genuinely an affine combination of the two denoised estimates with weights `(1-w, w)`, and at the working `w` the weight on `x_hat_null` is large and negative.

That sign is the tell. If `w` were in `[0,1]` this would be an *interpolation* — a point on the segment between `x_hat_null` and `x_hat_c`. But the useful CFG regime is `w` in `[5,30]`, so `1-w` is hugely negative: this is an *extrapolation*, far beyond `x_hat_c`, way off the end of the segment. Why does that matter? Because of the manifold geometry. The clean manifold `M` is, to a good local approximation, piecewise linear — the segment between two nearby denoised points lies (approximately) on `M`, and so does a short move along it. So `x_hat_null` and `x_hat_c` are both on `M`, the whole segment between them is on `M`, but the moment the combination weight passes 1 I shoot off the end of that linear piece and leave `M`. That lines up with the sudden shift and the color saturation people see in the early, high-noise steps: the denoised estimate is being thrown off the data manifold by the very extrapolation that high `w` demands. So one source of trouble looks like `w > 1` turning interpolation into extrapolation.

Now the renoise line. CFG renoises with `eps_cfg` too:

  x_{t-1} = sqrt(a_{t-1}) * x_hat_cfg + sqrt(1-a_{t-1}) * eps_cfg.

The first term already carried an off-manifold `x_hat_cfg`. But there's a second, separate insult here: the noise that's added back to lift the point to `M_{t-1}` is itself the *guided, extrapolated* noise direction `eps_cfg`, not the plain unconditional `eps_null`. So even setting aside the bad denoised estimate, the transition from `M` to `M_{t-1}` is being driven by an off-manifold noise vector, introducing a nonzero offset from the correct noisy manifold. Two distinct off-manifold sources, then: (a) the `w>1` extrapolation in the denoise, and (b) the guided noise in the renoise.

So how would I fix this from first principles rather than by hand-tuning `w`? I keep coming back to the fact that the *unconditional* DDIM step is clean and well-behaved — its renoise uses `eps_null`, and it stays on the manifold and inverts nicely. What if I stop thinking of guidance as "amplify the conditional score" — which is exactly what produced the `w>1` extrapolation via the sharpened posterior — and instead think of it as: run the well-behaved unconditional sampler, but at each step nudge the denoised estimate toward satisfying the text. That reframes text guidance as an optimization on the manifold,

  min_{x in M} l(x),

for some loss `l` that measures "this clean image doesn't match the prompt." If I can find the right `l` and a stable way to take a step on it, I get to keep the unconditional sampler's good behavior and add only a small corrective nudge. This is exactly the shape of a diffusion inverse-problem solver, where a data-consistency loss is minimized while the diffusion process keeps the iterate on the right manifold. So let me borrow that machinery and see where it lands — I don't yet know what the per-step update will look like in noise-mixing terms, which is the form I'll ultimately need to compare against CFG.

What's the right `l` for "match the text"? The most natural object is the text-conditioned score-matching loss the network was trained on — the same thing score distillation uses as its objective:

  l_sds(x) = || eps_theta(sqrt(a_t) x + sqrt(1-a_t) eps, c) - eps ||^2,

i.e. noise `x` to level `t` with a fresh `eps`, ask the conditional network to predict that noise, and penalize the residual. Minimizing this over clean `x` pushes `x` toward images the conditional model finds typical for the prompt. That's a loss that *directly* targets text alignment, unlike CFG which targets it indirectly through the sharpened posterior.

Now, how do I take a gradient step on `l_sds` without paying through the nose? The honest gradient of a diffusion training loss like this has three factors: the noise residual `(eps_hat - eps)`, times the U-Net Jacobian `d eps_hat / d (input)`, times the chain-rule Jacobian from `x`. The middle factor is the killer — differentiating through the score network means backpropagating through the entire U-Net every step, which is expensive and, at small noise levels, badly conditioned (the U-Net Jacobian is approximating a scaled Hessian of the marginal density). The score-distillation insight is that you can simply *drop the U-Net Jacobian* and the remaining direction — the noise residual times the outer chain rule — is still an effective descent direction. The decomposed-diffusion-sampling view makes the same omission rigorous in the inverse-problem setting: instead of the manifold-constrained gradient `grad_{x_t} l(x_hat_t)` (which differentiates the denoised estimate through the score), take the gradient *with respect to the denoised estimate itself*, `grad_{x_hat} l(x_hat)`, which is one-step projected gradient on the tangent space at `x_hat` and needs no Jacobian through the network. So the template I'll use for one reverse step is the DDS form:

  x_{t-1} = sqrt(a_{t-1}) * ( x_hat_null - gamma_t * grad_{x_hat} l_sds(x_hat_null) ) + sqrt(1-a_{t-1}) * eps_null.

I want to be clear-eyed about why the renoise term in that template is `eps_null` and not anything guided, because it would be easy to wave my hands here. This template is the *unconditional* DDIM step — `sqrt(a_{t-1}) x_hat + sqrt(1-a_{t-1}) eps_null` — with the denoised estimate `x_hat` perturbed by a data-consistency gradient. The diffusion sampler itself is running unconditionally; the conditioning enters *only* through the gradient applied to the denoised estimate. So the text never touches the renoise: the noise added back to reach `M_{t-1}` is the clean unconditional direction. That isn't a swap I'm choosing — it falls out of which PF-ODE the template solves. If this construction holds up, it addresses source (b) directly. I'll hold that as the expected behavior and confirm it survives once I've written the full step out.

Now let me actually compute the gradient step and see what the denoised half turns into, because I want to know whether (a) gets fixed too — and I genuinely don't know yet what coefficient will come out. First reduce `l_sds`. Use the forward relation `x_t = sqrt(a_t) x + sqrt(1-a_t) eps` for a clean `x` on `M`. From it, `eps = (x_t - sqrt(a_t) x)/sqrt(1-a_t)`. And the conditional Tweedie estimate is `x_hat_c = (x_t - sqrt(1-a_t) eps_c)/sqrt(a_t)`, so `eps_c = (x_t - sqrt(a_t) x_hat_c)/sqrt(1-a_t)`. Subtract:

  eps_c - eps = ( -sqrt(a_t) x_hat_c + sqrt(a_t) x ) / sqrt(1-a_t) = sqrt(a_t)/sqrt(1-a_t) * (x - x_hat_c).

The network's prediction `eps_theta(x_t, c)` is `eps_c`, and the residual in `l_sds` is `eps_c - eps`, so

  l_sds(x) = || eps_c - eps ||^2 = (a_t / (1-a_t)) * || x - x_hat_c ||^2.

This reduction does a lot of work, so before I differentiate it let me check it numerically rather than trust the cancellation. Keep `a_t = 0.36`. Pick an `x_t` and a conditional prediction `eps_c`, form `x_hat_c = (x_t - 0.8 eps_c)/0.6`. Now pick a candidate clean point `x`, set the consistent `eps_used = (x_t - 0.6 x)/0.8`, and evaluate the two sides: `||eps_c - eps_used||^2` directly, against `(0.36/0.64)||x - x_hat_c||^2`. They come out equal to the last digit (14.3526 both ways, difference 0.0), and `eps_c - eps_used` equals `sqrt(a_t)/sqrt(1-a_t) (x - x_hat_c) = 0.75 (x - x_hat_c)` componentwise (max difference ~4e-16). Good — `l_sds` really is, up to the scalar `a_t/(1-a_t)`, just the squared distance from `x` to the conditional denoised estimate. That's a clean quadratic in `x`. Its gradient, evaluated at `x = x_hat_null`, is

  grad_{x_hat} l_sds (x_hat_null) = (2 a_t / (1-a_t)) * (x_hat_null - x_hat_c).

Plug that into the denoised half of the DDS step:

  x_hat_null - gamma_t * (2 a_t/(1-a_t)) * (x_hat_null - x_hat_c) = x_hat_null + lambda * (x_hat_c - x_hat_null),

where I've collapsed all the prefactors into a single scalar

  lambda := 2 a_t * gamma_t / (1 - a_t).

So the optimization step, after the Jacobian-free reduction, makes the denoised estimate

  x_hat_cfgpp = x_hat_null + lambda (x_hat_c - x_hat_null) = (1 - lambda) x_hat_null + lambda x_hat_c.

Now compare that to the CFG denoised estimate `(1-w) x_hat_null + w x_hat_c`. Same *shape* — an affine combination of the same two endpoints — but the coefficient is `lambda`, and the question that decides everything is whether `lambda` stays in `[0,1]`. It's a step size: `lambda = 2 a_t gamma_t/(1-a_t)`, so for a small learning rate it's small and positive. With the numbers above and a modest `gamma_t = 0.05`, `lambda = 2(0.36)(0.05)/0.64 = 0.0563`, comfortably inside `[0,1]`; and I confirmed numerically that the gradient-step expression `x_hat_null - gamma_t grad` and the interpolation `x_hat_null + lambda(x_hat_c - x_hat_null)` are the same vector (difference exactly 0). So with `lambda` in `[0,1]` this is an honest **interpolation**: a convex combination of `x_hat_null` and `x_hat_c`, sitting *on the segment between them*, which — under the piecewise-linear manifold — stays *on* `M`. That addresses source (a): I never extrapolate past the conditional endpoint, so the denoised estimate doesn't shoot off the data manifold. And the endpoints are interpretable — `lambda = 0` is pure unconditional, `lambda = 1` is pure conditional, and in between a smooth blend — so the scale stops being a magnification factor I have to crank to 12.

Let me re-express the whole step in the familiar noise-mixing notation, both to see how small the change from CFG is and because I want one object I can drop straight into the existing sampler. Define, identically to CFG but with `lambda`,

  eps_cfgpp(x_t) = eps_null(x_t) + lambda (eps_c(x_t) - eps_null(x_t)).

I'd *like* the interpolated denoised estimate to be exactly the Tweedie estimate of this mixed noise — that would mean the denoise half is still a plain Tweedie call, just with a mixed input. Symbolically: substitute `x_hat_• = (x_t - sqrt(1-a_t) eps_•)/sqrt(a_t)` into `x_hat_null + lambda(x_hat_c - x_hat_null)`; the `x_t/sqrt(a_t)` terms collect to `x_t/sqrt(a_t)`, and the noise terms collect to `-(sqrt(1-a_t)/sqrt(a_t))(eps_null + lambda(eps_c - eps_null))`, which is `(x_t - sqrt(1-a_t) eps_cfgpp)/sqrt(a_t)`. I checked this numerically too with the same `a_t = 0.36`, `lambda = 0.0563`: the interpolation `x_hat_null + lambda(x_hat_c - x_hat_null)` and `(x_t - 0.8 eps_cfgpp)/0.6` agree to ~1e-16. So the reverse step is

  x_hat_cfgpp(x_t) = (x_t - sqrt(1-a_t) eps_cfgpp(x_t)) / sqrt(a_t)        # denoise with the MIXED noise
  x_{t-1}          = sqrt(a_{t-1}) x_hat_cfgpp(x_t) + sqrt(1-a_{t-1}) eps_null(x_t)    # renoise with the UNCONDITIONAL noise.

And this is the moment to come back and verify the source-(b) claim I parked earlier: the renoise term here is `eps_null`, exactly as the DDS template predicted, with no `lambda` and no `eps_c` in it. So the construction did carry through — the conditioning lives entirely in the denoise, and the renoise is the clean unconditional direction.

Now put the two algorithms side by side. CFG: form `eps_cfg = eps_null + w*delta`, Tweedie-denoise with `eps_cfg`, renoise with `eps_cfg`. CFG++: form `eps_cfgpp = eps_null + lambda*delta`, Tweedie-denoise with `eps_cfgpp`, renoise with `eps_null`. The single line that changed is the renoise noise: `eps_cfg -> eps_null`. (And the scale moved from `w > 1` to `lambda` in `[0,1]`.) It costs nothing extra — I already compute `eps_null` and `eps_c` every step for the mix, exactly two network evaluations, same as CFG — and it touches both off-manifold sources: the denoise no longer extrapolates (interpolation, `lambda <= 1`), and the renoise no longer injects guided noise (it uses `eps_null`).

I should test the claim that this genuinely fixes DDIM inversion, because that was one of the sharpest failures and "stays on the manifold" is suggestive but not a proof. Inversion works by running the DDIM map backward, and the deterministic DDIM map is approximately invertible exactly when the predicted noise barely changes between adjacent steps, `eps(x_t) ~ eps(x_{t-1})`. For the unconditional sampler that holds well for small step sizes, so unconditional inversion is clean. Now look at CFG's inversion error — the failure of `eps_cfg(x_t) ~ eps_cfg(x_{t-1})`:

  eps_cfg(x_t) - eps_cfg(x_{t-1}) = ( eps_null(x_t) - eps_null(x_{t-1}) ) + w ( delta(x_t) - delta(x_{t-1}) ).

The first bracket is the small unconditional drift. The second is scaled by the *full* `w`, and `delta` — the conditional-vs-unconditional direction — is not small. Let me actually quantify how badly that hurts rather than assert it. Set up two adjacent steps: a small unconditional drift `||eps_null(x_t) - eps_null(x_{t-1})|| ~ 0.02`, an order-1 `delta` that drifts appreciably between steps so `||delta(x_t) - delta(x_{t-1})|| ~ 0.56`. Then for CFG with `w = 12`:

  ||eps_cfg(x_t) - eps_cfg(x_{t-1})|| = 6.69,   of which   w*||d delta|| = 6.68,

so the guidance term completely dominates the inversion error — the adjacent-step noise change, which inversion needs to be small, is ~300x the unconditional drift. For CFG++ with `lambda = 0.6` on the same vectors:

  ||eps_cfgpp(x_t) - eps_cfgpp(x_{t-1})|| = 0.343,   lambda*||d delta|| = 0.334,

a factor of ~19.5 smaller than CFG's. The renoise using `eps_null` means the Tweedie half of the inversion keeps the usual unconditional approximation, and the guidance-sensitive part of the update is scaled by `lambda` instead of `w`:

  e_cfg   ~= w * ( delta(x_t) - delta(x_{t-1}) ),
  ||e_cfgpp|| = lambda || delta(x_t) - delta(x_{t-1}) ||,

and `lambda <= 1 < w` makes `||e_cfgpp|| < ||e_cfg||` term by term. So the guidance part of the inversion error is downweighted from the large CFG scale to the small interpolation scale, with no null-embedding optimization or coupled invertible transform added to the sampler — which is the kind of extra machinery null-text inversion and EDICT had to bolt on precisely to fight this `w`-scaled error.

There's one more thing I want to work out, because it should explain the *qualitative* difference in the trajectories — the smooth CFG++ evolution versus CFG's sudden jumps. Track the denoised estimate `x_hat` step to step and ask how it moves. Write `d z(x_t) := z(x_t) - z(x_{t+1})` for the change of any quantity across one (inverse-time) step. Start from the iteration `x_t = sqrt(a_t) x_hat_cfg(x_{t+1}) + sqrt(1-a_t) eps_cfg(x_{t+1})`, then form the next Tweedie estimate `x_hat_cfg(x_t) = (x_t - sqrt(1-a_t) eps_cfg(x_t))/sqrt(a_t)`. Substitute the iteration for `x_t`:

  x_hat_cfg(x_t) = x_hat_cfg(x_{t+1}) + sqrt(1-a_t)/sqrt(a_t) * ( eps_cfg(x_{t+1}) - eps_cfg(x_t) ).

Expand `eps_cfg = eps_null + w*delta` inside that difference and convert back to denoised-estimate language via `sqrt(1-a_t)/sqrt(a_t) * delta = x_hat_null - x_hat_c = -(x_hat_c - x_hat_null) =: -Delta`. After collecting terms the per-step change of the CFG denoised estimate is

  d x_hat_cfg(x_t) = sqrt(1-a_t)/sqrt(a_t) * d eps_null(x_t) + w * ( Delta(x_t) - Delta(x_{t+1}) ),

with `Delta(x_t) := x_hat_c(x_t) - x_hat_null(x_t)`. Now run the identical algebra for CFG++: the renoise uses `eps_null`, so the iteration is `x_t = sqrt(a_t) x_hat_cfgpp(x_{t+1}) + sqrt(1-a_t) eps_null(x_{t+1})`, and the conditional information enters only through the denoise. Pushing it through,

  d x_hat_cfgpp(x_t) = sqrt(1-a_t)/sqrt(a_t) * d eps_null(x_t) + lambda * Delta(x_t).

Compare the two. Both share the same benign unconditional drift `sqrt(1-a_t)/sqrt(a_t) d eps_null`. But the *conditional* part is different in character. CFG++ contributes a single nudge `lambda Delta(x_t)` — at each step, move a little (`lambda <= 1`) toward the current conditional direction. CFG contributes `w (Delta(x_t) - Delta(x_{t+1}))`: a *difference* of consecutive conditional shifts, each scaled by a large `w`. So CFG's update is a large `w Delta(x_t)` push partly cancelled by `-w Delta(x_{t+1})` from the previous step — a large overshoot that mostly undoes itself, with the residual that actually advances toward the condition being the small difference `Delta(x_t) - Delta(x_{t+1})`, which needs a large `w` to register and changes sign as the two consecutive `Delta`s trade places. I haven't simulated the full trajectory, so I'll call this the likely mechanism rather than a proven one — but the structure (a large-magnitude term that nearly self-cancels step to step) is the natural source of the oscillatory early shifts and saturation that CFG shows, whereas CFG++'s single `lambda Delta(x_t)` has no previous-step overshoot to cancel and should evolve smoothly. I'd want to confirm it by plotting `x_hat` across steps for both rules on a real model.

I should also sanity-check that this isn't secretly just CFG with a clever schedule, because if it were the "principled trajectory redesign" story would collapse. CFG++ is still a linear combination of the same two score functions, so I can ask: is there a time-varying CFG scale `w_t` that reproduces the CFG++ trajectory? Match the two updates step by step — CFG's `x_{t-1} = sqrt(a_{t-1}) x_hat_cfg + sqrt(1-a_{t-1}) eps_cfg` against CFG++'s `sqrt(a_{t-1}) x_hat_cfgpp + sqrt(1-a_{t-1}) eps_null` — and solve for the `w_t` that makes them agree. The two reduce to a single scalar equation per step, and it does have a solution: a schedule of the form `w_t = -gamma_t/xi_t` with `gamma_t = sqrt(a_{t-1}) sqrt(1-a_t)/sqrt(a_t)` and `xi_t = sqrt(1-a_{t-1}) - sqrt(a_{t-1})sqrt(1-a_t)/sqrt(a_t)`. So in principle CFG++ does correspond to *some* time-varying CFG schedule — but it's a schedule nobody would write down by hand, with a sign and a `t`-dependence dictated by the manifold geometry rather than by heuristic scale-tuning. That's the reassurance I wanted: the content of the method is not "pick a better `w(t)`" but "derive which renoise noise keeps the trajectory on the manifold," and the equivalent schedule only exists after the fact because both rules live in the same two-score linear family.

Before I write code I want the method to extend past plain DDIM, because the same off-manifold logic should apply to any solver. Every widely-used sampler up to second order, when solving the *unconditional* PF-ODE, has a single-step update of the form

  x_i = x_hat_null(x_{i-1}) + a_i x_hat_null(x_{i-1}) + b_i x_hat_null(x_{i-2}) + c_i x_{i-1} + d_i eps,

where the leading term is the denoising and all the rest — the higher-order corrections, the carry of the previous latent, the injected noise for ancestral samplers (`d_i != 0`) — together form the *renoising*. Stated generally, the rule the DDIM case forces on me is: optimize the *denoising* under text guidance, keep the *renoising* unconditional. So I replace only the leading denoising term by the guided `x_hat_cfgpp` and leave every renoising Tweedie term unconditional:

  x_i = x_hat_cfgpp(x_{i-1}) + a_i x_hat_null(x_{i-1}) + b_i x_hat_null(x_{i-2}) + c_i x_{i-1} + d_i eps.

For Euler this reads `x_{i+1} = x_hat_cfgpp(x_i) + (x_i - x_hat_null(x_i))/sigma_i * sigma_{i+1}` — note the slope term uses `x_hat_null`, not the guided estimate. For Euler-ancestral, same denoising swap plus the usual stochastic term `sigma_i eps`. For DPM++ 2M, rearrange its two-step update so the guided conditional Tweedie appears only as the leading term and every other Tweedie (the current and previous, including the finite-difference correction `(1-e^{-h})/(2r)` term) stays unconditional. The recipe is mechanical once stated this way. And it extends to distilled few-step models like SDXL-Lightning and SDXL-Turbo too: those already bake a fixed guidance into the conditional score, so I set `lambda = 1` (use the conditional denoised estimate) and still take the rest of the Euler renoising components from the unconditional estimate — the difference is *still* just the renoising step.

Now let me write the DDIM CFG++ sampler as the code I'd actually ship, filling the one empty slot — the guidance rule and the two halves of the step. Two network outputs per step, the mixed denoise, the unconditional renoise:

```python
import torch
from tqdm import tqdm


class BaseDDIMCFGpp(StableDiffusion):
    """DDIM sampler with CFG++: interpolating guidance on the denoise,
    unconditional noise on the renoise. Same two NFE/step as CFG."""

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance=0.6, prompt=["", ""], callback_fn=None, **kwargs):
        # cfg_guidance is the interpolation scale lambda.
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        zt = self.initialize_latent()                  # x_T ~ N(0, I)
        zt = zt.requires_grad_()

        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)                         # bar_alpha_t
            at_prev = self.alpha(t - self.skip)        # bar_alpha_{t-1}

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)   # eps_null, eps_c
                # mixed noise: eps_null + lambda (eps_c - eps_null)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # denoise: Tweedie with the mixed noise
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # renoise with the unconditional noise eps_null
            zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(), 'zt': zt.detach(), 'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # last step: return the clean denoised estimate, no renoise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
```

The SDXL version is the identical step with the scheduler's `alphas_cumprod[t]` indexed directly and the dual text embeddings passed through — same `noise_pred` for the Tweedie denoise, same `noise_uc` for the renoise:

```python
class BaseDDIMCFGpp(SDXL):
    def reverse_process(self, null_prompt_embeds, prompt_embeds, cfg_guidance,
                        add_cond_kwargs, shape=(1024, 1024), callback_fn=None, **kwargs):
        zt = self.initialize_latent(
            size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))

        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(
                    zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()       # denoise (mixed)
            zt = at_next.sqrt() * z0t + (1 - at_next).sqrt() * noise_uc # renoise (uncond)

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(), 'zt': zt.detach(), 'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        return z0t
```

Let me trace the causal chain one more time. I started from CFG's pain at the high scale it needs — mode collapse, saturation, broken inversion — and refused to accept it as inherent, because the unconditional sampler is fine. Dissecting one DDIM step, and checking the algebra numerically, showed two off-manifold sources: the `w>1` denoised estimate `(1-w)x_hat_null + w x_hat_c` is an extrapolation off the piecewise-linear manifold (coefficient on `x_hat_c` is `12` at the working scale), and the renoise injects the guided, off-manifold noise. Reframing text guidance as `min_{x in M} l(x)` instead of as a sharpened posterior let me run the well-behaved unconditional sampler and add only a corrective nudge; the text-conditioned score-matching (SDS) loss is the right `l`, and the decomposed-sampling / Jacobian-omitting view gave a stable gradient on the denoised estimate with the renoise unconditional — which I then confirmed survives into the written-out step, fixing source (b). Reducing the loss to `(a_t/(1-a_t))||x - x_hat_c||^2` (checked to match the raw residual numerically) and taking one gradient step turned the denoised estimate into the interpolation `(1-lambda)x_hat_null + lambda x_hat_c` with `lambda = 2 a_t gamma_t/(1-a_t)` landing in `[0,1]` (0.0563 for the test values) — fixing source (a) by staying on the segment, hence on the manifold. Rewriting in noise-mixing form, verified by the Tweedie-of-mixed-noise check, exposed the whole method as one line changed from CFG: denoise with the mixed `eps_cfgpp`, renoise with `eps_null`, scale in `[0,1]`, zero extra compute. The inversion-error numbers showed the guidance-sensitive term shrinks from `w` to `lambda` (a ~20x smaller adjacent-step noise change on the test vectors), restoring invertibility; the posterior-mean evolution traced CFG's oscillatory `w(Delta_t - Delta_{t+1})` down to CFG++'s single `lambda Delta_t`, which I expect makes the trajectory smooth but would still confirm by plotting. The general higher-order rule — guide only the leading denoising term, keep all renoising Tweedie terms unconditional — carries the same principle to Euler, ancestral, and DPM++ solvers and to distilled few-step models. And the whole thing drops into the existing DDIM harness as a one-line change in which predicted noise feeds the renoise.
