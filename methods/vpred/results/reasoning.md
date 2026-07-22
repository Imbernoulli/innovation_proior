The thing that keeps biting me is a choice I treated as cosmetic. I have a diffusion model: I noise a clean image into `z_t = alpha_t * x + sigma_t * eps` along a schedule where `alpha_t` slides from near one (almost clean) down to zero (pure noise) and `sigma_t = sqrt(1 - alpha_t^2)` slides the other way, and I train one network to undo it. The sampler doesn't consume the raw network output directly — it consumes a clean-image estimate `x_hat`, and at every step it takes `x_hat`, takes one short step back toward less noise, and repeats. So the only thing that ever has to be right is `x_hat`. And `x_hat` is some fixed linear function of the network's output, of `z_t`, and of the schedule. I've been treating "what does the network predict" as a free relabeling, because noise, clean image, and latent are all linearly related and you can convert any one into the others. At infinite precision and infinite sampling steps that's true. But I keep watching few-step samples degrade in a way that depends on this supposedly-cosmetic choice, so the relabeling is not free, and I need to figure out exactly why.

Let me write down the standard choice and actually stare at it, not skim past it. The network predicts the added noise, `eps_hat_theta(z_t)`. To get the clean image the sampler needs, I invert the forward equation `z_t = alpha_t * x + sigma_t * eps`: solve for `x`, replace the true `eps` with the network's `eps_hat`, and I get

  x_hat_theta(z_t) = (1 / alpha_t) * (z_t - sigma_t * eps_hat_theta(z_t)).

There's the divide by `alpha_t`. For most of training this is harmless — `alpha_t` is order one. But the whole point of the schedule is that `alpha_t -> 0` at the noisy end. So consider a small error `delta` in the network output, `eps_hat = eps + delta`. The induced error in `x_hat` is `-(sigma_t / alpha_t) * delta`. As `alpha_t -> 0` and `sigma_t -> 1`, that factor `sigma_t / alpha_t` blows up without bound. Let me put a number on "without bound" so I know whether it's a real problem or a theoretical one I can ignore: at `phi_t = 1.40` (`alpha_t = 0.170`, `sigma_t = 0.985`) the factor is already `0.985/0.170 = 5.8`, at `phi_t = 1.50` it is `0.998/0.0707 = 14.1`, and at `phi_t = 1.55` it is `1.00/0.0208 = 48`. So a 0.01 output error near the noisy end becomes a clean-image error of half an image. The same tiny output error that costs nothing at high SNR costs a gigantic clean-image error at low SNR, and the blowup arrives well before the literal endpoint. Push it to the literal limit: `alpha_t = 0`, `sigma_t = 1`. Then `z_t = eps` is pure noise, it carries zero information about `x`, and "predict the noise" is the question "given pure noise, what is the noise" — the answer is just the input, the task is degenerate, and there is no signal left to teach the network what `x` was. Worse, the noise-space loss `|| eps - eps_hat ||^2` equals `(alpha_t^2 / sigma_t^2) * || x - x_hat ||^2`, so at SNR zero the weight `alpha_t^2 / sigma_t^2` is exactly zero — the loss has stopped caring about the clean-image error precisely at the level where I most need it to. So noise prediction is excellent where `alpha_t` is order one and pathological where `alpha_t -> 0`.

Now I understand why I never noticed before: with hundreds of sampling steps the pathology hides. After each step the latent gets clipped back into a valid range, and any early misstep at the noisy end gets corrected by the dozens of steps that follow. The instability is real but it's masked by redundancy. The moment I cut to a few steps — or, in the worst case, a single step that starts from pure noise — there is no later step to clean up, and the `1 / alpha_t` amplification lands directly in the sample. So the failure mode I'm seeing is exactly this, surfacing only when I shorten the sampler. Good. That tells me the fix is not "tune the sampler" but "change what `x_hat` is a function of, so it stops dividing by something that goes to zero."

The obvious knee-jerk: just predict the clean image directly. Let the network output be `x_hat_theta(z_t)`, no conversion at all, loss `|| x - x_hat ||^2`. At the noisy end this is wonderful — the target is literally the thing the sampler wants, there is no `1 / alpha_t`, nothing blows up. So have I just won? Let me check the other end before I celebrate, because the noise route was fine there and I shouldn't trade one cliff for another. High SNR: `alpha_t -> 1`, `sigma_t -> 0`, so `z_t ≈ x`. The network is being asked to output `x` when its input already is essentially `x` — it's copying, learning almost nothing, wasting capacity at the levels where there's the most signal to exploit. And if I ever need the noise estimate from this parameterization — which the sampler implicitly does, since a DDIM step moves along the noise direction — I'd compute `eps_hat = (z_t - alpha_t * x_hat) / sigma_t`, and now I'm dividing by `sigma_t -> 0`. The noise side blows up exactly where the noise route was healthy. So clean-image prediction is the mirror image: well-conditioned at the noisy end, ill-conditioned at the clean end. Each of the two natural choices is good at one end and bad at the other, and they fail at *opposite* ends. That's not a coincidence I should shrug at — it's the shape of a solution. I want something that behaves like noise prediction where noise prediction is good (high SNR) and like clean-image prediction where that is good (low SNR), with a smooth handoff in between, and crucially a conversion to `x_hat` that never divides by a vanishing coefficient.

So let me stop guessing parameterizations and ask geometrically what's actually going on, because the two failures-at-opposite-ends smell like a coordinate problem, not a fundamental one. The latent is `z_t = alpha_t * x + sigma_t * eps` with `alpha_t^2 + sigma_t^2 = 1`. That constraint is the clue: `(alpha_t, sigma_t)` is a point on the unit circle. So introduce the angle. Let `phi_t = arctan(sigma_t / alpha_t)`, which gives `alpha_t = cos(phi_t)` and `sigma_t = sin(phi_t)`, and then

  z_phi = cos(phi) * x + sin(phi) * eps.

As `phi` runs from `0` to `pi/2`, `z_phi` rotates from `x` (pure signal) to `eps` (pure noise). The whole diffusion forward process is *rotation in the `(x, eps)` plane*. That reframes everything: `x` and `eps` are just two orthogonal axes, and `z_phi` is the same vector seen at angle `phi`. Noise prediction reads off the `eps` axis; clean-image prediction reads off the `x` axis. Both are *fixed* axes, and the trouble is that to recover `x` from `z_phi` by projecting onto a fixed axis I need to divide by `cos(phi)` or `sin(phi)`, one of which goes to zero. Fixed axes are the disease.

What's the natural direction that *moves with* `z_phi` instead of staying fixed? The velocity of `z_phi` as I rotate — the tangent to the circle. Differentiate:

  v_phi = d z_phi / d phi = -sin(phi) * x + cos(phi) * eps = cos(phi) * eps - sin(phi) * x.

So define the network to predict this velocity, `v = cos(phi) * eps - sin(phi) * x`, which in schedule terms is `v = alpha_t * eps - sigma_t * x`. Let me see what makes `v` special. Look at the pair `{z_phi, v_phi}`:

  z_phi =  cos(phi) * x + sin(phi) * eps,
  v_phi = -sin(phi) * x + cos(phi) * eps.

That is exactly a rotation matrix applied to `(x, eps)`:

  [ z_phi ]   [  cos(phi)   sin(phi) ] [ x   ]
  [ v_phi ] = [ -sin(phi)   cos(phi) ] [ eps ].

A rotation is orthogonal, so its inverse is its transpose — no determinant, no division, the inverse coefficients are just `cos` and `sin` again, all bounded in `[-1, 1]`. Invert it:

  [ x   ]   [ cos(phi)  -sin(phi) ] [ z_phi ]
  [ eps ] = [ sin(phi)   cos(phi) ] [ v_phi ],

which reads

  x   = cos(phi) * z_phi - sin(phi) * v_phi = alpha_t * z_t - sigma_t * v,
  eps = sin(phi) * z_phi + cos(phi) * v_phi = sigma_t * z_t + alpha_t * v.

Let me check `x` directly to be sure I didn't fumble a sign: `cos(phi) * z_phi - sin(phi) * v_phi = cos(phi)(cos x + sin eps) - sin(phi)(cos eps - sin x) = cos^2 x + cos sin eps - sin cos eps + sin^2 x = (cos^2 + sin^2) x = x`. Clean. And `eps`: `sin(phi)(cos x + sin eps) + cos(phi)(cos eps - sin x) = sin cos x + sin^2 eps + cos^2 eps - cos sin x = eps`. Clean. So if the network predicts `v`, the clean-image estimate is

  x_hat = alpha_t * z_t - sigma_t * v_hat,

and that's it — no `1 / alpha_t`, no `1 / sigma_t`, just `alpha_t` and `sigma_t` themselves multiplying things, both in `[0, 1]`. The amplification factor that was `sigma_t / alpha_t -> infinity` for noise prediction is gone; an error `delta` in `v_hat` induces an error `-sigma_t * delta` in `x_hat`, bounded by `delta` everywhere. This is the well-conditioned recovery I was after, and it fell straight out of choosing the moving (tangent) direction instead of a fixed axis.

And the smooth-handoff property I wanted is automatic, just from the definition `v = alpha_t * eps - sigma_t * x`. At the clean end, `alpha_t -> 1`, `sigma_t -> 0`, so `v -> eps`: predicting `v` is essentially predicting the noise, which is exactly the regime where noise prediction was the good choice. At the noisy end, `alpha_t -> 0`, `sigma_t -> 1`, so `v -> -x`: predicting `v` is essentially predicting (minus) the clean image, which is the regime where clean-image prediction was the good choice. So `v` interpolates between the two good behaviors and uses each where it's appropriate, with the crossover happening continuously through the schedule. The two parameterizations that each had a cliff are recovered as the two *limits* of this one, and neither limit divides by zero because the conversion is the bounded rotation, not a fixed-axis projection.

Now I should check this isn't just nice at the endpoints but coherent with the sampler I actually run, the deterministic DDIM step. If the geometry is the real story, DDIM ought to look clean in this `(z, v)` frame too. The DDIM update going from noise level `t` to a lower level `s` is

  z_s = alpha_s * x_hat + sigma_s * (z_t - alpha_t * x_hat) / sigma_t.

The `(z_t - alpha_t * x_hat) / sigma_t` piece is just the implied noise estimate `eps_hat`, so DDIM is `z_s = alpha_s * x_hat + sigma_s * eps_hat` — it rebuilds the latent at the new level from the clean-image and noise estimates. Substitute my rotation expressions, `x_hat = cos(phi_t) z_t - sin(phi_t) v_hat` and `eps_hat = sin(phi_t) z_t + cos(phi_t) v_hat`, with `alpha_s = cos(phi_s)`, `sigma_s = sin(phi_s)`:

  z_s = cos(phi_s)[cos(phi_t) z_t - sin(phi_t) v_hat] + sin(phi_s)[sin(phi_t) z_t + cos(phi_t) v_hat].

Collect the `z_t` and `v_hat` terms:

  z_t coefficient:  cos(phi_s)cos(phi_t) + sin(phi_s)sin(phi_t) = cos(phi_s - phi_t),
  v_hat coefficient: -cos(phi_s)sin(phi_t) + sin(phi_s)cos(phi_t) = sin(phi_s - phi_t),

using the angle-difference identities `cos(A - B) = cos A cos B + sin A sin B` and
`sin(A - B) = sin A cos B - cos A sin B`. So

  z_s = cos(phi_s - phi_t) * z_t + sin(phi_s - phi_t) * v_hat.

That is gorgeous. A DDIM step is a *rotation by the angle `phi_s - phi_t`* in the `(z, v_hat)` plane. Since I'm denoising, `s < t` means I'm reducing the noise, so `phi_s < phi_t` and the step is by `-delta` where `delta = phi_t - phi_s > 0`; writing it that way,

  z_{phi_t - delta} = cos(delta) * z_t - sin(delta) * v_hat,

so I move `z_t` along the `-v_hat` direction by angle `delta`. Sign slips on rotations are exactly the kind of error this whole derivation is supposed to avoid, so before I trust the rewrite I'll put numbers through both the raw DDIM update and this rotation form and see if they actually agree. Take `phi_t = 1.10` (the noisy level I'm stepping from) and `phi_s = 0.80`, so `alpha_t = cos 1.10 = 0.4536`, `sigma_t = sin 1.10 = 0.8912`, `alpha_s = cos 0.80 = 0.6967`, `sigma_s = sin 0.80 = 0.7174`. Pick a scalar `x = 0.37` and `eps = -1.24`, giving `z_t = 0.4536*0.37 + 0.8912*(-1.24) = -0.9373`, and let the network's (deliberately imperfect) output be `v_hat = 0.55`. Raw DDIM: `x_hat = alpha_t*z_t - sigma_t*v_hat = 0.4536*(-0.9373) - 0.8912*0.55 = -0.9154`, then the implied noise `eps_hat = (z_t - alpha_t*x_hat)/sigma_t = (-0.9373 - 0.4536*(-0.9154))/0.8912 = -0.5854`, and `z_s = alpha_s*x_hat + sigma_s*eps_hat = 0.6967*(-0.9154) + 0.7174*(-0.5854) = -1.0579`. Rotation form: `delta = phi_t - phi_s = 0.30`, `z_s = cos(0.30)*z_t - sin(0.30)*v_hat = 0.9553*(-0.9373) - 0.2955*0.55 = -1.0579`. They agree to every digit I carried (and to machine precision when I run it). The rotation rewrite is right, including the sign. The reason this matters beyond elegance: the step is expressed purely as an angle increment `delta` on the unit circle, with coefficients `cos(delta)` and `sin(delta)` that depend *only on the step size in angle*, not on where on the circle I am — not on the SNR. With noise prediction the effective step coefficients are tangled up with `alpha_t, sigma_t` and behave very differently at different noise levels; here every step is the same kind of rotation regardless of SNR. That SNR-independence of the step is the deep reason `v` is stable for taking large, few steps: the angular step no longer adds a schedule-endpoint amplification on top of the model error, whereas a fixed-axis parameterization can make the same output error explode near its bad end.

So the predicted quantity is settled: the network outputs `v`, the training target is `v = alpha_t * eps - sigma_t * x`, and the sampler recovers `x_hat = alpha_t * z_t - sigma_t * v_hat`. But I haven't pinned the loss weighting, and the standard choice for noise prediction was part of the problem, so I can't just inherit it. The simplest thing is plain mean-squared error on `v` itself, `|| v - v_hat ||^2`. Let me work out what weighting on the clean-image error that secretly corresponds to, because the clean-image error is the thing I actually care about and the comparison to the old choices has to be in those units. I have `v - v_hat = alpha_t(eps - eps_hat) - sigma_t(x - x_hat)`. And the implied noise error in terms of the clean-image error is `eps - eps_hat = (z_t - alpha_t x) / sigma_t - (z_t - alpha_t x_hat) / sigma_t = -(alpha_t / sigma_t)(x - x_hat)`. Substitute:

  v - v_hat = alpha_t * [-(alpha_t / sigma_t)(x - x_hat)] - sigma_t (x - x_hat)
            = -[ alpha_t^2 / sigma_t + sigma_t ] (x - x_hat)
            = -[ (alpha_t^2 + sigma_t^2) / sigma_t ] (x - x_hat)
            = -(1 / sigma_t)(x - x_hat),

where the last step uses `alpha_t^2 + sigma_t^2 = 1`. Square it:

  || v - v_hat ||^2 = (1 / sigma_t^2) * || x - x_hat ||^2 = (1 + alpha_t^2 / sigma_t^2) * || x - x_hat ||^2,

since `1 / sigma_t^2 = (alpha_t^2 + sigma_t^2) / sigma_t^2 = alpha_t^2/sigma_t^2 + 1`. So mean-squared error on `v` is a clean-image loss with weight `w(lambda_t) = 1 + exp(lambda_t)` — call it "SNR + 1". Compare to the noise-prediction weight, which was `exp(lambda_t)`, plain SNR. Let me read off both weightings at a few angles to see whether "SNR + 1" actually behaves the way I'm claiming. At `phi = 0.01` (high SNR), `alpha^2/sigma^2 = (1/sigma^2 - 1) = 9999`, so the v-weight is `10000` and plain SNR is `9999` — the `+1` is a rounding error, the two agree, and I keep the proven high-SNR behavior. At `phi = 1.00`, v-weight `1.41` vs SNR `0.41`. At `phi = 1.55` (near the noisy end), `sigma = 0.9998` so plain SNR is `alpha^2/sigma^2 = 0.0004`, essentially zero — the loss has abandoned the low-SNR end, the exact pathology — whereas the v-weight is `1 + 0.0004 = 1.0004`, i.e. it floors at one. The `+1` is precisely the floor that keeps the noisy end of the schedule contributing to the gradient instead of being ignored. So the v-loss isn't an arbitrary choice; it is the weighting that retains the good high-SNR emphasis while refusing to zero out the low-SNR end, which is the half of the schedule that few-step sampling depends on. I could try to floor the SNR weight explicitly, but that would only address the weighting half of the problem. Plain MSE on `v` gives me the SNR+1 floor and the well-conditioned recovery in the same parameterization.

Let me make sure I haven't talked myself into something I can't actually build consistently. The one hard constraint is that the sampler's recovery must be the exact inverse of the training target, or the model is trained to predict one thing and the sampler decodes it as another. Training target: `v = alpha_t * eps - sigma_t * x`. The noised input the network sees during training is `x_t = alpha_t * x + sigma_t * eps` (I'll use `x_t` for the latent in code). Sampler recovery: `x_hat = alpha_t * x_t - sigma_t * v_hat`. Are these consistent? Plug the true `v` into the recovery and I'd better get the true `x`: `alpha_t * x_t - sigma_t * v = alpha_t(alpha_t x + sigma_t eps) - sigma_t(alpha_t eps - sigma_t x) = alpha_t^2 x + alpha_t sigma_t eps - sigma_t alpha_t eps + sigma_t^2 x = (alpha_t^2 + sigma_t^2) x = x`. Exactly `x`. So the pair is consistent — the recovery is the algebraic inverse, not an approximation — and that's the whole correctness requirement.

In the schedule tensors I actually have, `sqrt_alpha = sqrt(alphas_cumprod)` is `alpha_t` and `sqrt_one_minus_alpha = sqrt(1 - alphas_cumprod)` is `sigma_t`. So the two coupled slots become exactly: target `= sqrt_alpha * noise - sqrt_one_minus_alpha * x_0`, and recovery `x_0 = sqrt_alpha * x_t - sqrt_one_minus_alpha * v`. That's the whole method — one definition and its inverse, dropped into the existing noising-plus-MSE training step and the existing DDIM step, with the fixed backbone, cosine schedule, optimizer and sampler all untouched.

```python
import torch


def compute_training_target(x_0, noise, timesteps, schedule):
    # Predict the velocity v = alpha_t * eps - sigma_t * x_0, the tangent to the
    # noising circle z_t = alpha_t*x_0 + sigma_t*eps. This is the orthonormal
    # rotation partner of z_t, so the clean-image recovery below needs no
    # division by a vanishing coefficient.
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)            # alpha_t
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)  # sigma_t
    return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0


def predict_x0(model_output, x_t, timesteps, schedule):
    # Invert the rotation to recover the clean image from the predicted velocity:
    #   x_t = alpha_t*x_0 + sigma_t*eps,  v = alpha_t*eps - sigma_t*x_0
    #   => x_0 = alpha_t*x_t - sigma_t*v       (bounded coeffs; no 1/alpha_t blowup)
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)            # alpha_t
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)  # sigma_t
    return sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output
```
