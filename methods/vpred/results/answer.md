# v-prediction (velocity parameterization), distilled

v-prediction trains a diffusion denoiser to output the **velocity**
`v = alpha_t * eps - sigma_t * x_0` — the tangent to the noising trajectory
`z_t = alpha_t * x_0 + sigma_t * eps` viewed as rotation on a circle — instead of the noise
`eps` or the clean image `x_0`. Because `{z_t, v}` is an orthonormal rotation of `{x_0, eps}`,
the clean-image estimate the sampler needs is recovered by the bounded inverse rotation
`x_0 = alpha_t * z_t - sigma_t * v`, with no division by a coefficient that vanishes. This keeps
the implied `x_0` well-conditioned across the entire signal-to-noise range, which is what makes
diffusion stable in the few-step (and one-step) sampling regime.

## Problem it solves

Standard noise (`eps`) prediction recovers the clean image via
`x_0 = (z_t - sigma_t * eps_hat) / alpha_t`, which divides by `alpha_t -> 0` at the noisy end of
the schedule: a small output error is amplified without bound, and at SNR zero the input is pure
noise (an uninformative, degenerate target whose loss weight is zero). Predicting `x_0` directly
fixes the noisy end but breaks symmetrically at the clean end, where the implied noise divides by
`sigma_t -> 0`. With hundreds of sampling steps the noise-prediction pathology is masked by
clipping and later corrections; with few steps it surfaces directly in the sample. A
parameterization is needed whose implied `x_0` stays stable at every noise level.

## Key idea

The variance-preserving forward process `z_t = alpha_t * x_0 + sigma_t * eps`,
`alpha_t^2 + sigma_t^2 = 1`, is rotation in the `(x_0, eps)` plane. Writing
`alpha_t = cos(phi_t)`, `sigma_t = sin(phi_t)` with `phi_t = arctan(sigma_t / alpha_t)`, the
latent is `z_phi = cos(phi) x_0 + sin(phi) eps`. Its velocity along the arc,

```
v = dz_phi / d phi = cos(phi) eps - sin(phi) x_0 = alpha_t * eps - sigma_t * x_0,
```

is the orthonormal rotation partner of `z_t`:

```
[ z ]   [  cos phi   sin phi ] [ x_0 ]
[ v ] = [ -sin phi   cos phi ] [ eps ]   (orthogonal: inverse = transpose)
```

so inverting (no division) gives the recovery used at sampling time:

```
x_0 = cos(phi) z - sin(phi) v = alpha_t * z_t - sigma_t * v,
eps = sin(phi) z + cos(phi) v = sigma_t * z_t + alpha_t * v.
```

The recovery coefficients `alpha_t, sigma_t` are bounded in `[0, 1]`; the `1/alpha_t` blowup of
noise prediction is gone. `v` also interpolates the two good behaviors: at high SNR
(`alpha_t -> 1`) `v -> eps` (≈ noise prediction); at low SNR (`alpha_t -> 0`) `v -> -x_0`
(≈ clean-image prediction).

## Why it is stable for few-step sampling

A deterministic DDIM step from level `t` to a lower level `s` is, in this basis, a pure
rotation:

```
z_s = cos(phi_s - phi_t) z_t + sin(phi_s - phi_t) v_hat
    = cos(delta) z_t - sin(delta) v_hat,    delta = phi_t - phi_s > 0,
```

derived from `z_s = alpha_s x_hat + sigma_s eps_hat` and the rotation expressions for
`x_hat, eps_hat`, using `cos(A-B)=cos A cos B + sin A sin B`, `sin(A-B)=sin A cos B - cos A sin B`.
The step coefficients depend only on the angular step size `delta`, not on the SNR, so large,
few steps avoid the extra endpoint amplification that fixed-axis parameterizations introduce.

## Loss weighting

Plain mean-squared error on `v` equals a clean-image loss weighted by `SNR + 1`:

```
|| v - v_hat ||^2 = (1 + alpha_t^2 / sigma_t^2) || x_0 - x_hat ||^2 = (1/sigma_t^2) || x_0 - x_hat ||^2,
```

since `eps - eps_hat = -(alpha_t/sigma_t)(x_0 - x_hat)` and
`v - v_hat = -[(alpha_t^2 + sigma_t^2)/sigma_t](x_0 - x_hat) = -(1/sigma_t)(x_0 - x_hat)`. This
matches the standard SNR weighting (`exp(lambda_t)`) at high SNR but floors the weight at `1` at
SNR zero, instead of vanishing there — keeping the low-SNR end, on which few-step sampling
depends, in the gradient.

## Consistency

The recovery is the exact algebraic inverse of the target: plugging the true
`v = alpha_t eps - sigma_t x_0` into `x_0 = alpha_t z_t - sigma_t v` returns `x_0`
(`(alpha_t^2 + sigma_t^2) x_0 = x_0`), so training target and sampling recovery are consistent.

## Working code

The two coupled functions that fill the harness slots (schedule tensors
`sqrt_alpha = sqrt(alphas_cumprod) = alpha_t`,
`sqrt_one_minus_alpha = sqrt(1 - alphas_cumprod) = sigma_t`):

```python
import torch


def compute_training_target(x_0, noise, timesteps, schedule):
    # v = alpha_t * eps - sigma_t * x_0
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0


def predict_x0(model_output, x_t, timesteps, schedule):
    # x_0 = alpha_t * x_t - sigma_t * v   (inverse rotation; no division by alpha_t)
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output
```

Equivalent canonical forms, all consistent with the above:

```python
# google-research/diffusion_distillation (JAX), with alpha=sqrt(sigmoid(logsnr)),
# sigma=sqrt(sigmoid(-logsnr)):
#   predict_v_from_x_and_eps:  v = alpha * eps - sigma * x
#   predict_x_from_v:          x = alpha * z   - sigma * v
#   training loss:             v_mse = mean((model_v - v_target) ** 2)

# diffusers DDPMScheduler:
#   get_velocity:              velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
#   step (v_prediction):       pred_original_sample = alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
#   (beta_prod_t = 1 - alpha_prod_t, so beta_prod_t**0.5 = sqrt_one_minus_alpha)
```
