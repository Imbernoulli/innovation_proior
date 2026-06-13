# Context: choosing what a denoising diffusion network should predict (unconditional CIFAR-10)

## Research question

A denoising diffusion model generates images by training one neural network to reverse a fixed
Gaussian noising process. The forward process takes a clean image `x_0` and a noise level
indexed by `t` and produces a noisy image

```
x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon,   epsilon ~ N(0, I),
```

where `alpha_t` (the cumulative product of `1 - beta` over the schedule) runs from near 1 at
`t = 0` (almost-clean) to near 0 at the final step (almost-pure-noise). Generation runs the
process backward: start from noise and repeatedly denoise. The reverse step needs, at every
noise level, an estimate of the clean image `x_0` implied by the current noisy `x_t` — a
deterministic sampler such as 50-step DDIM literally consumes that estimate at each step.

The network has to learn one regression. But there is a free choice in *what tensor the network
emits* — its output can be read as the clean image, or the noise, or some mixture — and a
matched rule that converts that output back into the `x_0` estimate the sampler needs. The three
readings are algebraically interchangeable (each is an exact linear function of the others at a
fixed `t`), so in the infinite-data, infinite-capacity limit they define the same model. Under a
*finite* budget — a fixed UNet, a fixed number of training steps, a fixed optimizer, a fixed
sampler — they are not equivalent: they place the regression target in different spaces, scale
the per-timestep signal differently, and so produce different gradients, different effective
loss weightings across noise levels, and different final sample quality (FID). The precise goal
here is to pick the target tensor and its matching `x_0`-recovery rule that gives the best FID on
unconditional CIFAR-10 under that fixed pipeline — and the two pieces must be *consistent*, in
that the recovery rule exactly inverts the training target so the sampler is fed the correct
clean-image estimate.

## Background

A diffusion model is a latent-variable generative model (Sohl-Dickstein, Weiss, Maheswaranathan &
Ganguli 2015; Ho, Jain & Abbeel 2020). A fixed forward Markov chain adds a little Gaussian noise
at each of `T` steps until the data is destroyed into `N(0, I)`; a learned reverse chain
`p_theta(x_{t-1} | x_t)` of Gaussian denoising steps undoes it. Because each forward step adds
only a small amount of noise, each reverse conditional is also approximately Gaussian, so the
learned reverse step only needs to predict a mean (the variances are fixed to schedule
constants). The forward marginal has the closed form `q(x_t | x_0) = N(sqrt(alpha_t) x_0,
(1 - alpha_t) I)`, which is exactly the reparameterization `x_t = sqrt(alpha_t) x_0 +
sqrt(1 - alpha_t) epsilon` above — so a training pair is generated cheaply at any `t` by drawing
one image, one Gaussian noise, and mixing them by the schedule.

The load-bearing facts about the reverse step:

- **The reverse step is a denoiser.** Conditioned on the clean image, the forward process can be
  run in reverse in closed form: `q(x_{t-1} | x_t, x_0)` is Gaussian with a mean that is a fixed
  linear combination of `x_t` and `x_0`,

  ```
  tilde_mu_t(x_t, x_0) = ( sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t) ) * x_0
                       + ( sqrt(alpha_step_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) ) * x_t,
  ```

  where `alpha_step_t = 1 - beta_t` is the per-step factor and `alpha_bar_t` the cumulative
  product. The only unknown in this mean is `x_0`. So the network's entire job, at every noise
  level, is to supply the clean image `x_0` (equivalently any quantity from which `x_0` can be
  recovered); the rest of the reverse mean is fixed arithmetic. A deterministic sampler makes
  this explicit — given `x_t`, first form an estimate of `x_0`, then step toward `x_{t-1}` using
  the closed-form posterior with that estimate substituted for the true `x_0`.

- **The denoising target is, up to scale, the score.** For a Gaussian corruption
  `N(x_t; sqrt(alpha_t) x_0, (1 - alpha_t) I)`, the score `grad_{x_t} log q(x_t | x_0)` equals
  `-(x_t - sqrt(alpha_t) x_0)/(1 - alpha_t) = -epsilon / sqrt(1 - alpha_t)` (Vincent 2011). So
  regressing onto the added noise `epsilon` is regressing onto the score up to a positive scale,
  and a denoiser that outputs `epsilon` doubles as a learned score estimate that a
  Langevin-/score-based sampler can use directly.

- **The three readings of the output, and their conversions.** At a fixed `t`, writing
  `a = sqrt(alpha_t)` and `b = sqrt(1 - alpha_t)` (so `a^2 + b^2 = 1`), the noisy image is
  `x_t = a*x_0 + b*epsilon` and the three candidate quantities are linearly related:

  ```
  x_0     = (x_t - b*epsilon) / a
  epsilon = (x_t - a*x_0) / b
  v       = a*epsilon - b*x_0            (the "velocity", Salimans & Ho 2022)
  ```

  Any one determines the other two given `x_t`. So whatever the network emits, the `x_0` estimate
  the sampler needs is one fixed linear map away.

- **The output reading sets the loss weighting across noise levels.** A plain mean-squared error
  on the network output is *not* the same objective in different readings, because the linear
  maps above carry `t`-dependent scale factors. From `epsilon - epsilon_hat = (a/b)(x0_hat - x_0)`
  one gets, exactly,

  ```
  || epsilon - epsilon_hat ||^2 = (a^2 / b^2) * || x_0 - x0_hat ||^2 = SNR_t * || x_0 - x0_hat ||^2,
  ```

  with the signal-to-noise ratio `SNR_t = alpha_t / (1 - alpha_t) = a^2 / b^2`. So a unit-weight
  MSE on noise is a *SNR-weighted* MSE on the clean image: it down-weights the high-noise (low
  `SNR_t`) steps and concentrates on the low-noise steps, while a unit-weight MSE on the clean
  image weights every noise level equally. The choice of what to predict silently chooses how the
  finite training budget is spread across noise levels.

- **The high-noise regime is intrinsically hard for the clean-image target.** When `t` is large,
  `a -> 0` and `x_t` is almost pure noise: a single `x_t` is consistent with an enormous set of
  clean images, so the best-possible regressor onto `x_0` is the *conditional mean* `E[x_0 | x_t]`,
  which collapses toward the dataset mean — a smooth, low-detail image. There is little learnable
  signal there for the clean-image target, yet a flat (equal-weight) MSE on the clean image still
  asks the network to spend capacity matching it.

- **Strong off-the-shelf backbone.** A UNet encoder-decoder with residual blocks, group
  normalization, sinusoidal timestep embeddings and self-attention at coarse resolutions is the
  standard same-shape image-to-image network used as the denoiser; it maps a noisy image and a
  timestep to an image-shaped output. Nothing about it dictates which reading the output carries.

The prevailing default, established when these models first matched GANs on CIFAR-10, is to have
the network emit the noise `epsilon` and train with a flat MSE on it (the "simple" objective),
because that reading both connects to score matching and, with its implicit SNR weighting,
emphasizes the perceptually important low-noise steps. The early reports on this default also
record one diagnostic observation about the alternatives, included here because it is a
pre-method fact about the design space: among the readings tried, having the network emit the
clean image directly was observed to give *worse* sample quality early on, under the same flat
objective — a known wrinkle, not yet a settled verdict, since it had not been retested with the
matched loss weightings or the deterministic samplers that came later.

## Baselines

The alternatives a target-tensor choice is measured against, all sharing the fixed UNet, schedule,
training procedure, and DDIM sampler:

**Noise prediction (epsilon; Ho, Jain & Abbeel 2020).** The network emits `epsilon_theta(x_t, t)`
and is trained with the flat objective `|| epsilon - epsilon_theta(x_t, t) ||^2`. The `x_0`
estimate the sampler consumes is recovered by inverting the forward equation,
`x0_hat = (x_t - sqrt(1 - alpha_t) * epsilon_theta) / sqrt(alpha_t)`. As shown above this is, in
clean-image space, an `SNR_t`-weighted MSE: it down-weights the high-noise steps where `x_0` is
barely recoverable and up-weights the low-noise steps that carry fine detail, and the network
output doubles as a score estimate. **Gap / cost:** the recovery divides by `sqrt(alpha_t)`,
which goes to 0 as the noise level grows. So a small error in the emitted `epsilon` is amplified
by `1/sqrt(alpha_t)` in the implied clean-image estimate at high noise, and in the limit of pure
noise (`alpha_t = 0`) the emitted `epsilon` implies *no* clean-image estimate at all — the link
degenerates exactly where a few-step sampler must take its first, largest steps.

**Velocity prediction (v; Salimans & Ho 2022).** The network emits
`v_theta = sqrt(alpha_t) * epsilon - sqrt(1 - alpha_t) * x_0`, a fixed rotation of the
`(x_0, epsilon)` pair; in the angular variable `phi = arctan(sqrt(1-alpha_t)/sqrt(alpha_t))` it
is `v = cos(phi) epsilon - sin(phi) x_0`, the velocity of the point `z = cos(phi) x_0 +
sin(phi) epsilon` as it moves on the circle. The `x_0` estimate is recovered as
`x0_hat = sqrt(alpha_t) * x_t - sqrt(1 - alpha_t) * v_theta`. As a clean-image loss its implicit
weight is `(1 + SNR_t)`. **Gap / cost:** it is a deliberately balanced compromise between the
noise and clean-image readings rather than either extreme, so it neither maximally emphasizes the
perceptual low-noise steps (the way noise prediction does) nor places the regression target
purely in the space the sampler ultimately consumes; whether the balance it strikes is the best
target under a *flat* MSE and a fixed budget on this dataset is left open.

## Evaluation settings

The yardstick fixed across every target-tensor choice:

- **Dataset:** CIFAR-10, 32x32, unconditional; pixels mapped to `[-1, 1]`.
- **Backbone:** a `UNet2DModel` denoiser at three channel scales — small
  (`block_out_channels = (64, 128, 128, 128)`, ~9M params, batch 128), medium
  (`(128, 256, 256, 256)`, ~36M, batch 128), large (`(256, 512, 512, 512)`, ~140M, batch 64).
- **Training:** 35,000 steps per scale, AdamW at learning rate `2e-4`, EMA of weights with rate
  `0.9995`, multi-GPU DDP.
- **Sampler:** 50-step DDIM (Song, Meng & Ermon 2020), the deterministic sampler that at each
  step forms the clean-image estimate and moves toward the next noise level.
- **Metric:** FID computed by clean-fid against the 50,000-image CIFAR-10 train set; lower is
  better.
- **Protocol:** the target-tensor parameterization is the only thing that changes — architecture,
  dataset, optimizer, noise schedule, sampling procedure, and metric are all held fixed — and the
  contribution must be a transferable target parameterization expressed in those two coupled
  functions.

## Code framework

The training script is fixed except for one editable region: two coupled functions that define
what the network is trained to emit and how that emission is converted into the clean-image
estimate the DDIM sampler consumes. Everything around them — the data pipeline, the noising step,
the UNet forward pass, the MSE, the optimizer, the EMA, the DDIM loop — already exists and does
not change. The precomputed `schedule` dict provides the per-timestep noise-schedule tensors
(`alphas_cumprod`, `sqrt_alpha = sqrt(alphas_cumprod)`,
`sqrt_one_minus_alpha = sqrt(1 - alphas_cumprod)`), broadcastable to image tensors.

```python
import torch
import torch.nn.functional as F


# ---- already exists: schedule tensors, indexed by the per-example timestep ----
# schedule["alphas_cumprod"]        : alpha_t  (cumulative product of 1 - beta)
# schedule["sqrt_alpha"]            : sqrt(alpha_t)
# schedule["sqrt_one_minus_alpha"]  : sqrt(1 - alpha_t)


# =========================== editable region ===========================
def compute_training_target(x_0, noise, timesteps, schedule):
    # The tensor the network is trained to emit at (x_t, t).
    # TODO: choose the regression target.
    pass


def predict_x0(model_output, x_t, timesteps, schedule):
    # Convert the network's emission into the clean-image estimate the DDIM
    # sampler consumes. MUST exactly invert compute_training_target so the two
    # are consistent.
    # TODO: choose the matching x_0-recovery rule.
    pass
# ========================= end editable region =========================


# ---- already exists: one training step (fixed) ----
def training_step(model, x_0, schedule):
    noise = torch.randn_like(x_0)
    t = torch.randint(0, schedule["alphas_cumprod"].shape[0], (x_0.shape[0],), device=x_0.device)
    sqrt_alpha = schedule["sqrt_alpha"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)

    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise   # forward noising
    target = compute_training_target(x_0, noise, t, schedule)
    output = model(x_t, t).sample                           # UNet denoiser
    return F.mse_loss(output, target)                       # flat MSE on the chosen target


# ---- already exists: one DDIM step (fixed) — consumes predict_x0 ----
@torch.no_grad()
def ddim_step(model, x_t, t, t_prev, schedule):
    output = model(x_t, t).sample
    x0_hat = predict_x0(output, x_t, t, schedule)           # clean-image estimate
    a_t = schedule["sqrt_alpha"][t].view(-1, 1, 1, 1)
    b_t = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)
    a_prev = schedule["sqrt_alpha"][t_prev].view(-1, 1, 1, 1)
    b_prev = schedule["sqrt_one_minus_alpha"][t_prev].view(-1, 1, 1, 1)
    eps_hat = (x_t - a_t * x0_hat) / b_t                    # implied noise direction
    return a_prev * x0_hat + b_prev * eps_hat               # deterministic step to x_{t_prev}
```

The two empty functions are exactly what a target-tensor choice fills in; their bodies must line
up so that `predict_x0` inverts `compute_training_target`.
