# Context: choosing a stable denoising target for few-step diffusion sampling

## Research Question

A variance-preserving diffusion model corrupts a clean image `x_0` into a noisy latent

```text
x_t = a_t * x_0 + b_t * eps,        eps ~ N(0, I),
a_t^2 + b_t^2 = 1,
```

where `a_t` decreases from nearly one to nearly zero as the signal-to-noise ratio falls, and
`b_t` increases from nearly zero to nearly one. Sampling reverses this process: start from noise,
evaluate one denoising network at a sequence of noise levels, and update the latent toward an
image.

The design choice is what the network's raw output should represent. The sampler ultimately needs
a clean-image estimate at every step, but the network could be trained to emit a noise coordinate,
a data coordinate, or another invertible coordinate, as long as a matching conversion gives the
clean-image estimate used by the sampler. In the infinite-capacity limit these coordinates are
linearly related. Under a fixed U-Net, fixed optimizer, finite training budget and few sampling
steps, the choice changes both numerical conditioning and how the loss weights different
signal-to-noise levels. The goal is to fill two coupled functions: a training target and the
matching clean-image recovery rule.

## Background

The forward marginal has a closed-form reparameterization, so a training example at any noise level
is made by drawing one image, one Gaussian noise tensor and mixing them with the schedule. The
reverse transition is also Gaussian when conditioned on the clean image; in the DDPM notation its
posterior mean is a fixed linear combination of `x_t` and `x_0`:

```text
mu_tilde_t(x_t, x_0)
  = (sqrt(abar_{t-1}) * beta_t / (1 - abar_t)) * x_0
  + (sqrt(alpha_step_t) * (1 - abar_{t-1}) / (1 - abar_t)) * x_t.
```

All schedule coefficients are known. The learnable part is therefore a denoising estimate from the
current latent and noise level.

The standard DDPM target is the added noise. This is natural because for a Gaussian corruption the
score satisfies

```text
grad_{x_t} log q(x_t | x_0) = -(x_t - a_t * x_0) / b_t^2 = -eps / b_t.
```

Thus predicting `eps` is predicting the score up to a known scale. The deterministic DDIM sampler
then recovers a clean-image estimate and combines it with the implied noise direction:

```text
x_hat = (x_t - b_t * eps_hat) / a_t,
x_prev = a_prev * x_hat + b_prev * eps_hat.
```

This makes the output coordinate and recovery rule a single contract: the sampler is only correct
when the recovery exactly matches the target used in training.

## Baselines

Noise prediction is the dominant baseline. It trains with `||eps - eps_hat||^2`, and by substituting
the clean-image estimate into the forward equation,

```text
eps - eps_hat = (a_t / b_t) * (x_hat - x_0),
||eps - eps_hat||^2 = (a_t^2 / b_t^2) * ||x_hat - x_0||^2.
```

So flat noise MSE is signal-to-noise-ratio weighted clean-image MSE. It emphasizes high-SNR,
low-noise details and gives almost no clean-image weight to the pure-noise endpoint. The numerical
cost is the inverse conversion: as `a_t -> 0`, small output errors are amplified by `b_t / a_t`
when expressed as clean-image error, and at zero SNR the conversion no longer defines a clean-image
estimate at all.

The older direct denoising view points in the opposite direction: the reverse mean and DDIM update
consume a clean-image estimate, so one might place the regression directly in data space. That
removes the low-SNR division in the clean-image estimate, but it separates the parameterization
question from the loss-weighting question. Plain data-space MSE weights all noise levels equally,
including high-noise latents where the clean image is weakly identified by the observation. A
faithful experiment has to keep those two knobs distinct.

The remaining constraint is endpoint behavior. Few-step sampling exposes the noisy end of the
trajectory because the first steps are large and later corrections are scarce. Any target choice
that relies on dividing by a vanishing schedule coefficient can look acceptable with hundreds or
thousands of steps and fail when the sampler is compressed.

## Evaluation Setting

The target choice is evaluated in an unconditional CIFAR-10 diffusion pipeline with a U-Net
denoiser, variance-preserving cosine log-SNR schedule, Adam-family optimization, EMA weights,
gradient clipping, and DDIM sampling. The target parameterization is isolated from the rest of the
system: the architecture, noising equation, schedule, optimizer, clipping policy, sampler and FID
protocol are held fixed while the output interpretation and matching loss weighting are varied.

In the canonical implementation, CIFAR-10 uses a U-Net with 256 channels, three residual blocks per
resolution, attention at 16x16 and 8x8, batch size 128, learning rate `2e-4` for original training,
EMA `0.9999`, gradient clipping at 1.0, continuous-time training and DDIM evaluation. Distillation
starts from an 8192-step teacher and repeatedly halves the number of steps, using the same output
interpretation as the CIFAR teacher. Larger datasets use a combined data/noise output instead, so
the CIFAR setting is the relevant reference for the data-coordinate target.

## Code Scaffold

The scaffold already supplies the noising step, model call, MSE site and DDIM update. Only the
target tensor and the clean-image recovery rule are open:

```python
import torch
import torch.nn.functional as F


def compute_training_target(x_0, noise, timesteps, schedule):
    # Choose the tensor the network should emit for x_t at this timestep.
    pass


def predict_x0(model_output, x_t, timesteps, schedule):
    # Convert the raw network output into the clean-image estimate consumed by DDIM.
    # This must be the algebraic inverse of compute_training_target.
    pass


def training_step(model, x_0, schedule):
    noise = torch.randn_like(x_0)
    t = torch.randint(0, schedule["alphas_cumprod"].shape[0], (x_0.shape[0],), device=x_0.device)
    a_t = schedule["sqrt_alpha"][t].view(-1, 1, 1, 1)
    b_t = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)

    x_t = a_t * x_0 + b_t * noise
    target = compute_training_target(x_0, noise, t, schedule)
    output = model(x_t, t).sample
    return F.mse_loss(output, target)


@torch.no_grad()
def ddim_step(model, x_t, t, t_prev, schedule):
    output = model(x_t, t).sample
    x0_hat = predict_x0(output, x_t, t, schedule)
    a_t = schedule["sqrt_alpha"][t].view(-1, 1, 1, 1)
    b_t = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)
    a_prev = schedule["sqrt_alpha"][t_prev].view(-1, 1, 1, 1)
    b_prev = schedule["sqrt_one_minus_alpha"][t_prev].view(-1, 1, 1, 1)
    eps_hat = (x_t - a_t * x0_hat) / b_t
    return a_prev * x0_hat + b_prev * eps_hat
```

If the surrounding harness exposes a per-timestep loss-weighting hook, that hook is separate from
these two functions. The functions decide the output coordinate and inverse conversion; loss
weights decide how strongly each noise level trains that coordinate.
