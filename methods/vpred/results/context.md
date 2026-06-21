# Context: choosing what a denoising network predicts in a diffusion model (circa 2021)

## Research question

A diffusion model corrupts a clean image `x` into a noisy latent
`z_t = alpha_t * x + sigma_t * eps` along a schedule of signal-to-noise ratios, and trains one
neural network to undo the corruption. The network is shown `z_t` (and the noise level) and is
optimized by a squared-error loss, then a sampler runs the trained denoiser backwards from pure
noise to an image. A free design choice sits at the center of this: **what quantity should the
network's output represent?** The network could be read as predicting the clean image, the
added noise, or something else, and a fixed conversion turns its output into the clean-image
estimate `x_hat` that the sampler actually consumes at every step. The question is how to choose
the predicted quantity, the training loss, and the corresponding inverse conversion used at
sampling time.

## Background

By 2021 the diffusion / score-based generative model is mature and its moving parts are well
understood.

- **The forward process and its marginals.** With a variance-preserving schedule one sets
  `sigma_t^2 = 1 - alpha_t^2`, so `alpha_t^2 + sigma_t^2 = 1` and the noised latent
  `z_t = alpha_t * x + sigma_t * eps` (with `eps ~ N(0, I)`) has unit-scale marginals at every
  `t`. The log signal-to-noise ratio `lambda_t = log(alpha_t^2 / sigma_t^2)` runs monotonically
  from large positive (almost-clean, `alpha_t -> 1`, `sigma_t -> 0`) to large negative
  (almost-pure-noise, `alpha_t -> 0`, `sigma_t -> 1`). Kingma, Salimans, Poole & Ho (2021)
  formalize the continuous-time version in terms of `lambda_t`, and show the variance-preserving
  and variance-exploding specifications are equivalent up to a rescaling of `z_t`, so working in
  the variance-preserving frame is without loss of generality.
- **Denoising as regression.** Training minimizes a weighted mean-squared error between the
  network's implied clean-image estimate `x_hat_theta(z_t)` and the true `x`, over noise levels
  sampled along the schedule: `E[ w(lambda_t) * || x_hat_theta(z_t) - x ||^2 ]`. This is
  justified both as a weighted variational bound on the data log-likelihood (Kingma et al. 2021)
  and as denoising score matching (Vincent 2011; Song & Ermon 2019). The weighting `w(lambda_t)`
  decides how much each noise level contributes to the gradient, and is itself a design choice.
- **The denoising/score/noise identities.** For a Gaussian corruption, the score
  `grad_z log q(z_t)` is, up to scale, the negative of the added noise, so "predict the noise"
  and "estimate the score" and "denoise" are the same task viewed three ways (Vincent 2011).
  This is exactly why the predicted-quantity choice is *available*: noise, score and clean image
  are linearly interconvertible given `z_t` and the schedule.
- **The deterministic ODE view.** Song et al. (2021) show the trained denoiser defines a
  probability-flow ODE whose solution maps noise to data deterministically; numerically
  integrating it generates samples, and the integration error vanishes as the step count grows.
  In this view the denoiser must define a *smooth* vector field across noise levels.
- **The few-step pressure.** Drawing a sample requires running the denoiser many times in
  sequence, which is the dominant cost. There is heavy pressure to cut the number of sampling
  steps to a few dozen or fewer.
- **Cosine noise schedules.** Nichol & Dhariwal (2021) replace the original linear beta schedule
  with a cosine cumulative schedule, using a small offset in their implementation to avoid the
  endpoint singularity. In a continuous variance-preserving harness the same idea can be written
  as the simpler `alpha_t = cos(0.5 * pi * t)`, `sigma_t = sin(0.5 * pi * t)`, so the path spans
  pure signal at one end and pure noise at the other.

## Baselines

These are the predicted-quantity choices already on the table, each with its conversion to
`x_hat` and the loss it implies.

**Noise prediction (Ho, Jain & Abbeel 2020).** The network output is read as the added noise
`eps_hat_theta(z_t)`, and the clean-image estimate is recovered by inverting
`z_t = alpha_t * x + sigma_t * eps`:

```
x_hat_theta(z_t) = (1 / alpha_t) * (z_t - sigma_t * eps_hat_theta(z_t))
```

The training loss is mean-squared error in noise space, `|| eps - eps_hat_theta(z_t) ||^2`,
which equals a clean-image reconstruction loss weighted by the signal-to-noise ratio:
`|| eps - eps_hat ||^2 = (alpha_t^2 / sigma_t^2) * || x - x_hat ||^2`, i.e. `w(lambda_t) =
exp(lambda_t)`. This is the dominant choice and produces excellent samples in the many-step
regime.

**Clean-image prediction.** The network output is read directly as the clean image
`x_hat_theta(z_t)`, with the identity conversion (the output *is* `x_hat`). The loss is
`|| x - x_hat_theta(z_t) ||^2`. The implied noise estimate is recovered as
`eps_hat = (z_t - alpha_t * x_hat) / sigma_t`.

**Loss weightings already in use.** Beyond the predicted quantity, the loss weighting is its own
lever. The standard signal-to-noise weighting `w(lambda_t) = exp(lambda_t)` concentrates weight
at high SNR; uniform weighting (`w = 1`) treats all noise levels equally.

## Evaluation settings

The natural yardstick is unconditional image generation under a fixed backbone and pipeline,
varying only the predicted-quantity / loss choice.

- **Dataset:** CIFAR-10, 32x32, unconditional (50,000 train images). Larger-resolution
  class-conditional sets (downsampled ImageNet, LSUN) are the secondary stress tests.
- **Backbone:** a U-Net denoiser (residual blocks, group normalization, scalar noise-level
  embedding, self-attention at coarse resolutions), held fixed across the parameterization
  choices and trained at several channel widths.
- **Schedule / training:** variance-preserving cosine schedule `alpha_t = cos(0.5 * pi * t)`,
  `t ~ U[0, 1]`; AdamW; an exponential moving average of the weights; a fixed number of
  optimization steps per setting; multi-device data-parallel training.
- **Sampler:** deterministic DDIM (Song, Meng & Ermon 2021) with a fixed, small number of steps
  (tens of steps).
- **Metric:** Fréchet Inception Distance against the training set (50,000 samples), lower is
  better; Inception Score as a secondary readout. The protocol holds backbone, dataset,
  optimizer, schedule, sampler and metric fixed, isolating the effect of the predicted-quantity
  and loss choice.

## Code framework

The harness already exists: a training loop that draws a clean batch, samples a noise level and
Gaussian noise, forms the noised latent `z_t = sqrt(alpha) * x_0 + sqrt(1 - alpha) * eps`, runs
the fixed U-Net on `z_t`, and minimizes a squared error between the network output and a
*target*; and a DDIM sampler that, at each step, needs the model's clean-image estimate `x_0`
recovered from the raw network output. Two coupled slots are left open — one defining what the
network is trained to output, and one defining how the sampler turns that output back into a
clean-image estimate. They must be consistent inverses: the recovery used at sampling time has
to correctly undo the target used at training time. The schedule tensors `sqrt_alpha =
sqrt(alphas_cumprod)` and `sqrt_one_minus_alpha = sqrt(1 - alphas_cumprod)` are precomputed and
available to both.

```python
import torch
import torch.nn.functional as F


def compute_training_target(x_0, noise, timesteps, schedule):
    """What the network is trained to predict for a noised input
    z_t = sqrt(alpha)*x_0 + sqrt(1-alpha)*noise.

    schedule["sqrt_alpha"][t]            = sqrt(alphas_cumprod[t])
    schedule["sqrt_one_minus_alpha"][t]  = sqrt(1 - alphas_cumprod[t])
    """
    # TODO: define the regression target the network output should match.
    pass


def predict_x0(model_output, x_t, timesteps, schedule):
    """Recover the clean-image estimate x_0 from the raw network output,
    used by the DDIM sampler at every step. Must be the consistent inverse
    of compute_training_target above."""
    # TODO: convert the network output back into a clean-image estimate.
    pass


# existing training step the two functions plug into
def train_step(model, x_0, schedule, optimizer):
    noise = torch.randn_like(x_0)
    t = torch.randint(0, schedule["num_steps"], (x_0.shape[0],), device=x_0.device)
    sqrt_a = schedule["sqrt_alpha"][t].view(-1, 1, 1, 1)
    sqrt_1ma = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)

    x_t = sqrt_a * x_0 + sqrt_1ma * noise          # forward noising
    target = compute_training_target(x_0, noise, t, schedule)
    model_output = model(x_t, t).sample
    loss = F.mse_loss(model_output, target)        # plain MSE on the chosen target

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


# existing DDIM sampling step the recovery plugs into
@torch.no_grad()
def ddim_step(model, x_t, t, t_prev, schedule):
    sqrt_a_prev = schedule["sqrt_alpha"][t_prev].view(-1, 1, 1, 1)
    sqrt_1ma_prev = schedule["sqrt_one_minus_alpha"][t_prev].view(-1, 1, 1, 1)
    sqrt_1ma = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)

    model_output = model(x_t, t).sample
    x0 = predict_x0(model_output, x_t, t, schedule)        # clean-image estimate
    eps = (x_t - schedule["sqrt_alpha"][t].view(-1, 1, 1, 1) * x0) / sqrt_1ma
    return sqrt_a_prev * x0 + sqrt_1ma_prev * eps          # deterministic DDIM update
```

The two `# TODO` bodies are the only open slots; everything else — the noising, the MSE, the
DDIM update — is fixed.
