# x_0-prediction (data / "x" parameterization), distilled

x0pred is a denoising-diffusion **target parameterization**: the network is trained to emit the
clean image `x_0` directly (rather than the noise `epsilon` or the velocity `v`), and the
clean-image estimate the sampler consumes is recovered by the identity. It is one of the three
standard, mutually interchangeable parameterizations of the same diffusion model, distinguished
only by what tensor the network outputs and how that output is converted back to `x_0`.

## Problem it solves

The reverse step of a diffusion model is a denoiser. Conditioned on `x_0`, the forward process
reverses in closed form with posterior mean linear in `x_0` and `x_t`, so the only learnable
quantity is `x_0`; a deterministic sampler (DDIM) forms an estimate of `x_0` at every step.
The free choice is *what the network emits* and the matching `x_0`-recovery rule — the two must
be consistent (the recovery inverts the target). The candidates are algebraically equivalent in
the infinite-capacity limit but, under a fixed UNet / step budget / sampler, allocate the
training budget across noise levels differently and give different FID.

## Key idea

At a fixed `t`, with `a = sqrt(alpha_t)`, `b = sqrt(1 - alpha_t)`, `a^2 + b^2 = 1`, and
`x_t = a*x_0 + b*epsilon`:

```
x_0     = (x_t - b*epsilon) / a
epsilon = (x_t - a*x_0) / b
v       = a*epsilon - b*x_0
```

so all three readings are linear bijections given `x_t`. The reading sets the loss weighting,
because flat MSE on one reading is a `t`-weighted MSE on another. Exactly:

```
|| epsilon - epsilon_hat ||^2 = (a^2 / b^2) || x_0 - x0_hat ||^2 = SNR_t * || x_0 - x0_hat ||^2,
SNR_t = alpha_t / (1 - alpha_t).
```

- **epsilon** = flat MSE on noise = `SNR_t`-weighted clean-image MSE: emphasizes low-noise
  (high-SNR) perceptual detail, near-ignores high-noise steps. But its `x_0` recovery
  `x0_hat = (x_t - b*epsilon_hat)/a` divides by `a = sqrt(alpha_t) -> 0`, so a small output
  error is amplified by `1/sqrt(alpha_t)` at high noise (and is undefined at `alpha_t = 0`) —
  unstable for a few-step sampler.
- **x_0 (x0pred)** = flat MSE on the clean image = equal weight across all noise levels. The
  network output **is** the `x_0` estimate the sampler wants, so recovery is the **identity** —
  no `1/sqrt(alpha_t)` amplification, implied clean prediction stable at every noise level. Its
  cost: flat weighting also spends budget on high-noise steps where `x_0` is barely recoverable
  from `x_t` (the optimal regressor there collapses toward `E[x_0|x_t]`, a blur), which is why
  predicting `x_0` under flat MSE historically trailed `epsilon` on sample quality.
- **v** = flat MSE on velocity = `(1 + SNR_t)`-weighted clean-image MSE: the balanced
  compromise between the two, with stable implied `x_0`.

x0pred is the sampler-aligned corner: predict exactly what DDIM consumes, recover it trivially.

## Final form

The two coupled functions in the editable region; `predict_x0` exactly inverts
`compute_training_target` (here, trivially):

```python
def compute_training_target(x_0, noise, timesteps, schedule):
    # X0-prediction: the network directly predicts the clean image.
    return x_0


def predict_x0(model_output, x_t, timesteps, schedule):
    # The output IS x_0 -- recovery is the identity, no conversion, and no
    # 1/sqrt(alpha_t) amplification at high noise.
    return model_output
```

These plug into the fixed pipeline unchanged. Training:

```python
import torch
import torch.nn.functional as F


def training_step(model, x_0, schedule):
    noise = torch.randn_like(x_0)
    t = torch.randint(0, schedule["alphas_cumprod"].shape[0], (x_0.shape[0],), device=x_0.device)
    sqrt_alpha = schedule["sqrt_alpha"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)

    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise        # forward noising
    target = compute_training_target(x_0, noise, t, schedule)    # == x_0
    output = model(x_t, t).sample                               # UNet denoiser
    return F.mse_loss(output, target)                          # flat MSE on x_0
```

DDIM sampling (deterministic), consuming `predict_x0` to get the clean-image estimate each step:

```python
@torch.no_grad()
def ddim_step(model, x_t, t, t_prev, schedule):
    output = model(x_t, t).sample
    x0_hat = predict_x0(output, x_t, t, schedule)              # == output
    a_t = schedule["sqrt_alpha"][t].view(-1, 1, 1, 1)
    b_t = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)
    a_prev = schedule["sqrt_alpha"][t_prev].view(-1, 1, 1, 1)
    b_prev = schedule["sqrt_one_minus_alpha"][t_prev].view(-1, 1, 1, 1)
    eps_hat = (x_t - a_t * x0_hat) / b_t                       # implied noise direction
    return a_prev * x0_hat + b_prev * eps_hat                  # step to x_{t_prev}
```

## Fixed evaluation pipeline (held constant across parameterizations)

CIFAR-10 32x32 unconditional, pixels in `[-1, 1]`; `UNet2DModel` at small/medium/large channel
scales; 35,000 training steps; AdamW lr `2e-4`; weight EMA rate `0.9995`; 50-step DDIM sampler;
FID via clean-fid against the 50k-image train set (lower better). Only the target
parameterization changes.
