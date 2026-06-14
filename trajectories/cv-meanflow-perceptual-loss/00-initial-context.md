## Research question

Flow matching trains a network to predict a velocity field that transports Gaussian noise to
data, and **MeanFlow** is the variant that learns the *average* velocity over a time interval so
that generation collapses to very few steps. The MeanFlow training objective, the DiT backbone,
the data pipeline, and the ten-step Euler sampler are all fixed. The single thing being designed
is the **training loss** that sits on top of the predicted mean velocity: starting from the
canonical mean-squared error on the velocity, can an *auxiliary* loss — applied to the image the
velocity implies — improve the FID of the generated CIFAR-10 samples without destabilizing the
velocity target? Everything else about the model and the evaluation is frozen.

The leverage point is one identity the harness already exposes. The predicted mean velocity
implies a denoised image at every noise level,

```
x_denoised = x_t - t * pred_mean_vel,
```

so any image-space distance — perceptual, gradient-domain, multiscale, frequency-domain — can be
turned into a differentiable signal on `x_denoised` and added to the MSE. The whole ladder is a
sequence of choices about *what to measure on `x_denoised`* and *how to schedule it across the
noise level `t`*.

## Prior art before the first rung (loss-design lineage)

The first rung is the floor loss — pure MSE on the velocity — and it reacts against a single,
concrete pathology that earlier attempts on this exact scaffold ran into, plus the standard image
losses the later rungs reach for.

- **Per-pixel / per-coordinate squared error (the `L2` floor).** Squared error treats every
  coordinate as independent: as a *velocity* target it is the principled MeanFlow regression
  signal, but on *images* it is blind to structure (a faint blur is a tiny `L2` change yet a large
  perceptual one) and, on multi-modal targets, it drives a predictor to the blurry conditional
  mean. Gap: correct for the velocity, but image quality is left entirely to the backbone.
- **Loss-adaptive (inverse-loss) reweighting.** A tempting refinement is to divide each sample's
  MSE by its own magnitude, `weight = 1 / (loss_mse.detach() + 1e-3)`, so the optimizer spends more
  on hard samples. On this scaffold that reverse-focal weighting amplifies *easy* samples and the
  run diverges around step 35-40k. Gap: a reweighting that looks principled actively destabilizes
  this MeanFlow loop — it is the pathology the floor must remove.
- **Perceptual feature-space losses (Gatys 2015; Johnson 2016; LPIPS, Zhang 2018).** Distance in
  the activations of a frozen ImageNet-trained network tracks human similarity far better than
  pixels and is differentiable, so it can supervise `x_denoised`. The harness exposes it as
  `lpips_fn`. Gap on its own: a feature-space distance is spatially tolerant, so it leaves fine
  edges and the high-frequency spectrum under-supervised.
- **Gradient-domain and multiscale image losses (Mathieu 2016; LapSRN, Lai 2017).** A
  finite-difference (Sobel) edge `L1` forces edges to land where the target's edges are, and a
  multi-resolution downsampled `L1` supervises coarse layout; the harness exposes both as
  `compute_gradient_loss` and `compute_multiscale_loss`. Gap: both are spatial-domain and mostly
  local, so the diffuse high-frequency deficit a generator accumulates is seen only weakly.
- **Frequency-domain losses (Fourier-space terms; Mathieu 2016; Fuoli 2021).** Transforming
  `x_denoised` and the clean image by the FFT and comparing spectra concentrates the diffuse
  high-frequency deficit into a focused error exactly where it lives. Gap of the *naive* version:
  comparing only FFT magnitude with a uniform `L1` discards phase and weights every frequency
  equally.

## The fixed substrate

A self-contained MeanFlow training script, `custom_train_perceptual.py`, is frozen and must not be
touched outside the loss region. It trains a `SmallDiT` (~512 hidden, ~8 layers, ~40M params;
Peebles & Xie 2023) on CIFAR-10 (32x32). Each step samples trajectory parameters
`t, t_next, dt, alpha` from a logit-normal scheme (`sample_traj_params`), forms the noised image
`x_t = (1 - t) * x + t * noise` and the instantaneous velocity `velocity = noise - x`, computes the
MeanFlow mean-velocity target via a single JVP (`compute_mean_velocity_target`), and predicts
`pred_mean_vel = net(x_t, sigma=t, sigma_next=t_next)`. The optimizer (AdamW, lr 2e-4, weight decay
1e-4), the AMP `GradScaler`, the global grad-norm clip at 1.0, and the ten-step Euler sampler are
all fixed. Sampling FID is computed by clean-fid against the CIFAR-10 train set (lower is better).

The loop also pre-builds the helpers a loss may use, all expecting images on `[-1, 1]`:

- `lpips_fn(x_denoised, x_target)` — LPIPS perceptual loss (frozen VGG backbone).
- `compute_gradient_loss(x_pred, x_target)` — Sobel-style gradient-domain `L1` (returns a scalar).
- `compute_multiscale_loss(x_pred, x_target)` — multi-resolution downsampled MSE (returns a scalar).

## The editable interface

Exactly one region is editable — the loss computation inside the training loop (lines 384-401 of
`custom_train_perceptual.py`). Every method on the ladder is a fill of this same contract: from the
in-scope tensors `pred_mean_vel`, `mean_vel_target`, `x` (clean image, `[-1, 1]`), `x_t`, `t`, the
helpers above, and `device`, assign a scalar `loss` that will be back-propagated. Auxiliary losses
on `x_denoised = x_t - t * pred_mean_vel` must be applied only where `x_denoised` is numerically
meaningful: at very small `t` the implied image is ill-conditioned and the auxiliary gradients
dominate the velocity target, so a `t <= 0.1` mask is required.

The starting point is the scaffold default: **pure MSE on the mean velocity**. Each later method
replaces exactly this region and nothing else.

```python
# EDITABLE region of custom_train_perceptual.py (lines 384-401) — default fill
# Current: pure MSE on velocity.
loss_mse = ((pred_mean_vel - mean_vel_target) ** 2).mean()
loss = loss_mse
```

## Evaluation settings

Training runs at three scales / budgets — `small`, `medium`, `large` — each on CIFAR-10 with batch
size 128 and the fixed ten-step Euler sampler, seed 42. The metric is FID per scale (`best_fid_small`,
`best_fid_medium`, `best_fid_large`), computed by clean-fid against the CIFAR-10 train set; **lower
is better** on all three. A useful loss improves visual sample quality (lower FID) without
destabilizing the velocity target, and applies auxiliary terms only where `x_denoised` is
numerically meaningful. The architecture, data pipeline, sampler, number of evaluation steps, and
metric computation are not to be changed.
