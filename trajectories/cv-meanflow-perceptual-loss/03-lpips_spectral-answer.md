**Problem.** The spatial perceptual stack (FID 17.79 / 17.19 / 14.49) closed most of the floor's
image-quality gap but plateaued: all three of its terms (LPIPS, Sobel gradient, multiscale) are
spatial-domain and mostly local, so the diffuse high-frequency energy a generator under-produces —
spread thin across space — is seen only weakly. That residual is concentrated in the frequency
domain.

**Key idea (spatial + frequency).** Add a frequency-domain term to the existing spatial stack, on
the same `x_denoised = x_t - t * pred_mean_vel`, under the same `(1 - t)^2 * 1[t > 0.1]` schedule and
`[-1, 1]` clamp (the schedule reasoning is path-determined and carries over). Take the real-input
2D FFT (`rfft2`, half-spectrum by Hermitian symmetry) of `x_denoised` and the clean image per
channel, take amplitude, and average the `L1` of the difference. Penalize **magnitude** (the missing
energy) not phase (the spatial terms handle position and amplitude is shift-tolerant); use `L1` to
match the robust flavor and avoid low-frequency-coefficient dominance.

**Why it differs from step 2.** The frequency term is global guidance aimed at exactly what the
local spatial terms miss. This rung also *drops* the step-2 Charbonnier velocity term back to a clean
unscaled MSE anchor — fewer interacting knobs on the correctness target, so the perceptual gains are
attributable entirely to the image terms.

**Weights.** Velocity MSE unscaled (anchor). Image terms by centrality under the shared schedule:
LPIPS 0.5 (primary) > gradient 0.3 (edges) > multiscale 0.2 (coarse) = spectral 0.2 (high-frequency
energy). The new spectral term shares multiscale's weight because it plays the same complementary
(not primary) role.

**Hyperparameters.** MSE weight 1.0; no Charbonnier; schedule `(1 - t)^2 * 1[t > 0.1]`; LPIPS 0.5,
gradient 0.3, multiscale 0.2, spectral 0.2. `x_denoised` clamped `[-1, 1]`; helpers and FFT run only
on the `t > 0.1` subset.

**What to watch.** FID should fall below 17.79 / 17.19 / 14.49 at every scale, by a *smaller* margin
than the floor-to-spatial jump (a refinement on the residual, not a new primary signal), with the
largest relative help at the larger budgets. If it fails to move, the spatial stack was already
capturing the spectrum; if it rises, the magnitude-only, uniformly-weighted `L1` is too crude — which
points at the principled fix beyond this rung: weight the hard-to-synthesize frequencies and use the
full complex spectrum.

```python
# EDITABLE region of custom_train_perceptual.py (lines 384-401) — step 3: spatial + frequency
# MSE on velocity
err = pred_mean_vel - mean_vel_target
loss_mse_unscaled = (err ** 2).flatten(1).mean(1)

# Auxiliary perceptual losses on denoised image (mask t<=0.1 edge case)
x_denoised = x_t - t * pred_mean_vel
t_flat = t.view(B)
mask = (t_flat > 0.1)
perceptual_w = ((1.0 - t_flat) ** 2) * mask.float()

loss_lpips = torch.zeros(B, device=device)
loss_grad = torch.zeros(B, device=device)
loss_multi = torch.zeros(B, device=device)
loss_spec = torch.zeros(B, device=device)
if mask.any():
    xd = x_denoised[mask].clamp(-1, 1).float()
    xc = x[mask].clamp(-1, 1).float()
    loss_lpips[mask] = lpips_fn(xd, xc).view(-1).float()
    loss_grad[mask] = compute_gradient_loss(xd, xc).float()
    loss_multi[mask] = compute_multiscale_loss(xd, xc).float()
    # FFT magnitude L1: per-channel rfft2, abs, L1 of difference
    fd = torch.fft.rfft2(xd, dim=(-2, -1)).abs()
    fc = torch.fft.rfft2(xc, dim=(-2, -1)).abs()
    loss_spec[mask] = (fd - fc).abs().mean(dim=(1, 2, 3)).float()

loss_total = (
    loss_mse_unscaled
    + perceptual_w * (
        0.5 * loss_lpips
        + 0.3 * loss_grad
        + 0.2 * loss_multi
        + 0.2 * loss_spec
    )
)
loss = loss_total.mean()
```
