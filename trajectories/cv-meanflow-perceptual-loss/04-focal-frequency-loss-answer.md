**Problem.** The strongest baseline (FID 17.38 / 15.82 / 13.63) added a frequency term built as
crudely as possible — a uniform `L1` on FFT *magnitude*. That term discards phase and, fatally,
weights every frequency equally, so the few large low-frequency coefficients dominate the mean while
the small high-frequency coefficients — the band the term was added to fix — barely contribute. The
residual is a frequency loss that under-attends the hardest-to-synthesize frequencies.

**Key idea (Focal Frequency Loss, Jiang et al., ICCV 2021, arXiv:2012.12821).** Replace *only* the
naive spectral term with the focal frequency loss, fixing both weaknesses. (1) Compare the **full
complex** spectrum: the per-frequency distance is the squared complex difference
`d(u,v) = (Δreal)^2 + (Δimag)^2` (amplitude *and* phase). (2) **Adaptively focus** on hard
frequencies with a per-frequency weight built from the current error itself,
`w(u,v) = |F_target - F_pred|^alpha = d(u,v)^(alpha/2)`, normalized per-image to `[0, 1]` and
**stop-gradiented** so it acts as a fixed importance mask — gradient flows only through `d`, never
through `w` (the focal-loss discipline; otherwise the model could lower the loss by shrinking the
weight). FFL = `mean(w * d)` over coefficients and channels. Default `alpha = 1.0`, orthonormal
2D DFT.

**What changes vs step 3.** Exactly one term: the magnitude-`L1` spectral term is swapped for FFL at
the *same* `0.2` weight and the *same* scheduled, masked, `[-1, 1]`-clamped slot on `x_denoised`. The
velocity MSE anchor (unscaled), LPIPS (0.5), gradient (0.3), multiscale (0.2), the
`(1 - t)^2 * 1[t > 0.1]` schedule, and the mask are all unchanged — keeping the term's construction
the only variable so the FID change is attributable to the focal mechanism, not a reweighting.

**Hyperparameters.** FFL `alpha = 1.0`, `norm='ortho'`, online error-adaptive weight matrix
(detached, per-image max-normalized, clamped `[0, 1]`, NaNs zeroed), term weight 0.2 under the shared
schedule; everything else as step 3.

**Bar to clear / what to validate.** Must beat 17.38 / 15.82 / 13.63 at every scale, with the gain
concentrated at the larger budgets (most purely high-frequency residual). Expected magnitude is
modest — a refinement of one of four image terms — so a few tenths of FID per scale, most clearly at
`large` (below 13.63). Validate stability first (the detached, `[0, 1]`-normalized weight cannot
blow the term up) and attribution (real gain at the unchanged 0.2 weight). No improvement would
falsify the claim that error-adaptive frequency weighting matters at CIFAR-10 scale.

```python
# EDITABLE region of custom_train_perceptual.py (lines 384-401) — finale: Focal Frequency Loss
# MSE on velocity (unscaled correctness anchor)
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
loss_ffl = torch.zeros(B, device=device)
if mask.any():
    xd = x_denoised[mask].clamp(-1, 1).float()
    xc = x[mask].clamp(-1, 1).float()
    loss_lpips[mask] = lpips_fn(xd, xc).view(-1).float()
    loss_grad[mask] = compute_gradient_loss(xd, xc).float()
    loss_multi[mask] = compute_multiscale_loss(xd, xc).float()

    # Focal Frequency Loss (Jiang et al., ICCV 2021) on x_denoised vs clean image.
    # Full complex spectrum (amplitude + phase), error-adaptive per-frequency
    # focusing weight (detached), alpha = 1.0.
    alpha = 1.0
    freq_d = torch.fft.fft2(xd, dim=(-2, -1), norm='ortho')
    freq_c = torch.fft.fft2(xc, dim=(-2, -1), norm='ortho')
    d_real = freq_d.real - freq_c.real
    d_imag = freq_d.imag - freq_c.imag
    freq_distance = d_real ** 2 + d_imag ** 2          # |F_pred - F_target|^2 per (c,u,v)

    # Spectrum weight matrix: w = (sqrt(freq_distance))^alpha, per-image max-normalized,
    # clamped to [0,1], NaNs->0, and STOP-GRADIENTED (gradient flows only through d).
    weight = torch.sqrt(freq_distance + 1e-12) ** alpha
    w_max = weight.amax(dim=(-2, -1), keepdim=True)
    weight = weight / (w_max + 1e-12)
    weight = torch.nan_to_num(weight, nan=0.0).clamp(0.0, 1.0).detach()

    loss_ffl[mask] = (weight * freq_distance).mean(dim=(1, 2, 3)).float()

loss_total = (
    loss_mse_unscaled
    + perceptual_w * (
        0.5 * loss_lpips
        + 0.3 * loss_grad
        + 0.2 * loss_multi
        + 0.2 * loss_ffl
    )
)
loss = loss_total.mean()
```
