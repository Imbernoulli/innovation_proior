**Problem.** The pure-MSE floor (FID 22.33 / 21.91 / 18.363) trains a competent velocity field but
leaves the samples perceptually soft — velocity MSE is structurally blind to image quality. The fix
must be an image-space signal on the denoised image the velocity implies, without destabilizing the
velocity anchor.

**Key idea (spatial-domain perceptual stack).** Apply perceptual losses to
`x_denoised = x_t - t * pred_mean_vel` against the clean image `x`, scheduled so they act only where
`x_denoised` is a faithful target. The schedule is *derived*: since
`x_denoised - x = t * (v_target - pred_mean_vel)`, the image-space gradient back to the velocity is
scaled by `t`, so small `t` is a weak-leverage, ill-conditioned regime. Hence
`perceptual_w = (1 - t)^2 * 1[t > 0.1]` — rise toward the clean endpoint, hard-gate the tiny-`t`
region. Three complementary image terms, each from where the previous is blind: **LPIPS** (primary
feature-space perceptual signal, spatially tolerant), a **Sobel gradient `L1`** (forces sharp edges
LPIPS forgives), a **multiscale `L1`** (coarse structure the fine terms miss). A small
**Charbonnier** robust `L1` rides on the velocity error as a hedge against MSE's outlier
sensitivity.

**Why these weights.** The velocity MSE stays unscaled — the correctness anchor; the network's job
is still to predict the right mean velocity. Charbonnier `0.1` (tail hedge). The image terms by
centrality: `0.5` LPIPS > `0.3` gradient > `0.2` multiscale, all well below the anchor. `x_denoised`
is clamped to `[-1, 1]` (VGG backbone range), image helpers run only on the `t > 0.1` subset, and
per-sample zero losses keep masked-out samples contributing only the velocity terms.

**Hyperparameters.** MSE weight 1.0 (unscaled); Charbonnier 0.1 (`eps^2 = 1e-6`); schedule
`(1 - t)^2 * 1[t > 0.1]`; LPIPS 0.5, gradient 0.3, multiscale 0.2.

**What to watch.** FID should fall below the floor at every scale, with stability and
budget-monotonicity preserved (no leak of image gradients into the velocity target); the largest
relative gain at the larger budgets where the velocity is already well-fit. The expected residual
weakness — high-frequency texture, since all three image terms are spatial-domain and mostly local —
is what motivates a frequency-domain term at step 3.

```python
# EDITABLE region of custom_train_perceptual.py (lines 384-401) — step 2: LPIPS + grad + multiscale
# MSE on velocity + Charbonnier smooth-L1 pixel loss on velocity
err = pred_mean_vel - mean_vel_target
loss_mse_unscaled = (err ** 2).flatten(1).mean(1)
loss_charb = torch.sqrt(err ** 2 + 1e-6).flatten(1).mean(1)

# Auxiliary perceptual losses on denoised image (mask t<=0.1 edge case)
x_denoised = x_t - t * pred_mean_vel
t_flat = t.view(B)
mask = (t_flat > 0.1)
perceptual_w = ((1.0 - t_flat) ** 2) * mask.float()

loss_lpips = torch.zeros(B, device=device)
loss_grad = torch.zeros(B, device=device)
loss_multi = torch.zeros(B, device=device)
if mask.any():
    xd = x_denoised[mask].clamp(-1, 1).float()
    xc = x[mask].clamp(-1, 1).float()
    loss_lpips[mask] = lpips_fn(xd, xc).view(-1).float()
    loss_grad[mask] = compute_gradient_loss(xd, xc).float()
    loss_multi[mask] = compute_multiscale_loss(xd, xc).float()

loss_total = (
    loss_mse_unscaled
    + 0.1 * loss_charb
    + perceptual_w * (0.5 * loss_lpips + 0.3 * loss_grad + 0.2 * loss_multi)
)
loss = loss_total.mean()
```
