**Problem.** MeanFlow trains a DiT to predict the mean velocity that the ten-step Euler sampler
integrates into images; the editable region is the loss on `pred_mean_vel` vs `mean_vel_target`.
The floor must train this objective *correctly and stably* and establish the honest MSE-only
reference that every auxiliary loss has to beat — exploration of image quality is left to a later
rung.

**Key idea (the floor loss).** Pure mean squared error on the mean velocity, with no adaptive
reweighting. The principled-looking refinement — divide each sample's MSE by its own magnitude,
`weight = 1 / (loss_mse.detach() + 1e-3)` — is a *reverse*-focal weighting: it amplifies easy
samples (small denominator -> huge weight) and, as most samples become easy, a few near-solved
samples blow up the gradient and the run diverges around step 35-40k. The floor removes that
pathology and reduces with a clean unweighted mean, keeping a per-sample vector so later rungs can
add per-sample scheduled auxiliaries on top.

**Why it is the floor.** Squared error on the velocity is the correct MeanFlow regression signal
but is blind to image structure: it never tells the network that its implied denoised image
`x_denoised = x_t - t * pred_mean_vel` should be sharp rather than the blurry conditional mean.
The leverage to fix that — image-space losses on `x_denoised` via `lpips_fn`,
`compute_gradient_loss`, `compute_multiscale_loss` — exists in the substrate but is deliberately
unused here.

**Hyperparameters.** No auxiliary terms, no schedule, no `t`-mask (none is needed without an
image-space term). Reduction: per-sample flatten-mean of the squared velocity error, then batch
mean.

**What to watch.** FID should be a real (non-divergent) number that *improves monotonically* with
the training budget `small -> medium -> large` — the signature of a stable floor — while staying
perceptually mediocre (slightly soft samples, under-produced high frequencies). That image-quality
gap, which velocity MSE structurally cannot see, is what forces an image-space perceptual term at
step 2.

```python
# EDITABLE region of custom_train_perceptual.py (lines 384-401) — step 1: pure MSE on velocity
# Pure MSE on mean velocity prediction.
# No inverse-loss reweighting (which would amplify easy samples
# and destabilise training around step 35k).
loss_mse_unscaled = ((pred_mean_vel - mean_vel_target) ** 2).flatten(1).mean(1)
loss = loss_mse_unscaled.mean()
```
