## Research question

Unconditional CIFAR-10 diffusion, everything frozen except one decision: **what should the network be
trained to predict?** The forward process, the UNet backbone, the optimizer, the noise schedule, the
DDIM sampler, and the FID metric are all fixed. The single object being designed is the **prediction
parameterization** — the regression target the model fits at training time, paired with the rule that
recovers the clean image from the model's output at sampling time. The three standard targets (the
noise $\epsilon$, the clean image $x_0$, the velocity $v$) are mathematically interchangeable — any one
converts to the others — but they give different loss landscapes, different signal scaling across
timesteps, and different gradient magnitudes, so under a *finite* training budget they land at
different FID. The whole task is to choose, derive, and correctly invert that target.

## Prior art before the first rung (the DDPM target lineage)

The first rung reacts to the choices baked into the DDPM training recipe (Ho et al. 2020,
arXiv:2006.11239) and the forward process it inherited from Sohl-Dickstein et al. (2015).

- **The fixed forward process (Ho et al. 2020).** A datapoint is corrupted by a variance-preserving
  Gaussian chain whose any-$t$ marginal is closed-form:
  $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$, with $\epsilon\sim\mathcal N(0,I)$
  and $\bar\alpha_t$ the cumulative product of $(1-\beta_t)$. The training loop here samples a random
  $t$, draws this $x_t$, and fits the network to a *target* by plain MSE. The forward process is given;
  the target is the open slot.
- **The reverse mean and its three readings.** The optimal reverse step regresses onto the
  forward-posterior mean $\tilde\mu_t(x_t,x_0)$, and that mean can be written as a linear function of
  $x_0$, of $x_t$ and $\epsilon$, or of the velocity $v$. So there is a *family* of equivalent targets,
  related by the same two schedule scalars $\sqrt{\bar\alpha_t}$ and $\sqrt{1-\bar\alpha_t}$. Predicting
  any one, then converting, yields the same ideal model — but not the same finite-budget optimization.
  Gap: equivalence at the optimum says nothing about which target a network learns *fastest* and
  *most evenly across noise levels* under a fixed step count.
- **Direct $x_0$ regression (the naive target).** The most literal thing to ask of a denoiser is the
  clean image itself: train the network to output $x_0$, and at sampling time use its output directly.
  No schedule algebra at recovery time. Gap: the target's difficulty and scale swing wildly with $t$ —
  at high noise the network must hallucinate a full clean image from near-pure noise, at low noise it
  must copy its input — so a single network with a single MSE spends its budget unevenly. This is the
  rung the ladder starts from.

The fixed pipeline below is what these choices plug into; the editable interface is the one decision
left open.

## The fixed substrate

A self-contained CIFAR-10 training script (`diffusers-main/custom_train.py`) is frozen and must not be
touched outside the editable region. It fixes: the dataset (CIFAR-10, $32\times32$, unconditional, random
horizontal flip, pixels mapped to $[-1,1]$); the backbone (`diffusers` `UNet2DModel`,
`google/ddpm-cifar10-32` architecture, evaluated at three channel scales — small $\sim$9M, medium
$\sim$36M, large $\sim$140M params); training (35,000 steps per scale, AdamW lr $2\times10^{-4}$, weight
decay $10^{-4}$, gradient clip 1.0, mixed precision, EMA rate 0.9995, multi-GPU DDP); the noise schedule
(linear $\beta$ from $10^{-4}$ to $0.02$ over 1000 steps, via `DDPMScheduler`); and the sampler (50-step
DDIM, Song et al. 2020, arXiv:2010.02502). The loop draws $x_t$ with `add_noise`, calls
`compute_training_target(...)` for the regression target, and fits the UNet by `F.mse_loss(pred,
target)`. At eval it uses EMA weights and computes FID with clean-fid against the CIFAR-10 train set.

One detail of the sampler is load-bearing for the contract. The DDIM scheduler is constructed with
`prediction_type="sample"` — it expects a **predicted $x_0$**, not $\epsilon$ or $v$. So the loop calls
the editable `predict_x0(...)` to convert the network's raw output into $\hat x_0$, and hands *that* to
`scheduler.step`. The schedule scalars are precomputed in a `schedule` dict: `alphas_cumprod`,
`sqrt_alpha` $=\sqrt{\bar\alpha_t}$, and `sqrt_one_minus_alpha` $=\sqrt{1-\bar\alpha_t}$.

## The editable interface

Exactly one region is editable — two coupled functions in `custom_train.py` (lines 83–118):

- `compute_training_target(x_0, noise, timesteps, schedule)` — the MSE regression target, shape
  $[B,C,H,W]$.
- `predict_x0(model_output, x_t, timesteps, schedule)` — recovers $\hat x_0$ from the model output, used
  by the DDIM sampler.

The hard contract is **consistency**: `predict_x0` must be the exact algebraic inverse of whatever
`compute_training_target` defines, under the same schedule scalars; if the sampler's recovery does not
invert the training target, the model is sampled off-parameterization and FID collapses. Every rung on
the ladder is a fill of exactly these two functions and nothing else — no change to architecture,
dataset, optimizer, noise schedule, sampling procedure, or metric.

The starting point is the scaffold default: **both functions unimplemented**. Each rung replaces exactly
these two definitions.

```python
# EDITABLE region of diffusers-main/custom_train.py (lines 83-118) — default (unimplemented)
def compute_training_target(x_0, noise, timesteps, schedule):
    """Compute the training target given clean images and noise.

    The model will be trained to predict this target via MSE loss.
    Must be consistent with predict_x0() below.

    Args:
        x_0:       [B, C, H, W] clean images
        noise:     [B, C, H, W] sampled Gaussian noise
        timesteps: [B] integer timesteps (0 to T-1)
        schedule:  dict with keys 'alphas_cumprod', 'sqrt_alpha',
                   'sqrt_one_minus_alpha', each [T] tensors
    Returns: [B, C, H, W] target tensor
    """
    raise NotImplementedError("Implement compute_training_target")


def predict_x0(model_output, x_t, timesteps, schedule):
    """Recover predicted x_0 from the model's output.

    Must be consistent with compute_training_target() above.
    Used during DDIM sampling to convert model prediction back to x_0.

    Args:
        model_output: [B, C, H, W] model prediction
        x_t:          [B, C, H, W] noisy sample
        timesteps:    [B] integer timesteps
        schedule:     dict (same as compute_training_target)
    Returns: [B, C, H, W] predicted clean image
    """
    raise NotImplementedError("Implement predict_x0")
```

## Evaluation settings

The candidate parameterization is trained from scratch at each of the three channel scales (small,
medium, large) and scored by FID. Single seed (42). Metric: FID computed by clean-fid against the
CIFAR-10 train set, lower is better, reported at three scales — `best_fid_small`, `best_fid_medium`,
`best_fid_large` (each the best checkpoint over training, EMA weights, 50-step DDIM). The contribution
must be a transferable target parameterization, not a change to anything else in the pipeline.
