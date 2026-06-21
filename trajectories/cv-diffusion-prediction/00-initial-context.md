## Research question

Unconditional CIFAR-10 diffusion, everything frozen except one decision: **what should the network be
trained to predict?** The forward process, the UNet backbone, the optimizer, the noise schedule, the
DDIM sampler, and the FID metric are all fixed. The single object being designed is the **prediction
parameterization** — the regression target the model fits at training time, paired with the rule that
recovers the clean image from the model's output at sampling time. Standard choices such as the noise
$\epsilon$ or the clean image $x_0$ are mathematically interchangeable, but they give different loss
landscapes, signal scaling across timesteps, and gradient magnitudes, so under a finite training budget
they land at different FID. The task is to choose, derive, and correctly invert that target.

## Prior art / Background / Baselines

The forward process is fixed: a variance-preserving Gaussian chain with closed-form marginals
$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$. Training samples a random $t$, draws
this $x_t$, and fits the network by MSE to a target; the target is the open slot. The optimal reverse
mean can be written as a linear function of $x_0$ or of $x_t$ and $\epsilon$, so there is a family of
equivalent targets under the same schedule scalars. Equivalence at the optimum says nothing about which
target a finite-budget network learns fastest and most evenly across noise levels.

- **DDPM $\epsilon$-prediction (Ho et al. 2020).** The network predicts the Gaussian noise $\epsilon$
  added to $x_0$; at sampling time $\hat x_0$ is recovered from the predicted noise using the schedule
  scalars. Gap: at low noise $x_t$ is nearly $x_0$, so inferring $\epsilon$ requires taking a small
  difference and is numerically fragile; at high noise $x_t$ is nearly pure noise, so the task is
  easier, and optimization effort concentrates unevenly across the noise range.
- **Direct $x_0$ prediction.** The network predicts the clean image $x_0$ directly, and the output is
  used as the recovered $\hat x_0$. Gap: at low noise the target is almost visible in the input, while
  at high noise the network must reconstruct a full clean image from near-pure noise, so the target
  scale and difficulty vary heavily across timesteps.

## Fixed substrate / Code framework

A self-contained CIFAR-10 training script (`diffusers-main/custom_train.py`) is frozen and must not be
touched outside the editable region. It fixes: the dataset (CIFAR-10, $32\times32$, unconditional, random
horizontal flip, pixels mapped to $[-1,1]$); the backbone (`diffusers` `UNet2DModel`,
`google/ddpm-cifar10-32` architecture, evaluated at three channel scales — small $\sim$9M, medium
$\sim$36M, large $\sim$140M params); training (35,000 steps per scale, AdamW lr $2\times10^{-4}$, weight
decay $10^{-4}$, gradient clip 1.0, mixed precision, EMA rate 0.9995, multi-GPU DDP); the noise schedule
(linear $\beta$ from $10^{-4}$ to $0.02$ over 1000 steps, via `DDPMScheduler`); and the sampler (50-step
DDIM, Song et al. 2020). The loop draws $x_t$ with `add_noise`, calls
`compute_training_target(...)` for the regression target, and fits the UNet by `F.mse_loss(pred,
target)`. At eval it uses EMA weights and computes FID with clean-fid against the CIFAR-10 train set.

The DDIM scheduler is constructed with `prediction_type="sample"` — it expects a **predicted $x_0$**,
not $\epsilon$. So the loop calls the editable `predict_x0(...)` to convert the network's raw output into
$\hat x_0$, and hands that to `scheduler.step`. The schedule scalars are precomputed in a `schedule`
dict: `alphas_cumprod`, `sqrt_alpha` $=\sqrt{\bar\alpha_t}$, and `sqrt_one_minus_alpha`
$=\sqrt{1-\bar\alpha_t}$.

## Editable interface

Exactly one region is editable — two coupled functions in `custom_train.py` (lines 83–118):

- `compute_training_target(x_0, noise, timesteps, schedule)` — the MSE regression target, shape
  $[B,C,H,W]$.
- `predict_x0(model_output, x_t, timesteps, schedule)` — recovers $\hat x_0$ from the model output, used
  by the DDIM sampler.

The hard contract is **consistency**: `predict_x0` must be the exact algebraic inverse of whatever
`compute_training_target` defines, under the same schedule scalars; if the sampler's recovery does not
invert the training target, the model is sampled off-parameterization and FID collapses. The
contribution must be a fill of exactly these two functions and nothing else — no change to architecture,
dataset, optimizer, noise schedule, sampling procedure, or metric.

The scaffold default is **both functions unimplemented**. Each candidate replaces exactly these two
definitions.

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
