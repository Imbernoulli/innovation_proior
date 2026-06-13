# Warmup + cosine annealing schedule, distilled

The warmup-cosine learning-rate schedule fuses two ideas into one curve `eta(t)`: a short
**linear warmup** that ramps the rate up from a small value to `base_lr` over the first few
epochs, followed by **cosine annealing** that smoothly decays the rate from `base_lr` toward 0
over the rest of training. It needs no milestone hyperparameters, introduces no value discontinuities,
and keeps the optimizer, loss, and data pipeline untouched — it only changes the per-epoch
learning rate.

## Problem it solves

Choosing the learning rate as a function of training time for SGD-with-momentum training of a
deep network, when a single constant rate fails at both ends: too large at the start (the
freshly-initialized net sits in a high-curvature region, so a full step exceeds the local
stability bound `eta < 2/lambda_max(H)` and the loss spikes or diverges) and too large at the
end (near a minimum the stochastic-gradient noise floor does not vanish, so big steps rattle
instead of settling). The schedule must be small at both ends and high in between, smoothly,
without adding tunable milestones.

## Key idea

Split the run into two phases and make the curve continuous across the seam.

- **Linear warmup** (epochs `0 .. warmup-1`): ramp the rate from `base_lr/warmup` up to `base_lr`
  in equal increments, `eta = base_lr * (epoch+1) / warmup`. Starting small is aimed at keeping
  early steps under the curvature ceiling `2/lambda_max` while the sharp initial region relaxes into a
  flatter one where the full rate becomes admissible. The ramp is *gradual* on purpose: a
  *constant* low prefix that then snaps to the full rate just postpones the loss spike to the snap
  (the post-prefix region may still not admit the full rate), so a continuous climb is needed.
  The smooth ramp is also gentle on the momentum velocity (no sudden rate change to mismatch the
  carried-over history).
- **Cosine annealing** (epochs `warmup .. total-1`): decay from `base_lr` toward `0` along a half
  cosine, with the cosine clock measured from the end of warmup:

  ```
  progress = (epoch - warmup) / (total_epochs - warmup)        # 0 at seam; floor at next boundary
  eta = eta_min + (eta_max - eta_min) * 0.5 * (1 + cos(pi * progress))
      = base_lr * 0.5 * (1 + cos(pi * progress))               # eta_min=0, eta_max=base_lr
  ```

  The seam matches by construction: `progress=0 -> cos(0)=1 -> eta=base_lr`, continuous with the
  top of the ramp. With the scheduler clock `T_i = total_epochs - warmup`, the mathematical floor
  `progress=1 -> cos(pi)=-1 -> eta=0` occurs at the epoch boundary after the last in-loop epoch;
  the last sampled training epoch is therefore just above 0, as in the usual closed-form cosine
  scheduler convention. The shape is the reason for the choice: the derivative of the scaled
  learning rate is `-(eta_max - eta_min) * pi/2 * sin(pi*progress)`, which is 0 at both ends of
  the cosine interval and most negative at the midpoint. The rate *lingers near `base_lr` early*
  (fast progress while the gradient is large and far from a minimum), *falls fastest in the
  middle*, and *lingers near 0 late* (a long low-rate tail to settle below the noise floor). This
  is the smooth replacement for step decay's hand-picked cliffs.

## Defaults and why

`warmup = 5`, `eta_min = 0`, `eta_max = base_lr` (= 0.1 in the fixed pipeline).

- `warmup = 5` epochs: long enough to give the initial curvature a chance to relax, short enough
  to be a small fixed prefix (2.5% of a 200-epoch budget) rather than a tuned milestone schedule.
- Ramp from `base_lr/warmup` (not from 0): a true 0 start wastes the first epoch and is
  unproductive; `base_lr/warmup` gives a clean constant-increment sampled ramp `base_lr/5,
  2base_lr/5, ..., base_lr`, making the first rate five times smaller than the target while the
  local stability ceiling `2/lambda_max` is tightest.
- `eta_min = 0`: for a single (non-restarting) anneal you want the smallest steps possible at
  the very end; 0 is the natural floor.
- Cosine clock from end of warmup (not from epoch 0): makes the seam continuous at `base_lr`,
  puts the cosine floor at the boundary after training, and gives the full post-warmup interval
  to the decay.
- No `config` (arch/dataset) conditioning: both mechanisms — survive the sharp init, then anneal
  to settle — are architecture-agnostic, so the same curve serves ResNet-20/CIFAR-10,
  ResNet-56/CIFAR-100, and MobileNetV2/FashionMNIST.

## Final schedule

```python
import math


def get_lr(epoch, total_epochs, base_lr, config):
    """Linear warmup (5 epochs) then cosine annealing toward 0.

    Epochs 0..4:   linear ramp base_lr/5 -> base_lr (each step base_lr/5),
                   reducing early steps while the curvature ceiling is tight,
                   with no discontinuity.
    Epochs 5..end: cosine anneal base_lr toward 0, clock measured from end of warmup
                   so it opens at base_lr (continuous with the ramp); the floor is
                   at the next epoch boundary, with slope ~0 at both cosine ends.
    """
    warmup = 5
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / (total_epochs - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
```

## Equivalent canonical forms

The cosine piece is exactly the standard closed form, and the full schedule follows the same
warmup-prefix branch structure used by common schedulers.

Cosine annealing alone is the closed form of PyTorch's `CosineAnnealingLR`:

```python
# eta_min + (base_lr - eta_min) * (1 + cos(pi * t / T_max)) / 2
import math

def cosine_annealing_lr(t, T_max, base_lr, eta_min=0.0):
    return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * t / T_max)) / 2
```

The warmup-prefixed cosine structure matches timm's `CosineLRScheduler` with
`warmup_prefix=True`: a linear warmup segment, then a cosine segment whose progress is measured
after the warmup:

```python
import math

def warmup_cosine_lr(t, T_total, base_lr,
                     warmup_t=5, warmup_lr_init=0.0, eta_min=0.0):
    if t < warmup_t:
        # linear: warmup_lr_init + t * (base_lr - warmup_lr_init) / warmup_t
        step = (base_lr - warmup_lr_init) / warmup_t
        return warmup_lr_init + t * step
    t_cur = t - warmup_t                      # clock from end of warmup (warmup_prefix)
    t_i = T_total - warmup_t
    return eta_min + (base_lr - eta_min) * 0.5 * (1 + math.cos(math.pi * t_cur / t_i))
```

The task's `get_lr` uses the same `warmup_prefix` cosine branch. Its warmup samples the
zero-start line at `epoch + 1`, giving `base_lr/warmup, ..., base_lr` during epochs `0..4`
instead of timm's default `warmup_lr_init, ..., base_lr - step`; this avoids a zero-LR first
epoch while preserving the continuous seam at `epoch == warmup`.

## Relation to prior schedules

- **Step decay** divides the rate by a fixed factor at hand-picked milestones; warmup-cosine
  replaces the cliffs with one smooth monotone decay (no milestones, no discontinuity shocks, no
  momentum-velocity mismatch at drops) and additionally fixes the untreated *start* with warmup.
- **Constant-prefix warmup** holds a low rate then snaps to full; the snap re-creates the
  early-instability spike. Linear warmup removes the snap.
- **Cosine annealing with warm restarts** is the same cosine bump repeated, raising the rate at
  each restart (period `T_i`, optionally doubling). Warmup-cosine uses one such cosine interval
  from `base_lr` to `0` and never triggers a restart inside the training run.
- **Cyclical (triangular) learning rates** oscillate between bounds with sharp corners; the
  cosine gives a smooth curve and a clean single anneal to a small final rate.
