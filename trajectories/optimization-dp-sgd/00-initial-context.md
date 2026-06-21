## Research question

Train a classifier under a fixed `(epsilon, delta)`-differential-privacy budget and get the highest
test accuracy out of that budget. The released object is a model whose parameters an adversary may read,
and the guarantee must be worst-case: removing any single training example must barely change the
output distribution. The only thing being designed is the **DP mechanism** — how per-sample gradients
are clipped, how Gaussian noise is calibrated, and whether either of those adapts over training.
Everything else (model, data pipeline, optimizer, privacy accountant) is frozen.

## Prior art / Background / Baselines

The starting point is the line of work that established gradient-level privatization.

- **Output / objective perturbation for private ERM (Chaudhuri–Monteleoni–Sarwate 2011; Bassily–Smith–Thakurta 2014).**
  Add noise to the final weights or to the objective, calibrated to the sensitivity of the minimizer.
- **Noisy SGD in the loop (Song–Chaudhuri–Sarwate 2013; Bassily et al. 2014).**
  Add noise inside the optimizer, at the gradient, each step.
- **Advanced (strong) composition (Dwork–Rothblum–Vadhan 2010).**
  Composing `k` mechanisms each `(epsilon, delta)`-private costs roughly `epsilon·sqrt(2k ln(1/delta'))` instead of `k·epsilon`.

## Fixed substrate / Code framework

A DP-SGD harness is frozen and must not be touched. Two small ConvNets (a 4-layer MNISTNet for MNIST/Fashion-MNIST, a GroupNorm CIFAR10Net), standard normalization, `SGD(lr=0.1, momentum=0.9)` with cosine annealing, cross-entropy, batch size 256, 20 epochs, `drop_last=True`. Per-sample gradients are computed for the loop and handed to the mechanism as a list of tensors `[B, *param_shape]`, one per parameter.

The accountant is fixed and single-`sigma`: before training, `calibrate_noise_to_epsilon` binary-searches the constant noise multiplier `sigma` that spends exactly `target_epsilon` over `total_steps = steps_per_epoch · epochs` at sampling rate `q = batch_size/dataset_size`, using an RDP bound `compute_epsilon(steps, sigma, q, delta)` that assumes **one uniform `sigma` across all steps**. The mechanism's `get_effective_sigma(step, epoch)` is the only thing the accountant reads back — so any non-uniform schedule must report the *equivalent uniform multiplier* or the budget it claims will be wrong.

## Editable interface

Exactly one region is editable — the `DPMechanism` class in `opacus/custom_dpsgd.py` (lines 152–233). The contract:

- `__init__(self, max_grad_norm, noise_multiplier, n_params, dataset_size, batch_size, epochs, target_epsilon, target_delta)` — `noise_multiplier` is the calibrated constant `sigma`; `max_grad_norm` is the default clip `C` (=1.0).
- `clip_and_noise(self, per_sample_grads, step, epoch)` — takes the per-sample gradients `[B, *shape]`, returns the aggregated **noised** gradients `[*shape]`, one per parameter, that the optimizer steps on.
- `get_effective_sigma(self, step, epoch)` — the noise multiplier the external accountant uses; for the default it is just the constant `sigma`.

Available inside the region: `torch`, `math`, `numpy`, `scipy.optimize`. The starting point is the scaffold default — **standard flat clipping with constant noise**.

```python
# EDITABLE region of opacus/custom_dpsgd.py (lines 152-233) — default fill (standard DP-SGD)
class DPMechanism:
    """Differentially private gradient mechanism.

    Standard DP-SGD: clip per-sample gradients to max_grad_norm,
    then add Gaussian noise calibrated to (noise_multiplier * max_grad_norm).
    """

    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.n_params = n_params
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

    def clip_and_noise(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # Compute per-sample gradient norms (flat norm across all parameters)
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)  # [B]

        # Clip per-sample gradients
        clip_factor = (self.max_grad_norm / norms.clamp(min=1e-8)).clamp(max=1.0)  # [B]

        noised_grads = []
        for g in per_sample_grads:
            shape = [batch_size] + [1] * (g.dim() - 1)
            clipped = g * clip_factor.reshape(shape)

            # Average over batch
            avg = clipped.mean(dim=0)

            # Add calibrated Gaussian noise
            noise = torch.randn_like(avg) * (
                self.noise_multiplier * self.max_grad_norm / batch_size
            )
            noised_grads.append(avg + noise)

        return noised_grads

    def get_effective_sigma(self, step, epoch):
        return self.noise_multiplier
```

## Evaluation settings

Trained and evaluated on three datasets, each over three seeds {42, 123, 456}, at a fixed budget
`epsilon = 3.0`, `delta = 1e-5`:

- **MNIST** — 28×28 grayscale digits, 10 classes (MNISTNet).
- **Fashion-MNIST** — 28×28 grayscale clothing, 10 classes (MNISTNet).
- **CIFAR-10** — 32×32 color images, 10 classes (CIFAR10Net, GroupNorm).

Metric: **test accuracy** (higher is better) under the same privacy budget. The consumed
`epsilon` is also recorded per dataset — a mechanism whose accounting drifts above 3.0 has spent more
budget than the others and its accuracy is not a like-for-like comparison.
