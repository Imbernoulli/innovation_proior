**Problem.** Privatize training of a non-convex deep net at a fixed, *certifiable* `(epsilon, delta)`.
Output perturbation is dead (the SGD endpoint has no usable sensitivity); and any non-uniform noise schedule
must be laundered through the harness's uniform-`sigma` accountant, where the claimed and spent budgets can
come apart. The reference rung is the one mechanism the harness accountant was written for — so its budget is
exact.

**Key idea (canonical DP-SGD).** Privatize the *process* at the gradient. Per step: clip each per-sample
gradient flat in `L2` to `C = max_grad_norm` (this caps each example's contribution, giving the summed
gradient sensitivity exactly `C` — the clip must be per-sample *before* averaging, unlike non-private
clip-after-average), average over the batch, and add `N(0, (sigma·C/B)^2)`. The optimizer steps on the noised
average. Privacy is composed by the moments/RDP accountant, which is exactly the harness's fixed
`compute_epsilon`; `calibrate_noise_to_epsilon` binary-searches the constant `sigma` that spends
`target_epsilon`.

**Why it is the trustworthy floor.** Because the mechanism reports the calibrated constant `sigma` unchanged,
the reported and spent budgets are the same object — no translation, no drift. It sits at `epsilon ≈ 3.0` by
construction, the honest budget every later rung must beat *at*, not above.

**Hyperparameters.** `C = max_grad_norm = 1.0`; `sigma` = the calibrated constant from the accountant
(`noise_multiplier`); noise std on the average = `sigma·C/B`. No schedule, no adaptation.

```python
class DPMechanism:
    """Standard DP-SGD (Abadi et al., 2016).

    Fixed per-sample gradient clipping + constant Gaussian noise.
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
