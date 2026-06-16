**Problem.** Standard DP-SGD carries a clip threshold `C` that does two opposite jobs — it sets the noise
scale (`sigma·C/B`) and the clipping bias — has no formula from `(epsilon, delta)`, and must be grid-searched,
with accuracy savagely sensitive to it. The standard floor fixed `C = 1.0` with no argument it is right. Remove
`C` entirely.

**Key idea (AUTO-S automatic clipping).** In the small-`C` regime where good private models live, Abadi's clip
`min(C/||g_i||, 1)` degenerates into per-sample normalization `C·g_i/||g_i||` — and normalization is exactly the
per-sample factor that maximizes alignment with the true gradient under the sensitivity budget (the first-order
descent term). So replace clipping with normalization. `C` is then gauge: privacy depends only on the
noise-to-sensitivity ratio `sigma`, so any constant `C` is equivalent — fix it at `1`. Pure normalization
(AUTO-V) erases all magnitude and can stall in a lazy region (opposite-class gradients cancel); a denominator
constant `gamma` — `g_i/(||g_i|| + gamma)` — restores magnitude order, turns into SGD near convergence, and
keeps sensitivity `< 1`, so the accountant is unchanged.

**This task vs. the generic method.** The harness pins `gamma = 1.0` (not the canonical ~0.01) explicitly to
keep the *frozen* `SGD(lr=0.1)` schedule stable — so this is a soft, magnitude-aware down-scaling, not aggressive
normalization. Noise is calibrated to the bounded sensitivity `1` (`sigma·1/B`), not to `max_grad_norm`.
`get_effective_sigma` returns the unchanged constant `sigma`, so unlike a noise schedule it cannot drift off
budget.

**Why it should win (modestly).** It removes the `C` knob at the same certified `3.0` budget; with `gamma = 1.0`
it behaves near convergence like the SGD the loop already runs, so a near-tie-or-slight-gain on the average is the
honest expectation — leaving the real CIFAR-10 headroom to *adapting* `C` to the drifting gradient distribution.

**Hyperparameters.** `gamma = 1.0`; scale `= 1/(||g_i|| + gamma)`; noise std on the average `= sigma·1/B`
(sensitivity 1); `sigma` = the calibrated constant. No clip threshold to tune.

```python
class DPMechanism:
    """AUTO-S Automatic Clipping (Bu et al., NeurIPS 2023).

    Per-sample gradient normalization: g_i / (||g_i|| + gamma).
    Sensitivity bounded by 1, no clipping threshold to tune.
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
        # AUTO-S gamma for this benchmark harness. gamma=1.0 keeps the
        # existing learning-rate schedule stable.
        self.gamma = 1.0

    def clip_and_noise(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # Compute per-sample gradient norms
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)  # [B]

        # AUTO-S normalization: scale each gradient by 1/(||g_i|| + gamma)
        # This bounds sensitivity to 1 (since ||g_i / (||g_i|| + gamma)|| <= 1)
        scale = 1.0 / (norms + self.gamma)  # [B]

        noised_grads = []
        for g in per_sample_grads:
            shape = [batch_size] + [1] * (g.dim() - 1)
            normalized = g * scale.reshape(shape)

            # Average over batch
            avg = normalized.mean(dim=0)

            # Add noise calibrated to sensitivity=1 (AUTO-S bound)
            # sigma * C / B where C=1 for AUTO-S
            noise = torch.randn_like(avg) * (
                self.noise_multiplier * 1.0 / batch_size
            )
            noised_grads.append(avg + noise)

        return noised_grads

    def get_effective_sigma(self, step, epoch):
        return self.noise_multiplier
```
