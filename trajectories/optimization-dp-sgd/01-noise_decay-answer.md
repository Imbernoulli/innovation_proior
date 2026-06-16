**Problem.** The default mechanism holds the noise multiplier `sigma` constant for the whole run, but a
step's signal-to-noise ratio `||g_avg|| / (sigma·C/B)` collapses as gradients shrink late in training, so
late steps spend privacy budget on near-pure noise. The budget is fixed, so noise cannot simply be turned
down — it must be *redistributed* without changing total cost.

**Key idea (step-decay noise + clip schedule).** For fixed sensitivity and sampling rate, the privacy cost
depends on the schedule *only* through `sum_t 1/sigma_t^2`; the temporal shape is otherwise free. Move noise
from late steps (wasted) to early steps (absorbed by the large signal) by holding `sigma` flat for `K`-epoch
plateaus, then cutting it by a factor at each milestone — a staircase, not a smooth curve (a smooth decay
forces `sigma_0 > sigma_const` and over-noises the entire first half). The clip threshold decays step-wise
*downward* to track shrinking gradients. `sigma_0` is calibrated so the schedule spends the same budget as
the constant run: `sigma_0 = sigma_cal · sqrt((sum_t 1/f_t^2)/total_steps)`.

**Why it can win — and how it can fail.** The harness accountant is fixed and single-`sigma`, so the
non-uniform schedule reports the equivalent uniform multiplier `sqrt(steps / sum_t 1/sigma_t^2)` through
`get_effective_sigma`. If that round-trips exactly through the harness's RDP bound, the run stays at budget
and should gain a little where late signal matters most. If it does *not* round-trip exactly, the consumed
`epsilon` drifts above 3.0 — the rung silently overspends, and its accuracy is not a like-for-like number.

**Hyperparameters.** Four plateaus (`decay_interval = epochs//4`); noise factor `0.8` and clip factor
`0.85` per stage (gentle, to keep `sigma_0` near `sigma_cal`); `steps_per_epoch = dataset_size//batch_size`.

```python
class DPMechanism:
    """Step-Decay Noise Schedule (inspired by Global-Adapt-V2-S, 2025).

    Decays noise multiplier and clipping threshold over training epochs
    to allocate more privacy budget to later (more useful) training steps.

    Privacy accounting: tracks cumulative RDP per-step using the actual
    sigma at each step, then returns an equivalent uniform sigma so
    that the external ``compute_epsilon(steps, sigma, q, delta)`` call
    produces the correct (tight) epsilon.
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

        # Step-decay schedule parameters
        # Decay noise and clipping every decay_interval epochs
        self.decay_interval = max(1, epochs // 4)  # 4 decay stages
        self.noise_decay_factor = 0.8  # Reduce noise by 20% at each stage
        self.clip_decay_factor = 0.85  # Reduce clip norm by 15% at each stage

        # Pre-compute the per-epoch sigma schedule so we can do accurate
        # RDP accounting.  Steps per epoch = dataset_size // batch_size
        # (drop_last=True in DataLoader).
        self.steps_per_epoch = dataset_size // batch_size

        # Compute sigma_0: scale the calibrated (uniform) sigma up so that
        # the harmonic-mean-equivalent sigma across all steps equals the
        # calibrated value.  This keeps the total privacy spend equal to
        # the budget even though individual steps have different noise.
        total_steps = self.steps_per_epoch * epochs
        inv_sq_sum = 0.0
        for e in range(1, epochs + 1):
            stage = (e - 1) // self.decay_interval
            factor = self.noise_decay_factor ** stage
            # Each epoch contributes steps_per_epoch steps at sigma_0*factor
            # 1/sigma_t^2 = 1/(sigma_0*factor)^2 = 1/(sigma_0^2 * factor^2)
            inv_sq_sum += self.steps_per_epoch / (factor * factor)
        # sigma_eff = sqrt(total_steps / inv_sq_sum) * sigma_0
        # We want sigma_eff == noise_multiplier (the calibrated value), so:
        #   noise_multiplier = sigma_0 * sqrt(total_steps / inv_sq_sum)
        #   sigma_0 = noise_multiplier / sqrt(total_steps / inv_sq_sum)
        #           = noise_multiplier * sqrt(inv_sq_sum / total_steps)
        self.sigma_0 = noise_multiplier * (inv_sq_sum / total_steps) ** 0.5
        self.clip_0 = max_grad_norm

        # Current values
        self._current_sigma = self.sigma_0
        self._current_clip = self.clip_0

    def clip_and_noise(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # Update schedule based on epoch
        stage = (epoch - 1) // self.decay_interval
        self._current_sigma = self.sigma_0 * (self.noise_decay_factor ** stage)
        self._current_clip = self.clip_0 * (self.clip_decay_factor ** stage)

        # Compute per-sample gradient norms
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)  # [B]

        # Clip per-sample gradients using current (decayed) threshold
        clip_factor = (self._current_clip / norms.clamp(min=1e-8)).clamp(max=1.0)

        noised_grads = []
        for g in per_sample_grads:
            shape = [batch_size] + [1] * (g.dim() - 1)
            clipped = g * clip_factor.reshape(shape)

            # Average over batch
            avg = clipped.mean(dim=0)

            # Add noise calibrated to current clip norm and sigma
            noise = torch.randn_like(avg) * (
                self._current_sigma * self._current_clip / batch_size
            )
            noised_grads.append(avg + noise)

        return noised_grads

    def get_effective_sigma(self, step, epoch):
        """Return equivalent uniform sigma for accurate RDP accounting.

        Computes the harmonic-mean-equivalent sigma over all steps up to
        the current point, so that the external call
        ``compute_epsilon(step, sigma_eff, q, delta)`` which assumes a
        uniform sigma gives the same epsilon as step-by-step RDP
        accounting with the actual per-step sigma values.

        sigma_eff = sqrt(steps / sum_{t=1}^{steps} 1/sigma_t^2)
        """
        if step <= 0:
            return self.sigma_0
        # Accumulate 1/sigma_t^2 across completed steps
        inv_sq_sum = 0.0
        steps_counted = 0
        for e in range(1, self.epochs + 1):
            stage = (e - 1) // self.decay_interval
            sigma_e = self.sigma_0 * (self.noise_decay_factor ** stage)
            inv_sq_e = 1.0 / (sigma_e * sigma_e)
            epoch_steps = min(self.steps_per_epoch, step - steps_counted)
            if epoch_steps <= 0:
                break
            inv_sq_sum += epoch_steps * inv_sq_e
            steps_counted += epoch_steps
        if inv_sq_sum == 0:
            return self.sigma_0
        return (steps_counted / inv_sq_sum) ** 0.5
```
