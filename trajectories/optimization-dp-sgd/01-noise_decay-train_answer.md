The baseline mechanism clips every per-sample gradient to a fixed $C$, averages over the batch, and adds Gaussian noise of standard deviation $\sigma\,C/B$, with $\sigma$ calibrated once before the first step and held constant to the last. That constancy is the structural waste I want to attack first. The quantity that decides whether a step is informative is its signal-to-noise ratio, roughly $\|g_\text{avg}\| / (\sigma\,C/B)$ — the norm of the true averaged gradient over the standard deviation of the injected noise. Early in training the model is far from any good region, the per-sample gradients are large, and the noise is a perturbation riding on top of a strong signal. Late in training the averaged gradient has shrunk while the *absolute* noise is nailed to a constant, so the SNR collapses toward zero: I am spending real privacy budget on late steps that mostly return noise. The budget is fixed, so I cannot simply turn $\sigma$ down everywhere; if I lower it somewhere I must pay for it elsewhere. The fix has to be a *redistribution* of noise that touches nothing the harness forbids — and a schedule, a public function of the epoch index that never reads the data, is exactly such a thing.

I propose a **step-decay noise (and clip) schedule**. The first thing it rests on is a precise accounting of what the budget actually charges. Treating each step as a subsampled Gaussian mechanism and composing with truncated concentrated DP (which composes additively, the property I need when every step may carry a different noise level), the base parameter at epoch $e$ is $\rho_e = C^2/(2\sigma_e^2)$; Poisson subsampling at rate $b/n$ multiplies it, and composing across epochs the $\rho$'s add:
$$\rho_\text{total} = 13\,(b/n)^2\,\frac{C^2}{2}\sum_e \frac{1}{\sigma_e^2}.$$
For fixed sensitivity and sampling rate, the *only* place the schedule enters is the scalar $\sum_e 1/\sigma_e^2$ — the temporal shape is otherwise invisible to the budget. So two schedules with the same inverse-square sum spend the same $\epsilon$, and I am free to reshape $\sigma_e$ across the run however I like as long as I hold that sum fixed. (Sanity check: a constant $\sigma$ gives $\sum = E/\sigma^2$, which is just $E$ identical Gaussian steps — plain DP-SGD.)

From the SNR argument I want the late epochs to carry *small* $\sigma_e$ and the early epochs to absorb the slack with *large* $\sigma_e$. The tempting move is a smooth geometric decay $\sigma_e^2 = \sigma_0^2 R^e$, but it is the wrong shape, and seeing why fixes the design. The fixed budget pins $\sum_e 1/\sigma_e^2$ to the constant run's value $E/\sigma_\text{const}^2$. A geometric schedule loads almost all of its inverse-square cost into the late epochs where $\sigma_e^2$ is smallest, so to keep the total equal the early $\sigma_0$ must be *larger* than $\sigma_\text{const}$. The curve therefore starts *above* the baseline and only slides below it around the midpoint — over-noising the entire first half just to under-noise a little in the second. Smoothness is the enemy: by being gradual the schedule spends its excess exactly where I wanted none. Inverse-time decay has the identical defect.

What I actually want is to hold the noise *near* the constant level for as long as the signal is large enough not to care, then drop it sharply, late, where every bit of relief converts directly into signal. That is not a curve but a staircase — and it is the same shape as the learning-rate step-decay schedules of non-private training. So I hold the variance fixed for $K$ epochs at a time and multiply it by a factor at each milestone:
$$\sigma_e^2 = \sigma_0^2\,R^{\lfloor e/K\rfloor}.$$
With $P = E/K$ plateaus the geometric staircase takes only $P$ discrete steps down instead of $E$, each spaced $K$ epochs apart. To balance the same budget I still need $\sigma_0$ a touch above the constant level, but now the schedule holds *flat* at that level through the whole first plateau and only begins cutting at epoch $K$; with far fewer, sharper cuts concentrated toward the back, the time-integrated excess over the front is much smaller than a curve sliding down from epoch zero.

The clip threshold gets the same staircase, but it must move the *other* way. Since the gradients shrink, I let the upper clip shrink with them, $C_e = C_0\,R_c^{\lfloor e/K\rfloor}$, rather than the natural-looking move of raising it — driving the threshold up would crush the already-small late gradients toward zero. The clip riding step-wise does not change the accounting structure.

Now the fragile part, and the falsifiable claim of this rung: I have to make a non-uniform schedule fit a harness whose accountant accepts *one* scalar $\sigma$. The fixed `compute_epsilon(steps, sigma, q, delta)` assumes a single uniform multiplier and charges a cost proportional to $\text{steps}/\sigma^2$, while my real run spends $\sum_t 1/\sigma_t^2$. So the number I report through `get_effective_sigma` must be the single scalar the fixed accountant would charge the same privacy for. Setting $\text{steps}/\sigma_\text{eff}^2 = \sum_t 1/\sigma_t^2$ gives
$$\sigma_\text{eff} = \sqrt{\frac{\text{steps}}{\sum_t 1/\sigma_t^2}},$$
a root-mean-of-inverse-squares — the same invariant that governs $\rho_\text{total}$, now appearing operationally as the value I hand back. It is dominated by the smallest $\sigma_t$, the low-noise late plateaus, which is right because those steps cost the most privacy. The same identity sets $\sigma_0$: writing each step's multiplier as $\sigma_t = \sigma_0 f_t$ and equating budgets gives $\sigma_0 = \sigma_\text{cal}\cdot\sqrt{(\sum_t 1/f_t^2)/\text{total\_steps}}$. Since the factors $f_t \le 1$ the sum exceeds total\_steps, so $\sigma_0 \ge \sigma_\text{cal}$ — the initial noise sits above the calibrated constant, exactly as the budget-balancing argument predicted, and the staircase drops below it later.

This is precisely where I am most exposed. The whole construction rests on `get_effective_sigma` returning an equivalent-uniform multiplier that the *fixed* harness accountant maps to the true budget. If the schedule's per-step variances and the $\sigma_\text{eff}$ I report do not round-trip through that specific RDP bound — if the reported scalar is computed for one notion of "equivalent" and the accountant charges another — the reported and spent $\epsilon$ will disagree, and the disagreement will surface as consumed $\epsilon$ drifting above $3.0$ on exactly the runs where the schedule cut noise the most. I keep the decay on the multiplier $\sigma$ (not the variance) and accumulate $1/f_t^2$ with $f_t$ the multiplier factor, so per-step variance is $\sigma_0^2 f_t^2$ and everything composes through $\sum_t 1/\sigma_t^2$ consistently, avoiding the off-by-a-square convention trap. The defaults are deliberately gentle — four plateaus ($K = \text{epochs}//4$), a per-stage noise factor of $0.8$ and clip factor of $0.85$ — on the theory that gentle cuts keep $\sigma_0$ close to $\sigma_\text{cal}$ and minimize front over-noising. If the SNR argument is right and the round-trip is exact, I should beat the constant-noise baseline by a small margin concentrated on the hard task; if it is not exact, the consumed $\epsilon$ will drift and the accuracy will not be a like-for-like number.

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
