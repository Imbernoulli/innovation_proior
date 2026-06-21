We want to train a neural network under a hard, externally checked $(\epsilon, \delta)$-differential-privacy guarantee while giving up as little test accuracy as possible. The privacy lives inside the gradient step: each per-sample gradient is clipped to an $L_2$ ball of radius $C$ so that one example can move the aggregate by at most $C$, the clipped gradients are averaged, and zero-mean Gaussian noise calibrated to that sensitivity is added before the optimizer sees anything. An accountant calibrates the multiplier to the target budget and tracks the budget each step consumes. The trouble is that the standard mechanism, DP-SGD, picks one multiplier $\sigma$ before the first step and then applies that same $\sigma$ identically from the first epoch to the last — it pretends epoch one and epoch ninety-nine are the same kind of step, and they are not. Early in training the per-sample gradients are large and a noisy update of magnitude $\sigma C / B$ on top of a big signal still points somewhere useful: the noise is a perturbation. Late in training the model is near a low-loss region, the useful gradient has shrunk, and that same absolute $\sigma C / B$ noise no longer perturbs the signal, it *is* most of what the optimizer receives. The quantity that decides whether a step is informative is a signal-to-noise ratio, roughly $\|g_{\text{avg}}\| / (\sigma C / B)$; the numerator collapses as the model converges while the denominator is nailed to a constant, so the SNR of late steps falls toward zero and we spend real budget on steps that mostly return noise. That is the waste to kill.

The naive fix, "add less noise late," cannot stand alone, because the budget is fixed: lowering $\sigma$ somewhere must be paid for somewhere else. The known attempts do not close the gap. The direct ancestor, linearly decaying noise (Zhang et al. 2021), lowers the variance geometrically every epoch, $\sigma_e^2 = \sigma_0^2 R^e$ with $R \in (0,1)$, and accounts for it with tCDP — but a smooth every-epoch decay is reported to give little or no advantage over constant noise. Adaptive-clipping and global-scaling methods (Andrew et al. 2021; Bu et al. 2021; Esipova et al. 2022) reshape the *clipping* rule rather than the noise; worse, DP-Global-Adapt drives its upper threshold $z_e$ *upward* geometrically, and since the in-bound scaling factor is $c_0 / z_e$, an exploding $z_e$ scales gradients toward zero exactly when they are already small, stalling convergence and costing extra budget to adapt the threshold privately. So we need a noise-and-clip policy that respects the same total $(\epsilon, \delta)$, stays compatible with the existing accountant, and actually concentrates relief where the SNR argument says it is worth the most.

I propose DP-SGD-Global-Adapt-V2-S, whose noise-schedule core is the mechanism I call *noise decay*: a *staircase* (step-decay) schedule on the noise multiplier and the clipping threshold. The license for reshaping the schedule comes from the accounting. Account for each training step as a subsampled Gaussian mechanism with truncated concentrated DP, because tCDP composes *additively* and so handles a per-step-varying $\sigma$ cleanly where strong $(\epsilon,\delta)$-composition gets awkward. A Gaussian step of sensitivity $C$ and noise level $\sigma_e$ is $(C^2/(2\sigma_e^2), \infty)$-tCDP; Poisson subsampling at rate $h = b/n$ multiplies the $\rho$ by $13h^2$, giving $\rho_e = 13(b/n)^2 C^2 / (2\sigma_e^2)$ and $\omega_e = \log(n/b)\,\sigma_e^2 / (2C^2)$. Composition adds the $\rho$'s and takes the minimum of the $\omega$'s, so over $E$ epochs
$$\rho_{\text{total}} = 13\,(b/n)^2\,\tfrac{C^2}{2}\sum_{e=0}^{E-1}\frac{1}{\sigma_e^2}, \qquad \omega_{\text{total}} = \frac{\log(n/b)}{2C^2}\,\min_e \sigma_e^2,$$
and conversion gives $\epsilon = \rho_{\text{total}} + 2\sqrt{\rho_{\text{total}}\ln(1/\delta)}$ once $\delta \ge 1/\exp((\omega_{\text{total}}-1)^2 \rho_{\text{total}})$. The load-bearing observation is staring out of $\rho_{\text{total}}$: for fixed $C$ and sampling rate, the *only* place the schedule enters the budget is the inverse-square sum $\sum_e 1/\sigma_e^2$. Two schedules with the same value of that sum spend the same privacy, full stop; the $\omega$ only sets a truncation floor. A predefined schedule is otherwise free, because it is a public function of the epoch index that never touches the data. So we are free to reshape $\{\sigma_e\}$ however we like as long as we hold $\sum_e 1/\sigma_e^2$ fixed — moving noise from late steps, where it is wasted, to early steps, where the big signal swallows it, at zero extra cost.

The question is then which *shape* to use, and here the obvious smooth options fail for a subtle reason. With the budget fixed, $\sum_e 1/\sigma_e^2$ equals the constant run's $E/\sigma_{\text{const}}^2$. A geometric decay loads almost all of its inverse-square cost into the late epochs where $\sigma_e^2 = \sigma_0^2 R^e$ is smallest, so to keep the total sum equal the early terms must be small, which forces $\sigma_0 > \sigma_{\text{const}}$. The decay therefore *starts above* the constant level and only slides below it after $R^e$ has decayed enough — with $R$ near 1, the crossover does not happen until roughly halfway through. For the first half of training a geometric (or, identically, an inverse-time $\sigma_e^2 = \sigma_0^2/(1+Re)$) schedule injects *more* noise than constant DP-SGD, and the early over-noising eats most of the late savings. Smoothness is the enemy: a curve that bleeds its decay evenly across every epoch overpays exactly where we wanted no excess. What we actually want is to hold the noise *near* the constant level for as long as the signal is large enough not to care, then drop it sharply, late, when every bit of relief converts directly into signal. That is not a curve, it is a staircase — borrowed from step-decay learning-rate schedules. Hold the variance fixed for $K$ epochs at a time and multiply it by $R$ at each milestone, $\sigma_e^2 = \sigma_0^2 R^{\lfloor e/K \rfloor}$. With $P = E/K$ plateaus the sum collapses plateau by plateau,
$$\sum_{e=0}^{E-1}\frac{1}{\sigma_e^2} = \frac{K}{\sigma_0^2}\sum_{p=0}^{P-1} R^{-p} = \frac{K}{\sigma_0^2}\cdot\frac{1-R^P}{R^{P-1}-R^P},$$
(geometric series of ratio $1/R$, then clearing the negative powers by multiplying numerator and denominator by $R^P$), so $\rho_{\text{total}} = 13(b/n)^2 C^2 K (1-R^P) / (2\sigma_0^2 (R^{P-1}-R^P))$, and folding $K$ into a per-plateau $\sigma_p^2 = R^p \sigma_0^2 / K$ reproduces the same sum and makes the collapsed $P$-term minimum the last plateau, giving $\omega_{\text{total}} = \log(n/b)\,\sigma_0^2 R^{P-1} / (2C^2 K)$. Because the staircase takes only $P$ discrete cuts spaced $K$ epochs apart instead of decaying every epoch, it holds *flat* near the constant for the first whole plateau and concentrates its sharp drops toward the back — its time-integrated excess noise over the front is far smaller than a smooth curve sliding down from epoch zero. Larger $K$ and $R$ closer to 1 (fewer, gentler cuts) shrink that excess further, trading a little late aggressiveness for less front-loaded over-noising; these are the utility knobs.

The clipping threshold gets the same staircase treatment, but it must move the *other* way. In the global-scaling view the *lower* bound $c_0$ is the sensitivity — all per-sample norms are scaled to be bounded by $c_0$, so the privacy formulas use $C = c_0$ independent of where the *upper* threshold $z_e$ sits. The in-bound scaling is $c_0/z_e$, so driving $z_e$ up geometrically (as DP-Global-Adapt does) crushes the late gradients toward zero exactly when they are smallest. The fix is to let the upper threshold *shrink* to track the gradients, $z_e = z_0 R^{\lfloor e/K \rfloor}$, keeping $c_0/z_e$ moderate; the sensitivity is still $c_0$, so this rides along without changing the accounting at all. Out-of-bound gradients are scaled by the smooth bounded factor $c_0/(\|g_i\| + w/(\|g_i\| + w))$ rather than hard-discarded, keeping the noised average close to the true batch average.

Finally the schedule must fit a harness whose accountant accepts only *one* scalar $\sigma$ and charges a cost proportional to $\text{steps}/\sigma^2$. My run spends $\sum_t 1/\sigma_t^2$, so I report the single multiplier that the fixed accountant would charge the same privacy for. Setting $\text{steps}/\sigma_{\text{eff}}^2 = \sum_t 1/\sigma_t^2$ and solving,
$$\sigma_{\text{eff}} = \sqrt{\frac{\text{steps}}{\sum_t 1/\sigma_t^2}},$$
a root-mean-of-inverse-squares — the same $\sum 1/\sigma^2$ invariant that governs $\rho_{\text{total}}$, now appearing operationally — dominated by the smallest, most expensive late-plateau $\sigma_t$, which is correct. The same identity calibrates $\sigma_0$ from the given calibrated constant $\sigma_{\text{cal}}$: writing $\sigma_t = \sigma_0 f_t$ with $f_t$ the per-step multiplier factor and demanding equal budget, $\sigma_0 = \sigma_{\text{cal}}\sqrt{(\sum_t 1/f_t^2)/\text{total\_steps}}$, and since every $f_t \le 1$ this gives $\sigma_0 \ge \sigma_{\text{cal}}$ — the initial noise sits above the constant, exactly as the budget-balancing argument predicted, then the staircase drops below it later. One convention trap matters: the closed-form table decays the *variance* by $R$, so the *multiplier* factor is $\sqrt{R}$; the harness code below decays the multiplier $\sigma$ directly, so its `noise_decay_factor` is the multiplier factor $f_t$ itself and `inv_sq_sum` accumulates $1/f_t^2$ at the multiplier level, keeping everything consistent through $\sum_t 1/\sigma_t^2$.

```python
import torch


class DPMechanism:
    """Step-decay noise multiplier + step-decay clipping threshold for DP-SGD.
    Holds noise near the calibrated constant for K-epoch plateaus, drops it by a fixed factor
    at each milestone; clip threshold decays by its own public factor to track shrinking
    gradients.
    Reports the schedule's privacy to a fixed single-sigma accountant via the equivalent
    uniform multiplier sqrt(steps / sum_t 1/sigma_t^2)."""

    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier        # calibrated CONSTANT multiplier sigma_cal
        self.n_params = n_params
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        self.decay_interval = max(1, epochs // 4)       # K: epochs per plateau (4 plateaus)
        self.noise_decay_factor = 0.8                   # multiplier factor per stage (acts on sigma)
        self.clip_decay_factor = 0.85                   # clip threshold factor per stage
        self.steps_per_epoch = max(1, (dataset_size + batch_size - 1) // batch_size)

        # sigma_0 = sigma_cal * sqrt( sum_t (1/f_t^2) / total_steps )  -> equal budget to constant run.
        total_steps = self.steps_per_epoch * epochs
        inv_sq_sum = 0.0
        for e in range(epochs):
            stage = self._stage_for_epoch(e)
            f = self.noise_decay_factor ** stage
            inv_sq_sum += self.steps_per_epoch / (f * f)
        self.sigma_0 = noise_multiplier * (inv_sq_sum / total_steps) ** 0.5
        self.clip_0 = max_grad_norm
        self._current_sigma = self.sigma_0
        self._current_clip = self.clip_0

    def _stage_for_epoch(self, epoch):
        epoch_idx = min(max(int(epoch), 0), self.epochs - 1)
        return epoch_idx // self.decay_interval

    def clip_and_noise(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]
        stage = self._stage_for_epoch(epoch)
        self._current_sigma = self.sigma_0 * (self.noise_decay_factor ** stage)
        self._current_clip = self.clip_0 * (self.clip_decay_factor ** stage)

        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)                                          # per-sample L2 norm [B]
        clip_factor = (self._current_clip / norms.clamp(min=1e-8)).clamp(max=1.0)

        noised_grads = []
        for g in per_sample_grads:
            shape = [batch_size] + [1] * (g.dim() - 1)
            clipped = g * clip_factor.reshape(shape)
            avg = clipped.mean(dim=0)                                        # batch average
            noise = torch.randn_like(avg) * (                               # std = sigma * C / B
                self._current_sigma * self._current_clip / batch_size
            )
            noised_grads.append(avg + noise)
        return noised_grads

    def get_effective_sigma(self, step, epoch):
        if step <= 0:
            return self.sigma_0
        inv_sq_sum = 0.0
        steps_counted = 0
        for e in range(self.epochs):
            stage = self._stage_for_epoch(e)
            sigma_e = self.sigma_0 * (self.noise_decay_factor ** stage)
            take = min(self.steps_per_epoch, step - steps_counted)
            if take <= 0:
                break
            inv_sq_sum += take / (sigma_e * sigma_e)                         # accumulate 1/sigma_t^2
            steps_counted += take
        if inv_sq_sum == 0:
            return self.sigma_0
        return (steps_counted / inv_sq_sum) ** 0.5                           # sqrt(steps / sum 1/sigma^2)
```
