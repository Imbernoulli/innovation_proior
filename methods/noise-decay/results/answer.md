# Step-decay noise multiplier for DP-SGD, distilled

A DP-SGD variant that schedules the noise multiplier (and the upper clipping threshold)
downward in a *staircase* — holding the noise level near the calibrated constant for `K`-epoch
plateaus, then cutting it by a fixed factor at each milestone. It redistributes a fixed
`(epsilon, delta)` budget so the noise-heavy steps fall early (where the gradient signal is
large and absorbs noise) and the low-noise steps fall late (where the gradient has shrunk and
every bit of relief converts to signal), at no extra privacy cost beyond the Gaussian noise it
adds. This is DP-SGD-Global-Adapt-V2-S; its noise-schedule core is the "noise decay" mechanism.

## Problem it solves

DP-SGD calibrates one constant noise multiplier `sigma` for the entire run and adds noise of
standard deviation `sigma C / B` to every batch-average gradient. Because the gradient signal
shrinks as training converges while that absolute noise stays fixed, late steps have a
collapsing signal-to-noise ratio and spend privacy on near-pure noise. The budget is fixed, so
noise cannot simply be lowered everywhere; it has to be *reallocated* in time, under an
accountant that tolerates a per-step-varying `sigma`.

## Key idea

For fixed sensitivity `C` and sampling rate `q = b/n`, the total privacy of a sequence of
subsampled Gaussian steps depends on the noise schedule **only through the inverse-square sum**
`sum_e 1/sigma_e^2` (tCDP: Gaussian step cost `~ 1/sigma_e^2`, subsampling `13 q^2`, additive
composition). A predefined schedule (a function of the epoch index, never of the data) is
otherwise free. So one can reshape `{sigma_e}` freely while holding `sum_e 1/sigma_e^2`.

A *smooth* decay (geometric or inverse-time) is the wrong shape: balancing a fixed budget forces
the initial `sigma_0` above the constant level, and a gradual curve then injects *more* noise
than constant DP-SGD for roughly the first half of training, eating the late savings — hence its
near-zero net advantage. A *step* decay holds flat near the constant for each plateau and drops
sharply at milestones, concentrating the noise relief late where it is worth most. The clipping
threshold is decayed step-wise too — *downward*, to track shrinking gradients — avoiding the
opposite (geometric-*increase*) global-scaling rule, whose exploding upper threshold scales late
gradients toward zero and stalls convergence.

## Decay schedules and their privacy (tCDP)

Per epoch `e`, a subsampled Gaussian step has `rho_e = 13(b/n)^2 C^2 / (2 sigma_e^2)`,
`omega_e = log(n/b) sigma_e^2 / (2 C^2)`; composition adds the `rho_e` and takes `min` of the
`omega_e`, giving `rho_total = 13(b/n)^2 (C^2/2) sum_{e=0}^{E-1} 1/sigma_e^2`,
`omega_total = (log(n/b)/(2 C^2)) min_e sigma_e^2`, and
`epsilon = rho_total + 2 sqrt(rho_total ln(1/delta))` once `delta >= 1/exp((omega_total - 1)^2 rho_total)`.

| Schedule | `sigma_e^2` | `rho_total` | `omega_total` |
|---|---|---|---|
| No decay | `sigma^2` | `13(b/n)^2 C^2 E / (2 sigma^2)` | `log(n/b) sigma^2 / (2 C^2)` |
| Linear (geometric) | `sigma_0^2 R^e` | `13(b/n)^2 C^2 (1 - R^E) / (2 sigma_0^2 (R^{E-1} - R^E))` | `log(n/b) sigma_0^2 R^{E-1} / (2 C^2)` |
| Time | `sigma_0^2 / (1 + R e)` | `13(b/n)^2 C^2 [2E + R E(E-1)] / (4 sigma_0^2)` | `log(n/b) sigma_0^2 / (2 C^2 (1 + R(E-1)))` |
| **Step** | `sigma_0^2 R^{floor(e/K)}` | `13(b/n)^2 C^2 K (1 - R^P) / (2 sigma_0^2 (R^{P-1} - R^P))` | `log(n/b) sigma_0^2 R^{P-1} / (2 C^2 K)` |

with `R in (0, 1)`, `K` = epoch drop rate, `P = E/K` = number of plateaus (assume `K | E`).

**Step closed form.** With `P` plateaus of `K` epochs each at variance `sigma_0^2 R^p`,
`sum_e 1/sigma_e^2 = (K/sigma_0^2) sum_{p=0}^{P-1} R^{-p} = (K/sigma_0^2)(1 - R^P)/(R^{P-1} - R^P)`
(geometric series with ratio `1/R`, then multiply numerator and denominator by `R^P`). Folding
`K` into a per-plateau `sigma_p^2 = R^p sigma_0^2 / K` reproduces the same sum and makes the
collapsed `P`-term `omega` minimum the last plateau, giving the step-row `omega_total`.

## Canonical algorithm (DP-SGD-Global-Adapt-V2-S)

```
Input: dataset D, sampling rate q = b/n, lower clip c_0, upper clip z_0, epochs E,
       decay rate R, epoch drop rate K, sigma decay sigma_e^2 = sigma_0^2 R^{floor(e/K)},
       clip decay z_e = z_0 R^{floor(e/K)}, learning rate eta_t, constant w, T = E/q iters.
Init  theta_0, sigma_0, z_0.
for t = 0, ..., T-1:
    B  <- Poisson sample of D at rate q;   e <- floor(q*t)
    for (x_i, y_i) in B:
        g_i   <- grad_theta loss(S_theta(x_i), y_i)
        gamma_i <- c_0 / z_e                                  if ||g_i|| <= z_e   # global scaling, sensitivity c_0
                   c_0 / (||g_i|| + w/(||g_i|| + w))          if ||g_i|| >  z_e   # DP-PSAC bounded scaling
        g_bar_i <- gamma_i * g_i
    sigma_e^2 <- sigma_0^2 R^{floor(e/K)}                                         # update once per epoch
    g_tilde_B <- (1/|B|) ( sum_i g_bar_i + N(0, sigma_e^2 I) )
    theta_{t+1} <- theta_t - eta_t * g_tilde_B
    z_e <- z_0 R^{floor(e/K)}                                                     # step-decay upper threshold
return theta_T
```

Sensitivity is the *lower* bound `c_0` (all scaled norms are bounded by `c_0`), so the privacy
formulas use `C = c_0`, independent of `z_e`. Privacy via the step-decay tCDP row above.

## Harness-faithful code (single-sigma accountant)

When the harness exposes only per-sample `L2` clipping to one threshold and a fixed accountant
`compute_epsilon(steps, sigma, q, delta)` that assumes a single uniform `sigma` (per-step cost
`~ 1/sigma^2`), the schedule is reported through the **equivalent uniform multiplier**
`sigma_eff = sqrt(steps / sum_t 1/sigma_t^2)` (root-mean-of-inverse-squares — the same
`sum 1/sigma^2` invariant). `sigma_0` is calibrated from the constant `sigma_cal` by
`sigma_0 = sigma_cal * sqrt((sum_t 1/f_t^2) / total_steps)` with `f_t` the per-step *multiplier*
factor (note: a variance-decay `R` corresponds to a multiplier factor `sqrt(R)`; the code below
decays the multiplier `sigma` directly, so `f_t` is the factor itself).

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

## Why step over smooth, and hyperparameters

- **Step vs linear/time:** a smooth decay starts above the constant `sigma` (fixed-budget
  balancing) and over-noises the front half; step decay holds flat near constant then cuts
  sharply, so its time-integrated noise is lower and the relief lands late.
- **Larger `K`, larger `R` (closer to 1):** fewer/gentler cuts shrink the inverse-square excess
  relative to `sigma_0`, giving less front-loaded over-noising and higher utility.
- **Step-decay clip (downward):** tracks shrinking gradients at fixed sensitivity `c_0`, avoiding
  the exploding upper-threshold pathology of geometric-increase global scaling.
