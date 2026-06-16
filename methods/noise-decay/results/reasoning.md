Let me start from what actually breaks when I train one of these private models, not from the privacy formulas. The mechanism I run is the standard one: for each minibatch I compute per-sample gradients, clip every one of them into an `L2` ball of radius `C` so that no single example can move the aggregate by more than `C`, average the clipped gradients, and add zero-mean Gaussian noise to that average. If the noise multiplier is `sigma` and the batch size is `B`, the average picks up noise with standard deviation `sigma * C / B`. The optimizer then sees a noised average shaped exactly like an ordinary gradient, and an accountant somewhere has calibrated `sigma` so the whole run hits a target `(epsilon, delta)`. Fine. The thing that nags me is that I pick `sigma` once, before the first step, and then I apply that same `sigma` to every step until the last. I am pretending epoch one and epoch ninety-nine are the same kind of step. They are not.

Here is the asymmetry I keep seeing. Early in training the per-sample gradients are large — the model is far from any good region, every example is screaming a direction — and a noisy update of magnitude `sigma C / B` on top of a large signal still points somewhere useful; the noise is a perturbation, not a replacement. Late in training the model is near a low-loss region, the useful gradient has shrunk, and that same absolute `sigma C / B` noise no longer perturbs the signal, it *is* most of what the optimizer receives. I can make this precise without measuring anything: the quantity that decides whether a step is informative is something like a signal-to-noise ratio, the norm of the true averaged gradient over the standard deviation of the injected noise, roughly `||g_avg|| / (sigma C / B)`. The numerator falls as the model converges; diagnostic traces for fixed-noise private mechanisms show the average norm failing to enter the small-gradient regime I would want from ordinary training. Meanwhile the denominator, `sigma C / B`, is nailed to a constant. So the SNR of the late steps collapses toward zero, and I am spending real privacy budget on steps that are mostly returning noise. That is the waste I want to kill.

The naive fix is "add less noise late." But the budget is fixed, and I cannot just turn `sigma` down everywhere — that would blow the budget. If I lower `sigma` somewhere, I have to pay for it somewhere else. So before I touch the schedule I need to know, exactly, what the budget is *charging me for*. If the cost of a step depends on `sigma` in a particular way, then maybe I can move noise from the late steps, where it is wasted, to the early steps, where the signal is big enough to swallow it — at no change in total cost. Let me write the cost down and see.

A single training step is a subsampled Gaussian mechanism. I will account for it with truncated concentrated DP, because I already suspect I am going to want a *changing* noise level and the strong-composition machinery for `(epsilon, delta)`-DP gets awkward when every step has a different `sigma`; tCDP composes additively, which is exactly the property I will lean on. The Gaussian mechanism with sensitivity `C` and noise level `sigma_e` at epoch `e` is `(C^2 / (2 sigma_e^2), infinity)`-tCDP — that is the base parameter `rho_e = C^2 / (2 sigma_e^2)` with truncation `omega = infinity`. I am not releasing the gradient of the whole dataset though; I subsample a Poisson batch at rate `b/n`. The subsampling-amplification property says a `(rho, omega)`-tCDP mechanism run on a uniformly random `hn` of `n` entries is `(13 h^2 rho, log(1/h)/(4 rho))`-tCDP, so with `h = b/n` the per-epoch base picks up a factor `13 (b/n)^2`:

`rho_e = 13 (b/n)^2 * C^2 / (2 sigma_e^2)`,  `omega_e = log(n/b) sigma_e^2 / (2 C^2)`.

Now compose across the epochs. The composition property is `(rho_1, omega_1) o (rho_2, omega_2) = (rho_1 + rho_2, min(omega_1, omega_2))`: the `rho`'s simply add, and the `omega` takes the minimum. It does not care whether the steps are identical — that is the whole point of using it — so summing over `e = 0, ..., E-1`,

`rho_total = 13 (b/n)^2 (C^2 / 2) * sum_{e=0}^{E-1} 1/sigma_e^2`,

`omega_total = (log(n/b) / (2 C^2)) * min_e sigma_e^2`.

And the conversion property turns this into the `(epsilon, delta)` I actually have to report: `epsilon = rho_total + 2 sqrt(rho_total ln(1/delta))`, valid once `delta` clears the truncation condition `delta >= 1/exp((omega_total - 1)^2 rho_total)`.

Stare at `rho_total`. For a fixed sensitivity `C` and a fixed sampling rate `b/n`, the *only* place the schedule enters is the sum `sum_e 1/sigma_e^2`. The `omega` only sets the truncation floor; the budget I spend is governed by that inverse-square sum and nothing else about the temporal shape. So two completely different schedules `{sigma_e}` and `{sigma_e'}` that happen to have the same `sum_e 1/sigma_e^2` spend the *same* `rho_total`, convert to the *same* `epsilon`, cost the *same* privacy. That is the lever. I am free to reshape the noise level across the run however I like, as long as I hold `sum_e 1/sigma_e^2` fixed — and reshaping it costs nothing extra, because the schedule is a public function of the epoch index that never touches the data. The privacy is paid entirely in that one scalar sum.

Let me sanity-check the constant case so I am sure I have not dropped a factor. If `sigma_e = sigma` for all `e`, the sum is `E / sigma^2`, so

`rho_total = 13 (b/n)^2 C^2 E / (2 sigma^2)`,  `omega_total = log(n/b) sigma^2 / (2 C^2)`.

That is just `E` identical Gaussian steps composed, which is what plain DP-SGD is. Good, the bookkeeping is right.

So now I get to spend the same `sum_e 1/sigma_e^2` but distribute it in time. From the SNR argument I want the late epochs to have *small* `sigma_e` (low noise, because the signal is small and precious) and the early epochs to absorb the slack with *large* `sigma_e`. The first idea that is already on the table is the geometric one: lower the variance every epoch, `sigma_e^2 = sigma_0^2 R^e` with `R in (0, 1)`, so the noise variance shrinks by a constant fraction `R` each epoch. Let me run it through the accountant. The inverse-square sum is

`sum_{e=0}^{E-1} 1/sigma_e^2 = (1/sigma_0^2) sum_{e=0}^{E-1} R^{-e}`.

The ratio of the geometric series is `1/R > 1`, so using `sum_{k=0}^{n-1} r^k = (r^n - 1)/(r - 1)` with `r = 1/R`,

`sum_{e=0}^{E-1} R^{-e} = ((1/R)^E - 1) / ((1/R) - 1)`.

Those negative powers are ugly; clear them by multiplying numerator and denominator by `R^E`. The numerator becomes `R^E ((1/R)^E - 1) = 1 - R^E`, and the denominator becomes `R^E ((1/R) - 1) = R^{E-1} - R^E`. So

`sum_{e=0}^{E-1} 1/sigma_e^2 = (1/sigma_0^2) (1 - R^E) / (R^{E-1} - R^E)`,

and the geometric-decay budget is

`rho_total = 13 (b/n)^2 C^2 (1 - R^E) / (2 sigma_0^2 (R^{E-1} - R^E))`,

with the smallest variance at the last epoch, `omega_total = log(n/b) sigma_0^2 R^{E-1} / (2 C^2)`. Clean closed form. And it does push noise toward the front: the late variances `sigma_0^2 R^e` are tiny, the early ones are large. So this should work.

Except — let me actually picture the schedule against the constant baseline, because something is off. I have a fixed budget, which fixes `sum_e 1/sigma_e^2` at the same value as the constant run, `E / sigma_const^2`. The geometric schedule loads almost all of its inverse-square cost into the late epochs, where `sigma_e^2 = sigma_0^2 R^e` is smallest and `1/sigma_e^2` is largest. To keep the *total* inverse-square sum equal to `E / sigma_const^2` while those late terms are so large, the early terms must be small, which means `sigma_0` itself has to be *bigger* than `sigma_const`. The decay starts from above the constant level and slides down through it. So where does it cross? Early, when `R^e` is near 1, `sigma_e approx sigma_0 > sigma_const` — I am adding *more* noise than the constant baseline. Only after `R^e` has decayed enough does `sigma_e` drop below `sigma_const`. With `R` close to 1, which is what I need for a gentle, non-collapsing schedule, that crossover does not happen until roughly halfway through training. So for the first half of the run a geometric decay injects *more* noise than just leaving `sigma` constant, and only in the back half does it inject less.

That is the wall. The whole premise was that late noise is wasted, so move it to the early steps where the big signal can absorb it. But a geometric decay does not concentrate the relief at the end — it smears the schedule so thin that I overpay in noise for the entire first half just to underpay a little in the second. Net, the early over-noising eats most of the late savings, and the schedule ends up tying or barely beating constant `sigma`. This matches the uncomfortable report that linear decay gives "little or no advantage." A smooth every-epoch decay is the wrong *shape*: by being gradual it spends its excess where I did not want any excess.

Maybe a different smooth shape escapes this. Try inverse-time decay, `sigma_e^2 = sigma_0^2 / (1 + R e)` — the variance still decreases, but harmonically rather than geometrically. Now `1/sigma_e^2 = (1 + R e)/sigma_0^2`, so the sum is a plain arithmetic sum:

`sum_{e=0}^{E-1} 1/sigma_e^2 = (1/sigma_0^2) sum_{e=0}^{E-1} (1 + R e) = (1/sigma_0^2) [E + R * E(E-1)/2]`,

using `sum_{e=0}^{E-1} e = E(E-1)/2`. So

`rho_total = 13 (b/n)^2 C^2 [2E + R E(E-1)] / (4 sigma_0^2)`,

`omega_total = log(n/b) sigma_0^2 / (2 C^2 (1 + R(E-1)))`.

Another clean formula — but the qualitative defect is identical. It is still a monotone, gradual drift that starts above the constant level (same budget-balancing argument: front terms must be small in inverse-square cost, so `sigma_0 > sigma_const`) and only undercuts the constant near the end. Smoothness is the enemy here, not the particular functional form. Any schedule that bleeds its decay evenly across every epoch will over-noise the front.

So what do I actually want? I want to hold the noise *near* the constant level for as long as the signal is still large enough not to care, and then drop it sharply, late, exactly when the gradient has shrunk and every bit of relief converts directly into signal. That is not a smooth curve; it is a staircase. And I have seen this exact shape before, in non-private training: learning-rate schedules that hold the rate flat and then cut it by a factor at fixed milestones — step decay. Borrow it for the noise. Hold the noise variance fixed for `K` epochs at a time, then multiply it by `R` at each milestone:

`sigma_e^2 = sigma_0^2 R^{floor(e/K)}`,  `R in (0, 1)`.

Now check what this does to the budget. Assume `E` is divisible by `K` and let `P = E/K` be the number of plateaus. Plateau `p` (for `p = 0, ..., P-1`) covers `K` epochs, each with variance `sigma_0^2 R^p`. To see the schedule concretely, take `E = 100`, `K = 10`, so `P = 10`: epochs 0 through 9 sit at `sigma_0^2`, epochs 10 through 19 at `sigma_0^2 R`, on down to epochs 90 through 99 at `sigma_0^2 R^9`. The inverse-square sum collapses plateau by plateau:

`sum_{e=0}^{E-1} 1/sigma_e^2 = sum_{p=0}^{P-1} K / (sigma_0^2 R^p) = (K/sigma_0^2) sum_{p=0}^{P-1} R^{-p}`.

It is the same geometric series as the linear case, but with only `P` terms instead of `E`, and a `K` out front. Using `sum_{p=0}^{P-1} R^{-p} = ((1/R)^P - 1)/((1/R) - 1)` and again clearing negative powers by multiplying top and bottom by `R^P`,

`sum_{e=0}^{E-1} 1/sigma_e^2 = (K/sigma_0^2) (1 - R^P) / (R^{P-1} - R^P)`,

so the step-decay budget is

`rho_total = 13 (b/n)^2 C^2 K (1 - R^P) / (2 sigma_0^2 (R^{P-1} - R^P))`.

For the truncation parameter I have to be careful not to confuse the plateau index with the epoch index. There is a tidy bookkeeping trick: I can fold the `K` epochs of a plateau into one effective per-plateau variance by writing `sigma_p^2 = R^p sigma_0^2 / K`, so that `sum_{p=0}^{P-1} 1/sigma_p^2` reproduces `sum_e 1/sigma_e^2` exactly — the `1/K` in `sigma_p^2` becomes the `K` multiplier in the sum. With this collapsed `P`-term indexing, the minimum parameter is the last plateau, `R^{P-1} sigma_0^2 / K`, so

`omega_total = log(n/b) sigma_0^2 R^{P-1} / (2 C^2 K)`.

And the comparison with the smooth schedules is the whole payoff. With `P` plateaus instead of `E` epochs, the geometric staircase only takes `P` discrete steps down, so each step is the *same* factor `R` but they are spaced `K` epochs apart. To balance the same fixed budget I still need `sigma_0` a bit above the constant level — that has not changed — but now the schedule holds *flat* at that level for the first whole plateau and only begins cutting at epoch `K`, and because there are far fewer, sharper cuts concentrated toward the back, the time-integrated excess noise over the front is much smaller than a curve that was sliding down from the very first epoch. The staircase sits *near* the constant line, then drops in deliberate jumps right where the SNR argument says the relief is worth the most. If the noise schedule is the thing preventing the optimizer from entering the small-gradient regime, this is the shape that gives it room late without spending extra privacy. Larger `K` (longer plateaus, fewer cuts) and `R` closer to 1 (gentler cuts) both shrink the total inverse-square excess relative to `sigma_0`, so they trade a little late aggressiveness for less front-loaded over-noising — the knobs I tune for utility.

While I am redistributing noise by training phase, the clipping threshold deserves the same treatment, and there is a specific pathology I want to avoid. In the global-scaling view, the *lower* bound `c_0` is the sensitivity — all per-sample norms get scaled to be bounded by `c_0`, so the global sensitivity used in the privacy formulas is `c_0`, independent of where the *upper* threshold `z_e` sits. The adaptive global-scaling rule moves `z_e` *upward* with a geometric update so it stays above every per-sample norm. But the scaling applied to an in-bound gradient is `c_0 / z_e`, inversely proportional to the upper threshold. If `z_e` is driven up geometrically, then toward the end of training it explodes, and `c_0 / z_e` scales the gradients down toward zero — precisely when the gradients are already small. That actively suppresses the late signal I am trying to protect, and it costs extra privacy to keep adapting `z_e` upward. The fix is the same staircase, but the threshold must go the *other* way: since the gradients shrink, let the upper threshold shrink to track them,

`z_e = z_0 R^{floor(e/K)}`,

so the scaling `c_0 / z_e` stays moderate and the late updates are not crushed. The sensitivity is still `c_0`, so this stepwise threshold rides along without changing the privacy accounting at all. When a gradient does exceed the current `z_e`, I scale it by the smooth bounded factor `c_0 / (||g_i|| + w/(||g_i|| + w))` rather than hard-discarding it, which keeps the noised average close to the true batch average instead of throwing information away.

Now I have to make all of this fit a harness whose accountant only accepts *one* scalar `sigma`. The fixed `compute_epsilon(steps, sigma, q, delta)` assumes a single uniform multiplier across all `steps`, and charges a cost proportional to `steps / sigma^2`. But my real run spends `sum_t 1/sigma_t^2` summed over the actual per-step multipliers. So I need the single scalar that the fixed accountant would charge the *same* privacy for as my whole schedule. Setting the two costs equal,

`steps / sigma_eff^2 = sum_t 1/sigma_t^2`,

and solving,

`sigma_eff = sqrt(steps / sum_t 1/sigma_t^2)`.

That is a root-mean-of-inverse-squares — exactly the same `sum 1/sigma^2` invariant that governs `rho_total`, now appearing operationally as the number I hand the harness. It is the harmonic-mean-of-variances flavor: the equivalent uniform multiplier is dominated by the *smallest* `sigma_t`, the low-noise late plateaus, which is right, because those are the steps that cost the most privacy.

The same identity also tells me how to set `sigma_0`. I am given a *calibrated constant* multiplier `sigma_cal` — the value that, run uniformly, hits the target budget. I want my schedule to spend exactly that same budget, i.e. to have `sigma_eff` over the whole run equal to `sigma_cal`. Write each step's multiplier as `sigma_t = sigma_0 * f_t`, where `f_t` is the schedule's per-step factor (1 on the first plateau, then decaying). Equal budget means

`sum_t 1/(sigma_0^2 f_t^2) = total_steps / sigma_cal^2`,

so

`sigma_0 = sigma_cal * sqrt((sum_t 1/f_t^2) / total_steps)`.

Since the factors `f_t <= 1`, the sum `sum_t 1/f_t^2 >= total_steps`, so `sigma_0 >= sigma_cal` — the initial noise sits above the calibrated constant, exactly as the budget-balancing argument predicted, and the staircase then drops below it later.

There is one convention trap I have to be exact about or I will silently square the schedule. The closed-form tCDP table above decays the *variance* by `R`, `sigma_e^2 = sigma_0^2 R^{floor(e/K)}`, so the *multiplier* factor is `f_e = R^{floor(e/K)/2}` — a square root, because `R` acts on `sigma^2`. But a harness that decays the multiplier directly, `sigma_t = sigma_0 * d^{stage}` with some decay factor `d`, has `f_t = d^{stage}` already at the multiplier level. The inverse-square invariant `sum_t 1/f_t^2` is the same object either way, but if I plug a variance-decay `R` into a place expecting a multiplier-decay `d`, I am off by a square. So I keep it straight: in the harness code the decay factor is applied to `sigma` (the multiplier), and `inv_sq_sum` accumulates `1/f_t^2` with `f_t` the multiplier factor; the per-step variance is then `sigma_t^2 = sigma_0^2 f_t^2`, and everything composes through `sum_t 1/sigma_t^2` consistently.

Now I can write the mechanism for the harness. The constructor precomputes `sigma_0` from the calibrated constant multiplier and the planned per-stage multiplier factors via the equal-budget identity. At each step it reads the current stage from the zero-based public epoch index, decays the multiplier and the clip threshold to that stage, clips the per-sample gradients to the decayed threshold, averages them, and adds Gaussian noise with standard deviation `current_sigma * current_clip / batch_size`. For accounting it returns the equivalent uniform multiplier over the steps completed so far, `sqrt(steps / sum_t 1/sigma_t^2)`, so the fixed single-`sigma` accountant certifies the non-uniform run at the right budget.

```python
import torch


class DPMechanism:
    """Stagewise (step-decay) noise multiplier and clipping threshold for DP-SGD.

    Holds the noise level near the calibrated constant for K-epoch plateaus, then drops it by
    a fixed factor at each milestone, concentrating noise relief in late training where the
    gradient signal is small. The clip threshold decays by its own public factor to track
    shrinking gradients.
    Privacy of the non-uniform schedule is reported to the fixed single-sigma accountant via
    the equivalent uniform multiplier sqrt(steps / sum_t 1/sigma_t^2), the same sum 1/sigma^2
    invariant that governs the tCDP rho_total."""

    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier        # the calibrated CONSTANT multiplier sigma_cal
        self.n_params = n_params
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        # Step-decay schedule: four public plateaus; decay sigma and the clip threshold at each.
        self.decay_interval = max(1, epochs // 4)       # K: epochs per plateau
        self.noise_decay_factor = 0.8                   # multiplier factor f per stage (acts on sigma)
        self.clip_decay_factor = 0.85                   # clip threshold factor per stage
        self.steps_per_epoch = max(1, (dataset_size + batch_size - 1) // batch_size)

        # Calibrate sigma_0 so the whole schedule spends the same budget as the constant run:
        #   sigma_0 = sigma_cal * sqrt( sum_t (1/f_t^2) / total_steps ),  f_t the multiplier factor.
        total_steps = self.steps_per_epoch * epochs
        inv_sq_sum = 0.0
        for e in range(epochs):
            stage = self._stage_for_epoch(e)
            f = self.noise_decay_factor ** stage        # multiplier factor on this plateau
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

        # Current plateau: zero-based public epoch index, step-wise public schedule.
        stage = self._stage_for_epoch(epoch)
        self._current_sigma = self.sigma_0 * (self.noise_decay_factor ** stage)
        self._current_clip = self.clip_0 * (self.clip_decay_factor ** stage)

        # Per-sample L2 norms across all parameters jointly (one clip per sample).
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)                                          # [B]
        clip_factor = (self._current_clip / norms.clamp(min=1e-8)).clamp(max=1.0)

        noised_grads = []
        for g in per_sample_grads:
            shape = [batch_size] + [1] * (g.dim() - 1)
            clipped = g * clip_factor.reshape(shape)                         # clip per sample
            avg = clipped.mean(dim=0)                                        # batch average
            # Gaussian noise on the average: std = sigma * C / B  (sensitivity of the average is C/B).
            noise = torch.randn_like(avg) * (
                self._current_sigma * self._current_clip / batch_size
            )
            noised_grads.append(avg + noise)
        return noised_grads

    def get_effective_sigma(self, step, epoch):
        """Equivalent uniform multiplier over the completed steps: sigma_eff = sqrt(steps / sum 1/sigma_t^2).
        Makes the fixed single-sigma accountant charge exactly the schedule's privacy."""
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
        return (steps_counted / inv_sq_sum) ** 0.5
```

The chain holds end to end. Constant `sigma` is structurally wrong because the SNR of a step is `||g_avg|| / (sigma C / B)` and the numerator collapses as gradients shrink, so late steps spend privacy on near-pure noise. The tCDP accounting shows that for fixed `C` and sampling rate the budget depends on the schedule *only* through `sum_e 1/sigma_e^2`, and that a predefined schedule is otherwise free, so I can move noise from late to early at no cost as long as that sum is held. A geometric (and equally an inverse-time) decay redistributes correctly in principle but is the wrong *shape*: balancing a fixed budget forces `sigma_0` above the constant level, and a smooth curve then injects more noise than constant for roughly the first half of training, eating the late savings — which is why gradual decay gives little advantage. A step-decay staircase holds near the constant for `K`-epoch plateaus and cuts sharply at milestones, concentrating the relief late where the SNR argument says it converts directly to signal; its closed form is `rho_total = 13(b/n)^2 C^2 K (1 - R^P)/(2 sigma_0^2 (R^{P-1} - R^P))`, with `omega_total = log(n/b) sigma_0^2 R^{P-1}/(2 C^2 K)` and `epsilon = rho_total + 2 sqrt(rho_total ln(1/delta))`. The clip threshold decays step-wise too — downward, to track shrinking gradients and avoid the exploding-upper-threshold pathology that scales late gradients to zero — at fixed sensitivity `c_0`. And for the single-`sigma` harness the same `sum 1/sigma^2` invariant reappears as the equivalent uniform multiplier `sqrt(steps / sum_t 1/sigma_t^2)`, which both calibrates `sigma_0` from the constant `sigma_cal` and reports the schedule's true privacy to the fixed accountant.
