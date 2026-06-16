# Automatic Clipping (AUTO-V / AUTO-S)

Automatic clipping replaces the per-sample gradient *clip* `min(R/||g_i||, 1)` of DP-SGD with
per-sample *normalization* `g_i/(||g_i|| + gamma)`, eliminating the clipping-threshold
hyperparameter `R`. With `R` gone, a private model has no DP-specific hyperparameter left to tune
(the noise multiplier `sigma` is fixed analytically by the privacy accountant), so private training
is as cheap to tune as regular training — a 1D search over the learning rate instead of a 2D search
over `(R, eta)`.

## Problem it solves

Abadi's DP-SGD needs two extra knobs versus regular training: the noise multiplier `sigma` and the
clipping threshold `R`. `sigma` is derivable from `(epsilon, delta, p, T)` by an accountant; `R` is
not, must be grid-searched, and accuracy is extremely sensitive to it (e.g. ResNet18/ImageNet drops
`45% -> 31%` at `2R`, `-> 0.1%` at `4R`). The 2D `(R, eta)` search dominates the tuning cost at scale.

## Key idea

In the regime where the best private models live, `R` is small and the clip is active on most
samples, so `min(R/||g_i||, 1) = R/||g_i||` — Abadi's clip is already per-sample normalization times
`R`. Make that the rule:

- **AUTO-V (vanilla):** `Clip(g_i; R) = R/||g_i||`, i.e. `g_i -> R g_i/||g_i||`.
- **AUTO-S (stable):** `Clip(g_i; R) = R/(||g_i|| + gamma)` with stability constant `gamma > 0`.

**`R` is gauge — it can be removed.** Privacy depends only on the noise-to-sensitivity ratio, which
is `sigma` for any `R` (post-normalization sensitivity is `R`, noise is `sigma R`). And `R` either
*couples* with the learning rate (non-adaptive optimizers: `R`-dependent AUTO with `(eta, lambda)` ==
`R`-independent AUTO with `(eta R, lambda/R)`) or *cancels* entirely (adaptive optimizers: the `R` in
numerator and denominator of `m/sqrt(v)` cancels). Either way `R` never needs tuning. Fix `R = 1`:
```
Clip_AUTO-S(g_i) = 1/(||g_i|| + gamma),   default gamma = 0.01.
```

**Why the stability constant.** AUTO-V (`gamma = 0`) maps every gradient to length `R`, erasing all
magnitude information (scale-invariance). This causes a "lazy region": opposite-class per-sample
gradients of different magnitudes cancel, so the update is zero even when the true gradient is nonzero
— the optimizer freezes at a non-stationary point. With `gamma > 0`, as `||g_i|| -> 0` the clipped
gradient tends to `g_i/gamma` (keeps direction, shrinks with magnitude); magnitude order is preserved
(`||g_i|| > ||g_j|| <=> ||C_i g_i|| > ||C_j g_j||`); and near convergence `sum_i g_i/(||g_i||+gamma)
-> (1/gamma) sum_i g_i`, i.e. it becomes ordinary SGD. `gamma > 0` is what lets the true gradient
norm converge to zero. The fixed default is `gamma = 0.01`.

## Algorithm (R-independent automatic DP optimizer)

```
Fix sigma such that  Accountant(epsilon, delta, p = B/n, T) <= epsilon.   # no R needed
for t = 1 .. T:
    sample batch B_t (each point i.i.d. with prob p)
    per-sample gradients g_i = grad of l_i at w_t
    ghat_i = g_i / (||g_i||_2 + gamma)            # AUTO-S clip, gamma = 0.01 (gamma = 0 is AUTO-V)
    ghat   = sum_i ghat_i + sigma * N(0, I)        # sensitivity 1, noise std = sigma
    w_t    = Optimizer.step(w_t, ghat, eta_t)      # any optimizer (SGD/Adam/AdamW/LAMB/...)
```

## Privacy

`||g_i/(||g_i||+gamma)|| < 1`, so the global `L2` sensitivity is bounded by `1`, exactly as Abadi's
clip bounds it by `R`. The noise-to-sensitivity ratio is `sigma` in both cases, so DP-SGD with
AUTO-V/AUTO-S satisfies the same `(epsilon_Accountant(delta, sigma, B/n, T), delta)`-DP as Abadi's
clipping under any valid accountant. No privacy budget is spent choosing `R` (there is none).

## Convergence (non-convex)

Under (A1) `L(w) >= L_*`, (A2) `L`-smoothness, (A3) per-sample noise mean-zero with `E||.||^2 <= xi^2`
and *centrally symmetric* about `g_t`, DP-SGD with AUTO-S and `eta ~ 1/sqrt(T)` gives
```
min_t E||g_t|| <= xi/r + (M_cvx)^{-1}( (4/sqrt(T)) sqrt((L_0 - L_*) L (1 + sigma^2 d/B^2)) ),
```
where `M(x; r, xi, gamma) = ( gamma/((r-1)(x+xi/r)+gamma) - gamma/((r+1)(x+xi/r)+gamma) ) x` for
`r > 1`. Minimizing the resulting hyperbola over the threshold ratio `r` yields
```
min_t E||g_t|| = O(T^{-1/4}),
```
the same asymptotic rate as standard non-private SGD (which gives
`min_t E||g_t|| <= T^{-1/4} sqrt(2(L_0 - L_*)L + xi^2/B)`).

The proof reduces the per-step decrease via smoothness to the alignment term
`g_t^T E(tilde g_t/(||tilde g_t||+gamma))` plus a noise penalty `sigma^2 d/B^2`; the symmetric-noise
half-space pairing lower-bounds the paired bracket by `min_{0<c<=1} f(c, r; gamma/||g_t||)
(||g_t|| - xi/r)`, so the alignment term carries the outer factor `1/2`, with
```
f(c, S; Gamma) = (1+Sc)/(sqrt(S^2+2Sc+1)+Gamma) + (1-Sc)/(sqrt(S^2-2Sc+1)+Gamma).
```
For AUTO-V (`gamma = 0`), `r < 1` leaves the residual floor `xi/r > xi`, while `r >= 1` makes
`min_c f(c, r; 0) = 0`, so AUTO-V cannot reach zero gradient norm — the lazy region, seen in the bound. For AUTO-S (`gamma > 0`)
and `r > 1`, `min_c f(c, r; Gamma) = f(1, r; Gamma) = Gamma/(r-1+Gamma) - Gamma/(r+1+Gamma) > 0`;
substituting `Gamma = gamma/||g||` gives the positive `M` above and restores convergence.

## Privacy-utility factors (via `mu`-GDP, `mu = (B/n) sqrt(T(e^{1/sigma^2}-1))`)

Minimizing the bound's argument: **train longer with larger noise** (decreasing in `T`), **use larger
batches** (decreasing in `B`), **pretrain** (smaller `L_0`, `L`, `xi`). The optimal learning rate is
`sqrt((L_0 - L_*) mu^2 n^2 / (L(mu^2 n^2 + dT)))` — larger for smaller models, weaker privacy, or
shorter runs. The learning rate is the one remaining knob.

## Working code

Filling the `clip_and_noise` slot of the DP mechanism (matches the Opacus one-line change
`per_sample_clip_factor = max_grad_norm/(per_sample_norms + 0.01)` and the FastDP ghost-clipping
`C = max_grad_norm/(norm_sample + numerical_stability_constant)` with `numerical_stability_constant =
1e-2`, `max_grad_norm = 1`):

```python
import torch


class DPMechanism:
    """Automatic clipping (AUTO-S): per-sample normalization g_i/(||g_i|| + gamma),
    in place of Abadi's clip min(R/||g_i||, 1). No threshold R to tune (R is gauge,
    fixed at 1). Sensitivity bounded by 1, so the privacy accountant is unchanged.
    gamma = 0 recovers AUTO-V (pure normalization, can stall in a lazy region);
    gamma > 0 (AUTO-S) preserves magnitude order and converges at O(T^{-1/4})."""

    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        self.noise_multiplier = noise_multiplier  # sigma, fixed by the accountant
        self.n_params = n_params
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = 1.0                  # R is redundant under normalization -> pin to 1
        self.numerical_stability_constant = 1e-2  # gamma for AUTO-S; 0 gives AUTO-V

    def clip_and_noise(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # per-sample L2 norm over all parameters jointly (sensitivity is of the
        # whole per-sample gradient vector)
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norm_sample = flat.norm(2, dim=1)                    # [B] = ||g_i||

        # AUTO-S clip factor C_i = R / (||g_i|| + gamma), R = 1
        # ||C_i g_i|| = ||g_i|| / (||g_i|| + gamma) < 1  ->  sensitivity <= 1
        C = self.max_grad_norm / (norm_sample + self.numerical_stability_constant)

        noised_grads = []
        for g in per_sample_grads:
            shape = [batch_size] + [1] * (g.dim() - 1)
            normalized = g * C.reshape(shape)                # C_i * g_i
            summed = normalized.sum(dim=0)                    # sum_i C_i g_i

            # Gaussian noise at sensitivity R = 1: std = sigma * R; ratio = sigma,
            # identical to Abadi's clip, so the accountant / (eps, delta) is unchanged
            noise = torch.randn_like(summed) * (self.noise_multiplier * self.max_grad_norm)
            noised_grads.append(summed + noise)              # any optimizer steps on this

        return noised_grads

    def get_effective_sigma(self, step, epoch):
        return self.noise_multiplier
```
