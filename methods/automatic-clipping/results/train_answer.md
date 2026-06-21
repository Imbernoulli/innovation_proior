Training a deep network with a formal differential-privacy guarantee means the optimizer never sees the raw minibatch gradient. Each per-sample gradient must first be bounded in L2 norm, then summed and masked with Gaussian noise calibrated to that bound. This adds two knobs that ordinary training lacks: the noise multiplier sigma and the clipping threshold R. Sigma is cheap: once the privacy budget, subsampling rate, and number of steps are fixed, a privacy accountant returns the exact sigma that spends the budget. R is expensive: there is no formula that maps the privacy budget to a good threshold, so practitioners must grid-search it, and accuracy is brutally sensitive to the choice. A ResNet18 on ImageNet can drop from 45% to 31% accuracy when R is merely doubled, and collapse to 0.1% when quadrupled. Private training therefore requires a two-dimensional search over (R, eta), while regular training needs only a one-dimensional search over the learning rate. On large models that extra dimension dominates tuning cost.

The natural response is to adapt R during training or set it from a quantile of the per-sample-norm distribution. Adaptive quantile clipping tracks a target quantile online, but it replaces the threshold knob with new knobs: the target quantile and the privacy-budget split between estimating the quantile and perturbing the gradient. Re-parameterized clipping makes accuracy less sensitive to R, yet R still appears in both branches of the min and still rescales weight decay. Global clipping drops large-gradient samples entirely and keeps the threshold. None of these removes R; they only move it around. The goal is to eliminate the threshold completely while preserving the same privacy guarantee and matching ordinary SGD's convergence rate.

The method is Automatic Clipping, specifically the stable variant AUTO-S. The starting observation is that the best private models consistently live in a small-R regime, and in that regime Abadi's clip min(R/||g_i||, 1) is active on most samples, so it degenerates to R/||g_i||. In other words, the threshold is not really acting as a threshold; it is acting as a per-sample normalizer. Automatic clipping makes that the rule: replace clipping with normalization, g_i -> g_i/(||g_i|| + gamma), where gamma is a small positive stability constant. The vanilla form AUTO-V sets gamma = 0 and normalizes every gradient to unit length; AUTO-S uses gamma > 0, with a canonical default of 0.01.

After normalization the sensitivity is bounded by 1, so R is gauge. Privacy depends only on the noise-to-sensitivity ratio sigma, and any constant R gives the same guarantee. For non-adaptive optimizers R couples into the learning rate, so tuning R and eta separately searches a one-dimensional manifold. For adaptive optimizers such as Adam or AdaGrad, R cancels between numerator and denominator entirely. Therefore R can simply be fixed at 1, and the only remaining optimizer knob is the learning rate, just as in non-private training.

Pure normalization would erase all magnitude information: a tiny gradient and a huge gradient both become unit vectors. That creates a lazy region where opposite-class gradients of similar count cancel even when the true gradient is nonzero, so the optimizer freezes at a non-stationary point. Adding gamma in the denominator fixes this. When ||g_i|| is large, g_i/(||g_i|| + gamma) behaves like g_i/||g_i||; when ||g_i|| is small, it behaves like g_i/gamma and keeps its direction while shrinking with magnitude. Magnitude order is preserved, and near convergence the aggregate smoothly becomes ordinary SGD. Under standard non-convex assumptions, lower-bounded smooth loss and centrally symmetric per-sample gradient noise with bounded variance, AUTO-S with step size eta ~ 1/sqrt(T) drives min_t E||g_t|| to zero at rate O(T^{-1/4}), matching standard non-private SGD. AUTO-V cannot match this because its scale-invariance makes the convergence prefactor vanish.

```python
import torch


class DPMechanism:
    """Automatic clipping (AUTO-S): per-sample normalization g_i / (||g_i|| + gamma).

    Replaces Abadi's per-sample clip min(R/||g_i||, 1). The clipping threshold R
    is gauge under normalization, so it is pinned to 1 and needs no tuning. The
    sensitivity is bounded by 1, so the privacy accountant is unchanged.
    """

    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        self.noise_multiplier = noise_multiplier  # sigma, fixed by the accountant
        self.n_params = n_params
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        # R is redundant under normalization; pin it to 1.
        self.max_grad_norm = 1.0
        # Stability constant for AUTO-S. gamma = 0 gives AUTO-V (pure
        # normalization), which can stall in a lazy region.
        self.numerical_stability_constant = 1e-2

    def clip_and_noise(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # Flat per-sample L2 norm across all parameters; sensitivity is of the
        # whole per-sample gradient vector.
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norm_sample = flat.norm(2, dim=1)  # [B] = ||g_i||

        # AUTO-S scale: C_i = R / (||g_i|| + gamma), with R = 1.
        # ||C_i g_i|| = ||g_i|| / (||g_i|| + gamma) < 1, so sensitivity <= 1.
        C = self.max_grad_norm / (norm_sample + self.numerical_stability_constant)

        noised_grads = []
        for g in per_sample_grads:
            shape = [batch_size] + [1] * (g.dim() - 1)
            normalized = g * C.reshape(shape)   # C_i * g_i
            summed = normalized.sum(dim=0)      # sum_i C_i g_i

            # Gaussian noise at sensitivity R = 1; std = sigma * R.
            # The noise-to-sensitivity ratio is sigma, identical to Abadi's clip,
            # so the privacy accountant and (eps, delta) guarantee are unchanged.
            noise = torch.randn_like(summed) * (self.noise_multiplier * self.max_grad_norm)
            noised_grads.append(summed + noise)

        return noised_grads

    def get_effective_sigma(self, step, epoch):
        # sigma is constant and was fixed up front by the accountant.
        return self.noise_multiplier
```
