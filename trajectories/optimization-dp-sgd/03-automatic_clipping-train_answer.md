The standard rung came back exactly as I needed: consumed $\epsilon$ is $3.0016$ on every grayscale seed and $2.9986$ on every CIFAR-10 seed — no drift — reaching MNIST 96.03, FMNIST 79.64, CIFAR-10 56.42, beating the schedule's 95.62 / 78.93 / 54.02 even though the schedule overspent to nearly $5.27$/$4.13$. So decaying the noise bought nothing once the budget is held honest, and I now have a floor I trust. Its accuracy is governed by a clip threshold $C = 1.0$ that I picked once and never touched, and that is the knob I want to attack — not because I have a better value, but because in standard DP-SGD $C$ does two opposite jobs at once and there is no formula that turns $(\epsilon,\delta)$ into a good one. $C$ is the sensitivity, so the noise $\sigma C/B$ rises with it; and it is the clipping radius, so lowering it crushes the large per-example gradients, erasing the magnitude differences that say which examples mattered and biasing the aggregate. The sweet spot is narrow, accuracy is savagely sensitive to it, and unlike $\sigma$ — which the accountant hands me for free — $C$ would have to be grid-searched. I want that knob gone entirely.

I propose **AUTO-S automatic clipping** (Bu et al., NeurIPS 2023): replace clipping with per-sample normalization, which removes $C$ as a tunable quantity. The argument starts from where good private models actually live — consistently at a *small* $C$, small enough that the clip is active on most examples every step. In that regime stare at the clip factor $\min(C/\|g_i\|, 1)$: when $C$ is small relative to a typical $\|g_i\|$, the $\min$ just picks $C/\|g_i\|$, so the clipped gradient is $g_i\cdot C/\|g_i\| = C\,g_i/\|g_i\|$. Every clipped gradient comes out with the same length $C$, pointed in its own direction. In the regime everyone uses, Abadi's clipping is *not really clipping* — it is per-sample normalization scaled by $C$.

Before committing, I check that normalization is a *sensible* operation, not just an algebraic limit. Ask what I want from the aggregated clipped gradient: it should point the same way as the true gradient so the step makes progress. Maximize the dot-product alignment between the aggregate of clipped gradients and the aggregate of raw ones, subject to each clipped gradient having length at most $C$ (the sensitivity constraint). The objective is linear in the per-sample scale factors over a box, so the optimum sits at a corner — push each factor to its max $C/\|g_i\|$ when $g_i$ correlates positively with the aggregate. When the per-sample gradients are concentrated, mostly agreeing with their own sum, every factor is at its max and the optimal factor is exactly $C/\|g_i\|$. And that dot-product is the first-order descent term in the smoothness expansion, so "maximize alignment" literally means "maximize the guaranteed one-step decrease." Normalization has a reason to exist.

Does the $C$ in $C\,g_i/\|g_i\|$ do any real work? After normalization each per-sample contribution has length exactly $C$, so the sensitivity is $C$, the noise is $\sigma C$, and the noise-to-sensitivity ratio is $\sigma$ — *independent of $C$*. The accountant only ever sees $\sigma$, $q$, $T$, so any constant $C > 0$ gives the same privacy; from the privacy side $C$ is pure gauge. The optimizer confirms it: $C$ multiplies the whole privatized gradient, so it is glued to the learning rate — a run with $(C, \text{lr})$ equals a $C=1$ run with $\text{lr}' = \text{lr}\cdot C$ — and the harness fixes the lr. So I fix $C=1$ and never tune it; the two DP-specific knobs collapse to one, $\sigma$, free from the accountant.

But pure normalization throws something away, and a bad bonus is worse than none. After normalization *every* clipped per-sample gradient has length exactly $1$: a tiny gradient and a huge one come out indistinguishable unit vectors, so the magnitude information is gone. That is harmless until the directions disagree. Picture a balanced batch where positives push one way and negatives the other, with a genuinely nonzero true gradient. With normalization every positive contributes a unit vector one way, every negative a unit vector the other, and in a balanced batch they cancel — the aggregated normalized gradient is *zero*, so the optimizer sits still though it is not at a stationary point. A "lazy region" where it should be moving; this is the AUTO-V defect. The cure is to restore a little magnitude dependence near zero without reintroducing a threshold: put a positive constant in the denominator,
$$g_i \ \longrightarrow\ \frac{g_i}{\|g_i\| + \gamma}.$$
When $\|g_i\| \gg \gamma$ this is $\approx g_i/\|g_i\|$, the normalization behavior intact; when $\|g_i\| \to 0$ it is $\approx g_i/\gamma$, which keeps the direction and *shrinks with the magnitude* instead of inflating to a unit vector. Magnitude order is preserved because $\|g\|/(\|g\|+\gamma)$ is strictly increasing, and that monotonicity is what kills the lazy region — opposite-class gradients of different sizes no longer cancel to zero. Near convergence the batch sum $\to (1/\gamma)\sum_i g_i$, the ordinary SGD direction, so the mechanism turns smoothly *into* SGD where SGD behaves well. Privacy is untouched: $\|g_i/(\|g_i\|+\gamma)\| < 1$, the sensitivity is still bounded by $1$, the ratio is still $\sigma$. This is AUTO-S.

Two implementation choices are specific to this harness and matter. First, the stability constant: the canonical AUTO-S keeps $\gamma \approx 0.01$, tiny so the denominator barely perturbs normalization, but here $\gamma$ is pinned at $1.0$ to keep the *frozen* `SGD(lr=0.1, momentum=0.9)` cosine schedule stable — a tiny $\gamma$ makes the effective step (which scales like $1/\gamma$ near convergence) far larger than the fixed schedule was tuned for. With $\gamma = 1.0$ the scale is $1/(\|g_i\| + 1)$, sitting near $0.5$ for typical clipped gradients: this is *not* the pure-normalization limit but a soft, magnitude-aware down-scaling that behaves near convergence like plain SGD scaled by $1/\gamma = 1$. Second, the noise: the sensitivity after AUTO-S clipping is $1$, so the noise is calibrated to $1$ (`sigma * 1.0 / B`), with the $C$ of the standard rung replaced by the constant. `get_effective_sigma` returns the unchanged constant $\sigma$ — so, unlike the schedule, this rung reports the calibrated $\sigma$ and *cannot* drift off budget, which makes it a legitimate sibling of the floor. With $\gamma = 1.0$ making this a gentle variant, I expect a near-tie with the standard floor — a wash around 96 on MNIST, near 79.6 on FMNIST, and at most a hair on CIFAR-10 — which would mean the real headroom is not in the per-example clip shape but in making the clip *adapt* to the drifting gradient distribution, the move the next rung makes.

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
