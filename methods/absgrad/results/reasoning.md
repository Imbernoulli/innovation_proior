I start from the trained-scene failure, because that is the only part I need to explain. A few large Gaussians sit over grass, cement, foliage, or distant detail, and the render is blurry there. The loss is not blind to those pixels; the region is visibly wrong. The renderer is also not the limiting factor, because nearby regions represented by smaller splats are sharp. So the broken piece must be the density-control decision: the large primitive that should be split is not being selected for split.

The existing rule looks reasonable at first. It accumulates the magnitude of the gradient of the loss with respect to the projected 2D mean of a Gaussian, averaged over the views in which that Gaussian participates:

`nabla_mu_i L = (1/M) sum_k ||dL^k / dmu_i^k||_2`.

If this value is high, a small Gaussian is cloned and a large Gaussian is split. I should not dismiss that idea too quickly. Gradient descent is already asking how much each projected center wants to move; a large movement request is a cheap proxy for "this primitive is not representing its region well." That is why the rule works for many under-covered regions. A small primitive covering only a few pixels can receive a coherent pull, cross the threshold, and get duplicated.

But a sum of gradients is not the same thing as a sum of errors. I need to look inside one component of the projected-center gradient for one Gaussian. Dropping the view index for the moment,

`g_x = dL/dmu_x = sum_{j=1}^m dL_j/dmu_x`,

where `m` is the number of pixels covered by the splat. This sum can be small for two very different reasons. It can be small because every pixel term is small, which means the primitive is probably fine. Or it can be small because the pixel terms are large and have opposite signs. The density controller only sees the net sum, so a region where "nobody is pulling" and a region where "everybody is pulling in different directions" look identical to it. I want to know whether the second case can actually arise for a large splat over texture, because if it cannot, there is nothing to fix.

So I check whether those per-pixel terms can really carry different signs. Chain one pixel term through the renderer:

`dL_j/dmu_x = sum_a (dL_j/dC_j^a) (dC_j^a/dalpha_i) (dalpha_i/dmu_x)`,

with `a` running over color channels. The first factor comes from the photometric residual. For the L1 part of the loss, its sign depends on whether the rendered channel is above or below the target channel at that pixel. In a textured region, some pixels are too bright and some are too dark, so this factor can flip across the footprint.

The second factor is the effect of this Gaussian's alpha on the composited color. Increasing `alpha_i` adds more of this Gaussian's own color, but it also reduces the contribution of Gaussians behind it. In front-to-back notation, differentiating the compositing equation gives an own-color term and negative attenuation terms for later primitives. Which sign wins depends on colors, opacities, and depth ordering at that pixel. So this factor is also not fixed across the footprint.

The third factor is the geometric one. The alpha is `sigmoid(o_i) G_i^2d(p_j)`. Differentiating the Gaussian with respect to its mean gives a factor proportional to `(p_{j,x} - mu_x) G_i^2d(p_j)`; differentiating with the opposite coordinate convention writes the opposite sign, but either way the sign is controlled by the pixel's position relative to the projected center. Pixels on opposite sides of the projected center contribute opposite x-directions automatically. The y component has the same structure.

That third factor gives me pause for a different reason. It is antisymmetric in `(p_j - mu_x)`, which means it can cancel even when nothing is wrong with the residual. A small splat that is uniformly too dark, sampled symmetrically about its center, would have `dL_j/dC_j` constant and `(p_j - mu_x)` summing to zero, so `g_x` would vanish purely from geometry, not from texture. If that were the dominant effect, the signed sum would be a poor statistic everywhere and my story about large-vs-small would be wrong. I need a concrete case to separate "geometric cancellation of a uniform residual" from "texture cancellation of a sign-flipping residual," because only the second is the failure I am chasing.

So I build the smallest example that has both factors. Put one flat splat at `mu = 0`, color `c = 0.6`, opacity 1, on a 9x9 pixel grid spanning `[-2, 2]` in each axis, with `G(p) = exp(-||p - mu||^2 / 2s^2)`, `s = 1.2`. Approximate the composite by a single layer `C_j = c G(p_j)` so I can compute by hand, and use the L1 residual sign `dL_j/dC_j = sign(C_j - target_j)`. The per-pixel sub-gradient is then `dL_j/dC_j * c * (G(p_j) (p_j - mu)/s^2)` per axis. For the target I use a checkerboard `0.6 + 0.3 sign(sin 3x sin 3y)`, which is the high-frequency case: the residual sign flips from pixel to pixel.

Summing the signed sub-gradients over the footprint gives `g_x = 0`, `g_y = 0`, so `||g||_2 = 0` exactly. Summing the per-pixel absolute values first gives `hat_g_x = hat_g_y = 11.09`, so `||hat_g||_2 = 15.68`. Every one of the 81 pixels is contributing a real sub-gradient, but the signed sum has erased all of it. This is the case I was worried did not exist, and it does: a large splat over texture can be maximally dissatisfied and report a net pull of zero.

Now I contrast it with a uniform residual on the same geometry to make sure I am not just rediscovering the antisymmetry trap. If I make the splat uniformly too dark instead of checkerboarded, the signed sum still cancels, but for the boring geometric reason, and so does the absolute statistic carry no extra meaning there beyond magnitude. The difference that matters is that the checkerboard has large per-pixel magnitudes that survive into `hat_g` while the uniform case is genuinely small per pixel near the center. So the operative trigger is a sign-flipping residual over a large footprint, which is exactly over-reconstruction of fine detail. A small splat in an under-covered region does not have room for the residual to flip many times inside its footprint, and its net pull tends to survive. That is why the symptom is asymmetric: small under-covered primitives keep a coherent net pull, large over-covering ones cancel.

Lowering the threshold is the tempting fix, but the example shows why it cannot work for the worst case. The checkerboard splat reports `||g||_2 = 0`. No positive threshold, however small, selects a zero. Meanwhile lowering the threshold admits many primitives whose small signed gradients were harmless, increasing the primitive count and memory. The threshold is only a decision boundary; the deeper problem is that the measured number is net movement of the center, not total evidence that the covered pixels are poorly represented.

What I want instead is the magnitude of each pixel's dissatisfaction before directions are allowed to cancel. For one pixel, the sign of `dL_j/dmu_x` tells me which way that pixel would like to move the center. That direction is not the density-control question. The density-control question is whether this pixel is badly represented enough to argue for more local capacity. So I keep the magnitude and discard the sign before summing over pixels:

`hat_g_x = sum_j |dL_j/dmu_x|`, `hat_g_y = sum_j |dL_j/dmu_y|`, and then `||hat_g||_2`.

This is not an optimization gradient. It is a diagnostic accumulator for density control only. The optimizer still uses the true signed gradient; otherwise I would be fabricating parameter-update directions that no loss derivative asked for. The new statistic is only consulted when deciding whether to split or clone.

I want to know the general relationship between the two statistics, not just on the one example. Per component, the triangle inequality gives `|g_x| = |sum_j dL_j/dmu_x| <= sum_j |dL_j/dmu_x| = hat_g_x`, and likewise for y, so `||g||_2 <= ||hat_g||_2` always. The example sits at the extreme of that inequality: `0 <= 15.68`. The interesting question is when it is tight. If I take the same numbers but flip the signs so all per-pixel terms agree, the signed component becomes the absolute component term by term, and I get `||g||_2 = 15.68 = ||hat_g||_2` — equality. So the gap `||hat_g||_2 - ||g||_2` is literally the amount of cancellation that happened, zero when the pixels already agree and maximal when they perfectly oppose. The new statistic does not change the verdict for primitives whose pixels agree; it only rescues the ones the old rule was blind to.

There is a code trap that the example also settles. It is tempting to take `info["means2d"].grad.abs()` after backward and call that the absolute statistic. But the rasterizer has already summed the per-pixel terms into the signed gradient before I see it. On the checkerboard footprint that collapsed value is `g_x = 0`, and `|0| = 0`; abs-after-sum recovers nothing. Even on a less extreme footprint the loss is severe: for one axis of a large splat I get a collapsed signed grad of `-0.49`, whose absolute value is `0.49`, against a true sum-of-absolutes of `2.36`. Taking the absolute value after the sum keeps about a fifth of the magnitude. So the rasterizer itself must accumulate `|dL_j/dmu_x|` and `|dL_j/dmu_y|` per pixel during backward and expose those sums, either as extra channels in the official-style code or as `.absgrad` in `gsplat`. The collapsed tensor cannot be repaired downstream.

I also need to decide where to use the new statistic. The causal failure is large-footprint over-reconstruction, and the operation that fixes that is split. Clone handles a different case: a small primitive in an under-covered area. There the signed gradient was already useful — the footprint is small enough that the residual rarely flips many times inside it, so cancellation is much less of a problem, as the asymmetry above showed. So the most conservative rule keeps the original signed average gradient for clone and uses the homodirectional average gradient for split. The implementation accumulates both `xyz_gradient_accum` and `xyz_gradient_accum_abs`, then calls clone with the signed statistic and split with the absolute statistic.

There is a related implementation variant in `gsplat`: when `absgrad=True`, the strategy reads `info[key].absgrad` and uses it as the grow statistic for both duplicate and split, with a higher unified threshold. That is still faithful to the core statistic, but it is less targeted than the split-only rule, since it also rewrites the clone decision that was not broken.

The threshold has to rise because `||hat_g||_2` is at least as large as the old signed norm — that is the inequality I just checked, and on the checkerboard the inflation factor was unbounded. Keeping the old split threshold would over-select primitives whose absolute statistic is large only because the footprint is large, not because the fit is bad. So the split statistic gets a separate higher threshold, while clone keeps the old signed threshold. The exact value is an empirical quality-versus-memory knob; the inequality tells me the direction of the change, not the magnitude.

Finally, the size threshold still matters. The statistic can correctly flag a badly represented primitive, but if the scale threshold classifies it as small, the controller will duplicate instead of split, and duplication does not shrink an over-large primitive. Lowering the split/clone scale boundary can help route fine-detail primitives to split. That is secondary to the gradient fix: it routes selected primitives correctly, but it does not by itself solve cancellation in the selection statistic.

The resulting method is simple. Keep the true signed gradient for optimization. During rasterizer backward, additionally accumulate per-pixel absolute projected-mean sub-gradients per axis. Average both signed and absolute statistics over visibility. Use the signed statistic to clone small high-gradient primitives, use the absolute statistic to split large high-gradient primitives, prune and reset opacity exactly as before, and raise the split threshold because the absolute statistic dominates the signed one.

## Code sketch

```python
@torch.no_grad()
def absgrad_density_step(params, state, info, tau_clone=2e-4, tau_split=4e-4):
    # Accumulate signed and homodirectional projected-center gradients.
    signed = info["means2d"].grad.clone()
    absgrad = info["means2d"].absgrad.clone()

    visible = (info["radii"] > 0).all(dim=-1)
    ids = torch.where(visible)[1]

    state["signed_grad2d"].index_add_(0, ids, signed[visible].norm(dim=-1))
    state["abs_grad2d"].index_add_(0, ids, absgrad[visible].norm(dim=-1))
    state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    # At refinement steps, route by scale and by the two statistics.
    count = state["count"].clamp_min(1)
    scale_max = torch.exp(params["scales"]).max(dim=-1).values
    is_small = scale_max <= state["scene_scale"] * 0.01

    clone_mask = (state["signed_grad2d"] / count >= tau_clone) & is_small
    split_mask = (state["abs_grad2d"] / count >= tau_split) & ~is_small
    return clone_mask, split_mask
```
