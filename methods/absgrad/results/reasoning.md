I start from the trained-scene failure, because that is the only part I need to explain. A few large Gaussians sit over grass, cement, foliage, or distant detail, and the render is blurry there. The loss is not blind to those pixels; the region is visibly wrong. The renderer is also not the limiting factor, because nearby regions represented by smaller splats are sharp. So the broken piece must be the density-control decision: the large primitive that should be split is not being selected for split.

The existing rule looks reasonable at first. It accumulates the magnitude of the gradient of the loss with respect to the projected 2D mean of a Gaussian, averaged over the views in which that Gaussian participates:

`nabla_mu_i L = (1/M) sum_k ||dL^k / dmu_i^k||_2`.

If this value is high, a small Gaussian is cloned and a large Gaussian is split. I should not dismiss that idea too quickly. Gradient descent is already asking how much each projected center wants to move; a large movement request is a cheap proxy for "this primitive is not representing its region well." That is why the rule works for many under-covered regions. A small primitive covering only a few pixels can receive a coherent pull, cross the threshold, and get duplicated.

But a sum of gradients is not the same thing as a sum of errors. I need to look inside one component of the projected-center gradient for one Gaussian. Dropping the view index for the moment,

`g_x = dL/dmu_x = sum_{j=1}^m dL_j/dmu_x`,

where `m` is the number of pixels covered by the splat. This sum can be small for two very different reasons. It can be small because every pixel term is small, which means the primitive is probably fine. Or it can be small because the pixel terms are large and have opposite signs. The density controller only sees the net sum, so it confuses "nobody is pulling" with "everybody is pulling in different directions." The latter is exactly what I expect from a large splat averaging over texture.

Now I check whether those per-pixel terms can really have different signs. Chain one pixel term through the renderer:

`dL_j/dmu_x = sum_a (dL_j/dC_j^a) (dC_j^a/dalpha_i) (dalpha_i/dmu_x)`,

with `a` running over color channels. The first factor comes from the photometric residual. For the L1 part of the loss, its sign depends on whether the rendered channel is above or below the target channel at that pixel. In a textured region, some pixels are too bright and some are too dark, so this factor can flip across the footprint.

The second factor is the effect of this Gaussian's alpha on the composited color. Increasing `alpha_i` adds more of this Gaussian's own color, but it also reduces the contribution of Gaussians behind it. In front-to-back notation, differentiating the compositing equation gives an own-color term and negative attenuation terms for later primitives. Which sign wins depends on colors, opacities, and depth ordering at that pixel. So this factor is also not fixed across the footprint.

The third factor is the geometric one. The alpha is `sigmoid(o_i) G_i^2d(p_j)`. Differentiating the Gaussian with respect to its mean gives a factor proportional to `(p_{j,x} - mu_x) G_i^2d(p_j)`; differentiating with the opposite coordinate convention writes the opposite sign, but either way the sign is controlled by the pixel's position relative to the projected center. Pixels on opposite sides of a large splat's center naturally contribute opposite x-directions. The y component has the same issue.

So the per-pixel sub-gradients do not have a stable sign. The L1 residual can flip them, compositing can flip them, and the projected Gaussian derivative almost invites opposite signs across a large footprint. The old statistic adds these signed terms before taking a norm. A large splat over high-frequency detail can have many dissatisfied pixels and still produce a small `g_x` and `g_y` after cancellation. That explains the observed asymmetry: small under-covered primitives often keep a coherent net pull, while large over-covering primitives can have their per-pixel pulls collide.

Lowering the threshold is the tempting fix, but it does not repair the statistic. If a net signed sum is small because of cancellation, I can lower the threshold and still miss the worst cases. Meanwhile I admit many primitives whose small signed gradients were harmless, increasing the primitive count and memory. The threshold is only a decision boundary; the deeper problem is that the measured number is net movement of the center, not total evidence that the covered pixels are poorly represented.

What I want instead is the magnitude of each pixel's dissatisfaction before directions are allowed to cancel. For one pixel, the sign of `dL_j/dmu_x` tells me which way that pixel would like to move the center. That direction is not the density-control question. The density-control question is whether this pixel is badly represented enough to argue for more local capacity. So I should keep the magnitude and discard the sign before summing over pixels.

That gives the homodirectional statistic:

`hat_g_x = sum_j |dL_j/dmu_x|`, `hat_g_y = sum_j |dL_j/dmu_y|`, and then `||hat_g||_2`.

This is not an optimization gradient. It is a diagnostic accumulator for density control only. The optimizer still uses the true signed gradient; otherwise I would be fabricating parameter-update directions that no loss derivative asked for. The new statistic is only consulted when deciding whether to split or clone.

The triangle inequality confirms that this statistic has the right ordering. For each component,

`|g_x| = |sum_j dL_j/dmu_x| <= sum_j |dL_j/dmu_x| = hat_g_x`,

and the same holds for y. Therefore `||g||_2 <= ||hat_g||_2`. The inequality becomes strict exactly when cancellation was present in at least one component. That is the missing case: the old rule sees a small net pull, while the new rule sees the large total magnitude underneath it. In the no-collision case, where pixel terms already agree in sign, the new statistic stays close to the old one.

I also need to decide where to use it. The causal failure is large-footprint over-reconstruction, and the operation that fixes that is split. Clone handles a different case: a small primitive in an under-covered area. There the signed gradient was already useful, and the footprint is small enough that collision is much less of a problem. So the most faithful rule is conservative: keep the original signed average gradient for clone, and use the homodirectional average gradient for split. The canonical implementation accumulates both `xyz_gradient_accum` and `xyz_gradient_accum_abs`, then calls clone with the signed statistic and split with the absolute statistic.

There is a related implementation variant in `gsplat`: when `absgrad=True`, the strategy reads `info[key].absgrad` and uses it as the grow statistic for both duplicate and split, with a higher unified threshold. That is still faithful to the core statistic, but it is less targeted than the split-only rule.

One code trap is important. Taking `info["means2d"].grad.abs()` after backward is not enough. That tensor is already the signed gradient after per-pixel terms have been summed by the rasterizer. Taking an absolute value there only changes the sign of the collapsed result; it cannot recover the magnitudes that canceled before the sum. The rasterizer must accumulate `|dL_j/dmu_x|` and `|dL_j/dmu_y|` per pixel during backward and expose those sums, either as extra channels in the official-style code or as `.absgrad` in `gsplat`.

The threshold has to rise because `||hat_g||_2` is at least as large as the old signed norm. Keeping the old split threshold would over-select primitives. The source implementation therefore uses a separate higher threshold for the absolute split statistic, while clone keeps the old signed threshold. The exact value is an empirical quality-versus-memory knob, not a theorem.

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
