The absolute-gradient rung did what I argued it would, and the per-scene numbers also show where it ran out: PSNR landed at garden 29.745, bicycle 27.026, bonsai 33.051, stump 27.850 — a four-scene mean of 29.418. bicycle and stump are the laggards, both sprawling outdoor scenes with thin high-frequency structure (spokes, grass blades, bark) seen sharply in only a handful of views, while bonsai sits comfortably ahead because its detail is structured and seen consistently. So AbsGS fixed *sign cancellation within a footprint*, but it left a second, orthogonal blind spot: the densification signal is still an *average over views*, and an average washes out exactly the failures concentrated in a few views. The accumulator at each refinement step is $\text{avg\_g} = \frac{1}{M}\sum_k \lVert \hat g_i^k \rVert$, the per-view absolute gradient norm summed over the $M$ views the Gaussian appeared in, divided by the count. A Gaussian that is badly wrong in *one* view but adequate in the other $M-1$ gets one large per-view norm buried under many small ones, so the average is dragged toward "fine," the Gaussian never crosses `grow_grad2d`, and the detail stays soft. The average is a low-pass filter on precisely the view-specific spikes that mark under-reconstruction in sparse-view regions, and lowering the threshold again would only flood in uniformly mediocre primitives while still possibly missing the spikes.

I propose two stackable refinements on top of AbsGS — the **Taming-3DGS max-grad blend** for selection and a **revised-opacity split** for the operation. I want a statistic that fires when *either* a Gaussian is persistently wrong across its views *or* sharply wrong in a few. The average captures the first; the natural complement of a mean is a *max*, so alongside $\text{avg\_g}$ I track the per-Gaussian running maximum per-view gradient norm over the accumulation window, $\text{max\_g}_i = \max_k \lVert \hat g_i^k \rVert$. A Gaussian that spikes in one view has a large $\text{max\_g}$ even when its $\text{avg\_g}$ is small; a uniformly slightly-wrong one has a moderate average and an unremarkable max. Since the harness gives a single threshold, I blend them into one criterion,

$$\text{combined} = w_{\text{avg}}\cdot\text{avg\_g} + w_{\text{max}}\cdot\text{max\_g}, \qquad w_{\text{avg}} = 0.7,\ w_{\text{max}} = 0.3,$$

weighting the average more heavily because persistent error is the more trustworthy signal and the max is a corrective minority report. A pure max would over-densify on single noisy views — the over-selection I keep refusing — and a pure average is what just failed, so the split is deliberate: the average stays dominant so growth does not regress to noise, but the max gets enough weight (0.3, not 0.05) that a genuine view-specific spike can carry a Gaussian over the threshold on its own. Computing the max per Gaussian over a window of visible ids is a scatter-reduce with `amax` (`scatter_reduce_(..., reduce="amax", include_self=True)`) into a zero-initialized `grad2d_max` that I reset each refinement step alongside the sum and count — one new piece of state.

The absgrad numbers also point at the split *operation*, not just its selection. The inherited covariance-sampled split replaces a flagged large Gaussian with two children, each keeping the parent's full shape and *full opacity*, at offsets drawn from the parent's covariance. The opacity is the subtle defect: before the split a ray through the region passes one bump of opacity $\alpha$ and is attenuated by $(1-\alpha)$; after, with two children each at $\alpha$, it passes two bumps and is attenuated by $(1-\alpha)^2$, which is *smaller* — the region jumps darker and more opaque the instant the split fires, and the optimizer must spend iterations undoing a density jump I introduced. With AbsGS densifying more often, I pay this tax more often. The fix is to make the split *cumulative-$\alpha$ invariant*: choose the children's opacity so two of them composite to the parent's single-bump transmittance. Solving $(1 - \alpha_{\text{child}})^2 = (1 - \alpha_{\text{parent}})$ gives

$$\alpha_{\text{child}} = 1 - \sqrt{1 - \alpha_{\text{parent}}}.$$

The limits check out: for small $\alpha$, $1 - \sqrt{1-\alpha} \approx \alpha/2$, so two children of half the parent's opacity compose to roughly the parent — exactly "don't double the occlusion" — and for $\alpha \to 1$ each child goes toward $1$ as it must, since two transparent children cannot reproduce a fully opaque parent. The compound alpha-blending stays invariant under the split, so the rendered image barely moves and the optimizer resumes from where it was instead of clawing back a brightness jump. The fixed substrate exposes this directly: the `split` op takes a `revised_opacity=True` flag that applies exactly the $1 - \sqrt{1-\alpha}$ child opacity, so I flip the flag rather than reimplement the geometry, and I keep the covariance-sampled child *positions* unchanged this rung — the opacity invariance is the piece that removes the density jump.

Both changes keep late densification productive: the blend surfaces view-specific candidates the average had given up on, and the invariant split means late splits no longer cost a recovery period. With the inherited `refine_stop_iter = 15000` I would freeze the set right when the max-grad signal is still surfacing genuine candidates, so I extend it to 18000 — three thousand more productive steps, safe only *because* the invariant split stopped making late edits disruptive. I also nudge `grow_grad2d` down to `0.0005` from absgrad's `0.0006`: the blend's max term is, for spiking Gaussians, smaller than a pure-average-of-large-views, so a marginally lower threshold keeps the growth rate where I want it while leaving the precise selection to the blend rather than the threshold. This is deliberately *not* the full Taming-3DGS — the harness exposes no per-pixel blending-weight, distance, or saliency accumulations, no writable rasterizer backward, and no budget interface, so the parabolic count schedule, the multi-signal importance score, the `multinomial` selection, and the rasterizer rewrite are out of reach; only the max-grad blend and the consistent split fit this contract, and I take only those. I expect the largest gains on the laggard outdoor scenes from view-specific selection, a modest move on garden, and bonsai as the open question — its detail is already seen consistently, so the max term adds little there and any gain comes from the invariant split keeping the dense indoor structure from being perturbed.

```python
@dataclass
class CustomStrategy(Strategy):
    """AbsGS + Taming-3DGS (max-grad blend) + New Split (revised opacity)."""

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0005   # slightly lower than absgrad (more aggressive growth)
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 18_000  # later stop — max-grad keeps finding splits
    reset_every: int = 3000
    refine_every: int = 100
    # Taming-3DGS blend weights
    avg_weight: float = 0.7
    max_weight: float = 0.3

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        return {
            "grad2d": None, "count": None, "grad2d_max": None,
            "scene_scale": scene_scale,
        }

    def step_pre_backward(self, params, optimizers, state, step, info):
        info["means2d"].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step >= self.refine_stop_iter:
            return

        # AbsGS: absolute gradients (key vs. default)
        if hasattr(info["means2d"], "absgrad"):
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.abs().clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n = len(list(params.values())[0])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n, device=grads.device)
            state["count"] = torch.zeros(n, device=grads.device)
            state["grad2d_max"] = torch.zeros(n, device=grads.device)

        sel = (info["radii"] > 0.0).all(dim=-1)
        gs_ids = torch.where(sel)[1]
        grad_norms = grads[sel].norm(dim=-1)
        state["grad2d"].index_add_(0, gs_ids, grad_norms)
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
        # Taming-3DGS: track per-Gaussian max gradient (catches view-specific spikes)
        state["grad2d_max"].scatter_reduce_(0, gs_ids, grad_norms, reduce="amax", include_self=True)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            avg_grads = state["grad2d"] / state["count"].clamp_min(1)
            # Blended signal: avg for persistent errors, max for view-specific
            combined = self.avg_weight * avg_grads + self.max_weight * state["grad2d_max"]
            scene_scale = state["scene_scale"]

            is_grad_high = combined > self.grow_grad2d
            scale_max = torch.exp(params["scales"]).max(dim=-1).values
            is_small = scale_max <= self.grow_scale3d * scene_scale

            is_dupli = is_grad_high & is_small
            if is_dupli.sum() > 0:
                duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

            # New Split: revised_opacity=True preserves alpha-blending under splits
            is_split = is_grad_high & ~is_small
            is_split = torch.cat([is_split, torch.zeros(is_dupli.sum(), dtype=torch.bool, device=is_split.device)])
            if is_split.sum() > 0:
                split(params=params, optimizers=optimizers, state=state, mask=is_split, revised_opacity=True)

            is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
            if step > self.reset_every:
                scale_max = torch.exp(params["scales"]).max(dim=-1).values
                is_prune = is_prune | (scale_max > self.prune_scale3d * scene_scale)
            if is_prune.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

            state["grad2d"].zero_()
            state["count"].zero_()
            state["grad2d_max"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params=params, optimizers=optimizers, state=state,
                      value=self.prune_opa * 2.0)
```
