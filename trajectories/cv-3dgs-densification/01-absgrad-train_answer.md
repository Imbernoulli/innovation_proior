The original 3DGS densification automaton is sound in almost every part, and the one place it fails is concrete and reproducible: render a trained scene and most of the image is sharp, but grass, foliage, uneven pavement, and distant texture stay soft, because a few large flat splats sit over regions that need many small primitives and are never selected for subdivision. The loss is not blind to those pixels — the region is visibly wrong — and the renderer is not the limit, since nearby regions drawn with smaller splats are sharp. The fault is the density-control *decision*: the large primitive that ought to be split is not flagged. The inherited rule scores each Gaussian by the magnitude of the loss gradient with respect to its projected 2D center, averaged over the views it appears in, and where that average is large it clones the Gaussian if small and splits it if large. The intuition is reasonable — gradient descent is already asking how far each projected center wants to move, and a large movement request is a cheap proxy for "this primitive does not represent its region well" — and the signal is free, riding on gradients already computed. The defect is that a sum of gradients is not a sum of *errors*.

Look inside one component of the projected-center gradient for one Gaussian, summing over the $m$ pixels its footprint covers:

$$g_x = \frac{\partial L}{\partial \mu_x} = \sum_{j=1}^{m} \frac{\partial L_j}{\partial \mu_x}.$$

This net sum can be small for two completely different reasons: because every per-pixel term is small (the primitive is genuinely fine), or because the per-pixel terms are *large but of opposite sign* and cancel before the sum is taken. The density controller only ever sees the net, so it cannot tell "nobody is pulling" from "everybody is pulling in different directions" — and the second case is exactly what a large splat averaging over texture produces. Chaining one per-pixel term through the renderer, $\frac{\partial L_j}{\partial \mu_x} = \sum_a \frac{\partial L_j}{\partial C_j^a}\frac{\partial C_j^a}{\partial \alpha_i}\frac{\partial \alpha_i}{\partial \mu_x}$, every factor invites a sign flip across a large footprint: the photometric residual flips as pixels go from too-bright to too-dark, the compositing term pits a Gaussian's own color against the attenuation of everything behind it, and the geometric factor $\frac{\partial \alpha_i}{\partial \mu_x} \propto (p_{j,x} - \mu_x)\,G_i^{2d}(p_j)$ is antisymmetric about the projected center, so pixels on opposite sides contribute opposite directions almost by construction. The signed statistic adds these terms before taking a norm, so a large splat over high-frequency detail can have many dissatisfied pixels and still report a small gradient. That is a structural blind spot, not noise — and lowering the threshold does not fix it, because a net that is near zero *from cancellation* stays below any reachable cut while a lower threshold floods in harmless primitives elsewhere. I need to change *what* is measured, not where I cut it.

I propose **AbsGS**, the homodirectional absolute gradient. The fix is to take each pixel's dissatisfaction *before* directions are allowed to cancel: keep the per-pixel magnitude and discard its sign before summing over pixels,

$$\hat g_x = \sum_j \left|\frac{\partial L_j}{\partial \mu_x}\right|, \qquad \hat g_y = \sum_j \left|\frac{\partial L_j}{\partial \mu_y}\right|,$$

and densify on $\lVert \hat g \rVert_2$. The sign of $\partial L_j/\partial \mu_x$ tells me which way pixel $j$ would push the center — an optimization question, not a density-control one — so I discard it for the split/clone decision while leaving it untouched for the optimizer, which still steps with the true signed gradient. This is not a fabricated update direction; it is consulted *only* for densification. The triangle inequality guarantees the right ordering: $|g_x| = |\sum_j \partial L_j/\partial \mu_x| \le \sum_j |\partial L_j/\partial \mu_x| = \hat g_x$, hence $\lVert g \rVert_2 \le \lVert \hat g \rVert_2$, with the gap becoming strict exactly when cancellation was present. That gap is the missing signal — where the old rule sees a small net pull, the new rule sees the large total magnitude underneath — and where the per-pixel terms already agree in sign, as in small coherent primitives, the two statistics nearly coincide, so the cases the automaton already handled are undisturbed.

The one place this can silently go wrong is the implementation. Taking `info["means2d"].grad.abs()` after backward is *not* the right thing: that tensor is already the signed per-pixel terms summed by the rasterizer, so an absolute value there only flips the sign of the collapsed result and cannot recover magnitudes that canceled *before* the sum. To get $\sum_j |\partial L_j/\partial \mu|$ I need the rasterizer to accumulate the absolute per-pixel sub-gradients *during* backward and expose them — and the fixed substrate already calls `rasterization(..., absgrad=True)`, so after backward `info["means2d"].absgrad` holds exactly that. I read it directly and keep `.grad.abs()` only as a defensive fallback for the case where the attribute is absent.

One design choice is forced by the harness. The most surgical form of the idea keeps the original *signed* statistic for clone — where the footprint is small and there is little cancellation — and uses the homodirectional statistic only for split, which needs two thresholds, a low signed $\tau_{\text{clone}}$ and a higher $\tau_{\text{split}}$. But the scaffold exposes a *single* densification accumulator and a single `grow_grad2d` feeding both `duplicate` and `split`, so I take the `gsplat` drop-in path it is shaped for: `.absgrad` is the grow statistic for both clone and split at one unified threshold. The cancellation fix is intact and the optimizer still uses signed gradients; it is simply less targeted than the two-threshold rule, which I do not pretend to here. That decision forces the threshold up. Because $\lVert \hat g \rVert_2 \ge \lVert g \rVert_2$ everywhere, keeping the old `0.0002` would over-select — every primitive that used to qualify still does, plus all the ones whose cancellation the absolute statistic now uncovers — so I raise `grow_grad2d` to `0.0006`, three times the signed threshold, in the band the `.absgrad` drop-in commonly uses. Everything else in the default fill stays exactly as inherited — the clone-small / split-large size split at `grow_scale3d`, the opacity-floor-plus-size prune, the periodic reset to $2\cdot$`prune_opa`, and the `500 / 15000 / 3000 / 100` schedule — because the cancellation fix is the only change this rung makes. I expect the gain to concentrate on the textured outdoor scenes (garden, bicycle, stump) where signed cancellation cost the most fine detail, and a smaller move on the structured indoor scene (bonsai); the residual risk I carry forward is that one blunt threshold for both clone and split grows aggressively, which is the pressure on the next rung.

```python
@dataclass
class CustomStrategy(Strategy):
    """AbsGS: absolute gradient densification for fine detail recovery."""

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0006
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        return {"grad2d": None, "count": None, "scene_scale": scene_scale}

    def step_pre_backward(self, params, optimizers, state, step, info):
        info["means2d"].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step >= self.refine_stop_iter:
            return

        if hasattr(info["means2d"], "absgrad"):
            grads = info["means2d"].absgrad.clone()       # per-pixel |dL/dmu| summed in backward
        else:
            grads = info["means2d"].grad.abs().clone()    # fallback only; cannot undo cancellation
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n = len(list(params.values())[0])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n, device=grads.device)
            state["count"] = torch.zeros(n, device=grads.device)

        sel = (info["radii"] > 0.0).all(dim=-1)
        gs_ids = torch.where(sel)[1]
        state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

        if step > self.refine_start_iter and step % self.refine_every == 0:
            avg_grads = state["grad2d"] / state["count"].clamp_min(1)
            scene_scale = state["scene_scale"]

            is_grad_high = avg_grads > self.grow_grad2d
            scale_max = torch.exp(params["scales"]).max(dim=-1).values
            is_small = scale_max <= self.grow_scale3d * scene_scale

            is_dupli = is_grad_high & is_small            # clone (single abs threshold, drop-in form)
            if is_dupli.sum() > 0:
                duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

            is_split = is_grad_high & ~is_small            # split (same abs threshold)
            is_split = torch.cat([is_split, torch.zeros(is_dupli.sum(), dtype=torch.bool, device=is_split.device)])
            if is_split.sum() > 0:
                split(params=params, optimizers=optimizers, state=state, mask=is_split)

            is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
            if step > self.reset_every:
                scale_max = torch.exp(params["scales"]).max(dim=-1).values
                is_prune = is_prune | (scale_max > self.prune_scale3d * scene_scale)
            if is_prune.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

            state["grad2d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params=params, optimizers=optimizers, state=state,
                      value=self.prune_opa * 2.0)
```
