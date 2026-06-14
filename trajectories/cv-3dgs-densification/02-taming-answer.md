**Problem (from step 1).** Absolute gradients fixed *within-footprint* sign cancellation, but the
densification signal is still an *average over views*, which buries Gaussians that are sharply wrong in a
few views under the many views where they look fine — the bicycle/stump failure (thin structure seen
broadside in only a handful of cameras). Separately, the inherited covariance-sampled split gives both
children the parent's *full opacity*, so two stacked children transmit `(1 − α)²` instead of `(1 − α)`:
the region jumps darker the instant the split fires, and the optimizer must claw it back.

**Key idea.** Two stackable refinements on top of AbsGS. (1) **Taming-3DGS max-grad blend** (Mallick et
al. 2024): alongside the averaged absolute gradient, track the per-Gaussian *running maximum* per-view
gradient, and densify on `combined = 0.7·avg + 0.3·max` — the average catches persistent error, the max
catches view-specific spikes the average washes out. (2) **Revised-opacity split** (Rota Bulo et al.,
ECCV 2024): split with child opacity `α_child = 1 − √(1 − α_parent)`, so two children compose to the
parent's transmittance `(1 − α)` and the alpha-blending stays invariant under the split.

**Why it works.** The blend's max term lets a Gaussian that fails in only a few views cross the
threshold (recovering thin high-frequency detail), with the average kept dominant (0.7) so growth does
not regress to over-densifying on single noisy views. The revised opacity removes the density jump at
each split, so the optimizer resumes immediately instead of undoing a brightness perturbation — which
matters more here because abs-gradient already densifies aggressively.

**Scaffold edit / what the harness exposes.** Add `grad2d_max` to the state and fill it with
`scatter_reduce_(..., reduce="amax")` over visible ids; blend at refine time. Flip the harness's
`split(..., revised_opacity=True)` — the `gsplat.strategy.ops.split` flag applies exactly
`1 − √(1 − sigmoid)`, so I do not re-implement the geometry. Keep AbsGS's `.absgrad`. Both changes keep
late densification productive, so extend `refine_stop_iter` to 18000 (only safe because the invariant
split stopped making late edits disruptive); nudge `grow_grad2d` to 0.0005. **Not** the full
Taming-3DGS: the harness exposes no per-pixel blending-weight/distance/saliency accumulations, no
writable rasterizer backward, and no budget interface, so the parabolic count schedule, the multi-signal
importance score, the `multinomial` selection, and the rasterizer rewrite are omitted — only the
max-grad blend and the consistent split fit this contract.

**What to watch.** Largest gains on the laggard outdoor scenes (bicycle 27.026, stump 27.850) from
view-specific selection; modest on garden (29.745); bonsai (33.051) is the open question — its detail is
already seen consistently across views, so the max term adds little and any gain comes from the invariant
split. Outdoor up, bonsai flat is the design signature.

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
