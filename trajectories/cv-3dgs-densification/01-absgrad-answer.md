**Problem.** The original 3DGS densification signal is a *signed*, summed projected-center gradient
`g_x = Σ_j ∂L_j/∂μ_x`. Over a large splat in a textured region the per-pixel sub-gradients have opposite
signs (the L1 residual, compositing, and the projected-Gaussian derivative all flip across the
footprint), so they cancel and the net stays small — the worst over-reconstructed Gaussians are never
selected for split, leaving grass/foliage/distant texture soft. Lowering the threshold cannot fix a
statistic that measures *net center motion* rather than *total error*.

**Key idea (AbsGS).** Accumulate the per-pixel *absolute* sub-gradients before the sum:
`ĝ_x = Σ_j |∂L_j/∂μ_x|`, `ĝ_y = Σ_j |∂L_j/∂μ_y|`, and densify on `‖ĝ‖_2`. By the triangle inequality
`‖g‖_2 ≤ ‖ĝ‖_2`; the gap is exactly the cancellation the signed rule was blind to. This is a
density-control statistic only — the optimizer still steps with the true signed gradient.

**Why it works.** Where per-pixel terms cancel, the absolute statistic reports the large magnitude
underneath, so over-reconstructed splats finally cross the threshold and get split; where terms already
agree in sign (small coherent primitives), it nearly equals the signed statistic, so the cases the
automaton already handled are undisturbed.

**Scaffold edit / what the harness exposes.** The fixed loop renders with `absgrad=True`, so
`info["means2d"].absgrad` (the per-pixel-absolute accumulation) is populated after backward — read it
directly. Do *not* use `info["means2d"].grad.abs()`, which only flips the sign of the already-collapsed
signed gradient and cannot undo cancellation (kept only as a defensive fallback). The harness exposes a
**single** densification accumulator and **one** `grow_grad2d`, so this is the `gsplat` drop-in form:
`.absgrad` feeds **both** clone and split at one threshold (not a surgical signed-clone /
abs-split two-threshold rule, which the harness does not allow). Because `‖ĝ‖_2 ≥ ‖g‖_2`, the threshold
must rise — `grow_grad2d = 0.0006` (vs `0.0002` signed). Everything else is the inherited default:
clone-small / split-large at `grow_scale3d`, opacity-floor + size prune, periodic reset, same schedule.

**What to watch.** Largest PSNR gains on the textured outdoor scenes (garden, bicycle, stump) where
cancellation cost the most fine detail; smaller move on the structured indoor scene (bonsai). The
single-threshold drop-in grows less discriminately than a two-threshold rule, so the next pressure is a
*richer selection signal* and a *less disruptive split*, not more flooding.

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
