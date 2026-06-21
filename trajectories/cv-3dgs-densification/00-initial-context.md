## Research question

3D Gaussian Splatting represents a scene as a collection of anisotropic 3D Gaussians and renders by differentiable rasterization. Training starts from a sparse Structure-from-Motion point cloud — far too few primitives to represent the scene — so the representation must change size during training: it has to add Gaussians where coverage is missing, subdivide Gaussians that smear over too much detail, and remove ones that go transparent or grow oversized. The single thing being designed is the **densification strategy** — the rule, called every refinement step, that decides which Gaussians to clone, split, prune, or leave alone. Everything else about the pipeline is fixed. The failure to fix is visible in real scenes: most regions render sharply, but grass, foliage, uneven pavement, and distant texture stay soft, because a few large flat splats sit over regions that need many small primitives and are never selected for subdivision. The metric is per-scene PSNR (higher is better) on novel views; the contribution must be a transferable density-control rule, not a change to the renderer, loss, optimizer, dataset, or evaluation protocol.

## Prior art / Background / Baselines

- **3D Gaussian Splatting (Kerbl et al., 2023).** Core idea: represent a scene as anisotropic 3D Gaussians and, during training, average each Gaussian's view-space positional gradient over the views it appears in; wherever that average exceeds a threshold, clone the Gaussian if it is small or split it into two children if it is large, while pruning near-transparent or oversized Gaussians and periodically resetting opacities to force pruning. Gap: the averaged gradient signal can be small in regions with mixed reconstruction error, so large over-reconstructed primitives are sometimes missed entirely.
- **Gradient aggregation over high-frequency regions.** Core idea: the densification statistic is the net view-space center gradient obtained by summing per-pixel contributions over each Gaussian's footprint. Gap: in textured or partially occluded regions the per-pixel contributions can partially cancel, so a splat that covers many poorly reconstructed pixels can still report a low net gradient and evade splitting.
- **Split, prune, and reset operations.** Core idea: splitting creates children by sampling new positions from the parent's covariance while preserving its scale, rotation, opacity, and color; pruning removes only Gaussians whose opacity falls below a small floor; opacity reset forces re-pruning. Gap: each split abruptly changes the rendered density and shape away from the optimized state, and opacity-floor pruning cannot see Gaussians that retain ordinary opacity but overfit only a few training views.

## Fixed substrate / Code framework

A `gsplat` CUDA training loop is frozen and must not be touched. It initializes Gaussians from the SfM points; renders each view with `rasterization(..., packed=False, absgrad=True)`, so `info["means2d"].grad` and `info["means2d"].absgrad` are populated after backward; computes the photometric loss `0.8·L1 + 0.2·(1 − SSIM)`; and runs AdamW with per-parameter learning rates and an exponential LR decay. Training is **30,000 steps per scene**; SH degree rises to 3 over the first few thousand steps. The loop calls the strategy through three hooks: `initialize_state(scene_scale)` once at the start, `step_pre_backward(...)` before `loss.backward()` (to retain the screen-space gradient), and `step_post_backward(...)` after backward and the optimizer step (where all densification happens). The loop hands the strategy an `info` dict with `means2d` (and its `.grad` / `.absgrad`), `width`, `height`, `n_cameras`, `radii`, `gaussian_ids`, and a `params` dict with `means [N,3]`, `scales [N,3]` (log), `quats [N,4]`, `opacities [N]` (logit), `sh0`, `shN`. It also provides the in-place operations in `gsplat.strategy.ops`: `duplicate`, `split` (with an optional `revised_opacity` flag), `remove`, `reset_opa`, `relocate`, `sample_add`, `inject_noise_to_position`.

## Editable interface

Exactly one region is editable — the `CustomStrategy` dataclass in `gsplat/custom_strategy.py` (lines 20–90): its dataclass hyperparameter fields and the three hook bodies. Every method on the ladder is a fill of this same contract. The starting point is the scaffold **default fill: original 3DGS densification** — signed averaged view-space gradient, clone-small / split-large at one threshold, opacity-floor + size pruning, periodic opacity reset. Each later method replaces exactly this dataclass and nothing else.

```python
# EDITABLE region of gsplat/custom_strategy.py (lines 20-90) — default fill: original 3DGS densification
@dataclass
class CustomStrategy(Strategy):
    """Original 3DGS densification: clone small, split large, prune transparent."""

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
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

        # Signed averaged view-space gradient (the inherited densification signal)
        grads = info["means2d"].grad.clone()
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

            is_dupli = is_grad_high & is_small          # clone small under-covered Gaussians
            if is_dupli.sum() > 0:
                duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

            is_split = is_grad_high & ~is_small          # split large over-covering Gaussians
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
            reset_opa(params=params, optimizers=optimizers, state=state, value=self.prune_opa * 2.0)
```

## Evaluation settings

Mip-NeRF 360 scenes (Barron et al., 2022) with every 8th image held out for testing, one seed (42). Four scenes span the range — **garden** and **bicycle** (outdoor, unbounded, heavy foliage and high-frequency texture), **bonsai** (indoor, structured), and **stump** (outdoor, held out from the public scenes). Metrics, per-scene: **PSNR** (higher is better, primary), **SSIM** (higher is better), **LPIPS** (lower is better). Scoring is per-scene PSNR at the end of the 30,000-step budget.
