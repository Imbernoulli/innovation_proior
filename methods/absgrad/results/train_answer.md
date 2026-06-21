The densification stage of 3D Gaussian Splatting starts from a sparse Structure-from-Motion point cloud and must grow the primitive set to match the scene. The inherited controller accumulates the magnitude of the signed loss gradient with respect to each primitive's projected 2D center, averaged over the views in which the primitive appears. When this average exceeds a threshold, the primitive is cloned if it is small or split if it is large. The rule works naturally for small under-covered primitives: their footprints cover only a few pixels, so the per-pixel sub-gradients usually agree in sign and produce a coherent net pull. It fails for large primitives that cover fine texture, such as grass, foliage, or distant detail. Across a large footprint, the per-pixel sub-gradients have opposite signs: the L1 residual flips between bright and dark pixels, alpha-compositing effects can flip signs depending on depth and color ordering, and the projected Gaussian derivative pulls in opposite directions on opposite sides of the center. After summation these terms cancel, so the net gradient stays small even though many pixels are poorly represented. Lowering the threshold does not solve the problem, because it would also densify primitives with genuinely small gradients while still missing the worst cancellation cases.

What is needed is a densification statistic that measures total pixel dissatisfaction before signs are allowed to cancel. This statistic must remain an auxiliary signal: the optimizer still steps with the true signed gradient, but the clone-or-split decision should consult a homodirectional summary of the per-pixel sub-gradients.

The method is AbsGS, absolute-gradient densification. Instead of accumulating the signed projected-center gradient, AbsGS accumulates the per-pixel absolute sub-gradient along each image axis before summing. For primitive `i` the homodirectional statistic is `ĝ_{i,x} = Σ_j |∂L_j / ∂μ_{i,x}|` and `ĝ_{i,y} = Σ_j |∂L_j / ∂μ_{i,y}|`, and the densification score is `||ĝ_i||_2`. By the triangle inequality, `||g_i||_2 ≤ ||ĝ_i||_2`; the gap is exactly the cancellation that blinded the original rule. Where per-pixel terms already agree in sign, the two statistics are nearly equal, so the cases the original controller handled well are left undisturbed. Where cancellation occurs, the absolute statistic exposes the large hidden error and triggers splitting.

AbsGS is implemented as a conservative two-channel rule. The original signed average gradient continues to drive clone decisions for small primitives, because small under-covered primitives already produce a coherent signed signal. The absolute average gradient is reserved for split decisions on large primitives, because that is the failure mode caused by cancellation. The split threshold must be raised relative to the clone threshold, since the absolute statistic is systematically larger. Pruning, opacity reset, and the refinement schedule remain unchanged. The only implementation requirement is that the rasterizer must expose per-pixel absolute-gradient accumulations, for example through an `absgrad` flag; taking the absolute value of the already-collapsed signed gradient after backward, `info["means2d"].grad.abs()`, cannot undo cancellation. A practical gsplat-style drop-in variant renders with `absgrad=True` and applies the absolute statistic to both clone and split at one raised threshold; that is faithful to the core idea but less surgical than the two-threshold rule, which keeps the original signed signal for clone and reserves the absolute signal for split.

```python
import torch
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class AbsGS:
    """Absolute-gradient densification for 3D Gaussian Splatting."""
    prune_opa: float = 0.005
    grow_grad2d: float = 2e-4       # signed threshold for clone
    grow_grad2d_abs: float = 4e-4   # absolute threshold for split
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        return {
            "signed_grad2d": None,
            "abs_grad2d": None,
            "count": None,
            "scene_scale": scene_scale,
        }

    def step_pre_backward(self, params, optimizers, state, step, info):
        info["means2d"].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info,
                           duplicate, split, remove, reset_opa):
        if step >= self.refine_stop_iter:
            return

        # Signed gradient for parameter updates and clone decisions.
        signed = info["means2d"].grad.clone()
        # Homodirectional gradient for split decisions; requires rasterizer support.
        if hasattr(info["means2d"], "absgrad"):
            absgrad = info["means2d"].absgrad.clone()
        else:
            absgrad = signed.abs()  # fallback; cannot undo rasterizer cancellation

        # Undo image-space normalization used by gsplat-style gradients.
        signed[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        signed[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        absgrad[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        absgrad[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n = len(list(params.values())[0])
        if state["signed_grad2d"] is None:
            device = signed.device
            state["signed_grad2d"] = torch.zeros(n, device=device)
            state["abs_grad2d"] = torch.zeros(n, device=device)
            state["count"] = torch.zeros(n, device=device)

        visible = (info["radii"] > 0.0).all(dim=-1)
        ids = torch.where(visible)[1]

        state["signed_grad2d"].index_add_(0, ids, signed[visible].norm(dim=-1))
        state["abs_grad2d"].index_add_(0, ids, absgrad[visible].norm(dim=-1))
        state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

        if step > self.refine_start_iter and step % self.refine_every == 0:
            count = state["count"].clamp_min(1)
            signed_avg = state["signed_grad2d"] / count
            abs_avg = state["abs_grad2d"] / count
            scene_scale = state["scene_scale"]

            scale_max = torch.exp(params["scales"]).max(dim=-1).values
            is_small = scale_max <= self.grow_scale3d * scene_scale

            clone_mask = (signed_avg >= self.grow_grad2d) & is_small
            split_mask = (abs_avg >= self.grow_grad2d_abs) & ~is_small

            if clone_mask.any():
                duplicate(params=params, optimizers=optimizers, state=state, mask=clone_mask)

            # Newly cloned primitives are appended; do not split them this step.
            split_mask = torch.cat([
                split_mask,
                torch.zeros(int(clone_mask.sum().item()),
                            dtype=torch.bool, device=split_mask.device),
            ])
            if split_mask.any():
                split(params=params, optimizers=optimizers, state=state, mask=split_mask)

            # Opacity-floor and oversized prune, inherited from the baseline.
            is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
            if step > self.reset_every:
                is_prune = is_prune | (scale_max > self.prune_scale3d * scene_scale)
            if is_prune.any():
                remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

            state["signed_grad2d"].zero_()
            state["abs_grad2d"].zero_()
            state["count"].zero_()

        if step > 0 and step % self.reset_every == 0:
            reset_opa(params=params, optimizers=optimizers, state=state,
                      value=self.prune_opa * 2.0)
```
