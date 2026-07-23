The densification stage of 3D Gaussian Splatting starts from a sparse Structure-from-Motion point cloud and must grow the primitive set to match the scene. The inherited controller accumulates the magnitude of the signed loss gradient with respect to each primitive's projected 2D center, averaged over the views in which the primitive appears. When this average exceeds a threshold, the primitive is cloned if it is small or split if it is large. The rule works naturally for small under-covered primitives: their footprints cover only a few pixels, so the per-pixel sub-gradients usually agree in sign and produce a coherent net pull. It fails for large primitives that cover fine texture, such as grass, foliage, or distant detail. Across a large footprint, the per-pixel sub-gradients have opposite signs: the L1 residual flips between bright and dark pixels, alpha-compositing effects can flip signs depending on depth and color ordering, and the projected Gaussian derivative pulls in opposite directions on opposite sides of the center. After summation these terms cancel, so the net gradient stays small even though many pixels are poorly represented. Lowering the threshold does not solve the problem, because it would also densify primitives with genuinely small gradients while still missing the worst cancellation cases.

What is needed is a densification statistic that measures total pixel dissatisfaction before signs are allowed to cancel. This statistic must remain an auxiliary signal: the optimizer still steps with the true signed gradient, but the clone-or-split decision should consult a homodirectional summary of the per-pixel sub-gradients.

The method is AbsGS, absolute-gradient densification. Instead of accumulating the signed projected-center gradient, AbsGS accumulates the per-pixel absolute sub-gradient along each image axis before summing. For primitive `i` the homodirectional statistic is `ĝ_{i,x} = Σ_j |∂L_j / ∂μ_{i,x}|` and `ĝ_{i,y} = Σ_j |∂L_j / ∂μ_{i,y}|`, and the densification score is `||ĝ_i||_2`. By the triangle inequality, `||g_i||_2 ≤ ||ĝ_i||_2`; the gap is exactly the cancellation that blinded the original rule. Where per-pixel terms already agree in sign, the two statistics are nearly equal, so the cases the original controller handled well are left undisturbed. Where cancellation occurs, the absolute statistic exposes the large hidden error and triggers splitting.

AbsGS is implemented as a conservative two-channel rule. The original signed average gradient continues to drive clone decisions for small primitives, because small under-covered primitives already produce a coherent signed signal. The absolute average gradient is reserved for split decisions on large primitives, because that is the failure mode caused by cancellation. The split threshold must be raised relative to the clone threshold, since the absolute statistic is systematically larger. Pruning, opacity reset, and the refinement schedule remain unchanged. The only implementation requirement is that the rasterizer must expose per-pixel absolute-gradient accumulations, for example through an `absgrad` flag; taking the absolute value of the already-collapsed signed gradient after backward, `info["means2d"].grad.abs()`, cannot undo cancellation. A practical gsplat-style drop-in variant renders with `absgrad=True` and applies the absolute statistic to both clone and split at one raised threshold; that is faithful to the core idea but less surgical than the two-threshold rule, which keeps the original signed signal for clone and reserves the absolute signal for split.

```python
@torch.no_grad()
def update_density_state(params, state, info, key="means2d"):
    signed = info[key].grad.clone()
    absgrad = info[key].absgrad.clone()  # per-pixel |dL/dmu_x|, |dL/dmu_y| summed in backward

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


@torch.no_grad()
def grow_abs_gs(params, optimizers, state, duplicate, split,
                tau_clone=0.0002, tau_split=0.0004, grow_scale3d=0.01):
    count = state["count"].clamp_min(1)
    signed_avg = state["signed_grad2d"] / count
    abs_avg = state["abs_grad2d"] / count

    scale_max = torch.exp(params["scales"]).max(dim=-1).values
    is_small = scale_max <= grow_scale3d * state["scene_scale"]

    clone_mask = (signed_avg >= tau_clone) & is_small
    split_mask = (abs_avg >= tau_split) & ~is_small

    n_clone = int(clone_mask.sum().item())
    if n_clone:
        duplicate(params=params, optimizers=optimizers, state=state, mask=clone_mask)

    # Newly cloned primitives are appended and should not be split in the same refinement.
    split_mask = torch.cat([
        split_mask,
        torch.zeros(n_clone, dtype=torch.bool, device=split_mask.device),
    ])
    if split_mask.any():
        split(params=params, optimizers=optimizers, state=state, mask=split_mask)
```
