# AbsGS: homodirectional view-space gradient for densification

AbsGS fixes the over-reconstruction failure in 3D Gaussian Splatting density control. The inherited
controller uses the signed, summed projected-center gradient

```text
g_i = dL/dmu_i,          ||g_i||_2 > tau_p
```

as the densification signal. For a large Gaussian, each component is a signed sum over covered pixels:

```text
g_{i,x} = sum_j dL_j/dmu_{i,x},        g_{i,y} = sum_j dL_j/dmu_{i,y}.
```

Per-pixel terms can have opposite signs, so large terms cancel and the primitive is not split. AbsGS
uses a homodirectional, per-axis absolute accumulation before the pixel sum:

```text
hat_g_{i,x} = sum_j |dL_j/dmu_{i,x}|,
hat_g_{i,y} = sum_j |dL_j/dmu_{i,y}|,
score_i = ||hat_g_i||_2.
```

By the triangle inequality, `||g_i||_2 <= ||hat_g_i||_2`; the gap is exactly the cancellation signal.
`hat_g_i` is an auxiliary density-control statistic, not the gradient used for parameter updates.

## Faithful Rule

The canonical implementation keeps the original signed statistic for clone and uses the
homodirectional statistic for split:

```text
signed_avg_i = average_k ||dL^k/dmu_i^k||_2
abs_avg_i    = average_k ||hat_g_i^k||_2

clone if signed_avg_i >= tau_clone and scale_i <= tau_S
split if abs_avg_i    >= tau_split and scale_i >  tau_S
```

with `tau_clone` inherited around `0.0002` and `tau_split` raised, commonly around `0.0004` in the
official-style code. The higher split threshold is required because the absolute statistic is
systematically larger.

## Code Contract

The rasterizer must expose per-pixel absolute-gradient accumulation. Do not replace it with
`info["means2d"].grad.abs()`: that is only the absolute value of the already-collapsed signed gradient
and cannot undo cancellation.

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

`gsplat` exposes a related drop-in variant: call `rasterization(..., absgrad=True)` and use
`DefaultStrategy(absgrad=True, grow_grad2d=0.0008)`. That implementation uses `.absgrad` for both
duplicate and split, so it is a faithful implementation of the homodirectional statistic but less
surgical than the original split-only rule.
