In 3D Gaussian Splatting, the scene starts as a sparse Structure-from-Motion point cloud and must grow during training. That growth is handled by adaptive density control, a rule that is interleaved with Adam's optimization of every Gaussian's position, scale, rotation, opacity, and color. Because every densification edit changes the set of primitives in the middle of optimization, each edit is a perturbation: the rendered image jumps, the loss jumps, and the optimizer has to spend iterations recovering before it can make further progress. The standard 3DGS rule uses two operations. Clone copies a small Gaussian to the same position with identical parameters; the copy gets no gradient that iteration, so the two only separate if the parent happens to take a large step, and when it does not they stay coincident and receive identical gradients forever. Split replaces a large Gaussian with two children that keep the parent's full shape and opacity and are placed at random offsets drawn from the parent's covariance, which changes the covered shape, injects run-to-run variance, and makes the region too opaque because two stacked children transmit less light than one parent. A separate issue is pruning: some Gaussians overfit a handful of training views while hurting held-out views, yet they retain ordinary opacity, so a simple opacity floor cannot remove them. The problem, then, is not just which Gaussians to densify, but how to densify without disturbing the renderer and how to prune Gaussians whose contribution is inconsistent across views.

The method I propose is EDC, which stands for Efficient Density Control. It replaces the clone-and-split pair with a single deterministic subdivision and adds a recovery-aware pruning rule. The core observation is that under continuous gradient descent a Gaussian's size converges to a single equilibrium that balances under-reconstruction against over-reconstruction, so the under-versus-over distinction that motivated clone versus split collapses. A single split operation suffices. To make that split as invisible to the renderer as possible, EDC splits along the Gaussian's longest principal axis, because that is the direction of maximum smearing and because it gives the two children the largest natural separation. The children are placed deterministically at mu plus and minus one-half of the longest scale times the world-frame long-axis direction, so their centers are one max-radius apart and remain inside the parent's original extent. The long axis is halved so that two displaced bumps tile the parent's elongated profile, while the other two axes are shrunk to 0.85 of their original length so the union's cross-section stays close to the parent's and thin slivers round out over successive splits. The opacity of each child is set to 0.6 times the parent's opacity. That value is near the interior minimum of the mismatch between the parent's unimodal density and the children's bimodal density, and it also keeps through-ray transmittance close to the parent's, so the region does not become suddenly darker at the moment of the split. Deterministic placement removes the sampling variance of the old split, and the shape and opacity choices keep the density field nearly unchanged, which lets the optimizer resume immediately.

The second piece of EDC is recovery-aware pruning. After each opacity reset, view-consistent Gaussians recover opacity monotonically and quickly because every view pushes them up, whereas view-conflicted overfit Gaussians recover slowly because the views they help push them up and the views they hurt push them down. EDC therefore waits 300 iterations after each reset and prunes any Gaussian whose sigmoid opacity is still below 0.05. That threshold is well above the usual junk floor of 0.005, so it catches overfit primitives that ordinary pruning misses. Removing redundant Gaussians reallocates capacity, while removing overfit Gaussians and letting densification refill their spots with fresh primitives slowly purifies the cloud and improves generalization without changing the resource budget. The periodic opacity reset is kept because it both rebalances contribution away from near-saturated Gaussians and provides the poke that makes the recovery-rate signature visible.

```python
from dataclasses import dataclass
from typing import Any, Dict
import torch
import torch.nn.functional as F

from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import remove, reset_opa, _update_param_with_optimizer
from gsplat.utils import normalized_quat_to_rotmat


@dataclass
class EDCStrategy(Strategy):
    """Efficient Density Control: Long-Axis Split + Recovery-Aware Pruning."""

    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    grow_grad2d: float = 0.0002      # raise to 0.0004 with AbsGS-style gradients
    prune_opa: float = 0.005         # standard low-opacity prune floor
    absgrad: bool = False
    key_for_gradient: str = "means2d"
    # Long-Axis Split constants
    las_opa_factor: float = 0.6      # child opacity = 0.6 * parent
    las_short_factor: float = 0.85   # non-longest axes shrunk to 0.85
    las_long_div: float = 2.0        # longest axis halved
    # Recovery-Aware Pruning
    recovery_offset: int = 300       # iters after each reset
    recovery_opa: float = 0.05       # prune Gaussians not recovered above this

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        return {"grad2d": None, "count": None, "scene_scale": scene_scale}

    def step_pre_backward(self, params, optimizers, state, step, info):
        info[self.key_for_gradient].retain_grad()

    @torch.no_grad()
    def _long_axis_split(self, params, optimizers, state, mask, scene=None):
        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]
        if len(sel) == 0:
            return

        scales = torch.exp(params["scales"][sel])
        quats = F.normalize(params["quats"][sel], dim=-1)
        rotmats = normalized_quat_to_rotmat(quats)

        a = scales.argmax(dim=-1, keepdim=True)
        e_local = torch.zeros_like(scales).scatter_(1, a, 1.0)
        direction = torch.einsum("sij,sj->si", rotmats, e_local)
        s_max = scales.gather(1, a).squeeze(-1)

        offset = 0.5 * s_max.unsqueeze(-1) * direction
        samples = torch.stack([offset, -offset], dim=0)

        new_scales = scales * self.las_short_factor
        new_scales.scatter_(1, a, s_max.unsqueeze(-1) / self.las_long_div)

        a_child = (self.las_opa_factor *
                   torch.sigmoid(params["opacities"][sel])).clamp(1e-6, 1 - 1e-6)
        new_opa_logit = torch.logit(a_child)

        def param_fn(name, p):
            repeats = [2] + [1] * (p.dim() - 1)
            if name == "means":
                p_split = (p[sel] + samples).reshape(-1, 3)
            elif name == "scales":
                p_split = torch.log(new_scales).repeat(2, 1)
            elif name == "opacities":
                p_split = new_opa_logit.repeat(repeats)
            else:
                p_split = p[sel].repeat(repeats)
            return torch.nn.Parameter(
                torch.cat([p[rest], p_split]), requires_grad=p.requires_grad
            )

        def optimizer_fn(key, v):
            v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=v.device)
            return torch.cat([v[rest], v_split])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                repeats = [2] + [1] * (v.dim() - 1)
                state[k] = torch.cat((v[rest], v[sel].repeat(repeats)))
        if scene is not None:
            scene.on_split(sel, rest)

    def step_post_backward(self, params, optimizers, state, step, info,
                           packed=False, scene=None):
        if step >= self.refine_stop_iter:
            return

        key = self.key_for_gradient
        grads = (info[key].absgrad if self.absgrad else info[key].grad).clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n = len(params["means"])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n, device=grads.device)
        if packed:
            ids = info["gaussian_ids"]
            grad_values = grads
        else:
            vis = (info["radii"] > 0.0).all(dim=-1)
            ids = torch.where(vis)[1]
            grad_values = grads[vis]
        state["grad2d"].index_add_(0, ids, grad_values.norm(dim=-1))
        state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

        if step > self.reset_every and (step - self.recovery_offset) % self.reset_every == 0:
            not_recovered = torch.sigmoid(params["opacities"].flatten()) < self.recovery_opa
            if not_recovered.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state,
                       mask=not_recovered, scene=scene)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            avg_grads = state["grad2d"] / state["count"].clamp_min(1)
            is_high = avg_grads > self.grow_grad2d
            if is_high.sum() > 0:
                self._long_axis_split(params, optimizers, state, is_high, scene=scene)
            is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
            if is_prune.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state,
                       mask=is_prune, scene=scene)
            state["grad2d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params=params, optimizers=optimizers, state=state, value=0.01)
```
