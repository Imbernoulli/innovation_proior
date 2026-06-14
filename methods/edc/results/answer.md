# EDC (Efficient Density Control), distilled

EDC is a drop-in adaptive-density-control rule for 3D Gaussian Splatting that makes each
densification edit minimally disturb the rendered scene and that prunes Gaussians which have
overfit specific views. It has two parts: **Long-Axis Split** (a single deterministic split
operation that replaces 3DGS's clone and covariance-sampled split) and **Recovery-Aware
Pruning** (cut Gaussians that fail to recover opacity after a reset). It is plug-and-play on
top of 3DGS, TamingGS, MiniGS, and the absolute-gradient (AbsGS) densification criterion.

## Problem it solves

In 3DGS, density control is interleaved with the gradient optimization of the Gaussians, so
every densification op is a perturbation the optimizer must spend iterations recovering from.
The standard operations perturb badly: *clone* makes near-coincident duplicates that receive
near-identical gradients and can never be optimized apart; *split* gives two children the
parent's full shape and opacity at random sampled positions, so the covered shape and the
through-ray density jump away from the optimized geometry. Separately, some Gaussians overfit
a few training views while harming held-out views yet keep ordinary opacity, so opacity-floor
pruning cannot remove them.

## Key idea

1. **Make the split invisible to the renderer.** Subdivide one Gaussian into two so that the
   union's shape, position, and density profile reproduce the parent as closely as possible —
   then the optimizer barely notices the edit and resumes immediately. Since Gaussian size
   converges to a single equilibrium under backprop, the under/over-reconstruction distinction
   collapses, so a single split operation (no clone) suffices.

2. **Reveal overfit Gaussians by a dynamical signature.** Overfit and useful Gaussians look
   identical at rest, but respond differently to an opacity reset: a view-consistent Gaussian
   recovers opacity monotonically and fast; a view-conflicted (overfit) one oscillates and
   recovers slowly. Prune the slow recoverers.

## Long-Axis Split (replaces clone + split)

For each Gaussian flagged for densification (mean `mu`, true scales `s`, rotation `R`, opacity
`alpha`):

- **Axis.** Split along the longest axis `a = argmax_d s_d`; world direction `d = R e_a`.
- **Position.** Two children at `mu +- (1/2) s_max d` — centers one max-radius apart, both
  inside the parent's extent, deterministic (no covariance sampling, so no run-to-run variance
  and no clustering).
- **Shape.** Longest-axis scale halved (`s_a -> s_a / 2`) so two displaced half-width bumps
  tile the parent's long profile; the other two axes shrunk to `0.85x` so the union's
  cross-section matches the parent's and elongated Gaussians round out.
- **Opacity.** Each child `0.6 * alpha`. This is the interior minimum of the mismatch between
  the parent's unimodal density and the children's bimodal density, and it simultaneously keeps
  the through-ray transmittance close to the parent's (`(1 - 0.6 alpha)^2 ≈ 1 - 1.2 alpha` vs
  the parent's `1 - alpha`, far better than two full-opacity children's `(1-alpha)^2`).
- `0.85` and `0.6` are extrema found by sweeping; `+-0.05` gives near-identical results.
- **Split only** (no clone): the converged-size observation makes one operation enough.

## Recovery-Aware Pruning

Opacity is reset every 3000 iterations to 0.01 (kept from 3DGS — it also rebalances rendering
contribution away from near-saturated Gaussians). **300 iterations after each reset**, prune
every Gaussian whose sigmoid-opacity is still **below 0.05**.

- Normal Gaussians (consistent positive contribution across views) have recovered above 0.05;
  overfit Gaussians (conflicting contributions) are still stuck low. The threshold (0.05) sits
  above the standard junk floor (0.005), so this catches Gaussians the ordinary prune misses.
- Two groups are removed: redundant Gaussians (always near-zero opacity — reallocates capacity,
  helps train and test) and overfit Gaussians (opacity was respectable but did not recover —
  may regrow, but regrowths are less likely to overfit; over many rounds the overfit fraction
  falls). The overfit case changes generalization without changing resource allocation, so it
  lifts test quality while leaving training quality roughly flat.
- It is complementary to Long-Axis Split: the sharper deterministic split fits detail harder
  and thus overfits more, which recovery-aware pruning then cleans up.

## Final form (densification strategy)

Faithful to the gsplat `Strategy` API and the 3DGS/gsplat edit primitives (`remove`,
`reset_opa`, `_update_param_with_optimizer`, `normalized_quat_to_rotmat`). Parameters are
stored as log-scales and logit-opacities.

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
    """Efficient Density Control: Long-Axis Split (split-only) + Recovery-Aware Pruning."""

    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    grow_grad2d: float = 0.0002      # raise (e.g. 0.0004) when using AbsGS homodirectional grads
    prune_opa: float = 0.005         # standard low-opacity prune floor
    absgrad: bool = False
    key_for_gradient: str = "means2d"
    # Long-Axis Split constants (sweep extrema; +-0.05 robust)
    las_opa_factor: float = 0.6      # child opacity = 0.6 * parent
    las_short_factor: float = 0.85   # non-longest axes -> 0.85
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

        scales = torch.exp(params["scales"][sel])                  # [S,3]
        quats = F.normalize(params["quats"][sel], dim=-1)
        rotmats = normalized_quat_to_rotmat(quats)                 # [S,3,3]

        a = scales.argmax(dim=-1, keepdim=True)                    # longest-axis index
        e_local = torch.zeros_like(scales).scatter_(1, a, 1.0)
        direction = torch.einsum("sij,sj->si", rotmats, e_local)   # world dir d = R e_a
        s_max = scales.gather(1, a).squeeze(-1)                    # [S]

        offset = 0.5 * s_max.unsqueeze(-1) * direction             # +- (1/2) s_max d
        samples = torch.stack([offset, -offset], dim=0)            # [2,S,3]

        new_scales = scales * self.las_short_factor                # other axes -> 0.85
        new_scales.scatter_(1, a, s_max.unsqueeze(-1) / self.las_long_div)  # long axis / 2

        a_child = (self.las_opa_factor * torch.sigmoid(params["opacities"][sel])).clamp(1e-6, 1 - 1e-6)
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
            return torch.nn.Parameter(torch.cat([p[rest], p_split]), requires_grad=p.requires_grad)

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

    def step_post_backward(self, params, optimizers, state, step, info, packed=False, scene=None):
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

        # Recovery-Aware Pruning: 300 iters after each reset
        if step > self.reset_every and (step - self.recovery_offset) % self.reset_every == 0:
            not_recovered = torch.sigmoid(params["opacities"].flatten()) < self.recovery_opa
            if not_recovered.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=not_recovered, scene=scene)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            avg_grads = state["grad2d"] / state["count"].clamp_min(1)
            is_high = avg_grads > self.grow_grad2d
            if is_high.sum() > 0:
                self._long_axis_split(params, optimizers, state, is_high, scene=scene)   # split-only
            is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
            if is_prune.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=is_prune, scene=scene)
            state["grad2d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params=params, optimizers=optimizers, state=state, value=0.01)
```

## Relation to prior density control

- **3DGS clone + split** are both replaced by a single deterministic Long-Axis Split; the
  clone-coincidence failure and the split shape/opacity/sampling discontinuities are removed.
- **Homodirectional / absolute gradient (AbsGS, GOF)** is orthogonal — it fixes *which*
  Gaussians are selected (no sign cancellation across pixels); EDC fixes *how* they are split.
  EDC stacks on top of it (raise the gradient threshold accordingly).
- **Budgeted score densification (TamingGS)** is orthogonal — it fixes *how many* and *which*;
  EDC stacks on top, supplying the operation.
- **Revised split opacity `1 - sqrt(1 - alpha)` (RevisingGS)** addresses the two-stacked-child
  transmittance only; EDC's `0.6 * alpha` is derived to match the full spatial density
  distribution and incidentally keeps transmittance close too.
- **Longest-axis placement (VCR-GauS)** shares the deterministic long-axis idea but keeps the
  parent's full shape and opacity; EDC additionally halves the long axis, shrinks the short
  axes to 0.85, and reduces opacity to 0.6 so the whole density profile is preserved.
- **Opacity-based / importance pruning (LightGaussian, EAGLES, MiniGS)** aim to shrink the
  Gaussian count; Recovery-Aware Pruning instead targets overfit Gaussians via recovery rate,
  improving generalization rather than just compressing.
