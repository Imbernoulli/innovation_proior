**Problem (from step 2).** The max-grad blend fixed *which* Gaussians are selected but left two faults:
the split is still covariance-*sampled* (random offsets, full-shape children that don't tile the parent —
run-to-run variance, residual shape jump), and pruning is blind to Gaussians that overfit a few training
views while holding ordinary opacity (the opacity floor at 0.005 cannot see them). The stump regression
under taming is the overfitting fingerprint: train-view fit up, novel-view PSNR down.

**Key idea (EDC, Deng et al. 2024), two parts.** (1) **Long-Axis Split** for large Gaussians: replace
the covariance-sampled split with a *deterministic* one — split along the longest principal axis, place
children at `μ ± 0.5·s_max·d` (`d = R·e_a`), divide the longest scale by 1.6 (other axes unchanged), set
child opacity `0.6·sigmoid(parent)`. Deterministic placement kills the variance; `0.6·α` is the interior
minimum of the unimodal→bimodal density mismatch and keeps through-ray transmittance near the parent's.
(2) **Recovery-Aware Pruning**: the opacity reset is a poke — view-consistent Gaussians recover fast,
view-conflicted (overfit) ones recover slowly — so at reset+300 prune Gaussians still below 0.05
sigmoid-opacity, catching overfit primitives the opacity floor cannot.

**Why it works.** The long-axis split makes the edit nearly invisible to the renderer (shape tiled,
density and transmittance matched), so the optimizer resumes immediately. Recovery-aware pruning removes
overfit and redundant Gaussians each reset cycle; densification refills high-gradient spots with fresh
view-consistent Gaussians, so the overfit *fraction* falls over training. The two are complementary: the
sharper deterministic split overfits detail more, which the recovery prune then cleans up.

**Scaffold edit / what the harness exposes.** I **keep** clone/split as separate ops (small→`duplicate`,
large→long-axis split) and the Taming max-grad blend wired to them — so this is *not* the split-only EDC
variant, and the longest axis is shrunk by `/1.6` with the **other two axes unchanged** (not `/2` and
0.85, which the split-only form uses to also absorb the clone case). The harness `split` samples
covariance offsets and shrinks all axes, so I write `_long_axis_split` directly via the same low-level
utilities (`_update_param_with_optimizer`, `normalized_quat_to_rotmat`). Recovery prune fires at
`step > reset_every and (step − 300) % reset_every == 0`. Extend `refine_stop_iter` to 22000 (recovery
prune holds the count stable). AbsGS gradient and 0.7/0.3 blend retained.

**What to watch (the bar to clear: taming 30.044 / 27.141 / 33.353 / 27.794).** Deterministic split
edges garden and bonsai up; the decisive test is **stump** (27.794, slipped under taming) and bicycle
(27.141) — if recovery-aware pruning removes the overfit primitives the blend over-added on the sprawling
outdoor scenes, stump should be the largest single gain and rise back above the absgrad floor. Every
public scene over taming, stump recovered.

```python
@dataclass
class CustomStrategy(Strategy):
    """EDC-TamingGS-Abs: Long-Axis Split + Recovery-Aware Pruning + Taming + AbsGS."""

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0005
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 22_000  # extended (recovery prune keeps count stable)
    reset_every: int = 3000
    refine_every: int = 100
    # Taming-3DGS blend
    avg_weight: float = 0.7
    max_weight: float = 0.3
    # EDC: Long-Axis Split
    long_axis_opa_factor: float = 0.6   # child opacity = 0.6 * parent
    long_axis_scale_div: float = 1.6    # longest axis scale shrunk by 1.6
    long_axis_offset: float = 0.5       # child offset = +-0.5 * longest_axis
    # EDC: Recovery-Aware Pruning
    recovery_offset: int = 300          # iters after each opacity reset
    recovery_opa: float = 0.05          # prune below this sigmoid-opacity

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        return {
            "grad2d": None, "count": None, "grad2d_max": None,
            "scene_scale": scene_scale,
        }

    def step_pre_backward(self, params, optimizers, state, step, info):
        info["means2d"].retain_grad()

    def _long_axis_split(self, params, optimizers, state, mask):
        """EDC long-axis split: children placed deterministically along
        longest axis, opacity = 0.6 * sigmoid(parent), longest axis / 1.6.
        """
        from gsplat.strategy.ops import _update_param_with_optimizer
        from gsplat.utils import normalized_quat_to_rotmat
        import torch.nn.functional as F

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]
        if len(sel) == 0:
            return

        scales = torch.exp(params["scales"][sel])                  # [N, 3]
        quats = F.normalize(params["quats"][sel], dim=-1)
        rotmats = normalized_quat_to_rotmat(quats)                 # [N, 3, 3]
        # longest axis index per Gaussian
        max_axis = scales.argmax(dim=-1, keepdim=True)             # [N, 1]
        # local one-hot direction along longest axis
        e_local = torch.zeros_like(scales)
        e_local.scatter_(1, max_axis, 1.0)                          # [N, 3]
        # rotate to world frame
        direction = torch.einsum("nij,nj->ni", rotmats, e_local)    # [N, 3]
        longest = scales.gather(1, max_axis).squeeze(-1)            # [N]
        # offsets +-0.5 * longest along world direction
        offset = self.long_axis_offset * longest.unsqueeze(-1) * direction
        samples = torch.stack([offset, -offset], dim=0)             # [2, N, 3]

        # new scales: longest axis / 1.6, others unchanged
        new_scales = scales.clone()
        new_scales.scatter_(1, max_axis, longest.unsqueeze(-1) / self.long_axis_scale_div)

        # new opacity: 0.6 * alpha, following the EDC long-axis split rule
        new_opa_alpha = (self.long_axis_opa_factor * torch.sigmoid(params["opacities"][sel])).clamp(1e-6, 1.0 - 1e-6)
        new_opa_logit = torch.logit(new_opa_alpha)

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

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step >= self.refine_stop_iter:
            return

        # AbsGS: absolute gradients
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
        # Taming: per-Gaussian max gradient
        state["grad2d_max"].scatter_reduce_(0, gs_ids, grad_norms, reduce="amax", include_self=True)

        # EDC Recovery-Aware Pruning: triggered 300 iters after each opacity reset (after first reset)
        if step > self.reset_every and (step - self.recovery_offset) % self.reset_every == 0:
            opa = torch.sigmoid(params["opacities"].flatten())
            is_recovery_prune = opa < self.recovery_opa
            if is_recovery_prune.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=is_recovery_prune)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            avg_grads = state["grad2d"] / state["count"].clamp_min(1)
            combined = self.avg_weight * avg_grads + self.max_weight * state["grad2d_max"]
            scene_scale = state["scene_scale"]

            is_grad_high = combined > self.grow_grad2d
            scale_max = torch.exp(params["scales"]).max(dim=-1).values
            is_small = scale_max <= self.grow_scale3d * scene_scale

            is_dupli = is_grad_high & is_small
            if is_dupli.sum() > 0:
                duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

            # EDC long-axis split (replaces stochastic split)
            is_split = is_grad_high & ~is_small
            is_split = torch.cat([is_split, torch.zeros(is_dupli.sum(), dtype=torch.bool, device=is_split.device)])
            if is_split.sum() > 0:
                self._long_axis_split(params, optimizers, state, is_split)

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
