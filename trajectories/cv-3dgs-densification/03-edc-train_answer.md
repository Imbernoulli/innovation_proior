The max-grad blend and the consistent split lifted the mean to 29.583 (garden 30.044, bicycle 27.141, bonsai 33.353, stump 27.794), but the per-scene shape was not what I predicted, and that discrepancy is the most informative thing on the table. Against the absgrad floor (29.745 / 27.026 / 33.051 / 27.850), garden jumped +0.30 and bonsai jumped +0.30, bicycle moved only +0.115, and stump *slipped* $-0.056$. I had expected the laggard outdoor scenes to gain most from view-specific selection; instead bonsai — already strongest, indoor, structured — gained as much as garden, and stump went backwards. Two readings. First, the revised-opacity split is doing real work most where the scene is densely structured and consistently viewed, because there the dominant cost was the density perturbation of each split, and removing it let the indoor structure refine cleanly. Second, the stump regression is the fingerprint of view-overfitting: I added a mechanism that densifies more and later, and on the sprawling outdoor scene that benefits least from extra primitives, those extra primitives fit the training views without improving held-out views — train-view fit up, novel-view PSNR down. The blend fixed *which* Gaussians are selected but left two faults untouched: the split *operation* is still covariance-*sampled* (random offsets, full-shape children that do not tile the parent), and pruning is still blind to Gaussians that overfit a few views while holding ordinary opacity.

I propose **EDC**, two parts that attack exactly those faults — a deterministic **long-axis split** and a **recovery-aware prune**. Take the split first. The inherited operation, even with revised opacity, replaces a flagged large Gaussian with two children, each keeping the parent's full ellipsoidal shape (uniformly shrunk by 1.6), placed at $\mu + R\,z$ with $z \sim \mathcal{N}(0, S^2)$. Three things still jump. The positions are *random* — two runs split the same Gaussian differently, and a bad draw puts both children on one side or too far out, pure variance injected into the representation. The shape is wrong — an ellipsoid is not the union of two smaller similar ellipsoids placed off-center, so two displaced copies bulge past the parent's footprint in some directions and leave gaps in others. And the uniform 1.6 shrink shrinks *all three* axes when the subdivision is really happening along *one* direction. A split is morally one-dimensional: I am cutting one blob into two and laying them along a line. The right line is the *longest* principal axis. The parent has axis lengths $s = (s_x, s_y, s_z)$ in its local frame oriented by $R = R(q)$; splitting across a short axis leaves the children almost coincident, while splitting along the longest gives maximum separation for free and is exactly where an elongated Gaussian smears most. So $a = \arg\max_d s_d$, local unit vector $e_a$, world direction $d = R\,e_a$, and the children go to $\mu \pm \text{offset}\cdot d$. The parent's reach along that axis is on the order of $s_{\max}$, so pushing each child out by $\tfrac{1}{2} s_{\max}$ puts the centers one longest-scale apart with each still inside the parent's extent — $\text{offset} = 0.5\,s_{\max}$, deterministic, which directly kills the stump-style run-to-run variance.

The shapes force a narrower choice than the most aggressive form of the idea. The fuller long-axis split also shrinks the two short axes (to $\sim 0.85$), halves the long axis ($/2$), and *replaces both clone and split* with this one operation, on the argument that under backprop a Gaussian's size converges to one equilibrium so the under/over-reconstruction distinction dissolves. But I am filling a contract that keeps `duplicate` and `split` separate, selected by the size test, with a Taming max-grad blend already wired to the clone-small / split-large branch and a clone path that has worked on the small under-covered primitives since the first rung. Collapsing to split-only would mean rebuilding the whole selection branch and discarding that clone path. So I keep the clone/split *distinction* and make only the *large-Gaussian split* the long-axis one. That sets the shrink factors: I divide the longest axis by $1.6$ (matching the inherited split's shrink, so the children tile the long extent without the more aggressive $/2$ the split-only variant uses to also cover the clone case) and leave the other two axes *unchanged* — the conservative choice that fits keeping clone, since clone still rounds out the small-primitive case and I need no extra short-axis trimming. The opacity is the third piece, set by the geometry of two displaced bumps. Before the split the density along the long axis is one bump of opacity $\alpha$; after, two bumps a max-radius apart. At $\alpha_{\text{child}} = \alpha$ the central overlap pushes density above the parent's single-peak profile (too bimodal); at $\alpha_{\text{child}} = 0$ the parent is erased; the unimodal-to-bimodal mismatch has an interior minimum, and for two half-spaced bumps it bottoms out around

$$\alpha_{\text{child}} \approx 0.6\,\alpha.$$

This single factor also nearly preserves the through-ray transmittance — two children at $0.6\alpha$ transmit $(1 - 0.6\alpha)^2 \approx 1 - 1.2\alpha$, far closer to the parent's $1 - \alpha$ than the $\sim 1 - 2\alpha$ of full-opacity children — so the spatial density and the transmittance both stay near the parent's and the edit is nearly invisible to the renderer. I write it in logit space, $\text{logit}(\text{clamp}(0.6\cdot\text{sigmoid}(\text{parent}), \epsilon, 1-\epsilon))$, keeping the $0.6$ multiplicative and *not* additionally routing it through $1 - \sqrt{1-\alpha}$, because the $0.6$ factor is itself the density-matching choice for this spacing. Because the harness `split` samples covariance offsets and shrinks all axes, I cannot call it; I write a `_long_axis_split` helper that does the parameter surgery directly with the same low-level utilities the harness uses — `_update_param_with_optimizer` to swap parameters and reset the Adam state for the new children, and `normalized_quat_to_rotmat` to rotate the local long-axis vector to world frame — selecting the long axis, offsetting by $\pm 0.5\,s_{\max}$, dividing the longest scale by 1.6, setting child opacity to $0.6\alpha$, copying quats and colors, and resizing the running state tensors alongside.

The second fault is pruning, and the stump slip is its symptom. Some Gaussians are harmful on held-out views while looking great on a few training views — they have found a configuration that nails two or three camera angles and produces garbage from the rest. The loss only ever sees training views, so the optimizer keeps them, and at rest they hold perfectly respectable opacity, so the 0.005 floor cannot touch them; there is nothing in their opacity, scale, or position *at rest* that flags them. So I cannot tell an overfit Gaussian from a useful one while it sits still — I have to *poke the system* and watch them respond differently, and I already have a poke built in: the opacity reset every 3000 steps clamps all opacities down and lets the optimizer re-grow the ones it needs. A normal Gaussian contributes positively in every view it appears in — every view votes "more opacity here" — so after the reset its opacity climbs steadily and recovers fast. An overfit Gaussian contributes positively in a few views and *negatively* in the rest — the views it helps push opacity up, the views it hurts push it down — so its recovery is slow and conflicted, the votes fight. The discriminator I could not find at rest is *recovery speed after a reset*. So some iterations after each reset I look at who recovered and cut the laggards: long enough that the normals have visibly recovered (so the laggards are genuinely the conflicted ones, not everyone still climbing), but not so long that the overfit ones have grudgingly crawled back and hidden. A few hundred iterations is where the two populations separate most, so I check at reset $+\,300$, firing when `step > reset_every and (step - 300) % reset_every == 0`. The threshold is well above the 0.005 junk floor — I am catching the ones that *failed to recover*, not enforcing a final floor — so I prune Gaussians still below 0.05 sigmoid-opacity. A recovered normal sails past 0.05; a conflicted one stuck near the reset value gets cut, along with genuinely redundant near-zero Gaussians whose removal frees capacity. This is a slow purification, not a hard constraint: densification refills the high-gradient spot with a fresh Gaussian that may converge view-consistently, so each reset-recover-prune-regrow round lowers the *fraction* of overfit Gaussians. It pairs especially well with the deterministic long-axis split, which is a stronger detail-fitter and therefore *more* prone to overfitting fine regions — exactly the overfitting the recovery prune then cleans up, which is why I add both this rung and not one without the other. Because the prune holds the count stable, I can refine longer before freezing the set, so I extend `refine_stop_iter` to 22000 (from taming's 18000); the grow threshold stays at 0.0005, the blend weights stay 0.7/0.3, and the AbsGS path stays. The decisive test is stump (27.794, the one that slipped) and bicycle (27.141): if the recovery prune removes the overfit primitives the blend over-added on the sprawling outdoor scenes, stump should be the largest single gain and rise back above the absgrad floor.

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
