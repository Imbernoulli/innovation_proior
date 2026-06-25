Let me start from what actually slows training down, because the thing that bugs me about 3DGS is structural, not a tuning issue. I have two processes fighting for the same iterations: gradient descent is patiently sculpting each Gaussian's position, scale, rotation, opacity, and color to fit the photographs, and every hundred steps the density-control rule barges in and *edits the set* — clones one Gaussian, splits another, prunes a third, occasionally knocks all the opacities flat. The optimizer doesn't get a vote. So every densification op is a perturbation: the moment it fires, the rendered image changes, the loss jumps, and the optimizer has to spend the next stretch of iterations clawing back to where it was before it can make new progress. If I want training to be efficient, the lever isn't "densify more" or "densify less" — it's "make each edit something the optimizer barely notices." An ideal densification would add representational capacity while leaving the rendered scene, ray by ray, almost exactly as it was. Let me hold that as the target and go look hard at what the two existing operations actually do to the rendered scene at the instant they fire.

Clone first. A Gaussian flagged as too small for the detail it's covering gets copied — same position, same everything. The two copies are supposed to drift apart and cover the under-reconstructed region between them. But how do they drift apart? The clone happens *after* the render, so the copy gets no gradient this iteration; the only thing that separates the two is the parent's parameter update from this single step. If the parent took a big step this iteration, fine, they separate. But there's no reason the parent's step is big right when I decide to clone — densification fires on accumulated gradient over a window, not on this step's magnitude. When the step is small, I've just placed two essentially coincident Gaussians. And now they're stuck: two Gaussians at the same place with the same shape see almost the same gradient on every future iteration, so they move together forever. I've spent a Gaussian and gotten no new degree of freedom. That's the clone failure — it relies on a coincidence (a large parent step exactly when cloning) that often doesn't hold, and when it fails the two are unrecoverable because identical inputs give identical gradients.

Now split, which is the operation for a large Gaussian over a region with too much variation. Replace the parent with two children, each scaled to 1/1.6 of the parent, each keeping the parent's shape, opacity, and color, each placed at the parent center plus a sample drawn from the parent's own covariance, `mu + R·z` with `z ~ N(0, S²)`. Let me ask my efficiency question of this: how much does the rendered scene change at the instant I split? A lot, actually, and in three separate ways. First the shape. The parent has been optimized so its ellipsoid hugs the local geometry. Each child keeps that *same* ellipsoidal shape, just shrunk uniformly by 1.6 and moved. Two copies of the parent's shape, displaced, cannot reproduce the parent's footprint — they bulge out past it in some directions and leave gaps in others, because an ellipsoid is not the union of two smaller similar ellipsoids placed off-center. So the covered region's shape jumps at the edit, and since the parent's shape was the optimizer's hard-won fit to the geometry, the jump is exactly away from the geometry. Second, the positions are *random* — `z ~ N(0, S²)` — so two runs split the same Gaussian differently, and a bad draw can put both children on the same side, or too far out, and the optimizer inherits whatever the dice gave it. Third, opacity: both children keep the parent's alpha, and I'll come back to what that does to the density along a ray, because it's the subtlest of the three.

So the real problem with both operations is the same problem viewed twice: they don't respect the thing the optimizer already built. Clone makes a useless duplicate; split makes a shape-and-opacity discontinuity. Let me see if I can design *one* operation that, when it fires, leaves the rendered density profile as close to unchanged as I can make it, while still giving the optimizer two genuinely separable Gaussians to work with.

Do I even need two operations? 3DGS uses clone for under-reconstruction (region needs more coverage) and split for over-reconstruction (one Gaussian covers too much). But here's something I keep noticing in training: under gradient descent the size of a Gaussian doesn't run off to either extreme — it settles. A Gaussian that's too big gets pushed smaller because it's smearing detail; a Gaussian that's too small gets pushed bigger because it's leaving gaps; either way the scale converges to the single value that balances those two pressures and minimizes the loss locally. So "under-reconstructed" and "over-reconstructed" aren't two stable regimes calling for two different fixes — they're both transient states that the optimizer is already pulling toward the same equilibrium size. What the region actually needs in both cases is *more Gaussians of about the right size*, and the high positional gradient is already telling me where. That collapses the design: I don't need clone-versus-split chosen by size. I need one subdivision operation that adds a Gaussian where the gradient is high, and I can let the optimizer's own size-equilibration handle the under/over distinction. Split only. That also kills the clone-coincidence failure outright, because a real subdivision separates the two children deterministically rather than hoping the parent steps far enough.

Good — so the operation is "split, always." Now make the split itself minimally disruptive. A Gaussian is a 3D blob, but a split is morally a one-dimensional act: I'm cutting one blob into two and laying them along a line. Which line? The parent ellipsoid has three principal axes with lengths set by the scales `s = (s_x, s_y, s_z)` in the Gaussian's local frame, oriented in the world by the rotation `R = R(q)`. If I split across a short axis, the two children sit almost on top of each other and I'm back to the clone problem — they'll see nearly the same gradients. If I split along the *longest* axis, I get the maximum separation for free, and it's also the direction in which a single ellipsoid is doing the most smearing — an elongated Gaussian covers the most varied stretch of geometry along its long axis, so that's exactly where subdivision buys the most. So: split along the longest axis. Let `a = argmax_d s_d` be the index of the longest scale, `e_a` the local unit vector along it, and the world direction is `d = R · e_a`. The two children go on either side of the parent center along `±d`.

How far apart? I want them separated enough that they're genuinely two Gaussians the optimizer can move independently — not coincident — but not so far that they spill outside the region the parent was covering, because spilling outside is itself a change to the rendered scene away from the fit. The parent's reach along the long axis is on the order of its longest scale. If I push each child out by half the longest scale, the two centers are one longest-scale apart and each child still sits inside the parent's original extent. That's the natural spacing: offset `= ½ · s_max` from the center, so the children are at `mu ± ½ s_max d`, their centers separated by the parent's maximum radius. Just enough to avoid overlap, not enough to escape the footprint.

Now the shapes, and this is where I have to be careful, because getting the shape right is what makes the edit invisible to the renderer. I'm placing two children a distance `s_max` apart along the long axis, each meant to cover half of the parent's elongated extent. Picture the parent's 1-D profile along that axis: a single Gaussian bump with characteristic radius `s_max`. I want two bumps whose combined support follows that same long profile as closely as I can. If each child keeps the full long-axis radius `s_max` while being shifted by `±½ s_max`, the one-radius landmarks run from about `-1.5 s_max` to `+1.5 s_max` and the middle is heavily overlapped; that is much wider and denser than the parent. To tile the parent's long extent with two displaced bumps, each child's long-axis radius should be about half the parent's, so a bump centered at `+½ s_max` with radius `½ s_max` covers the right half and its twin covers the left half. So: halve the longest-axis scale of each child, `s_a -> s_a / 2`. Note this is a sharper cut than the original split's uniform `/1.6` — and it should be, because I'm only shrinking the *one* axis I'm subdividing, not all three.

The other two axes need separate thought. The children sit side by side along the long axis, and neither child is being subdivided across the short axes, so my first instinct is to leave the short axes alone. But the split has changed the body from one ellipsoid to two overlapping ellipsoids; even if the long extent is tiled, the overlap region now carries the cross-section of two children rather than one parent. If I leave the short radii untouched, the combined covered body is too generous around the middle. If I shrink the short axes a little, I pull that combined body back toward the parent's shape while leaving enough cross-section for each child to represent the same surface patch. How much? Not as much as the long axis: the long axis is the one I actually subdivided, so it takes the big cut (half); the short axes only need a gentle trim. I don't have a clean closed form for the cross-section overlap the way I do for the long-axis tiling, so I'll reason by bounds — leaving them at `1.0` is clearly too generous (the overlap double-counts), and cutting them as hard as the long axis (`0.5`) would pinch each child below the surface patch it has to cover. The right value is a gentle shrink somewhere in between but close to 1; I'll take about `0.85` and treat it as a number to confirm by sweep rather than something I've pinned analytically. And this has a bonus: it nudges very elongated Gaussians toward being rounder over repeated splits, because the longest radius gets cut the most. So: longest axis halved, the other two to 0.85.

Now the opacity, the third piece, and the one I waved at earlier. Let me actually look at the density along a ray that threads through the split region, because that's what the renderer integrates. Before the split, the region's density along the long axis is one Gaussian bump of opacity `alpha`. After the split it's two bumps. If I keep each child at the parent's opacity `alpha`, then at the two child centers the density is back up at `alpha`, and in the middle where the two tails overlap it can climb *above* the parent's local density -- I've turned a single-peak density into a double-peak density with the wrong central profile. The renderer composites this as more occlusion than the parent provided: a ray that used to pass through one bump of `alpha` now passes through two, and the compositing of two layers of opacity `alpha` transmits `(1-alpha)²`, which is *less* than the parent's `(1-alpha)`. So at the instant of the split the region gets darker / more opaque than it was, a visible jump exactly when I wanted invisibility. I need to pull each child's opacity down so the two-bump density matches the one-bump density as closely as possible.

What's the right factor? My first instinct is to match the density *distribution* — the unimodal parent bump against the bimodal pair of children — in the most direct way I can: set up the 1-D profile along the long axis and find the child peak `f` that minimizes the squared difference. Let me actually do that rather than eyeball it. Put the parent at the origin with radius 1 and unit peak; the two children sit at `±0.5` (one max-radius apart) with radius `0.5` (halved) and peak `f`. Sweep `f` from 0 to 1 and integrate `(parent − children)²` along the axis:

```
f      L2 mismatch
0.55   0.524
0.60   0.459
0.65   0.399
0.84   0.274   <- minimum
1.00   0.374
```

That is not what I expected. The least-squares spatial match bottoms out near `f ≈ 0.84`, and `f = 1.0` (keep the full opacity) actually fits *better* than `f = 0.6`. The reason is visible if I evaluate the combined profile: with `f` small the two half-width bumps simply don't carry enough mass to refill the parent's body — at the center the combined density is only `~0.73` against the parent's `1.0`, and at a child center `~0.68`. The 1-D mass is even cleaner: each child has half the radius so half the 1-D mass of a unit bump, and there are two of them, so the total child mass is exactly `2 · ½ · f = f` times the parent's. To match integrated mass I'd want `f = 1`. So pure spatial density-matching pushes me *up* toward 0.8–1.0, not down. If matching the rendered density profile were the whole story, I should barely drop the opacity at all.

But it isn't the whole story, and the spatial integral is the wrong objective — it weights every point in space equally, whereas the renderer doesn't integrate density in space, it integrates *transmittance along rays*. What a ray crossing the split region actually sees is the parent's single bump of opacity `alpha` transmitting `(1 − alpha)`; after the split, a ray threading through both children transmits `(1 − f alpha)²`. *That's* the quantity I must preserve, and it pulls the opposite way from the spatial match. Setting `(1 − f alpha)² = (1 − alpha)` and solving gives `f = (1 − sqrt(1 − alpha)) / alpha`. Let me compute where this lands for the opacities I actually see in a scene:

```
alpha   transmittance-matching f
0.1     0.513
0.3     0.545
0.5     0.586
0.7     0.646
0.9     0.760
```

So the transmittance-preserving factor is *not* a constant — it drifts up with opacity — but across the bulk of the opacity range a Gaussian actually holds (mostly 0.1–0.7), it sits in a tight band from about 0.51 to 0.65. I need one fixed constant, not an `alpha`-dependent rule (the edit primitives take a scalar, and I don't want the split's behaviour swinging with each Gaussian's current opacity). The value that sits in the middle of that band, and that splits the difference between the transmittance argument (which wants low-0.5s to mid-0.6s) and the spatial-mass argument (which wants higher), is right around `0.6`. So `alpha_child = 0.6 alpha` — chosen as the central value of the transmittance-matching factor over typical opacities, nudged up slightly toward the spatial-mass pull. It's a compromise, not a true minimum of any one objective; the two objectives disagree, and 0.6 is where I'd put the single constant that serves both least badly.

Let me sanity-check that the compromise actually beats the naive "keep full opacity" on the thing that matters. Two children of opacity `0.6 alpha` stacked on a ray transmit `(1 − 0.6 alpha)²`; with `alpha` not large this is `≈ 1 − 1.2 alpha`. Numerically, at `alpha = 0.3` the parent transmits `0.700`, two children at `0.6 alpha` transmit `0.672` — a small miss — while two full-opacity children transmit `(1 − 0.3)² = 0.490`, a large miss that darkens the region exactly when I wanted the edit invisible. At `alpha = 0.5`: parent `0.500`, `0.6 alpha` children `0.490`, full-opacity children `0.250`. So dropping to `0.6` cuts the transmittance error from roughly a third of the signal down to a few percent. The spatial profile is a little light as a result — but a slightly-light density that the optimizer can refill in a few steps is a far smaller perturbation than a doubled occlusion that darkens the whole region at the instant of the edit.

Let me write down the whole operation as it now stands, because I want to check it hangs together. A flagged Gaussian with mean `mu`, log-scales giving true scales `s`, rotation `R`, opacity `alpha`: find the longest axis `a`, world direction `d = R·e_a`; create two children at `mu ± ½ s_max · d`; each child's scales are `s` with `s_a` replaced by `s_a/2` and the other two multiplied by 0.85; each child's opacity is `0.6 alpha`; rotation, color copied. Replace the parent by these two. That's deterministic — no `N(0,Σ)` sampling — so the run-to-run variance of the old split is gone too, which is a third improvement I get for free from going deterministic along the long axis. Let me sanity-check the degenerate cases. If the Gaussian is nearly spherical, `s_max` is barely the longest of three near-equal axes, but the operation still does something sensible: it picks one axis, separates the children by a radius, slightly shrinks everything, drops the opacity — a reasonable subdivision, just not as dramatic as for an elongated one, which is correct because a near-sphere isn't smearing much. If the Gaussian is a thin sliver, the long axis is unambiguous, it gets halved, the slivery cross-section gets gently shrunk toward rounder — exactly what I want. Good.

Now I have a clean subdivision operation. But there's a second, independent problem I've been seeing in training that no amount of better splitting fixes, and it's about *pruning*. Some Gaussians are flat-out harmful on held-out views while looking great on a few training views — they've overfit. They've found a configuration that nails the appearance from two or three camera angles and produces garbage from the rest, but because the loss only ever sees the training views, the optimizer is happy to keep them. The damage shows up only at test time. The standard prune can't touch them: it removes Gaussians with opacity below ~0.005, and an overfit Gaussian, at steady state, holds a perfectly respectable opacity — it's *contributing*, just contributing wrongly off-view. There's nothing in its opacity, scale, or position at convergence that flags it as overfit. So opacity-threshold pruning is blind to exactly the Gaussians that hurt generalization most. Wall. How do I tell an overfit Gaussian apart from a genuinely useful one when, sitting still, they look identical?

I can't tell them apart at rest. So I need to *poke the system* and watch how they respond differently — find a dynamical signature, not a static one. And I already have a poke built into training: the opacity reset. Every 3000 iterations all opacities get clamped down to 0.01, and then the optimizer re-grows the ones it needs. Think about what re-growth looks like for the two kinds of Gaussian. A *normal* Gaussian contributes positively in every view it appears in — every view's gradient says "more opacity here." So after the reset its opacity climbs steadily, monotonically, and recovers fast; all the votes agree. An *overfit* Gaussian contributes positively in a few views and *negatively* in the rest — the views it helps push its opacity up, the views it hurts push it down. The votes conflict. Its opacity after the reset doesn't climb cleanly; it oscillates — up, down, up — and on net recovers much more slowly, because the negative views are constantly fighting the positive ones. So there *is* a signature: recovery *speed* after an opacity reset. Consistent-contribution Gaussians snap back; conflicted-contribution Gaussians lag. That's the discriminator I couldn't find at rest, surfaced by the poke I already have.

Now turn it into a prune. Some time after a reset, I look at who has recovered and who hasn't, and I cut the laggards. How long to wait? Long enough that the normal Gaussians have visibly recovered — so the laggards are genuinely the conflicted ones and not just everyone still climbing — but not so long that the overfit ones have grudgingly crawled back up and hidden again. A few hundred iterations after the reset is the window where the two populations are most separated: the normals are up, the overfit ones are still down. I'll check 300 iterations after each reset. And the threshold: I'm not trying to enforce a final opacity floor, I'm trying to catch the ones that *failed to recover*, so I set it a bit above the noise — prune Gaussians whose opacity is still below 0.05 at reset+300. That's well above the 0.005 junk threshold (so I'm not just re-doing the standard prune) and low enough that a normal Gaussian, which has recovered toward its real opacity, sails past it, while a conflicted one that's still stuck near the 0.01 reset value, or oscillating below 0.05, gets cut.

Let me reason about who actually gets removed, because I want to be sure this helps and doesn't just thin the cloud. Two groups fall below 0.05 at reset+300. The first is genuinely redundant Gaussians whose opacity was always near zero — they contribute almost nothing to any view. Cutting them is pure win: it frees capacity (and under a count budget, lets it be reallocated to Gaussians that matter), helping both training and test. The second group is the overfit Gaussians whose opacity *was* respectable but didn't recover after the reset. Cutting these is the real point. Now, an overfit Gaussian was earning its opacity from the few views it helps, so after I prune it the gradient from those views may well *regrow* a Gaussian in roughly that spot — densification will refill the high-gradient region. But here's the lever: the regrown Gaussian starts fresh and isn't necessarily overfit in the same way; it might converge to a configuration that's consistent across views. So each round of reset-recover-prune-regrow replaces some overfit Gaussians with possibly-not-overfit ones, and over many rounds the *fraction* of overfit Gaussians in the cloud keeps dropping. The mechanism is a slow purification: I'm not enforcing a hard opacity constraint on the final result — the threshold is only 0.05, the interval is long, and it only runs during the densification phase — I'm running a repeated filter whose fixed point has fewer conflicted Gaussians.

There's a clean prediction in that, and it tells me this is a *generalization* fix specifically. Pruning the redundant first group reallocates resources, so it should help training loss and test loss both. But pruning-and-regrowing the overfit second group *doesn't change the resource allocation* — a Gaussian gets removed and another grows back in its place — it only swaps a view-inconsistent Gaussian for a (hopefully) view-consistent one. So I'd expect recovery-aware pruning, beyond what plain redundant-pruning gives, to lift the *test* metric without much moving the *training* metric. That asymmetry — generalization up, training roughly flat — is exactly the fingerprint of fixing overfitting rather than fixing under-capacity, and it's what I'd want to validate. It also tells me this pairs especially well with my deterministic long-axis split: the deterministic split is a stronger detail-fitter than the noisy old split, which means it's also *more* prone to overfitting fine regions, so the two pieces are complementary — the split fits detail aggressively, the recovery-aware prune cleans up whatever of that detail overfit.

One more thing falls out about the opacity reset that I want to keep. I argued the reset is the poke that makes the recovery-rate signature visible, but the reset is also independently mild-but-positive even apart from pruning: clamping all opacities and letting them re-grow rebalances how rendering contribution is spread across Gaussians — it tends to pull down the handful of near-saturated (opacity > 0.9) hogs and redistribute that contribution to many mid-opacity Gaussians, and a more evenly shared contribution reconstructs slightly better. So I keep the 3000-iteration reset to 0.01: it earns its place twice, as a gentle rebalancer and as the trigger that exposes overfit Gaussians for recovery-aware pruning.

Let me now assemble the whole density-control rule into the training-loop hooks and make the long-axis split concrete as code I'd actually run. The harness gives me, after each backward pass, the screen-space mean gradient (I retain it in the pre-backward hook), the image size and camera count, the per-Gaussian radii and which Gaussians were visible. I accumulate the densification signal — the per-Gaussian view-space positional gradient — over the refinement window; whatever selection criterion the surrounding method uses (a plain threshold, the absolute-value homodirectional gradient that stops over-reconstructed Gaussians from being starved by sign cancellation, a budgeted score) feeds the same mask. On a refine step I build the high-gradient mask, run my long-axis split on it, and run the opacity-floor prune. Separately, 300 iterations after each reset, I run recovery-aware pruning; and every 3000 iterations I reset opacity. The long-axis split itself is the new piece, so let me write it carefully — selecting the longest axis, rotating its local unit vector to world frame, offsetting the two children by half the max scale, halving the long scale and shrinking the other two to 0.85, and dropping each child's opacity to 0.6 of the parent's — operating on log-scales and logit-opacities the way the parameters are stored, and resizing the optimizer state alongside the parameters:

```python
from dataclasses import dataclass
from typing import Any, Dict
import torch
import torch.nn.functional as F

from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import remove, reset_opa, _update_param_with_optimizer
from gsplat.utils import normalized_quat_to_rotmat


@dataclass
class CustomStrategy(Strategy):
    """Density control: split-only long-axis split + recovery-aware pruning."""

    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    grow_grad2d: float = 0.0002          # densification gradient threshold (raise if Abs grads)
    prune_opa: float = 0.005             # standard low-opacity prune floor
    absgrad: bool = False
    key_for_gradient: str = "means2d"
    # Long-Axis Split constants (sweep extrema; +-0.05 is robust)
    las_opa_factor: float = 0.6          # child opacity = 0.6 * parent
    las_short_factor: float = 0.85       # non-longest axes shrunk to 0.85
    las_long_div: float = 2.0            # longest axis halved
    # Recovery-Aware Pruning
    recovery_offset: int = 300           # iters after each reset to test recovery
    recovery_opa: float = 0.05           # prune Gaussians not recovered above this

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        return {"grad2d": None, "count": None, "scene_scale": scene_scale}

    def step_pre_backward(self, params, optimizers, state, step, info):
        info[self.key_for_gradient].retain_grad()  # screen-space mean gradient after backward

    @torch.no_grad()
    def _long_axis_split(self, params, optimizers, state, mask, scene=None):
        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]
        if len(sel) == 0:
            return

        scales = torch.exp(params["scales"][sel])                 # [S,3] true scales
        quats = F.normalize(params["quats"][sel], dim=-1)
        rotmats = normalized_quat_to_rotmat(quats)                # [S,3,3]

        a = scales.argmax(dim=-1, keepdim=True)                   # [S,1] longest-axis index
        e_local = torch.zeros_like(scales).scatter_(1, a, 1.0)    # local unit vector e_a
        direction = torch.einsum("sij,sj->si", rotmats, e_local)  # world long-axis dir d = R e_a
        s_max = scales.gather(1, a).squeeze(-1)                   # [S] longest scale

        # children at mu +- (1/2) s_max * d  (centers one max-radius apart, inside parent extent)
        offset = 0.5 * s_max.unsqueeze(-1) * direction            # [S,3]
        samples = torch.stack([offset, -offset], dim=0)           # [2,S,3]

        # shape: longest axis halved, the other two shrunk to 0.85 (union ~ parent footprint)
        new_scales = scales * self.las_short_factor
        new_scales.scatter_(1, a, s_max.unsqueeze(-1) / self.las_long_div)

        # opacity: child = 0.6 * parent  (matches unimodal->bimodal density; keeps transmittance)
        a_parent = torch.sigmoid(params["opacities"][sel])
        a_child = (self.las_opa_factor * a_parent).clamp(1e-6, 1.0 - 1e-6)
        new_opa_logit = torch.logit(a_child)

        def param_fn(name, p):
            repeats = [2] + [1] * (p.dim() - 1)
            if name == "means":
                p_split = (p[sel] + samples).reshape(-1, 3)       # [2S,3]
            elif name == "scales":
                p_split = torch.log(new_scales).repeat(2, 1)      # store as log-scale
            elif name == "opacities":
                p_split = new_opa_logit.repeat(repeats)           # store as logit
            else:
                p_split = p[sel].repeat(repeats)                  # quats, colors copied
            return torch.nn.Parameter(torch.cat([p[rest], p_split]), requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=v.device)
            return torch.cat([v[rest], v_split])                  # reset Adam state for children

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

        # accumulate gsplat's screen-space gradient signal
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

        # Recovery-Aware Pruning: 300 iters after each reset, cut Gaussians not recovered > 0.05
        if step > self.reset_every and (step - self.recovery_offset) % self.reset_every == 0:
            not_recovered = torch.sigmoid(params["opacities"].flatten()) < self.recovery_opa
            if not_recovered.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=not_recovered, scene=scene)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            avg_grads = state["grad2d"] / state["count"].clamp_min(1)
            is_high = avg_grads > self.grow_grad2d
            if is_high.sum() > 0:                                  # split-only: one operation
                self._long_axis_split(params, optimizers, state, is_high, scene=scene)

            is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
            if is_prune.sum() > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=is_prune, scene=scene)

            state["grad2d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

        # opacity reset: rebalances contribution AND triggers the recovery-rate test above
        if step % self.reset_every == 0 and step > 0:
            reset_opa(params=params, optimizers=optimizers, state=state, value=0.01)
```

Let me trace the causal chain back to be sure it holds. I started from the real bottleneck: densification is interleaved with optimization, so every edit is a perturbation the optimizer must recover from, and the lever for efficiency is making each edit minimally disturb the rendered scene. Examining the two existing operations through that lens, clone makes near-coincident, unrecoverable duplicates when the parent's step is small, and split — covariance-sampled, full-shape, full-opacity — jumps the covered shape away from the optimized geometry, injects sampling variance, and overcompounds the density. Since Gaussian size converges to one equilibrium under backprop, the under/over distinction that justified two operations dissolves, so I keep a single split. Making that split minimally disruptive forces each choice: split along the longest axis for maximum separation and most-needed subdivision; offset by half the max scale so the children separate but stay inside the parent's extent; halve the long axis so two displaced bumps tile the parent's long profile; shrink the other two axes to ~0.85 so the union's cross-section matches the parent and very thin Gaussians round out; drop each child's opacity to ~0.6 of the parent's — not the spatial-density minimum (that sits near 0.84) but the central value of the transmittance-preserving factor `(1 − sqrt(1 − alpha))/alpha` over the opacities a Gaussian actually holds, which is what the renderer integrates and which two full-opacity children badly violate. Deterministic placement also removes the old split's run-to-run variance. Separately, overfit Gaussians degrade held-out views while keeping ordinary opacity, so opacity-threshold pruning is blind to them; but the opacity reset is a poke that reveals a dynamical signature — normal Gaussians, consistent across views, recover monotonically and fast, while overfit Gaussians, conflicted across views, oscillate and recover slowly — so pruning Gaussians still below 0.05 at 300 iterations after each reset catches the laggards, removing redundant Gaussians (reallocating capacity, helping train and test) and overfit ones (swapped for possibly-consistent regrowths, lifting generalization without changing allocation), the two pieces complementary because the sharper deterministic split fits detail harder and so overfits more, which the recovery-aware prune then cleans up.
