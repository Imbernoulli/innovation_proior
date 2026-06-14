# Taming 3DGS, distilled

Taming 3DGS makes point-based radiance-field optimization *budgeted and controllable*. It replaces the
open-loop, threshold-based growth of standard 3D Gaussian Splatting with (1) a deterministic, purely
constructive growth schedule that reaches an exact user-specified Gaussian count B, (2) score-based
multinomial sampling that spends that budget on the primitives most worth densifying, (3) late-stage
high-opacity Gaussians for cheaper opaque-surface modeling, and (4) runtime optimizations around the
dominant backward, SH-update, and SSIM costs.

## Problem it solves

Standard 3DGS densifies every Gaussian whose view-averaged view-space positional gradient exceeds a fixed
threshold, prunes near-transparent ones, and resets opacities periodically. The final primitive count is an
uncontrolled emergent number (up to an order of magnitude variation between scenes from the same SfM init,
millions of Gaussians, >1 GB on disk); the over-grow-then-prune behavior makes the *peak* count far exceed
the *final* count, so peak memory is large and unpredictable and can OOM constrained devices; per-iteration
cost (backward pass, then Adam updates) scales with the count. Goal: hit an exact budget B deterministically
and constructively (peak ≈ final), spend it where it raises quality, and train faster.

## Key ideas

**1. Predictable model growth (parabolic target schedule).** Standard 3DGS grows quickly early and flattens
later as fewer Gaussians satisfy the gradient threshold. Define a *target-count* curve A over the N
densification steps by three constraints:
A(0) = S (SfM count), A(N) = B (budget), A'(N) = 0 (growth flattens at the end). The unique quadratic is

```
A(x) = a x^2 + b x + c,   a = -(B - S) / N^2,   b = 2 (B - S) / N,   c = S.
```

Its vertex is at x = N, so the count climbs monotonically to exactly B with no overshoot. Follow A as an
*accumulated* target (densify the gap between A's value and the current count) so along-the-way pruning is
compensated. Densification runs every 500 iterations — a fifth of the original cadence — so the optimizer can
lower the opacity of misplaced Gaussians or move them before loss-steered densification would otherwise
duplicate them.

**2. Score-based steerable densification.** At each event, sample N_cam = 10 random training views. Per view v,
form a per-pixel saliency map

```
S_v = 1_ROI ⊙ ( λ1 · L1(v, r_v) + λ2 · E(v) ),   λ1 = λ2 = 0.5,
```

with E a Laplacian edge filter and 1_ROI an optional region-of-interest mask. Per Gaussian g, compute a score

```
S_g = Σ_i  P_i · F( ∇_g, c_g^i, D_g^i, S_v^i, B_g^i, z_g^i, o_g, s_g ),
```

where P_i = 0.8·L1 + 0.2·(1 − SSIM) is the view's photometric loss, and F is a weighted sum of each
attribute *median-scaled* (divide positive values by their median — outlier-robust — times a per-attribute
weight). The signals and their roles: ∇_g positional gradient (under-sampling detector, kept but not trusted
alone); c_g pixel count (large projection → blur); D_g pixel-to-center distance (thin "slivers"); S_v saliency
(routes loss/edges/ROI into densification); B_g blending-weight sum (actual image contribution → biggest
quality gain); z_g depth (zero outside frustum ⇒ doubles as visibility × distance); o_g opacity (highest
weight — steer *away* from low-opacity floaters); s_g scale product (penalize oversized Gaussians that hurt
novel-view generalization). Candidates are drawn by `multinomial(S_g, budget, replacement=False)` — score-
weighted but stochastic, keeping exploration rather than greedy top-k. The per-step budget is split between
clone (small Gaussians) and split (large, N = 2 covariance-sampled children, scale ÷ 0.8N) in proportion to
the qualifying-candidate populations. Pruning is *constructive*: set a removal cap to half of the
would-be-pruned low-opacity/oversized count, sample inverse-score candidates with weight 1/(ε + S_g), and prune
only the intersection with the dead-weight mask, only early in training — the schedule, not pruning, governs
the count.

Representative weights (per-Gaussian: grad 25, opacity 100, depth 5, radii 10, scale 25; per-pixel:
distance 50, loss 10, count 0.1, blend 50; outer view multiplier 50).

**3. High-opacity Gaussians.** After densification ends (~iteration 15000), switch the opacity activation from
sigmoid to `abs` and clamp blending weights to 1 from above during rendering. A primitive's effective opacity
can then exceed 1, letting a single Gaussian model an opaque surface that would otherwise need several.

**4. Runtime optimizations.** *Per-splat backward:* the original maps threads to pixels and atomically adds each
pixel's gradient into the splat — heavy atomic contention. Instead parallelize over splats (one thread owns one
splat's gradient → no atomics); reconstruct the needed per-pixel blending state by checkpointing it every 32nd
splat in the forward pass and exchanging the small per-pixel state (transmittance T, accumulated RGB) via
intra-warp shuffle in the backward; skip occluded splat tails via a per-tile last-contributor; apply tighter
culling. Numerically equivalent. *Batched SH:* SH are 48 of 59 per-Gaussian attributes; update the higher bands
once every 16 iterations with a separate optimizer (the one non-equivalent change, compensated with a higher
SH learning rate),
read DC and higher SH from separate tensors to drop a concat, and compute SSIM via two separable 1D
convolutions plus a fused kernel.

## Final algorithm

```
S <- SfM point count;  B <- user budget;  N <- number of densification steps
target[x] = -((B-S)/N^2) x^2 + (2(B-S)/N) x + S          # x = 0..N accumulated targets
for step in 1..30000:
    render view; loss = 0.8*L1 + 0.2*(1-SSIM); backward
    accumulate view-space positional-gradient stats
    if step % 500 == 0 and densifying:
        sample 10 views; S_g = sum_v P_v * F(median-scaled attributes)
        budget_step = target[next] - current_count
        split budget_step into clone/split by qualifying-candidate populations
        clone <- multinomial(S_g over small qualifiers, clone_budget)
        split <- multinomial(S_g over large qualifiers, split_budget)  # N=2 children, scale/0.8N
        prune <- dead_weight_mask AND inverse_score_sample(cap=0.5*dead_weight_count, early only)
    if step == 15000:  opacity activation sigmoid -> abs;  clamp blend <= 1
    Adam step  (SH bands>0 every 16 steps)
```

## Working code

The code mirrors the canonical implementation: the parabolic count schedule, the median-scaled score over sampled
views, multinomial clone/split selection with population-proportional budget split, and constructive
inverse-score pruning. Fills the densification slot of the 3DGS training harness.

```python
import torch


def get_count_array(start_count, budget, num_steps):
    """Accumulated target counts. A(0)=S, A(N)=B, A'(N)=0 ->
    A(x) = -((B-S)/N^2) x^2 + (2(B-S)/N) x + S."""
    S, B, N = float(start_count), float(budget), num_steps
    a = -(B - S) / (N * N)
    b = 2.0 * (B - S) / N
    c = S
    return [int(a * x * x + b * x + c) for x in range(num_steps + 1)]


def normalize(weight, value):
    """Outlier-robust per-attribute scaling: positive values / their median, * weight."""
    value = torch.nan_to_num(value, nan=0.0).to(torch.float32)
    out = torch.zeros_like(value)
    pos = value > 0
    if pos.any():
        out[pos] = weight * (value[pos] / value[pos].median())
    return out


# representative score weights
SCORE_W = {"view": 50, "edge": 50, "mse": 50, "grad": 25, "dist": 50,
           "opac": 100, "dept": 5, "loss": 10, "radii": 10, "scale": 25,
           "count": 0.1, "blend": 50}


def compute_gaussian_score(scene, cams, edge_maps, gaussians, renderer, bg, w):
    """S_g = sum_i P_i * F(per-Gaussian + per-pixel-accumulated, median-scaled)."""
    n = len(scene.gaussians.get_xyz)
    score = torch.zeros((len(cams), n), device="cuda", dtype=torch.float32)
    opac = scene.gaussians.get_opacity.detach().squeeze()
    scale = torch.prod(scene.gaussians.get_scaling.detach(), dim=1)
    grads = scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
    grads[grads.isnan()] = 0.0
    grads = grads.detach().squeeze()

    for i, cam in enumerate(cams):
        img = renderer(cam, gaussians, bg)["render"]
        P_i = 0.8 * l1_loss(img, cam.gt) + 0.2 * (1.0 - ssim(img, cam.gt))   # photometric loss
        # per-pixel saliency S_v = 0.5*L1_map + 0.5*edges (min-max normalized)
        l1m = (img - cam.gt).abs().mean(0).detach()
        l1m = (l1m - l1m.min()) / (l1m.max() - l1m.min())
        S_v = w["mse"] * l1m + w["edge"] * edge_maps[i].cuda()
        pkg = renderer(cam, gaussians, bg, pixel_weights=S_v)               # carry weights through raster
        loss_acc, dist_acc = pkg["accum_weights"], pkg["accum_dist"]
        blend, rev_count = pkg["accum_blend"], pkg["accum_count"]
        vis = pkg["visibility_filter"].detach()
        depth, radii = pkg["gaussian_depths"].detach(), pkg["gaussian_radii"].detach()

        g_part = (normalize(w["grad"], grads) + normalize(w["opac"], opac) +
                  normalize(w["dept"], depth) + normalize(w["radii"], radii) +
                  normalize(w["scale"], scale))
        p_part = (normalize(w["dist"], dist_acc) + normalize(w["loss"], loss_acc) +
                  normalize(w["count"], rev_count) + normalize(w["blend"], blend))
        agg = w["view"] * P_i * (p_part + g_part)
        score[i][vis] = agg[vis]
    return score.sum(dim=0)                                                 # S_g


class GaussianModel:
    # ... existing 3DGS tensors, optimizers, clone/split/prune helpers ...

    def add_densification_stats(self, viewspace_points, update_filter):
        # running view-space positional-gradient signal
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_points.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_clone_taming(self, scores, budget, mask):
        scores[~mask] = 0
        sel = torch.zeros(self.get_xyz.shape[0], dtype=bool, device="cuda")
        sel[torch.multinomial(scores, budget, replacement=False)] = True
        self.densification_postfix(self._xyz[sel], self._features_dc[sel],
                                   self._features_rest[sel], self._opacity[sel],
                                   self._scaling[sel], self._rotation[sel],
                                   self.tmp_radii[sel])

    def densify_and_split_taming(self, scores, budget, mask, N=2):
        scores[~mask] = 0
        sel = torch.zeros(self.get_xyz.shape[0], dtype=bool, device="cuda")
        sel[torch.multinomial(scores, budget, replacement=False)] = True
        stds = self.get_scaling[sel].repeat(N, 1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = build_rotation(self._rotation[sel]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[sel].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[sel].repeat(N, 1) / (0.8 * N))
        self.densification_postfix(new_xyz, self._features_dc[sel].repeat(N, 1, 1),
                                   self._features_rest[sel].repeat(N, 1, 1),
                                   self._opacity[sel].repeat(N, 1),
                                   new_scaling, self._rotation[sel].repeat(N, 1),
                                   self.tmp_radii[sel].repeat(N))
        self.prune_points(torch.cat((sel, torch.zeros(int(N * sel.sum()), device="cuda", dtype=bool))))

    def densify_with_score(self, scores, max_screen_size, min_opacity, extent, budget, radii, iter_num):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.tmp_radii = radii

        qualifies = torch.norm(grads, dim=-1) >= 0.0002
        max_scale = self.get_scaling.max(dim=1).values
        clone_ok = qualifies & (max_scale <= self.percent_dense * extent)
        split_ok = qualifies & (max_scale > self.percent_dense * extent)
        n_clone, n_split = int(clone_ok.sum()), int(split_ok.sum())

        cur = len(self.get_xyz)
        budget = min(budget, n_clone + n_split + cur)  # budget is the accumulated target count
        denom = max(n_clone + n_split, 1)
        clone_budget = ((budget - cur) * n_clone) // denom
        split_budget = ((budget - cur) * n_split) // denom

        self.densify_and_clone_taming(scores.clone(), clone_budget, clone_ok)
        self.densify_and_split_taming(scores.clone(), split_budget, split_ok, N=2)

        # constructive pruning: inverse-score candidates, intersected with dead weight, early only
        prune = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            prune |= self.max_radii2D > max_screen_size
            prune |= self.get_scaling.max(dim=1).values > 0.1 * extent
        remove_budget = int(0.5 * int(prune.sum()))
        if remove_budget and iter_num < 27:
            inv = torch.zeros(self.get_xyz.shape[0], dtype=torch.float32, device="cuda")
            inv[: scores.shape[0]] = 1.0 / (1e-6 + scores.squeeze())
            keep = torch.zeros_like(inv, dtype=bool)
            keep[torch.multinomial(inv, remove_budget, replacement=False)] = True
            self.prune_points(prune & keep)
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def modify_functions(self):
        # high-opacity Gaussians: sigmoid -> abs (call once after densification ends)
        old_opacities = self.get_opacity.clone()
        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = identity_gate
        self._opacity = self.opacity_activation(old_opacities)
```

Training-loop wiring (every 500 iterations once densifying; high-opacity at 15000; batched SH every 16 steps):

```python
counts = get_count_array(len(scene.gaussians.get_xyz), args.budget, num_densify_steps)
densify_idx = 0
for it in range(1, 30001):
    render; loss = 0.8*L1 + 0.2*(1 - SSIM); loss.backward()
    if it < densify_until:
        gaussians.add_densification_stats(viewspace_points, visibility_filter)
        if it > densify_from and it % 500 == 0:
            cams, edges = sample_views(scene, n=10)
            S_g = compute_gaussian_score(scene, cams, edges, gaussians, render, bg, SCORE_W)
            gaussians.densify_with_score(S_g, max_screen_size=(20 if it > opacity_reset else None),
                                         min_opacity=0.005, extent=scene.extent,
                                         budget=counts[densify_idx + 1],
                                         radii=radii, iter_num=densify_idx)
            densify_idx += 1
        if it % opacity_reset == 0:
            gaussians.reset_opacity()
    if it == 15000:
        gaussians.modify_functions()                       # enable high-opacity
    gaussians.optimizer.step(); gaussians.optimizer.zero_grad(set_to_none=True)
    if it % 16 == 0:                                        # batched higher-SH update
        gaussians.shoptimizer.step(); gaussians.shoptimizer.zero_grad(set_to_none=True)
```
