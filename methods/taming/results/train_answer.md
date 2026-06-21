When I train a point-based radiance field I start from a sparse Structure-from-Motion point cloud, turn each point into an optimizable anisotropic 3D Gaussian — a mean $\mu$, a covariance $\Sigma = R S S^\top R^\top$, an opacity $o$, and view-dependent color via order-3 spherical harmonics — and let the optimizer *grow* the set: every hundred iterations it averages each Gaussian's view-space positional gradient, and wherever that average exceeds a fixed threshold $\tau_{\text{pos}} = 0.0002$ it densifies (clones the Gaussian if it is small, splits it into two covariance-sampled children if it is large), separately prunes near-transparent ones, and periodically resets all opacities to force more pruning. This works for quality, but the failure I keep hitting is about *control*, not quality. The number of Gaussians I end up with is whatever this open-loop automaton decides: two scenes started from the same number of SfM points can finish an order of magnitude apart, a single unbounded scene can converge to several million Gaussians and more than a gigabyte on disk, and I can neither set nor predict the count in advance. Worse, because the heuristic over-produces and then prunes, the *peak* count partway through training is far above the final count — and it is the peak that decides whether the run even fits in GPU memory, so on a constrained device a run can simply die mid-training, unpredictably. Anything downstream that needs a fixed-size input — a classifier over the primitive set, a fixed VRAM envelope, a streaming budget — cannot touch a representation whose size I cannot name. The existing baselines do not close this. The threshold-based adaptive density control of standard 3DGS has no notion of a target count at all. Reframing optimization as MCMC sampling gives easier control and a principled view of densification but not an *exact* budget hit along a smooth, peak-free schedule. On-disk compression acts after the fact and never bounds the training-time peak. Concurrent limited-count splatting still over-samples before pruning back, so the peak-to-final gap remains. And backward-pass accelerators reduce the cost of the rasterizer atomics without removing the contention.

I propose Taming 3DGS: an optimization that lands on an exact, user-specified Gaussian budget $B$ deterministically and *constructively* — the count climbs smoothly toward $B$ with no overshoot-then-prune hump, so peak memory is essentially final memory — and that spends the fixed budget where it most raises image quality, while removing the main training bottlenecks. The first move is to separate two problems I should never have conflated: *how many* Gaussians to add over time, and *which* ones to add. The count problem carries the hard guarantee, so I take it first. I want a target-count curve $A(x)$ that says, at densification step $x$, how many Gaussians the model should have; then at each event I add the gap between $A$ and my current count and land on $B$ by construction. The shape of $A$ should match how growth actually behaves, because the rest of the machinery — the decaying positional learning rate especially — is tuned to a particular rhythm, and the observed rhythm is front-loaded: across outdoor scenes the number added per step falls over training as fewer primitives clear the threshold, so the total rises fast early and flattens near the end. That gives me exactly three constraints. The curve must start at the SfM count, $A(0) = S$; it must end exactly at the budget, $A(N) = B$ over the $N$ densification steps; and to mimic the flattening, its slope must vanish at the end, $A'(N) = 0$. Three constraints is precisely a quadratic $A(x) = a x^2 + b x + c$, and a downward parabola is exactly a curve that rises and then flattens — I need nothing fancier. Solving, $A(0) = c = S$. The derivative $A'(x) = 2ax + b$ with $A'(N) = 0$ forces $b = -2aN$. Then $A(N) = aN^2 + bN + c = -aN^2 + S = B$ gives $a = -(B-S)/N^2$, and back-substituting $b = 2(B-S)/N$, so

$$A(x) = -\frac{B-S}{N^2}\,x^2 + \frac{2(B-S)}{N}\,x + S.$$

The corners check out: $A(0) = S$ and $A(N) = -(B-S) + 2(B-S) + S = B$. The vertex sits at $x = -b/(2a) = N$, so the count climbs monotonically over $[0, N]$ and tops out at exactly $B$ at the final step, never overshooting — that is the constructive, peak-free schedule, and since $a < 0$ the per-step additions decay linearly to zero, reproducing the early-fast, late-flat shape. One wrinkle: the optimizer still prunes genuinely dead Gaussians along the way, so if I followed the per-step *increments* of $A$ I would undershoot $B$. The fix is to treat $A$ as an *accumulated* target — at each event add (target at this step − current actual count) — so along-the-way pruning is automatically compensated and I still hit $B$.

Now the harder problem: of all Gaussians, which $B_{\text{step}}$ do I densify at this event? "Everything above the gradient threshold" over-densifies and leaves redundancy. The positional gradient is a decent under-sampling detector but a single blunt scalar — it says nothing about whether a Gaussian actually contributes to the image, whether its projection is so large it is just blurring, whether it is a near-transparent floater the optimizer is already killing, or whether the error it sits on is real under-sampling versus a misplaced primitive. On a strict budget, densifying a floater or duplicating a misplaced Gaussian is pure waste. The right framing is point-cloud downsampling turned inside out — score each Gaussian by how much densifying it would help, and select against the budget — and the sampling view of densification (the gradient step plus noise is a Langevin update, so adding a primitive is a *sampling* operation and the loss landscape should say where to sample) pushes the same way. But loss-steered densification has a trap that fixes the *frequency*: a Gaussian reads high loss either because its neighborhood is under-sampled or because it is itself misplaced, and if I densify every hundred iterations I will keep cloning misplaced primitives into more misplaced primitives — a feedback loop that burns budget. What saves me is time: given enough iterations between events, ordinary optimization lowers the opacity of a misplaced Gaussian or moves it, so by the next event the misplaced ones have faded or been fixed and what still reads "high loss" is much more likely to be true under-sampling. So I densify every $500$ iterations — a fifth of the original cadence — and that reduced frequency is not a speed hack, it is what makes loss-as-signal safe.

The score itself is a weighted basket, where each signal earns its place by the failure mode it catches. The positional gradient $\nabla_g$ stays in as the under-sampling detector. The pixel count $c_g$ a Gaussian covers flags large projections that produce blur. The accumulated pixel-to-center distance $D_g$ catches thin elongated slivers that cover few pixels but stretch far. The per-pixel saliency $S_v$ — the photometric L1 map plus a Laplacian edge term, $S_v = \mathbb{1}_{\text{ROI}} \odot (\lambda_1 L_1(v) + \lambda_2 E(v))$ with $\lambda_1 = \lambda_2 = 0.5$ — routes the image loss and high-frequency content directly into densification. The sum of blending weights $B_g$ over the pixels a Gaussian touches is the most direct measure of how much it actually *contributes* to the render, so densifying a high-contribution Gaussian is the change most likely to be visible. The depth $z_g$ is zero outside the view frustum, so summed over views it doubles as visibility × distance, prioritizing commonly-seen foreground without starving rare background. The opacity $o_g$ gets the *highest* weight so the score steers hard *away* from low-opacity floaters. And the scale product $s_g$ penalizes oversized Gaussians that hurt novel-view generalization. These attributes live on wildly different magnitudes, so I cannot just add them; and a few outliers (a giant background splat, a degenerate sliver) would wreck any mean-based normalization. So for each attribute I rescale by its *median* over the Gaussians with a positive value — divide by the median, multiply by a per-attribute weight — which is outlier-robust where the mean is not and makes every attribute order-one and comparable. The per-Gaussian score then aggregates the median-scaled weighted sum $F$ over $N_{\text{cam}} = 10$ sampled views, each view weighted by its own photometric loss $P_i = 0.8\,L_1 + 0.2\,(1 - \text{SSIM})$ so badly-reconstructed views dominate:

$$S_g = \sum_{i=1}^{N_{\text{cam}}} P_i \cdot F\!\left(\nabla_g,\, c_g^i,\, D_g^i,\, S_v^i,\, B_g^i,\, z_g^i,\, o_g,\, s_g\right).$$

Ten views is the compromise between viewpoint diversity and the cost of scoring every view. To turn $S_g$ into a selection I deliberately reject hard top-$k$: the score is a noisy estimate from ten random views and the MCMC reading says densification should retain exploration, so I draw candidates by $\texttt{multinomial}(S_g, B_{\text{step}}, \text{replacement=False})$ — high-scoring Gaussians are very likely picked, a lower-scoring one is occasionally explored, and none is densified twice. A chosen Gaussian is cloned if small (max scale $\le$ percent\_dense $\cdot$ extent) or split into two covariance-sampled children (scale $\div\, 0.8N$, $N=2$) if large, so I divide $B_{\text{step}}$ between the two in proportion to the qualifying-candidate populations: $\text{clone\_budget} = B_{\text{step}} \cdot n_{\text{clone}}/(n_{\text{clone}} + n_{\text{split}})$ and likewise for split. Pruning is the one thing that could reintroduce the sawtooth, so it is kept gentle and constructive: cap removals at half the usual dead-weight (low-opacity / oversized) count, draw inverse-score candidates with weight $1/(\epsilon + S_g)$ to protect high-score primitives, prune only the *intersection* with the dead-weight mask, and only early in training — the schedule, not pruning, governs the count.

Two pieces lift quality at a low count, and a set of runtime changes pay for the speed. For quality, after densification ends (around iteration $15000$) I switch the opacity activation from sigmoid to $\texttt{abs}$ and clamp blending weights to $1$ from above, so a primitive's effective opacity can exceed $1$ and a single Gaussian can model an opaque surface that would otherwise need several stacked — done late so it sharpens the fixed set rather than perturbing growth. For speed, the backward pass dominates and is atomic-reduction-bound: the standard rasterizer maps threads to pixels and each thread atomically adds its per-pixel gradient into a splat, so many threads hammer the same splat's gradient memory and the atomics serialize. I flip the parallelization to one thread per *splat* — then thread $i$ alone accumulates splat $i$'s gradient and there are no atomics. The state it needs (transmittance $T$ and accumulated RGB at the moment splat $i$ composites) is reconstructed by checkpointing per-pixel state every 32nd splat during the forward pass and exchanging the small per-pixel state along the warp by intra-warp shuffle in the backward, with occluded splat tails skipped via a per-tile last-contributor and tighter culling — all numerically equivalent to the original gradient. Then the Adam step: spherical harmonics are 48 of the 59 per-Gaussian attributes and the higher bands barely move per iteration, so I update them once every 16 iterations with a separate optimizer (the one non-equivalent change, compensated by a higher SH learning rate), read DC and higher SH from separate tensors to drop a concat, and compute SSIM via two separable 1D convolutions in a fused kernel.

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
