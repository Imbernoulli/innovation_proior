Let me start from what actually goes wrong when I train one of these Gaussian scenes, because the failure is concrete and it's not about quality at all — it's about control. I start from a sparse SfM point cloud, turn each point into an anisotropic 3D Gaussian, and let the optimizer grow the set: every hundred iterations it looks at each Gaussian's view-space positional gradient, and wherever that averaged gradient exceeds a fixed threshold τ_pos = 0.0002 it densifies — clones the Gaussian if it's small, splits it into two covariance-sampled children if it's large — and separately it prunes near-transparent ones and periodically resets all opacities to force more pruning. It works. But the number of Gaussians I end up with is whatever this open-loop automaton decides. I run two scenes from the same number of SfM points and one finishes with a few hundred thousand Gaussians and the other with several million — an order of magnitude apart, and over a gigabyte on disk for a single unbounded scene. I cannot predict it, I cannot set it, and worse: because the heuristic over-produces and then prunes, the *peak* count partway through training is far above the final count. And it's the peak that decides whether the run even fits in GPU memory. So on a constrained device a run can simply die mid-training, unpredictably. Anything downstream that wants a fixed-size input — a classifier over the primitive set, a fixed VRAM envelope, a streaming budget — can't touch a representation whose size I can't name in advance.

So the thing I want is sharp: hit an *exact* user-given Gaussian count B, deterministically, on any scene, and get there *constructively* — the count should climb smoothly toward B without a big overshoot-then-prune hump, so that peak memory is essentially final memory and never spikes past the device's budget. And I'd like to spend that fixed budget B well, since if I use fewer Gaussians than the automaton would, each one has to earn its keep. Those are really two problems — *how many* to add over time, and *which* ones — and I should not conflate them.

Take the count problem first, because it's the one with the hard guarantee attached. I want a target-count schedule: a function that says, at densification step x, how many Gaussians the model *should* have. If I have that, then at each densification event I just add the difference between the target and my current count, and I land on B by construction. The question is the shape of that schedule. I could go linear from S (the SfM count) to B over the N densification steps, but that doesn't match how growth actually behaves, and matching the natural behavior matters because the optimizer's other machinery — the decaying positional learning rate especially — is tuned to a particular rhythm of growth. So let me look at how the automaton actually grows. Across outdoor scenes, growth is front-loaded: the number of Gaussians added per step falls over training as progressively fewer Gaussians clear the gradient threshold, and the total count rises fast early and flattens toward the end.

That gives me the three constraints a target curve A(x) has to satisfy. It has to start at the SfM count, A(0) = S. It has to end exactly at the budget, A(N) = B. And to mimic the observed flattening — additions per step decaying to near zero by the end — its slope at the end should vanish, A'(N) = 0. Three constraints means I want exactly three free parameters. A quadratic A(x) = a x² + b x + c has three, so let me try the simplest thing that could fit before reaching for anything more elaborate. Solve. A(0) = c = S, so c = S. The derivative is A'(x) = 2a x + b, and A'(N) = 0 forces 2aN + b = 0, i.e. b = −2aN. Then A(N) = aN² + bN + c = aN² − 2aN² + S = −aN² + S, and setting that equal to B gives −aN² = B − S, so a = −(B − S)/N², and back-substituting, b = −2N·a = 2(B − S)/N. So

  A(x) = −((B − S)/N²) x² + (2(B − S)/N) x + S.

Before I trust this I should put numbers in and watch the whole curve, not just argue about it. Take a concrete unbounded scene: S = 50,000 SfM points, a budget B = 1,000,000, over N = 60 densification events. The endpoints come out A(0) = 50,000 and A(60) = 999,999 — lands on B up to the integer floor, as it must. The vertex is at x = −b/(2a) = N = 60, so the maximum of the parabola sits exactly at the final step; over [0, 60] the curve is strictly increasing and never rises above B, which is the no-overshoot property I was after. And the per-step increments fall from 31,402 at the first event to 263 at the last — a 119× front-loading. That's the early-fast, late-flat shape the automaton produces empirically, recovered from a < 0 making the slope shrink linearly to zero. So a downward parabola isn't just plausible here; the corner values, the vertex location, and the decaying increments all check out on real numbers.

There's a wrinkle I almost missed, and it would have quietly broken the guarantee. The optimizer keeps pruning genuinely low-opacity Gaussians *between* my densification events, so the actual count drifts below the curve. My first instinct is to add A's per-step increment each event — add 31,402, then 30,875, and so on. Let me simulate that against B with a few thousand Gaussians pruned at random between events: it finishes at ~902,000, undershooting the budget by ~98,000, about ten percent low. The increments never know about the pruning, so the deficit accumulates. The fix is to read A as an *accumulated* target and add the gap to the current actual count, target_at_this_step − current_count, which snaps the population back onto the curve every event regardless of what was pruned. Re-running the same pruning simulation under this rule: immediately after the final event the count is exactly A(N) = 999,999, i.e. B, and the only residual is whatever pruning happens *after* the last densification (a few hundred Gaussians) — not a compounding ten-percent gap. So the accumulated reading is what actually delivers "land on B," and the naive increment reading does not.

Now the harder, more interesting problem: of all the Gaussians, which B_step do I actually densify at this event? The automaton's answer is "everything above the gradient threshold," and I already know that over-densifies and leaves redundancy. The positional gradient is a decent under-sampling detector — high gradient does tend to mean the region needs more primitives — but it's a single blunt scalar. It says nothing about whether the Gaussian actually contributes much to the image, whether its projection is so large it's just blurring things, whether it's a near-transparent floater the optimizer is in the middle of killing, or whether the error it sits on is real under-sampling versus a Gaussian that's simply in the wrong place. If I'm on a strict budget, spending a slot to densify a floater, or to duplicate a misplaced Gaussian, is pure waste. I want to *rank* Gaussians by how much densifying them would actually help, and spend my B_step on the top of that ranking.

Reframing it as ranking-and-selecting is suggestive, because that's exactly point-cloud downsampling turned inside out. There's a whole literature on picking a compact, salient subset of points from a dense cloud — score each point by some importance and keep a budget of the best, whether by local density, a learned sampler, critical-point layers, or differentiable soft selection. "High quality at a low primitive count" is the same shape of problem. So let me build a per-Gaussian *score* S_g that estimates the value of densifying Gaussian g, and select against a budget using it.

And there's a second hint pushing me the same way. If I take seriously the view that the Gaussian set is a *sample* from a distribution tied to the loss — the SGLD / MCMC reading, where the gradient step plus noise is a Langevin update and densification is a state transition of the sample set — then adding a primitive is fundamentally a *sampling* operation, and the loss landscape ought to tell me where to sample. So image loss should steer densification directly. That's intuitive on its face: high image loss in a region means either it's under-sampled (densify) or something there is wrong.

But "high loss → densify" has a trap I should think through before I lean on it, and it's the trap that determines how *often* I'm allowed to densify. A Gaussian causes high image loss for one of two reasons: its neighborhood is genuinely under-sampled, or the Gaussian itself is misplaced. If I densify on loss every hundred iterations like the automaton, I'll keep duplicating misplaced Gaussians — the loss is high, so I clone it, but cloning a misplaced primitive just makes two misplaced primitives, and next event I clone those too. That's a feedback loop that burns budget on garbage. What saves me is *time*. Given enough iterations between densification events, the ordinary optimization handles the misplaced case on its own: it lowers the opacity of an out-of-place Gaussian, or moves it, so by the next densification event the misplaced ones have either been fixed or faded toward pruning, and what's left reading "high loss" is much more likely to be true under-sampling. So loss-steered densification *requires* a lower frequency — I should densify far less often than every hundred steps to let the optimizer self-correct first. Every five hundred iterations, a fifth of the original cadence, gives that breathing room. The reduced frequency isn't a speed hack; it's what makes loss-as-signal safe.

So at a densification event I want a per-Gaussian score. What should go into it? Let me reason through each signal by asking, for each, "what failure mode does densifying-by-this catch, and what does ignoring it cost?"

The positional gradient ∇_g stays in — it's the under-sampling detector, and throwing away a working signal would be foolish; I just refuse to trust it alone. Then: the number of pixels a Gaussian covers, c_g. A Gaussian with a large projection covering many pixels is exactly the one producing blur, so a large c_g is a reason to densify (break it up to sharpen). The accumulated distance from the covered pixels to the Gaussian's center, D_g — this catches the thin elongated "slivers" that cover few pixels but stretch far from their center; those want densifying into more compact primitives. The image saliency itself, S_v — the per-pixel loss and edge content — folded in so that loss and high-frequency regions pull densification toward themselves directly; this is the loss-steering, made concrete. The sum of a Gaussian's blending weights over the pixels it touches, B_g — this is the single most direct measure of how much a Gaussian actually *contributes* to the rendered image, and densifying a high-contribution Gaussian is the change most likely to be *visible*, i.e. to move quality, whereas densifying something that barely contributes is wasted budget. The depth z_g — and here a nice subtlety: z_g is zero for any Gaussian outside the view frustum, so summed across the sampled views it doubles as a combined measure of *how often this Gaussian is seen* and its average distance to the camera; it lets me prioritize commonly-observed foreground primitives without entirely starving rarely-seen background ones. Opacity o_g — I want to steer *away* from low-opacity Gaussians, because low opacity is the signature of a floater or a primitive the optimizer is phasing out, and densifying those is the textbook waste case; so opacity gets a *high* weight to strongly down-rank floaters. And scale s_g, the product of the three axis scales — oversized Gaussians hurt generalization to unseen views even if they look fine in the training views, so scoring on scale pushes toward more uniformly sized primitives.

Now I have a basket of heterogeneous attributes — a gradient norm, a pixel count, a distance sum, a blending-weight sum, a depth, an opacity in (0,1), a scale product — living on wildly different magnitudes. If I just add them I'd let whichever has the largest raw numbers dominate, which is meaningless. I need to put them on a common footing, and robustly, because a few outlier Gaussians (a giant background splat, a degenerate sliver) would wreck any mean-based normalization. So for each attribute I rescale by its *median* over the Gaussians that have a positive value — divide by the median, multiply by a per-attribute weight. The median is outlier-robust where the mean isn't, and dividing by it makes each attribute roughly order-one and comparable; the per-attribute weight then sets how much each one matters. (Concretely the weights I'll use put the highest emphasis on opacity — steering hardest away from floaters — with substantial weight on gradient, the distance/sliver term, the blending-weight contribution, and scale, and small weights on depth and pixel count.)

So the per-view contribution of Gaussian g is a weighted sum of these median-scaled attributes, and I want to aggregate over the N_cam views I sample at this event. Here's where the loss-steering enters at the *view* level too: I weight each view's contribution by that view's photometric loss P_i = 0.8·L1 + 0.2·(1 − SSIM), so views that are currently badly reconstructed dominate the score — the densification budget flows toward the views that need it. The per-Gaussian score is then

  S_g = Σ_{i=1}^{N_cam} P_i · F(∇_g, c_g^i, D_g^i, S_v^i, B_g^i, z_g^i, o_g, s_g),

with F the median-scaled weighted sum, and the per-pixel quantities (c, D, B, and the saliency accumulation) gathered during a render that carries per-pixel weights through the rasterizer. I sample N_cam = 10 views per event as the practical compromise: enough viewpoint diversity to keep the score from being a single-camera accident, but not the cost of scoring every training view at every densification event.

How many views to sample is one knob; the other is how to turn S_g into a selection. The downsampling literature would say "keep the top-B by score," deterministic top-k. But I deliberately don't want hard top-k, and the sampling view tells me why: the score is a *noisy estimate* of importance from ten random views, and the MCMC reading says densification should retain an element of *exploration*, not collapse onto a greedy argmax. So I select stochastically — draw B_step candidates by `multinomial(S_g, B_step, replacement=False)`, i.e. sample without replacement with probability proportional to score. High-scoring Gaussians are very likely to be picked, but a lower-scoring one occasionally gets explored, and I never densify the same one twice in an event. This is the "steerable sampling": the score steers, but it samples rather than dictates.

There's one more thing to settle in the selection: a chosen Gaussian gets *cloned* or *split* depending on its size, exactly as in the original — clone if it's small (max scale ≤ percent_dense·extent), split into two covariance-sampled children if it's large. So my budget B_step for this event has to be divided between a clone budget and a split budget. The clean way is to split it in proportion to the available candidate populations: among Gaussians that also clear the basic gradient qualifier ‖∇‖ ≥ 0.0002, count how many are clone-type (small) versus split-type (large), and allocate

  clone_budget = (B_step) · n_clone / (n_clone + n_split),   split_budget = (B_step) · n_split / (n_clone + n_split),

so the two operations share the budget according to where the qualifying primitives actually are. Then I multinomial-sample clone_budget from the clone-eligible scores and split_budget from the split-eligible scores. Split children each get scale divided by 0.8·N with N = 2, the same shrink the original uses so two children roughly tile the parent's footprint.

And pruning — I have to be careful here, because pruning is precisely what creates the grow-then-prune peak I'm trying to eliminate. The whole point of the constructive schedule is that the count *rises* toward B; if I aggressively prune I reintroduce the sawtooth. So I keep pruning, but make it gentle and constructive: I compute the usual opacity/size dead-weight mask, set a removal budget to half of that mask's count, draw inverse-score candidates with weights 1/(ε + S_g), and prune only the intersection of those candidates with the dead-weight mask. High-score Gaussians are protected, low-score ones are much more likely to go, and the intersection means I may prune less than the cap rather than forcing the count down. I only do this early in training. The schedule, not pruning, governs the count; pruning just trims genuinely dead weight without cutting into the budget.

That's the densification machinery. Two more pieces matter for hitting "high quality at few primitives," and then the speed.

First, the primitives themselves. A standard Gaussian has a rigid exponential falloff, and a single one capped at opacity 1 can only do so much to represent a genuinely opaque surface — I'd need several stacked to look solid, which spends budget. There's a trick from the level-of-detail / hierarchical line: use *high-opacity* Gaussians, clamped, where a primitive's opacity can effectively exceed 1, to approximate clusters with fewer primitives. Let me adopt it for the same reason — to model opaque surfaces with fewer Gaussians. Concretely, after the midpoint of training, once densification is done (around iteration 15000), I switch the opacity activation from sigmoid to abs and clamp the blending weight to 1 from above during rendering. The abs activation lets the underlying opacity parameter push past what sigmoid could reach, and the clamp keeps blending physical; a single primitive can now act fully opaque. I do this late, after the budget is filled, so it sharpens the fixed set rather than perturbing the growth.

Second — and this is where most of the wall-clock actually goes — the per-iteration cost. Profile the standard pipeline and the backward pass is the largest cost, with the Adam optimizer step climbing as the primitive count grows. Take the backward first. The standard rasterizer backward maps threads to *pixels*: each thread walks the depth-sorted splats back-to-front and, for each splat, *atomically adds* its per-pixel gradient contribution into that splat's accumulated gradient. The problem is contention — within a tile, many threads are hammering the *same* splat's gradient memory at the same time, so the atomics serialize, and it's worse here than in a typical reduction because each splat carries a whole vector of attribute gradients (position, covariance, opacity, all the SH). The diagnosis is that it's atomic-reduction-bound, not arithmetic-bound. The known mitigation is to do warp-level reduction to cut the atomic traffic, but that still keeps the per-pixel mapping and the atomics. Let me go further and ask whether I can eliminate the atomics entirely by flipping the parallelization: parallelize over the *splats* instead of the pixels.

If each thread owns one splat, then thread i is solely responsible for accumulating splat i's gradient — no other thread writes to it, so *no atomics*. What thread i needs is, for every pixel j in the tile, the blending state at the moment splat i is composited — the transmittance T and accumulated color RGB after the front i−1 splats. The catch: that state is per-(splat, pixel), and recomputing it from scratch for every splat would be quadratic. But I don't need to store the whole table. In a simplified tile where #threads = #pixels = #splats = M, the idea is: during the *forward* pass, each thread stores one per-pixel blending state every M splats — so I have checkpoints (0, j), (M, j), … for all pixels j. At the start of the backward, each thread reconstructs its needed state (i, j) by stepping forward from the nearest checkpoint, and then threads *exchange* per-pixel states by fast intra-warp shuffling — thread i+1 takes the state (i, j) it received and applies one step of the ordinary alpha-blend to get (i+1, j), folding that into its gradient. So the threads pass the small per-pixel state (just T and RGB) along the warp, instead of every thread storing the large per-splat data.

I want to be sure this reconstruction actually reproduces the same transmittances the forward pass saw, because the gradient is only correct if the T_i it multiplies are bit-for-bit the ones used in compositing — a checkpointing scheme that drifts would silently corrupt every gradient. Let me trace one pixel with four splats, alphas (0.5, 0.4, 0.6, 0.3), and checkpoint every 2nd splat (a toy stand-in for every 32nd). Forward, the recurrence is T₀ = 1, T_{i+1} = T_i(1 − α_i): so T = 1, 0.5, 0.3, 0.12, and I store checkpoints at i = 0 (T = 1) and i = 2 (T = 0.3). Now reconstruct in backward: thread 1 needs T₁, takes the i = 0 checkpoint and steps once, 1·(1 − 0.5) = 0.5; thread 3 needs T₃, takes the i = 2 checkpoint and steps once, 0.3·(1 − 0.6) = 0.12. Lining the reconstructed series (1, 0.5, 0.3, 0.12) against a fresh forward pass (1, 0.5, 0.3, 0.12), the maximum absolute difference is 0.0. The reconstruction is the same recurrence run from a stored waypoint, so it returns exactly the forward state — which is the property the gradient depends on. In practice I checkpoint every 32nd splat in the sorted list during forward, schedule buckets of 32 splats to a CUDA warp in backward, and let the warp shuffle reconstruct its slice of the state table cheaply. The net effect: I exchange a small per-pixel state via registers instead of contending on per-splat memory via atomics — the contention is gone, not just reduced. I also notice the tail of each tile's sorted splat list is often fully occluded — the forward pass already terminates on saturation — so in backward I track the last contributor per tile and skip whole groups of occluded splat–tile pairings, and I apply tighter culling to drop redundant splats from both passes. Because the reconstructed per-pixel state matches the forward state exactly, the gradient this computes is the same one the atomic version computes, just without the bottleneck — and skipping fully-occluded tails drops only terms that contributed nothing.

Then the optimizer step. As the count grows the Adam updates dominate, and the spherical-harmonic coefficients are the culprit — 48 of the 59 per-Gaussian attributes are SH, so they're the bulk of every Adam step. The base color (the zeroth SH band) changes meaningfully every iteration, but the higher bands, which carry the finer view-dependent detail, barely move per step. So I put the higher SH bands on a *batched* update — one Adam step every 16 iterations instead of every iteration — using a separate optimizer for them. This is the one change that isn't strictly equivalent, so I compensate for the rarer updates with a higher SH learning rate and keep the exact-update path available when I want it. Two smaller, fully-equivalent wins ride along: the standard pipeline concatenates the zeroth and higher SH bands into one tensor before rasterization, which costs a surprising slice of the forward pass, so I extend the rasterizer to read DC and higher SH from separate tensors and skip the concat; and the SSIM in the loss uses an 11×11 Gaussian-kernel 2D convolution, but a Gaussian kernel is separable, so I do it as two 1D convolutions and fuse the SSIM evaluation into one kernel — which matters most exactly when the primitive count is low relative to the image resolution, i.e. on a budget.

Let me put the whole thing into the strategy code I'd actually run, filling the empty densification slot in the harness. I keep the running view-space-gradient statistics (the under-sampling signal), I precompute the parabolic target-count array once, and at each densification event I build the score from sampled views, read the per-step budget off the curve, split it between clone and split by population, multinomial-sample candidates, and constructively trim. The score computation gathers per-Gaussian and per-pixel attributes from a render that carries pixel weights through the rasterizer.

```python
import torch


def get_count_array(start_count, budget, num_steps):
    # Parabolic target schedule A(x) with A(0)=S, A(num_steps)=B, A'(num_steps)=0.
    # A(x) = a*x^2 + b*x + c,  a = -(B-S)/N^2,  b = 2(B-S)/N,  c = S.
    S, B, N = start_count, budget, num_steps
    a = -(B - S) / (N * N)
    b = 2.0 * (B - S) / N
    c = S
    return [int(a * x * x + b * x + c) for x in range(num_steps + 1)]  # caller skips x=0


def median_scale(weight, value):
    # Outlier-robust per-attribute normalization: divide positive values by their
    # median, scale by the attribute's weight. (NaNs -> 0, non-positive -> 0.)
    value = torch.nan_to_num(value, nan=0.0).float()
    pos = value > 0
    out = torch.zeros_like(value)
    if pos.any():
        out[pos] = weight * (value[pos] / value[pos].median())
    return out


def compute_gaussian_score(scene, cams, edge_maps, gaussians, renderer, bg, w):
    # Per-Gaussian importance S_g = sum_i P_i * F(per-Gaussian + per-pixel attrs).
    n = len(scene.gaussians.get_xyz)
    score = torch.zeros((len(cams), n), device="cuda", dtype=torch.float32)
    opac = scene.gaussians.get_opacity.detach().squeeze()
    scale = torch.prod(scene.gaussians.get_scaling.detach(), dim=1)
    grads = scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
    grads[grads.isnan()] = 0.0
    grads = grads.detach().squeeze()

    for i, cam in enumerate(cams):
        img = renderer(cam, gaussians, bg)["render"]
        P_i = 0.8 * l1_loss(img, cam.gt) + 0.2 * (1.0 - ssim(img, cam.gt))
        l1m = (img - cam.gt).abs().mean(0).detach()
        l1m = (l1m - l1m.min()) / (l1m.max() - l1m.min())
        pixel_weights = w["mse"] * l1m + w["edge"] * edge_maps[i].cuda()
        pkg = renderer(cam, gaussians, bg, pixel_weights=pixel_weights)

        loss_acc = pkg["accum_weights"]
        dist_acc = pkg["accum_dist"]
        blend = pkg["accum_blend"]
        count = pkg["accum_count"]
        vis = pkg["visibility_filter"].detach()
        depth = pkg["gaussian_depths"].detach()
        radii = pkg["gaussian_radii"].detach()

        g_part = (median_scale(w["grad"], grads) + median_scale(w["opac"], opac) +
                  median_scale(w["dept"], depth) + median_scale(w["radii"], radii) +
                  median_scale(w["scale"], scale))
        p_part = (median_scale(w["dist"], dist_acc) + median_scale(w["loss"], loss_acc) +
                  median_scale(w["count"], count) + median_scale(w["blend"], blend))
        score[i][vis] = (w["view"] * P_i * (g_part + p_part))[vis]
    return score.sum(dim=0)


class GaussianModel:
    # ... existing 3DGS tensors, optimizers, clone/split/prune helpers ...

    def add_densification_stats(self, viewspace_points, update_filter):
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

        prune = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            prune |= self.max_radii2D > max_screen_size
            prune |= self.get_scaling.max(dim=1).values > 0.1 * extent
        remove_budget = int(0.5 * int(prune.sum()))
        if remove_budget and iter_num < 27:
            inv = torch.zeros(self.get_xyz.shape[0], dtype=torch.float32, device="cuda")
            inv[:scores.shape[0]] = 1.0 / (1e-6 + scores.squeeze())
            pick = torch.zeros_like(inv, dtype=bool)
            pick[torch.multinomial(inv, remove_budget, replacement=False)] = True
            self.prune_points(prune & pick)
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def modify_functions(self):
        old_opacities = self.get_opacity.clone()
        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = identity_gate
        self._opacity = self.opacity_activation(old_opacities)
```
