## Research question

A point-based radiance field is trained by starting from a sparse Structure-from-Motion (SfM) point cloud,
turning each point into an optimizable anisotropic 3D Gaussian, and then *growing* the set during optimization:
new primitives are inserted wherever the current reconstruction is too coarse, and useless ones are removed.
This growth is what lets the representation resolve fine detail. But the growth process in use is an open-loop
heuristic, and that creates a concrete, painful problem: **you cannot control how many primitives you end up with,
and you cannot bound the memory the run will use.**

The symptoms are specific. Starting two scenes from the same number of SfM points, the final Gaussian count can
differ by an order of magnitude; a single unbounded scene can converge to several million Gaussians and more than a
gigabyte on disk. The growth schedule over-produces and then prunes, so the *peak* primitive count during training
is much larger than the *final* count — and it is the peak that determines whether the run fits in GPU memory at all.
On a constrained device, that unpredictable peak means a run may simply fail to complete. Training time fluctuates
strongly for the same reason. And any downstream consumer that needs a *fixed-size* input (a classifier over the
primitive set, a streaming budget, a fixed VRAM envelope) cannot use a representation whose size it cannot predict.

The precise goal is an optimization that simultaneously: (1) reaches an *exact, user-specified* number of Gaussians B,
deterministically, regardless of scene; (2) reaches it *constructively* — the primitive count rises smoothly toward B
without a large over-grow-then-prune peak, so peak memory ≈ final memory and never spikes past the device budget;
(3) spends that fixed budget *well*, putting primitives where they most raise reconstruction quality, so the fixed-size
model does not waste capacity; and (4) trains faster, since the per-iteration cost is dominated by a backward pass and
optimizer updates that scale with the primitive count. The existing growth heuristic achieves none of these directly.
Closing that gap is the problem.

## Background

By this time the dominant point-based radiance-field method represents a scene as a set of anisotropic 3D Gaussians,
each with a mean μ, a covariance Σ = R S Sᵀ Rᵀ (rotation R, scaling S), an opacity o, and view-dependent color via
spherical harmonics (SH) of order 3 plus a direct color term. Its theoretical per-point contribution is

```
G(x) = o · exp( -½ (x - μ)ᵀ Σ⁻¹ (x - μ) ).
```

For a viewpoint, the visible Gaussians are projected to 2D "splats" and composited front-to-back by α-blending in a
tile-based, differentiable rasterizer. Training minimizes a photometric loss against ground-truth images,
`L = (1 - λ) L1 + λ (1 - SSIM)` with `λ = 0.2`, optimizing μ, R, S, o and the SH coefficients of every Gaussian by Adam.

The load-bearing piece for this problem is the **growth / adaptive density control** that runs alongside optimization.
The prevailing recipe is open-loop and threshold-based, and it carries several pain points that motivate everything
that follows:

- **Positional gradient as the only densification signal.** The view-space gradient of a Gaussian's screen position,
  averaged over the views it appeared in, is treated as a "3D discontinuity detector": a high `‖∇‖` is read as "this
  region is under-sampled, add detail here." It is effective but blunt — it is a single scalar that ignores how much a
  Gaussian actually contributes to the image, how large its projection is, whether it is a near-transparent floater
  the optimizer is already phasing out, or whether the error it sits on is genuine under-sampling versus a misplaced
  primitive. Relying on it alone is documented to be wasteful, producing superfluous primitives.

- **Open-loop count.** Densification fires at fixed intervals and acts on *every* Gaussian past a fixed gradient
  threshold; there is no notion of a target count, so the number added at each stage — and the final total — is an
  emergent property of the scene and the threshold, not a quantity the user sets.

- **Grow-then-prune peaks.** Because the heuristic over-produces and then removes low-opacity primitives (and resets
  opacities periodically to force pruning), the trajectory of the primitive count overshoots: peak ≫ final. This is the
  direct cause of the unpredictable memory spikes.

- **Per-iteration cost scales with the primitive count, and the backward pass dominates.** Profiling the standard
  pipeline shows gradient backpropagation is the largest per-iteration cost, followed by the Adam optimizer step as the
  primitive count grows. Within the optimizer step, the SH coefficients — 48 of the 59 optimized attributes per
  Gaussian — account for the bulk of the work.

Two further pre-method observations matter. First, an empirical regularity in the standard method: across outdoor
scenes, the number of Gaussians added per densification step falls over training as progressively fewer primitives
qualify. The total-count curve therefore rises quickly early and flattens near the end. Second, a structural fact about
the backward pass: it maps GPU threads to
*pixels* and has each thread, walking the depth-sorted splats, *atomically* add its gradient contribution into the
splat's accumulated gradient. Many threads hit the same splat's memory at once, so the atomics serialize — the
contention, not the arithmetic, is the bottleneck, and it is aggravated because each splat carries many attribute
gradients.

A second framing is also in the air. The optimized Gaussian set can be viewed probabilistically: treat the current set
as samples from a distribution tied to the loss, in which case the ordinary gradient step plus injected noise is a
Stochastic Gradient Langevin Dynamics update, and densification/pruning become state transitions of that sample set
(Kheradmand et al. 2024, arXiv:2404.09591). In that view, adding a primitive is a *sampling* operation, and it is
natural to let the image loss say *where* to sample.

A third, older body of work is relevant by analogy. Interpreting Gaussian means as points in space, "high quality at a
low primitive count" is essentially **point-cloud downsampling**: choosing a compact yet salient subset of points.
Classical and learned approaches — resampling by local density (Goldberger et al. 2004; Plötz & Roth 2018), learning a
task-specific sampler (Dovrat et al. 2019), critical-point layers that pass the most significant points forward
(Nezhadarya et al. 2020), Gumbel subset sampling (Yang et al. 2019), and differentiable point sampling via softened
nearest-neighbor projection (SampleNet, Lang et al. 2020) — all frame the problem as scoring points by importance and
keeping a budget of the salient ones.

## Baselines

These are the prior methods a controllable, budgeted optimizer would be measured against and would react to.

**The threshold-based adaptive density control of standard 3DGS (Kerbl et al. 2023).** Every fixed interval (100
iterations), compute each Gaussian's view-averaged view-space positional gradient. If `‖∇‖` exceeds a threshold
`τ_pos = 0.0002`, densify it: *clone* it if it is small (max scale ≤ `percent_dense · scene_extent`), or *split* it
into N = 2 children sampled from its covariance, each with scale divided by `0.8·N`, if it is large. Separately, prune
Gaussians whose opacity falls below a small threshold, and additionally prune those with too-large screen-space radius
or world-space scale; periodically (every 3000 iterations) reset all opacities to a small value to force the pruning of
redundant primitives. Core idea: positional gradient localizes under-reconstruction; clone/split adds detail there;
opacity-driven pruning removes the dead weight. **Gap:** every quantity here is a fixed threshold with no target — the
count is an uncontrolled emergent number that varies by up to an order of magnitude between scenes; the single
gradient signal over-densifies and leaves redundancy; and the over-grow-then-prune behavior produces a peak primitive
count far above the final one, so peak memory is large and unpredictable and can exceed a constrained device's budget.

**Reframing optimization as sampling (Kheradmand et al. 2024, 3DGS-MCMC).** Treat the Gaussian set as MCMC samples from
a loss-tied distribution; the standard update plus noise is an SGLD step; densification and pruning are rewritten as
deterministic state transitions, with cloning replaced by a *relocation* that approximately preserves the sample
probability, plus a regularizer encouraging removal of unused Gaussians. Core idea: densification is a sampling /
exploration operation, and the loss landscape should drive it. **Gap:** it gives easier control over the number of
Gaussians and a principled view of densification, but it does not, on its own, deliver an *exact* user-set budget hit
along a smooth, peak-free schedule, nor a multi-signal notion of which primitives are worth sampling beyond the
loss/relocation mechanics.

**On-disk compression of trained models (Compact-3DGS / R-VQ, Lee et al. 2023; Compressed-3DGS, Niedermayr et al. 2023;
LightGaussian, Fan et al. 2023; memory-footprint reduction, Papantonakis et al. 2024).** Cluster and codebook-index the
Gaussian parameters, prune low-significance primitives by volume/opacity, quantize and entropy-code. Core idea: shrink
the storage of an *already-trained* model. **Gap:** these act after the fact and do little to make the optimization
itself *controllable* — they neither bound the training-time peak count nor target an exact size, and the primitive
reductions they achieve are typically modest (≈2×). They are complementary to, not a substitute for, controlling growth.

**Limited-count splatting concurrent with this line (Mini-Splatting, Fang & Wang 2024).** Constrain the number of
Gaussians via blur-aware splitting and importance-weighted stochastic sampling. **Gap:** it, like the standard method,
heavily over-samples the scene before pruning back down, so the gap between peak and final count remains large — the
peak-memory problem is not solved.

**Backward-pass acceleration (DISTWAR, Durvasula et al. 2023, arXiv:2401.05345).** Diagnose the rasterizer backward as
atomic-reduction-bound: most threads in a warp atomically update the same memory, overwhelming the atomic units. Speed
it up with warp-level reduction in registers plus distribution between the SM and L2 atomic units. **Gap:** it reduces
the *cost* of the atomics but keeps the per-pixel parallelization and the atomic-reduction structure itself; the
contention is mitigated, not removed. A separate line (StopThePop, Radl et al. 2024) reduces redundant splats via
tighter culling for sorted rendering.

## Evaluation settings

The natural yardsticks already in use for point-based radiance fields:

- **Datasets:** MipNeRF360 (Barron et al. 2022; 9 indoor/outdoor unbounded scenes), Tanks&Temples (Knapitsch et al.
  2017; 2 scenes), and Deep Blending (Hedman et al. 2018; 2 scenes). These span bounded indoor and unbounded outdoor
  captures with detailed backgrounds.
- **Protocol:** the standard 3DGS train/test split (every 8th image held out for testing), trained for a fixed number
  of steps, SH degree raised gradually during training, optimized by Adam, under the fixed photometric loss
  `0.8·L1 + 0.2·(1 - SSIM)` and the fixed differentiable rasterizer.
- **Quality metrics:** PSNR (primary), SSIM (higher is better), and LPIPS (Zhang et al. 2018; lower is better).
- **Resource metrics:** wall-clock training time, the *final* number of Gaussians, and — crucially for this problem —
  the *peak* number of Gaussians reached during training.
- **Budget configurations to sweep:** a reasonable per-scene budget (scaled by scene type, e.g. a multiplier on the SfM
  count), and a configuration set to match a reference final count exactly so that quality is compared at equal size.

## Code framework

The growth/densification logic plugs into the standard point-based-radiance-field training harness that already exists.
The renderer (a CUDA tile-based differentiable rasterizer), Adam with per-parameter learning rates, the photometric
loss, the 30k-step training loop, opacity reset, the gradual SH schedule, and the primitive operations for clone,
split, and prune are all fixed and given. What is not settled — and is exactly what is to be designed — is the policy
that decides, each densification event, which primitives to add or remove and how many. So the substrate is only the
generic machinery plus an empty growth-policy slot.

The harness already keeps view-space positional-gradient statistics after each backward pass and exposes the existing
Gaussian-set operations:

```python
class GaussianModel:
    # Existing optimized tensors: xyz, scaling, rotation, opacity, SH, max_radii2D.
    # Existing optimizers: one for xyz/scaling/rotation/opacity/DC color and one for higher SH.

    def add_densification_stats(self, viewspace_points, update_filter):
        self.xyz_gradient_accum[update_filter] += norm(
            viewspace_points.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_clone(self, selected_mask):
        # Existing operation: duplicate selected small Gaussians in place.
        ...

    def densify_and_split(self, selected_mask, N=2):
        # Existing operation: sample children from the selected Gaussians' covariance,
        # shrink their scale, append them, and remove the selected parent.
        ...

    def prune_points(self, prune_mask):
        # Existing operation: remove selected Gaussians and optimizer state.
        ...

    def growth_policy(self, rendered_views, radii, step, extent):
        # TODO: the policy we will design.
        # Given the gradients, rendered views, radii, and per-primitive attributes,
        # decide what to add and what to remove this event.
        ...


# existing training loop the strategy plugs into
def train(scene, renderer, optimizer, strategy, n_steps=30_000):
    for step in range(n_steps):
        cam = scene.sample_camera()
        info = renderer(scene.gaussians, cam)          # forward: rasterize splats
        loss = 0.8 * l1(info["image"], cam.gt) + 0.2 * (1 - ssim(info["image"], cam.gt))
        loss.backward()                                 # fills means2d.grad, etc.
        scene.gaussians.add_densification_stats(info["means2d"], info["visibility"])
        if should_run_growth_policy(step):
            scene.gaussians.growth_policy(sample_rendered_views(), info["radii"], step, scene.extent)
        optimizer.step()
        optimizer.zero_grad()
```

The outer loop supplies the rendered views, the per-primitive attributes, and the view-space gradients;
`growth_policy` is the empty slot.
