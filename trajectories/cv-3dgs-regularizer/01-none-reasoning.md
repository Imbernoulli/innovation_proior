The regularizer is the whole point of this task, but it bolts onto an optimization that already runs,
and with no regularizer that optimization is the floor — so the first thing I have to establish, before
I add any preference of my own, is exactly what the pure data term gives me on these scenes and exactly
*why* it leaves room for a preference at all. Everything around the regularizer is fixed and I cannot
touch it: the `gsplat` CUDA rasterizer projects each Gaussian to its analytic 2D footprint and
composites a sampled training view front-to-back; the loss is `0.8·L1 + 0.2·(1−SSIM)` against the
photograph; `DefaultStrategy` clones, splits, and prunes the Gaussians; per-parameter Adam steps every
quantity with the canonical learning rates and an exponential decay on the means; thirty thousand steps
per scene, SH degree grown one band at a time. The single empty slot is `compute_regularizer`, and at
this rung I leave it exactly as the scaffold hands it to me — it returns `torch.zeros(())`, so the loop
adds zero to the photometric loss and the Gaussians are supervised by the photographs and nothing else
(the distilled module is in the answer). The reason to run this first is not ceremony: it is the only
way to *measure* where the missing preference actually costs PSNR, scene by scene, so that the term I
design next is aimed at the real headroom rather than at a failure mode I merely imagine.

Let me be precise about why this configuration is the weakest *by construction*, because that argument
is simultaneously what makes it the right place to start and what dictates what the next rung must do.
The photometric loss is a pure data term. It rewards any arrangement of Gaussians whose rendered
training views match the photographs, and it is completely indifferent to *how* that arrangement is
built — to the shapes, sizes, opacities, or spatial layout of the primitives that produce those pixels.
Now think about how many degrees of freedom there are. Each Gaussian carries a 3-vector mean, three
log-scales, a four-component quaternion, a logit opacity, and a stack of spherical-harmonic
coefficients; a scene holds on the order of a million of them. The number of free parameters dwarfs the
number of pixel constraints from the held-out-free training views, and the image-formation model is a
*sum* over overlapping, semi-transparent primitives — `C = Σ_i c_i α_i Π_{j<i}(1−α_j)` — so the same
pixel can be explained by a single well-placed Gaussian, by two faint ones that happen to composite to
the same color, by a large diffuse one tinting a whole region, or by a cluster of slivers. The fit is
therefore under-determined in a strong sense: a vast equivalence class of Gaussian configurations
renders the training views to within a hair of one another, and gradient descent does not select the
"nicest" member of that class — it settles into whichever member it first descends into from the SfM
initialization. There is no term anywhere in the objective that says "among all configurations that fit
the photos, prefer the one built from compact, well-shaped, parsimonious Gaussians." That missing
preference is the entire opening for a regularizer, and naming it precisely is the work of this rung.

What does the optimizer actually stumble into, concretely? The configurations it finds are the wasteful
ones, and they come in a small taxonomy I want fixed in my head because each later rung will attack one
branch of it. First, *floaters*: faint, near-transparent Gaussians parked in mid-air just outside where
any camera looks closely — in the volume between the camera path and the background, in foliage gaps, in
the sky. They cost almost nothing on any single training view (low opacity, seen edge-on or barely seen)
but they read as haze or specks the moment the camera moves to a held-out pose. Second, *needles*:
hugely anisotropic Gaussians with one variance enormous and the other two collapsed almost to zero. A
forest of needles can tile a training image acceptably — each one shaves a little error along its long
axis on the view that produced it — but rotate to a novel view and each needle is a spike sticking out
of the surface. Third, *over-reconstruction*: one oversized splat smeared across fine geometry it cannot
resolve, because for the data term a single broad primitive that is "mostly right" over a region is
cheaper than the work of subdividing it. All three are local minima of a pure data term, and all three
are invisible to it precisely because they were *selected* for being cheap on the training views. They
are not bugs in the optimizer; they are the optimizer doing exactly what the objective asks.

The obvious objection is that the fixed `DefaultStrategy` already adds and removes Gaussians, so why
doesn't it clean these up on its own? It is worth working through, because the answer is what tells me a
regularizer is *needed* and not redundant. Densification is the only mechanism in the whole loop that
changes the *number* of primitives — the photometric gradient can reshape, recolor, move, and fade the
Gaussians that exist, but it has no move that creates one where a region is empty or deletes one that is
junk. So `DefaultStrategy` interleaves with optimization: it clones small Gaussians under large
view-space positional gradient (under-reconstruction — the region wants more coverage), splits large
ones under the same signal (over-reconstruction — the region wants more resolution), and prunes
Gaussians whose activated opacity falls below a small threshold (around 0.005) or whose scale has grown
too large in world or screen space; it even periodically resets opacity downward to force every Gaussian
to re-justify itself. But notice what its trigger is: *view-space positional gradient* and the prune
*thresholds*. It reacts to where the loss is pulling a primitive's position, and to opacity/size crossing
fixed cutoffs. It has no signal that says a Gaussian's *shape* is a needle, or that a faint
region-spanning splat is wasteful while it sits just *above* the prune threshold. The pure data term
never pushes those pathological Gaussians across the cutoffs the strategy is watching, so the strategy
grows and prunes the population without stopping the optimizer from re-forming the same needles and
floaters in the slack the photos leave. Densification manages the count; it does not express the missing
preference. That is exactly the gap a parameter regularizer is asked to fill — and confirming the gap is
real, rather than already closed by the strategy, is part of what this floor rung establishes.

So my edit at step 1 is the trivial one, and deliberately so: leave the function at its scaffold
default, returning a scalar zero on the Gaussians' device. No constants, no schedule, no penalty — the
honest lower bound that every later rung must beat, on every scene, at once.

Now I reason about what this floor should *do* across the four reported scenes, because predicting the
per-scene shape in advance is how I turn the measurement into a diagnosis rather than a number. The data
term will fit the training views well everywhere — that is what it is good at — so the question is only
where the under-constraint hurts the *held-out* views most, and that is governed by how completely each
scene is observed. The three unbounded outdoor scenes — garden, bicycle, stump — have large volumes seen
by few cameras: distant background that recedes toward infinity, ground far from the camera path, dense
foliage whose interior no view penetrates. Over those volumes the data term is flat, the equivalence
class of fits is widest, and the optimizer has the most freedom to leave floaters and needles that cost
nothing on training views and bleed on novel ones. I therefore expect those scenes to sit lower in
absolute PSNR and to carry the most headroom for any regularizer. Among them I expect bicycle the lowest
and the most under-observed — a large scene with a thin camera orbit and a lot of weakly-seen background
and ground — so if the missing-preference diagnosis is right, bicycle should be the loudest evidence for
it. The indoor scene — bonsai — is the opposite: a compact, fully-observed room where nearly every
surface is seen from many angles, so the data term pins the Gaussians tightly, the equivalence class is
narrow, and there is little floater slack for a parsimony prior to reclaim. I expect bonsai highest in
absolute PSNR and, importantly, the scene with the *least* room for a size-and-count regularizer to
help — a prediction I will only be able to test once I have a real regularizer to compare against this
floor. stump I expect between the two extremes among the outdoor scenes, an unbounded scene but with a
dominant near-field subject. (stump is the held-out scene during development; I reason about it the same
way and only see its number at scoring.)

Whatever the exact per-scene split turns out to be, the diagnosis is already pointed at the next rung,
and the floor is what will let me check it. This is not a learning-rate problem and not a densification
problem — the substrate is the converged 3DGS recipe and it is fixed. It is a *missing-preference*
problem: the objective has no way to say "use Gaussians efficiently." The cheapest preference the data
term cannot express, and therefore the natural first thing to add, is to drive the Gaussians the photos
are indifferent to toward not existing — turning the empty `compute_regularizer` into a real
parameter-level penalty on the quantities that make a Gaussian wasteful. The numbers from this floor —
per scene, with the unbounded outdoor scenes carrying the headroom and bonsai carrying the least — are
the bar that the first real regularizer has to clear on every scene at once, and the per-scene gaps are
what will tell me whether the term I add is reaching the slack I predicted or missing it.
