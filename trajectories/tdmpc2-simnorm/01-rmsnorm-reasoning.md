Let me start from what the latent world model actually asks of its final activation, because the
choice is not cosmetic — the latent is the only thing the model has. There is no decoder. The encoder
sends an observation to a latent, the dynamics net predicts the next latent from the current latent and
the action, and reward and value heads read off the latent. The losses that define the latent are
self-referential: a self-predictive consistency term `‖d(z,a) − sg(h(s'))‖²` with a stop-gradient on the
target; a reward regression; and a bootstrapped value loss whose target `q = r + γ Q̄(z', p(z'))` is
itself a function of the very latents the network produces. So the latent is pinned by nothing external,
and that is precisely where instability lives. A squared-error-in-feature-space consistency loss has a
cheap, useless way to go down that has nothing to do with learning dynamics: change the *scale* of the
representation. If the last linear map can inflate or deflate the magnitude of the latent freely, the
losses get a free parameter that moves the landscape around without encoding anything — and because the
value target reads the network's own latents, this closes a feedback loop with a runaway direction. The
exploding gradients people see on harder tasks are that magnitude blowing up. So the final activation's
first job is structural: remove the magnitude degree of freedom, bound the latent.

The runaway is what makes boundedness non-negotiable. Say the last
linear map's weights drift up by a factor `α`, so every latent `z` becomes `αz`. The consistency term
scales as `α²` — both prediction and stop-gradient target inflate, so the residual inflates and the
gradient it sends back scales with `α`. The value target `q = r + γ Q̄(z', p(z'))` reads `αz'`, and if
`Q̄` is even locally linear its output moves with `α`, so the target the network chases is a function of
the inflation it just produced: larger latents → larger targets → larger gradients → larger weights →
larger latents. Nothing in the loss opposes it, because scaling everything together leaves every
*relative* comparison unchanged — the loss is blind to the one direction that runs away. Pinning `‖z‖`
independently of the upstream weights breaks the triangle at its base: the weights have nothing to gain
by growing. So whatever final activation I pick, the non-negotiable property is that it caps `‖z‖`
regardless of the upstream weights.

Why doesn't the LayerNorm already inside the MLP stacks save me? Because it normalizes the *intermediate*
pre-activations flowing through each layer — it fixes the per-layer mean and variance of the river — but
the network's *output*, the final latent fed to dynamics, reward, and value, is produced by a last linear
map *after* the last LayerNorm, and that map can have arbitrarily large weights. LayerNorm controls the
river, not the mouth. The thing I actually consume is unbounded. So I need a constraint on the *final*
layer's output — a final activation that defines the geometry of the latent itself. That is the editable
slot: `CustomSimNorm.forward`, inserted at the output of both the encoder and the dynamics MLP, taking
`(*batch, 128)` and returning the same shape, with everything else frozen.

"Bound the latent" admits several genuinely different answers, so let me weigh them. Identity is out on
the argument above — it leaves the runaway direction wide open, a control not a contender. Discrete latent
codes are the opposite extreme: split each group into a one-hot from a learned codebook, giving a hard
bound and maximal sparsity — but the argmax is non-differentiable, so it demands a straight-through
estimator plus a commitment loss, and drags codebook collapse and dead codes along. In a *recurrent*
world model the latent is fed back through the dynamics rollout after rollout, so a straight-through bias
compounds across steps and a collapsed codebook silently caps capacity. That fragile machinery is the
wrong tool for a first rung where I want to isolate the effect of *bounding* from any exotic estimator, so
I set it aside.

That leaves the two smooth rescalings — L2 onto a hypersphere, and RMSNorm — and the arithmetic makes
them near-cousins. L2 sends `z ↦ s · z/‖z‖₂` with a single learnable radius `s`. RMSNorm divides by the
root-mean-square, and for an `n`-vector `RMS(z) = ‖z‖₂/√n`, so `z/RMS(z) = √n · z/‖z‖₂`: full-vector
RMSNorm is L2 normalization scaled by `√n`, with a per-coordinate gain replacing the single scalar radius.
They bound magnitude identically; the only real difference is that L2 hands back one shared scale knob
while RMSNorm hands back a gain vector, one knob per coordinate. Since the intermediate LayerNorms already
committed this model to a per-coordinate gain style, RMSNorm's gain is the less arbitrary match, and it
keeps the layer inside the LayerNorm lineage rather than introducing a bespoke single-radius sphere. So
the cheapest disciplined bound in the idiom of the rest of the net is RMSNorm; the L2 sphere is the
near-equivalent I pass over precisely because it is near-equivalent but off-idiom.

Reaching for RMSNorm follows a long-standing intuition about LayerNorm: I get a big per-step improvement
but pay for it on every forward pass, so I have asked whether I am paying for something I do not need.
LayerNorm bundles two distinct operations. One: subtract the mean — recenter to zero mean. Two: divide by
the standard deviation — rescale to unit spread. They feel like one move, "standardize," but they buy
*different* invariances: subtracting the mean buys re-centering invariance (add a constant to every entry,
the centered vector is unchanged), dividing by σ buys re-scaling invariance (scale every entry by α and
`(a−μ)/σ` is unchanged).

The mean is the suspicious one. Stabilizing training is about controlling *spread* — keeping activations
and gradients from blowing up or vanishing. Subtracting the mean recenters but does nothing to variance:
`var(a−μ) = var(a)`. The thing that controls spread — that pulls the magnitude to a fixed scale no matter
how big the weights got — is the *division by a scale statistic*. That is exactly the degree of freedom I
diagnosed as the latent's runaway: its overall scale. So the part of LayerNorm I do not need is the part I
was already suspicious of, and the part I do need is the part that solves the world-model's actual problem.

If I drop the mean, what do I divide by? Not σ, because `σ = √((1/n)Σ(aᵢ−μ)²)` is *defined* through μ. I
need a measure of scale that never references μ: the spread around *zero* rather than around the mean, the
root-mean-square `RMS(a) = √((1/n)Σ aᵢ²)`. The normalization becomes `āᵢ = aᵢ / RMS(a) · gᵢ` with a
learned gain `g` — no μ, one statistic. If the entries happen to have zero mean, `σ = RMS(a)` and the rule
coincides with LayerNorm, so this is the same layer with re-centering switched off. And it does the one
job I need: `RMS(αa) = |α|·RMS(a)`, so `αa / RMS(αa) = a / RMS(a)` — re-scaling invariance, the
load-bearing one, survives; only re-centering, the weaker suspect, is discarded.

Now to fit it to this task's edit surface, where the implementation deviates from textbook RMSNorm on
purpose. The default fill is SimNorm, which reshapes the 128-dim latent into 16 groups of `simnorm_dim = 8`
and softmaxes within each group; the whole harness — consistency loss, heads — is built around a latent
that has this group structure as its native shape. The cleanest slot-in, and the true apples-to-apples
swap against the default, is to keep the *same partition* and apply RMS normalization *within each group of
8* rather than over the full 128-vector: reshape to `(*batch, 16, 8)`, compute the root-mean-square over
the last axis, divide, multiply by a learnable gain of size 8 shared across groups, reshape back. This is
**group-wise** RMSNorm, and the difference is deliberate — it mirrors SimNorm's group structure so the
comparison isolates "softmax-on-a-simplex vs. RMS-rescale" rather than confounding it with "16 groups vs.
one block." The gain is per group element (size 8), so each coordinate within a group can be re-weighted.

The cost is negligible: the gain is 8 parameters against a ~1M-parameter model, under one part in a
hundred thousand, and the compute is one square, mean, sqrt, divide, multiply per element — all
elementwise and cheap next to the MLP matmuls it sits after. There is no reduction over the batch, so it
behaves identically at train and eval and needs no running statistics, unlike BatchNorm — another reason
this family is low-risk for a recurrent model whose batch composition shifts as the replay buffer fills.

Group-wise rather than full-vector is not just idiom-matching; it removes an inter-group failure mode.
Suppose the raw vector has group A `[3,…]` (mean-square 9) and group B `[0.3,…]` (mean-square 0.09), a
hundred-to-one energy ratio — the kind of imbalance an untamed last linear map produces. Full-vector
RMSNorm computes one `rms` over all 128 entries, so after the divide A stays near magnitude 3 and B near
0.3: the imbalance *survives*, because a single global scalar cannot equalize groups that differ from each
other. Group-wise instead divides A by 3 and B by 0.3, sending both to root-mean-square one, so it removes
the *inter-group* magnitude imbalance and gives every one of the 16 groups the same say in the consistency
and value losses. That is the property I want when the harness treats the latent as 16 native sub-blocks:
no group can shout over the others merely because its slice of the last linear map grew large. The cost is
16 tiny reductions instead of one — already negligible.

The bound holds on the backward pass too, not just the forward magnitude. The map `x ↦ x/RMS(x)` is
scale-invariant, so its Jacobian has zero radial component: moving `x` along its own direction does not
change the output. The derivative of `xᵢ/RMS(x)` picks up a `−xᵢ(x·δ)/(n·RMS³)` term that cancels any
incoming perturbation along `x`, and the gradient flowing back is normalized by the same `RMS` in the
denominator, so it cannot blow up with the weights the way an unnormalized last layer's would. That is the
runaway triangle cut on both passes: forward magnitude pinned to one, backward magnitude scaled by
`1/RMS`, so growing the weights buys neither a larger latent nor a larger gradient.

One numerics choice: I put `eps` *inside* the square root, `rms = √(mean(x²) + eps)`, rather than clamping
the norm from below afterward. With groups of only 8 entries the mean-square can get small, and eps under
the root keeps the division well-conditioned without a zero denominator; it also keeps the sqrt gradient
finite at the origin — `d/dx √x = 1/(2√x)` blows up as `x→0`, and `+eps` caps that at `1/(2√eps)`. I use
`eps = 1e-8`: at initialization the LayerNorm-controlled pre-activations make a group's mean-square order
one, so `1e-8` shifts `rms` by a part in `10⁸`, invisibly, and only becomes load-bearing exactly when a
group collapses toward the origin — the one case where I want a floor. Clamping *after* the divide would
leave the sqrt gradient singular at the origin, so eps under the root fixes both value and gradient at
once. The gain initializes to ones, so the layer starts as a pure RMS rescale.

I should be clear-eyed about what this buys and what it does *not*, because the gap is the whole reason
this is the first rung and not the last. RMSNorm bounds the *spread* of each group — after the divide,
root-mean-square one up to the gain — so the magnitude degree of freedom that drives the runaway loop is
controlled and gradients should stay tame. That is real and it is the point. But RMSNorm does nothing
else. It induces no sparsity: every coordinate is generically nonzero, the group a bounded but dense blob.
It induces no competition between coordinates: each entry is rescaled by the same scalar, so there is no
pressure to prioritize a few directions. And the learnable gain can partially re-inflate scale per
coordinate, so the bound is soft, a leash rather than a wall. For a latent that has to support stable
bootstrapped value learning, a bounded-but-shapeless code is exactly the weak representation I worried
about: the value head reads a dense vector with no structure to exploit. I expect that to cost nothing on
easy tasks — where almost any bounded latent saturates the reward — and to bite on a task whose dynamics
are rich enough that the latent geometry matters, a richer running gait rather than a swing-up or a walk.

So I run RMSNorm as the first rung and let the numbers say how much the missing structure costs. I do not
get to assume it costs anything — the number decides, and a shortfall showing up only where the dynamics
are rich would be the evidence that bounding magnitude was necessary but not sufficient. The full scaffold
module — the group-wise RMS rescale with per-group gain and eps under the root — is in the answer.
