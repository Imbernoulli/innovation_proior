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

Let me make that feedback loop concrete, because it is the whole reason boundedness is not optional. Say
the last linear map's weights drift up by a factor `α`, so every latent `z` becomes `αz`. The consistency
term `‖d(z,a) − sg(h(s'))‖²` scales as `α²` — both the prediction and the (stop-gradient) target inflate,
so the *residual* inflates too, and the gradient it sends back into the weights scales with `α` as well.
The value target `q = r + γ Q̄(z', p(z'))` reads `αz'`, and if `Q̄` is even locally linear in its input
its output moves with `α`, so the regression target the network is chasing is itself a function of the
inflation it just produced. That is a positive-feedback triangle: larger latents → larger targets →
larger gradients → larger weights → larger latents. Nothing in the loss opposes it, because scaling
everything together leaves every *relative* comparison unchanged; the loss is blind to the one direction
that is running away. A bound on the latent magnitude breaks the triangle at its base — if `‖z‖` is
pinned no matter how big the weights get, the weights have nothing to gain by growing, and the loop is
starved. So whatever final activation I pick, the non-negotiable property is that it caps `‖z‖`
independently of the upstream weights.

Why doesn't the LayerNorm already inside the MLP stacks save me? Because LayerNorm normalizes the
*intermediate* pre-activations flowing through each layer — it fixes the per-layer mean and variance of
the river — but the network's *output*, the final latent fed to dynamics, reward, and value, is produced
by a last linear map *after* the last LayerNorm, and that map can have arbitrarily large weights.
LayerNorm controls the river, not the mouth. The thing I actually consume is unbounded. So I need a
constraint on the *final* layer's output, a final activation that defines the geometry of the latent
itself. That is the editable slot the task hands me: `CustomSimNorm.forward`, inserted at the output of
both the encoder and the dynamics MLP. The contract is narrow — take `(*batch, 128)`, return the same
shape — and everything else is frozen, so my whole design space is "what map do I apply to the raw
latent."

Before I commit to a family, let me actually walk the candidates the setup makes available, because
"bound the latent" admits several genuinely different answers and I want the one whose failure modes I can
live with. The do-nothing option, identity, is out on the argument I just made: it leaves the runaway
direction wide open, so it is a control, not a contender. Discrete latent codes are the opposite extreme:
split each group into a one-hot drawn from a learned codebook. That gives a hard bound (a one-hot has
norm one) and maximal within-group sparsity — attractive on paper — but the argmax that selects the code
is non-differentiable, so it demands a straight-through estimator plus a commitment loss to keep the
encoder near the codebook, and it drags codebook collapse and dead codes along for the ride. In a
*recurrent* world model the latent is fed back through the dynamics net rollout after rollout, so gradient
has to flow cleanly through many applications of this activation; a straight-through gap that is a small
bias per step compounds across the rollout, and a collapsed codebook silently caps capacity. That fragile
machinery is the wrong tool for the first rung, where I want to isolate the effect of *bounding* from the
effect of any exotic estimator. I set it aside deliberately.

That leaves the two smooth rescalings — L2 normalization onto a hypersphere, and RMSNorm — and I should
be honest that these are close cousins, because the arithmetic makes the relationship exact. L2
normalization sends `z ↦ s · z/‖z‖₂` with a single learnable radius `s`. RMSNorm divides by the
root-mean-square, and for an `n`-vector `RMS(z) = √((1/n)Σzᵢ²) = ‖z‖₂/√n`, so `z/RMS(z) = √n · z/‖z‖₂`.
Full-vector RMSNorm is therefore *L2 normalization scaled by `√n`, with a per-coordinate gain replacing
the single scalar radius*. They bound magnitude the same way; the only real difference is that L2 hands
back one shared scale knob while RMSNorm hands back a gain vector, one knob per coordinate, so the network
can re-weight directions rather than only the overall radius. Given that the intermediate LayerNorms have
already committed this model to a per-coordinate gain style, matching that with RMSNorm's gain is the
less arbitrary choice, and it keeps the layer inside the LayerNorm lineage I understand rather than
introducing a bespoke single-radius sphere. So the family that best fits "cheapest disciplined bound that
lives in the same idiom as the rest of the net" is RMSNorm, and the L2 sphere is the near-equivalent I
pass over precisely because it is near-equivalent but off-idiom.

The simplest disciplined answer in the "normalize the magnitude" family is to reach for the cheapest
normalization I trust. I have a long-standing intuition about LayerNorm: when I train normalized
networks I get a big per-step improvement, but I pay for it on every forward pass, and a good chunk of
the per-step gain gets handed back in wall-clock. So I have asked before: am I paying for something I do
not need? LayerNorm bundles two distinct operations. One: subtract the mean — recenter the vector to
zero mean. Two: divide by the standard deviation — rescale it to unit spread. They feel like one move,
"standardize," but they are two, and they buy *different* things. Subtracting the mean buys
re-centering invariance: add the same constant to every entry and the centered vector is unchanged.
Dividing by σ buys re-scaling invariance: scale every entry by α and `(a−μ)/σ` is unchanged. Two
operations, two invariances, cleanly separated.

The mean is the suspicious one. Stabilizing training is fundamentally about controlling *spread* —
keeping activations and gradients from blowing up or vanishing. Subtracting the mean recenters the
vector but does nothing to its variance: `var(a−μ) = var(a)`. Recentering moves the cloud; it does not
shrink or grow it. The thing that actually controls spread — that pulls the magnitude to a fixed scale
regardless of how big the weights got — is the *division by a scale statistic*. So if I have to bet
which of the two invariances is doing the stabilizing work, I bet on re-scaling and treat re-centering
as dead weight. That is exactly the degree of freedom I diagnosed as the source of the latent's runaway:
its overall scale. The re-scaling normalization is precisely the operation that pins it. So in *this*
model the part of LayerNorm I do not need is the part I was already suspicious of, and the part I do need
is the part that solves the world-model's actual problem.

If I drop the mean, what do I divide by? I cannot use σ, because `σ = √((1/n)Σ(aᵢ−μ)²)` is *defined*
through μ — it is the spread *around the mean*. If I refuse to compute μ, I need a measure of scale that
never references it. The natural candidate is the spread around *zero* instead of around the mean: the
root-mean-square, `RMS(a) = √((1/n)Σ aᵢ²)`. This is the quadratic mean of the entries — how big they
are on average, measured from the origin — and the normalization becomes `āᵢ = aᵢ / RMS(a) · gᵢ` with a
learned gain `g`. No μ anywhere, no subtraction, one statistic. Check the degenerate case: if the
entries happen to have zero mean, `σ = RMS(a)` and the rule coincides exactly with LayerNorm — so this
is not a wild departure, it is the same layer with re-centering switched off. And check that it does the
one job I need: `RMS(αa) = |α|·RMS(a)`, so `αa / RMS(αa) = a / RMS(a)` — re-scaling invariance, the
load-bearing one, survives. Re-centering invariance is the only thing discarded, and I just argued it is
the weaker suspect. This is the right tool for bounding a self-predictive latent's magnitude with the
least machinery.

Now I have to fit it to *this* task's edit surface, and this is where the implementation deviates from
the textbook RMSNorm in a way I want to be honest about. The default fill is SimNorm, which reshapes the
128-dim latent into 16 groups of `simnorm_dim = 8` and softmaxes within each group. The whole harness —
the consistency loss, the heads — is built around a latent that has this group structure as its native
shape. The cleanest way to slot RMSNorm into the same slot, and the version that is a true
apples-to-apples swap against the default, is to keep the *same partition* and apply RMS normalization
*within each group of 8* rather than over the full 128-vector. So: reshape to `(*batch, 16, 8)`, compute
the root-mean-square over the last axis (the 8 entries of each group), divide, and multiply by a
learnable gain of size 8 shared across groups, then reshape back. This is **group-wise** RMSNorm, not
the canonical full-vector RMSNorm, and the difference is deliberate: it normalizes each of the 16 groups
to a common magnitude independently, mirroring SimNorm's group structure so the comparison isolates
"softmax-on-a-simplex vs. RMS-rescale" rather than confounding it with "16 groups vs. one block." I keep
the gain per group element (size 8) so each coordinate within a group can be re-weighted, the same role
the learned gain plays in LayerNorm.

Let me walk the shape and the cost so I know exactly what I am inserting. The input arrives as
`(*batch, 128)`; `view(*shp[:-1], -1, 8)` reshapes it to `(*batch, 16, 8)` because `128/8 = 16`, and the
`-1` infers the 16 without my having to name it, which keeps the module agnostic to any latent width that
is a multiple of 8. The mean-square reduction over `dim=-1` with `keepdim=True` produces `(*batch, 16, 1)`,
which broadcasts back against `(*batch, 16, 8)` on the divide — that is the whole reason for `keepdim`,
so I never have to unsqueeze by hand. The gain is a length-8 vector broadcasting across all 16 groups and
all batch dims, so it costs exactly 8 parameters. Against a ~1M-parameter world model that is under one
part in a hundred thousand — the layer is free in parameters, and in compute it is one square, one mean,
one sqrt, one divide, one multiply per element, all elementwise and cheap next to the MLP matmuls it sits
after. There is no reduction over the batch, so it behaves identically at train and eval and needs no
running statistics, unlike BatchNorm — another reason this family is the low-risk pick for a recurrent
model whose batch composition shifts as the replay buffer fills.

Let me hand-trace one group to be sure the algebra and the code agree. Take a group `[2, −2, 1, −1, 0, 0,
0, 0]`. The mean-square is `(4 + 4 + 1 + 1 + 0 + 0 + 0 + 0)/8 = 10/8 = 1.25`, so with `eps` negligible
`rms = √1.25 ≈ 1.1180`. Dividing gives `[1.789, −1.789, 0.894, −0.894, 0, 0, 0, 0]` (before the gain,
which starts at ones). Check the invariant I care about: the mean-square of the output is
`(1.789² + 1.789² + 0.894² + 0.894²)/8 = (3.20 + 3.20 + 0.80 + 0.80)/8 = 8.0/8 = 1.0` — root-mean-square
exactly one, as designed. Now scale the input by `α = 10`: the mean-square becomes `125`, `rms = 11.18`,
and dividing returns the *same* `[1.789, −1.789, …]` — the re-scaling invariance I proved abstractly,
confirmed on a number. And notice what the trace does *not* do: the four zero entries stay exactly zero,
the two `±1.789` and two `±0.894` keep their relative proportions untouched. The map only set the group's
overall size; it did not redistribute mass among the coordinates or push any of them toward zero. That is
the concrete face of "bounds spread, adds no structure," and it is the property I will be watching pay a
price later.

The hand-trace also lets me pin down *why* group-wise and not full-vector, with a number rather than a
slogan. Suppose the raw 128-vector has two groups with very different sizes — say group A is `[3,3,3,3,3,
3,3,3]` (mean-square 9) and group B is `[0.3,0.3,…]` (mean-square 0.09), a hundred-to-one ratio in
energy, which is the kind of imbalance an untamed last linear map produces. Full-vector RMSNorm computes
one `rms` over all 128 entries; if the rest of the vector is order one, that global `rms` is order one,
so after the divide group A stays around magnitude 3 and group B stays around 0.3 — the imbalance
*survives*, because a single global scalar cannot equalize groups that differ from each other. Group-wise
RMSNorm instead divides A by `√9 = 3` and B by `√0.09 = 0.3`, sending *both* to root-mean-square one. So
the group-wise form does not just bound the whole vector; it removes the *inter-group* magnitude
imbalance too, giving every one of the 16 groups the same say in the consistency and value losses. That
is the property I actually want when I insert this into a slot the harness treats as 16 native
sub-blocks: no group is allowed to shout over the others merely because its slice of the last linear map
grew large. Full-vector RMSNorm would leave that failure mode open; group-wise closes it, at the cost of
16 tiny reductions instead of one — a cost I already showed is negligible.

There is one more thing I should check before trusting the bound: that the *gradient* the layer sends
back is itself tame, not just the forward magnitude. The map `x ↦ x/RMS(x)` is scale-invariant, and a
scale-invariant function has a Jacobian whose radial component is zero — moving `x` along its own
direction does not change the output, so the output can only respond to *directional* changes in the
input. Concretely, the derivative of `xᵢ/RMS(x)` picks up a `−xᵢ(x·δ)/(n·RMS³)` term that exactly cancels
any component of the incoming perturbation along `x`. The upshot is that no matter how large the upstream
weights drive `‖x‖`, the gradient flowing back through this layer is normalized by the same `RMS` in the
denominator, so it cannot blow up with the weights the way an unnormalized last layer's would. That is the
same runaway triangle from before, now cut on the backward pass as well as the forward: the forward
magnitude is pinned to one and the backward magnitude is scaled by `1/RMS`, so growing the weights buys
neither a larger latent nor a larger gradient. Both directions of the loop are starved, which is exactly
the guarantee I wanted from a "bound the latent" activation and could not get from identity or from a
LayerNorm buried mid-stack.

A subtlety in the numerics: I put the `eps` *inside* the square root, `rms = √(mean(x²) + eps)`, rather
than clamping the norm from below afterward. With groups of only 8 entries the mean-square can get small,
and adding eps under the root keeps the division well-conditioned without ever producing a zero
denominator; it also keeps the gradient of the sqrt finite at the origin — `d/dx √x = 1/(2√x)` blows up
as `x→0`, and the `+eps` caps that derivative at `1/(2√eps)` instead of infinity. I use `eps = 1e-8`.
To see that it is small enough not to bias a real group: at initialization the pre-activations feeding the
last linear map are LayerNorm-controlled to order one, so a group's mean-square is order one too, and
`1e-8` is eight orders of magnitude below it — it shifts `rms` by a part in `10⁸`, invisibly. It only
becomes load-bearing exactly when a group collapses toward the origin, which is the one case where I want
a floor rather than a division by near-zero. Clamping *after* the divide would instead leave the sqrt
gradient singular at the origin and only patch the denominator, so putting eps under the root is the
choice that fixes both the value and the gradient at once. The gain initializes to ones so the layer
starts as a pure RMS rescale and learns any per-coordinate emphasis from there.

Let me be clear-eyed about what this buys and what it does *not*, because the gap is the whole reason
this is the first rung and not the last. RMSNorm bounds the *spread* of each group — after the divide,
each group has root-mean-square one (up to the gain) — so the magnitude degree of freedom that drives the
runaway loop is controlled, and gradients should stay tame. That is real and it is the point. But
RMSNorm does *nothing else*. It does not induce sparsity: every coordinate is generically nonzero, the
group is a bounded but dense blob. It does not induce any competition between coordinates: each entry is
rescaled by the same scalar, so there is no pressure to prioritize a few directions. And the learnable
gain can partially re-inflate the scale per coordinate — if a coordinate's gain learns its way up to,
say, ten, that coordinate's post-norm magnitude is ten times what the RMS divide left it, so the bound I
imposed is soft, a leash rather than a wall. For a latent that has to support stable bootstrapped value
learning, a bounded-but-shapeless code is exactly the weak representation I worried about in the lineage:
the value head reads a dense vector with no structure to exploit. I expect that to show up not on the
easy tasks — where almost any bounded latent saturates the reward — but on the one task whose dynamics
are rich enough that the latent geometry matters.

So here is where I think this lands and what I am watching, stated as falsifiable expectations. On
walker-walk and cartpole-swingup I expect RMSNorm to sit right at the top of the reward range, near
whatever the other normalizations achieve — these tasks do not stress the latent, so bounding the
magnitude is enough and the lack of structure costs nothing. The honest test is cheetah-run, where the
running gait is a richer, more dynamic behavior and the value head has to read fine structure out of the
latent. There I expect group-wise RMSNorm to be the *weakest* of the disciplined normalizations,
precisely because it bounds magnitude without shaping the code — a dense rescaled latent gives the value
head less to work with than a sparse, competitive one would. If that is what the numbers say — saturated
on the two easy tasks, visibly lower on cheetah-run — then the diagnosis for the next rung writes
itself: bounding the magnitude is necessary but not sufficient, and the missing ingredient is
*structure* — a competition between coordinates that biases the latent toward a sparse, overcomplete
code the readouts prefer. That is exactly the move from a plain rescale to a within-group softmax. But I
do not get to assume that; I run RMSNorm first and let the cheetah-run number tell me how much the
missing structure actually costs. The full scaffold module — the group-wise RMS rescale with the
per-group gain and the eps-under-root — is in the answer.
