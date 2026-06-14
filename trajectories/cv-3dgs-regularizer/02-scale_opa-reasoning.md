The no-regularizer floor came back and it draws the map. Garden 29.067, bicycle 26.641, stump 27.460,
bonsai 32.531 — mean 28.925 — and the per-scene spread is exactly the under-constraint story I expected
to read off it. Bonsai, the fully-observed indoor scene, sits highest at 32.531: when nearly every
surface is seen from many angles, the data term pins the Gaussians tightly and there is little slack for
junk to hide in. The three unbounded outdoor scenes sit far lower — garden in the 29s, stump in the
high 27s, bicycle the lowest at 26.641 — and that is precisely where large volumes are seen by few
cameras, the data term goes flat over foliage and distant background, and the optimizer has the room to
leave floaters and needles that cost nothing on the training views and bleed on the held-out ones. So
the floor confirms the diagnosis I started from: this is a *missing-preference* problem, not a learning
problem, and the headroom is concentrated where the photos under-constrain the fit. The bare optimizer
has no term that says "use Gaussians efficiently," and bicycle's 26.641 is the loudest evidence — the
weakest-observed scene is exactly the one where the absent preference hurts most.

So I want to add the cheapest preference the data term cannot express: when a Gaussian is not earning
its keep, push it toward not existing. Not shrink it a little, not fade it a little — drive it toward
the place where the surrounding machinery can be rid of it. And here I have to ground this in the loop
this task actually fixes, because what "be rid of it" means is set by the densification strategy, which
is `DefaultStrategy` — the original 3DGS clone/split/prune — not any relocation scheme. `DefaultStrategy`
prunes a Gaussian whose activated opacity falls below a small threshold (around 0.005), and also prunes
Gaussians that have grown too large in world space (scale above a fraction of the scene scale) or too
large on screen; periodically it even resets opacity downward to force every Gaussian to re-justify
itself. So the loop already *deletes* primitives that are faint or oversized — but only once they cross
those thresholds, and the pure data term has no reason to push anything across them. If I apply a steady
downward pressure on the quantities those thresholds watch — opacity and extent — then I am not
designing a destructive penalty so much as feeding the prune loop the cases it is waiting for. That
reframing matters: "pushed below threshold" does not mean "lost," it means the strategy reclaims the
capacity, and densification re-spends it where the data term actually wants resolution. So I can afford
to press, because pressing where the photos are indifferent is free, and pressing where they are not is
resisted by the data term.

Now, what is the right quantity to push down? My first instinct is opacity. A Gaussian I want gone
should fade, so penalize opacity, and once it is faint enough it drops below the prune threshold and the
strategy removes it. Concretely, add a small weight times the mean activated opacity. Clean, cheap,
`O(N)`, exactly the kind of term the contract wants. But before I commit I have to stare at whether
"faint" actually means "harmless," because that is the assumption the whole scheme leans on. Go back to
how a Gaussian shows up in a pixel. Its contribution is a *product*: the opacity out front, times a
spatial bump whose width is set by the covariance. The total stuff a Gaussian dumps into the image is
opacity times the size of that bump, and the size of the 3D bump grows with the Gaussian's volume, which
is the product of its per-axis extents. So a Gaussian's real footprint is opacity *and* spatial extent,
together — opacity alone is only half of it.

And now I can see exactly how opacity-only pressure fails, and it fails on the very scenes the floor
told me have the most headroom. Picture one of bicycle's or garden's big space-filling Gaussians — a
large covariance faintly tinting a whole patch of foliage or background — that the optimizer is holding
at an opacity of, say, 0.01, just above the prune threshold. My opacity penalty barely touches it: the
mean activated opacity it contributes is already tiny, and because opacity is stored as a logit and
squashed through a sigmoid, the gradient on the stored parameter is scaled by `o·(1−o)`, which is small
near the threshold, so the push it feels is weak. The data term is happy to keep that one big faint
Gaussian smearing a region, because deleting it would change many pixels by a little. So it sits there,
a loud region-spanning artifact, paying almost nothing to my opacity term and never crossing the prune
threshold — precisely the over-reconstruction floater that costs bicycle its PSNR. Opacity-only pressure
leaves that case completely untouched. And there is a mirror-image failure: a tiny but fully opaque
speck doing real work contributes almost nothing to the image yet my opacity term would actively
penalize it for being opaque and try to fade a Gaussian that was helping. The opacity axis is simply the
wrong single axis. Existence is not opacity; existence is opacity-times-extent.

The fix falls right out of the product. If a Gaussian's footprint is set jointly by opacity and extent,
then to reliably make one disappear I must be able to drive *either* factor toward zero, so I press on
*both*. Penalize opacity so faint-and-useless Gaussians fade across the prune threshold; penalize extent
so big-and-useless Gaussians shrink even when opacity alone would leave them alive — and once a Gaussian
shrinks far enough it stops being a region-spanning artifact, and if it grows the strategy's
oversized-scale prune catches it too. A genuinely unneeded Gaussian now has both factors pushed: opacity
is the explicit death channel into the prune threshold, and scale collapse removes the footprint that
made it harmful in the first place. The big-floater case is covered by the extent term, and the needle
case too, because a needle has at least one huge axis whose extent the term hammers.

Let me make "extent" concrete from the parameters I am handed. The covariance is stored factored as
`Σ = R S Sᵀ Rᵀ` with `S = diag(s₁, s₂, s₃)` the per-axis scales and `R` a rotation. Because `R` is
orthogonal, `Σ = R · diag(s₁², s₂², s₃²) · Rᵀ` is already the eigendecomposition of `Σ` — its
eigenvalues are exactly the squared scales — so `√eigⱼ(Σ) = sⱼ`, the standard deviation, the physical
half-width along axis `j`. The thing I want to shrink is just the scale vector, which the contract hands
me directly (as a log-scale; `exp` recovers it). No eigensolver, no determinant at runtime. The volume
is the product of the three scales and I could penalize that, but a sum of the scales is the cleaner,
more robust choice: it pushes *every* axis down rather than letting the optimizer satisfy a volume
penalty by collapsing one axis to zero while keeping the others huge — which is precisely the needle
shape I am trying to kill. Penalize each axis additively and a needle pays full freight on its long
axis. So: a sum (a mean, normalized) over Gaussians and over their three axes of the activated scale.

Now what *kind* of penalty — what function of opacity and of scale? I want useless Gaussians to go to
*zero*, not hover at small-but-nonzero values forever, because only zero-ish opacity crosses the prune
threshold. That is the textbook split between L1 and L2. With an L2 penalty the gradient in the
penalized value is proportional to the value itself, so it fades as the value goes to zero: it shrinks
things softly but the pressure dies right where I need it most, near the threshold, leaving a haze of
small-but-present Gaussians that never quite get pruned. With L1 the subgradient is the constant sign of
the value: the push toward zero does not weaken just because the value is already small. That is the
sparsity-inducing behavior I want — steady pressure in the physical opacity and scale variables, with
the stored logits and log-scales receiving the smooth chain-rule gradient through their activations. So
L1 on both opacity and scale.

There is a subtlety about *which* number I take the L1 of. The opacity is stored as a logit through a
sigmoid; the scale as a log-scale through an exp. I must not penalize the raw stored parameters —
penalizing a logit toward −∞ just sends opacity to zero with ever-vanishing gradient and no physical
zero inside the raw coordinate, and penalizing a log-scale toward −∞ is the same trap. The quantity that
actually determines existence is the *activated* value: the real opacity `sigmoid(logit) ∈ (0,1)` and
the real scale `exp(log_scale) > 0`. Those are what enter the blending equation, and the prune threshold
itself is checked on `sigmoid(opacity)`, so those are exactly what I want small. And a numerical gift
drops out for free: both activated quantities are *already nonnegative* — sigmoid lands in `(0,1)`, exp
in `(0,∞)` — so the absolute value in the L1 is a no-op, and the L1 collapses to a plain mean of the
activation. That sidesteps differentiating an absolute value at zero and any `log(0)`/NaN hazard the
contract warns about; the exp could in principle overflow if a scale exploded, but the whole point of
the term is to *prevent* scales from blowing up, so it is self-stabilizing in the direction that matters.

So the term, before the weight, is the mean activated opacity plus the mean activated scale. It is
opacity pressure plus extent pressure, both L1, both on the activated quantity, both normalized as means
so the coefficient does not scale with the Gaussian count — just an elementwise activation and a
reduction, no neighbour search, no `N×N` over the means, affordable at every one of thirty thousand
steps. Now the weight. This is a *prior*, not a competing objective, and it must never overwhelm the data
term — if it did, the optimizer would happily fade and shrink everything into a uniform nothing that
minimizes the regularizer while ignoring the photos. The contract puts the photometric loss in the
0.03–0.1 range and asks me to keep the regularizer in roughly `1e-4` to `1e-1`. Because each piece is a
mean of activated values, a hundredth-scale coefficient keeps the pressure gentle and count-independent;
I set the scale and opacity weights equal at `1e-2` each. Small, equal weights on the two footprint
factors: the term bites only where the data term is flat — the floater regions and the redundant
Gaussians, the places the photos do not constrain — and is resisted everywhere the photos do.

Let me sanity-check the dynamics on the cases the floor exposed. A faint useless floater in bicycle's
background: low opacity, the opacity L1 keeps pressing the activated opacity down, the data term does not
object because the region is unconstrained, it crosses 0.005, `DefaultStrategy` prunes it. A big
over-reconstructing splat held just above the prune threshold: the opacity term barely moves it, but the
scale L1 keeps shrinking each activated scale through the exp chain rule, so it collapses toward a point
— either small enough to stop being an artifact, or the oversized-scale prune catches it, or shrinking
finally lets the data term find a better local configuration. A needle: its long axis carries a large
activated scale, so it takes the largest scale gradient of all and gets squashed back toward isotropy. A
tiny opaque speck doing real work: it is small, so the scale term barely touches it, and if it is
genuinely contributing the data term holds its opacity up against the gentle opacity pressure — it
survives, correctly. Each pathology gets pressure on the axis that defines it, and useful Gaussians are
protected by the data term they are actively helping.

So the delta from step 1 is concrete: where the floor left `compute_regularizer` returning a scalar
zero, I now return a gentle two-pronged L1 — `1e-2·mean(exp(scales)) + 1e-2·mean(sigmoid(opacities))` —
that presses on both factors of a Gaussian's footprint at every step (the full module is in the answer).
Reading the floor's shape, here is what I expect this to fix and where I am unsure. The unbounded scenes
should gain the most, because that is where the floor told me the floaters live: I expect garden and
stump to move up clearly past their 29.067 and 27.460, and I especially expect bicycle — the weakest at
26.641, the most under-observed — to gain, since its big faint background splats are exactly what the
two-pronged term collapses. Bonsai is the open question: it already sits at 32.531 because it is
fully-observed, so there is little floater slack for the prior to reclaim, and a parsimony pressure
that is good at deleting junk does not, by itself, sharpen geometry that is already well-constrained — I
expect at most a small gain there, possibly the smallest of the four. If that is what the numbers show —
broad gains on the outdoor scenes, a thin gain on bonsai — then the diagnosis for the next rung is
already written: a term that controls *how big* and *how many* Gaussians says nothing about *what shape*
each one is, so the needles that survive at small size on the indoor and high-detail regions would be
the next thing to attack.
