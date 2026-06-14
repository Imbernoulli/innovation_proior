The compactness L1 came back and it did most of what I predicted, but it also drew a clean boundary
around what a size-and-count prior can and cannot do, and that boundary is the next rung. Garden moved
29.067 → 29.318, stump 27.460 → 27.720, bicycle 26.641 → 26.844 — every outdoor scene gained, the mean
went 28.925 → 29.142, and the gains landed exactly where I expected them: the weakly-observed unbounded
scenes where the floaters lived. That is the two-pronged footprint penalty working as designed — faint
background splats faded across the prune threshold, big over-reconstructing splats collapsed, the slack
the photos left got reclaimed. But look at bonsai: 32.531 → 32.685, a gain of only 0.154, the thinnest
of the four and roughly half the size of garden's. That is the boundary I suspected in the last rung's
close. Bonsai is the fully-observed indoor scene; it has almost no floater slack for a parsimony prior to
reclaim, so once the few faint and oversized Gaussians are gone the compactness term has nothing left to
bite on. And yet bonsai is not perfect — it still leaves PSNR on the table. The compactness L1 controls
*how big* and *how many* Gaussians there are; it says nothing about *what shape* each surviving Gaussian
is. A needle and a disk of the same total extent get penalized identically — the additive per-axis scale
term squashes a needle's long axis, yes, but only by shrinking it, and a *small* needle is still a
needle. So what survives the compactness pass, especially on the high-detail and fully-constrained
geometry where there is no floater to delete, is a population of correctly-sized but badly-shaped
primitives. That is the missing preference now: not size, shape.

Let me name the pathology precisely, because the fix has to target it and not over-reach. The photometric
loss only ever sees pixels from the training cameras, and it says nothing about what an individual
primitive should look like in 3D. So the optimizer, doing exactly what I asked, finds whatever pile of
Gaussians reproduces the training pixels — and a great many come out extremely elongated, one variance
enormous and the other two collapsed almost to zero. Needles. From a training camera a forest of needles
can tile an image fine; rotate to a held-out view and each needle is a spike sticking out of the surface.
That is the residual that the compactness term cannot reach: it can make a needle smaller, but it cannot
make it *not a needle*. So I want a second term, on top of the compactness L1 I am keeping, that is about
shape — and the requirements are the same as before plus one. It must be computed only from the
parameters (no depth, no normals — that would cheat the setup), differentiable so it rides the same
backward pass, and `O(N)` cheap over millions of primitives across thirty thousand steps. And the new
one: it cannot be a sledgehammer, because thin, genuinely slender structures — a wire, a leaf edge — do
exist in these scenes and have to stay representable.

What is already on the table is the compactness term, and I keep it: it is a clean parsimony prior, it
keeps the primitive set tidy, and a tidy set is clean material for a shape prior to work on. But it
controls extent, not the *ratio* of the three axes, so I need to build the shape number from scratch.
Other ways people attack shape are instructive in where they stall. The flatten-it instinct — drive one
scale small so the Gaussian lies flat against the surface — is right that density should concentrate near
the surface, but "one scale small" is exactly the ambiguity I am fighting: a Gaussian with one small
scale can be a *disk* (two big axes, one tiny — good, it covers area) or a *needle* (one big axis, two
tiny — bad). Flatness does not separate them, and the optimization drifts right back into needles. An
aspect-ratio cap — bound max-scale over min-scale — is a two-axis comparison; it cannot tell a disk from
a needle either, and it over-punishes legitimately thin objects, exactly the sledgehammer I want to
avoid. The common failure is the same: they look at one or two scales at a time, and a Gaussian's shape
is a three-way thing. I need a single number that folds all three scales together and says, cleanly,
where on the sphere–disk–needle spectrum a Gaussian sits.

Let me try to build that number. What I want is "how many axes effectively matter" for this Gaussian. The
honest version is the rank of the covariance — sphere is rank 3, disk rank 2, needle rank 1. But integer
rank is useless on two counts: it is discrete, so it has no gradient and I cannot backprop through it; and
in floating point two tiny-but-nonzero scales make everything numerically rank 3, so there is no gradient
to push a near-needle toward a disk. I need a real-valued, differentiable surrogate that varies smoothly,
sitting near 1 for a needle, 2 for a disk, 3 for a sphere. How do I turn "spread of the three scales"
into a smooth scalar between 1 and 3? The three scales describe how the Gaussian's energy is distributed
across its principal directions. Normalize them into a probability distribution — what fraction of the
"size" lives on each axis — and a measure of how *spread out* that distribution is gives exactly the
count I want: concentrated on one axis means low spread (needle), spread evenly over two means medium
(disk), over three means high (sphere). The canonical measure of spread is Shannon entropy, and there is
a fact about it that makes it the right choice and not just *a* choice: the entropy of a distribution on
`n` equal atoms is exactly `log n`. So `exp(entropy)` gives `exp(0)=1` for one atom, `exp(log 2)=2` for
two equal, `exp(log 3)=3` for three equal. The exponential-of-entropy of a normalized spectrum is a
continuous, differentiable number that equals the integer rank at the clean cases and interpolates
smoothly between — the effective rank from signal processing (Roy & Vetterli, 2007), the real-valued
extension of rank, defined as `exp(H)` of the normalized singular values. It was built to answer
"effective dimensionality," which is my question, primitive by primitive.

I have to be careful which normalized quantity I feed the entropy, because the covariance and the scales
are related by a square. With `Σ = R S Sᵀ Rᵀ`, the rotation only orients it; the singular values of `Σ`
are the eigenvalues of `S Sᵀ`, which are the *squared* scales, not the scales. So "normalize the singular
values" means normalize the squared scales: form `q_j = s_j² / Σ_k s_k²`, the entropy `H(q) = −Σ q_j log
q_j`, and `erank = exp(H)`. Check the geometry with numbers in my head. Scales `(1, 0.01, 0.01)` —
`q ≈ (1, 0, 0)`, entropy near 0, `erank ≈ 1`: needle. `(1, 1, 0.01)` — `q ≈ (0.5, 0.5, 0)`, entropy near
`log 2`, `erank ≈ 2`: disk. `(1, 1, 1)` — `q = (⅓,⅓,⅓)`, `erank = 3`: sphere. It lands exactly where I
want and is smooth between. So `erank` is my shape coordinate, and it is strictly better than the pairwise
comparisons the other methods use, because it folds all three axes into one number and never confuses a
disk with a needle — the failure that sank flatness. And it suggests the target: I want the mass near 2,
disks, not pushed all the way to 3.

Now turn `erank` into a penalty, and the requirements are subtle enough that the naive symmetric quadratic
`(2 − erank)²` gets two of them wrong. First, I must *not* punish `erank > 2`: a Gaussian between disk and
sphere is fine, it is not a needle, and over-flattening everything to exactly planar would fight the
slight thickness scenes need. A quadratic punishes `erank = 2.5` as much as `1.5`. Second, the thin-but-
legitimate structures live at `erank` somewhere in `(1, 2)` — a slender disk, a wire — and I should only
*gently* discourage being too needle-like, with the pressure ramping up hard only as I approach the
genuine pathology at `erank = 1`. A quadratic treats `erank = 1.9` and `1.1` as merely a bit different; I
want `erank → 1` to feel like a wall and `erank` near 2 to feel like nothing. A one-sided barrier that
explodes at `erank = 1` and is exactly zero for `erank ≥ 2` is what those two requirements describe. What
function explodes as its argument → 0 and is gentle as it grows? The negative log. Let the argument be the
distance from the needle limit, `x = erank − 1`: then `−log(erank − 1)` blows up as `erank → 1` and equals
exactly 0 at `erank − 1 = 1`, i.e. `erank = 2`, the disk. Almost magically aligned — the wall sits at the
needle, the barrier naturally hits zero at the disk. The only problem is `erank > 2` makes the log
negative, which would *reward* sphericity; so clamp it at zero. Now the penalty is a steep wall as
`erank → 1`, decreasing through `(1, 2)`, exactly 0 at `erank = 2`, and flat zero above. Numerically: at
1.001 the penalty is ~6.9, at 1.1 ~2.3, at 1.5 ~0.69, at 2.0 it is 0, at 2.5 the raw value is negative so
the clamp gives 0 — exactly the asymmetric soft barrier I argued for.

There is a numerical landmine: at a true needle `erank = 1` exactly, `−log(0)` is `+∞` and a NaN gradient,
which the contract explicitly warns against. The barrier fix is a small floor inside the log:
`max(−log(erank − 1 + eps), 0)` with `eps = 1e-5`. This does two jobs — it keeps the value finite at the
needle (`−log(1e-5) ≈ 11.5`, a large-but-finite wall height) and keeps the gradient finite, so a primitive
sitting at the needle still gets a real, large push outward instead of a NaN.

Now I think I am done, but tracing what the optimizer *does* with this barrier reveals a loophole. The
penalty is a function of `erank`, and `erank` is scale-*invariant*: it depends only on the ratios
`q_j = s_j²/Σ s²`, not on overall size. Multiply all three scales by any constant and `erank` does not
change. So the barrier says "make the two smaller axes more comparable to the largest" but expresses no
preference about *how*. The optimizer can raise `erank` from 1 toward 2 by *growing the two small axes* to
match the big one — inflating the needle into a fat disk or even a blob — instead of by *flattening*,
keeping the third axis genuinely thin while letting the second grow. A bloated disk is bad: it pulls
density off the surface, the exact opposite of the concentrate-near-the-surface prior I am chasing. The
barrier alone is shape-correct but size-blind. I break the scale-invariance with an absolute downward
pressure on the *smallest* axis, so the cheapest way to satisfy the barrier is to flatten rather than
inflate. Add the smallest scale itself. Now consider the two routes from a needle `(big, tiny, tiny)`.
Inflate: grow the two tiny axes toward `big` — raises `erank` but *increases* the smallest axis, so the
new term punishes it. Flatten: grow one tiny axis to match `big` and keep the third small — also raises
`erank` toward 2 but keeps the smallest axis small, so the new term stays cheap. The combination prefers
flattening: a flat disk near the surface, exactly what the barrier alone could not pin down.

A timing question I have to get right, because applying this from step zero would be a mistake — and the
last rung's bonsai result is the warning. At initialization the Gaussians come from sparse SfM points,
roughly isotropic, `erank` near 3, and early training is when densification is busy cloning and splitting
to grow capacity and discover where the geometry even is. If I clamp shapes toward disks immediately I am
imposing a strong structural prior on a scene whose structure has not formed, fighting the exploratory
growth that is supposed to happen first. The fix is coarse-to-fine: let the early phase run free, let the
geometry and the count settle, and switch the shape barrier on partway through. I schedule it to begin at
step 7000 — well into the run, after the bulk of densification — and leave the compactness L1 on
throughout. The `step` argument the contract gives me is there precisely so I can gate the term on this
schedule. There is also a second-order interaction with densification I would otherwise miss: the standard
split rule reacts to the *norm of the summed* view-space gradient, and a disk covers many pixels whose
gradients point in many directions and cancel under the norm of the sum, so disks can fail to split even
when they should. The fixed harness here renders with `absgrad=True`, which accumulates the absolute
(sum-of-magnitudes) gradient for the split test, so disks keep splitting — this is the AbsGS/Bulò-style
densification compatibility, and it lives in the fixed pipeline, not in my term; but it is the reason the
disk-favoring prior and the densifier pull the same way rather than against each other.

One detail about the weight that is specific to *this* task's contract, and it is where I depart from a
clean separation of weights. The regularizer's return is added to the loss unscaled, so it carries its own
coefficients, and I keep the compactness weights at `1e-2` each. For the shape term I use a single weight
`ERANK_REG = 1e-2` that multiplies the *whole* shape contribution — the barrier mean and the smallest-
scale flattening term *together*. So the flattening term rides at the same `1e-2` scale as the barrier
rather than at full strength: it is a tie-breaker that decides *how* `erank` is raised, not a primary force
that would otherwise dominate the data term by pressing every smallest scale toward zero at weight one.
Keeping both shape pieces inside the one `1e-2` factor keeps the term in the `1e-4`–`1e-1` band the
contract asks for and keeps it gentle. So the full term: square the exp-activated scales, normalize to a
distribution over the three axes, take entropy, exponentiate to `erank`; the per-Gaussian shape penalty is
`ERANK_REG · (barrier + smallest_scale)` averaged over Gaussians, gated on after step 7000, with the
compactness L1 always on; every numerical edge guarded — a tiny constant in the normalization denominator,
a tiny constant inside the entropy log, the `eps = 1e-5` inside the barrier — and everything elementwise
over `N` and reduced with a mean, `O(N)` (the full module is in the answer).

So the delta from step 2 is concrete: where the compactness L1 controlled size and count and gained the
most on the floater-rich outdoor scenes while barely moving bonsai, I now keep that term and add an
effective-rank shape barrier that, after warmup, pulls the surviving needles onto flat disks. Reading the
two prior rungs, here is what I expect. Bonsai is the scene I am most hopeful for and the cleanest test of
the hypothesis: its thin 0.154 gain under compactness said the residual there is shape, not size, so if
the needle diagnosis is right bonsai should gain *more* under the shape prior than it did under
compactness — that is the falsifiable claim. Garden, also high-detail with lots of fine foliage where
needles tile training views, should gain again past 29.318. The risk is on the scenes where the shape
prior can *over-reach*: bicycle and stump have genuinely thin structures (spokes, twigs) at `erank` in
`(1, 2)`, and even with the asymmetric barrier and the warmup, the pressure could nudge a legitimately
slender primitive and cost a little — so I would not be surprised if bicycle, which gained cleanly under
pure compactness to 26.844, gives a little of that back, or if stump moves less than garden and bonsai.
If the numbers come out that way — bonsai and garden up clearly, bicycle and stump roughly flat or
slightly down — then the shape prior has done its job on the high-detail geometry where needles dominate,
at a small cost on the thin-structure scenes, and the net across the four is the question the mean will
answer.
