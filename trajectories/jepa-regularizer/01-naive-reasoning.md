The regularizer is the whole point, but it sits on top of an invariance objective, and before I can
judge any anti-collapse term I have to see what the invariance objective does *without* one. So the
floor I start from is the barest possible fill of the contract: pull the two views together and add
nothing to stop them from imploding. That is the lower bound by construction, and running it is not a
throwaway — it is the diagnostic that tells me, in measured numbers, exactly how bad collapse is on
this harness, which is the failure every later rung has to beat.

Let me write down what the loop actually hands me, because the contract is narrow and I want to design
to it, not to a paper. `forward(z1, z2)` receives two `[B, D]` projected embeddings — view 1 and view 2
of the same batch of images, each pushed through the same backbone and the same projector — and must
return a dict with a scalar `"loss"`. That is the entire surface. There is no place to put negative
samples beyond what is already in the batch, no second network, no predictor, no stop-gradient hook,
no teacher branch — the harness gives me two embedding tensors and asks for a number. So the simplest
honest thing to optimize is "make the two views agree," and the cleanest scalar that says that is the
mean squared distance between the paired embeddings, `F.mse_loss(z1, z2)`. No coefficient, no margin,
no second term. Whatever survives the augmentations — crop, color jitter, grayscale, flip — is what
this loss declares to be the content, and it asks the encoder to map both views of an image to the same
embedding.

Before I commit to that even as the floor, I want to be honest that there is a *different* candidate for
"barest fill," and thinking it through is what tells me the MSE version is the one worth running. The
literal barest fill is the placeholder itself: `loss = 0`, return it and nothing else. But look at what
that does to the encoder given the rest of the harness. The training step backpropagates the
regularizer's `"loss"` plus the online probe's cross-entropy, and the probe reads the backbone through a
`detach()`. So if my regularizer contributes zero gradient and the probe's gradient is severed from the
encoder, the encoder receives *no gradient at all* — it never leaves its random initialization. The
metric would then be measuring a **random-feature** ResNet under a linear probe, and random features on
CIFAR-10 are not near chance: a random-init convolutional stack still carries a lot of low-level,
input-correlated structure, so a linear probe on top typically lands well above 10% — plausibly in the
thirties or forties. That is emphatically *not* the floor I want. The whole purpose of this rung is to
measure how far the *invariance objective itself* drags the representation down, and a zero-loss encoder
never engages the invariance objective, so it would report the wrong quantity — and, worse, a
misleadingly high one. A collapsed encoder can actually score *below* random init, because active
collapse spends gradient steps destroying the very input-correlated variance that random init handed it
for free. So the MSE fill is strictly the better floor: it exercises the objective under test and gives
me its genuine worst case, whereas the zero-loss fill measures an untrained network and would flatter the
setup. I keep the MSE.

I should be precise about why the MSE version is the floor and not merely a weak choice, because the
precision is the whole reason to run it first. The invariance term is, in the metric-learning lineage,
exactly the *attract-only* half of a contrastive pair loss: take the similar-pair branch
`½‖G(x₁) − G(x₂)‖²`, treat two views as the similar pair, and delete the margin repulsion that would push
dissimilar pairs apart. Picture the embeddings as point masses joined by springs whose rest length is
zero — every spring pulls its two endpoints toward coincidence, nothing ever pushes anything apart. Ask
gradient descent to relax that network and it has a configuration that makes *every* spring perfectly
happy at once, with exactly zero energy: put all the points at the same location. If the encoder learns
to ignore its input and emit one constant vector `c`, then for every image both views map to `c`, every
paired distance is zero, the MSE is zero — a global minimum of the objective I wrote. It is not a bad
local optimum I might dodge with luck; the constant map is *the* global minimizer, and it carries no
information about the image at all. There is no internal force opposing the collapse because the loss
only ever asks for energy to be lowered and never asks for it to be raised anywhere, so the trivial way
to make it low everywhere I look is to make the encoder flat. That missing counter-pressure — the
repulsion, the negatives, the decorrelation, whatever a real method supplies — is precisely the thing
this baseline omits, and naming the omission is what makes the rest of the ladder legible.

Let me make the collapse claim concrete rather than leave it at the spring picture, because I want to
know that gradient descent actually *flows* into the constant and does not merely tolerate it as one
minimizer among many. Write the loss with the mean reduction `F.mse_loss` actually uses,
`L = (1/BD) Σ_b Σ_i (z1_{b,i} − z2_{b,i})²`, and take the gradient with respect to one view's embedding:
`∂L/∂z1_{b,i} = (2/BD)(z1_{b,i} − z2_{b,i})`. It pulls view 1 of image b toward view 2 of the same image
and, by symmetry, view 2 toward view 1 — every step contracts each paired pair toward its own midpoint.
Now, the two views share the encoder, so the only way to make `z1_b` and `z2_b` coincide for a given
image is for the encoder to become insensitive to whatever augmentation separated the two views. There
are two ways to buy that insensitivity. The *hard* way is to learn features that genuinely survive crop,
jitter, grayscale and flip while still varying across *different* images — real invariant content. The
*easy* way is to become insensitive to the input entirely: if the encoder's output does not depend on
its input, then it certainly does not depend on the augmentation, and every paired distance is zero at
once, for every image, with no representational work done. Nothing in the objective rewards the hard way
over the easy way — both reach `L = 0` — and the easy way is reachable by simply shrinking the
information the encoder passes forward, a low-effort direction in parameter space that also happens to be
favored by weight decay. So gradient descent does not merely permit the constant; it is pulled toward it,
because the constant is the path of least resistance to the same zero the informative solution reaches
only with effort. That is the floor made mechanical.

And this harness's augmentation recipe sharpens the pull, which is worth spelling out because it tells me
how *fast* and how *completely* the collapse should arrive. The two views are separated by
`RandomResizedCrop(32, scale=0.2–1.0)`, color jitter, grayscale, light solarize, and horizontal flip —
an aggressive stack; a scale-0.2 crop can keep as little as a fifth of the image area, so the two views
of one image can be genuinely disjoint patches. The harder the augmentation, the harder the "hard way"
becomes: for the encoder to make two nearly-disjoint crops map to the same non-trivial embedding, it has
to discover features that are stable under that much geometric and photometric distortion *and* still
discriminative across images, which is a real representational problem. The "easy way" — ignore the input,
emit `c` — is completely indifferent to how violent the augmentation is; a constant map is invariant to
*everything* for free. So aggressive augmentation widens the effort gap between the two routes to zero
loss, which makes the shortcut relatively more attractive and the collapse more thorough. I should not
read this as "the augmentation causes the collapse" — the missing repulsion causes it — but the
augmentation strength is a multiplier on how decisively the naive objective should lose, and it argues
for a clean, low, unambiguous number rather than a marginal one. It also means I should not expect the
warmup to save anything: the 10 warmup epochs slow the *early* steps, so the probe may briefly read
half-decent features before the invariance term finishes contracting, but `val_acc` is reported at the
`is_final` endpoint, by which point the encoder has had the full schedule to reach the constant. The
number I get is the collapsed endpoint, not the transient.

There is one detail of *this* harness that makes the collapse louder than the pure argument suggests,
and it is worth pinning down because it shapes what the numbers will look like. The metric is not the
regularizer's own loss; it is a **linear probe trained online on the frozen, detached backbone
features**. The probe never sees the projector output `z`; it reads the 512- or 2048-dim representation
straight off the backbone and fits a 10-way classifier to it each step. So the question the metric
actually asks is: *after the encoder has been driven by this loss, do the backbone features linearly
separate the ten CIFAR classes?* If the encoder collapses — features with little variance, or all
information crammed into one direction — a linear classifier on top has nothing to separate and lands
near chance. CIFAR-10 has ten roughly balanced classes, so chance is 10%. I therefore expect the
naive run to sit not far above 10% on every backbone: a collapsed or near-collapsed representation,
with whatever residual signal the BatchNorm in the projector and the joint probe gradient happen to
leak through. It will not be exactly 10%, because the projector's batch-norm layers inject a little
per-feature variance and the probe is trained on whatever the backbone does retain early — before the
invariance term finishes pulling everything together — but it should be unambiguously broken: a
two-digit accuracy, low in the teens, that any real regularizer beats by a wide margin.

I want to think through the joint probe more carefully, because in principle it could *soften* the
collapse and I need to know whether to count on it. The training step does not minimize the regularizer
alone: it minimizes the regularizer's `"loss"` *plus* a cross-entropy term from the online linear probe,
both backpropagated in the same `backward`. Write it as `L_total = L_reg(z1, z2) + CE(W · sg(h), y)`,
where `h` is the backbone feature, `sg` is the stop-gradient the harness inserts with `detach()`, and
`W` is the probe's own weight. Differentiate with respect to any encoder parameter `θ`:
`∂L_total/∂θ = ∂L_reg/∂θ + ∂CE/∂θ`, and the second term is zero, because the only path from `θ` to the
cross-entropy runs through `h`, which the stop-gradient has severed — `∂CE/∂θ = (∂CE/∂h)(∂ sg(h)/∂θ) = 0`.
So the encoder is shaped *purely* by my MSE invariance term; the probe's gradient flows only into `W`.
The probe is a passive reader, fit each step to whatever the encoder currently produces, and it cannot
push back against collapse — by the detach, it structurally cannot. The encoder is free to march to the
constant, and the probe simply reports how unseparable the resulting features are. Good: that confirms
the floor is a clean test of the objective and not contaminated by the metric's own training signal.

It helps to see *why* a collapsed encoder pins the probe at chance in the probe's own terms, because it
also tells me what the residual few points above 10% actually are. The probe solves a ridge/least-squares
problem on the batch feature matrix `H ∈ ℝ^{B×d}`; its fitted directions live in the row space of `H`,
and its ability to separate classes is bounded by how much class-discriminative variance `H` carries.
When the encoder collapses, every row of `H` is nearly the same vector, so `HᵀH` — the Gram matrix the
probe effectively inverts — is dominated by a single rank-1 mean component with a thin skirt of
near-zero-variance directions coming from BatchNorm's per-feature rescaling and whatever the backbone has
not yet flattened. The probe can only reach for structure inside that thin skirt, and the skirt has no
reason to align with the ten CIFAR classes, so the classifier fits noise and lands at chance plus a small
capacity-dependent overfit. That is the mechanism behind the "not exactly 10%" I predicted: it is not
retained class signal, it is the probe extracting a little spurious separation from the low-variance
residue. Which immediately sharpens the cross-backbone expectation — the size of that residue is a
function of feature width and probe capacity, not of anything the objective learned.

A note on what I deliberately do *not* do here. I do not touch `CONFIG_OVERRIDES`; the default projector
`2048 → 2048` is fine, because reshaping the projector cannot rescue a loss that has no anti-collapse
term — collapse is a property of the objective, not the width, and narrowing or widening the projector
only changes how many coordinates share the same zero. I do not add an epsilon, a temperature, a
normalization of the embeddings, or any stabilizer, because every one of those would be a partial
anti-collapse mechanism smuggled in, and the point of the floor is to have *none*. The most tempting one
to reach for is a negative cosine similarity in place of MSE — `−cos(z1, z2)`, the invariance form the
predictor-plus-stop-gradient methods use — so let me be explicit about why I refuse it even here. Cosine
similarity first l2-normalizes each embedding onto the unit sphere. That normalization is *itself* a weak
variance constraint: it forbids the all-zero vector outright, since you cannot normalize zero, so it
quietly rules out one flavor of collapse for free. It does not rule out the important one — the encoder
can still send every image to the *same* unit vector, cosine 1, a constant direction, fully collapsed and
uninformative — but it launders away the cleanest constant and puts a thumb on the scale. I want the floor
to carry no such thumb, so I keep the raw, unnormalized MSE, whose global minimizer is the honest constant
`c` with no sphere to hide on. The loss is one line. I return the scalar under `"loss"` plus an
`"invariance_loss"` echo so the log shows the two are identical — there is nothing else in the objective.

One more scale observation, because it will matter for reading the number. `F.mse_loss` reduces by the
mean over *all* `B·D` elements, so the naive loss is an average squared coordinate error, an `O(1)`
quantity regardless of the 2048-wide projector — its gradient norm stays modest and the LARS optimizer
sees a well-behaved, small objective. There is nothing here that could stall training on numerical
grounds; if the encoder collapses it is because the objective *wants* it to, not because the optimizer
failed to move. That keeps the diagnosis clean: a broken number here is a statement about the objective,
full stop.

Now reason about what running this must produce across the three backbones, because the cross-backbone
spread is itself information for the next rung. All three see the identical loss; the only difference is
the backbone capacity and the feature width feeding the probe (512 for ResNet-18 and ResNet-34, 2048
for ResNet-50). Collapse is not a capacity problem — a bigger network collapses just as completely, and
arguably *more* cleanly, because it has more freedom to find the constant solution. So I do not expect
the larger backbones to rescue accuracy. If anything the three should all sit in the same broken band,
with only noise separating them, and possibly ResNet-50's wider features giving the probe a marginally
larger chance-level cushion: 2048 detached coordinates of near-zero-variance noise give a linear
classifier a little more room to fit a spurious separation than 512 do, which nudges its floor a hair
above the others without meaning anything. What I am watching for is exactly that: a tight, low band
across all three, with no clean monotone-with-scale ordering — because under collapse the accuracy is
noise, not signal, so I should *not* see the backbones line up by capacity. If instead one backbone
scored well, that would mean the invariance term alone was somehow not collapsing on that architecture,
and I would have to rethink — but the spring argument says that should not happen.

Let me make the prediction sharp enough to be wrong. I expect all three backbones in the low teens to
below twenty, well inside two digits; I expect no clean capacity ordering, because collapse-residue is
noise; and I expect, if anything, the widest-feature backbone (ResNet-50, 2048 dims into the probe) to
sit a hair *above* the two 512-dim backbones purely on the overfit-cushion argument, not because it
learned more. Three specific things would falsify the collapse reading. If any backbone cleared, say, the
forties, the invariance term would not be collapsing there and the whole floor premise would be wrong. If
the three lined up cleanly by capacity with a real spread, that would suggest the number is tracking
learned signal rather than noise, again contradicting collapse. And if the run were merely *under-trained*
rather than collapsed, I would expect the accuracy to still be climbing at the endpoint and to differ a
lot run-to-run; instead collapse should give a stable, converged-low number. Whatever the exact figures,
the shape I am betting on is a tight low band, and the gap between that band and where a real regularizer
lands is the budget I hand to step 2.

Before I close, I want to name the *two* shapes collapse can take, because the distinction is what makes
the next rung's fix precise rather than vague. The loud one is total collapse: the encoder emits one
constant vector, feature covariance rank zero, every dimension dead. That is the global minimizer I
traced. But there is a quieter cousin that the same zero-repulsion objective also permits — dimensional
collapse, where the encoder keeps *some* variance but crams all of it into a low-rank subspace, a handful
of active directions with the remaining `d − k` dimensions flat. The invariance term is just as content
with a low-rank informative-looking representation as with a point, as long as the paired distances are
small, and there is no term anywhere that rewards *using* all the dimensions. So the failure I am about to
measure is really "the feature covariance has collapsed toward low rank," with the constant map as its
extreme corner. That matters for what comes next: a fix has to raise the loss both when variance vanishes
(the constant) *and* when variance survives but is confined and redundant (the low-rank cousin). A term
that only prevents the constant — say, by forbidding zero variance — would leave the low-rank escape open;
the counter-pressure I add later has to speak to the whole spectrum of the covariance, not just its total
scale. Pinning that down now means the next rung is not "add something that stops collapse" but "add a
term that keeps the embedding's variance both nonzero *and* spread across dimensions," which is a much
more specific instruction to design against.

The diagnosis the floor sets up for step 2 is already written into the construction. The failure is not
a learning-rate problem, not a schedule problem, not a backbone problem; it is that the objective has
no term that raises the energy of the collapsed configuration. So the fix is forced in *kind* even
before I pick the exact one: I must add an anti-collapse term that makes the constant map — and the
softer dimensional collapse where all variance crowds into a low-rank subspace — a high-loss state
rather than a minimizer. The metric-learning ancestor would restore the margin repulsion against
negatives; the information-maximization line would instead demand that the embedding dimensions stay
spread and decorrelated, which is the branch I expect to climb on this harness because it needs no
negatives and no second network — exactly the machinery this contract does not give me room for. So
step 1's job is done the moment it produces a broken number: it converts "collapse is a theoretical
worry" into "collapse costs roughly N points of val_acc on this exact setup," and that gap is the
budget every later regularizer is trying to recover. I expect the naive baseline to be unambiguously
the weakest thing on the board — a two-digit accuracy near the ten-class chance floor on all three
backbones — and the size of that hole is what the next rung's decorrelation-or-variance term has to
fill. (The distilled module and the one-line scaffold edit are in the answer.)
