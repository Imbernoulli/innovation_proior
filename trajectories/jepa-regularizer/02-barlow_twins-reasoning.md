The naive floor came back exactly as broken as the construction predicted, and it came back in a way
that pins the diagnosis cleanly. Seed 42 gave 14.56 on ResNet-18, 13.77 on ResNet-34, 17.28 on
ResNet-50. Let me actually read those three numbers rather than wave at them. The mean is
`(14.56 + 13.77 + 17.28)/3 = 15.20`, so on average the encoder is fifteen points, i.e. about five points
above the ten-class chance floor of 10%. The above-chance residues are `+4.56`, `+3.77`, `+7.28` — tiny,
and exactly the "low-variance residue the probe overfits" I expected, not learned class signal. The band
is 13.77 to 17.28, a spread of 3.51 points, and crucially the ordering is *not* monotone with capacity:
ResNet-50 leads (17.28) but the middle backbone ResNet-34 is the *lowest* (13.77), below the smallest
ResNet-18 (14.56). If the invariance term were merely weak I would see the backbones line up by capacity
with one climbing; instead the ordering is scrambled and the two 512-dim backbones sit within 0.79 of
each other, which is noise. That scramble is the signature of collapse: the accuracy is not tracking
anything the objective learned, it is tracking how much spurious separation each probe scrapes out of a
dead feature covariance, and the only systematic effect is ResNet-50's 2048-wide features giving its
probe a slightly bigger cushion (`+7.28` vs `+4.56`/`+3.77`) — the overfit-cushion argument, confirmed.
So the hole is roughly `90 − 15 ≈ 75` points of `val_acc` waiting to be recovered, and the fix is forced
in kind: I have to add a term that makes the collapsed configuration a *high*-loss state instead of a
minimizer. The only question is which statistic to penalize, and the contract here — `forward(z1, z2)`
returns a scalar, no negatives slot, no second network, no stop-gradient hook — rules out two of the
three classic answers before I start.

Let me walk the families against this contract so I land on the one that fits, and do the elimination
with numbers rather than taste. The contrastive answer restores the missing repulsion as a softmax over
in-batch negatives: pull the positive pair together, push every other image in the batch apart. It kills
collapse — a constant maximizes every negative similarity, so the loss can never reach zero there. But
that repulsion is a non-parametric estimate of the spread of the embedding distribution computed from all
the pairwise sample distances in the batch, and high-dimensional spread estimates from pairwise distances
are brutally sample-hungry. Count it out on this harness: batch 256 gives each positive exactly 255
in-batch negatives, and those 255 samples have to characterize the geometry of a `D = 2048`-dimensional
embedding space. 255 points in 2048 dimensions do not even span the space — the sample covariance is
rank-deficient by an order of magnitude — so the partition function the InfoNCE softmax estimates is
wildly under-determined. This is exactly why the contrastive methods that work reach for enormous
batches (thousands) or memory queues (tens of thousands) of negatives; this harness fixes the batch at
256 and gives me no queue, so I would be running contrastive learning in precisely the regime where its
spread estimate is weakest. I distrust it here. The asymmetric answer — a predictor MLP on one branch
plus a stop-gradient — does not collapse, but there is no single scalar being minimized, the
non-collapse is a dynamics accident of the predictor-plus-EMA interplay, and crucially the contract gives
me two *finished* embedding tensors and asks for one number: there is no place in `forward(z1, z2)` to
insert an extra predictor head or arrange a stop-gradient between branches without smuggling a whole
sub-network into the loss module. So both negatives and asymmetry are awkward-to-impossible on this
surface. That points me at the third family, the one that states what a good representation *is* and gets
non-collapse as a *consequence* of the objective: information-maximization. And that family fits the
contract perfectly, because the whole statistic lives inside the two tensors I am handed — no negatives,
no queue, no extra network.

So go back to first principles about what I want from the representation, beyond "the two views agree."
I want it to be *informative* about the image — and a constant is the maximally uninformative thing,
which is exactly the failure I measured. There is an old, usable statement of "informative": Barlow's
redundancy-reduction principle, that a good sensory code recodes redundant input into components that
are statistically non-overlapping — if two units always carry the same information, one is wasted.
"Informative" becomes operational: don't let the embedding dimensions duplicate each other.
*Decorrelate* them. Now I have two desiderata that pull against each other in precisely the way I need:
(1) invariance — the two views of one image should produce the same feature values; (2) non-redundancy
— distinct feature components should not duplicate each other, i.e. be decorrelated. Bare invariance
alone collapses (everything to a point); a non-redundancy demand is the missing pressure, because a
constant has no variance and a set of duplicated features is maximally *correlated*, not decorrelated.
The place where the two desiderata balance cannot be the constant. That is non-collapse as a consequence
of asking for non-redundancy — exactly the property the naive floor lacked, and it speaks to both
collapse modes I named: forbidding zero variance kills the constant, forbidding duplication kills the
low-rank cousin.

Now turn both desiderata into one differentiable scalar from `z1, z2`. I have two `[B, D]` batches, view
1 and view 2 of the same images, `b` indexing the batch and `i, j` indexing features. The statistic that
captures both at once is a correlation between feature `i` of view 1 and feature `j` of view 2, taken
across the batch. So first standardize each feature across the batch — subtract its batch mean, divide by
its batch standard deviation — and then form, for every pair `(i, j)`, the cross-correlation
`C_ij = (1/B) Σ_b ẑ1_{b,i} ẑ2_{b,j}` where `ẑ` is the standardized embedding. Let me be careful about the
shapes so I know exactly what object this is: `ẑ1` and `ẑ2` are each `[B, D]`; `ẑ1ᵀ ẑ2` is `[D, D]`;
dividing by `B` gives the `D×D` cross-correlation matrix, each entry in `[−1, 1]` — `+1` perfectly
correlated, `−1` anti-correlated, `0` decorrelated. Read the two desiderata straight off it. The diagonal
`C_ii` is the correlation of feature `i` in view 1 with feature `i` in view 2; if feature `i` is invariant
to the augmentation, the two views move together across the batch and `C_ii = 1`. So invariance is "drive
every diagonal entry to 1." The off-diagonal `C_ij` (`i ≠ j`) is how much feature `i` and a *different*
feature `j` co-vary; non-redundancy is "drive every off-diagonal to 0." Together: make the
cross-correlation matrix the identity, `C = I`. The loss is the squared distance from the identity, with a
knob trading the two halves: `L = Σ_i (1 − C_ii)² + λ · Σ_{i≠j} C_ij²` — an on-diagonal invariance term
and an off-diagonal redundancy-reduction term, one scalar, no negatives, no predictor, no stop-gradient,
no momentum encoder.

Does this actually exclude collapse by construction? I do not want to assert that; I want to *evaluate*
the loss at the collapsed configurations and read the numbers. With the default `D = 2048` projector the
matrix has `2048² = 4,194,304` entries: `2048` on the diagonal and `4,194,304 − 2048 = 4,192,256` off it,
so the off-diagonal block outnumbers the diagonal by a factor of `2047`. Take the constant first:
`z_{b,i} = c_i`, every feature dead. Standardizing feature `i` maps every value to `(c_i − c_i)/std = 0/0`,
which the batch-normalization standardization sends to zero, so `ẑ1 = ẑ2 = 0` and `C = 0`. Then the
on-diagonal term is `Σ_i (0 − 1)² = 2048` and the off-diagonal term is `0`. So the constant is *not* a
minimizer — it costs `2048` in the diagonal term alone. A *correlation* needs non-zero variance in both
arguments even to reach 1, and a feature that does not vary across the batch can never satisfy `C_ii = 1`;
the diagonal target by itself already forbids the constant, and that is the load-bearing role the
standardization plays — exactly the counter-pressure the naive floor had none of. Now the subtler cheap
escape, the low-rank cousin in its sharpest form: find one invariant direction and copy it into all `D`
features, `z_{b,i} = s_b` for every `i`. Standardizing collapses each column to the *same* unit-variance
vector `ŝ_b`, so `C_ij = (1/B) Σ_b ŝ_b ŝ_b = Var(ŝ) = 1` for *every* pair `(i, j)` — `C` is the all-ones
matrix. The on-diagonal term is `Σ_i (1 − 1)² = 0` (invariance is perfectly satisfied — the copies *are*
invariant), but the off-diagonal term is `λ · Σ_{i≠j} 1² = λ · 4,192,256`, and with the redundancy weight
`λ = 0.0051` that is `21,380`. So the copy escape is not free either — it costs `21,380`, an order of
magnitude *more* than the constant's `2048`, precisely because the off-diagonal term sees the duplicated
signal as maximal redundancy and slams it. Only `C = I` — `D` features each invariant *and* mutually
decorrelated — makes both terms zero. Delete the off-diagonal term and the copy-one-direction escape
returns to zero loss; delete the on-diagonal term and nothing ties the two views together — both halves
are necessary, the dual of the naive collapse and the copy-collapse I just ruled out, and the numbers
say the two failure modes cost `2048` and `21,380` respectively where the target costs `0`.

It is worth connecting this back to the two collapse *modes* I separated at the floor, because the
off-diagonal term is precisely the piece that speaks to the quieter one. Total collapse (the constant) is
killed by the diagonal target — a dead feature cannot self-correlate to 1. But dimensional collapse, where
variance survives but crowds into a low-rank subspace, is not about any single feature being dead; it is
about the features being *redundant*, all re-expressing the same few directions. Redundancy is exactly
off-diagonal correlation: if the embedding lives in a `k`-dimensional subspace with `k < D`, then the
`D` coordinates must be linear combinations of `k` underlying signals, so many pairs `(i, j)` are strongly
correlated and the off-diagonal block lights up. Driving `C_ij → 0` for all `i ≠ j` forces the `D`
coordinates to carry statistically distinct content, which is the same as demanding the feature covariance
have spread eigenvalues rather than a rank-`k` spike. So the diagonal term addresses the loud collapse and
the off-diagonal term addresses the low-rank cousin — the pair covers the whole spectrum of the covariance,
which is exactly the instruction I extracted from the floor: keep the variance both nonzero *and* spread.

That `λ = 0.0051` is not arbitrary and the count above tells me why. There are ~`D²` off-diagonal entries
against `D` diagonal ones, so an unweighted sum would let the off-diagonal block drown the diagonal
alignment: `4.19M` small squared correlations against `2048` invariance terms. Ask what `λ` balances the
two blocks in the regime the optimizer actually lives in. At the start of training the diagonal is near
zero (features not yet invariant), so the diagonal term is near its max `2048`; suppose the off-diagonals
are each around some typical magnitude `ρ`. The off-diagonal term is then `λ · 4,192,256 · ρ²`, and setting
it equal to the diagonal's `2048` gives `ρ² = 2048 / (0.0051 · 4,192,256) = 2048 / 21,381 = 0.0958`, i.e.
`ρ ≈ 0.31`. So `λ = 0.0051` is tuned to make the redundancy pressure comparable to the invariance pressure
when off-diagonal correlations sit around `0.3` — strong enough to force decorrelation, weak enough that
it does not overwhelm the diagonal alignment before invariance has a chance to form. That is a sensible
operating point and it survives the sanity check.

There is one more thing I should check before trusting the target, because it is a genuine limitation of
computing this from a finite batch: can `C = I` actually be reached at all? The standardized views are
`[B, D] = [256, 2048]`, so `ẑ1ᵀ ẑ2` is a `2048×2048` matrix formed as a product through a shared
dimension of size `B = 256` — its rank is at most `256`. The identity `I_2048` has rank `2048`. A
rank-≤256 matrix cannot equal a rank-2048 matrix, so within any *single* batch the empirical
cross-correlation literally cannot be the identity, and the loss can never reach exactly zero in one step.
That is not a bug — it is the reminder that the loss is a *stochastic estimate* of a population statistic.
Each batch hands the optimizer a noisy, rank-deficient snapshot whose gradient still points toward "raise
the diagonals, zero the off-diagonals," and it is the *expected* cross-correlation over many batches of
fresh samples that can approach the full-rank identity, not any one batch's matrix. This reframes the
convergence question as one of averaging: the encoder needs enough batches for the per-batch noise to
wash out. And it sharpens the projector-width reasoning I am about to lean on — a wider projector makes
the matrix *more* rank-deficient per batch (`256 ≪ 8192` is far worse than `256 ≪ 2048`), so it needs
proportionally more batches, i.e. a longer schedule, to average the target into focus. At this budget the
default `2048` width is already asking `256`-sample snapshots to fill a `2048×2048` matrix; going wider
without more epochs would leave the diagonal unfilled.

Now I have to be careful, because the version this harness actually runs is not the textbook
correlation-matrix loss verbatim — it is the practical port, and three implementation choices matter for
predicting the numbers. First, the standardization. Rather than dividing by per-feature std by hand, I
run each view's embeddings through a non-affine `BatchNorm1d`, which *is* the center-and-divide-by-batch-
std I need, and then the cross-correlation is just `bn(z1).T @ bn(z2) / B`. Because the loss module is
constructed with no arguments and only sees the feature dimension `D` at the first forward, I make the
BatchNorm *lazy* — `nn.LazyBatchNorm1d(affine=False)` — so it registers in `__init__` (and rides along
with `.to(device)`/dtype) but materializes `D` on the first call. I should also note the BatchNorm is
`affine=False` on purpose: a learnable scale-and-shift would let the network partly *undo* the
standardization, restoring a per-feature scale the diagonal target is supposed to own, so I want the pure,
parameter-free center-and-divide. Second, the loss is written with the on-diagonal and off-diagonal as
*sums* (`(diag − 1)².sum() + λ·offdiag².sum()`), not means, and with `λ = 0.0051`, for exactly the balance
I just computed. The off-diagonal extraction uses the standard flatten-view-slice trick, and since it is
load-bearing (a wrong index set would penalize the diagonal or miss entries) I want to trace it once on a
`D = 3` matrix to be sure it returns what I think. A `3×3` matrix flattens to
`[a₀₀, a₀₁, a₀₂, a₁₀, a₁₁, a₁₂, a₂₀, a₂₁, a₂₂]`; dropping the last element with `[:-1]` leaves eight
entries; viewing as `(n−1, n+1) = (2, 4)` gives rows `[a₀₀, a₀₁, a₀₂, a₁₀]` and `[a₁₁, a₁₂, a₂₀, a₂₁]`;
dropping the first column with `[:, 1:]` gives `[a₀₁, a₀₂, a₁₀]` and `[a₁₂, a₂₀, a₂₁]`; flattening yields
`[a₀₁, a₀₂, a₁₀, a₁₂, a₂₀, a₂₁]` — exactly the six off-diagonal entries of a `3×3`, with every diagonal
element `a₀₀, a₁₁, a₂₂` dropped. The trick works: the padding-by-one that the view introduces is precisely
what shifts each row past its own diagonal element. Good, the off-diagonal sum is the object I meant. Third, and this is the one that is easy to miss:
the raw summed loss with a 2048-wide projector is on the order of `10³–10⁴` — my traces above put the
copy configuration at `21,380` and the constant at `2048`, so early-training values in the thousands are
the norm. That is a large objective to hand to LARS, whose per-layer step is rescaled by roughly
`‖p‖ / (‖g‖ + wd·‖p‖)`. I want to be honest about the mechanism here: that `‖p‖/‖g‖` ratio is *nominally*
scale-invariant — multiply the whole loss by a constant and the gradient norm scales with it, so the ratio
should cancel — which is exactly why the need for a scale multiplier is subtle rather than obvious. What
breaks the invariance is the interaction with `clip_lr=True`, the `eta = 0.02` trust coefficient, and the
weight-decay term in the denominator: at this loss magnitude and this budget, the clipped adaptive rate
and the warmup-cosine schedule do not lift the diagonal toward 1, and a backbone can sit stuck near 10%.
Rather than pretend I have cleanly derived the threshold from the LARS formula, I take the matched CIFAR
recipe's remedy: a fixed multiplier `scale_loss = 0.1` on the whole loss, which the solo-learn CIFAR
settings pair with the default `2048 → 2048` projector, batch 256, and these three backbones. That is a
different regime from the large-scale one, where an 8192-wide projector pairs with a much smaller
`scale_loss` and a 1000-epoch, batch-2048 schedule; that recipe would need the long schedule to converge
and at *this* budget would leave the diagonal stuck. So I keep the default projector (no
`CONFIG_OVERRIDES`) and `scale_loss = 0.1`. With the multiplier, the constant configuration costs
`0.1 · 2048 = 204.8` and the copy configuration `0.1 · 21,380 = 2138` — the same ordering, now at a
gradient scale LARS can drive.

So the delta from step 1 is concrete: where naive returned `F.mse_loss(z1, z2)` and let the encoder relax
to a point, I now standardize each view across the batch, form the `D×D` cross-correlation, and push it
toward the identity — diagonal to 1 for invariance, off-diagonal to 0 for decorrelation, scaled by 0.1 so
LARS can drive it. The standardization makes the constant a high-loss state (no zero-variance feature can
self-correlate to 1, `204.8` after scaling) and the off-diagonal term forbids the copy-one-direction
escape (`2138` after scaling), so the collapse that cost ~75 points is excluded by construction and by
arithmetic.

Here is what I expect against the naive numbers, falsifiably. The two-digit band — 14.56 / 13.77 / 17.28
— should vanish entirely; this is the first rung with a real anti-collapse term, so I expect a jump into
the high-80s on every backbone, recovering the bulk of that ~75-point hole. The cross-backbone ordering
should *repair itself* relative to naive: under collapse the ordering was scrambled (ResNet-34 the
lowest, below ResNet-18) because accuracy was noise, but with a working objective the larger backbones
should genuinely separate the classes better, so I expect a clean monotone with scale —
ResNet-50 > ResNet-34 > ResNet-18 — rather than the scramble. The risk I am watching is exactly the
LARS-starvation failure mode: if `scale_loss` were too small or the projector too wide, a backbone could
stay stuck near 10% with the diagonal never reaching 1 — so the tell of a healthy run is that *all three*
backbones clear the high-80s with no stragglers, and a single backbone marooned near chance would
immediately indict the scale/width pairing rather than the objective. And if barlow lands solidly but a
touch below where an explicit per-dimension variance floor would — if, say, ResNet-18 sits lowest and a
point or two under the larger backbones — that gap is the opening for the next rung: redundancy reduction
pins the diagonal to 1 via the *cross*-correlation, which couples the two branches and standardizes the
embeddings, and a method that instead enforces variance and decorrelation on each branch *separately*,
with no embedding standardization, might shape a slightly better-transferring geometry. That is the
thread I would pull next if barlow clears the floor but leaves a point or two on the table. (The distilled
module and the literal scaffold edit are in the answer.)
