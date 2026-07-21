The naive floor came back as broken as the construction predicted: 14.56 on ResNet-18, 13.77 on
ResNet-34, 17.28 on ResNet-50, mean `(14.56 + 13.77 + 17.28)/3 = 15.20` — about five points above the
10% chance floor. The above-chance residues `+4.56 / +3.77 / +7.28` are tiny, exactly the low-variance
residue the probe overfits rather than learned class signal. And the ordering is *not* monotone with
capacity: ResNet-50 leads but the middle ResNet-34 is the *lowest*, below ResNet-18, and the two 512-dim
backbones sit within 0.79 of each other. That scramble is the signature of collapse — accuracy tracks
how much spurious separation each probe scrapes from a dead feature covariance, not anything the
objective learned, with ResNet-50's wider features giving only a slightly bigger cushion. So the hole is
roughly `90 − 15 ≈ 75` points, and the fix is forced in kind: a term that makes the collapsed
configuration a *high*-loss state. The only question is which statistic to penalize.

The contract rules out two of the three classic answers before I start. The contrastive answer restores
the missing repulsion as a softmax over in-batch negatives — a constant maximizes every negative
similarity, so it kills collapse — but that repulsion is a non-parametric estimate of the embedding
distribution's spread from pairwise sample distances, and high-dimensional spread estimates are
sample-hungry. Batch 256 gives each positive 255 negatives to characterize a `D = 2048` space; 255
points do not even span it, the sample covariance is rank-deficient by an order of magnitude, and the
InfoNCE partition function is wildly under-determined. That is why working contrastive methods reach for
thousands of negatives or memory queues of tens of thousands; this harness fixes batch 256 and gives no
queue, so I would be running contrastive learning in exactly its weakest regime. The asymmetric answer —
a predictor MLP plus stop-gradient — does not minimize a single scalar, and more decisively,
`forward(z1, z2)` hands me two *finished* embeddings and asks for one number: there is no place to insert
a predictor head or a between-branch stop-gradient without smuggling a sub-network into the loss. So
negatives and asymmetry are both awkward-to-impossible here. That points at the third family —
information-maximization — which states what a good representation *is* and gets non-collapse as a
consequence, and whose whole statistic lives inside the two tensors I am handed.

What do I want beyond "the two views agree"? An *informative* representation — and a constant is
maximally uninformative, the failure I measured. Barlow's redundancy-reduction principle makes
"informative" operational: a good code does not let its dimensions duplicate each other; decorrelate
them. That gives two desiderata pulling against each other exactly as I need — (1) invariance: the two
views produce the same feature values; (2) non-redundancy: distinct components are decorrelated. Bare
invariance alone collapses to a point; a constant has no variance and duplicated features are maximally
correlated, so the balance point of the two demands cannot be the constant. Non-collapse falls out as a
consequence, and it speaks to both collapse modes I named at the floor: forbidding zero variance kills
the constant, forbidding duplication kills the low-rank cousin.

Turn both into one differentiable scalar from `z1, z2`. Standardize each feature across the batch —
subtract its batch mean, divide by its batch std — then form the cross-correlation
`C_ij = (1/B) Σ_b ẑ1_{b,i} ẑ2_{b,j}`. In shapes, `ẑ1, ẑ2` are `[B, D]`, so `ẑ1ᵀ ẑ2 / B` is the `D×D`
cross-correlation matrix, each entry in `[−1, 1]`. Read the two desiderata off it: the diagonal `C_ii` is
feature `i` of view 1 against feature `i` of view 2 — invariance drives it to 1; the off-diagonal `C_ij`
(`i ≠ j`) is how much different features co-vary — non-redundancy drives it to 0. Together, make `C` the
identity, and the loss is the squared distance from it with a knob trading the halves:
`L = Σ_i (1 − C_ii)² + λ · Σ_{i≠j} C_ij²`.

Does this exclude collapse by construction? Evaluate it at the collapsed configurations rather than
assert. At `D = 2048` the matrix has `2048` diagonal entries and `4,192,256` off it. The constant
`z_{b,i} = c_i` has every feature dead; standardizing a zero-variance feature sends `0/0 → 0`, so
`C = 0`, the on-diagonal term is `Σ_i(0 − 1)² = 2048` and off-diagonal `0`. The constant costs `2048` —
not a minimizer. A correlation needs non-zero variance in both arguments even to reach 1, so the diagonal
target alone forbids the constant, and that is the load-bearing role of the standardization. The subtler
escape is the low-rank cousin's sharpest form: copy one invariant direction into all `D` features,
`z_{b,i} = s_b`. Then every standardized column is the same unit-variance vector and `C_ij = 1` for
*every* pair — the all-ones matrix. The on-diagonal term is `0` (the copies *are* invariant) but the
off-diagonal is `λ · 4,192,256`, which at `λ = 0.0051` is `21,380` — an order of magnitude more than the
constant's `2048`, because the off-diagonal term reads the duplicated signal as maximal redundancy. Only
`C = I` zeros both. Delete the off-diagonal term and the copy escape returns to zero loss; delete the
on-diagonal and nothing ties the views together — both halves are necessary, and they address the two
collapse modes I separated at the floor: the diagonal target kills the loud constant (a dead feature
cannot self-correlate to 1), the off-diagonal term kills the low-rank cousin (redundancy *is*
off-diagonal correlation, so driving `C_ij → 0` forces spread eigenvalues rather than a rank-`k` spike).

The `λ = 0.0051` is not arbitrary. With ~`D²` off-diagonal entries against `D` diagonal ones, an
unweighted sum lets the off-diagonal block drown the invariance alignment. Balance the two blocks in the
early regime where the diagonal is near its max `2048` and off-diagonals sit around some magnitude `ρ`:
setting `λ · 4,192,256 · ρ² = 2048` gives `ρ² ≈ 0.096`, `ρ ≈ 0.31`. So `λ = 0.0051` makes the redundancy
pressure comparable to invariance when off-diagonal correlations are around `0.3` — strong enough to
force decorrelation, weak enough not to overwhelm alignment before invariance has a chance to form.

One genuine limitation of a finite-batch estimate: can `C = I` even be reached? `ẑ1ᵀ ẑ2` is `2048×2048`
formed through a shared dimension of size `B = 256`, so its rank is at most `256`; it cannot equal the
rank-2048 identity within any single batch, and the loss never reaches exactly zero in one step. That is
not a bug — the loss is a stochastic estimate of a population statistic, and it is the *expected*
cross-correlation over many fresh batches that approaches the identity. This also sharpens the width
reasoning: a wider projector is *more* rank-deficient per batch (`256 ≪ 8192` is far worse than
`256 ≪ 2048`), so it needs proportionally more epochs to average the target into focus. At this budget
the default 2048 already asks 256-sample snapshots to fill a 2048×2048 matrix; going wider without more
epochs would leave the diagonal unfilled.

The version this harness runs is the practical port, and a few choices matter for the numbers. The
standardization is a non-affine `nn.LazyBatchNorm1d(affine=False)` — the center-and-divide-by-batch-std
I need, made lazy because the module is built with no args and only sees `D` at the first forward, and
`affine=False` on purpose so a learnable scale-and-shift cannot partly undo the standardization the
diagonal target is supposed to own. The on- and off-diagonal terms are *sums* (not means), the
off-diagonal extracted by the standard flatten-slice-view trick. And the one easy to miss: the raw
summed loss at 2048 width is `10³–10⁴` — my traces put the copy at `21,380` and the constant at `2048`.
That is a large objective for LARS, whose per-layer step is rescaled by roughly `‖p‖/(‖g‖ + wd·‖p‖)`.
That ratio is *nominally* scale-invariant — multiply the loss by a constant and the gradient scales with
it, so it should cancel — which is why the need for a multiplier is subtle rather than obvious. What
breaks the invariance is the interaction with `clip_lr=True`, the `eta = 0.02` trust coefficient, and
the weight-decay term in the denominator: at this magnitude and this budget the clipped adaptive rate
does not lift the diagonal toward 1, and a backbone can sit stuck near 10%. So rather than pretend I
derived a threshold from the LARS formula, I take the matched CIFAR recipe's remedy — a fixed
`scale_loss = 0.1` on the whole loss, paired with the default `2048 → 2048` projector and batch 256.
That is a different regime from the large-scale one, where an 8192-wide projector pairs with a much
smaller `scale_loss` and a 1000-epoch, batch-2048 schedule; that recipe would leave the diagonal stuck
at this budget. So I keep the default projector (no `CONFIG_OVERRIDES`) and `scale_loss = 0.1`; the
constant then costs `204.8` and the copy `2138`, same ordering at a gradient scale LARS can drive.

So the delta from step 1 is concrete: where naive returned `F.mse_loss(z1, z2)` and let the encoder
relax to a point, I now standardize each view across the batch, form the `D×D` cross-correlation, and
push it toward the identity — diagonal to 1 for invariance, off-diagonal to 0 for decorrelation, scaled
by 0.1 so LARS can drive it. The standardization makes the constant a high-loss state (`204.8`) and the
off-diagonal term forbids the copy-one-direction escape (`2138`), so the collapse that cost ~75 points is
excluded by construction and by arithmetic.

Against the naive numbers, falsifiably: the two-digit band — 14.56 / 13.77 / 17.28 — should vanish
entirely, this being the first step with a real anti-collapse term, so I expect a jump into the high-80s
on every backbone, recovering the bulk of that ~75-point hole. The ordering should *repair* from the
collapse scramble to a clean monotone with scale, the larger backbones genuinely separating classes
better. The risk to watch is exactly the LARS-starvation mode: a single backbone marooned near 10% would
indict the scale/width pairing rather than the objective, so the tell of a healthy run is all three
clearing the high-80s with no straggler. And if barlow clears the floor but leaves a point or two on the
table — plausibly on the smallest, narrowest-feature backbone, where the coupling-plus-standardization I
lean on should cost the most — that gap is the opening for the next step: enforce variance and
decorrelation on each branch *separately*, with no cross-branch coupling and no embedding
standardization, which might shape a slightly better-transferring geometry. (The distilled module and the
literal scaffold edit are in the answer.)
