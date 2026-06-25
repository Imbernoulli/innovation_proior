The naive floor came back exactly as broken as the construction predicted, and it came back in a way
that pins the diagnosis cleanly. Seed 42 gave 14.56 on ResNet-18, 13.77 on ResNet-34, 17.28 on
ResNet-50 — a tight band of two-digit accuracies hugging the ten-class chance floor of 10%, with the
larger backbone only marginally higher (its wider 2048-dim features give the probe a slightly bigger
chance-level cushion, nothing more). That is the signature of collapse, not of under-training: if the
invariance term were merely weak I would see one backbone climb and the others lag, but instead all
three sit in the same broken band, which says the failure is in the *objective*, identical across
architectures. The MSE invariance term drove the encoder toward the constant map — the zero-rest-length
spring network relaxing to a point — and the detached linear probe, reading frozen features with almost
no class-separating variance, lands near chance. So the hole is roughly 75 points of `val_acc` waiting
to be recovered, and the fix is forced in kind: I have to add a term that makes the collapsed
configuration a *high*-loss state instead of a minimizer. The only question is which statistic to
penalize, and the contract here — `forward(z1, z2)` returns a scalar, no negatives slot, no second
network, no stop-gradient hook — rules out two of the three classic answers before I start.

Let me walk the families against this contract so I land on the one that fits. The contrastive answer
restores the missing repulsion as a softmax over in-batch negatives: pull the positive pair together,
push every other image in the batch apart. It kills collapse — a constant maximizes every negative
similarity, so the loss can never reach zero there. But that repulsion is a non-parametric estimate of
the spread of the embedding distribution computed from all the pairwise sample distances in the batch,
and high-dimensional spread estimates from pairwise distances are brutally sample-hungry, which is why
contrastive methods want enormous batches or memory queues. This harness fixes the batch at 256 — small
— and gives me no queue. A sample-space repulsion estimated from 256 examples is exactly the regime
where contrastive learning is weakest, so I distrust it here. The asymmetric answer — a predictor MLP on
one branch plus a stop-gradient — does not collapse, but there is no single scalar being minimized, the
non-collapse is a dynamics accident, and crucially the contract gives me two finished embedding tensors
and asks for one number: there is no place in `forward(z1, z2)` to insert an extra predictor head or
arrange a stop-gradient between branches without smuggling a whole sub-network into the loss module.
So both negatives and asymmetry are awkward-to-impossible on this surface. That points me at the third
family, the one that states what a good representation *is* and gets non-collapse as a *consequence* of
the objective: information-maximization. And that family fits the contract perfectly, because the whole
statistic lives inside the two tensors I am handed.

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
of asking for non-redundancy — exactly the property the naive floor lacked.

Now turn both desiderata into one differentiable scalar from `z1, z2`. I have two `[B, D]` batches, view
1 and view 2 of the same images, b indexing the batch and i, j indexing features. The statistic that
captures both at once is a correlation between feature i of view 1 and feature j of view 2, taken across
the batch. So first standardize each feature across the batch — subtract its batch mean, divide by its
batch standard deviation — and then form, for every pair (i, j), the cross-correlation
`C_ij = (1/B) Σ_b ẑ1_{b,i} ẑ2_{b,j}` where ẑ is the standardized embedding. This is a D×D matrix, each
entry in [−1, 1]: +1 perfectly correlated, −1 anti-correlated, 0 decorrelated. Read the two desiderata
straight off it. The diagonal `C_ii` is the correlation of feature i in view 1 with feature i in view 2;
if feature i is invariant to the augmentation, the two views move together across the batch and
`C_ii = 1`. So invariance is "drive every diagonal entry to 1." The off-diagonal `C_ij` (i ≠ j) is how
much feature i and a *different* feature j co-vary; non-redundancy is "drive every off-diagonal to 0."
Together: make the cross-correlation matrix the identity, `C = I`. The loss is the squared distance from
the identity, with a knob trading the two halves:
`L = Σ_i (1 − C_ii)² + λ · Σ_{i≠j} C_ij²` — an on-diagonal invariance term and an off-diagonal
redundancy-reduction term, one scalar, no negatives, no predictor, no stop-gradient, no momentum
encoder.

Does this actually exclude collapse by construction? Try to collapse it. If the encoder emits a constant
`z_{b,i} = c_i`, then after standardizing feature i across the batch every value becomes
`(c_i − c_i)/std = 0/0` — a zero-variance feature, which the batch-normalization standardization maps to
zero. Its self-correlation `C_ii` is then not 1; a *correlation* needs non-zero variance in both
arguments even to reach 1, and a feature that does not vary across the batch can never satisfy
`C_ii = 1`. So the diagonal target alone already forbids the constant: "be perfectly correlated with
yourself across the batch" is impossible for a constant feature. That is the load-bearing role the
standardization plays — exactly the counter-pressure the naive floor had none of. And the subtler cheap
escape — find one invariant direction and copy it into all D features — is killed by the off-diagonal
term: the copies are the same signal, so |C_ij| = 1 off-diagonal and the redundancy term slams them. The
only zero-loss target is `C = I`: D features each invariant *and* mutually decorrelated. The
decorrelation term does the job that negatives or stop-gradients did elsewhere, but as part of the
objective. Delete the off-diagonal term and the copy-one-direction escape returns; delete the on-
diagonal term and nothing ties the two views together — both halves are necessary, the dual of the
naive collapse and the copy-collapse I just ruled out.

Now I have to be careful, because the version this harness actually runs is not the textbook
correlation-matrix loss verbatim — it is the practical port, and three implementation choices matter for
predicting the numbers. First, the standardization. Rather than dividing by per-feature std by hand, I
run each view's embeddings through a non-affine BatchNorm1d, which *is* the center-and-divide-by-batch-
std I need, and then the cross-correlation is just `bn(z1).T @ bn(z2) / B`. Because the loss module is
constructed with no arguments and only sees the feature dimension D at the first forward, I make the
BatchNorm *lazy* — `nn.LazyBatchNorm1d(affine=False)` — so it registers in `__init__` (and rides along
with `.to(device)`/dtype) but materializes D on the first call. Second, the loss is written with the
on-diagonal and off-diagonal as *sums* (`(diag − 1)².sum() + λ·offdiag².sum()`), not means,
and with `λ = 0.0051` — the standard redundancy weight, small because there are ~D² off-diagonal entries
against D diagonal ones, so an unweighted sum would let the off-diagonal block drown the diagonal
alignment before there is any reason its per-entry errors are correspondingly smaller. Third, and this
is the one that is easy to miss: the raw summed loss with a 2048-wide projector is on the order of
10³–10⁴, and LARS rescales each layer's step by `‖p‖ / (‖g‖ + …)`. An enormous raw gradient norm starves
that adaptive rescaling, so the diagonal of the cross-correlation never climbs toward 1 and the run
stalls (in a degenerate regime it can leave a backbone stuck near 10%). The fix the CIFAR recipe uses is
a fixed multiplier `scale_loss = 0.1` on the whole loss, taming the gradient norm so LARS can actually
move the diagonal. That multiplier is not needed in the large-scale ImageNet regime — there the 8192-wide
projector pairs with a smaller `scale_loss` and a 1000-epoch, batch-2048 schedule — but on this harness
(batch 256, the three CIFAR backbones, LARS `eta=0.02`, `clip_lr=True`) the solo-learn CIFAR settings
are the right match, so I keep the default `2048 → 2048` projector (no `CONFIG_OVERRIDES`) and
`scale_loss = 0.1`. An 8192-wide projector would need the long schedule to converge and at this
budget would leave the diagonal stuck.

So the delta from step 1 is concrete: where naive returned `F.mse_loss(z1, z2)` and let the encoder
relax to a point, I now standardize each view across the batch, form the D×D cross-correlation, and push
it toward the identity — diagonal to 1 for invariance, off-diagonal to 0 for decorrelation, scaled by
0.1 so LARS can drive it. The standardization makes the constant a high-loss state (no zero-variance
feature can self-correlate to 1) and the off-diagonal term forbids the copy-one-direction escape, so the
collapse that cost ~75 points is excluded by construction.

Here is what I expect against the naive numbers, falsifiably. The two-digit band — 14.56 / 13.77 / 17.28
— should vanish entirely; this is the first rung with a real anti-collapse term, so I expect a jump into
the high-80s on every backbone, recovering the bulk of that ~75-point hole. The cross-backbone ordering
should *invert* relative to naive in a small way: under collapse ResNet-50 led only because wider
features gave a bigger chance cushion, but with a working objective the larger backbones should genuinely
separate the classes better, so I expect ResNet-34 and ResNet-50 to edge above ResNet-18 rather than all
three sitting flat. The risk I am watching is exactly the LARS-starvation failure mode: if `scale_loss`
were too small or the projector too wide, a backbone could stay stuck near 10% with the diagonal never
reaching 1 — so the tell of a healthy run is that *all three* backbones clear the high-80s with no
stragglers. And if barlow lands solidly but a touch below where an explicit per-dimension variance floor
would, that gap is the opening for the next rung: redundancy reduction pins the diagonal to 1 via the
*cross*-correlation, which couples the two branches and standardizes the embeddings — a method that
instead enforces variance and decorrelation on each branch *separately*, with no embedding
standardization, might shape a slightly better-transferring geometry. That is the thread I would pull
next if barlow clears the floor but leaves a point or two on the table. (The distilled module and the
literal scaffold edit are in the answer.)
