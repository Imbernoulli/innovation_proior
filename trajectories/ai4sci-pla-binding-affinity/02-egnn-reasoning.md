The SchNet numbers came back and they confirm exactly the under-informed-but-honest floor I expected,
which is useful because it tells me precisely where to push. On the 2013 core set it lands at RMSE
1.4765 / Rp 0.792, on 2016 at 1.3702 / 0.777, and on the 2019 temporal holdout at 1.5624 / 0.570. Let
me actually read the table rather than glance at it, because two things in it are worth pulling apart.
First, the ordering of RMSE across benchmarks is *not* monotone in what I would naively call
difficulty: 2013 (107 complexes) posts a worse RMSE than 2016 (285), yet 2013's Rp (0.792) is *higher*
than 2016's (0.777). Those two facts only look contradictory until I remember the metrics measure
different things. Rp is scale- and offset-free — it asks whether the predicted ranking tracks the true
ranking — while RMSE is absolute and punishes any miscalibration of scale or bias. So 2013 is a small,
noisy set on which the model ranks complexes *well* (good Rp) but whose absolute errors are inflated by
sample noise and a slight scale miscalibration (worse RMSE); 2016 is larger and better-calibrated in
absolute terms but the ranking is a touch looser. Neither of those is the alarming number. The
alarming number is 2019: RMSE 1.5624 is the worst it posts anywhere *and* Rp 0.570 is the worst
correlation anywhere, and both fail together. The drop in Rp from 2016 to 2019 is 0.777 − 0.570 =
0.207, and the RMSE gap 2016→2019 is 0.192 — a large, coherent collapse, not a wobble.

It is worth making the 2013 metric divergence quantitative, because it tells me *what kind* of error
SchNet makes and therefore what a fix has to attack. Write the predictions on a benchmark as `p_i`
against truths `y_i`. Rp is the correlation `cov(p, y) / (σ_p σ_y)`, which is invariant to any affine
rescaling `p → a·p + b`: rank the complexes correctly and Rp is high no matter how the scale is set.
RMSE, `√mean((p − y)²)`, is not — it picks up every bit of `a ≠ 1` and `b ≠ 0`. The `-logKd/Ki`
labels span several log units (the tightest and loosest binders differ by many orders of magnitude in
`Kd`), so if a regressor ranks the 107 CASF-2013 complexes well but *compresses* its predicted range
toward the training mean — the textbook behaviour of an over-regularized model on a small,
distribution-shifted set — it will show a high Rp and an inflated RMSE at the same time. A compression
that shrinks a true span of, say, several units down to a predicted span a couple of units smaller
injects on the order of a unit of RMSE from range-compression *alone*, entirely independent of ranking
quality. That is exactly the 2013 signature: Rp 0.792 (ranking is fine) alongside RMSE 1.4765 (scale
is off). So SchNet's 2013 problem is calibration, not comprehension, and I should not expect the next
rung to move 2013 RMSE dramatically unless it also sharpens the dynamic range — a caution I will hold
against whatever number comes back.

That collapse is the tell, and it points cleanly at the mechanism. The 2019 holdout is the largest
test set (4366 complexes) and the one whose chemistry is most removed from training by the temporal
split, so it is precisely the set where the model cannot lean on memorized motifs and must *reason*
about contact geometry it has not seen. SchNet sees the interface and is invariant, but its single
window onto geometry is the scalar distance through the RBF; the angle and triangle-area statistics in
the 11-dim edge features never enter the message. On near-training complexes (2016) it can recall
distance signatures it has seen and post a respectable 0.777; on the temporally distant 2019 set, where
recall is worthless, it has nothing extra to reason with and the correlation falls apart. So the
diagnosis is sharp and it exonerates the parts I was worried about: the bottleneck is *not* the
message-passing skeleton — the heterogeneous four-convolution structure and the dual interface readout
are doing their job, because if they were broken the near-training 2016 number would be bad too, and it
is not. The bottleneck is what the message is *allowed to depend on*. SchNet's filter is a function of
`d` alone, so two contacts at the same distance but different orientation are identical to it, and
binding is exquisitely orientation-dependent. I need to widen what a message can condition on — and I
want to widen it along *one* axis at a time, so the next number is attributable.

Let me lay out the axes I could push, because there is more than one and conflating them would waste
the rung. Axis one: make the message depend on *both endpoint atoms*, not just one gated neighbour — a
richer, learned dependence on who is touching whom. Axis two: make the message depend on *more of the
geometry* than the scalar distance — pour the 11-dim angle/area/neighbour statistics in. Axis three:
change how neighbours are *aggregated* — attention instead of a filtered sum. And there is a tempting
axis four to dispose of explicitly: go fully *equivariant*, carrying vector features per node and
updating coordinates, the signature half of the equivariant layer. SchNet is thin on axes one and two
both. If I push several at once — say, a full transformer over contacts fed the entire 11-dim edge
vector — and the number improves, I will not know whether the endpoint-pair richness, the extra
geometry, or the attention bought it, and the ladder loses its diagnostic value. Axis four I can
eliminate on hard grounds rather than taste: the target is an invariant scalar and the double-centering
argument below shows distances are already a sufficient geometric statistic for such a target, so
vector features add no expressiveness a scalar readout can use — and in any case the edit surface
exposes no coordinates to carry as vectors, so it is unimplementable here. Axis three, attention
aggregation, is a real and orthogonal lever, but it is a readout-side change, and the readout already
carries an attention-normalized bias term I inherited; I do not want to entangle a new message-side
attention with the existing head-side one this rung. So I deliberately hold the geometry input fixed at
the scalar distance this rung and move only axis one, the node-pair expressiveness. That also keeps a
clean two-rung ablation in reserve: if node-richness-at-fixed-geometry helps the hard set but costs the
easy one, the *next* rung's job — add the geometry back — is already written by the result.

The cleanest way to widen the message along the node axis is the equivariant-message-passing edge
function, because its whole premise is a learned dependence of the message on the endpoints and the
geometry rather than SchNet's single distance-gated filter. Let me re-derive it from the symmetry,
because the symmetry is what makes it safe and also tells me which half of it I actually get to use on
this edit surface. The affinity is an invariant scalar, so I want every message to be invariant: rotate
or translate the complex and the message is unchanged. What does a rigid motion leave alone? Sending
`r → Q r + t`, a raw coordinate scrambles, a difference `r_i − r_j → Q(r_i − r_j)` still rotates, but
the squared distance `‖r_i − r_j‖²` is fixed because `QᵀQ = I` and `t` cancels in the difference. So
the canonical equivariant layer feeds its edge function the *invariant* squared distance alongside the
node features: `m_ij = φ_e(h_i, h_j, ‖r_i − r_j‖², a_ij)`. Where this differs from SchNet, and where
its extra power lives, is that the message is a full function of *both* endpoint feature vectors and
the distance — not just a distance-derived filter multiplying one neighbour. Two contacts at the same
distance can now produce different messages because the *atoms* differ, and the function can carve the
distance dependence per atom-pair type instead of forcing it through one shared radial filter. That is
the widening I want, and it is exactly axis one.

Now the famous second half of the equivariant layer is the coordinate update: move each point along a
weighted sum of relative-difference vectors, `x_i ← x_i + C Σ_j (x_i − x_j) φ_x(m_ij)`, where the
weight `φ_x(m_ij)` is an *invariant scalar* read off the message, so `Q` factors out of the differences
and the update is equivariant. That update is the reason the method can emit a *vector* target — an
updated position, a velocity. But I have to ask whether I need it here, and the answer is a clean no,
on three grounds that stack. First, my target is a single invariant *scalar*, `-logKd/Ki`; there is no
vector to emit, so the equivariant coordinate channel has nothing to point at. Second, and this is the
reassurance that dropping it costs no expressiveness in principle: for a fixed node indexing the
pairwise distance matrix already determines the geometry up to a rigid motion. The classical
double-centering argument makes this exact — given the squared-distance matrix `D`, the matrix `B =
−½ J D J` with `J = I − (1/n)11ᵀ` is the centered Gram matrix `X_c X_cᵀ`, whose eigendecomposition
recovers the coordinates up to `O(3)`, and for `n` points in 3D that Gram matrix has rank at most 3.
So the distances carry every bit of geometric information the difference *vectors* would, for any
target that must itself be `O(3)`-invariant; the coordinate update adds expressiveness only when the
*output* is required to be equivariant. Third, and decisively for this task, the harness does not hand
me coordinates at all. I get a `PLABatch` whose geometry is precomputed into invariant edge features;
there is no `pos` to update, no relative-difference vector to form, so the coordinate update is not
merely unnecessary — it is *unimplementable* on this edit surface. All three point the same way. I keep
the half of the method I can use and that the target wants: the invariant equivariant-style message,
`φ_e` of the endpoints and the distance, summed into the node update.

So I read the distance the only way the harness exposes it, `edge_attr[:, -1:] * 10`, giving a 1-dim
scalar distance per edge in angstroms, and that single scalar is the geometric input to every message.
The message itself I structure as the equivariant layer's edge function, decomposed into three additive
SiLU-MLP terms so the source atom, the destination atom, and the edge distance each get their own
learned transform before they are combined: `msg = mlp_u(x_src) + mlp_v(x_dst) + mlp_e(dist)`, summed
over neighbours, then a node MLP on the concatenation of the destination's own feature and the
aggregate, `node_mlp([x_dst, agg])`. SiLU throughout because, like SchNet's Softplus, it is the smooth
activation the equivariant layer uses on its invariant channels — and here every channel is invariant
(I never touch a coordinate), so there is no equivariance to endanger by a pointwise nonlinearity. This
is a strictly richer message than SchNet's on the node side: SchNet gated one projected neighbour by a
distance filter; here both endpoints *and* the distance pass through their own MLPs and add, so the
message can express "this kind of ligand atom meeting that kind of pocket atom at this separation" in a
way a single radial filter cannot.

I should be honest about how much node-pair richness this additive form actually buys, because it is
less than the slogan suggests and knowing the gap keeps my expectations calibrated. The canonical edge
function is a *joint* map `φ_e(h_i, h_j, d)` that can, in principle, form arbitrary cross terms between
the two atoms — the message for "this donor meeting that acceptor" can be genuinely different from the
sum of "this donor to anything" and "anything to that acceptor." My decomposition `mlp_u(src) +
mlp_v(dst) + mlp_e(dist)` is *separable*: the three contributions add, they do not multiply, so at the
message-formation step the source and destination cannot interact — there is no `src ⊗ dst` cross
term. What the additive message really provides over SchNet is that *both* endpoints get to speak, each
through its own learned transform, rather than only the source being gated. The true atom-pair
*interaction* — the multiplicative "these two specifically, together" signal — does not enter at the
message; it enters later, partly through `node_mlp([x_dst, agg])`, which mixes the destination's own
feature with the aggregated messages nonlinearly and so can manufacture some cross term after the fact,
and decisively through the readout's per-contact triple product, which *is* multiplicative in source,
destination, and edge. So the widening this rung buys is real but modest and mostly on the "both
endpoints heard" axis, with the sharp pairwise interaction deferred to the head. That is a reason to
temper how large a gain I predict, not a reason to abandon the move — and it is another argument for
holding the geometry axis fixed, since piling the 11-dim edge features into a still-separable message
would confound "more geometry" with "still no cross term" and muddy the read.

I want to be scrupulous about the one place this is *not* richer than SchNet, because it is the whole
reason I am holding geometry fixed and it sets up the next rung. EGNN's geometric input is still only
the scalar distance — the same single number SchNet used, just consumed through `mlp_e` instead of a
radial filter. And feeding one scalar into `mlp_e = Linear(1, H) → SiLU → Linear(H, H) → SiLU`
reintroduces exactly the pathology the RBF was invented to remove. At initialization every channel of
the first `Linear(1, H)` is an affine function of the *same* single variable `d`, so the pre-activation
columns are all scalar multiples of the distance vector plus a bias — rank one in the distance
direction — and a smooth monotone nonlinearity keeps them comonotone, so the geometry channel has
effectively *one* degree of freedom in `d` at init. Where SchNet expanded that one distance into 60
decorrelated RBF bumps, giving its filter fine distance resolution from step one, EGNN feeds the
distance raw and its `mlp_e` can only learn a single monotone warp of `d` to start from. So on pure
geometric resolution EGNN is *thinner* than SchNet: it trades SchNet's distance-resolution for
node-pair expressiveness. Whether that trade nets out positive is the empirical question this rung
answers, and I can already sharpen where each side of the trade should show up.

Everything around the message stays the heterogeneous interface skeleton, because the SchNet rung
already showed that skeleton is not the bottleneck — the near-training 2016 number was fine, so the
four-convolution structure and the dual readout are load-bearing and correct, and I am swapping only
the convolution. Four EGNN convolutions per layer, one each for covalent-ligand, covalent-pocket,
non-covalent ligand→pocket, and non-covalent pocket→ligand, all computed in parallel from the same
input features and summed per destination node type: a pocket atom gets its covalent update plus the
non-covalent update from the ligand contacts pointing into it, a ligand atom gets its covalent update
plus the pocket contacts. Three layers, hidden width 256 — the same receptive-field-versus-oversmoothing
reasoning as before, a local 5 Å shell that three rounds traverse without homogenizing the complex.
On the parameter budget this is roughly a wash with SchNet: an `EGNNConv` carries two endpoint MLPs
(`256→256→256` each), a distance MLP (`1→256→256`), and a node MLP (`512→256→256`), so where SchNet
spent its per-block budget on a 60-input filter and an output MLP, EGNN spends a comparable budget on
the second endpoint MLP — a few hundred thousand parameters per block either way, twelve blocks, a few
million total. I am not buying node richness with extra capacity; I am *reallocating* SchNet's geometry
capacity to the node side. The readout is the shared dual bidirectional interface scorer the ladder
uses — a per-contact triple product of projected source atom, projected destination atom, and a
projection of the contact distance, summed over a complex's contacts in both directions, each direction
corrected by an attention-normalized bias term whose softmax over the complex's contacts gives a
scale-stable per-complex reference to subtract, then averaged — except that now the edge projection
into the score is a `Linear(1, H)` of the raw distance rather than a `Linear(60, H)` of the RBF, the
same thinning carried consistently into the head. I keep the harness's plain MSE on the single
`forward` output — no `compute_loss` hook — because, as with SchNet, this fill produces one prediction.

Two checks before I commit. First, invariance has to survive end to end, and it does by the same
argument as SchNet, only easier to see: the only geometric quantity that ever enters is
`edge_attr[:, -1:] * 10`, a distance, which is rigid-motion invariant; the node features are chemical
one-hots, also frame-independent; every operation downstream — the SiLU MLPs, the sum aggregation, the
`node_mlp` concatenation, the triple-product readout, the softmax bias — is a function of those
invariant inputs, and no coordinate or difference vector is ever formed. So the prediction is invariant
by construction, no augmentation, exactly as before. Second, a small worked check that the additive
message does what I claim — distinguish two same-distance contacts by their atoms. Take a ligand
carbonyl oxygen contacting a pocket backbone amide N-H at 3.0 Å, and a ligand methyl carbon contacting
a pocket aromatic carbon at the same 3.0 Å. SchNet's filter, a function of `d = 3.0` alone, produces
the *identical* gate `W` for both and can only differ in the message through the projected source
feature — one atom's channel. My message is `mlp_u(src) + mlp_v(dst) + mlp_e(3.0)`: the distance term
`mlp_e(3.0)` is shared, but `mlp_u` fires differently on an oxygen than on a methyl carbon and `mlp_v`
fires differently on an amide nitrogen than on an aromatic carbon, so the two messages differ in *both*
endpoint terms. The polar H-bond-like contact and the nonpolar van-der-Waals-like contact land in
different regions of message space despite the identical distance — which is precisely the
orientation-blind failure mode of SchNet that I set out to widen, at least on the identity of the
partners if not yet on their relative orientation.

So here are the falsifiable expectations against SchNet's numbers, and they follow directly from the
two sides of the trade. The node-pair-aware message should help most where the chemistry of the atom
pair carries signal that a distance filter alone misses, and that is where *reasoning* beats *recall* —
the temporal holdout. SchNet's worst result was 2019 Rp 0.570; if letting both endpoints into the
message buys real generalization, EGNN should lift 2019 the most, and I would expect its 2019 Rp to
clear 0.60 and its 2019 RMSE to drop below SchNet's 1.5624, because that set is where extra reasoning
capacity, not memorization, decides the score. On 2013 I expect a clear improvement too — richer
messages should sharpen the ranking on the small CASF-2013 set. The risk lives on 2016, and I can name
it exactly: SchNet's 60-channel RBF gave it fine distance resolution on the near-training core set where
memorized distance signatures pay off, and EGNN's single raw-distance channel — the rank-one-at-init
`mlp_e` — is coarser there, so it is entirely possible EGNN *loses* to SchNet on the 2016 RMSE even
while winning overall, the node-side richness helping the hard sets while the geometry-side coarseness
costs it on the easy one. If that split appears — EGNN ahead on 2013 and 2019 but behind on 2016 RMSE —
it is not noise; it is the precise, mechanistic statement that the next rung must stop discarding the
11-dim edge geometry and feed the *full* angle/area/distance description into the message, recovering
SchNet's geometric resolution while keeping EGNN's node-pair expressiveness. That is the gap I expect
EGNN to leave, and it is the gap the next method has to close. (The full scaffold module is in the
answer.)
