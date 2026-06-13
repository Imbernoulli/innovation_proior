The SchNet numbers came back and they confirm exactly the under-informed-but-honest floor I expected,
which is useful because it tells me precisely where to push. On the 2013 core set it lands at RMSE
1.4765 / Rp 0.792, on 2016 at 1.3702 / 0.777, and on the 2019 temporal holdout at 1.5624 / 0.570 —
that last one is the tell. The 2019 holdout is the largest test set (4366 complexes) and the one whose
chemistry is most removed from training by the temporal split, and SchNet's Rp there collapses to
0.57, the worst correlation it posts anywhere. So the distance-only continuous filter generalizes
*least* exactly where the held-out structures demand the most from the geometry — which is what I
argued it would do. The model sees the interface and is invariant, but its single window onto geometry
is the scalar distance through the RBF; the angle and triangle-area statistics in the 11-dim edge
features never enter the message, and on near-training complexes (2016) it can lean on memorized motifs
to post a respectable 0.777, while on the temporally distant 2019 set, where it must actually
*reason* about contact geometry it has not seen, it has nothing extra to reason with and the
correlation falls apart. The diagnosis is sharp: the bottleneck is not the message-passing skeleton —
the heterogeneous four-convolution structure and the dual interface readout are doing their job — it
is what the message is *allowed to depend on*. SchNet's filter is a function of `d` alone, so two
contacts at the same distance but different orientation are identical to it, and binding is exquisitely
orientation-dependent. I need to widen what a message can condition on.

The cleanest next move is the equivariant-message-passing idea, because its whole premise is a
*richer, learned* dependence of the message on the endpoints and the geometry, rather than SchNet's
single distance-gated filter. Let me re-derive it from the symmetry, because the symmetry is what
makes it safe and also tells me which half of it I actually get to use on this edit surface. The
affinity is an invariant scalar, so I want every message to be invariant: rotate or translate the
complex and the message is unchanged. What does a rigid motion leave alone? Sending `r → Q r + t`, a
raw coordinate scrambles, a difference `r_i − r_j → Q(r_i − r_j)` still rotates, but the squared
distance `‖r_i − r_j‖²` is fixed because `QᵀQ = I` and `t` cancels in the difference. So the canonical
equivariant layer feeds its edge function the *invariant* squared distance alongside the node features:
`m_ij = φ_e(h_i, h_j, ‖r_i − r_j‖², a_ij)`. Where this differs from SchNet, and where its extra power
lives, is that the message is a full MLP of *both* endpoint feature vectors and the distance — not just
a distance-derived filter multiplying one neighbour. Two contacts at the same distance can now produce
different messages because the *atoms* differ, and the MLP can carve the distance dependence per
atom-pair type instead of through one shared radial filter. That is the widening I want.

Now the famous second half of the equivariant layer is the coordinate update: move each point along a
weighted sum of relative-difference vectors, `x_i ← x_i + C Σ_j (x_i − x_j) φ_x(m_ij)`, where the
weight `φ_x(m_ij)` is an *invariant scalar* read off the message, so `Q` factors out of the differences
and the update is equivariant. That update is the reason the method can emit a *vector* target — an
updated position, a velocity. But I have to ask whether I need it here, and the answer is a clean no,
twice over. First, my target is a single invariant *scalar*, `-logKd/Ki`; there is no vector to emit,
so the equivariant coordinate channel has nothing to do. There is even a uniqueness fact that makes
this not a loss: for a fixed node indexing, the pairwise distance matrix already determines the geometry
up to a rigid motion, so for an invariant target the distances carry all the geometric information the
difference *vectors* would — the coordinate update adds expressiveness only when the *output* must be
equivariant. Second, and decisively for this task, the harness does not hand me coordinates at all. I
get a `PLABatch` whose geometry is precomputed into invariant edge features; there is no `pos` to
update, no relative-difference vector to form. So the coordinate update is not merely unnecessary here —
it is *unimplementable* on this edit surface. I keep the half of the method I can use and that the
target wants: the invariant equivariant-style message, `φ_e` of the endpoints and the distance, summed
into the node update.

So I read the distance the only way the harness exposes it, `edge_attr[:, -1:] * 10`, giving a 1-dim
scalar distance per edge in angstroms, and that single scalar is the geometric input to every message.
The message itself I structure as the equivariant layer's edge function, decomposed into three additive
SiLU-MLP terms so the source atom, the destination atom, and the edge distance each get their own
learned transform before they are combined: `msg = mlp_u(x_src) + mlp_v(x_dst) + mlp_e(dist)`, summed
over neighbours, then a node MLP on the concatenation of the destination's own feature and the
aggregate, `node_mlp([x_dst, agg])`. SiLU throughout because, like SchNet's Softplus, it is the smooth
activation the equivariant layer uses on its invariant channels — and here every channel is invariant
(I never touch a coordinate), so there is no equivariance to endanger by a pointwise nonlinearity. This
is a strictly richer message than SchNet's: SchNet gated one projected neighbour by a distance filter;
here both endpoints *and* the distance pass through their own MLPs and add, so the message can express
"this kind of ligand atom meeting that kind of pocket atom at this separation" in a way a single radial
filter cannot.

Everything around the message stays the heterogeneous interface skeleton, because the SchNet rung
already showed that skeleton is not the bottleneck — I am swapping only the convolution. Four EGNN
convolutions per layer, one each for covalent-ligand, covalent-pocket, non-covalent ligand→pocket, and
non-covalent pocket→ligand, all computed in parallel from the same input features and summed per
destination node type: a pocket atom gets its covalent update plus the non-covalent update from the
ligand contacts pointing into it, a ligand atom gets its covalent update plus the pocket contacts.
Three layers, hidden width 256. The non-covalent edges feed the convolution the same 1-dim distance,
so the interface message passing is geometry-aware through that one scalar. The readout is the shared
dual bidirectional interface scorer the ladder uses: a per-contact triple product of projected source
atom, projected destination atom, and a projection of the contact distance, summed over a complex's
contacts in both directions, each direction corrected by an attention-normalized bias term whose
softmax over the complex's contacts kills the size-dependent offset of a raw sum, then averaged. I keep
the harness's plain MSE on the single `forward` output — no `compute_loss` hook — because, as with
SchNet, this fill produces one prediction.

Let me be honest about the one place EGNN here is *not* richer than SchNet, because it sets up exactly
what the next rung should fix. EGNN's geometric input is still only the scalar distance — the same
single number SchNet used, just consumed through additive endpoint MLPs instead of a radial filter. The
extra expressiveness is all on the *node* side (both endpoints get their own learned transform); the
*geometry* side is no richer than SchNet's, and certainly no richer than the full 11-dim edge feature
sitting unused in the batch. SchNet expanded that one distance into 60 RBF channels; EGNN feeds it raw
as a single channel through `mlp_e`. So on pure geometric resolution EGNN may actually be *thinner*
than SchNet — it trades SchNet's distance-resolution for node-pair expressiveness. Which way that trade
nets out is the empirical question this rung answers.

So here are the falsifiable expectations against SchNet's numbers. The node-pair-aware message should
help most where the chemistry of the atom pair carries signal that a distance filter alone misses, and
I expect that to show up first on the *temporal holdout*: SchNet's worst result was 2019 Rp 0.570, and
if letting both endpoints into the message buys real generalization, EGNN should lift 2019 the most —
I would expect its 2019 Rp to clear 0.60 and its 2019 RMSE to drop below SchNet's 1.5624, because that
set is where extra reasoning capacity, not memorization, decides the score. On 2013 I expect a clear
improvement too (richer messages should sharpen the small CASF-2013 set). The risk is 2016: SchNet's
60-channel RBF gave it fine distance resolution on the near-training core set where memorized motifs
pay off, and EGNN's single raw-distance channel is coarser there, so it is entirely possible EGNN
*loses* to SchNet on the 2016 RMSE even while winning overall — the node-side richness helping the hard
sets while the geometry-side coarseness costs it on the easy one. If that split appears — EGNN ahead on
2013 and 2019 but behind on 2016 RMSE — it is not noise; it is the precise statement that the next rung
must stop discarding the 11-dim edge geometry and feed the *full* angle/area/distance description into
the message, recovering SchNet's geometric resolution while keeping EGNN's node-pair expressiveness.
That is the gap I expect EGNN to leave, and it is the gap the next method has to close. (The full
scaffold module is in the answer.)
