The SchNet numbers came back and they confirm the under-informed-but-honest floor I expected, which
tells me precisely where to push. On the 2013 core set it lands at RMSE 1.4765 / Rp 0.792, on 2016 at
1.3702 / 0.777, and on the 2019 temporal holdout at 1.5624 / 0.570. Two things in that table are worth
pulling apart. First, RMSE across benchmarks is not monotone in naive difficulty: 2013 (107 complexes)
posts a worse RMSE than 2016 (285), yet 2013's Rp (0.792) is *higher* than 2016's (0.777). That only
looks contradictory until I remember the two metrics measure different things. Rp is scale- and
offset-free — it asks whether the predicted ranking tracks the true ranking, invariant to any affine
rescaling `p → a·p + b` — while RMSE is absolute and punishes any miscalibration of scale or bias. So a
regressor that ranks the 107 CASF-2013 complexes well but *compresses* its predicted range toward the
training mean (the textbook behaviour of an over-regularized model on a small, shifted set) shows a high
Rp and an inflated RMSE at once: range-compression alone can inject on the order of a unit of RMSE
independent of ranking quality. That is the 2013 signature — calibration, not comprehension — so I
should not expect the node-richness move to shift 2013 RMSE much unless it also sharpens the dynamic
range, a caution I hold against whatever comes back.

The alarming number is 2019: RMSE 1.5624 is the worst it posts anywhere and Rp 0.570 the worst
correlation anywhere, both failing together, a drop in Rp from 2016 of 0.207 and an RMSE gap of 0.192 —
a coherent collapse, not a wobble. And it points cleanly at the mechanism. The 2019 holdout is the
largest test set (4366 complexes) and the one whose chemistry is most removed from training by the
temporal split, so it is precisely where the model cannot lean on memorized motifs and must *reason*
about contact geometry it has not seen. SchNet sees the interface and is invariant, but its single
window onto geometry is the scalar distance through the RBF; the angle and triangle-area statistics
never enter the message. On near-training complexes it can recall distance signatures and post a
respectable 0.777; on the temporally distant set, where recall is worthless, it has nothing extra to
reason with and the correlation falls apart. The diagnosis exonerates the parts I worried about: the
bottleneck is *not* the message-passing skeleton — the near-training 2016 number is fine, so the
four-convolution structure and the dual readout are doing their job — it is what the message is
*allowed to depend on*. SchNet's filter is a function of `d` alone, so two contacts at the same distance
but different orientation are identical to it, and binding is exquisitely orientation-dependent. I need
to widen what a message can condition on, along *one* axis at a time so the next number is attributable.

Lay out the axes. One: make the message depend on *both endpoint atoms*, not just one gated neighbour.
Two: make it depend on *more of the geometry* than the scalar distance — the 11-dim angle/area/neighbour
statistics. Three: change how neighbours are *aggregated* — attention instead of a filtered sum. And a
tempting fourth: go fully *equivariant*, carrying vector features per node and updating coordinates.
SchNet is thin on axes one and two both, and if I push several at once I will not know which bought the
gain. I can eliminate axis four on hard grounds rather than taste (below), and axis three is a
readout-side lever I do not want to entangle with the head's existing attention here. So I hold the
geometry input fixed at the scalar distance and move only axis one, node-pair expressiveness — which
keeps a clean two-step ablation in reserve: if node-richness-at-fixed-geometry helps the hard set but
costs the easy one, the next method's job (add the geometry back) is already written by the result.

The cleanest widening along the node axis is the equivariant-message-passing edge function, whose whole
premise is a learned dependence of the message on the endpoints and the geometry rather than SchNet's
single distance-gated filter. The invariant that feeds it is the same distance as before — `‖r_i −
r_j‖²` is fixed under `r → Q r + t` — so the canonical layer forms `m_ij = φ_e(h_i, h_j, ‖r_i − r_j‖²,
a_ij)`. Where this beats SchNet is that the message is a full function of *both* endpoint feature vectors
and the distance, not a distance filter multiplying one neighbour: two contacts at the same distance can
now produce different messages because the *atoms* differ. That is exactly axis one.

The famous second half of the layer is the coordinate update, `x_i ← x_i + C Σ_j (x_i − x_j) φ_x(m_ij)`,
whose weight is an invariant scalar so `Q` factors out and the update is equivariant. That is the reason
the method can emit a *vector* target — and I need none of it here, on three stacked grounds. My target
is a single invariant *scalar*, so there is no vector to point at. Dropping the update costs no
expressiveness in principle: for a fixed node indexing the pairwise-distance matrix determines the
geometry up to a rigid motion — the double-centering identity `B = −½ J D J` (with `J = I − (1/n)11ᵀ`)
recovers the centered Gram matrix `X_c X_cᵀ` from the squared distances `D`, whose eigendecomposition
returns the coordinates up to `O(3)`, so distances carry every bit of geometric information the
difference vectors would for any target that must itself be `O(3)`-invariant; the coordinate update adds
expressiveness only when the *output* is required to be equivariant. And decisively, the harness hands me
no coordinates — no `pos` to update, no difference vector to form — so the update is unimplementable here
anyway. All three point the same way: keep the invariant equivariant-style message, `φ_e` of the
endpoints and the distance, summed into the node update.

So I read the distance the only way exposed, `edge_attr[:, -1:] * 10`, and structure the message as
three additive SiLU-MLP terms so source atom, destination atom, and edge distance each get their own
learned transform before combining: `msg = mlp_u(x_src) + mlp_v(x_dst) + mlp_e(dist)`, summed over
neighbours, then a node MLP on the concatenation of the destination's own feature and the aggregate,
`node_mlp([x_dst, agg])`. SiLU throughout — the smooth activation the layer uses on its invariant
channels, and here every channel is invariant so no pointwise nonlinearity endangers anything.

I should be honest about how much node-pair richness this additive form buys, because it is less than the
slogan. The canonical edge function `φ_e(h_i, h_j, d)` is a *joint* map that can form arbitrary cross
terms between the two atoms — "this donor meeting that acceptor" genuinely different from the sum of
"this donor to anything" and "anything to that acceptor." My decomposition is *separable*: the three
terms add, they do not multiply, so at message formation the source and destination cannot interact — no
`src ⊗ dst` cross term. What it really provides over SchNet is that *both* endpoints get to speak through
their own transform, rather than only the source being gated. The true multiplicative atom-pair
interaction enters later — partly through `node_mlp([x_dst, agg])`, which mixes the destination's feature
with the aggregate nonlinearly, and decisively through the readout's per-contact triple product, which
*is* multiplicative in source, destination, and edge. So the widening is real but modest and mostly on
the "both endpoints heard" axis, with the sharp pairwise interaction deferred to the head — a reason to
temper how large a gain I predict, and another argument for holding the geometry axis fixed, since piling
the 11-dim features into a still-separable message would confound "more geometry" with "still no cross
term."

There is one place this is *not* richer than SchNet, and it sets up the next move.
EGNN's geometric input is still only the scalar distance, and feeding one scalar into
`mlp_e = Linear(1, H) → SiLU → …` reintroduces exactly the rank-one-at-init degeneracy the RBF was
invented to remove — the first layer's `H` channels are all affine in the same `d`, one effective degree
of freedom, where SchNet's 60 decorrelated RBF bumps gave fine distance resolution from step one. So on
pure geometric resolution EGNN is *thinner* than SchNet: it trades distance-resolution for node-pair
expressiveness. Whether that trade nets out positive is the empirical question here.

Everything around the message stays the heterogeneous interface skeleton, since SchNet showed the
skeleton is not the bottleneck — I swap only the convolution. Four EGNN convolutions per layer (covalent
ligand, covalent pocket, non-covalent ligand→pocket, pocket→ligand), computed in parallel from the same
inputs and summed per destination type; three layers, width 256, the same receptive-field-versus-
oversmoothing reasoning as before. On parameters this is roughly a wash: where SchNet spent per-block
budget on a 60-input filter and an output MLP, EGNN spends it on the second endpoint MLP — I am
*reallocating* SchNet's geometry capacity to the node side, not buying richness with extra capacity. The
readout is the shared dual bidirectional interface scorer — per-contact triple product, summed both
directions, each corrected by an attention-normalized bias, averaged — except the edge projection into
the score is now a `Linear(1, H)` of the raw distance rather than a `Linear(60, H)` of the RBF, the same
thinning carried consistently into the head. One `forward` output, so plain MSE, no `compute_loss`.

Invariance survives end to end by the same argument as SchNet, only easier: the only geometric quantity
that enters is the invariant distance, the node features are chemical one-hots, and every downstream
operation is a function of those invariant inputs with no coordinate ever formed. And the additive
message buys exactly what I claim on the node axis: two contacts at the same 3.0 Å — a ligand carbonyl
oxygen on a pocket amide N–H versus a ligand methyl carbon on a pocket aromatic carbon — get the
identical SchNet gate `f(3.0)`, but here they share only `mlp_e(3.0)` while `mlp_u` and `mlp_v` fire
differently on oxygen-vs-carbon and amide-N-vs-aromatic-C, so the polar and nonpolar contacts separate
in message space despite equal distance. That is partner-*identity* discrimination, not orientation —
the orientation blindness I set out against is still wide open.

So the falsifiable expectations against SchNet's numbers follow directly from the two sides of the
trade. The node-pair-aware message should help most where the atom-pair chemistry carries signal a
distance filter misses, and that is where reasoning beats recall — the temporal holdout. So I expect
EGNN to lift 2019 the most, its RMSE dropping below SchNet's 1.5624 and its Rp rising off SchNet's 0.570, and
to improve 2013 as richer messages sharpen the small-set ranking. The risk lives on 2016: SchNet's
60-channel RBF gave fine distance resolution on the near-training core set where memorized distance
signatures pay off, and EGNN's single raw-distance channel is coarser there, so it may *lose* to SchNet
on 2016 RMSE even while winning overall. If that split appears — ahead on 2013 and 2019, behind on 2016
RMSE — it is not noise; it is the mechanistic statement that the next method must stop discarding the
11-dim edge geometry and feed the full angle/area/distance description into the message, recovering
SchNet's resolution while keeping EGNN's node-pair expressiveness. (The full scaffold module is in the
answer.)
