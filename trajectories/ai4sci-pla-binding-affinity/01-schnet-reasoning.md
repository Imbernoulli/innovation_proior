The architecture is the whole contribution here, and before I reach for anything clever I want a
floor — the simplest fill of the `AffinityModel` slot that actually takes the binding problem
seriously, so the methods that follow have something to beat and something to diagnose against. The
scaffold default is not that floor: it runs two independent encoders over the ligand and the pocket,
mean-pools each, concatenates, and regresses, and it *ignores the inter-molecular edges entirely*.
That is a non-starter physically, and it is worth saying exactly why, because the why is the whole
problem. Binding affinity is a free energy, and the free energy of binding comes almost entirely from
the non-covalent contacts *across the interface* — hydrogen bonds, salt bridges, van der Waals
packing, π-stacking — each between a particular ligand atom and a particular pocket atom. The covalent
skeletons inside each molecule exist whether or not the ligand is in the pocket; they set the
conformation but they are not the binding. A model that encodes the two molecules separately and never
lets a ligand atom see the pocket atom it is touching has thrown away the one thing that determines
the answer. So the default's `-logKd/Ki` can only be guessed from the *marginal* shapes of the two
molecules, which is exactly the dataset shortcut I distrust. My floor has to at least let the interface
edges into the message passing.

So what is the simplest principled fill? Let me think about what physics will not let me get away
with, because that pins the design more than any architectural taste. The affinity is a scalar
property of the *arrangement* of atoms, and it cannot change if I pick up the whole complex and
translate it or rotate it. If I slide ligand and protein together by a vector `t` or spin them by a
rotation `Q`, the answer must be byte-for-byte the same. A model that gives two different affinities
for the same complex viewed from two angles is predicting something that is not physics. I refuse to
learn that invariance from augmentation — rotating every complex a hundred ways and hoping the network
smooths it out is expensive and only ever approximate. I want invariance by construction. What
survives a rigid motion? Send every position to `Q r + t`. A raw coordinate changes completely. A
difference of two positions `(Q r_i + t) − (Q r_j + t) = Q(r_i − r_j)` loses the translation but still
rotates. Its length, though, `‖Q(r_i − r_j)‖ = √((r_i−r_j)ᵀ Qᵀ Q (r_i−r_j)) = ‖r_i − r_j‖` because
`QᵀQ = I`, is invariant to translation and to rotation and reflection both. The interatomic *distance*
is the free invariant the geometry hands me. If I build everything out of distances I am invariant,
full stop; if I let a raw coordinate or a bare difference vector leak in, I break it.

Now I look at what this task's harness actually exposes, because the edit surface is more restrictive
than a from-scratch geometric net would assume, and that restriction shapes the floor. The model is
*not* handed raw 3D coordinates. It is handed a `PLABatch` whose edges already carry precomputed,
rigid-motion-invariant geometry: the intra-molecular edges have 17 features (bond chemistry plus 11
geometric numbers — angle and triangle-area and neighbour-distance statistics, plus pairwise L1 and L2
distances), and the inter-molecular edges have those same 11 geometric numbers. The invariance work is
already done for me in the features; I do not get to recompute distances from positions because there
are no positions. The last geometric channel is the L2 distance (scaled by 0.1), so if a layer wants a
scalar distance per edge it reads `edge_attr[:, -1:] * 10` and gets back angstroms. That is the only
handle on geometry the harness gives, and it is enough to build a distance-driven message.

So which geometric-GNN idea do I instantiate first? The cleanest distance-driven message passing is
the continuous-filter convolution: expand the scalar distance in a bank of Gaussian radial basis
functions, turn that expansion into a per-edge filter, gate the neighbour's features elementwise by
the filter, sum over neighbours. Why the RBF expansion and not the bare distance straight into an MLP?
Because a fresh MLP is nearly linear at init, so feeding it one scalar makes every output channel come
out as nearly the same near-linear ramp in `d`. The filter channels are all correlated, there is no
diversity, and training crawls on a plateau because the filter is effectively one-dimensional. A bank
of Gaussians centered on a grid of distances fixes that: a short distance lights up the near centers,
a long distance the far ones, so the inputs to the filter map are already decorrelated and the filter
starts diverse. I lay the centers from 0 to 6 Å at 0.1 Å spacing — fine enough to resolve a bond
length (~1.5 Å) from a contact distance (out to the 5 Å cutoff) — which gives 60 RBF channels. The
filter map is a small two-layer MLP with a Softplus in the middle, and the message is `node_proj(src) *
filter(rbf)`, summed over neighbours, with the destination's own feature added back as a residual and
a Softplus output MLP. Softplus rather than ReLU is the canonical choice for this filter family — it is
the smooth cousin of ReLU, and a response to a continuous physical distance ought to be smooth, no
kinks. Each interaction block reads its distance off `edge_attr[:, -1:] * 10`, expands it, and runs
the continuous filter; only the invariant distance enters, so every block is invariant and the whole
predictor is invariant by construction, exactly as the physics demanded — with zero augmentation.

But the original continuous-filter convolution was built for a *single homogeneous molecule* where
every edge is the same kind of edge, processed by one shared filter. My complex is not homogeneous: it
is two molecules joined by an interface, with two physically distinct kinds of edge — stiff covalent
bonds around a bond length apart, and soft non-covalent contacts reaching out to 5 Å. If I throw both
into one filter I am asking a single distance-to-filter map to model two different physical regimes at
once, and the homogeneous net's whole limitation was exactly that blur. The interaction-graph lineage
this task lives in already settled the skeleton that fixes this, and I will adopt it rather than
reinvent it: keep one node set per molecule but four *separate* convolutions per layer — covalent on
the ligand, covalent on the pocket, non-covalent ligand→pocket, non-covalent pocket→ligand — all
computed in parallel from the same input features, then for each destination node type sum the
contributions that land on it. A pocket atom receives its own-molecule covalent update plus the
non-covalent update from the ligand contacts pointing into it; a ligand atom receives its covalent
update plus the non-covalent update from pocket contacts. Sum, because covalent and non-covalent
influences on an atom are additive, and using the same input features for both branches cleanly
separates "what my own molecule tells me" from "what my binding partner tells me." Each of the four
convolutions is its own continuous-filter block with its own filter map, so the covalent and
non-covalent distance regimes get their own responses. Three layers — the contact graph is a local
5 Å shell, and three rounds already carry a protein atom's influence a few bonds into the ligand and
back, building a joint covalent-plus-non-covalent context without oversmoothing the whole complex into
mush. Hidden width 256.

Now the readout, and here I deliberately do *not* fall back to "pool all atoms and regress," because
the interaction-graph skeleton offers something better and it is the part that makes the floor
meaningful: read the affinity off the interface. For each non-covalent contact I score a per-contact
affinity from a low-rank triple product of the projected source atom, the projected destination atom,
and the projected edge geometry — `e_proj ⊙ src_proj ⊙ dst_proj`, collapsed to a scalar by a final
linear — and sum the scalars over a complex's contacts, in both directions (ligand→pocket and
pocket→ligand). A raw sum over a variable number of contacts carries a size-dependent offset
(complexes with more atoms simply have more contacts in the shell), so I subtract an attention-weighted
bias correction whose softmax over a complex's contacts normalizes that offset away, per direction.
Average the two directional estimates for the prediction. This is the shared readout the whole ladder
uses; for the SchNet rung it sits on top of continuous-filter message passing, and the edge geometry
fed into the scoring is the same RBF expansion of the contact distance. I keep the harness's plain
MSE loss here — the dual heads exist, but the SchNet fill predicts a single `forward` output, so the
harness regresses that against the label, and I do not expose a `compute_loss` hook.

One honest note about what this fill is *not*. It is the canonical continuous-filter convolution's
geometric core — RBF, Softplus filter, elementwise gating — but it is not the canonical SchNet
end-to-end: there are no forces to train (the target is a single scalar, not an energy whose gradient
is a force), so the second-derivative machinery and the energy-plus-force loss are irrelevant here,
and I drop the cosine cutoff because the contacts are already capped at 5 Å by the graph construction
and the RBF bank only reaches 6 Å, so there is no neighbour drifting across a boundary mid-training to
smooth. What survives is the move that matters: distance → Gaussian expansion → learned filter →
elementwise gate, slotted into the heterogeneous interface skeleton.

Now reason about what this floor should do, because that is the entire point of running it. The model
sees the interface (unlike the default) and is invariant (unlike feeding coordinates raw), but its
only window onto geometry is the *scalar distance* through the RBF. The 11 geometric numbers per edge
include angles and triangle areas and neighbour-distance statistics — a rich local description — and
the continuous filter ignores all of it except the single L2 distance channel. So I expect this to be
a competent but *under-informed* predictor: it should beat the interface-blind default comfortably,
but it should sit at the bottom of any rung that actually uses the full edge geometry or treats the
covalent and non-covalent message passing with more chemistry-specific structure. The distance-only
filter is a real ceiling: two contacts at the same distance but different orientations are
indistinguishable to it, and binding cares about orientation. So my falsifiable expectation is that
SchNet lands as the weakest of the geometric fills — worst RMSE and worst Rp on the harder benchmarks
where the held-out chemistry rewards richer geometry — and that the next rung's gain comes precisely
from letting more than the scalar distance into the message. If SchNet instead matched the
edge-feature-rich methods, that would tell me the angle/area geometry is not pulling its weight on this
data and I would stop adding it; I expect the opposite. That is the diagnosis I want the first set of
numbers to either confirm or overturn. (The full scaffold module is in the answer.)
