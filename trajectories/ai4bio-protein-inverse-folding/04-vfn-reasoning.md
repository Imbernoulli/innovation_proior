PiFold topped the ladder as I predicted, and the shape of its win tells me where to push. On CATH 4.2 it
recovered 0.4648 at perplexity 5.2943, on CATH 4.3 0.4772 / 5.1294, and on TS50 — the out-of-distribution
set — it jumped to 0.5228 / 4.5513. Read as a table, the trajectory of the deltas is the real signal.
Recovery on CATH 4.2 went 0.4310 → 0.4603 → 0.4648: a big first step of +0.0293 when ProteinMPNN poured in
distances, then a nearly flat +0.0045 when PiFold added attention, probes, and context. On TS50 the same
two steps went +0.0227 → +0.0399: the local-distance fix gained the *least* on TS50, the transferable
machinery the *most* — precisely the inversion I predicted, because the global gate and the learnable
virtual atoms are fold-agnostic structure that pays off where the folds are least like training. And CATH
4.2 recovery has nearly plateaued for this family of encoders — 0.4603 to 0.4648 is the encoder scraping the
ceiling of what argmax on that split can show.

But the perplexity column at that plateaued step is the crux. On CATH 4.2, while recovery crawled +0.0045,
perplexity fell 5.4723 → 5.2943, a drop of 0.178 — a large move on the smooth signal while the argmax barely
twitched. So the finer geometry PiFold added was *there*, sharpening the whole distribution, but recovery
could not register it because the argmax was already right on those residues; perplexity registered it
loudly. When the geometry gets finer, perplexity moves before and more than recovery, especially on the
in-distribution sets where recovery is near its ceiling. I build my falsifiable claim on exactly that.

And there is headroom on that channel. The perplexity deltas across the ladder: CATH 4.2 fell 0.427 then
0.178 — decelerating, near its floor; CATH 4.3 fell 0.357 then 0.335 — still dropping strongly; TS50 fell
0.400 then 0.531 — actually *accelerating* on the second step. So two of the three benchmarks show
perplexity still falling briskly with no sign of a floor, and only CATH 4.2 is flattening. If a finer
geometry is real, this is where it should be visible: continued clean perplexity drops on CATH 4.3 and TS50,
and at least a modest one on CATH 4.2 even where its recovery cannot move.

At each rung the lever was *how richly the encoder reads the local geometry*. So what is the richest reading
PiFold still leaves on the table? PiFold's learnable virtual atoms are read only through their pairwise
*distances*. It places three transferable probe points in the local frame and measures lengths between them
and the neighbor's probe points — RBF-coded scalars. But a distance is a lossy summary of the geometric
relationship between two frame-anchored point clouds: it keeps magnitude and throws away *direction*. Two
neighbors whose probe clouds sit at the same distances but rotated differently look nearly identical to
PiFold, and yet they are different environments. That is the same scalar-collapse disease the whole lineage
has, one notch finer: PiFold un-froze the *features* but still pools the virtual-atom geometry into scalars
before anything learnable touches the direction.

I want to be careful not to talk myself into the wrong fix. Going back to GVP-style equivariant vectors on
nodes and edges is tempting, but GVP was the *weakest* attempt — not because vectors are a bad idea, but
because it fed them almost nothing, one `CA→CA` direction per edge, so the equivariant machinery had no
geometry to carry. The lesson is "vectors need rich, learnable input," not "avoid vectors." The heavyweight
route — a fully steerable SE(3) network with higher-order tensor features and Clebsch–Gordan products — I
reject again: invariant-scalar richness plus one well-chosen geometric idea has beaten baroque equivariance
algebra at every step, and that route does not fit the dense `(B, L, K)` harness without a reimplementation
I have no evidence I need. So the move threads between them: read the *learnable frame-anchored probes
PiFold already proved transferable* as vectors rather than distances — PiFold's best feature combined with
the one thing GVP was right about, keeping direction, reducing to invariant scalars only at the very end.

Start with the frame. Each residue gets a rigid transform `T_i = (R_i, t_i)` built the AlphaFold way —
`t_i = Cα_i`, `R_i = [e1, e2, e3]` by Gram–Schmidt from `N, Cα, C`: `e1 = norm(C−Cα)`,
`e2 = norm((N−Cα) − (e1·(N−Cα))e1)`, `e3 = e1×e2`. This is a cleaner frame than the bisector frame I leaned
on earlier, and it is the canonical one. It has a property that matters physically: `e1` and `e2` are
orthonormal and `e3 = e1×e2`, so `[e1, e2, e3]` is right-handed with determinant `+1` — a proper rotation
in `SO(3)`, never a reflection. Amino acids are chiral; the native L-form and its mirror image are different
molecules, and a frame that could silently flip handedness would make a residue and its reflection encode
identically, which is exactly wrong here. Gram–Schmidt, being always determinant `+1`, preserves chirality
by design. Give each residue a set of `d_q` learnable virtual atoms `Q_i` whose coordinates are parameters
in the local frame, shared across residues — the same transferable-probe idea PiFold used, read differently.

To relate residue `i` to a neighbor `j` I bring `j`'s atoms into `i`'s frame: the neighbor's world atoms
`R_j Q_j + t_j` become `K_j = R_i^T(R_j Q_j + t_j − t_i)`, exactly `T_{i←j} ∘ Q_j`; the center's own atoms
in its frame are just `Q_i`. This is invariant under a global rigid motion `X → R_g X + t_g`: the frames
transform as `R_i → R_g R_i`, `t_i → R_g t_i + t_g`, so `world_j − t_i → R_g(R_j Q_j + t_j − t_i)` (the
`t_g` cancels) and the leading `R_i^T → R_i^T R_g^T` kills the `R_g` since `R_g^T R_g = I` — `K_j` is
unchanged, and `q_i = Q_i` is frame-local to begin with. So both inputs to what comes next are strictly
invariant: two sets of 3D points in `i`'s canonical frame.

Now the central operation, the one new idea past PiFold. Treat the two point sets like the inputs to a
*linear layer*, but with 3D vectors as the channel values instead of scalars. An ordinary linear layer
computes `y_k = Σ_l w_{k,l} x_l` over scalar inputs; I do the same over the virtual-atom *vectors*, so each
output channel is itself a vector: `h⃗_k = Σ_l w^a_{k,l} q⃗_l + Σ_l w^b_{k,l} k⃗_l`, learnable scalar weights
mixing the center atoms and the neighbor atoms into `d_out` output vectors in `R^3`. This is the vector
field operator. Because the weights are scalars multiplying whole vectors, it commutes with rotation of the
common frame — `Σ w (vU) = (Σ w v)U` — so the output vectors are equivariant in `i`'s frame; nothing has
collapsed yet. Then reduce, late and richly: for each output vector keep both its unit direction
`h⃗_k/‖h⃗_k‖` (three numbers, invariant because the center frame is canonical) and an RBF of its magnitude
`RBF(‖h⃗_k‖)`, concatenated over the `d_out` channels: `g_{i,j} = concat_k(h⃗_k/‖h⃗_k‖, RBF(‖h⃗_k‖))`. Where
PiFold kept a handful of scalar distances, this keeps `d_out` *learnable directions* plus their RBF-coded
magnitudes — the direction PiFold discarded is back, learnable rather than a fixed list of atom pairs.

I have to believe the gain is *information*, not width — PiFold already proved capacity without geometry
plateaus, and I just read a CATH 4.2 recovery plateau off the table. A distance `‖q − k‖` is a single
rotation-invariant number: fix the center probe and a neighbor probe at distance `d` can be anywhere on a
sphere of radius `d`, a two-parameter family, and the distance maps every point on that sphere to the same
`d` — it has thrown away two degrees of freedom, *which direction* on the sphere the neighbor sits. The map
from "the geometric relationship between two probe clouds" to "the multiset of their pairwise distances" is
many-to-one; PiFold lives downstream of that quotient and can never recover what it collapsed. The vector
field operator does not take that quotient: each output vector reduces to its unit direction in the
canonical center frame — three invariant numbers that place the neighbor *on* the sphere, not just name its
radius — plus an RBF of the magnitude. So per learned feature, VFN extracts direction-and-magnitude where
PiFold extracts magnitude only; for any pair of neighbors PiFold cannot tell apart but that present
different environments, there exist operator weights whose output directions *do* separate them. The
representation is strictly finer, not merely wider.

And "strictly finer" is constructive: set one output channel's weights to `w^a = +1` on the center's probe
`l`, `w^b = −1` on the neighbor's probe `l`, zero elsewhere, and `h⃗_k = q⃗_l − k⃗_l` with `‖h⃗_k‖` *exactly*
PiFold's same-index probe distance, so `RBF(‖h⃗_k‖)` reproduces its distance channel verbatim. Every PiFold
distance is thus the norm of some fixed weight setting of this operator, and on top of it VFN keeps
`h⃗_k/‖h⃗_k‖`, the direction PiFold discarded. So the operator contains PiFold's move as a corner and
extends it — it can only do worse if the extra freedom fails to train, which is the risk the ablation below
is there to catch. I RBF the magnitude rather than pass `‖h⃗_k‖` raw for the same reason the distance banks
needed it — a bare length reads only monotonically, while Gaussian bumps let the network carve out specific
length bands — and split direction from magnitude so each is read the way it wants to be.

There is a design inversion here worth naming. PiFold reached the top by pouring *more* hand-designed
invariant scalars into the graph — 204 node channels, 412 edge channels of dihedrals, angles, dot products,
distance banks. VFN goes the other way: the node features are the strictly-invariant backbone dihedrals
*only*, six channels; the edge features are just the `CA–CA` RBF plus the sinusoidal sequence offset. All
the directional geometry PiFold hand-built as scalar projections now enters through *one learnable operator*
that reads the frame-anchored probes as vectors. So the finale is "fewer hand-features, more learned
geometry" — strip the featurizer to the minimum and let the vector field operator generate the geometric
signal the previous rungs hand-curated. That is also exactly why it stays cleanly invariant: with no
world-frame vector fed anywhere as a raw scalar, the only geometry in the forward pass is the dihedrals, the
`CA–CA` distances, the offset, and the invariant reductions of the operator's output.

Everything else is the attention-plus-update machinery PiFold already justified, fed the richer geometric
feature. The attention weight comes from center node, neighbor node, geometric feature, and edge,
`a_{i,j} = softmax_j(MLP(s_i, s_j, g_{i,j}, e_{i,j}))` scaled by `1/√d_head`; the value from neighbor,
geometry, and edge; aggregate and residual-update the node, then a wide feed-forward. Update the edges from
the refreshed endpoints and the geometry, as PiFold did. And one new degree of freedom the vectors enable:
let each residue's virtual atoms be *re-predicted* from its node state every layer, `Q_i ← Linear(s_i)`, so
the probe specializes to what the embedding has learned instead of staying at the shared initialization —
PiFold's virtual atoms were learnable-but-static per residue, these are set per residue per layer. This is
the one piece with no analogue in PiFold, and the thing most likely to help *and* most likely to
destabilize: re-setting the probe each layer is a moving target for the operator to read, so it is exactly
what I will want to validate rather than assume. Fifteen such layers, deeper than PiFold's ten because each
layer's geometric feature is richer — a bank of `d_out` learnable directions rather than a fixed distance
list — and because the probe now *moves* each layer, so more rounds of refinement compound.

Size fifteen layers, since the budget is fixed. Each output vector reduces to `3 + num_rbf = 19` scalars, so
with `d_vec = 32` output channels the geometric feature per edge is `32 · 19 = 608`-dimensional — a wide
read, but computed, not stored as static features. The operator's own weights are tiny: `w^a` and `w^b` are
each `d_out × num_virtual = 32 × 4 = 128` numbers, 256 learnable scalars for the whole geometric engine.
What it spends is intermediate memory: forming `h⃗` on the `(B, L, K, d_out, 3)` grid is, for `L ≈ 500` and
`K = 30`, on the order of `2.9×10^6` floats per protein per layer, fifteen deep — the memory bill that
forces the same `CONFIG_OVERRIDES` trick PiFold used, `num_encoder_layers = 15` with `batch_size = 8`, both
within the four keys the harness permits. Four virtual atoms rather than PiFold's three, because the
operator's cost grows with `d_out`, the number of output vectors, not with the square of the probe count —
so a fourth probe adds inputs to mix without exploding the feature dimension the way a fourth distance-bank
probe would.

Now the part that makes this finale *fit this task*. PiFold already chose a one-shot linear decoder and
pushed all the work into the encoder — precisely what the harness isolates, and why PiFold lost nothing when
the autoregressive decoder was amputated. VFN inherits exactly that decision: `p(s_i | X) =
log_softmax(W h_i)`, one parallel forward pass, per-residue cross-entropy. So nothing about VFN's design
conflicts with the edit surface — no sequence input, no decoder loop, no second optimizer — it is an encoder
improvement layered on PiFold's decoding philosophy. The edit replaces the encoder with frame-anchored
virtual atoms and the vector field operator, keeps the scaffold's data flow (`forward(X, mask)`, KNN graph,
dense `(B, L, K)` tensors, the harness mask), and feeds a one-shot linear head. And because I derived the
invariance rather than assuming it, I confirm it numerically as a guard: a random `SE(3)` transform of the
input moves the output log-probabilities by about `1e-6`, floating-point noise from the Gram–Schmidt and
einsums, consistent with exact invariance and not a subtle leak.

The bar, since the finale has no measured result to hide behind — it must beat PiFold's real numbers or it
is not a finale. PiFold set 0.4648 / 0.4772 / 0.5228 recovery and 5.2943 / 5.1294 / 4.5513 perplexity.
Reading the frame-anchored probes as learnable *vectors* (direction + magnitude) instead of *distances*
(magnitude only) is a strict increase in geometric expressivity, so I expect recovery above PiFold on all
three benchmarks — but the *shape* is set by the delta table: CATH 4.2 is near its plateau, so only a modest
lift into the low-0.47s; CATH 4.3 has headroom, high-0.47s; TS50, where transferable geometry pays most,
past 0.53. The specific falsifiable claim the table earned is that the gain should be *cleanest on
perplexity* — I watched CATH 4.2 recovery flatten to +0.0045 while its perplexity still fell 0.178, so a
finer-geometry improvement shows on the smooth signal first; if VFN does not beat PiFold on perplexity across
the board, the "direction adds information over distance" story is wrong and the operator is just extra
capacity. What I would validate first, then, is exactly that mechanism: a controlled comparison holding
attention, virtual-atom count, and depth fixed, swapping only the vector field operator for PiFold's
distance readout, to confirm the gain comes from reading vectors rather than the deeper stack. Second, that
the per-layer virtual-atom re-prediction helps and does not destabilize training, since re-setting the probe
each layer is the one piece with no analogue in PiFold and the one I most distrust. If both hold, the single
thesis — richer invariant reading of local geometry is the lever — has been carried one step past the
strongest baseline.
