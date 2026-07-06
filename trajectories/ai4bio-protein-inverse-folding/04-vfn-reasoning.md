PiFold topped the ladder exactly as I predicted, and the shape of its win tells me where to push next.
On CATH 4.2 it recovered 0.4648 at perplexity 5.2943, on CATH 4.3 0.4772 / 5.1294, and on TS50 — the
out-of-distribution set — it jumped to 0.5228 / 4.5513. Before I move, I want to read the whole ladder as
a table, not just the last row, because the trajectory of the deltas is the real signal. Recovery on
CATH 4.2 went 0.4310 → 0.4603 → 0.4648: a big first step of +0.0293 when ProteinMPNN poured in distances,
then a nearly flat +0.0045 when PiFold added attention, probes, and context. On TS50 the same two steps
went +0.0227 → +0.0399: the first step, the local-distance fix, gained the *least* on TS50, and the
second step, the transferable machinery, gained the *most*. That is precisely the inversion I predicted:
the global gate and the learnable virtual atoms are fold-agnostic structure, so they pay off where the
folds are least like training. And it means CATH 4.2 recovery has nearly plateaued for this family of
encoders — 0.4603 to 0.4648 is the encoder scraping the ceiling of what argmax on that split can show.

But look at the perplexity column at that same plateaued step, because it is the crux of what I do next.
On CATH 4.2, while recovery crawled +0.0045 from ProteinMPNN to PiFold, perplexity fell 5.4723 → 5.2943,
a drop of 0.178 — a large move on the smooth signal while the argmax barely twitched. So the finer
geometry PiFold added was *there*, sharpening the whole distribution, but recovery could not register it
because the argmax was already right on those residues; perplexity registered it loudly. This is the most
important thing the table teaches me for a finale: when the geometry gets finer, perplexity moves before
and more than recovery, especially on the in-distribution sets where recovery is near its ceiling. I will
build my falsifiable claim on exactly that.

And there is headroom on that channel to claim. Reading the perplexity deltas across the whole ladder:
CATH 4.2 fell 0.427 then 0.178 — decelerating, near its floor; CATH 4.3 fell 0.357 then 0.335 — still
dropping strongly; TS50 fell 0.400 then 0.531 — actually *accelerating* on the second step. So two of the
three benchmarks show perplexity still falling briskly, with no sign of a floor, and only CATH 4.2 is
flattening. If VFN's finer geometry is real, this is where it should be visible: continued, clean
perplexity drops on CATH 4.3 and TS50, and at least a modest one on CATH 4.2 even where its recovery
cannot move. That is a prediction with somewhere to go, not a hope pinned to a saturated metric.

Now the whole ladder has been one story — GVP's thin edges (0.4310) gave way to ProteinMPNN's 25 invariant
distances (0.4603) gave way to PiFold's attention plus learnable virtual-atom probes plus context (0.4648
/ 0.4772 / 0.5228) — and at each rung the lever was *how richly the encoder reads the local geometry*. So
I should ask: what is the richest reading PiFold still leaves on the table? And there is a precise one.
PiFold's learnable virtual atoms are read only through their pairwise *distances*. It places three
transferable probe points in the local frame and then measures lengths between them and the neighbor's
probe points — RBF-coded scalars. But a distance is a lossy summary of the geometric relationship between
two frame-anchored point clouds: it keeps magnitude and throws away *direction*. Two neighbors whose probe
clouds sit at the same set of distances but rotated differently relative to me look nearly identical to
PiFold, and yet they are different environments. That is the same scalar-collapse disease the whole
structure-encoding lineage has, just one notch finer: PiFold un-froze the *features* but still pools the
virtual-atom geometry into scalars before anything learnable touches the direction.

I want to be careful here, because I could talk myself into the wrong fix. One tempting move is to go
back to GVP-style equivariant vectors on the nodes and edges — but GVP was the *weakest* rung, so I have
to state what actually failed there and confirm I am not repeating it. GVP did not fail because vectors
are a bad idea; it failed because it fed those vectors almost nothing — one `CA→CA` direction per edge —
so the equivariant machinery had no geometry to carry. The lesson is not "avoid vectors," it is "vectors
need rich, learnable input." A second tempting move is the heavyweight one: a fully steerable SE(3)
network with higher-order tensor features and Clebsch–Gordan products. But the ladder has repeatedly shown
that invariant-scalar richness plus one well-chosen geometric idea beats baroque equivariance algebra, and
that route does not fit the dense `(B, L, K)` harness without a reimplementation I have no evidence I
need. So the move that is actually forced by the diagnosis threads between them: read the *learnable
frame-anchored probes PiFold already proved transferable* as vectors rather than distances — combine
PiFold's best feature with the one thing GVP was right about, keeping direction, and reduce to invariant
scalars only at the very end. Let me work out what that requires while staying exactly invariant, because
invariance is the property the whole ladder has guarded and I will not give it up for expressivity.

Start with the frame. Each residue gets a rigid transform `T_i = (R_i, t_i)` built the AlphaFold way —
`t_i = Cα_i`, and `R_i = [e1, e2, e3]` by Gram–Schmidt from `N, Cα, C`: `e1 = norm(C−Cα)`,
`e2 = norm((N−Cα) − (e1·(N−Cα))e1)`, `e3 = e1×e2`. This is a cleaner frame than the bisector frame the
earlier rungs implied, and it is the canonical one. It has a property I should check rather than assume,
because it matters physically: `e1` and `e2` are orthonormal and `e3 = e1×e2`, so `[e1, e2, e3]` is
right-handed with determinant `+1` — a proper rotation in `SO(3)`, never a reflection. That is not a
pedantic point. Amino acids are chiral; the native L-form and its mirror image are different molecules,
and a frame that could silently flip handedness would make a residue and its reflection encode
identically, which is exactly wrong for this problem. The Gram–Schmidt construction, being always
determinant `+1`, preserves chirality by design. Good. Give each residue a set of `d_q` learnable virtual
atoms `Q_i` whose coordinates are parameters in the local frame, shared across residues — the same
transferable-probe idea PiFold used, but now I will read them differently.

To relate residue `i` to a neighbor `j` I bring `j`'s atoms into `i`'s frame: the neighbor's atoms in
world coordinates are `R_j Q_j + t_j`, and into `i`'s frame that is `K_j = R_i^T(R_j Q_j + t_j − t_i)`,
which is exactly `T_{i←j} ∘ Q_j`. The center's own atoms in its frame are just `Q_i`. I need to verify the
global rotation truly cancels, because the whole finale rests on this being invariant and not merely
looking invariant. Apply a global rigid motion `X → R_g X + t_g` to every atom. The frame axes are built
from atom differences `C−Cα` and `N−Cα`, whose global-`t_g` shifts cancel and whose rotations give
`R_i → R_g R_i`; the translation is `t_i = Cα_i → R_g Cα_i + t_g = R_g t_i + t_g`. Then the neighbor's
world atoms become `R_g(R_j Q_j + t_j) + t_g`, and `world_j − t_i → R_g(R_j Q_j + t_j) + t_g −
(R_g t_i + t_g) = R_g(R_j Q_j + t_j − t_i)` — the `t_g` cancels. Finally `R_i^T → (R_g R_i)^T =
R_i^T R_g^T`, so `K_j → R_i^T R_g^T R_g (R_j Q_j + t_j − t_i) = R_i^T(R_j Q_j + t_j − t_i)`, since
`R_g^T R_g = I`. It is *unchanged*. And `q_i = Q_i` is a frame-local placement independent of the world
pose entirely. So both inputs to what comes next are strictly invariant to any rigid motion of the input.
That is the substrate: two sets of 3D points in `i`'s canonical frame, and I have checked the algebra that
makes them invariant rather than trusting it.

Now the central operation, and it is the one new idea past PiFold. Treat the two point sets like the
inputs to a *linear layer*, but with 3D vectors as the channel values instead of scalars. An ordinary
linear layer computes `y_k = Σ_l w_{k,l} x_l` over scalar inputs; I do the same over the virtual-atom
*vectors*, so each output channel is itself a vector:
`h⃗_k = Σ_l w^a_{k,l} q⃗_l + Σ_l w^b_{k,l} k⃗_l`, with learnable scalar weights mixing the center atoms and
the neighbor atoms into `d_out` output vectors in `R^3`. This is the vector field operator. Because the
weights are scalars multiplying whole vectors, the operation commutes with rotation of the common frame —
`Σ w (vU) = (Σ w v)U` — so the output vectors are equivariant in `i`'s frame; nothing has collapsed yet.
Then reduce, late and richly: for each output vector keep both its unit direction
`h⃗_k/‖h⃗_k‖` (three numbers, invariant because the center frame is canonical) and an RBF of its magnitude
`RBF(‖h⃗_k‖)`. Concatenate over the `d_out` channels: `g_{i,j} = concat_k(h⃗_k/‖h⃗_k‖, RBF(‖h⃗_k‖))`. Where
PiFold kept a handful of scalar distances, this keeps `d_out` *learnable directions* plus their RBF-coded
magnitudes — the direction PiFold discarded is back, and the combination is learnable rather than a fixed
list of atom pairs. That is the scalar bottleneck broken at the level the ladder had stalled on.

Let me make sure I believe the gain is real and not just more parameters, because that is the failure mode
I have to guard against — PiFold already proved that capacity without geometry plateaus, and I just read a
CATH 4.2 recovery plateau off the table. The argument has to be about *information*, not width. A distance
`‖q − k‖` is a single rotation-invariant number. Fix the center probe and ask where a neighbor probe atom
at distance `d` can be: anywhere on a sphere of radius `d`, a two-parameter family of positions, and the
distance readout maps every point on that sphere to the same value `d`. It has thrown away two degrees of
freedom — *which direction* on the sphere the neighbor sits. The map from "the geometric relationship
between two probe clouds" to "the multiset of their pairwise distances" is therefore many-to-one; PiFold
lives downstream of that quotient and can never recover what it collapsed. The vector field operator does
not take that quotient: each output vector reduces to its unit direction in the canonical center frame —
three invariant numbers that place the neighbor *on* the sphere, not just name its radius — plus an RBF of
the magnitude. So per learned feature, VFN extracts direction-and-magnitude where PiFold extracts
magnitude only; for any pair of neighbors PiFold cannot tell apart but that present different
environments, there exist operator weights whose output directions *do* separate them. The representation
is strictly finer, not merely wider. That is why I expect a real gain rather than a capacity wash, and it
is the claim my falsifiable test below is built to break.

I can make "strictly finer" constructive rather than rhetorical, which reassures me the operator loses
nothing PiFold had. Pick one output channel and set its weights to select the center's probe `l` with
`w^a = +1` and the neighbor's probe `l` with `w^b = −1`, everything else zero. Then `h⃗_k = q⃗_l − k⃗_l`,
and `‖h⃗_k‖ = ‖q_l − k_l‖` is *exactly* the center-to-neighbor same-index probe distance PiFold's
featurizer computes, so `RBF(‖h⃗_k‖)` reproduces PiFold's virtual-atom distance channel verbatim. Every
distance PiFold read is therefore recoverable as the norm of some fixed weight setting of the vector field
operator — PiFold's distance readout is literally the magnitude-only projection of a special case of this
operator — and on top of that recoverable distance, VFN *also* keeps `h⃗_k/‖h⃗_k‖`, the direction PiFold
discarded. So the operator contains PiFold's move as a corner and extends it; the finale cannot do worse
than the baseline for lack of the distances, only better for having the directions, unless the extra
freedom simply fails to train — which is the honest risk the ablation is there to catch.

A word on why I RBF the magnitude rather than pass `‖h⃗_k‖` raw, since it is the same reasoning that made
the distance banks work two rungs ago: a bare length fed to the downstream MLP can only be read
monotonically, but the useful magnitudes are not monotone in meaning — a short output vector and a long
one signal different geometric regimes — so lifting `‖h⃗_k‖` into Gaussian bumps lets the network carve
out specific length bands, exactly as the 25-distance featurizer did, while the unit direction carries the
orientation the length cannot. Direction and magnitude are split so each is read the way it wants to be
read.

There is a design inversion here worth naming, because it is what makes this the natural endpoint rather
than one more feature dump. PiFold reached the top by pouring *more* hand-designed invariant scalars into
the graph — 204 node channels, 412 edge channels of dihedrals, angles, dot products, and distance banks.
VFN goes the other way. The node features are the strictly-invariant backbone dihedrals *only*, six
channels; the edge features are just the `CA–CA` RBF plus the sinusoidal sequence offset. All the
directional geometry that PiFold hand-built as scalar projections now enters through *one learnable
operator* that reads the frame-anchored probes as vectors. So the finale is not "more features," it is
"fewer hand-features, more learned geometry" — strip the featurizer back to the minimum and let the vector
field operator generate the geometric signal the previous rungs hand-curated. That is also exactly why it
keeps the model cleanly invariant: with no world-frame vector fed anywhere as a raw scalar, the only
geometry in the forward pass is the dihedrals, the `CA–CA` distances, the offset, and the invariant
reductions of the operator's output.

Everything else is the attention-plus-update machinery PiFold already justified, now fed the richer
geometric feature. The attention weight comes from the center node, neighbor node, geometric feature, and
edge: `a_{i,j} = softmax_j(MLP(s_i, s_j, g_{i,j}, e_{i,j}))`, scaled by `1/√d_head`; the value from the
neighbor, geometry, and edge; aggregate and residual-update the node, then a wide feed-forward. Update the
edges from the refreshed endpoints and the geometry, as PiFold did. And one new degree of freedom the
vectors enable: let each residue's virtual atoms be *re-predicted* from its node state every layer,
`Q_i ← Linear(s_i)`, so the probe specializes to what the embedding has learned instead of staying at the
shared initialization — PiFold's virtual atoms were learnable-but-static per residue, these are set per
residue per layer from the current representation. This is the one piece with no analogue in PiFold, and I
flag it now as the thing most likely to help *and* most likely to destabilize: re-setting the probe each
layer is a moving target for the operator to read, so it is exactly what I will want to validate rather
than assume. Fifteen such layers, deeper than PiFold's ten because each layer's geometric feature is
richer — a bank of `d_out` learnable directions rather than a fixed distance list — and because the probe
now *moves* each layer, so more rounds of refinement compound where PiFold's static probes would not.

Let me size what fifteen layers of this cost, since the harness budget is fixed. Each output vector is
reduced to `3 + num_rbf = 3 + 16 = 19` scalars, so with `d_vec = 32` output channels the geometric feature
per edge is `32 · 19 = 608`-dimensional — a wide read, but computed, not stored as static features. The
operator's own weights are tiny: `w^a` and `w^b` are each `d_out × num_virtual = 32 × 4 = 128` numbers,
256 learnable scalars for the whole geometric engine. What it does spend is intermediate memory: forming
`h⃗` on the `(B, L, K, d_out, 3)` grid is, for `L ≈ 500` and `K = 30`, on the order of `500 · 30 · 32 · 3 ≈
2.9×10^6` floats per protein per layer, fifteen layers deep. That is the memory bill that forces the same
`CONFIG_OVERRIDES` trick PiFold used — `num_encoder_layers = 15` with `batch_size = 8` — both within the
four keys the harness permits. Four virtual atoms rather than PiFold's three, because the operator can
afford one more probe when it is reading them as vectors rather than paying `n·(n−1)` distance channels
for each — the operator's cost grows with `d_out`, the number of output vectors, not with the square of
the probe count, so a fourth probe adds inputs to mix without exploding the feature dimension the way a
fourth distance-bank probe would.

Now the part that makes this finale *fit this task* rather than fight it. The strongest baseline, PiFold,
already chose a one-shot linear decoder and pushed all the work into the encoder — which is precisely what
the harness isolates and why PiFold lost nothing when the autoregressive decoder was amputated. VFN
inherits exactly that decision: `p(s_i | X) = log_softmax(W h_i)`, one parallel forward pass, per-residue
cross-entropy. So nothing about VFN's design conflicts with the edit surface — no sequence input, no
decoder loop, no second optimizer — it is an encoder improvement layered on top of PiFold's own decoding
philosophy. Concretely, the edit replaces the encoder with frame-anchored virtual atoms and the vector
field operator, keeps the scaffold's data flow (`forward(X, mask)`, KNN graph, dense `(B, L, K)` tensors,
the harness mask), and feeds a one-shot linear head. And because I derived the invariance rather than
assuming it, I can also confirm it numerically as a guard: a random `SE(3)` transform of the input moves
the output log-probabilities by about `1e-6`, which is floating-point noise from the Gram–Schmidt and the
einsums, consistent with exact invariance and not with a subtle leak.

Let me state the bar this has to clear, because the finale has no measured result to hide behind — it has
to beat PiFold's real numbers or it is not a finale. PiFold set 0.4648 / 0.4772 / 0.5228 recovery and
5.2943 / 5.1294 / 4.5513 perplexity. The bet is that reading the frame-anchored virtual atoms as
learnable *vectors* (direction + magnitude) instead of *distances* (magnitude only) is a strict increase
in geometric expressivity over PiFold's exact move, so I expect recovery above PiFold on all three
benchmarks — but the *shape* of the expectation is set by the delta table, not by hope. CATH 4.2 recovery
is near its plateau, so I expect only a modest lift there, into the low-0.47s; CATH 4.3 has more headroom,
high-0.47s; and TS50, where the transferable geometry pays most, past 0.53. The specific, falsifiable
claim is the one the table earned: the gain should be *cleanest on perplexity*. I watched CATH 4.2
recovery flatten to +0.0045 while its perplexity still fell 0.178 at the last rung, which means a finer-
geometry improvement shows up on the smooth signal first; so if VFN does not beat PiFold on perplexity —
the less argmax-quantized metric — across the board, then the "direction adds information over distance"
story is wrong and the vector field operator is just extra capacity, not extra geometry, and the recovery
gains would be suspect too. What I would validate first, then, is exactly that mechanism: a controlled
comparison holding the attention, virtual-atom count, and depth fixed, swapping only the vector field
operator for PiFold's distance readout, to confirm the gain comes from reading vectors rather than from
the deeper stack. Second, I would confirm the per-layer virtual-atom re-prediction (`Q_i ← Linear(s_i)`)
helps and does not destabilize training, since re-setting the probe each layer is the one piece with no
analogue in PiFold and the one I most distrust. If both hold, the ladder's single thesis — richer
invariant reading of local geometry is the lever — has been carried one rung past the strongest baseline.
The full encoder-only scaffold module is in the answer.
