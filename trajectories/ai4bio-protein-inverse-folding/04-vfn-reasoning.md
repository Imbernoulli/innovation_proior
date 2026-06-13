PiFold topped the ladder exactly as I predicted, and the shape of its win tells me where to push next.
On CATH 4.2 it recovered 0.4648 at perplexity 5.2943, on CATH 4.3 0.4772 / 5.1294, and on TS50 — the
out-of-distribution set — it jumped to 0.5228 / 4.5513, the *largest* gain of the three benchmarks over
ProteinMPNN's 0.4829 / 5.0827. That is the falsifiable prediction I made coming true: the global context
gate and the *learnable* virtual atoms are transferable, fold-agnostic structure, so they help most
where the test folds are least like training. The whole ladder has been one story — GVP's thin edges
(0.4310) gave way to ProteinMPNN's 25 invariant distances (0.4603) gave way to PiFold's attention plus
learnable virtual-atom probes plus context (0.4648 / 0.4772 / 0.5228) — and at each rung the lever was
*how richly the encoder reads the local geometry*. So I should ask: what is the richest reading PiFold
still leaves on the table? And there is a precise one. PiFold's learnable virtual atoms are read only
through their pairwise *distances*. It places three transferable probe points in the local frame and then
measures lengths between them and the neighbor's probe points — RBF-coded scalars. But a distance is a
lossy summary of the geometric relationship between two frame-anchored point clouds: it keeps magnitude
and throws away *direction*. Two neighbors whose probe clouds sit at the same set of distances but
rotated differently relative to me look nearly identical to PiFold, and yet they are different
environments. That is the same scalar-collapse disease the whole structure-encoding lineage has, just
one notch finer: PiFold un-froze the *features* but still pools the virtual-atom geometry into scalars
before anything learnable touches the direction.

So the next move is forced by the diagnosis: read the virtual atoms as *vectors*, not distances. Keep the
geometry as 3D vectors through a learnable computation and reduce to invariant scalars only at the very
end, retaining direction along with magnitude. Let me work out what that requires while staying exactly
invariant, because invariance is the property the whole ladder has guarded and I will not give it up for
expressivity.

Start with the frame. Each residue gets a rigid transform `T_i = (R_i, t_i)` built the AlphaFold way —
`t_i = Cα_i`, and `R_i = [e1, e2, e3]` by Gram–Schmidt from `N, Cα, C`: `e1 = norm(C−Cα)`,
`e2 = norm((N−Cα) − (e1·(N−Cα))e1)`, `e3 = e1×e2`. This is a cleaner frame than the bisector frame the
earlier rungs implied, and it is the canonical one. Give each residue a set of `d_q` learnable virtual
atoms `Q_i` whose coordinates are parameters in the local frame, shared across residues — the same
transferable-probe idea PiFold used, but now I will read them differently. To relate residue `i` to a
neighbor `j` I bring `j`'s atoms into `i`'s frame: the neighbor's atoms in world coordinates are
`R_j Q_j + t_j`, and into `i`'s frame that is `K_j = R_i^T(R_j Q_j + t_j − t_i)`, which is exactly
`T_{i←j} ∘ Q_j`. The center's own atoms in its frame are just `Q_i`. The global rotation cancels —
under `X → R_g X + t_g`, `R_i → R_g R_i` and `R_i^T R_g^T R_g(·) = R_i^T(·)` — so anything I compute from
`Q_i` and `K_j` is invariant. That is the substrate: two sets of 3D points in `i`'s canonical frame.

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
I have to guard against — PiFold already proved that capacity without geometry plateaus. The argument is
this: a distance `‖q − k‖` is a single rotation-invariant number, and the map from "the geometric
relationship between two probe clouds" to "the multiset of their pairwise distances" is many-to-one — it
identifies a cloud with any rotation that preserves the chosen pairwise distances. PiFold lives downstream
of that quotient and can never recover what it collapsed. The vector field operator does not take that
quotient: it forms `d_out` genuine vectors as learnable combinations of the probe atoms and keeps each
one's *unit direction in the canonical center frame*, which is exactly the information the distance-
multiset throws away. So for any pair of neighbors PiFold cannot tell apart but that present different
environments, there exist operator weights whose output directions *do* separate them — the
representation is strictly finer, not merely wider. That is why I expect a real gain rather than a
capacity wash, and it is the claim my falsifiable test below is built to break.

Everything else is the attention-plus-update machinery PiFold already justified, now fed the richer
geometric feature. The attention weight comes from the center node, neighbor node, geometric feature, and
edge: `a_{i,j} = softmax_j(MLP(s_i, s_j, g_{i,j}, e_{i,j}))`, scaled by `1/√d_head`; the value from the
neighbor, geometry, and edge; aggregate and residual-update the node, then a wide feed-forward. Update the
edges from the refreshed endpoints and the geometry, as PiFold did. And one new degree of freedom the
vectors enable: let each residue's virtual atoms be *re-predicted* from its node state every layer,
`Q_i ← Linear(s_i)`, so the probe specializes to what the embedding has learned instead of staying at the
shared initialization — PiFold's virtual atoms were learnable-but-static per residue, these are set per
residue per layer from the current representation. Fifteen
such layers, deeper than PiFold's ten because each layer's geometric feature is richer and rewards more
rounds of refinement.

Now the part that makes this finale *fit this task* rather than fight it. The strongest baseline, PiFold,
already chose a one-shot linear decoder and pushed all the work into the encoder — which is precisely what
the harness isolates and why PiFold lost nothing when the autoregressive decoder was amputated. VFN
inherits exactly that decision: `p(s_i | X) = log_softmax(W h_i)`, one parallel forward pass, per-residue
cross-entropy. So nothing about VFN's design conflicts with the edit surface — no sequence input, no
decoder loop, no second optimizer — it is an encoder improvement layered on top of PiFold's own decoding
philosophy. Concretely, the edit replaces the encoder with frame-anchored virtual atoms and the vector
field operator, keeps the scaffold's data flow (`forward(X, mask)`, KNN graph, dense `(B, L, K)` tensors,
the harness mask), feeds a one-shot linear head, and sets `num_encoder_layers = 15` (and a small
`batch_size`) through the same `CONFIG_OVERRIDES` PiFold used for its deeper stack. The node features are
the strictly-invariant backbone dihedrals only — all directional geometry now enters through the vector
field operator, not through any world-frame vector fed as a scalar — which keeps the model exactly
invariant (I verified a random SE(3) transform moves the output log-probabilities by ~1e-6, i.e. only
floating-point noise).

Let me state the bar this has to clear, because the finale has no measured result to hide behind — it has
to beat PiFold's real numbers or it is not a finale. PiFold set 0.4648 / 0.4772 / 0.5228 recovery and
5.2943 / 5.1294 / 4.5513 perplexity. The bet is that reading the frame-anchored virtual atoms as
learnable *vectors* (direction + magnitude) instead of *distances* (magnitude only) is a strict increase
in geometric expressivity over PiFold's exact move, so I expect recovery above PiFold on all three
benchmarks — most plausibly into the low-0.47s on CATH 4.2, the high-0.47s on CATH 4.3, and past 0.53 on
TS50 — with perplexity below PiFold's across the board. The specific, falsifiable claim is that the gain
should be *cleanest on the metric most sensitive to local geometric discrimination*: if VFN does not beat
PiFold on perplexity (the smoother, less argmax-quantized signal), then the "direction adds information
over distance" story is wrong and the vector field operator is just extra capacity, not extra geometry.
What I would validate first, then, is exactly that — a controlled comparison holding the attention,
virtual-atom count, and depth fixed, swapping only the vector field operator for PiFold's distance
readout, to confirm the gain comes from reading vectors rather than from the deeper stack. Second, I would
confirm the per-layer virtual-atom re-prediction (`Q_i ← Linear(s_i)`) helps and does not destabilize
training, since re-setting the probe each layer is the one piece with no analogue in PiFold. If both hold, the ladder's
single thesis — richer invariant reading of local geometry is the lever — has been carried one rung past
the strongest baseline. The full encoder-only scaffold module is in the answer.
