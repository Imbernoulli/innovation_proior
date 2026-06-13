ProteinMPNN did what I bet it would and the numbers confirm the diagnosis was right. On CATH 4.2 it
recovered 0.4603 at perplexity 5.4723, on CATH 4.3 0.4612 / 5.4639, on TS50 0.4829 / 5.0827 — a clean
jump over GVP's 0.4310 / 0.4337 / 0.4602, and the perplexity fell from the high-5.8s into the mid-5.4s.
So the lever was exactly what I claimed: pouring 25 invariant inter-atomic distances and a live edge
update into the graph beats one direction vector and frozen edges. The thin-edge starvation, not the
choice of invariance route, was GVP's problem. Good — but ProteinMPNN also stopped short, and I can name
where. Three things are still left on the table. First, its aggregation is an *unweighted* symmetric MLP
sum divided by a fixed `scale`; every valid neighbor contributes equally, and the model never learns
that a tightly-packed hydrophobic contact should weigh more than a distant glancing one. Second, its only
side-chain proxy is a *single fixed* virtual `Cβ` at the ideal tetrahedral position — one hand-placed
point, identical for every residue, when the geometry that distinguishes amino acids could use a learned,
transferable probe. Third, it has no global context: every residue sees only its `k`-NN neighborhood,
never a protein-level summary, so it cannot tell a buried core position from a surface one except through
whatever leaks in over three hops. Each of those is a concrete handle, and the next rung should grab all
three at once while keeping the rich invariant featurization that just paid off.

Start with the aggregation, because it is the cheapest big win. ProteinMPNN sums messages; I want the
graph to *decide* how much neighbor `j` matters to center residue `i`. Standard graph attention would
make a query from `i`, a key and value from `j`, and score by a dot product. But the dot product is the
restrictive part: in a protein graph the edge feature is not a small bias, it is the actual pairwise
geometry, and I want it to participate *directly* in the score. So instead of separate query/key
projections, score each edge with an MLP over the concatenation of the center node, the edge, and the
neighbor node, `w_ji = AttMLP([h_i ‖ e_ji ‖ h_j]) / sqrt(d_head)`, softmax over the incoming neighbors
of the same center, and form the value from the edge-and-neighbor concatenation `v_j = NodeMLP([e_ji ‖
h_j])`, then `ĥ_i = Σ_j a_ji v_j`. The `1/sqrt(d_head)` scaling matters for the Transformer reason —
without it large head dimensions saturate the neighbor softmax. This replaces ProteinMPNN's equal-weight
sum with a learned, geometry-aware weighting, which is precisely the missing lever on aggregation. I keep
the edge update ProteinMPNN introduced — after the node attention refines `h_i`, re-derive each edge from
its refreshed endpoints, `e_ji ← EdgeMLP([ĥ_j ‖ e_ji ‖ ĥ_i])` — so node and edge states still co-adapt
through depth.

Next, the global context, because the second weakness of ProteinMPNN was that residues never see the
protein as a whole. Full global attention would give every residue direct access to every other, but
that is quadratic in length and I refuse to pay it. I only need a cheap protein-level summary: mean-pool
the current node embeddings over the residues of the same protein, push that through an MLP, and use a
*sigmoid gate* to modulate each residue's channels, `h_i ← ĥ_i · σ(GateMLP(c))`. Why a multiplicative
gate and not an additive context? Addition would overwrite the local, residue-specific geometry that is
the whole carrier of the prediction; multiplication lets the global summary keep or suppress channels
while the local representation stays the carrier. This is linear in residues and slots in after the node
and edge updates. So one encoder block is now: MLP-scored neighbor attention, a wide feed-forward
refinement, the edge update, and the global channel gate — strictly more than ProteinMPNN's
node-then-edge block, addressing aggregation and context together.

Now the third weakness, the single fixed virtual atom, and this is the richest fix. ProteinMPNN's one
`Cβ` is a hand-placed point; I want *learnable* virtual atoms — points whose positions in the local
backbone frame are parameters shared across all residues. Build the frame from the three backbone atoms
around `CA` and place each virtual atom `V_i^k = x_k·b_i + y_k·n_i + z_k·(b_i×n_i) + CA_i` with shared
unit-normalized coefficients `(x_k, y_k, z_k)`. Sharing the coefficients across residues is the crucial
discipline: if every residue had its own virtual coordinates the points would be arbitrary per example,
but shared coordinates force a *transferable* placement — a consensus geometric probe that adds
side-chain-like information without ever seeing a side chain, and that the optimizer can tune to wherever
it most helps discriminate amino acids. With three virtual atoms I get extra invariant distances on both
nodes and edges: virtual-virtual pair distances at the node level, and same-index and cross-virtual
distances at the edge level, all in the RBF basis. This is the principled generalization of
ProteinMPNN's lone fixed `Cβ` — three learned probes instead of one fixed one.

Around those virtual atoms I enrich the rest of the featurization to the limit of what invariant scalars
can carry, since the ProteinMPNN run proved that feature richness is the dominant lever. For nodes:
intra-residue real-atom pair distances (`CA-N, CA-C, CA-O, N-C, N-O, O-C`) in the RBF basis, the virtual-
virtual distances, the six dihedral `{sin, cos}` channels, and the local orientation channels. For edges:
a long list of inter-residue real-atom pair distances in both orientations, the virtual-atom edge
distances, several direction features (the dot products of the backbone direction vectors with the
neighbor direction, and their cross-product magnitudes, all invariant because they are projections), a
few angle features, and the sinusoidal positional encoding of the offset. The exact channel counts close
the way PiFold's featurizer does; the point is that every channel is an invariant scalar and together
they describe the local geometry far more completely than ProteinMPNN's 25 distances alone. I normalize
with BatchNorm here rather than LayerNorm — the dense per-residue MLP stacks that embed these features
benefit from batch statistics, and BatchNorm is what the reference PiGNN uses.

Now the decoder, and here the encoder-only edit surface and PiFold *agree* for once, which is why this
rung fits the task so naturally. The autoregressive factorization `p(S|X) = Π_t p(s_t | s_<t, X)` is
expressive but pays a sequential cost, and many of the dependencies it models through previously
generated tokens are *already* induced by the shared structure — if two residues pack together, the
backbone geometry and the contact graph have told the encoder so. So the right trade is to make the
structural encoder strong enough that a *one-shot* decoder suffices: `p(S|X) = Π_i p(s_i | X)`, with the
decoder a single linear readout to 20 logits and a log-softmax. This is not a claim that residues are
physically independent; it is the claim that the final node embeddings already contain the structural
context for each marginal. And it is exactly what the harness wants: `forward(X, mask)`, no sequence
input, one parallel forward pass, no length-`L` decoding loop. Where GVP and ProteinMPNN had their
autoregressive decoders amputated by the edit surface and lost something, PiFold *chose* the one-shot
decoder, so it loses nothing — its whole design philosophy is to push the work into the encoder, which is
precisely the component this task isolates. This is the structural reason I expect PiFold to top the
ladder: it is the one method on the ladder whose strength was never in the decoder the harness removed.

Two task-specific implementation notes on which PiFold this is. First, the reference PiFold runs ten
PiGNN layers, far more than the scaffold's default three; the deeper stack is where the attention,
edge-update, and context-gate machinery earns its keep, so this rung sets `num_encoder_layers = 10` via
the `CONFIG_OVERRIDES` the edit surface allows. Ten attention-plus-context layers over the full
batch-normed feature stack are heavier per step, so the same override drops `batch_size` to 8 to fit
memory — both are within the four keys the harness permits (`learning_rate, dropout, num_encoder_layers,
batch_size`). Second, this is the *encoder* of PiFold transplanted into the dense batched `(B, L, K)`
harness with the scaffold `mask`, not the sparse scatter-based reference; the attention softmax is over
the neighbor axis with a `-1e9` mask on padded neighbors, and the BatchNorm is applied on flattened
`(B·L, hidden)` tensors. The featurizer, the PiGNN block, and the linear one-shot decoder are faithful;
the graph plumbing is the dense form the harness provides.

Let me check invariance once more, since I leaned entirely on invariant scalars again. All node and edge
distance features are inter-atomic distances (invariant); the dihedrals and the direction *dot products*
and *cross-product magnitudes* are projections, hence invariant; the virtual atoms are placed *in the
local frame*, so distances from them rotate and translate with the residue and stay invariant; the
positional encoding is frame-independent; the attention is a softmax over a symmetric neighbor set. The
linear readout consumes only these invariant node embeddings, so the output distribution is invariant by
construction. Cost is `O(L·k)` per layer with the context gate `O(L)`.

The falsifiable expectations against ProteinMPNN's measured numbers. The bet is that learned attention
(over equal-weight sums), three learned virtual atoms (over one fixed `Cβ`), a global context gate (over
none), a richer invariant feature set, and a deeper ten-layer stack together beat ProteinMPNN — and that
PiFold's one-shot encoder-only design fits this task better than any method whose decoder was amputated.
So I expect a clear gain over ProteinMPNN's 0.4603 / 0.4612 / 0.4829: recovery into the mid-0.46s on
CATH 4.2 and into the high-0.47s on CATH 4.3, with TS50 climbing past 0.52, and perplexity dropping below
ProteinMPNN's 5.47 / 5.46 / 5.08 toward the low-5s and mid-4s. I am specifically predicting the largest
jump on TS50, the out-of-distribution set, because the global context gate and the learned virtual atoms
are exactly the kind of transferable, fold-agnostic structure that should generalize past the training
distribution — if PiFold's TS50 gain is *not* the largest of the three benchmarks, my story about *why*
the new machinery helps (transferable global/probe features, not just more capacity) is suspect. If
PiFold tops the ladder as expected, the open question it leaves — and the one a finale would have to
answer — is whether the *frame-anchored* virtual atoms can be made to carry richer geometry still, by
letting the network compute *learnable vector operations* between them rather than only reading their
pairwise distances. The full encoder-only scaffold module is in the answer.
