ProteinMPNN did what I bet it would. On CATH 4.2 it recovered 0.4603 at perplexity 5.4723, on CATH 4.3
0.4612 / 5.4639, on TS50 0.4829 / 5.0827 — a clean jump over GVP's 0.4310 / 0.4337 / 0.4602, perplexity
falling from the high-5.8s into the mid-5.4s. So the lever was as claimed: 25 invariant inter-atomic
distances and a live edge update beat one direction vector and frozen edges. Thin-edge starvation, not the
choice of invariance route, was GVP's problem.

But *where* it went up tells me what is still missing. Take the recovery deltas benchmark by benchmark:
CATH 4.2 rose 0.0293, CATH 4.3 rose 0.0275, TS50 only 0.0227. The richest gain landed on the two
in-distribution CATH sets and the *smallest* on TS50, the out-of-distribution de novo set — exactly what a
local-geometry fix should look like. Distances sharpen the reading of each residue's immediate
neighborhood, and on structures drawn from the training distribution that sharper local read converts
directly into recovery; on TS50, whose folds are unlike anything trained on, a purely local improvement
helps less, because what transfers across distribution shift is not local sharpness but *global,
fold-agnostic* structure. So ProteinMPNN's edge distances are an in-distribution lever, and moving TS50
needs machinery that is transferable rather than merely local.

The perplexity column tells a complementary story. GVP → ProteinMPNN dropped perplexity by 0.43 on CATH
4.2, 0.36 on CATH 4.3, 0.40 on TS50 while recovery moved only 0.02–0.03. So the smooth `exp`-NLL signal
fell several times more, in relative terms, than the argmax rate rose — both metrics improved together, so
the gain was a genuine narrowing of the whole distribution and not argmax quantization, and perplexity is
the more *sensitive* readout of a geometry improvement, registering the encoder getting surer about every
residue even where the argmax was already right and cannot move. I'll want that when I judge this rung.

ProteinMPNN also stopped short in three ways. Its aggregation is an *unweighted* symmetric MLP sum divided
by a fixed `scale`; every valid neighbor contributes equally, and the model never learns that a
tightly-packed hydrophobic contact should weigh more than a distant glancing one. Its only side-chain proxy
is a *single fixed* virtual `Cβ`, one hand-placed point identical for every residue, when the geometry that
distinguishes amino acids could use a learned, transferable probe. And it has no global context: every
residue sees only its `k`-NN neighborhood, never a protein-level summary, so it cannot tell a buried core
position from a surface one except through whatever leaks in over three hops.

The cheap move — just stack more of ProteinMPNN's own block — would not get me there, and the delta table
says why. Adding layers to the unweighted-sum block buys more hops of local propagation, more of the
in-distribution lever that already gained the most and the least where I most need help. Depth alone cannot
learn to weight one neighbor over another — the sum is structurally equal-weight however deep it runs — and
it cannot manufacture a protein-level view a `k`-NN stack never forms except by slow leakage. So more depth
would nudge CATH and leave the TS50 gap roughly where it is, the opposite of what I want. The three
mechanisms address weighting, transferable probe geometry, and global context directly, so I grab them
together rather than reaching for the free parameter first.

Start with the aggregation, the cheapest big win. ProteinMPNN sums messages; I want the graph to *decide*
how much neighbor `j` matters to center `i`. The obvious template is graph attention — query from `i`, key
and value from `j`, score by dot product — but the dot product is the wrong scoring function here. In a
plain transformer the edge is at most a small additive positional bias on `q_i · k_j`; the geometry only
shifts the logit. In this graph the edge feature *is* the pairwise geometry, the 25-distance pose I worked
to build, and I want it to participate in the score multiplicatively and nonlinearly, not as an
afterthought. So instead of query/key projections and a dot product, score each edge with an MLP over the
concatenation of center node, edge, and neighbor node, `w_ji = AttMLP([h_i ‖ e_ji ‖ h_j]) / sqrt(d_head)`,
softmax over the incoming neighbors of the same center, and form the value from the edge-and-neighbor
concatenation `v_j = NodeMLP([e_ji ‖ h_j])`, then `ĥ_i = Σ_j a_ji v_j`. The `1/sqrt(d_head)` scaling
matters for the Transformer reason — without it large head dimensions saturate the neighbor softmax into a
near-argmax and the gradient through the weights dies. Four heads over the 128-dim hidden, `d_head = 32`,
because a residue's identity depends on several relations at once — how tightly a neighbor packs, how far
along the chain it sits, how its probe atoms align — and four heads let the same neighborhood be pooled
under four learned weightings in parallel. I keep the edge update ProteinMPNN introduced — after the node
attention refines `h_i`, re-derive each edge from its refreshed endpoints,
`e_ji ← EdgeMLP([ĥ_j ‖ e_ji ‖ ĥ_i])` — so node and edge states still co-adapt through depth.

Next the global context. Full global attention would give every residue direct access to every other, but
price it first: `O(L²)` per layer against the graph's `O(L·k)`; with `k = 30` and `L ≈ 500` that is a ratio
of `L/k ≈ 16`, sixteen times the compute per layer, ten layers deep, and in exchange a mechanism with *no
locality prior* that has to relearn from scratch that spatial neighbors matter most. I refuse to pay `16×`
to discard the inductive bias the KNN graph gave me for free. What I need is far less: a cheap protein-level
summary. Mean-pool the current node embeddings over the residues of the same protein, push through an MLP,
and use a *sigmoid gate* to modulate each residue's channels, `h_i ← ĥ_i · σ(GateMLP(c))`. Not an additive
context, because adding the same context vector to every `h_i` shifts all residues by an identical amount —
a per-protein bias that cannot say "keep this channel here, suppress it there," it moves everyone the same
way and washes out the residue-specific geometry that is the whole carrier. Multiplication with
`σ(·) ∈ (0, 1)` per channel lets the global summary keep or damp each channel independently while the local
representation stays the carrier. Linear in residues, slotted in after the node and edge updates. So one
encoder block is now MLP-scored neighbor attention, a wide feed-forward, the edge update, and the global
channel gate.

Now the third weakness, the single fixed virtual atom, and the richest fix. I want *learnable* virtual
atoms — points whose positions in the local backbone frame are parameters shared across all residues. Build
the frame from the three backbone atoms around `CA` and place each virtual atom
`V_i^k = x_k·b_i + y_k·n_i + z_k·(b_i×n_i) + CA_i` with shared unit-normalized coefficients. Two checks.
Invariance: `b_i`, `n_i` are backbone difference vectors and `b_i×n_i` their cross product, so all three
rotate with the residue; a fixed linear combination added to `CA_i` transforms exactly as the backbone
does, and every distance from `V_i^k` is invariant. And the discipline of *sharing* the coefficients across
residues: per-residue virtual coordinates would be arbitrary per example and overfit placements that mean
nothing across proteins, while shared coordinates force a *transferable* placement — a consensus geometric
probe that adds side-chain-like information without ever seeing a side chain, tuned by the optimizer to
wherever it most helps discriminate amino acids. There is a continuity worth noting: the first virtual atom
is seeded at exactly the tetrahedral coefficients ProteinMPNN used for its fixed `Cβ`, so this rung
literally *starts* from that one hand-placed probe and is free to move it and add two more. With three
virtual atoms I get extra invariant distances on both nodes and edges. Three is deliberate: each virtual
atom contributes `n·(n−1)` ordered virtual-virtual node channels through the RBF, so three already spend
`3·2·16 = 96` node channels — nearly half the 204-channel node feature — and pushing to four or five would
let the virtual-atom distances dominate the featurization and invite overfitting the probe placements, the
very failure the sharing discipline exists to prevent. Three is enough to triangulate a side-chain-like
direction while keeping the learned geometry a minority of the feature budget.

Around those probes I enrich the featurization to the limit of what invariant scalars carry, since the
ProteinMPNN run proved feature richness is the dominant lever. Nodes: intra-residue real-atom pair
distances (`CA-N, CA-C, CA-O, N-C, N-O, O-C`) in the RBF basis, the virtual-virtual distances, six dihedral
`{sin, cos}` channels, and local orientation channels — `6·16 + 12 + 3·2·16 = 204` invariant channels.
Edges: fifteen inter-residue real-atom pair distances in the RBF basis, the virtual-atom edge distances
(three same-index plus cross terms), direction features — dot products of the backbone direction vectors
with the neighbor direction and cross-product magnitudes, all invariant because they are projections and
lengths, not raw vectors — a few angle features, and the sinusoidal offset encoding —
`15·16 + 4 + 8 + 16 + (3·16 + 3·2·16) = 412` channels. The exact counts aren't the point; every channel is
an invariant scalar and together they describe the local geometry far more completely than 25 distances
alone. I normalize with BatchNorm rather than LayerNorm — the dense per-residue MLP stacks benefit from
batch statistics over the flattened `(B·L, hidden)` tensor — but the trade is that BatchNorm couples the
batch, so a residue's normalization now depends on the other proteins in its batch, one more reason the
deeper stack wants a modest but not tiny batch size.

Now the decoder, where the encoder-only edit surface and PiFold *agree*. The autoregressive factorization
`p(S|X) = Π_t p(s_t | s_<t, X)` is expressive but pays a sequential cost, and many of the dependencies it
models through previously generated tokens are *already* induced by the shared structure — if two residues
pack together, the backbone geometry and the contact graph have told the encoder so. So make the encoder
strong enough that a *one-shot* decoder suffices: `p(S|X) = Π_i p(s_i | X)`, the decoder a single linear
readout to 20 logits and a log-softmax. Not a claim that residues are physically independent — the claim
that the final node embeddings already contain the structural context for each marginal. And it is exactly
what the harness wants: `forward(X, mask)`, no sequence input, one parallel forward pass. Here is the
structural point that makes me expect this rung to *top* the ladder. GVP and ProteinMPNN each arrived
carrying an autoregressive decoder, and the edit surface *amputated* it — those two lost the half of the
model meant to carry residue-residue couplings and made do with a per-residue MLP they were never built
for. PiFold *chose* the one-shot decoder from the start; its whole philosophy pushes the work into the
encoder. So when the harness removes the autoregressive decoder, PiFold loses nothing — the component the
task keeps is precisely the component PiFold invested in. That alignment, not just the extra machinery, is
why I expect it on top.

Two implementation notes on which PiFold this is. PiFold runs ten layers, far more than the scaffold's
three; the deeper stack is where the attention, edge-update, and context-gate machinery earns its keep, so
this rung sets `num_encoder_layers = 10` via `CONFIG_OVERRIDES`. Depth buys two things: it deepens the
attention refinement, and it widens the receptive field — three hops of a `k = 30` graph reach only a few
residues out, whereas ten hops let signal propagate across essentially a whole domain, so even before the
explicit gate fires each residue's embedding integrates structure from far down the chain, and the gate
adds the protein-level summary on top. Ten layers over the full batch-normed feature stack are heavier per
step — each carries a value tensor on the `(B, L, K, hidden)` grid — so the same override drops
`batch_size` to 8 to fit, both within the four keys the harness permits. The neighbor softmax is guarded
the obvious way: padded neighbors have their logits pushed to `-1e9` before the softmax, which underflows
to zero, so no weight leaks onto padding. And this is the *encoder* of PiFold in the dense batched
`(B, L, K)` form the harness provides, with the scaffold `mask` rather than a sparse scatter.

Invariance holds again by construction: every node and edge feature is an inter-atomic distance, a
dihedral, a direction dot-product or cross-product magnitude, a frame-local virtual-atom distance, or a
frame-independent positional encoding; the attention is a softmax over a symmetric neighbor set; the linear
readout consumes only these invariant embeddings. Cost is `O(L·k)` per layer with the context gate `O(L)`.

The bet against ProteinMPNN's numbers: learned attention, three learned virtual atoms, a global context
gate, a richer invariant feature set, and a deeper ten-layer stack together beat ProteinMPNN, and PiFold's
one-shot encoder-only design fits this task better than any method whose decoder was amputated. So I expect
a clear gain over 0.4603 / 0.4612 / 0.4829 — a step up on both CATH sets, TS50 pushing past 0.5, perplexity
below 5.47 / 5.46 / 5.08. And the sharp claim the delta table earns: I expect the *largest* recovery jump on
TS50. The mechanism — when ProteinMPNN added purely local edge distances, TS50 gained the least (0.0227
against CATH's 0.0293 and 0.0275) because local sharpness does not transfer across distribution shift; the
machinery this rung adds, the global context gate and the shared learnable probes, is transferable,
fold-agnostic structure, which should help most where the test folds are least like training. So the
ordering of gains should invert: TS50 should now lead. If instead PiFold's TS50 gain is not the largest, the
machinery is just adding local capacity, my transferability story is wrong, and I would rethink what the
gate and probes are doing. If PiFold tops the ladder, the open question a finale faces is whether the
frame-anchored virtual atoms can carry richer geometry still, by letting the network compute *learnable
vector operations* between them rather than only reading their pairwise distances.
