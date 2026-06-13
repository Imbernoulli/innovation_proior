I start at the weakest rung deliberately, because it pins down the one thing every later rung will
argue about: how the encoder carries geometry through the graph while keeping the output invariant. The
scaffold default already gives me a working message-passing encoder — a KNN graph over `CA`, dihedral
and orientation scalars on the nodes, a distance-RBF plus a `CA→neighbor` direction vector on the
edges, three plain MPNN layers, an MLP decoder. It runs, it's invariant, and it is a fine floor. But I
want to understand *what kind* of floor it is, because the choice of where to start the ladder is the
choice of what failure mode I am trying to climb out of. And the default's failure mode is exactly the
one the prior-art lineage warned about: it is a scalar-only graph encoder. The instant the direction
vector `F.normalize(X_ca_neighbors - X_ca)` is concatenated onto the edge RBFs and pushed through
`edge_embed`, the geometry stops being geometry. It is three numbers in a feature vector, and three
layers deep the network can no longer treat them as an arrow it could rotate, project into a frame, or
recombine — it can only do affine arithmetic on the components. That is the Ingraham freeze restated in
the scaffold's own code: invariance bought by killing the geometry at the door.

So the question I want the first rung to answer is sharp: can I keep the relational message passing the
scaffold already has, but stop freezing the geometry — let directional information flow through the
graph as honest 3D objects I can keep manipulating at every layer, while the scalar predictions stay
invariant? If I can build the per-step primitive that does that, I have the cleanest possible baseline
encoder, and everything later is a refinement of the *features* and the *aggregation*, not of the
invariance machinery. That is why GVP is the right first rung: it is the minimal change to the default
that un-freezes the geometry.

Let me reason about what "carry vectors through the graph" actually requires, because the constraint is
unforgiving. Suppose a node (and an edge) carries, alongside its scalar features `s`, a set of vector
features `V` — a few arrows in `R^3`. Rotate or reflect the input, i.e. right-multiply every vector by
a unitary `U`. I need the scalar outputs to not move at all (invariance) and the vector outputs to
rotate the same way (equivariance). Which operations on `V` are allowed? Let me enumerate. A linear
combination across the vector channels, `W V`: since `W` touches the channel index and `U` touches the
spatial axis, `(WV)U = W(VU)` — they commute, so channel-mixing is equivariant, but only if it is
*bias-free* (adding a constant vector breaks it, because there is no nonzero rotation-invariant constant
vector). The L2 norm of a row, `‖v‖`: unitary `U` preserves length, so `‖vU‖ = ‖v‖` — flatly
invariant. That norm is my one bridge from a vector into a scalar. And the forbidden moves: a
coordinate-wise nonlinearity like `ReLU(v_x), ReLU(v_y), ReLU(v_z)` scrambles under rotation because
the components are defined only relative to my arbitrary axes — I can never put a pointwise nonlinearity
on coordinates. That last ban is the dangerous one, because a vector path with no nonlinearity is just
stacked linear maps and collapses. The escape is forced: take the norm of a vector (invariant), pass it
through any nonlinearity (it's a scalar now, anything goes), and use the result to *scale* the vector,
`v' = σ⁺(‖v‖)·v`. Check it: `σ⁺(‖vU‖)·(vU) = (σ⁺(‖v‖)·v)U`, equivariant, direction preserved, only the
length modulated. That is the only vector nonlinearity that survives, and it drops out of the
constraints rather than being chosen.

So the closed toolkit for vectors is: bias-free channel-linear maps, L2 norms, and scale-by-a-function-
of-the-norm — with the norm as the one-way door into the scalar path. Package it as a perceptron that
processes a tuple `(s, V)`. The scalar path must *see* the vectors (else the geometry never reaches the
prediction), so concatenate the norms of transformed vector channels onto `s` before the scalar linear
and nonlinearity. The vector path channel-mixes and then scales by its own norms. One subtlety I have
to get right or I'll tie two unrelated widths together: the number of norms I want to feed the scalars
(call it `h`) is about how much invariant geometric summary I want, while the number of output vector
channels (`μ`) is about how rich a directional representation I propagate. If one matrix does both,
`h = μ` by accident. So split it: `W_h` lifts the input vectors to an intermediate `V_h` with
`h = max(v_in, v_out)` channels whose row-norms feed the scalars, and a second bias-free `W_μ` produces
the actual output vectors `V_μ = W_μ V_h`, scaled by `σ⁺(‖V_μ‖)`. Now `h` and `μ` are decoupled. This
little module — scalar path reads vector norms, vector path channel-mixes then scales-by-norm — is the
geometric vector perceptron, and its equivariance is exactly the toolkit: every operation I ever apply
to a vector is `W_h`, `W_μ`, a norm, or a norm-based scaling, all of which commute with `U`.

I should also confirm the supporting pieces respect the same constraint, because the GNN needs
normalization. Ordinary LayerNorm subtracts a mean and applies per-feature scale/shift; on a vector
channel, subtracting a mean vector and per-coordinate scaling both break equivariance. What is allowed
is a single *invariant* rescaling: divide the vectors by their root-mean-square norm so their RMS length
is one — one global invariant scale, no per-coordinate parameters, commutes with `U`. The scalar path
keeps ordinary LayerNorm. Same for dropout: dropping a coordinate is axis-dependent, so I drop whole
vector channels (an invariant 0/1 per-channel scaling) and use ordinary dropout on scalars.

Now I have to be honest about *which GVP this task wants*, because the scaffold is not the generic
autoregressive design model the GVP derivation usually lands on. The full method would wrap the
encoder in a structure-only encoder followed by a sequence-conditioned autoregressive decoder with a
causal mask, computing backward messages from the encoder embeddings to keep causality exact. None of
that survives here. The harness gives me `forward(X, mask)` with **no sequence input at all** and a
fixed MLP decoder head; I produce all 20 marginals in one shot. So the autoregressive decoder, the
sequence embeddings on edges, the `src < dst` causal masking — all of it is amputated by the edit
surface. What I keep is precisely the *encoder*: build the GVP primitive, featurize the backbone into
scalar+vector node and edge tuples, run GVP message-passing layers, and project the final scalar part
to `hidden_dim` for the scaffold's decoder. The edit is an encoder-only GVP, dense and batched (no
`torch_geometric`, because the scaffold works in padded `(B, L, K)` tensors with a `mask`), and the
equivariant LayerNorm is simplified to the scalar LayerNorm plus a norm-preserving vector pass the
dense code can afford.

With that settled, choose the features, and here is where I cash in the un-freezing. Per node: the
scalar features are the backbone dihedrals `{sin, cos}(φ, ψ, ω)` — six invariant scalars from
`_dihedrals`. The vector features are the directional quantities kept as honest arrows in the global
frame: the unit vectors `CA→N`, `CA→C`, `CA→O`. Three vectors that pin the residue's orientation,
written once per node, absolutely — no per-neighbor frame redundancy. So `node_in = (6, 3)`. Per edge
`(j→i)`: scalar features are the `CA`-distance in 16 RBFs plus a 16-dim sinusoidal encoding of the
backbone offset `j − i` — 32 scalars; the vector feature is the single unit direction `CA_j − CA_i`
kept as a vector — `edge_in = (32, 1)`. Note the contrast with the default scaffold and with Ingraham:
both projected the neighbor direction into invariant scalars; I keep it as a vector and let the GVP take
whatever norms and inner products it wants, so the direction stays geometric downstream. I lift these
inputs through input GVPs (identity activations, just to raise dimension) to hidden node tuples
`(100, 16)` and edge tuples `(32, 1)`, each followed by the equivariant LayerNorm. Why `(100, 16)`?
100 scalar channels is the usual width for carrying residue context; 16 vector channels is a
deliberately modest geometric width — each is a full arrow, three numbers, and the value is in keeping
a handful of meaningful directions live, not in brute width. Edges stay thin at `(32, 1)` because an
edge's job is mostly to gate and direct the message, and the `k·L` edges would otherwise eat memory.

The GVP convolution layer is Gilmer message passing with GVPs as the learned functions. For each edge,
concatenate the source node tuple, the edge tuple, and the target node tuple (scalars to scalars,
vectors to vectors — they live in different spaces), run a short GVP stack whose last layer uses
identity activations so the summed message stays expressive, mask out padded neighbors, and aggregate
by a degree-stable mean over valid neighbors. Then a residual update with the equivariant LayerNorm,
followed by a wide GVP feed-forward (expand the scalar/vector widths, contract back, last layer
identity-activation), residual and normed again. Critically, because each GVP updates *both* scalar and
vector channels, the vector features at every node get refined as the network deepens — that is the
whole point, the geometry stays live instead of frozen at input. Three such layers give each residue a
few rounds of spatial-neighborhood context, matching the scaffold's three-layer budget. A final GVP
with a `(hidden_dim, 0)` output — scalars only — gives the per-residue embedding the scaffold decoder
consumes. The decoder head itself is the scaffold's two-layer MLP, untouched, mapping `h_V` to 20
logits and log-softmax; the loss is the harness's masked per-residue cross-entropy.

Let me sanity-check that nothing I did broke invariance, since the whole rung hinges on it. Node scalars
are dihedrals (invariant); node vectors are `CA→N/C/O` directions (equivariant); edge scalars are RBF
distances and sequence offset (invariant/frame-independent); edge vectors are `CA_j−CA_i` directions
(equivariant). Every layer's only contact with vectors is through the GVP toolkit — bias-free
channel-linear maps, norms, norm-scalings — and the equivariant LayerNorm. The final readout takes
`(hidden_dim, 0)`, i.e. only scalars, so the output distribution is invariant by construction. Rotate
the backbone and the predicted log-probabilities do not move. Good — and the cost is `O(L·k)` per
layer, linear in length, fits the budget.

Now reason about what this floor should *do*, because that is the point of running it first. GVP un-
freezes the geometry, which is the most principled fix to the default's weakness; but principled is not
the same as strong, and I can already see two reasons it may be the weakest rung even so. First, its
geometric expressivity per edge is thin: one direction vector and a `CA`-only distance. The richest
signal in this problem is the *relative pose* of two residues, which a single `CA→CA` direction plus one
distance under-determines — two residues at the same `CA` separation can be packed very differently, and
GVP's edge tuple barely distinguishes them. Second, the encoder-only amputation hurts GVP more than it
hurts a method that was designed to put its strength in the encoder: the GVP story leans on the
autoregressive decoder to model residue-residue couplings, and here that decoder is gone, replaced by a
per-residue MLP. So GVP is carrying its geometric machinery into a setting that has stripped away the
half of the model it expected to share the work with.

The falsifiable expectation, then: GVP should clear the dumb baselines and be invariant and stable, but
land at the *bottom* of the three-rung ladder on recovery, precisely because its per-edge geometry is
thin and its decoder is amputated. I expect recovery in the low-0.43 range on CATH 4.2 and CATH 4.3,
with perplexity well above 5.5 — comfortably worse than a method that enriches the edge geometry with
many atom-pair distances. If that holds, the diagnosis for the next rung writes itself: the fix is not
*more* invariance machinery but *richer invariant features* on the edges — pour more of the relative
pose into the graph through additional inter-atomic distances — which is exactly where I will go next.
The full encoder-only scaffold module is in the answer.
