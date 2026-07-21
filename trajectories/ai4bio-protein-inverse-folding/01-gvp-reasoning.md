The scaffold default already gives me a working message-passing encoder — a KNN graph over `CA`,
dihedral and orientation scalars on the nodes, a distance-RBF plus a `CA→neighbor` direction vector on
the edges, three plain MPNN layers, an MLP decoder. It runs, it's invariant, and it is a fine floor. But
I want to understand *what kind* of floor it is, because that choice fixes what failure mode I am trying
to climb out of. Its failure mode is exactly the one the prior-art lineage warned about: it is a
scalar-only graph encoder. The instant the direction vector `F.normalize(X_ca_neighbors - X_ca)` is
concatenated onto the edge RBFs and pushed through `edge_embed`, the geometry stops being geometry. It
is three numbers in a feature vector, and three layers deep the network can no longer treat them as an
arrow it could rotate, project into a frame, or recombine — it can only do affine arithmetic on the
components. That is the Ingraham freeze restated in the scaffold's own code: invariance bought by killing
the geometry at the door.

So the question is sharp: can I keep the relational message passing the scaffold already has, but stop
freezing the geometry — let directional information flow through the graph as honest 3D objects I can
keep manipulating at every layer, while the scalar predictions stay invariant? If I can build the
per-step primitive that does that, everything later is a refinement of the *features* and the
*aggregation*, not of the invariance machinery.

There is more than one way to un-freeze geometry. A fully steerable equivariant network —
spherical-harmonic feature types, Clebsch–Gordan tensor products, the SE(3)-transformer machinery — is
strictly more expressive but wrong for this substrate: the harness hands me padded `(B, L, K)` dense
tensors and a residue `mask`, not the sparse edge lists those libraries assume, so every Clebsch–Gordan
path would have to be reimplemented dense and batched at a per-edge cost that scales badly in the number
of feature types — machinery I have no evidence I need when the lever is how richly the encoder reads
local geometry, not how baroque the algebra is. The coordinate-update style, where each layer moves the
points and reads geometry off the moving cloud, I also reject: I want an *invariant readout* of a fixed
backbone, and pushing atoms around invites the exact instability the frozen scaffold was avoiding. So the
minimal un-freeze is the middle path: attach a few genuine 3D vectors to nodes and edges and process them
with the smallest closed set of operations that keeps scalars invariant and vectors equivariant.

What does "carry vectors through the graph" require? Suppose a node (and an edge) carries, alongside its
scalar features `s`, a set of vector features `V` — a few arrows in `R^3`. Rotate or reflect the input,
i.e. right-multiply every vector by a unitary `U`. I need the scalar outputs to not move at all
(invariance) and the vector outputs to rotate the same way (equivariance). Which operations on `V` are
allowed? A linear combination across the vector channels, `W V`: since `W` touches the channel index and
`U` touches the spatial axis, `(WV)U = W(VU)` — channel-mixing is equivariant, but only if it is
*bias-free*, and the code will have to turn the bias off by hand. If I added a constant vector `c`,
equivariance would demand `(vW + c)U = (vU)W + c` for all `U`, i.e. `cU = c` for every rotation, forcing
`c = 0`, since no nonzero vector is fixed by all of `SO(3)`. So `W_h` and `W_μ` must carry `bias=False`; a
bias here is not a harmless regularizer, it is a symmetry violation. The L2 norm of a row is invariant
(`‖vU‖ = ‖v‖`) — my one bridge from a vector into a scalar. And the forbidden move: a coordinate-wise
nonlinearity like `ReLU(v_x), ReLU(v_y), ReLU(v_z)` scrambles under rotation, because the components are
defined only relative to my arbitrary axes. That ban is the dangerous one, because a vector path with no
nonlinearity is just stacked linear maps and collapses. The escape is forced: take the norm of a vector
(invariant), pass it through any nonlinearity (it's a scalar now, anything goes), and use the result to
*scale* the vector, `v' = σ⁺(‖v‖)·v` — then `σ⁺(‖vU‖)·(vU) = (σ⁺(‖v‖)·v)U`, equivariant, direction
preserved, only length modulated. That is the only vector nonlinearity that survives, and it drops out of
the constraints rather than being chosen.

So the closed toolkit for vectors is: bias-free channel-linear maps, L2 norms, and
scale-by-a-function-of-the-norm — with the norm as the one-way door into the scalar path. Package it as a
perceptron on a tuple `(s, V)`. The scalar path must *see* the vectors, else the geometry never reaches
the prediction, so concatenate the norms of transformed vector channels onto `s` before the scalar linear
and nonlinearity; the vector path channel-mixes then scales by its own norms. One subtlety, or I'll tie
two unrelated widths together: the number of norms I feed the scalars (`h`) is about how much invariant
geometric summary I want, while the number of output vector channels (`μ`) is about how rich a directional
representation I propagate. If one matrix does both, `h = μ` by accident. So split it: `W_h` lifts the
input vectors to an intermediate `V_h` with `h = max(v_in, v_out)` channels whose row-norms feed the
scalars, and a second bias-free `W_μ` produces the output vectors `V_μ = W_μ V_h`, scaled by `σ⁺(‖V_μ‖)`.
Now `h` and `μ` are decoupled. This module is the geometric vector perceptron, and its equivariance is
exactly the toolkit.

Set the vector width to zero and the GVP degenerates to `s ↦ σ(W s)`, an ordinary scalar
linear-plus-activation; a message-passing layer of such GVPs over `[h_i, e, h_j]` summed across neighbors
is *exactly* the scaffold's `MPNNEncoderLayer`. So the plain MPNN is the `v = 0` corner of the GVP family
— turning the vector width up is strictly adding capacity, not swapping architectures.

The GNN needs normalization that respects the same constraint. Ordinary LayerNorm subtracts a mean and
applies per-feature scale/shift; on a vector channel both break equivariance. What is allowed is a single
*invariant* rescaling: divide the vectors by their root-mean-square norm so their RMS length is one — one
global invariant scale, no per-coordinate parameters. The scalar path keeps ordinary LayerNorm. Same for
dropout: dropping a coordinate is axis-dependent, so I drop whole vector channels and use ordinary dropout
on scalars.

Now, which GVP this task wants — because the scaffold is not the generic autoregressive design model the
GVP derivation usually lands on. The full method wraps the encoder in a sequence-conditioned
autoregressive decoder with a causal mask, computing backward messages to keep causality exact. None of
that survives here: the harness gives me `forward(X, mask)` with **no sequence input at all** and a fixed
MLP decoder head, and I produce all 20 marginals in one shot. So the autoregressive decoder, the sequence
embeddings on edges, the `src < dst` causal masking are all amputated by the edit surface. What I keep is
precisely the *encoder*: build the GVP primitive, featurize the backbone into scalar+vector node and edge
tuples, run GVP message-passing layers, and project the final scalar part to `hidden_dim` for the
scaffold's decoder. Dense and batched (no `torch_geometric`, since the scaffold works in padded
`(B, L, K)` tensors with a `mask`), the equivariant LayerNorm simplified to scalar LayerNorm plus a
norm-preserving vector pass the dense code can afford.

Now the features, where I cash in the un-freezing. Per node: scalar features are the backbone dihedrals
`{sin, cos}(φ, ψ, ω)` — six invariant scalars from `_dihedrals`; vector features are the directional
quantities kept as honest arrows in the global frame — the unit vectors `CA→N`, `CA→C`, `CA→O`, three
vectors that pin the residue's orientation, written once per node. So `node_in = (6, 3)`. Per edge
`(j→i)`: scalar features are the `CA`-distance in 16 RBFs plus a 16-dim sinusoidal encoding of the
backbone offset `j − i` — 32 scalars; the vector feature is the single unit direction `CA_j − CA_i` kept
as a vector — `edge_in = (32, 1)`. The contrast with the default and with Ingraham: both projected the
neighbor direction into invariant scalars; I keep it as a vector and let the GVP take whatever norms and
inner products it wants, so the direction stays geometric downstream. I lift these through input GVPs
(identity activations, just to raise dimension) to hidden node tuples `(100, 16)` and edge tuples
`(32, 1)`, each followed by the equivariant LayerNorm. 100 scalar channels is the usual width for residue
context; 16 vector channels is a deliberately modest geometric width — 48 live geometric numbers per node
against 100 scalars, the value being in keeping a handful of meaningful directions live, not in brute
width. Edges stay thin at `(32, 1)` because there are `k·L` edges — order `10^4` per example at `k = 30` —
and every intermediate the message GVP forms lives on that `(B, L, K, ·)` grid; widening the edge scalar
to node-width 100 roughly triples the largest edge tensors. An edge's job is to gate and direct the
message, not store a fold's worth of context, so thin edges are the right economy.

The GVP convolution layer is Gilmer message passing with GVPs as the learned functions. For each edge,
concatenate the source node tuple, the edge tuple, and the target node tuple (scalars to scalars, vectors
to vectors — they live in different spaces): message input `(edge_s + 2·node_s, edge_v + 2·node_v) =
(232, 33)`. Run a short GVP stack whose last layer uses identity activations so the summed message stays
expressive, mask out padded neighbors, and aggregate by a degree-stable mean over valid neighbors — a mean
rather than a sum because the neighbor count varies with local packing and a raw sum would fight the
LayerNorm that follows. Then a residual update with the equivariant LayerNorm, followed by a wide GVP
feed-forward (scalar width 100→400→100, last layer identity), residual and normed again. Because each GVP
updates *both* scalar and vector channels, the vector features at every node get refined as the network
deepens — that is the whole point, the geometry stays live instead of frozen at input. Three such layers
match the scaffold's three-layer budget, and the whole stack comes in well under half a million
parameters. A final GVP with a `(hidden_dim, 0)` output — scalars only — gives the per-residue embedding
the scaffold's untouched two-layer MLP decoder consumes; the loss is the harness's masked per-residue
cross-entropy.

The distance RBF lays 16 Gaussians across 0–20 Å, spacing about 1.25 Å, a hair under a bond length. The
`CA–CA` neighbor distances a `k = 30` graph produces — sequence-adjacent `CA`s at roughly 3.8 Å, packing
contacts from about 4 to 10–12 Å, the 30th neighbor in a compact core rarely past 12–15 Å — all fall in
the lower-to-middle of the basis where several Gaussians fire and resolve them, while the top RBFs at
16–20 Å almost never activate for this `k`. So the basis is not the bottleneck, which is what makes the
*number* of distances, one per edge, the real limitation I am setting up to attack. The sinusoidal offset
encoding makes a sequence-adjacent contact distinguishable from a long-range one without inventing a
frame, its lowest frequency far longer than any protein so the code never wraps.

Invariance holds by construction: node/edge scalars are invariant, node/edge vectors equivariant, every
layer touches vectors only through the GVP toolkit and the equivariant LayerNorm, and the final readout
takes scalars only — so rotating the backbone leaves the log-probabilities fixed. Cost is `O(L·k)` per
layer, linear in length.

Now what this floor should *do*. GVP un-freezes the geometry, the most principled fix to the default's
weakness, but principled is not the same as strong, and I can already see two reasons it may be the
weakest even so. Count the degrees of freedom the edge carries. The relative pose of two residues is a
rigid transform in `SE(3)`: six numbers, three for the separation and three for the neighbor's
orientation about it. GVP's edge tuple gives one `CA→CA` distance and one unit direction — a length plus
two angles, three numbers, and those three pin only the separation. The neighbor's *orientation* relative
to me — the remaining three degrees of freedom, the thing that decides which face it packs with — reaches
the center only indirectly, through the neighbor's own node vectors after they have been mixed and normed.
So per edge the geometry under-determines the pose by half: two residues at the same `CA` separation but
rotated differently look nearly identical to the edge. Second, the encoder-only amputation hurts GVP more
than it hurts a method built to put its strength in the encoder: the GVP story leans on the autoregressive
decoder to model residue-residue couplings, and here that decoder is a per-residue MLP.

So I expect GVP to clear the dumb baselines, be invariant and stable, but land at the *bottom* on recovery
— recovery in the low-0.4s and perplexity still well above the mid-5s a richer encoder ought to reach —
the signature of an encoder that sees the neighborhood but cannot resolve *which* neighbor is doing what.
TS50, being de novo and more idealized, should sit a little higher on recovery and lower on perplexity
than the two CATH sets even at this floor. If that holds, the next move is not *more* invariance machinery
— the plain MPNN is already the `v = 0` corner of GVP and live vectors did not rescue thin edges — but
*richer invariant features* on the edges, pouring more of the relative pose into the graph through
additional inter-atomic distances that recover the orientation degrees of freedom the single `CA→CA`
direction misses.
