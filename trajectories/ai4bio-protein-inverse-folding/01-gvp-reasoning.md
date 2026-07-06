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
invariance machinery.

Before I commit to that route I should be honest that there is more than one way to un-freeze geometry,
and I want to reject the others by argument rather than taste. The heavyweight option is a fully
steerable equivariant network — spherical-harmonic feature types, Clebsch–Gordan tensor products between
them, the SE(3)-transformer machinery. That is strictly more expressive than what I am about to build:
it can carry type-1 and type-2 features and mix them with learned paths. But it is the wrong tool for
this substrate on two counts I can actually count. First, the harness hands me padded `(B, L, K)` dense
tensors and a residue `mask`, not the sparse edge lists those libraries assume; every Clebsch–Gordan
path would have to be reimplemented dense and batched, and the per-edge tensor-product cost scales badly
in the number of feature types. Second, the ladder's whole premise is that the lever is *how richly the
encoder reads local geometry*, not how baroque the equivariance algebra is — spending my first rung on
type-2 tensors buys me machinery I have no evidence I need yet. A second alternative is the coordinate-
update style, where each layer moves the points and reads geometry off the moving cloud. But I do not
want to regress coordinates; I want an *invariant readout* of a fixed backbone, and letting the encoder
push atoms around invites the exact instability the frozen scaffold was avoiding. So the minimal, honest
un-freeze is the middle path: attach a few genuine 3D vectors to nodes and edges and process them with
the smallest closed set of operations that keeps scalars invariant and vectors equivariant. That is the
first rung — the minimal change to the default that stops freezing the geometry.

Let me reason about what "carry vectors through the graph" actually requires, because the constraint is
unforgiving. Suppose a node (and an edge) carries, alongside its scalar features `s`, a set of vector
features `V` — a few arrows in `R^3`. Rotate or reflect the input, i.e. right-multiply every vector by
a unitary `U`. I need the scalar outputs to not move at all (invariance) and the vector outputs to
rotate the same way (equivariance). Which operations on `V` are allowed? Let me enumerate. A linear
combination across the vector channels, `W V`: since `W` touches the channel index and `U` touches the
spatial axis, `(WV)U = W(VU)` — they commute, so channel-mixing is equivariant, but only if it is
*bias-free*. It is worth seeing exactly why the bias is fatal, because the code will have to turn it off
by hand. If I added a constant vector `c`, the map is `v ↦ vW + c`, and equivariance would demand
`(vW + c)U = (vU)W + c` for all `U`, i.e. `cU = c` for every rotation — which forces `c = 0`, since no
nonzero vector is fixed by all of `SO(3)`. So `W_h` and `W_μ` must carry `bias=False`; a bias is not a
harmless regularizer here, it is a symmetry violation. The L2 norm of a row, `‖v‖`: unitary `U`
preserves length, so `‖vU‖ = ‖v‖` — flatly invariant. That norm is my one bridge from a vector into a
scalar. And the forbidden moves: a coordinate-wise nonlinearity like `ReLU(v_x), ReLU(v_y), ReLU(v_z)`
scrambles under rotation because the components are defined only relative to my arbitrary axes — I can
never put a pointwise nonlinearity on coordinates. That last ban is the dangerous one, because a vector
path with no nonlinearity is just stacked linear maps and collapses. The escape is forced: take the norm
of a vector (invariant), pass it through any nonlinearity (it's a scalar now, anything goes), and use
the result to *scale* the vector, `v' = σ⁺(‖v‖)·v`. Check it: `σ⁺(‖vU‖)·(vU) = (σ⁺(‖v‖)·v)U`,
equivariant, direction preserved, only the length modulated. That is the only vector nonlinearity that
survives, and it drops out of the constraints rather than being chosen.

I want to watch this actually work on numbers once, because if the primitive is wrong everything above
it is wrong. Take a single input vector channel with value `v = (1, 0, 0)`, and a lift `W_h` from one
channel to two with weights `[2, −1]`. Transposing so the spatial axis is inside, the lift produces two
intermediate vectors, `2·v = (2, 0, 0)` and `−1·v = (−1, 0, 0)`; their spatial-axis norms are `2` and
`1`, so the scalar path reads `(2, 1)`. Now rotate the input by 90° about `z`, `v → (0, 1, 0)`. The two
intermediates become `(0, 2, 0)` and `(0, −1, 0)`, whose norms are again `2` and `1` — the scalar path
reads exactly `(2, 1)` as before. The norms did not move: invariance confirmed by construction, not
assertion. Push it through an output map `W_μ = [1, −1]`: the output vector is `2·(2,0,0)`-style mixing,
`1·(2,0,0) + (−1)·(−1,0,0) = (3, 0, 0)` for the original, and `1·(0,2,0) + (−1)·(0,−1,0) = (0, 3, 0)`
for the rotated input — which is precisely `(3,0,0)` rotated 90° about `z`. The output vector rotated
with the input; equivariance confirmed on numbers. And the gate that scales it, `σ⁺` of a function of
the (invariant) norms, is the same scalar in both frames, so it cannot break what I just verified. This
is the whole primitive and it survives the one test that matters.

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

There is a limit check hiding here that reassures me I am generalizing the floor rather than replacing
it. Set the vector width to zero — no input vectors, no output vectors. Then the GVP has nothing to lift
and no norms to concatenate; it degenerates to `s ↦ σ(W s)`, an ordinary scalar linear-plus-activation,
and a message-passing layer built from such GVPs over `[h_i, e, h_j]` summed across neighbors is
*exactly* the scaffold's `MPNNEncoderLayer`. So the plain MPNN is the `v = 0` corner of the GVP family;
turning the vector width up is strictly adding capacity, not swapping architectures. Whatever the floor
recovers, this encoder can recover too, and the only question is whether the live vectors buy anything
on top.

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
deliberately modest geometric width — each is a full arrow, three numbers, so 16 vectors is 48 live
geometric numbers per node against 100 scalars, and the value is in keeping a handful of meaningful
directions live, not in brute width. Edges stay thin at `(32, 1)` for a reason I can size. There are
`k·L` edges — with `k = 30` and proteins up to several hundred residues, an order of `10^4` edges per
example — and every intermediate the message GVP forms lives on that `(B, L, K, ·)` grid. Widening the
edge scalar from 32 to node-width 100 roughly triples the largest edge tensors; keeping the edge tuple
`(32, 1)` is what makes the dense implementation fit memory while the nodes stay wide. An edge's job is
mostly to gate and direct the message, not to store a fold's worth of context, so thin edges are the
right economy.

The GVP convolution layer is Gilmer message passing with GVPs as the learned functions. For each edge,
concatenate the source node tuple, the edge tuple, and the target node tuple (scalars to scalars,
vectors to vectors — they live in different spaces): the message input is `(edge_s + 2·node_s,
edge_v + 2·node_v) = (32 + 200, 1 + 32) = (232, 33)` scalar and vector channels. Run a short GVP stack
whose last layer uses identity activations so the summed message stays expressive, mask out padded
neighbors, and aggregate by a degree-stable mean over valid neighbors — a mean rather than a sum
because the neighbor count varies with the local packing and a raw sum would fight the LayerNorm that
follows. Then a residual update with the equivariant LayerNorm, followed by a wide GVP feed-forward
(expand the scalar width from 100 to 400, contract back, last layer identity-activation), residual and
normed again. Critically, because each GVP updates *both* scalar and vector channels, the vector
features at every node get refined as the network deepens — that is the whole point, the geometry stays
live instead of frozen at input. Three such layers give each residue a few rounds of spatial-
neighborhood context, matching the scaffold's three-layer budget. The bookkeeping stays small: a single
conv layer is dominated by its four GVP linears — the `(232, 33) → (100, 16)` message map alone is on
the order of `2.7×10^4` scalar-path weights, and the `100 → 400 → 100` feed-forward another `10^5` —
so the layer is roughly `1.4×10^5` parameters and the three-layer stack plus the thin input and edge
GVPs and the final projection comes in well under half a million. This is a lean floor, which is what I
want a floor to be. A final GVP with a `(hidden_dim, 0)` output — scalars only — gives the per-residue
embedding the scaffold decoder consumes. The decoder head itself is the scaffold's two-layer MLP,
untouched, mapping `h_V` to 20 logits and log-softmax; the loss is the harness's masked per-residue
cross-entropy.

One more featurization check before I trust the edges, because a basis that does not cover the relevant
distances would starve the encoder for a boring reason rather than an interesting one. The distance RBF
lays 16 Gaussians across 0–20 Å, spacing about `20/16 = 1.25` Å, a hair under a bond length, so the
resolution is fine. The question is whether the `CA–CA` neighbor distances a `k = 30` graph produces
land where the basis is dense. Sequence-adjacent `CA`s sit at roughly 3.8 Å; spatial contacts that
matter for packing run from about 4 Å out to 10–12 Å; and the 30th nearest neighbor in a compact core
rarely exceeds 12–15 Å. So essentially all the informative edges fall in the lower-to-middle of the
basis, where several Gaussians fire and resolve them well, while the top RBFs at 16–20 Å almost never
activate for this `k`. The basis is not the bottleneck — which is what makes the *number* of distances,
one per edge, the real limitation I am setting up to attack. The sinusoidal offset encoding is the same
kind of cheap, honest feature: it makes a sequence-adjacent contact distinguishable from a long-range
one without inventing a frame, and its lowest frequency has a period far longer than any protein so the
code never wraps.

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
the same as strong, and I can already see two reasons it may be the weakest rung even so, and I can put
numbers on the first. Count the degrees of freedom the edge actually carries. The relative pose of two
residues is a rigid transform in `SE(3)`: six numbers, three for the separation and three for the
neighbor's orientation about it. GVP's edge tuple gives one `CA→CA` distance and one unit direction —
one length plus two angles, three numbers, and those three pin only the separation. The neighbor's
*orientation* relative to me — the remaining three degrees of freedom, and the thing that decides which
face it packs with — reaches the center only indirectly, through the neighbor's own node vectors after
they have been mixed and normed. So per edge the geometry input under-determines the pose by half: two
residues at the same `CA` separation but rotated differently look nearly identical to the edge, and the
network has to reconstruct the missing orientation from the node channels rather than being handed it.
That is thin edge geometry stated precisely. Second, the encoder-only amputation hurts GVP more than it
hurts a method that was designed to put its strength in the encoder: the GVP story leans on the
autoregressive decoder to model residue-residue couplings, and here that decoder is gone, replaced by a
per-residue MLP. So GVP is carrying its geometric machinery into a setting that has stripped away the
half of the model it expected to share the work with.

The falsifiable expectation, then: GVP should clear the dumb baselines and be invariant and stable, but
land at the *bottom* of the three-rung ladder on recovery, precisely because its per-edge geometry is
thin and its decoder is amputated. Read what those two numbers would mean physically: recovery in the
low-0.43 range is the argmax landing on the native residue about three times in seven, roughly eight-
fold above the 0.05 a uniform guess over 20 amino acids gives, so the encoder is plainly reading real
structure; and a perplexity above 5.5 is `exp` of the mean per-residue NLL, i.e. the model has narrowed
twenty options down to between five and six effective ones. That is a genuine reduction but a long way
from the four-or-so a richer encoder ought to reach, and it is exactly the signature of an encoder that
sees the neighborhood but cannot resolve *which* neighbor is doing what. So I expect recovery in the
low-0.43 range on CATH 4.2 and CATH 4.3, with perplexity well above 5.5 — comfortably worse than a
method that enriches the edge geometry with many atom-pair distances; and because TS50 is de novo, more
idealized structures, I expect its recovery
a little higher and its perplexity a little lower than the two CATH sets even at this floor. If that
holds, the diagnosis for the next rung writes itself: the fix is not *more* invariance machinery — I
just proved the plain MPNN is the `v = 0` corner of GVP and that live vectors did not rescue thin edges —
but *richer invariant features* on the edges, pouring more of the relative pose into the graph through
additional inter-atomic distances that recover the orientation degrees of freedom the single `CA→CA`
direction misses. That is exactly where I will go next. The full encoder-only scaffold module is in the
answer.
