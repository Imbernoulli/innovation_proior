Let me start from the thing that actually limits me. I have a per-residue frame `T_i = (R_i, t_i)` built
from the backbone — translation at `Cα`, rotation from `N, Cα, C` by Gram–Schmidt — and I want a layer
that reads the geometric relationship between two residues' frames and turns it into features that update
the residue representation. The dominant primitive for this is invariant point attention: each residue
emits query and key *points* in its own frame, and the geometric attention term is the squared distance
between `i`'s query points and `j`'s key points after both are brought into a common frame. It works, and
it is invariant. But I want to understand exactly what information it keeps and what it throws away,
because that is where any improvement has to come from.

Write the IPA geometric term out and look at it. For center query points `Q_i` (a small cloud in `i`'s
frame) and neighbor key points `K_j` (the neighbor's points brought into `i`'s frame), the term is
`Σ_{a,b} ‖q_a − k_b‖²`. Expand it: `Σ_{a,b}(‖q_a‖² + ‖k_b‖² − 2 q_a·k_b) = M Σ_b‖k_b‖² + N Σ_a‖q_a‖²
− 2 (Σ_a q_a)·(Σ_b k_b)`. So as a function of the *neighbor* cloud, this scalar depends on `K_j` only
through two quantities: the total squared norm `Σ_b‖k_b‖²` and the centroid `Σ_b k_b`. Everything else
about the shape and orientation of the neighbor cloud — how its points are arranged relative to the
center — has been integrated out before any learnable weight touches it. That's a strong claim; check it
on numbers before trusting it.

Construct two genuinely different neighbor clouds with the *same* total norm and centroid and see whether
IPA's scalar really cannot tell them apart. Take a neighbor cloud `K_j` centered at the origin (so its
centroid is `0`), and rotate the whole cloud about the origin to get `K_j'`. A rotation about the origin
preserves every `‖k_b‖²` individually, hence `Σ_b‖k_b‖²`, and keeps the centroid at `0`. So by the
expansion the IPA scalar should be identical for `K_j` and `K_j'` even though the clouds are rotated
versions of each other. With four random points and a random rotation: the centroids stay at `~1e-8`,
`Σ‖k‖²` is `17.8390` versus `17.8390`, and the IPA squared-distance scalar comes out `89.314323` for `K_j`
and `89.314346` for `K_j'` — equal to six figures, the residual being float noise, while the two clouds
are demonstrably not the same array. Two neighbor environments that differ by a rotation relative to me
are invisible to the scalar IPA computes. The geometry between two residues is a relationship between two
little clouds of points in 3D, and IPA collapses that relationship into a *scalar* — a sum of squared
point distances — before anything learnable touches it, blind to *direction*. That is the bottleneck to
break.

So what would "not collapse to a scalar" look like while staying invariant? The instinct is to keep the
geometry as *vectors* and only reduce to scalars at the very end. What object should the vectors come
from? Each residue carries a set of points anchored in its frame. Real backbone atoms are an obvious set,
but they are few and fixed, and the side chain — which is what actually distinguishes amino acids — is not
in the input. So give each residue a set of *virtual atoms*: learnable points whose coordinates are
parameters expressed in the local frame, shared across residues. Because they live in the frame, they
rotate and translate with the residue, so any frame-relative quantity computed from them should be
invariant — confirmed later once the full feature is built. Shared coordinates force the optimizer to
learn a *transferable* geometric probe — a consensus set of points in the residue frame that best help
discriminate identity — rather than arbitrary per-example points. PiFold already has such virtual atoms,
but it reads only their pairwise distances, which is the same scalar collapse one notch down: from the
expansion above, pairwise distances feed exactly the norm-and-centroid projection just shown to be
direction-blind. I want to read *vectors* between them.

Now make the relationship between two residues concrete. Residue `i` carries virtual atoms `Q_i` in its
own frame; residue `j` carries `Q_j` in *its* frame. To relate them, put them in a common frame — the
center residue `i`'s frame is the natural choice. The neighbor's atoms in world coordinates are
`R_j Q_j + t_j`; bringing those into `i`'s frame gives `K_j = R_i^T(R_j Q_j + t_j − t_i)`, which should
equal `T_{i←j} ∘ Q_j` with `T_{i←j} = T_i^{-1} ∘ T_j`. Sign and transpose conventions are easy to get
wrong here, so check it numerically: with random frames `(R_i,t_i)`, `(R_j,t_j)` and random `Q_j`, compute
`K_j` by the one-line formula and separately by composing the two rigid maps directly — apply `T_j`
(`R_j q + t_j`), then `T_i^{-1}` (`R_i^T(· − t_i)`). The maximum absolute difference is `0.0` exactly: the
formula is the composition, not an approximation of it. The center's own atoms in its frame are just
`Q_i`. So in `i`'s frame there are now two sets of 3D points — `Q_i` (the center) and `K_j` (the
neighbor) — and the global rotation should cancel, because under `X → R_g X + t_g` the frames become
`R_i → R_g R_i`, `t_i → R_g t_i + t_g`, and `R_i^T R_g^T R_g (·) = R_i^T (·)`. I'll verify that
cancellation once I have the full feature, since that is the property everything rides on.

What computation do I run on two sets of vectors? Two existing recipes bound the space. GVP carries
vector features through the graph and restricts every operation on them to ones that provably commute
with rotation — channel-linear maps, taking norms, and scaling a vector by a (gated) function of its own
norm — but it has no explicit notion of two different residues' local frames being related by composing
rigid transforms; its vector features are displacement-style vectors already sitting in one shared frame,
not a learnable per-residue point set brought in from a neighbor's own frame the way `T_{i←j}` does here.
Tensor-field networks go further still: represent everything as spherical-harmonic irreps and combine
them with Clebsch–Gordan tensor products — fully general, but it pays for that generality with
angular-momentum bookkeeping and higher-degree channels I don't need, since I only ever have order-1
vectors (virtual-atom positions), never higher tensors. What I actually want is GVP's basic
primitive — a channel-linear combination of vector inputs — applied to a different substrate: a shared,
learnable virtual-atom set per residue, explicitly related across residues through the frame composition
just derived, and reduced richly (direction *and* magnitude) rather than collapsed to a norm alone.
Treat that combination like a *linear layer*, but with 3D vectors as the channel values instead of
scalars. An ordinary linear
layer takes scalar inputs `x_l` and outputs `y_k = Σ_l w_{k,l} x_l`. Do the same, but let the `x_l` be the
virtual-atom *vectors*, so the output channels are themselves vectors:
`h⃗_k = Σ_l w^a_{k,l} q⃗_l + Σ_l w^b_{k,l} k⃗_l`, with learnable scalar weights `w^a, w^b` mixing the
center atoms and the neighbor atoms into `d_out` output *vectors* in `R^3`. The weights need to be
*scalars* multiplying whole vectors, not `3×3` matrices, because scalar weights make the operation
commute with rotation of the common frame: if I rotate `Q_i, K_j` by some `U`, every `h⃗_k` should rotate
by the same `U`, since formally `Σ w (vU) = (Σ w v)U`. A `3×3`-matrix weight would secretly break that —
it would mix the x/y/z components of a vector, which does not commute with an arbitrary rotation `U` the
way scalar multiplication does. Check it against the implementation, not just the formula: with random
`w^a, w^b`, random clouds `q,k`, and a random rotation `U`, compare `(Σ w v) U` (rotate the output)
against `Σ w (vU)` (rotate the inputs, then combine) — max absolute difference `9.5e-7`, float noise. The
operator is genuinely equivariant in the common frame, which is what I want *before* I reduce.

Now reduce to invariant scalars, but reduce *late* and reduce *richly*. The norm of a vector is invariant;
so is the unit direction *relative to the (already-canonical) center frame*. So for each output vector
keep both: the unit direction `h⃗_k / ‖h⃗_k‖` (three numbers that survive because the center frame is
canonical) and the magnitude lifted into an RBF basis `RBF(‖h⃗_k‖)`. Concatenate over the `d_out`
channels: `g_{i,j} = concat_k( h⃗_k/‖h⃗_k‖ , RBF(‖h⃗_k‖) )`. This is the geometric feature for the edge
`(i, j)`, an invariant vector of length `d_out · (3 + num_rbf)`. The RBF on the magnitude (rather than the
raw norm) turns "this length is about `μ_n`" into a soft one-hot the downstream linear layer can act on
like a lookup.

Two properties need checking, since the whole design rests on them. First: is `g_{i,j}` actually
invariant to a global rigid motion? Apply a random `R_g, t_g` to the frames (`R_i → R_g R_i`,
`t_i → R_g t_i + t_g`, same for `j`), keep the virtual-atom *parameters* `Q_i, Q_j` fixed (they are frame
coordinates, so a global motion of the molecule does not change them), and recompute the whole feature —
neighbor-into-center-frame, vector field operator, unit-direction + RBF. Max absolute change in `g_{i,j}`:
`7.7e-7`. The cancellation sketched above holds through the full reduction, not just in the algebra.
Second — the payoff — does the feature actually *distinguish* the two clouds that IPA could not? On the
same `K_j` versus rotated-about-centroid `K_j'` pair from before (identical IPA scalar to six figures),
the vector geometric feature differs by `1.00` in max absolute value, almost all of it in the
unit-direction components. `g_{i,j}` keeps `d_out` learnable directions *and* their RBF-coded
magnitudes — many more geometric numbers than IPA's one scalar — and the *directions* IPA discarded are
demonstrably back: `1.00` of separation where IPA had `0`.

With the geometric feature in hand, the layer is an ordinary attention-plus-update block, except the
attention and value functions get to see `g_{i,j}`. Attention weight from the center node, neighbor node,
geometric feature, and edge feature: `a_{i,j} = softmax_j( MLP(s_i, s_j, g_{i,j}, e_{i,j}) )`, softmax
over the center's neighbors, scaled by `1/√d_head` for the usual saturation reason. Value from the
neighbor, geometry, and edge: `v_{i,j} = MLP(s_j, g_{i,j}, e_{i,j})`; aggregate `o_i = Σ_j a_{i,j} v_{i,j}`
and residual-update `s_i ← s_i + MLP(o_i)`, followed by a wide feed-forward, each with a norm. Update the
edges too, because a static edge wastes the most geometry-laden channel — re-derive each edge from the
refreshed endpoints and its geometry, `e_{i,j} ← e_{i,j} + MLP(s_i, s_j, g_{i,j}, e_{i,j})`.

There is one more thing to decide: the virtual atoms are fixed parameters so far, but they could *move* as
the representation refines, which would let the geometric probe adapt per residue through depth. Two
updates are on the table. A node-feature-based update simply re-predicts the atoms from the current node
state, `Q_i ← Linear(s_i)` — cheap, and it lets each residue's probe specialize to what its embedding now
knows. A coordinate-aggregating update instead pulls neighbor atoms in,
`Q_i ← V-MLP(Q_i, Σ_j a_{i,j} K_j)`. The second is more expressive but mixes coordinates across residues
inside the update, which is exactly the kind of step where an invariance bug can creep in; the
node-feature update predicts coordinates from already-invariant node features, so it stays frame-anchored
by construction and is the lighter, more stable choice. Lean on the node-feature update: predict the
virtual-atom coordinates directly from the node features at the end of each layer.

Now the real test: is the *whole stacked network* invariant, not just the geometric feature in isolation?
The per-feature check above is necessary but not sufficient — the attention MLPs, the edge update, and the
virtual-atom re-prediction could each leak a frame somewhere. Build a small instance of the actual model
(a few layers, `eval` mode to kill dropout), feed it a synthetic 12-residue backbone, and run two things:
the original coordinates, and the coordinates after a random global rotation *and* translation. The model
also consumes the kNN graph (`E_idx`), the neighbor distances, and the backbone dihedrals, so the
invariance claim only holds if *those* inputs are themselves invariant — check them first. After the
rigid motion the kNN graph is identical (the neighbor relation is by `Cα` distance, which a rigid motion
preserves), and the dihedral features match to `9.5e-7` (they are computed from invariant geometry). With
invariant inputs confirmed, the output log-probabilities move by `4.8e-7` under rotation+translation, and
by `4.8e-7` under pure translation — float noise in both cases. The output distribution does not move
when the whole protein is moved: SE(3)-invariant in practice, because every geometric quantity entered
the network through the center frame and the graph and dihedral inputs were invariant to begin with. The
test also pins down the honest caveat: the network is invariant *given* an invariant graph and invariant
node features, not automatically invariant if fed frame-dependent inputs. The cost is
`O(L·k·d_q·d_out)` per layer, linear in length; with modest `d_q` and `d_out` it scales to whole proteins.

For the design task itself, weigh autoregressive against one-shot decoding. The autoregressive
factorization models residue-residue dependencies through generated tokens, but most of those dependencies
are already induced by the shared structure — and the encoder now reads that structure with high
geometric fidelity, since it retains the directional information IPA discarded. A single parallel forward
pass plus a linear head should suffice: `p(s_i | X) = log_softmax(W h_i)`, trained with per-residue
cross-entropy. This is the bet PiFold makes too, and it pays off in speed; the risk is that one-shot
decoding underperforms on the residual sequence-level dependencies, which is an empirical question for
CATH recovery, not one I can settle here. For depth, fifteen layers, deeper than PiFold's ten, on the
reasoning that each layer's geometric feature is richer (it carries directions, not just distances) and so
benefits from more rounds of refinement — though the exact depth is a hyperparameter to tune, not derive.

Sanity-check by removing each piece. Drop the vector field operator and keep only scalar distances
between virtual atoms, and this is back to PiFold's distance-only reading — which the expansion above
showed feeds the same direction-blind norm-and-centroid projection, so the bottleneck returns. Drop the
virtual atoms and use only real backbone atoms, and the probe loses the side-chain-like capacity that
distinguishes residues. Reduce the geometric feature to just the norms (drop the unit directions), and
direction collapses back into magnitude — concretely, the `1.00` of separation between the two clouds
above lived almost entirely in the unit-direction components, so dropping them would put this back near
IPA's `0`. Freeze the virtual atoms (drop the node-feature update), and the probe cannot specialize
through depth. Each ablation removes something with a nameable consequence.

What ships is exactly that stack: per-residue frames with shared learnable virtual atoms; the vector
field operator turning frame-anchored atom vectors into a learnable linear combination, reduced to
invariant unit-directions plus RBF-coded magnitudes only at the very end; an attention-plus-edge-update
block that consumes this richer geometric feature; a node-feature virtual-atom update so the probe
specializes through depth; fifteen layers; and a one-shot linear decoder under per-residue cross-entropy —
modeling frames with the capacity of a vector-valued linear layer rather than a scalar distance-sum, while
staying exactly invariant and linear in length. The layer and encoder that implement this are the code:
`rigid_frames` and the harness's `_rbf`/`knn_graph` supply the invariant inputs the whole argument above
depends on, and the vector field operator and attention/edge/virtual-atom-update block are the pieces just
derived.
