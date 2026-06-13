Let me start from what actually hurts. I have a protein backbone — a chain of residues, each with its N, Cα, C, O atoms sitting at definite points in 3D — and I want a network that reads this structure and predicts, per residue, which amino acid belongs there. The thing that keeps biting me is that the problem has two natures at once. Part of it is pure geometry: which way a residue points, how the backbone curves, where a pocket opens up, whether two neighbors sit on the same side of me or on opposite sides — that's directions and angles and shapes in space. And part of it is relational: who is in contact with whom, the connectivity pattern of interactions, the order along the chain. The two architectures I have on the shelf each nail exactly one of these and fumble the other, and I keep being forced to give one up to get the other.

The convolutional route — voxelize the atoms into a 3D occupancy grid and run a 3D-CNN — is genuinely good at geometry. Its filters latch onto motifs and pocket shapes directly in space. But it pays for that three ways: it's not invariant to how I happen to orient the molecule, so I have to augment with rotations and hope; the grid resolution trades memory against geometric precision; and it has thrown away the residue-residue graph entirely, so any relational reasoning has to be painfully re-learned through dense volumetric filters. The graph route is the mirror image. Build a proximity graph over residues, do message passing à la Gilmer — for each directed edge form a message m_{j→i} = g(h_i, h_j, e_{j→i}), aggregate the incoming ones at each node, update h_i — and now relational reasoning is native and cheap. The catch is the geometry. A protein's answer doesn't change if I rotate or reflect the whole coordinate frame, so my scalar predictions must be invariant to any unitary R acting on all the coordinates. The standard way a graph method buys that invariance is to crush every directional quantity into rotation-invariant scalars before it ever enters the network.

I should look hard at how the best graph method for this task actually does that, because that's the wall I'm going to hit. Ingraham's Structured Transformer is the state of the art: it casts design as autoregressive language modeling conditioned on structure, p(s | x) = ∏_i p(s_i | x, s_{<i}), with an encoder that builds sequence-independent per-residue embeddings from a k-nearest-neighbor graph (k = 30 by Cα distance) and a decoder that predicts each residue attending to the full structure and the previously decoded residues under a causal mask. Their geometry trick is the part I want to stare at. They first noticed something I have to take seriously: distance-only edge features are not "locally informative." If all an edge carries is ‖x_j − x_i‖, then for two neighbors a and b I learn ‖x_a − x_i‖ and ‖x_b − x_i‖ but I genuinely cannot tell whether a and b are on the same side of me or opposite sides — the pairwise distances are blind to that. So a graph that wants to reason about local 3D geometry has to carry more than length on its edges, enough to reconstruct the neighbor positions up to rigid motion. Their fix: pin a local coordinate frame O_i = [b_i, n_i, b_i × n_i] to every node (b_i the bisector of the rays to my chain neighbors, n_i the normal to that plane), and then describe each edge as the triple ( RBF(‖x_j − x_i‖), O_i^T (x_j − x_i)/‖x_j − x_i‖, q(O_i^T O_j) ) — distance in a radial basis, the direction to the neighbor expressed in my own frame, and the quaternion of the relative rotation between our frames. Project everything into the local frame and it's invariant by construction; then run an ordinary scalar network.

It works, and I see exactly why it works, and I also see exactly where it dead-ends. The geometry gets collapsed into invariant scalars at the door. Once that first projection happens, every intermediate representation is a bag of scalars, and the network can never again touch the directional quantities as geometric objects. It cannot, three layers deep, decide to propagate a position in a shared frame, or point at a location in space that isn't itself one of the nodes, or update an orientation as the embedding refines. The geometry is frozen at input resolution. And it's stored redundantly — an orientation has to be re-encoded relative to every neighbor's frame O_i, rather than written down once, absolutely, per node. That redundancy and that freezing are two symptoms of one disease: the only way they knew to be invariant was to stop being geometric immediately.

The other route on the table goes the opposite way and refuses to give up the geometry at all — the tensor-field / Cormorant line. Carry actual geometric tensors through every layer, organized as irreducible representations of the rotation group: scalars, vectors, higher-order tensors, with convolutional filters built from a learned radial part times a spherical harmonic, coupled by tensor (Clebsch–Gordan) products so the whole network is exactly equivariant layer by layer. That is the principled thing; it's equivariant everywhere and needs no augmentation. But the spherical-harmonic and Wigner-D and tensor-product machinery is heavy — carrying and coupling higher-order irreps is expensive enough that in practice it's been stuck on small molecules. I have proteins with hundreds of residues. I can't afford it.

So here's the gap, stated cleanly: I want the graph method's full relational expressivity and cheapness, but I refuse to freeze the geometry into scalars at input the way Ingraham does, and I refuse to pay the irreps tax the way tensor field networks do. I want directional information to flow through the graph as honest 3D objects that I can keep manipulating at every layer, while my scalar outputs stay invariant. Let me think about what that even requires, mechanically.

Suppose I let a node (and an edge) carry, alongside its ordinary scalar features s ∈ ℝ^n, a set of vector features V ∈ ℝ^{ν×3} — ν little arrows in ℝ^3. The whole game is then: what operations am I allowed to apply to V such that when I rotate or reflect the input — multiply every vector on the right by a unitary U ∈ ℝ^{3×3} — the vector outputs rotate the same way (equivariance) and the scalar outputs don't move at all (invariance)? Let me just enumerate the moves and check each against right-multiplication by U.

Take a linear combination of the vector channels: a matrix W acting on the channel index, (WV). Does it commute with U? (WV)U = W(VU) — yes, trivially, because W touches the ν index and U touches the 3 index; they're on opposite sides and never interfere. So channel-mixing linear maps are fine, and they're equivariant: rotate the input, the output rotates identically. Scalar (per-channel) multiplication: scale row i of V by some number c_i. That's left-multiplication by a diagonal matrix D, and (DV)U = D(VU) — commutes again. Fine, equivariant, provided the scaling factors c_i are themselves invariant (don't depend on the orientation). And the L2 norm of a row, ‖v‖: since U is unitary, ‖vU‖ = ‖v‖ — the norm is flatly invariant. That's my bridge: the norm turns a vector into a scalar in a way that doesn't care how the molecule is oriented.

Now the things I must not do. A bias on a vector channel — add a fixed vector v0 — breaks it: (v + v0)U = vU + v0U ≠ vU + v0 unless v0 = 0, because there is no nonzero constant vector that's invariant under all rotations. So vector linear maps must be bias-free. And a coordinate-wise nonlinearity on the three components of a vector, like ReLU(v_x), ReLU(v_y), ReLU(v_z) — that's a disaster, because the components are defined only relative to my arbitrary axes; rotate the molecule and the components scramble, so ReLU applied component-wise gives a different, orientation-dependent answer. I cannot put a pointwise nonlinearity on coordinates.

That last constraint is the one that worries me, because a network with no nonlinearity on its vector path is just a stack of linear maps — it collapses. I need *some* nonlinearity on the vectors that respects equivariance. But the only invariant handle I have on a vector is its norm. So the move is forced: take the norm of each output vector (invariant), pass it through any nonlinearity I like (the norm is a scalar, I can do whatever), and use the result to *scale* the vector. v' = σ⁺(‖v‖) · v. Check it: ‖vU‖ = ‖v‖ so σ⁺(‖vU‖) = σ⁺(‖v‖), and σ⁺(‖vU‖)·(vU) = (σ⁺(‖v‖)·v)U — equivariant. It keeps the direction of v exactly and modulates only its length by a learned-or-fixed function of its own length. That's the only kind of vector nonlinearity that survives, and it drops out of the constraints rather than being chosen. Good.

So I have a closed toolkit for the vector path — channel-linear maps without bias, L2 norms, and scale-by-a-function-of-the-norm — and the norm as the one-way bridge from vectors into scalars. Let me now actually build the smallest module that processes a tuple (s, V) into a new tuple (s', V'), a drop-in replacement for a dense layer that happens to also handle vectors. The scalar path needs to be informed by the vectors (otherwise the geometry never reaches the prediction), the vector path needs to be transformable, and both need their own nonlinearity.

First cut. Scalar output: transform the scalars, but first let them see the vectors. The only invariant thing I can hand them is norms, so I'll concatenate the norms of (transformed) vector channels onto s and then apply a linear layer plus a scalar nonlinearity. Vector output: a channel-linear map followed by the scale-by-norm nonlinearity. Let me write it and then immediately poke at it. Say I transform the input vectors by a matrix W_h to get V_h ∈ ℝ^{h×3}, take their row-wise norms s_h = ‖V_h‖ ∈ ℝ^h, concatenate with s, run W_m and σ to get s'. And for the vectors, scale V_h by σ⁺ of its own norms: V' = σ⁺(‖V_h‖) ⊙ V_h.

Now the poke. If I use the *same* V_h both to feed norms into the scalar path and to be the vector output, I've tied two dimensionalities together that have no business being equal. The number of norms I want to extract for the scalars (call it h) is about how much rotation-invariant geometric information I want to summarize; the number of vector channels I want to output (call it μ) is about how rich a directional representation I want to pass on. With one matrix W_h doing both jobs, h must equal μ. That's an artificial coupling — I might want many norms but few output vectors, or vice versa. So I'll split it: W_h maps the input vectors to an intermediate V_h ∈ ℝ^{h×3} with h = max(ν, μ) channels (generous enough to extract the norms I need), and a *second* bias-free linear map W_μ produces the actual vector output V_μ = W_μ V_h ∈ ℝ^{μ×3}. The scalars read s_h = ‖V_h‖ ∈ ℝ^h (h norms), and the vector output is the scale-by-norm of V_μ, namely V' = σ⁺(‖V_μ‖) ⊙ V_μ ∈ ℝ^{μ×3}. Now h and μ are decoupled. The whole thing is:

V_h = W_h V, V_μ = W_μ V_h, s_h = ‖V_h‖ row-wise, v_μ = ‖V_μ‖ row-wise, s' = σ(W_m concat(s_h, s) + b), V' = σ⁺(v_μ) ⊙ V_μ. The only learned weights are W_h, W_μ (bias-free, on vectors) and W_m, b (the ordinary biased linear on scalars). I'll call this little thing a geometric vector perceptron — it's a perceptron that has grown a vector path.

Let me make sure I haven't quietly broken equivariance with the split, by doing the proof end to end, because I'm going to lean on this property hard. Rotate/reflect the input: V → VU with U unitary. The scalar output is s' = σ( W_m [ ‖W_h V‖ ; s ] + b ). The only place V enters is through ‖W_h V‖, and ‖W_h (VU)‖ = ‖(W_h V) U‖ = ‖W_h V‖ because U is unitary and acts on the right (the 3-axis) while W_h acts on the left (the channel axis). So s' is exactly unchanged — invariant. For the vectors: V' = σ⁺(‖W_μ W_h V‖) ⊙ (W_μ W_h V). The scaling σ⁺(‖·‖) is a diagonal matrix D acting on the channel index, and as just argued ‖W_μ W_h (VU)‖ = ‖W_μ W_h V‖, so D is invariant. Then D W_μ W_h (VU) = (D W_μ W_h V) U because every left-acting factor (D, W_μ, W_h) commutes with the right-acting U. So V'(VU) = (V'(V)) U — equivariant. Both halves check out, and it's because the only operations I ever applied to vectors were channel-linear maps, norms, and norm-based scalings, which is exactly the closed toolkit I derived. The σ⁺ footnote matters: σ⁺ is a *scaling* applied via the norm, not a function applied to coordinates — that's what keeps it equivariant.

I want one more guarantee before I trust this, because "it's invariant" is cheap — a network that ignores its vectors is also invariant. The question is whether this module is *expressive enough* to compute the invariant geometric functions I care about. Concretely: can a GVP that takes only vectors in and emits only scalars (n = 0, μ = 0), followed by a dense layer, approximate any continuous rotation/reflection-invariant scalar function of the input vectors? If yes, then the geometric reasoning isn't being lost — it's all recoverable from the norms.

Let me try to construct it. Take F : Ω^ν → ℝ continuous and invariant, F(R(V)) = F(V), where Ω^ν is the bounded set of V whose first three vectors v_1, v_2, v_3 are linearly independent (so I can canonically orient any element). The intuition: any invariant function only depends on the rotation-and-reflection-invariant "shape" of the vector cloud, and the shape is fully captured by the norms and pairwise inner products of the vectors. So if I can recover the inner products from norms — which is all a GVP's scalar path produces — I can recover the shape, and then it's an ordinary function-approximation problem.

Make this precise. Call V *oriented* if v_1 lies along e_x, v_2 in the xy-plane, v_3 has positive z, etc.; define the orientation function ω that orients its input and reads off the 3ν − 3 coefficients [x_1, x_2, y_2, x_3, y_3, z_3, …]. These coefficients can be written purely in norms and inner products: x_1 = ‖v_1‖; x_i = v_i · v_1 / x_1 for i ≥ 2; y_2 = √(‖v_2‖² − x_2²); y_i = (v_i · v_2 − x_i x_2)/y_2 for i ≥ 3; z_3 = √(‖v_3‖² − x_3² − y_3²); and so on — Gram–Schmidt in disguise. Every one of these is built from norms and inner products of the v_i, hence invariant, so F factors as F = F̃ ∘ ω with F̃ a function on a hypercube. Now, can the GVP's scalar path produce the raw ingredients? An inner product v_i · v_j I can get from norms via the cosine law: v_i · v_j = (‖v_i‖² + ‖v_j‖² − ‖v_i − v_j‖²)/2. So if I build W_h so that the rows of W_h V are exactly the original vectors v_i and the Gram-Schmidt differences — v_i − v_1 for i ≥ 2, v_i − v_2 for i ≥ 3, and v_i − v_3 for i ≥ 4 — then the row-wise norms ‖W_h V‖ hand me ‖v_i‖ and the needed ‖v_i − v_j‖ values, 4ν − 6 norms in all. The cosine law gives every needed inner product, and then ω(V) follows. The GVP computes exactly those row-wise norms as its intermediate scalar y = ‖W_h V‖; the remaining map from y through the dense layer with sigmoidal σ to F̃ ∘ (cosine law) is exactly the setting of the classical universal approximation theorem (Cybenko 1989): a one-hidden-layer net with a sigmoidal nonlinearity is dense in continuous functions on a compact set. So for any ε there is a form f(V) = w^T G_s(V) with |F − f| < ε. The GVP inherits universal approximation of invariant functions. That settles it — the norm bridge is not lossy; the scalar path can in principle reconstruct any invariant geometric quantity. And it tells me the σ in G_s wants to be sigmoidal for that argument to go through, while I'm free to use ReLU in the working layers.

Before I wire these into a GNN I need the supporting pieces — normalization and dropout — and they have to respect equivariance too, which immediately rules out the off-the-shelf versions on the vector path. Ordinary LayerNorm subtracts a mean and divides by a per-feature std and then applies learned per-feature scale and shift; on a vector channel, subtracting a mean vector and per-coordinate scaling would both break equivariance (a mean vector isn't invariant; per-coordinate scaling isn't a unitary-commuting operation). What *is* allowed is an overall rescaling of the vectors by an invariant scalar. So for the vectors I normalize by their root-mean-square norm: V ← V / √( (1/ν) ‖V‖²_F ), i.e. scale all the row vectors so their RMS length is one. No mean subtraction, no per-coordinate or even per-channel parameters — a single global invariant scale, so it commutes with U and stays equivariant. On the scalar path I keep ordinary LayerNorm with its trainable γ, β. Same logic for dropout: dropping individual coordinates of a vector would break equivariance (the dropped coordinate is axis-dependent), so I drop *entire vector channels* at random — zero out a whole arrow or keep it — which is just a per-channel invariant 0/1 scaling, equivariant by construction. The scalar path uses ordinary dropout.

Now assemble the GNN. I'll instantiate Gilmer's message passing with GVPs as the learned functions. Each node i carries a tuple h_v^{(i)} = (scalars, vectors); each edge (j→i) carries h_e^{(j→i)} = (scalars, vectors). The message from j to i is m^{j→i} = g( concat( h_v^{(j)}, h_e^{(j→i)}, h_v^{(i)} ) ) where g is a short stack of GVPs (concatenation here means concatenate the scalar parts and concatenate the vector parts separately — they live in different spaces); I feed in both endpoints, the neighbor h_v^{(j)} and the node itself h_v^{(i)}, alongside the edge, so the message can depend on the pair. Aggregate the incoming messages by averaging over the k' neighbors I actually have, and do a residual update with the equivariant LayerNorm: h_v^{(i)} ← LayerNorm( h_v^{(i)} + (1/k') Dropout( Σ_j m^{j→i} ) ). Between propagation steps, a per-node feed-forward update, also residual and normed: h_v^{(i)} ← LayerNorm( h_v^{(i)} + Dropout( g(h_v^{(i)}) ) ), with g here a shorter GVP stack. Critically, because each GVP updates both scalar and vector channels, the *vector* features at every node get refined as the network deepens — that's the whole point, the geometry stays live, unlike the scalar-only freeze. And the residual sums add tuples elementwise (scalar to scalar, vector to vector), which is a linear combination on the vector path, so equivariance survives the residual connections too.

For the message GVP stack g, I'll let the first GVP map the concatenated (source node ⊕ edge ⊕ target node) dimensions down to node dimensions, then a couple more at node dimensions, with the *last* GVP in the stack using identity activations (None, None) — I don't want a nonlinearity squashing the message right before I sum it; the nonlinearity already happened inside the stack, and a final raw linear keeps the message space expressive. Same trick on the feed-forward stack: expand to a wider hidden tuple (something like 4× the scalars and 2× the vectors), then contract back with a final identity-activation GVP. This mirrors the usual transformer-style wide feed-forward, just carried out on tuples.

Now I have to choose what scalars and vectors actually go on the nodes and edges, and here I can finally cash in the equivariant representation to *un-freeze* what Ingraham had to freeze. The diagnostic that pushed Ingraham to local frames — distance-only is not locally informative — still holds, but I no longer need a per-node frame to honor it, because I'm allowed to carry raw directions as vector features and let the GVPs extract whatever invariants they need via norms. So per node: the scalar features are the backbone dihedrals on the torus, {sin, cos} of (φ, ψ, ω) — six invariant scalars computed from consecutive C_{i-1}, N_i, Cα_i, C_i, N_{i+1}. The vector features are the directional quantities themselves, kept as honest arrows in the global frame: the forward unit vector toward Cα_{i+1} − Cα_i, the reverse unit vector toward Cα_{i-1} − Cα_i, and the imputed Cβ direction. That Cβ direction I get by assuming tetrahedral geometry: with n = N_i − Cα_i and c = C_i − Cα_i, the four bonds off the Cα sit at the tetrahedral angle, so the Cβ must lie opposite the average of the two backbone bonds (the in-plane N/C bisector) and tilt out of that plane along the normal. Decompose it into those two orthogonal pieces — the unit bisector (n+c)/‖n+c‖ and the unit normal (n×c)/‖n×c‖ — and fix the split by the tetrahedral angle: cos(half the in-plane opening) pins the bisector weight and the remainder goes to the normal. Working the geometry out, the Cβ contribution is √(2/3) along the plane-normal and −√(1/3) along the bisector, i.e. the unit direction √(2/3)(n×c)/‖n×c‖ − √(1/3)(n+c)/‖n+c‖. Let me pin the coefficients by checking against an exact tetrahedral residue (Cα at the origin with N, C, Cβ at perfect 109.47° angles): the bisector-and-normal weights √(1/3) on (n+c) and √(2/3) on (n×c) reproduce the true Cβ direction exactly, whereas swapping them lands ~19° off — so the larger weight √(2/3) belongs on the out-of-plane normal, not the bisector. (Equivalently, writing the normal as c×n flips its sign and the same vector reads −√(1/3)·bisector − √(2/3)·perp, with perp = c×n — the form the canonical data pipeline uses.) These three vectors together pin the residue's orientation, but as *vectors*, written once per node, absolutely — no per-neighbor redundancy. So node_in = (6 scalars, 3 vectors). Per edge (j→i): the scalar features are the Cα_j − Cα_i distance lifted into 16 Gaussian radial basis functions (centers 0–20 Å) plus a 16-dim sinusoidal encoding of the backbone offset j − i — 32 scalars; the vector feature is the single unit vector in the direction Cα_j − Cα_i, kept as a vector — 1 vector. So edge_in = (32, 1). Notice what I do with direction: Ingraham projected the neighbor direction into O_i to make it an invariant scalar triple; I keep it as a vector and let the GVP take whatever norms and inner products it wants. Same information, but now it stays geometric downstream.

Those are input dimensionalities; I want the hidden tuples richer. I'll first run input GVPs W_v, W_e (with identity activations, just to lift dimension) to bring nodes to (100 scalars, 16 vectors) and edges to (32 scalars, 1 vector), each followed by the equivariant LayerNorm, and then do graph propagation in those dimensions. Why those numbers — 100 scalars and 16 vectors at nodes? The scalar width is the usual "make the relational channel wide enough to carry the residue-context features"; 16 vector channels is a deliberately modest geometric width — each is a full arrow, so it's three numbers, and the value of the vector path is in keeping a handful of *meaningful* directions live, not in brute width. Edges stay narrow at (32, 1): an edge's job is mostly to gate and direct the message; one direction vector plus the distance/offset scalars is enough, and keeping edges thin saves the memory that the k·L edges would otherwise eat. Three propagation steps for the encoder give each residue several rounds of spatial-neighborhood context without turning the lightweight layer into the heavy machinery I was trying to avoid.

Now the task-level architecture for design. Following the autoregressive framing, the joint distribution factors as p(s | x) = ∏_i p(s_i | x, s_{<i}), so I want an encoder that sees structure only and a decoder that adds sequence causally. The encoder is the three GVP propagation steps on the structural graph, producing per-residue embeddings that depend on structure alone. Then I inject sequence: embed the known amino acids and, for each edge (j→i), append the *neighbor's* sequence embedding to the edge's scalar features — but mask it to zero whenever j ≥ i, so that residue i only ever sees the sequence of residues before it. That mask is what makes the factorization causal and lets me teacher-force at training (feed the true s_{<i}) while sampling left-to-right at inference. The decoder is three more GVP propagation steps on this sequence-augmented graph, but with a twist that keeps the causality exact even as the decoder deepens: for incoming edges from j ≥ i (the "future" direction), I must not let stale, sequence-contaminated decoder embeddings leak backward, so I compute those backward messages from the *encoder* embeddings (which never saw any sequence), while forward edges j < i use the live decoder embeddings; I sum the two and divide by the actual neighbor count. Finally one more GVP with a (20, 0) output — 20 scalars, no vectors — gives per-residue logits over the amino acids; softmax or log-softmax is applied when I sample or compute the cross-entropy loss.

Let me sanity-check the design against what would happen if I removed each piece, since that's the real test of whether every part is load-bearing. If I replace the GVP with a vanilla MLP on the merged features, I'm back to scalar-only geometry: the vectors get flattened into coordinates and an MLP on coordinates is not invariant, so to stay invariant I would have to pre-project to scalars and recreate Ingraham's freeze. If I propagate only scalars, direct geometric access is gone. If I propagate only vectors, I lose the dihedral scalars and the amino-acid identity, which are genuinely scalar inputs, and I also lose the scalar output path used in the approximation argument. If I drop W_μ and fuse the two vector transforms into one, I re-impose h = μ, the coupling I deliberately broke. Each piece earns its place by a specific failure it prevents.

So let me write the actual module. The GVP itself, processing a tuple (s, V):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    # L2 norm clamped above eps so the gradient never sees a 0/0 at the origin.
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


class GVP(nn.Module):
    """Geometric Vector Perceptron: (s, V) -> (s', V').
    s : (..., n) scalars (invariant);  V : (..., nu, 3) vectors (equivariant).
    Vectors are only ever channel-mixed (bias-free), normed, or scaled by a
    function of their norm -- the closed set of ops that commute with rotation."""

    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid)):
        super().__init__()
        self.si, self.vi = in_dims          # (n, nu)
        self.so, self.vo = out_dims         # (m, mu)
        self.scalar_act, self.vector_act = activations
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)        # h = max(nu, mu)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)   # W_h: nu -> h, no bias
            self.ws = nn.Linear(self.h_dim + self.si, self.so)     # W_m on concat(norms, s)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)  # W_mu: h -> mu, no bias
        else:
            self.ws = nn.Linear(self.si, self.so)

    def forward(self, x):
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)         # (..., 3, nu) so Linear hits the channel axis
            vh = self.wh(v)                         # V_h = W_h V   (..., 3, h)
            vn = _norm_no_nan(vh, axis=-2)          # s_h = ||V_h|| per channel -> invariant
            s = self.ws(torch.cat([s, vn], -1))     # scalars see the vector norms
            if self.vo:
                vm = self.wv(vh)                     # V_mu = W_mu V_h  (..., 3, mu)
                vm = torch.transpose(vm, -1, -2)     # (..., mu, 3)
                if self.vector_act:                  # V' = sigma+(||V_mu||) (.) V_mu  (row-wise)
                    vm = vm * self.vector_act(_norm_no_nan(vm, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                vm = torch.zeros(s.shape[:-1] + (self.vo, 3), device=s.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        return (s, vm) if self.vo else s
```

The equivariant LayerNorm and the vector-channel dropout, since the GNN layers need them:

```python
class GVPLayerNorm(nn.Module):
    """LayerNorm on a tuple: ordinary LayerNorm on scalars; vectors rescaled so
    their root-mean-square norm is 1 (a single invariant scale -> equivariant)."""
    def __init__(self, dims):
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)   # ||v||^2 per channel
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))      # RMS norm over channels
        return self.scalar_norm(s), v / vn


class _VDropout(nn.Module):
    """Drop whole vector channels (not coordinates) -> equivariant."""
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.drop_rate) *
                               torch.ones(x.shape[:-1], device=x.device)).unsqueeze(-1)
        return mask * x / (1 - self.drop_rate)


class Dropout(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)
```

And the message-passing layer — the GVP message stack, mean aggregation, residual + equivariant norm, then the wide GVP feed-forward — with the autoregressive option for the decoder, where backward (src ≥ dst) edges form their messages from a separate set of node embeddings:

```python
import functools
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add


def tuple_sum(*args):
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    return x[0][idx], x[1][idx]

def _merge(s, v):
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)

def _split(x, nv):
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    return x[..., :-3*nv], v


class GVPConv(MessagePassing):
    """m_{j->i} = g(concat(h_j, e_{j->i}, h_i)); aggregate over neighbors. g = stack of GVPs,
    last with identity activation so the summed message stays expressive. I concatenate both
    endpoints (source h_j AND target h_i) with the edge -- hence 2*si + se scalars in -- so the
    message can depend on the pair, not just the neighbor."""
    def __init__(self, in_dims, out_dims, edge_dims, n_layers=3, aggr="mean"):
        super().__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        GVP_ = functools.partial(GVP)
        module_list = [GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)]
        for _ in range(n_layers - 2):
            module_list.append(GVP_(out_dims, out_dims))
        module_list.append(GVP_(out_dims, out_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        x_s, x_v = x
        message = self.propagate(edge_index, s=x_s,
                                 v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                                 edge_attr=edge_attr)
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        m = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        return _merge(*self.message_func(m))


class GVPConvLayer(nn.Module):
    """Residual node update with aggregated GVP messages + equivariant LayerNorm,
    then a wide GVP feed-forward, residual + norm. Vectors are refined every step."""
    def __init__(self, node_dims, edge_dims, n_message=3, n_feedforward=2,
                 drop_rate=.1, autoregressive=False):
        super().__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                            aggr="add" if autoregressive else "mean")
        GVP_ = functools.partial(GVP)
        self.norm = nn.ModuleList([GVPLayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])
        hid = (4 * node_dims[0], 2 * node_dims[1])             # wide feed-forward tuple
        self.ff_func = nn.Sequential(
            GVP_(node_dims, hid),
            GVP_(hid, node_dims, activations=(None, None)))

    def forward(self, x, edge_index, edge_attr, autoregressive_x=None, node_mask=None):
        if autoregressive_x is not None:
            src, dst = edge_index
            fwd = src < dst                                    # past: use live decoder embeddings
            bwd = ~fwd                                         # future: use encoder embeddings only
            dh = tuple_sum(
                self.conv(x, edge_index[:, fwd], tuple_index(edge_attr, fwd)),
                self.conv(autoregressive_x, edge_index[:, bwd], tuple_index(edge_attr, bwd)))
            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)
        else:
            dh = self.conv(x, edge_index, edge_attr)
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x
```

Finally the inverse-folding model itself. The data pipeline has already featurized the backbone into the (6, 3) node and (32, 1) edge tuples. The model lifts those to node_h = (100, 16) and edge_h = (32, 1), runs three structure-only encoder layers, appends the causally masked sequence embedding to edge scalars, runs three autoregressive decoder layers, and outputs the 20 amino-acid logits:

```python
class CPDModel(nn.Module):
    """Structure-conditioned autoregressive protein design.
    node_in_dim=(6,3), node_h_dim=(100,16), edge_in_dim=(32,1), edge_h_dim=(32,1)."""
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.1):
        super().__init__()
        self.W_v = nn.Sequential(GVP(node_in_dim, node_h_dim, activations=(None, None)),
                                 GVPLayerNorm(node_h_dim))
        self.W_e = nn.Sequential(GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
                                 GVPLayerNorm(edge_h_dim))
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) for _ in range(num_layers))
        self.W_s = nn.Embedding(20, 20)
        edge_h_dim = (edge_h_dim[0] + 20, edge_h_dim[1])
        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate, autoregressive=True)
            for _ in range(num_layers))
        self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None))

    def forward(self, h_V, edge_index, h_E, seq):
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        encoder_embeddings = h_V
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0                 # residue i sees only seq[j] for j<i
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)
        return self.W_out(h_V)                                  # (n_nodes, 20) logits
```

The softmax over those logits gives the per-residue distribution; the training loss is the per-residue cross-entropy against the native sequence.

Let me trace the causal chain one more time. The problem is dual — geometric and relational — and the two architecture families each surrendered one face: CNNs and the SO(3)-irrep networks reason geometrically but are costly or non-relational, while graph methods reason relationally but, to stay invariant, froze all geometry into scalars at the input via per-node local frames, which made the geometry both redundant and un-manipulable downstream. I refused both surrenders and asked what operations on raw vector features commute with rotation; the answer — channel-linear maps without bias, L2 norms, and scaling by a function of the norm — is a closed toolkit that makes vectors equivariant and scalars invariant, with the norm as the one bridge from geometry into the scalar prediction. Packaging that toolkit as a perceptron with a scalar path that reads vector norms and a vector path that channel-mixes then scales-by-norm gives the GVP; splitting the vector transform into W_h (for the norms) and W_μ (for the output vectors) decoupled the number of extracted invariants from the number of propagated directions; and the universal-approximation argument — recover inner products from norms by the cosine law, then Cybenko — proved the norm bridge loses nothing. Dropping it into Gilmer's message passing, with an equivariant RMS LayerNorm and vector-channel dropout, gives a GNN whose vector features stay alive and get refined at every layer; feeding it the dihedral scalars and the three orientation vectors per node and the RBF/offset scalars plus the direction vector per edge — directions kept as honest arrows instead of local-frame scalars — un-freezes exactly what the prior graph method had to freeze; and wrapping it in the autoregressive masked encoder-decoder for design lands on a model that reasons relationally and geometrically at once, invariantly, at a cost that scales to whole proteins.
