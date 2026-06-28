Let me start from what actually hurts. I want a fast learned surrogate for PDEs on the meshes people really use — a car body, an airfoil, tens of thousands of irregularly placed points, complex non-periodic boundaries. The job is to take the discretized geometry, and whatever quantities are observed on it, and predict the solution field at every point in one forward pass. Two pressures pull against each other. The field is governed by intricate, often long-range physical interactions — the wake behind a car depends on its nose, the pressure on one surface couples to flow far away — so I need a model that captures *correlations* across the whole domain, not just local smoothing. And the domain has `N` in the tens of thousands, so anything quadratic in `N` is dead before it starts. Cheap on 32k irregular points, yet globally correlated, yet able to swallow an arbitrary unstructured geometry. Each tool I have does some of that. None does all of it.

The operator-learning view is the right frame, so let me set it up cleanly. Li and coworkers formalized PDE solving as learning a mapping between function spaces, and then sharpened each layer of that map into a non-local integral operator followed by a local nonlinearity. The nonlinearity is just a pointwise feed-forward layer; the thing with all the content is the integral operator

  G(u)(g*) = ∫_Ω κ(g*, ξ) u(ξ) dξ,

with κ a kernel on Ω×Ω. Every neural operator is a way to parameterize and evaluate that integral. FNO picks κ in the Fourier domain: a fixed basis, learnable spectral multipliers, truncated to low modes, evaluated with the FFT in O(N log N). That is fast and it works on a clean grid. But the FFT *is* the periodic-uniform-grid assumption. The moment the mesh is irregular with real boundaries — a car shape — the Fourier-deformation trick that maps the domain to a latent grid degenerates, and it degenerates seriously; geo-FNO falls apart on the practical design tasks even after a full hyperparameter sweep, because a car surface is nothing like a periodic torus. So fixed-basis spectral operators are out for the geometries I actually care about.

The graph-kernel route handles irregular meshes — approximate the integral with a learnable kernel over local graph neighborhoods, message passing on the mesh graph. Geometry-general, good. But the kernel is *local*. To carry information from the nose of the car to the wake I'd have to stack many message-passing steps, and even then global correlation is exactly what local graph kernels are worst at. So I get geometry but lose the long-range physics.

What gives me a *learnable* kernel and global reach at once? Attention. Cao, and Kovachki and coworkers, made this precise, and it's worth re-deriving because the whole approach is going to hinge on it. Take the integral operator and pick a particular row-normalized kernel — the attention kernel,

  κ(g*, ξ) = exp((W_q u(g*))(W_k u(ξ))^T / sqrt(d)) W_v / ∫_Ω exp((W_q u(g*))(W_k u(η))^T / sqrt(d)) dη.

Now discretize. With N mesh points g_1,…,g_N drawn from Ω, the normalizing integral for the fixed query g* becomes a Monte-Carlo sum,

  ∫_Ω exp((W_q u(g*))(W_k u(η))^T / sqrt(d)) dη ≈ (|Ω|/N) Σ_{j=1}^N exp((W_q u(g*))(W_k u(g_j))^T / sqrt(d)),

and using the same Monte-Carlo quadrature for the outer integral G(u)(g*) ≈ (|Ω|/N) Σ_i κ(g*, g_i) u(g_i), the |Ω|/N factors cancel between numerator and denominator and I'm left with

  G(u)(g*) ≈ Σ_{i=1}^N  exp((W_q u(g*))(W_k u(g_i))^T / sqrt(d)) / ( Σ_{j=1}^N exp((W_q u(g*))(W_k u(g_j))^T / sqrt(d)) )  W_v u(g_i),

which is exactly the row of softmax attention for query g*: the denominator sums over the key/value index while the query stays fixed. Before I lean on this I want to be sure the |Ω|/N quadrature factors really do drop out and leave a *bare* softmax — otherwise the "attention = integral" identification has a leftover mesh-density factor and isn't clean. Take the coefficient that multiplies u(g_i) in G: it's (|Ω|/N)·κ(g*,g_i), and κ itself carries a 1/((|Ω|/N)·Σ_j exp(·)) from its own normalizing integral. So the coefficient is (|Ω|/N)·exp(·)_i / ((|Ω|/N)·Σ_j exp(·)_j). With three toy logits exp(·) = (1.3, 0.7, 2.1), the coefficient on the first node at quadrature scale |Ω|/N = 10 is 1.3·10/(10·4.1) = 0.31707, and at scale 0.001 it is 1.3·0.001/(0.001·4.1) = 0.31707 — identical to five places, and both equal the plain softmax weight 1.3/4.1 = 0.31707. The |Ω|/N cancels exactly, so attention literally *is* the integral operator with the mesh points as quadrature nodes, no residual density factor, and unlike FNO the kernel is learned rather than fixed to Fourier modes. This is the most expressive, most geometry-agnostic parameterization of G on the table. It's also the one I can't afford: the quadrature uses every mesh point as a node, so the QK^T matrix and the softmax over it are O(N^2) in both compute and memory. At N = 32k, that score matrix alone is 32000^2 ≈ 1.0×10^9 entries per head; in fp32 that's ~4 GB for one head of one layer, and with 8 heads and 8 layers the attention maps don't come close to fitting on a 40 GB card even before activations and gradients. Quadratic attention over the raw mesh is simply off the table at these N. Wall.

The obvious patch is linear attention — replace softmax(QK^T)V with the Galerkin-style Q(K^T V), which reassociates the matmuls so the cost drops to O(N). OFformer, GNOT, the Galerkin Transformer all do this, and it does fix the complexity. But stare at what it leaves untouched: the attention, linear or not, is still being computed *over the N mesh points themselves*. And there's a real modeling problem hiding under the complexity problem. When I ask attention to learn reliable correlations among tens of thousands of individual points, the informative physical relationships get diluted in a sea of low-level point-to-point interactions; the cross-attention with mesh-point queries that OFformer uses is hard to even optimize at 32k points — the loss just jitters. So linear attention buys me speed and hands me back a harder learning problem. Making the quadrature cheaper isn't enough; the quadrature nodes are wrong.

Let me sit with that, because I think the nodes are the whole point. Why am I quadraturing over mesh points at all? Mesh points are an *artifact of discretization* — a finite, arbitrary sampling of an underlying continuous physics. The physics doesn't live at the points; it lives in the states. And here's the thing I keep noticing when I look at these fields: spatially distant points are often in the *same* physical state, and spatially adjacent points can be in *different* states. On a driving car the windshield, the license plate, and the headlights are all in the front region — that's one physical regime, the one that governs drag — even though they're scattered across the surface. Patchify in vision Transformers groups a square block of pixels into one token to cut the token count, and that's the right *instinct* — quadrature over groups, not pixels — but a square patch is a geometric block: it assumes a grid, and worse, it groups by location, so it would split that front region across many patches and merge the front bumper with whatever's geometrically next to it. A fixed geometric grouping can't represent a state that's spread non-locally. Domain decomposition in classical FEM has the same limitation — it carves the mesh into spatially-local computation areas by hand, which can never put the windshield and the license plate in the same piece.

So the move I want isn't "group points by where they are." It's "group points by *what physical state they're in*, learned from data, with the groups free to be any shape and to span the whole domain." If I had, say, M such groups — call them slices — then I could encode each slice into a single token summarizing its physical state, run attention among the M tokens, and broadcast the result back to the points. The attention is then O(M^2), the encoding and broadcast are O(NM), and with M a small constant the whole thing is linear in N. The quadrature would happen over *learned physical states* instead of mesh points — far fewer nodes, and nodes that actually mean something.

Now I have to make "ascribe each point to a slice" concrete and, crucially, differentiable, because the slicing has to be learned end-to-end against the final loss. A hard assignment — argmax each point to one slice — is the intuitive picture but it kills the gradient and freezes the grouping. So make it soft. For each point's feature x_i ∈ R^{1×C}, project to M numbers and normalize into weights. The question is which axis to normalize over, and it matters more than it looks. If I take softmax over the *points* (so each slice's weights over the N points sum to one), I've defined slices as distributions over points but a single point could be split arbitrarily and there's no clean "this point belongs here" reading. What I want is the opposite: each point gets a distribution *over the M slices*,

  w_i = Softmax(Project(x_i)) ∈ R^{1×M},   Σ_{j=1}^M w_{i,j} = 1,

so w_{i,j} is the degree to which point i belongs to slice j, and the per-point weights form a partition of unity across slices. Let me confirm that partition-of-unity actually holds the way I want it to: I push 5 random points (M = 6, a small linear projection, softmax over the slice axis) and sum each point's weights over the 6 slices — every row sums to 1.0, exactly, which is the property I'll lean on later when the slice mass has to integrate to one for the change of variables to close. Project is just a pointwise linear map C→M, which is the one thing that adapts to any geometry without assuming a grid. And the softmax-over-slices choice does a second job I want: softmax sharpens. If I'd used a plain normalization the model could sit at a lazy near-uniform assignment, every point smeared equally across all slices, and then every slice would encode essentially the same domain-wide average — the slices would collapse and carry no distinguishing information. The exponential in softmax pushes the assignment to be low-entropy, peaky, so a point commits mostly to one or a few slices and the slices are pressured to specialize into distinct physical states. Points with similar features get similar w, so they land in the same slice; that's the grouping, and it's learned. I can even reach for a knob here: divide the projection logits by a temperature τ before the slice softmax. τ below one sharpens further, τ above one softens; making τ a learnable parameter (and one per head, since I'm about to go multi-head) lets the model choose how committed each head's slicing should be, and I'll initialize it on the sharp side, around 0.5. This temperature belongs to the slice assignment; the token-to-token attention will still use the usual dim_head^{-1/2} scale.

Given the soft assignment, how do I turn a slice into one token? A slice is "the points, weighted by how much they belong." The summarizing token should be the slice's *representative physical state*, which is the weighted average of its members' features, not the weighted sum — because a slice that happens to own many points shouldn't get an artificially large token just for being big. So normalize by the slice's total mass:

  z_j = ( Σ_{i=1}^N w_{i,j} x_i ) / ( Σ_{i=1}^N w_{i,j} ),   z_j ∈ R^{1×C}.

Dividing by Σ_i w_{i,j} makes z_j a genuine weighted mean, scale-free in the number of points the slice gathers. (In code I'll floor that denominator with a tiny ε so an empty slice doesn't blow up; nothing conceptual, just don't divide by zero.) Now I have M physics-aware tokens z = {z_j}, each a compressed description of one learned physical state.

Among these M tokens I run ordinary attention — and here I deliberately keep the *softmax* attention, because M is small (32 to 64), so M^2 is trivial, and softmax attention is the most expressive learnable-kernel operator, the very thing I derived above. No need to cripple it with a linear approximation when there are only a few dozen tokens:

  q, k, v = Linear(z),   z' = Softmax(q k^T / sqrt(d)) v,   q,k,v,z' ∈ R^{M×d}.

This is where the global physical correlations get modeled: every physical state can attend to every other, so the front-region state and the wake state interact directly, in O(M^2 d) per head. The 1/sqrt(d) is the usual scaling so the dot products don't saturate the softmax as the per-head channel width grows.

Then I have to get back to the points — the loss lives on the mesh. I have a transited token z'_j per slice; I want a feature x'_i per point. The natural inverse of "average points into a slice with weights w" is "broadcast the slice back to its points with the same weights":

  x'_i = Σ_{j=1}^M w_{i,j} z'_j,   1 ≤ i ≤ N.

I want to dwell on the choice to reuse *the same* w for the broadcast that I used for the encoding, because it's tempting to learn a separate set of de-slice weights and that would be a mistake. Tying them makes slice-then-deslice a single change of variables: I move from the point domain into the slice domain, do the work there, and come back through the same map. If the two maps were independent, the round trip would be two unrelated projections glued together instead of one operator transported through a coordinate map. So: slice with w, attend, deslice with w. Call the sandwich Physics-Attn(x) = x'.

Let me count the cost to be sure I actually won. Slicing and token encoding: each of N points touches M slices over C channels, O(NMC). Attention among tokens: O(M^2 C). Deslicing: O(NMC) again. Total O(NMC + M^2 C). M is a fixed constant with M ≪ N, so this is linear in N — the thing I needed — while the attention itself, the part doing the global correlation, is over the M meaningful nodes, not the N noisy ones. Both problems, the complexity and the modeling, fall to the same move: change the quadrature nodes from mesh points to learned physical states.

One head is one decomposition of the domain into M states. But the domain has more than one meaningful decomposition at once — pressure regimes, velocity regimes, geometric regimes — so I do this multi-head, exactly as canonical attention does: split the C channels into several subspaces and run an independent slicing-attention-deslicing in each, then concatenate. Each head learns its *own* slice weights and so its own physical-state grouping, and different heads end up capturing different facets of the physics. Concretely, with `heads` heads and `dim_head = C / heads`, the per-head projections produce features of width dim_head, the slice projection maps dim_head→M, and the per-head outputs are concatenated back to width C and passed through an output linear.

Now I drop Physics-Attn into the Transformer skeleton in place of full attention and keep everything else standard, because the residual/pre-norm structure is not what I'm reinventing. Layer l is

  x̂^l = Physics-Attn(LayerNorm(x^{l-1})) + x^{l-1},
  x^l  = FeedForward(LayerNorm(x̂^l)) + x̂^l,

stacked L times, with the feed-forward sublayer the usual ~4×-wide MLP. The input x^0 is a linear embedding of the concatenated geometry and observed quantities, x^0 = Linear(Concat(g, u)); the output is a linear projection of x^L to the target quantities. The whole model is just the canonical Transformer with its one expensive, geometry-blind sublayer swapped for the cheap, geometry-general one.

Before I trust this I want to know whether it is more than an ad-hoc trick. The question I can actually pose is whether the slice-attention-deslice sandwich is *still* the integral operator G on Ω — just evaluated through slice coordinates instead of mesh coordinates — or whether the grouping step quietly changes the operator into something else. So I'll try to push G itself through a change of variables into the slice domain and see what discrete formula falls out the other end, and then compare that formula, symbol for symbol, against the sandwich I just built. If they match, the construction is the same operator; if they don't, I've smuggled in an assumption I should name.

I already have one half: canonical attention is the Monte-Carlo discretization of G with the mesh points as quadrature nodes (the lemma I re-derived above), with the query fixed and the denominator summing over key/value nodes. I need a bridge from the mesh domain Ω to a slice-coordinate domain Ω_s and a value function defined on that domain. The finite M learned tokens are the quadrature nodes used later; for the change-of-variables argument I first let Ω_s be a countable domain that can correspond one-to-one with countable Ω. Fix an integer K ≥ 1. March through the points i = 1, 2, 3, …; for point i, look at the block of K slice coordinates indexed by (⌊(i−1)/K⌋·K, (⌊(i−1)/K⌋+1)·K] and send i to the slice coordinate in that block with the largest slice weight w_{g_i, s_j} that has not already been claimed:

  g(g_i) = argmax_{s_j in this block, not yet projected} w_{g_i, s_j}.

Because each point claims a distinct slice coordinate in its own block, this is injective, and by taking Ω_s to be the claimed countable slice coordinates it is bijective, so Ω ≅ Ω_s. If the slice-weight function w_{·,·} is smooth on continuous extensions of both domains, I can treat this correspondence as a diffeomorphic projection g. This is the map I need for a change of variables; the finite slice tokens will be the learned quadrature nodes that approximate integrals on Ω_s.

Define the value function on the slice domain to be the continuous version of my token encoding:

  u_s(ξ_s) = ( ∫_Ω w_{ξ, ξ_s} u(ξ) dξ ) / ( ∫_Ω w_{ξ, ξ_s} dξ ),

which is exactly z_j in the limit — a slice token is the w-weighted average of the field over the points assigned to that slice, divided by the slice mass. Now push G through the change of variables. Start from G(u)(g) = ∫_Ω κ(g, ξ) u(ξ) dξ and substitute ξ = g^{-1}(ξ_s):

  G(u)(g) = ∫_{Ω_s} κ_ms(g, ξ_s) u_s(ξ_s) dg^{-1}(ξ_s)
          = ∫_{Ω_s} κ_ms(g, ξ_s) u_s(ξ_s) |det(∇_{ξ_s} g^{-1}(ξ_s))| dξ_s,

where κ_ms is the induced kernel between a mesh point and a slice. In the discrete implementation the slice domain is a set of token slots with counting measure, and relabeling the slots cannot change the operator. So for the implemented formula I take the permutation-invariant, measure-preserving case and set

  |det(∇_{ξ_s} g^{-1}(ξ_s))| = 1.

If that determinant were not one, it would have to be carried in the measure or absorbed into the induced kernel; the code corresponds to the determinant-one simplification. Now the mesh-to-slice kernel κ_ms should itself be expressed through the slice-to-slice kernel κ_ss weighted by the slice weights, because to get from a mesh point g to the influence of slice ξ_s I first relate g to output/query slices ξ_s' through the deslice weight w_{g, ξ_s'}, and then let those slices interact with input/key slices through κ_ss:

  κ_ms(g, ξ_s) = ( ∫_{Ω_s} w_{g, ξ_s'} κ_ss(ξ_s', ξ_s) dξ_s' ) / ( ∫_{Ω_s} w_{g, ξ_s'} dξ_s' ).

Substituting, taking the determinant-one simplification, and using the slice-softmax partition of unity over the M slice axis, whose continuum analogue is ∫_{Ω_s} w_{g, ξ_s'} dξ_s' = 1 for each fixed mesh point g,

  G(u)(g) = ∫_{Ω_s}∫_{Ω_s} w_{g, ξ_s'} κ_ss(ξ_s', ξ_s) u_s(ξ_s) dξ_s dξ_s',

and I can read the three pieces straight off: w_{g, ξ_s'} is the deslice weight for output/query slice ξ_s', the inner integral over ξ_s is attention from that query slice to key/value slices, and u_s is the slice token itself. Finally discretize the slice-domain integrals by Monte-Carlo over the M learned slice nodes. With z_t ≈ u_s(ξ_{s,t}), q_j = W_q z_j, k_t = W_k z_t, v_t = W_v z_t, and row-normalized attention over key/value slice t for each fixed query slice j,

  G(u)(g_i) ≈ Σ_{j=1}^M w_{i,j} Σ_{t=1}^M [ exp(q_j k_t^T / sqrt(d)) / Σ_{p=1}^M exp(q_j k_p^T / sqrt(d)) ] v_t,

where the outer j-sum is the deslice step, the inner t-sum is the token-attention row for query slice j, and the p-sum normalizes only over key/value slices for that fixed j. Now I do the comparison I set up: read this derived formula against the sandwich I built earlier, factor by factor. The sandwich was deslice(out_token)_i = Σ_j w_{i,j} (out_token)_j, with out_token = Softmax(qk^T/√d) v over the M tokens, and token z_t = mass-normalized mean of the points in slice t. The derived formula's outer factor w_{i,j} is the same w from the slice softmax — same symbol, same partition-of-unity — and it appears in the deslice position exactly as in the sandwich. Its inner bracket Σ_t [exp(q_j k_t^T/√d)/Σ_p exp(q_j k_p^T/√d)] v_t is the j-th row of softmax attention over the M tokens — character-for-character the (Softmax(qk^T/√d) v)_j of the sandwich. And the v_t, q_j, k_t are linear maps of z_t = u_s(ξ_{s,t}), the slice token, which is the token-encoding step. So all three factors line up; there is no extra term in the derived formula that the sandwich lacks, and no term in the sandwich missing from the derivation. They are the same operator.

But the match only holds *because of two simplifications I made along the way*, and honesty means naming them as the price rather than pretending the derivation was unconditional. The first is |det ∇g^{-1}| = 1: I set the Jacobian to one when changing variables, justified by the slice slots being a permutation-invariant counting measure. If the learned slicing were genuinely non-measure-preserving, that determinant would survive and the implemented einsum would be missing a per-slice density factor — the code does *not* carry one, so the code is committing to the determinant-one case, not deriving it. The second is ∫_{Ω_s} w_{g,ξ_s'} dξ_s' = 1, the continuum partition-of-unity, which is exactly the per-point softmax-over-slices sum I checked numerically above (every row summed to 1.0); that one I've actually verified holds in the discrete implementation, so it's not an assumption but a property. So the derivation is conditionally exact: granting the determinant-one simplification, Physics-Attn is the integral operator G evaluated by Monte-Carlo over the slice domain, with the tying of slice and deslice weights *forced* (it is the one map g used both ways, not a free design choice), and the slice temperature and dim_head^{-1/2} token scale riding along as implementation details inside w and the attention. That is weaker than "provably G" but it is what I can actually stand behind, and it's enough to tell me the sandwich isn't an arbitrary trick.

Now let me make the implementation honest, because there are a couple of refinements that matter and that the clean equations gloss over. First, in the equations I both *decide* a point's slice assignment and *aggregate* the point's content from the same feature x_i. But the feature that's best for *deciding which state a point is in* isn't necessarily the feature I want to *carry into the token*. So I'll project x twice with separate linear maps: one stream, call it x_mid, feeds the slice-weight projection (the "where does this point belong" signal), and a second stream, fx_mid, supplies the values that get averaged into the tokens (the "what does this point contribute" signal). Both are computed per head at width dim_head. This decoupling is cheap and gives the model room to learn an assignment criterion distinct from the carried content.

Second, the slice-projection layer maps dim_head → M, and if I initialize it carelessly the M slice logits start out correlated and the slices are slow to differentiate. Initializing that one linear layer with an orthogonal weight matrix decorrelates the slice directions from the start, so the M slices begin as distinct as possible and specialize faster. Everything else gets the standard truncated-normal init; LayerNorms start at unit weight, zero bias.

Let me also pin the per-head plumbing exactly, since the shapes are where bugs hide. With input x of shape (B, N, C): project to fx_mid and x_mid, each reshaped to (B, heads, N, dim_head). Slice weights are softmax over the slice axis of in_project_slice(x_mid)/τ, shape (B, heads, N, M). The slice norm is their sum over the N axis, (B, heads, M). The slice tokens are einsum("bhnc,bhng->bhgc", fx_mid, slice_weights) divided by (slice_norm + 1e-5), shape (B, heads, M, dim_head). Attention: q,k,v from per-head linear maps dim_head→dim_head with no bias, dots = q k^T · dim_head^{-1/2}, softmax over the token axis, optional dropout, out = attn·v, shape (B, heads, M, dim_head). Deslice: einsum("bhgc,bhng->bhnc", out_tokens, slice_weights), then merge heads back to (B, N, heads·dim_head) and pass through the output linear (inner_dim→C) with dropout. That's the whole sublayer.

A couple of choices I should justify rather than assert. The number of slices M is the one genuinely new hyperparameter, so I should understand its limits by reasoning about the extremes — and the M = 1 extreme I can actually run, so I do, rather than just claiming what it degenerates to. I set slice_num = 1 and trace the sublayer on 5 random points. The slice softmax over a single slice axis returns weights that are all 1.0 (softmax of a length-1 vector is 1, as it must be), so the partition-of-unity is the trivial one. The mass-normalized token then equals the plain mean of the point features over N — I compute both and they agree to 1e-5. The "attention" among one token: the score matrix is 1×1, softmax of it is exactly 1.0, so out = attn·v = v, no mixing at all. And the deslice broadcasts that single token back with weight 1 to every point, so I checked that points 0, 1, and 4 all come out with byte-identical features. So M = 1 *is* global pooling-and-broadcast — every point gets the same domain-average feature — and it throws away all physical correlations, since there is only one state and nothing for attention to relate it to. That confirms the degenerate end concretely. At the other end, push M toward the number of points: the physics domain gets fragmented into too many slivers, each slice owns almost no points, the tokens become noisy and numerous, and I've drifted back toward attention-over-points with all its distraction. The useful regime is in between, a few dozen to a couple hundred. How it should track the width C is worth getting right rather than hand-waving, so I count parameters for the two configs I have in mind. With heads = 8, the Physics-Attn linears (two C→C input projections, the dim_head→M slice projection, three dim_head→dim_head q/k/v, and the C→C output) come to about 51k parameters at (C=128, M=64) and about 202k at (C=256, M=32) — so it is *not* true that the two configs have matched parameter counts; the wider model is roughly 4× heavier, dominated by the C×C input/output projections, and M barely moves that. What the two configs *do* match is the per-point slicing work: each point touches M slices over the per-point feature, and M·C = 64·128 = 32·256 = 8192 either way, so the O(NMC) term — the part that scales with the big N — is held fixed across the two. That is the quantity I actually want comparable when N is 32k, so I'll pick M = 64 for C = 128 and M = 32 for C = 256 to hold the dominant per-point cost constant, and accept that the smaller-N parameter count differs. Heads I keep at the canonical 8, layers at L = 8, and the feed-forward at the usual 4× width, because none of those is where the contribution lives and matching the standard Transformer keeps the comparison fair.

One more on Project. For a genuinely unstructured mesh — a car, an airfoil point cloud — Project must be point-wise, so it's a plain linear layer; that's the geometry-general default and the one the design tasks need. But when the mesh *is* structured I can do better: instantiate the two point projections as local convolutions with kernel size 3, so a point's slice assignment can see its immediate neighbors, which is sensible on a grid where locality is meaningful. Same slicing idea, the projection just adapts to whether the discretization has exploitable local structure. The attention-among-tokens and the deslice are identical in both cases.

So let me write the sublayer and the block the way I'd actually ship them, on the unstructured path that the irregular-mesh tasks use, with the two-stream projection, the learnable per-head temperature, the orthogonally-initialized slice projection, and the mass-normalized tokens.

```python
import torch
import torch.nn as nn
from einops import rearrange


class Physics_Attention_Irregular_Mesh(nn.Module):
    """Physics-Attention for irregular meshes: soft-slice points into M physical states,
    attend among the M physics-aware tokens, then broadcast back. O(NMC + M^2 C)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5                         # 1/sqrt(dim_head) for the token attention
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # learnable, per-head sharpness for the slice assignment (init on the sharp side)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # two separate projections of x: one to DECIDE the slice (x_mid), one for CONTENT (fx_mid)
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)   # dim_head -> M slice logits
        nn.init.orthogonal_(self.in_project_slice.weight)        # decorrelate slices at init

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape

        # (1) Slice: soft-assign every point to M slices, encode each slice into one token
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()                # B H N dim_head  (content stream)
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()                # B H N dim_head  (assignment stream)
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature)  # B H N M, softmax over the M axis
        slice_norm = slice_weights.sum(2)                     # B H M  (mass of each slice)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)   # weighted sum
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None]
                                     .repeat(1, 1, 1, self.dim_head))           # -> weighted mean

        # (2) Attention among the M physics-aware tokens (global physical correlations, O(M^2))
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.softmax(dots))
        out_slice_token = torch.matmul(attn, v)               # B H M dim_head

        # (3) Deslice: broadcast tokens back to points with the SAME slice weights
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Transolver_block(nn.Module):
    """Canonical pre-norm residual block with full attention replaced by Physics-Attention."""

    def __init__(self, num_heads, hidden_dim, dropout=0., act='gelu', mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if last_layer:                                        # final block carries the read-out head
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx                    # x + PhysicsAttn(LN(x))
        fx = self.mlp(self.ln_2(fx)) + fx                     # x + FFN(LN(x))
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx
```

And the model wraps the stack with the embedding and the read-out, on the unstructured path the design tasks take — embed the concatenated coordinates and input features into width C, add a learned constant bias, run the L blocks (the last one projects to the output channels), and the forward signature ignores the graph connectivity because Physics-Attention needs no mesh graph at all:

```python
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__name__ = 'Transolver'
        self.args = args
        # lift (coordinates ++ input quantities) to width n_hidden
        self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                              n_layers=0, res=False, act=args.act)
        self.blocks = nn.ModuleList([
            Transolver_block(num_heads=args.n_heads, hidden_dim=args.n_hidden, dropout=args.dropout,
                             act=args.act, mlp_ratio=args.mlp_ratio, out_dim=args.out_dim,
                             slice_num=args.slice_num, last_layer=(i == args.n_layers - 1))
            for i in range(args.n_layers)])
        # small learned constant bias added to every point's embedding
        self.placeholder = nn.Parameter((1 / args.n_hidden) * torch.rand(args.n_hidden))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx, T=None, geo=None):
        # x: (1, N, space_dim) coordinates; fx: (1, N, fun_dim) features; geo (mesh graph) unused
        if fx is not None:
            fx = torch.cat((x, fx), dim=-1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx
```

Let me trace the whole causal chain once more to be sure it holds together. I needed a fast neural operator on huge irregular meshes that still models global physical correlations. The integral-operator view says each layer should learn a non-local kernel, and attention is exactly that kernel — a learnable Monte-Carlo quadrature of the integral with row-normalized softmax over key/value nodes — but with the mesh points as quadrature nodes it costs O(N^2) and, even made linear, it dilutes the physics across tens of thousands of meaningless point-to-point relations. The fix wasn't a cheaper quadrature; it was better nodes: group points by learned physical state rather than by location, because the physics lives in states, not in the discretization, and states span the domain non-locally in a way square patches and FEM subdomains can't capture. So I soft-assign each point across M slices with a per-point softmax over the slice axis (partition of unity, and the exponential sharpens so slices specialize), encode each slice into a mass-normalized weighted-mean token, run full softmax attention among the M tokens with the standard dim_head^{-1/2} scale, and broadcast back through the *same* weights — and that tying is exactly what makes the slice-attention-deslice sandwich a single change of variables that, when I push the integral operator through the slice-domain map and grant the determinant-one (measure-preserving) simplification, reproduces the sandwich factor for factor, as I checked by reading the derived discrete formula against the implemented one. Dropped into the canonical pre-norm Transformer in place of full attention, with two-stream point projections, a learnable per-head temperature for slicing, an orthogonally-initialized slice projection, and multi-head slicing so different heads learn different decompositions, the whole operator is O(NMC + M^2 C) — linear in N — and needs no mesh graph at all.
