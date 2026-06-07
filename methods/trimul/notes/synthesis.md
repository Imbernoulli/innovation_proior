# Synthesis — triangular multiplicative update + triangle attention

## The problem (in-frame, pre-method)
The pair representation z_ij ∈ ℝ^{c_z} (c_z=128) is a learned per-pair embedding over residues
i,j ∈ {1..N_res}. It is meant to encode the *relation* between residues i and j — ultimately the
geometry: distance, relative orientation. It is read out into a distogram (predicted distance
distribution) and feeds the structure module. So z_ij is, semantically, an *edge feature* on the
complete residue graph.

Pain point: if I update each edge z_ij independently (e.g. plain per-pair MLP, or axial
row/column attention that mixes along one residue axis at a time), nothing enforces that the
collection of pairwise relations is *mutually consistent in 3D*. A set of pairwise distances
{d_ij} is realizable as points in 3D only if it satisfies geometric constraints — the most basic
being the triangle inequality on every triple: d_ij ≤ d_ik + d_kj. An edge (i,j) is constrained
by the *other two edges of every triangle it sits in*: (i,k) and (k,j) for every third node k.
Independent edge updates can't see those constraints.

## The known math anchor
Triangle inequality / metric realizability. For points in 3D, all triples must satisfy
d_ij ≤ d_ik + d_kj (and the Cayley–Menger conditions more fully). The narrator invokes this as a
known geometric fact, not a measurement.

## The lineage (ancestors)
- **Pair / edge features on a residue graph** — treat residues as nodes, pairs as edges; the pair
  rep is initialized from relative position encoding (Alg 4: clipped relative sequence distance,
  one-hot to 65 bins [-32..32], linear) plus an outer sum of single embeddings (Alg 3), and updated
  from the MSA via OuterProductMean (Alg 10: m_si→a,b; o_ij=flatten(mean_s(a_si⊗b_sj)); z=Linear).
- **Message passing on graphs (GNN)** — an edge/node is updated by aggregating over neighbors. Here
  the natural aggregation for an *edge* (i,j) is over a third node k, combining the two incident
  edges (i,k),(k,j) — a "triangle" message. This is the GNN-flavored read of the operation.
- **Axial / row-column attention (Ho et al. 2019, Axial Attention in Multidimensional Transformers;
  Child et al. 2019)** — factorize attention over a 2D grid into attention along rows then columns,
  O(N) memory per axis instead of O(N²) over the full N²×N² interaction. Adapted to MSAs by the MSA
  Transformer (Rao et al. 2021): tied row attention + column attention over the (sequences × positions)
  matrix; row attention map ≈ contact prediction. Axial attention along a single residue axis mixes
  z_ik for fixed i (all k) OR z_kj for fixed j — but **never couples the two incident edges of a
  triangle at once**, so it can't impose triangle consistency. That is the precise gap.
- **Gated attention / gating** — sigmoid gates as a learned selective filter (as in the MSA stack's
  gated self-attention, Alg 7/8: g = sigmoid(Linear(·)), o = g ⊙ Σ a v).

## The derivation target
Each edge (i,j) should be updated by routing information through every third node k, combining the
two other edges of the triangle {i,j,k}. The cheapest symmetric way to "combine two edge vectors
and sum over the apex k" is an elementwise (Hadamard) product summed over k — an outer-product-style
contraction:
  update_ij = Σ_k a_ik ⊙ b_jk        ("outgoing" — both edges leave the shared apex... see note)
There are two non-equivalent ways to pick which two edges share node k:
  - **Outgoing** (Alg 11): Σ_k a_ik ⊙ b_jk  — code einsum 'ikc,jkc->ijc'. Edges (i,k),(j,k): both
    have k as their *second* index; i and j are the "starting" nodes that share their target k.
  - **Incoming** (Alg 12): Σ_k a_ki ⊙ b_kj  — code einsum 'kjc,kic->ijc'. Edges (k,i),(k,j): k is the
    *first* index; i and j share their source k.
Because z is not assumed symmetric (z_ij ≠ z_ji in general), these two contractions see different
information, so both are applied (consecutive Evoformer layers), giving symmetric coverage of the
triangle's orientations.

## Exact algorithm (verified against SI Alg 11/12 and DeepMind modules.py)
TriangleMultiplicationOutgoing (c=128 hidden):
  1. z_ij ← LayerNorm(z_ij)
  2. a_ij = sigmoid(Linear_g(z_ij)) ⊙ Linear_a(z_ij);  b_ij = sigmoid(Linear_g'(z_ij)) ⊙ Linear_b(z_ij)
     (each projection ℝ^{c_z}→ℝ^c; gate is a separate sigmoid(Linear) of same z; a,b ∈ ℝ^c)
  3. g_ij = sigmoid(Linear(z_ij))   ∈ ℝ^{c_z}   (output gate)
  4. z̃_ij = g_ij ⊙ Linear( LayerNorm( Σ_k a_ik ⊙ b_jk ) )
Incoming: identical except step 4 uses Σ_k a_ki ⊙ b_kj.
Notes on shape/cost: c=128 hidden, the einsum Σ_k is O(N³ c). LayerNorm appears twice (input + on the
summed result, "center_layer_norm") — input LN stabilizes the projections; output/center LN stabilizes
the magnitude after the sum over N terms (variance grows with N). Two gates: per-edge input gates on
a and b select which edges contribute; output gate g_ij lets edge (i,j) modulate how much update it
takes. fp16 path divides a,b by their std before matmul to avoid overflow (engineering detail).

## Triangle self-attention (Alg 13/14) — the attention sibling
The multiplicative update mixes *all* k uniformly (weighted only by learned gates, not by content
similarity). Attention lets edge (i,j) *choose* which k matter by query–key similarity AND still feel
the third edge:
TriangleAttentionStartingNode (c=32, N_head=4), updates edge ij from edges ik sharing start node i:
  q,k,v = LinearNoBias(z_ij);  b^h_jk = LinearNoBias(z_jk)  (scalar bias per head from the *third* edge jk)
  a^h_ijk = softmax_k( (1/√c) q^h_ij · k^h_ik + b^h_jk )
  o^h_ij = g^h_ij ⊙ Σ_k a^h_ijk v^h_ik ;  z̃_ij = Linear(concat_h o^h_ij)
So: query = central edge ij, key/value = left edge ik, and the affinity is *biased by the right edge
jk* — every triangle edge participates. EndingNode (Alg 14) is the transpose (operates on columns /
edges around ending node). In the Evoformer this is implemented as row attention on z for starting,
and "not starting" → transpose(-2,-3) then row attention for ending (OpenFold).

## Where it sits (Evoformer block, Alg 6, per block, N_block=48)
MSA stack: RowAttnWithPairBias → ColumnAttn → Transition.
Communication: z += OuterProductMean(m).
Pair stack (residual + dropout each):
  z += TriMulOutgoing(z)
  z += TriMulIncoming(z)
  z += TriAttnStartingNode(z)
  z += TriAttnEndingNode(z)
  z += PairTransition(z)
Order: multiplicative updates first (cheap, symmetric, impose consistency), then attention (content
routing), then a transition MLP (4× expand). Pair bias flows *into* MSA row attention (Alg 7 line 3:
b = LinearNoBias(LayerNorm(z))) — the pair stack informs the MSA stack, closing the loop.

## Design-decision → why table
- **Route through a third node k at all** → independent edge updates can't enforce triangle
  inequality / 3D realizability; the constraint on (i,j) lives in the *other two* edges of each triangle.
- **Multiplicative (Hadamard ⊙) + sum over k, not attention first** → cheapest, most symmetric way to
  "combine two edges and aggregate over the apex"; an outer-product mix. Developed as a frugal,
  symmetric alternative to attention.
- **Two directions (outgoing/incoming)** → z asymmetric; the two contractions ('ikc,jkc' vs 'kjc,kic')
  see different triangle orientations; using both restores symmetric coverage.
- **Per-edge input gates on a,b** → learned soft selection of which edges k carry usable signal
  (soft, differentiable stand-in for the hard "for all k" constraint).
- **Output gate g_ij** → lets the target edge control how much of the triangle-aggregated update it
  absorbs (residual-friendly, avoids overwriting confident edges).
- **Input LayerNorm + center LayerNorm** → stabilize projections, and renormalize after a sum of N
  products whose variance scales with N.
- **Add triangle attention on top** → multiplicative update weights all k by gates only (content-blind);
  attention lets ij pick relevant k by q·k similarity while the third-edge bias b_jk keeps the triangle
  closed. Multiplicative alone is cheaper/symmetric; the combination is more accurate (kept both).
- **Bias from the third edge jk in attention** (not plain axial attention) → plain row attention on z
  would couple only (i,j) and (i,k); adding the b_jk logit injects the missing edge so the decision is
  made over the *whole* triangle, not two of its sides.
- **c=128 hidden for mult, c=32 / 4 heads for attn** → mult update is the O(N³) workhorse and gets the
  wide hidden; attention is the more expensive refinement and runs narrower.

## Canonical code grounded in
- OpenFold openfold/model/triangular_multiplicative_update.py (forward path lines ~526-558) and
  triangular_attention.py (lines ~93-162). DeepMind alphafold/model/modules.py TriangleMultiplication
  (lines 1390-1468) with explicit einsum 'ikc,jkc->ijc' (out) / 'kjc,kic->ijc' (in).
