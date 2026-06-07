# GAT — Synthesis notes (Phase 1.5)

## Pain point being solved
Want a neural operator on arbitrary graphs (citation nets, PPI, meshes, social) that:
- assigns *different* importances to different neighbors (capacity),
- does NOT depend on knowing the global graph structure / Laplacian eigenbasis upfront,
- works **inductively** (generalize to graphs unseen at train time),
- handles variable-degree neighborhoods,
- is computationally cheap (no eigendecomposition, no matrix inversion), parallelizable across edges/nodes.

## The precise first-principles object
A per-node feature update h_i' = aggregate over N_i of transformed neighbor features. The central difficulty: how to weight neighbors *and* keep the operator (a) shared across all nodes/graphs (weight sharing like CNN), (b) independent of fixed graph structure, (c) able to handle arbitrary neighborhood sizes.

## Load-bearing ancestors (lineage)

### Spectral line
- **Bruna et al. 2014 (Spectral networks):** graph convolution defined in Fourier domain. Graph Laplacian L = D - A (or normalized L = I - D^{-1/2} A D^{-1/2}); eigendecomp L = U Λ U^T. Graph Fourier transform of signal x is U^T x. Spectral conv: g_θ ⋆ x = U g_θ(Λ) U^T x, where g_θ = diag of free params per eigenvalue. Limitations: (1) eigendecomp is O(N^3), forward is O(N^2); (2) filters are O(N) free params, not localized in space; (3) filters defined per-eigenvalue → tied to the eigenbasis U → tied to THIS graph.
- **Henaff et al. 2015:** smooth spectral multipliers (spline parameterization) → spatial localization. Still eigenbasis-dependent.
- **Defferrard et al. 2016 (ChebNet):** approximate g_θ(Λ) by Chebyshev polynomial of order K: g_θ(Λ) ≈ Σ_{k=0}^K θ_k T_k(Λ~), Λ~ = 2Λ/λ_max - I. Because T_k(L) is a degree-k polynomial in L, the filter is K-localized (touches only K-hop neighbors) and needs no eigendecomposition (just sparse matmuls with L). O(K|E|) cost. But the polynomial coefficients still implicitly depend on the spectrum/structure; still fundamentally spectral.
- **Kipf & Welling 2017 (GCN):** set K=1, λ_max≈2, single shared θ. Get g_θ ⋆ x ≈ θ(I + D^{-1/2} A D^{-1/2}) x. Eigenvalues of (I + D^{-1/2}AD^{-1/2}) in [0,2] → repeated stacking causes numerical instability/exploding-vanishing. **Renormalization trick:** I + D^{-1/2}AD^{-1/2} → D~^{-1/2} A~ D~^{-1/2} with A~ = A + I (self-loops), D~ = degree of A~. Layer rule: H^{(l+1)} = σ( D~^{-1/2} A~ D~^{-1/2} H^{(l)} W^{(l)} ). 
  - KEY LIMITATION for the narrative: the propagation matrix D~^{-1/2} A~ D~^{-1/2} is a **fixed** function of the whole graph's adjacency. The weight a node gives a neighbor is *structurally fixed* = 1/sqrt(d_i d_j); no learned per-neighbor importance. And because the operator IS the normalized adjacency of a specific graph, it's intrinsically transductive — a model trained on graph G's normalized-adjacency operator is not defined for a different graph G' (different N, different structure), so it can't transfer. This is the "filters depend on Laplacian eigenbasis → depend on graph structure → can't apply to a new graph" point.

### Non-spectral / inductive line
- **Duvenaud et al. 2015:** molecular fingerprints; learn a separate weight matrix per node degree → doesn't share weights across degrees, awkward.
- **Atwood & Towsley 2016 (DCNN):** powers of transition matrix define neighborhoods; weights per hop/channel.
- **Niepert et al. 2016 (PATCHY-SAN):** extract & normalize fixed-size ordered neighborhoods → needs a node ordering / graph normalization, CNN on it.
- **Monti et al. 2016 (MoNet):** unifying spatial framework — pseudo-coordinates u(x,y) over node pairs, weight functions w_j(u) (Gaussian kernels). GCN/DCNN are special cases. GAT is also a special case (with u = feature concat and w = softmax(MLP)). Important: MoNet's pseudo-coords are usually *structural* (degrees) → assume known structure.
- **Hamilton et al. 2017 (GraphSAGE):** the key inductive prior. Learns aggregator functions, not per-node embeddings → inductive. Algorithm: h_v^k = σ(W · CONCAT(h_v^{k-1}, AGG_k({h_u^{k-1}, u ∈ sampled N(v)}))). Aggregators: mean, LSTM, max-pool.
  - Limitation 1: **fixed-size neighborhood sampling** (sample S neighbors) → for a fixed compute footprint, it does NOT see the whole neighborhood at inference.
  - Limitation 2: mean/GCN aggregators treat all sampled neighbors **equally** (no learned per-neighbor importance).
  - Limitation 3: best results came from **LSTM aggregator**, but a neighbor set is unordered → must feed random permutations; assumes/forces an ordering that isn't really there.

### Attention line
- **Bahdanau et al. 2015 (additive attention):** in NMT, align decoder state to encoder states. Score e_ij = a(s_i, h_j) via a small feedforward net (single hidden layer, tanh), then α = softmax over j. Handles variable-length inputs, focuses on relevant parts, interpretable alignments. GAT's attention mechanism "closely follows" this — single-layer feedforward scoring. (GAT uses LeakyReLU instead of tanh.)
- **Vaswani et al. 2017 (Transformer / self-attention):** self-attention = a sequence attends over itself; "attention is all you need" — sufficient alone for SOTA. Multi-head attention: K independent attention functions in parallel, concatenated → stabilizes, lets heads attend to different subspaces. GAT borrows: self-attention idea + multi-head for stability.
- self-attention also: Cheng et al. 2016 (machine reading), Lin et al. 2017 (structured self-attentive sentence embedding).

## The GAT layer derivation (the method)
Input: h = {h_1..h_N}, h_i ∈ R^F. Output h_i' ∈ R^{F'}.
1. Shared linear map W ∈ R^{F'×F} applied to every node (need at least one learnable transform for expressivity; shared = weight sharing across nodes/graphs).
2. Shared attention mechanism a: R^{F'} × R^{F'} → R computes e_ij = a(W h_i, W h_j) = importance of j to i.
   - General form lets every node attend to every other → drops structure. Inject structure via **masked attention**: only compute e_ij for j ∈ N_i (first-order neighbors incl. self).
3. Normalize across j: α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_{k∈N_i} exp(e_ik).
4. Concrete a: single-layer feedforward, weight vector **a** ∈ R^{2F'}, LeakyReLU (neg slope 0.2):
   e_ij = LeakyReLU( a^T [W h_i ‖ W h_j] ).
   Full: α_ij = exp(LeakyReLU(a^T[Wh_i‖Wh_j])) / Σ_k exp(LeakyReLU(a^T[Wh_i‖Wh_k])).
5. Output: h_i' = σ( Σ_{j∈N_i} α_ij W h_j ).
6. **Multi-head** (K heads) for stability: concat: h_i' = ‖_{k=1}^K σ(Σ_j α_ij^k W^k h_j) → KF' features. On **final** layer concat is not sensible → average: h_i' = σ( (1/K) Σ_k Σ_j α_ij^k W^k h_j ).

## Why it hits all the goals (derive, don't assert)
- **Per-neighbor importance:** α_ij learned from features → leap in capacity vs GCN's fixed 1/sqrt(d_i d_j). Const-GAT (a≡1) is the GCN-like ablation isolating this.
- **No eigendecomposition / no fixed structural operator:** a is a shared function of node *features*, applied edge-wise; no Laplacian, no inversion. Complexity O(|V| F F' + |E| F') — on par with GCN.
- **Inductive / structure-agnostic:** because a and W are shared and depend only on features (not on a fixed N×N operator), the exact same parameters apply to any graph → works on completely unseen test graphs (PPI). Directed graphs fine (just omit α_ij for absent edges). No upfront global structure needed.
- **Whole neighborhood, no ordering:** uses all of N_i (vs GraphSAGE sampling), and softmax over a set is permutation-invariant (vs LSTM aggregator's forced ordering).
- **Parallelizable:** e_ij over all edges and h_i' over all nodes are independent → parallel.

## Design-decision → why table
- **Shared W applied to all nodes:** need expressivity (raw features insufficient); sharing = weight-sharing/inductive transfer; one W per head.
- **Self-attention over neighbors (not global):** global attention drops structure & is O(N^2); masking to N_i injects graph prior cheaply and keeps it sparse/local.
- **Mask = first-order neighbors incl. self:** receptive field grows with depth (stack layers for multi-hop); include self-loop so node keeps own features (mirrors GCN's A+I).
- **Single-layer feedforward a (additive, Bahdanau-style):** simplest learnable scoring that handles variable inputs; framework agnostic but additive chosen for simplicity/interpretability. a ∈ R^{2F'} because it scores the concatenation [Wh_i‖Wh_j].
- **LeakyReLU (slope 0.2) inside a:** nonlinearity for the scoring net; LeakyReLU (vs ReLU) keeps gradient on negative pre-activations so the score isn't saturated/dead — important since the logit can be negative and we need it to remain discriminative before softmax.
- **softmax over N_i:** makes coefficients comparable across nodes with different degrees (normalizes); gives probabilistic, well-behaved, interpretable weights.
- **Multi-head:** stabilizes self-attention learning (Vaswani); different heads capture different relations; K-fold capacity.
- **Concat for hidden, average for output:** concat preserves all heads' info in hidden layers; at the prediction layer concat would give K·C outputs which is nonsensical for C classes → average (then apply softmax/sigmoid). Average delays final nonlinearity.
- **ELU hidden nonlinearity:** smooth, negative values allowed → mean activations closer to zero, faster convergence (Clevert).
- **Dropout on attention coefficients (p=0.6 Cora):** with tiny train sets (20/class), heavy regularization; dropping α_ij = stochastically sampling the neighborhood each step → strong regularizer specific to graphs.
- **L2 (λ=5e-4 Cora):** small-data regularization.
- **Glorot init, Adam, early stopping (patience 100) on val loss & acc:** standard, small-data friendly.
- **Transductive arch:** 2 layers, head config [8 heads × 8 feat] then [1 head × C] (Cora/Citeseer); Pubmed output uses 8 heads averaged + stronger L2.
- **Inductive arch (PPI):** 3 layers, [4×256, 4×256, 6×121-avg], skip connection across the middle attentional layer (He et al.) because deeper; no dropout/L2 (enough data); batch 2 graphs; sigmoid (multilabel).
- **Const-GAT ablation:** a(x,y)=1 → uniform weights → GCN-like inductive operator; isolates the value of *attention*.

## Canonical implementation grounding (PetarV-/GAT, TF)
- `attn_head` (utils/layers.py): 
  - seq_fts = conv1d(seq, out_sz, 1, no bias)  → this is W h (1x1 conv over nodes = shared linear map).
  - f_1 = conv1d(seq_fts, 1, 1); f_2 = conv1d(seq_fts, 1, 1)  → these compute the two halves of a^T[.‖.]: a = [a1‖a2], a1^T(Wh_i) and a2^T(Wh_j). logits = f_1 + transpose(f_2) gives e_ij = a1·Wh_i + a2·Wh_j (additive decomposition of a^T[Wh_i‖Wh_j]).
  - coefs = softmax( leaky_relu(logits) + bias_mat )  → bias_mat is -1e9 * (1 - adj) mask = masked attention (non-neighbors → -inf → 0 after softmax).
  - dropout on coefs (attention dropout) and on seq_fts (feature dropout).
  - vals = coefs @ seq_fts ; + bias ; optional residual ; activation.
- `GAT.inference` (models/gat.py): first layer = n_heads[0] heads concatenated; middle layers concat; output layer = n_heads[-1] heads, `tf.add_n(out)/n_heads[-1]` = averaging.
- `base_gattn.py`: masked_softmax_cross_entropy (transductive), masked_sigmoid_cross_entropy & micro_f1 (inductive), L2 weight decay + Adam.
- `execute_cora.py`: hid_units=[8], n_heads=[8,1], lr=5e-3, l2=5e-4, dropout 0.6, ELU, patience 100, adj_to_bias(nhood=1).

The clever impl trick: instead of materializing [Wh_i‖Wh_j] for every edge (2F' per edge), decompose a^T[x‖y] = a_1^T x + a_2^T y, compute two N-vectors and broadcast-add → O(N) scoring then masked. This is the efficiency derivation worth living out in reasoning.md.
