# Synthesis — ChebNet (fast localized spectral filtering on graphs)

## Pain point / research question
CNNs win on grids (images, audio) because convolution = local, translation-invariant, weight-shared filters with O(S) parameters (S = support size, independent of input size n), composed hierarchically with pooling. We want the *same* on arbitrary graphs (social nets, brain connectomes, word-embedding graphs, k-NN feature graphs). Obstacle: convolution and pooling are only defined on regular grids. On a graph there is no canonical translation operator in the vertex/spatial domain — "shift the filter by one node" is ill-defined because neighborhoods have different sizes and no consistent ordering. So a naive spatial filter can't share weights cleanly.

Two routes:
- Spatial: localization is free (finite kernel) but matching/ordering neighborhoods to share weights is ill-posed (Bruna 2013 noted this).
- Spectral: translation/convolution get a clean definition through the graph Fourier transform, but (i) the filter is not naturally localized in space and (ii) applying it costs O(n²) and needs the eigendecomposition O(n³).

Goal: localized graph filters that are (a) provably K-hop localized, (b) O(K) parameters (same as classical CNNs), (c) O(K|E|) to evaluate, eigendecomposition-free.

## The math chain (all main-text §2.1; this NIPS paper has no separate appendix — it is self-contained)

**Graph + Laplacian.** Undirected weighted G=(V,E,W), |V|=n, signal x∈R^n. Combinatorial Laplacian L = D − W, D=diag(degree). Normalized L = I_n − D^{-1/2} W D^{-1/2}. L is real symmetric PSD ⇒ complete orthonormal eigenbasis {u_l}, eigenvalues 0 ≤ λ_0 ≤ … ≤ λ_{n-1} (the graph "frequencies"). Diagonalization L = UΛU^T, U=[u_0,…,u_{n-1}], Λ=diag(λ_l).

**Why these eigenvectors are "Fourier modes".** On a ring graph the Laplacian is the discrete second-difference operator; its eigenvectors are the classical sinusoids and eigenvalues ~ squared frequency. Generalizing: the Dirichlet energy x^T L x = ½Σ_{ij} W_{ij}(x_i/√d_i − x_j/√d_j)² (normalized) measures signal smoothness; eigenvectors of small λ vary slowly across edges (low frequency), large λ oscillate (high frequency). So L's eigenbasis is the natural Fourier basis for the graph.

**Graph Fourier transform.** x̂ = U^T x ∈ R^n; inverse x = U x̂. (Orthonormality U^T U = I gives invertibility.)

**Convolution via convolution theorem.** No spatial translation, so DEFINE convolution to diagonalize in the Fourier basis: x ∗_G y = U( (U^T x) ⊙ (U^T y) ). Filtering by g_θ:
  y = g_θ(L) x = g_θ(UΛU^T) x = U g_θ(Λ) U^T x.
(Key identity: for analytic g, g(UΛU^T) = U g(Λ) U^T because (UΛU^T)^k = UΛ^k U^T by orthonormality telescoping U^TU=I.)

**Non-parametric filter.** g_θ(Λ)=diag(θ), θ∈R^n free Fourier coefficients. Problems: (i) NOT localized in vertex space (a generic spectral multiplier has global spatial support); (ii) O(n) parameters; (iii) needs full U (EVD O(n³), storage n²) and two O(n²) multiplications by U per pass.

**Polynomial filter ⇒ localization.** Restrict g_θ(Λ)=Σ_{k=0}^{K-1} θ_k Λ^k (θ∈R^K). Then g_θ(L)=Σ_k θ_k L^k. Localization claim: the filter centered at i, applied to delta δ_i, is (g_θ(L)δ_i)_j = (g_θ(L))_{ij} = Σ_k θ_k (L^k)_{ij}. By Hammond–Vandergheynst–Gribonval (2011, Lemma 5.2): if shortest-path distance d_G(i,j) > k then (L^k)_{ij}=0. Reason: (L^k)_{ij} = Σ over length-k walks i→j of products of edge entries; if d_G(i,j)>k there is NO walk of length ≤k, so every term is 0. Hence a degree-(K−1) polynomial of L is exactly (K−1)-localized: nonzero only within K−1 hops (paper states "K-localized"; with K terms k=0..K-1 the max reach is K−1 hops). Learning complexity O(K) = support size, same as classical CNNs.

**Still O(n²).** Computing y = U g_θ(Λ) U^T x explicitly still multiplies by U (needs EVD + two dense n×n mults). But g_θ(L)=Σθ_k L^k can be applied DIRECTLY via L without ever forming U: each L^k x is k sparse mat-vecs, L sparse with |E| nonzeros ⇒ O(K|E|). Naive monomial recurrence L^k x = L(L^{k-1}x) works but L^k is numerically unstable (powers of an operator with spread spectrum). Want a STABLE orthogonal-polynomial recurrence.

**Chebyshev.** T_k(y)=2y T_{k-1}(y) − T_{k-2}(y), T_0=1, T_1=y; orthogonal on [−1,1] w.r.t. dy/√(1−y²); domain is [−1,1] so rescale eigenvalues: Λ̃ = 2Λ/λ_max − I_n ∈ [−1,1] (since λ∈[0,λ_max] ⇒ 2λ/λ_max−1 ∈ [−1,1]). Filter:
  g_θ(Λ) = Σ_{k=0}^{K-1} θ_k T_k(Λ̃),
  y = g_θ(L)x = Σ_{k=0}^{K-1} θ_k T_k(L̃)x, L̃ = 2L/λ_max − I_n.
Define x̄_k = T_k(L̃)x ∈ R^n. Recurrence in vectors:
  x̄_0 = x, x̄_1 = L̃x, x̄_k = 2 L̃ x̄_{k-1} − x̄_{k-2}.
Verify by applying T_k recurrence to L̃ and multiplying by x: T_k(L̃)=2L̃ T_{k-1}(L̃) − T_{k-2}(L̃) ⇒ x̄_k = 2L̃x̄_{k-1} − x̄_{k-2}. Each step = 1 sparse mat-vec L̃·(·) + a few adds ⇒ O(K|E|). Then y = [x̄_0,…,x̄_{K-1}] θ. Eigendecomposition-free, K-localized (same polynomial degree argument since T_k is degree-k in L̃, and L̃ is an affine function of L so a degree-(K-1) poly in L̃ is a degree-(K-1) poly in L ⇒ same K-localization).

For normalized L the spectrum is bounded by λ_max ≤ 2 (Chung), so practical choice λ_max=2 ⇒ L̃ = L − I_n (code uses rescale_L with lmax=2). Setting λ_max=2 frees the model from computing the true largest eigenvalue (which would need a power iteration); any slack just rescales coefficients, which learning absorbs.

**Filter bank / learning.** Layer maps F_in→F_out feature maps:
  y_{s,j} = Σ_{i=1}^{F_in} g_{θ_{i,j}}(L) x_{s,i} ∈ R^n,
with F_in×F_out coefficient vectors θ_{i,j}∈R^K trainable. Cost O(K|E| F_in F_out S). Backprop gradients: ∂E/∂θ_{i,j} = Σ_s [x̄_{s,i,0},…,x̄_{s,i,K-1}]^T ∂E/∂y_{s,j}; ∂E/∂x_{s,i} = Σ_j g_{θ_{i,j}}(L) ∂E/∂y_{s,j}. The Chebyshev basis [x̄_0..x̄_{K-1}] is computed once per input and reused; everything is sparse-mat-vec + dense mat-mul ⇒ GPU-friendly.

## Coarsening + pooling
Pooling needs meaningful neighborhoods (clusters). Graph clustering NP-hard ⇒ use a fast multilevel greedy method (Graclus, built on Metis): at each level pick unmarked vertex i, match with unmarked neighbor j maximizing local normalized cut W_{ij}(1/d_i + 1/d_j); merge weights; halves node count per level. This gives a hierarchy of coarser graphs (each ~½ size), analogous to grid subsampling.

Fast pooling trick: after coarsening, vertices have no useful order, so naive pooling needs a lookup table (slow, not parallel). Instead build a balanced binary tree: each coarse node has exactly 2 children — matched pairs give 2 regular children; singletons get 1 real + 1 FAKE (disconnected) child; fake nodes get 2 fake children. Fake nodes carry a neutral value (0 for ReLU+max-pool) and, being disconnected, are untouched by filtering. Order coarsest level arbitrarily, propagate: node k → children 2k, 2k+1. Result: the finest level is ordered so adjacent nodes are the ones merged upward ⇒ pooling = plain 1D pooling of size p (power of 2), memory-local, GPU-friendly.

## Baselines (prior art to elaborate)
- **Bruna et al. 2013 (Spectral Networks).** First spectral graph CNN. Filter g_θ(Λ)=Bθ, B = cubic B-spline basis (n×K), θ control points. Smoothness in Fourier domain → some spatial localization, but NO precise control of support; needs U ⇒ EVD O(n³) + two O(n²) mults per pass; doesn't scale. This is the work being directly improved.
- **Henaff, Bruna, LeCun 2015.** Extended above + learned graph from data; same O(n²) bottleneck; raised the 3 concerns (complexity, graph quality, validity of locality/stationarity assumptions).
- **Graph Neural Network (Scarselli 2009; Li GG-NN 2015).** RNN-style node embedding via transition function; with f a diffusion s=Wx and output θLx + x, a K-layer GNN reproduces degree-K Chebyshev-like diffusion. ChebNet = multiple diffusion + node-local layers, interpretation.
- **Local Receptive Fields (Gregor-LeCun 2010, Coates-Ng 2011).** Group features by similarity to cut connections — locality but NO weight sharing/stationarity.
- **Geodesic CNN (Masci 2015).** Spatial conv on 3D meshes via geodesic polar coords; only smooth low-dim manifolds.

## Ancestor concepts (Background)
- Spectral graph theory (Chung 1997); Graph Signal Processing review (Shuman 2013); graph wavelets via spectral graph theory (Hammond 2011 — source of the Chebyshev-on-Laplacian trick and the localization lemma); Deep Learning / CNNs (LeCun 1998, 2015); convolution theorem (Mallat).

## Evaluation settings (no outcomes)
MNIST (70k digits 28×28; build 8-NN graph of the 2D grid via W_{ij}=exp(−‖z_i−z_j‖²/σ²), z=pixel coord; n=976 incl fake nodes, |E|=3198). 20NEWS text categorization (18,846 docs, 20 classes; 10k most common words; bag-of-words per doc; 16-NN graph on word2vec embeddings, n=10k, |E|=132,834). Architectures notation GCk/Pk/FCk; ReLU; softmax + cross-entropy + ℓ2 on FC; SGD/momentum (MNIST) or Adam (20NEWS); batch 100; 20 epochs. Metric: classification accuracy; also wall-clock per minibatch CPU/GPU.

## Design decisions → why
- Spectral over spatial: spatial has no well-defined translation/weight-sharing on irregular neighborhoods.
- Polynomial of L: gives provable K-hop localization (Hammond Lemma 5.2) + O(K) params.
- Chebyshev over raw monomials L^k: numerically stable orthogonal recurrence; equi-ripple/minimax behavior; both still degree-k so same localization, but monomial powers are ill-conditioned.
- Recurrence on L̃ directly (not via U): kills the EVD and the two O(n²) U-mults → O(K|E|), eigendecomposition-free, no storing n² basis.
- λ_max=2 for normalized L: spectrum bounded by 2, avoids computing the true λ_max; slack absorbed by learned coefficients.
- Graclus coarsening (÷2 per level) + binary-tree fake-node reordering: makes graph pooling = 1D pooling, parallel/memory-local.
- Filter bank Fin×Fout each with K coeffs: mirrors classical conv layer weight-sharing across the whole graph (stationarity).

## Canonical code (mdeff/cnn_graph, lib/graph.py + lib/models.py)
- graph.laplacian(W, normalized): L = I − D^{-1/2} W D^{-1/2}.
- graph.rescale_L(L, lmax=2): L̃ = (2/lmax) L − I = L − I.
- graph.chebyshev(L, X, K): stack x̄_0=X, x̄_1=L X, x̄_k = 2 L x̄_{k-1} − x̄_{k-2}.
- models.cgcnn.chebyshev5(x,L,Fout,K): sparse-tensor recurrence building [x̄_0..x̄_{K-1}], reshape to N·M × Fin·K, multiply by W (Fin·K × Fout) → per-node Fout features. THE method.
- b1relu (one bias per filter + ReLU), mpool1 (1D max pool size p on the reordered signal), fc, _inference (graph conv layers → flatten → FC → softmax).
