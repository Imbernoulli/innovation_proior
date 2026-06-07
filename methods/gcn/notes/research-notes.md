# GCN research notes (Phase 1 synthesis)

Primary paper: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017, arXiv 1609.02907. Read full text + Appendix A (WL relation, random-weight karate club, semi-supervised embeddings) + Appendix B (model depth, residual variant). Source in src/main.tex, bib in src/main.bbl.

## Load-bearing ancestors (verified against primary paper)

1. Graph-Laplacian regularization for SSL (Zhu 2003 label propagation / Gaussian fields + harmonic functions; Belkin 2006 manifold reg; Weston 2012 deep semi-supervised embedding; Zhou 2004 local & global consistency). Loss L = L0 + lambda * L_reg with L_reg = sum_ij A_ij ||f(Xi)-f(Xj)||^2 = f(X)^T Delta f(X), Delta = D - A unnormalized Laplacian. CORE LIMITATION (stated in paper intro & discussion): assumes edges encode mere similarity (connected => same label); restricts modeling capacity because edges may carry info beyond similarity. Also f conditioned only on X, not on A.

2. Spectral graph theory / graph Laplacian. Normalized Laplacian L = I - D^{-1/2} A D^{-1/2} = U Lambda U^T. Eigenvectors U = graph Fourier basis; eigenvalues = frequencies; smoothness via x^T L x quadratic form. Normalized L has eigenvalues in [0,2]. Graph Fourier transform U^T x. Multiplying by U is O(N^2); eigendecomp prohibitive for big graphs.

3. Bruna 2014 (Spectral networks, ICLR 2014). First spectral CNN on graphs: filter = diag of free params in Fourier domain, g_theta(Lambda). LIMITATIONS: (a) O(N) free params per filter (non-parametric), (b) filters not spatially localized, (c) needs full eigendecomposition + O(N^2) multiply by U, (d) basis-dependent / non-transferable.

4. Defferrard 2016 ChebNet (NIPS 2016). Approximate g_theta(Lambda) by truncated Chebyshev expansion sum_{k=0}^K theta'_k T_k(tilde Lambda), tilde Lambda = (2/lambda_max) Lambda - I. Then g_theta * x ≈ sum_k theta'_k T_k(tilde L) x, tilde L = (2/lambda_max) L - I. K-localized (depends on K-hop neighborhood), O(|E|) cost, no eigendecomposition. Recurrence T_k = 2x T_{k-1} - T_{k-2}, T_0=1, T_1=x. Hammond 2011 is the source of the Chebyshev-on-Laplacian approximation. LIMITATION GCN reacts to: K>1 still K params per filter; risk of overfitting on local neighborhood for wide-degree graphs; can build deeper instead.

5. Skip-gram graph embeddings: DeepWalk (Perozzi 2014) = random walks + skip-gram (Mikolov 2013); LINE, node2vec extend walk schemes; Planetoid (Yang 2016) injects labels into embedding learning. LIMITATION: multi-step pipeline (walk gen + embedding + classifier), each optimized separately; not end-to-end.

6. Graph NN lineage: Gori 2005 / Scarselli 2009 (recurrent, contraction map to fixed point); Li 2016 GG-NN (modern RNN training); Duvenaud 2015 (conv-like, but degree-specific weight matrices -> doesn't scale to wide degree distributions); Atwood 2016 DCNN (O(N^2)); Niepert 2016 (node ordering needed).

## Key derivation chain (from Sec 2)
Eq2 g_theta * x = U g_theta U^T x  (spectral conv)
Eq3 Chebyshev approx of g_theta(Lambda)
Eq4 g_theta' * x ≈ sum theta'_k T_k(tilde L) x  (ChebNet)
limit K=1, approx lambda_max ≈ 2:
Eq5 g * x ≈ theta'_0 x + theta'_1 (L - I) x = theta'_0 x - theta'_1 D^{-1/2} A D^{-1/2} x
constrain theta = theta'_0 = -theta'_1:
Eq6 g * x ≈ theta (I + D^{-1/2} A D^{-1/2}) x ; eigenvalues of (I+D^{-1/2}AD^{-1/2}) in [0,2] -> repeated application unstable
Renormalization trick: I + D^{-1/2}AD^{-1/2} -> tilde D^{-1/2} tilde A tilde D^{-1/2}, tilde A = A + I, tilde D_ii = sum_j tilde A_ij
Eq7 Z = tilde D^{-1/2} tilde A tilde D^{-1/2} X Theta  (multi-channel), complexity O(|E| F C)
Layer rule Eq1: H^{l+1} = sigma( tilde D^{-1/2} tilde A tilde D^{-1/2} H^l W^l )
2-layer model Eq9: Z = softmax( Ahat ReLU( Ahat X W0 ) W1 ), Ahat = tilde D^{-1/2} tilde A tilde D^{-1/2}
Loss: cross-entropy over labeled nodes only.

## WL connection (Appendix A)
WL-1: h_i <- hash(sum_{j in N_i} h_j). Replace hash with sigma( sum_{j in N_i} (1/c_ij) h_j W ); choose c_ij = sqrt(d_i d_j) -> recovers GCN propagation in vector form. GCN = differentiable, parameterized generalization of 1-WL. Untrained random-weight GCN already gives community-structured embeddings on karate club.

## Code (canonical, in code/)
- tkipf/gcn (TensorFlow): utils.normalize_adj (symmetric D^{-1/2} A D^{-1/2}), preprocess_adj (adds I then normalizes = renormalization trick), chebyshev_polynomials (Eq4 support list), layers.GraphConvolution (sum over supports of Ahat (x W)), models GCN (2 layers).
- tkipf/pygcn (PyTorch, cleaner): layers.GraphConvolution forward = spmm(adj, x W); models.GCN = relu(gc1) -> dropout -> gc2 -> log_softmax; utils.normalize row-normalizes (note pygcn uses row-norm of A+I as a simplification), load_data.

## Field state at the time (mid-2016)
SSL on graphs split into (a) explicit Laplacian regularization and (b) graph-embedding/skip-gram pipelines. Spectral CNNs (Bruna -> Defferrard) existed but applied to graph-level / signal tasks, not large-scale transductive node classification. No simple, scalable, end-to-end node classifier conditioned on both X and A. Pain points: Laplacian reg's similarity assumption; embedding pipelines' multi-stage non-end-to-end optimization; spectral methods' cost/locality.

## NO hindsight (do not use): GAT, GraphSAGE, oversmoothing literature, SGC, PPNP, anything post-2016.
