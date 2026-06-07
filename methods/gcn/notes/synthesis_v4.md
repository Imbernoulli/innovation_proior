# GCN synthesis (V4) — notes-first, composed-from for results_v4

Primary: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks",
ICLR 2017, arXiv 1609.02907. Read full main text + Appendix A (WL relation, random-weight
karate club, semi-supervised embeddings) + Appendix B (depth, residual variant). Source
src/main.tex, bib src/main.bbl. Canonical code: code/tkipf-gcn (TF), code/pygcn (PyTorch).

This V4 notes file is written FIRST. results_v4/{context,reasoning,answer}.md are transcribed
FROM it. The big change vs V3: the context.md Code-framework section must be a MINIMAL
PRE-METHOD scaffold that presupposes NOTHING about spectral/Chebyshev/renormalized propagation.
At context time we do not yet know we will design any of that. So below I keep two things
strictly separate: (i) the design-decision -> why table (this is the METHOD, lives in reasoning
/answer), and (ii) the pre-method scaffold (this is what is knowable BEFORE the method, lives
in context.md Code-framework).

================================================================================
PART 0 — What is knowable BEFORE the method (drives context.md, esp. Code-framework)
================================================================================

Pre-method facts only. Someone in mid-2016 facing transductive node classification knows:
- The data: one fixed graph. A sparse adjacency matrix (binary/weighted, undirected),
  a dense per-node feature matrix X (e.g. sparse bag-of-words), labels on a small subset.
- The task: predict labels on the unlabeled nodes using BOTH X and the graph.
- The tooling that already exists: a deep-learning framework with autodiff + GPU
  (TensorFlow / PyTorch), SciPy sparse matrices, the Adam optimizer, dropout, Glorot init,
  cross-entropy loss, the generic notion of "stack layers, each = linear map + nonlinearity."
- The transductive trick that already exists: compute outputs for ALL nodes, supervise only
  on the labeled subset via a MASK on the loss. (This is generic SSL bookkeeping, not the
  method.)

What is NOT yet known at context time (must NOT appear in the scaffold):
- that the per-layer operation will be a graph convolution
- anything spectral: Laplacian, eigenbasis, Fourier, Chebyshev, polynomial-in-L
- the K=1 simplification, lambda_max ~ 2
- symmetric vs random-walk normalization, D^{-1/2} A D^{-1/2}
- the renormalization trick, A+I, D~, Ahat
- "support" lists of operators, sparse-dense matmul of a graph operator
- the names normalize_adj / preprocess_adj / chebyshev_polynomials / GraphConvolution
- the words "reference implementation" / "official repo" / method name "GCN".

So the pre-method scaffold is JUST a bare graph SSL node-classification harness:
  * load sparse adjacency A, dense features X, labels Y, train/val/test masks
  * a SINGLE big empty slot: def graph_layer(H, A, W):  # TODO: the propagation we'll design
  * a class GraphModel: # TODO  that stacks such layers + nonlinearity
  * Adam + dropout, masked cross-entropy on labeled nodes, full-batch loop.
Everything method-specific is one TODO. The scaffold must correspond piece-for-piece to the
final code: graph_layer -> the GraphConvolution layer; GraphModel -> the 2-layer GCN;
masked CE / Adam / dropout -> unchanged.

SCAFFOLD<->FINAL CODE CORRESPONDENCE TABLE (for self-check (d)):
  scaffold piece                  ->  final code piece
  load_graph_data()               ->  utils.load_data (sparse A, dense X, masks)  [unchanged]
  def graph_layer(H, A, W): TODO  ->  GraphConvolution: act( Ahat @ dropout(H) @ W )
  class GraphModel: TODO          ->  GCN: softmax(Ahat ReLU(Ahat X W0) W1)
  masked_cross_entropy(...)       ->  masked_softmax_cross_entropy  [unchanged]
  Adam + dropout + full-batch loop->  AdamOptimizer(0.01), dropout 0.5, 200 epochs [unchanged]
  (no scaffold entry for A's preprocessing) -> preprocess_adj is PART OF the method (the TODO),
       so the scaffold passes A in raw; the method fills in what to do to A.

NOTE: V3 scaffold leaked normalize_adj (D^{-1/2}), chebyshev_polynomials, the "support" list,
and the phrase "the method's propagation operator". All of that is METHOD and is REMOVED from
V4 context.md, moved into reasoning/answer where it is derived.

================================================================================
PART 1 — Load-bearing ancestors (context.md Background + Baselines; reasoning elaborates)
================================================================================

A1. Graph-Laplacian regularization SSL (Zhu 2003 label propagation / Gaussian fields +
    harmonic functions; Zhou 2004 local & global consistency; Belkin 2006 manifold reg;
    Weston 2012 deep semi-supervised embedding).
    Idea: L = L0 + lambda * L_reg, L_reg = sum_ij A_ij ||f(Xi)-f(Xj)||^2 = f(X)^T Delta f(X),
    Delta = D - A unnormalized Laplacian. Quadratic-form identity (DERIVE in reasoning):
      f^T Delta f = f^T D f - f^T A f = sum_i d_i ||f_i||^2 - sum_ij A_ij f_i^T f_j;
      sum_ij A_ij ||f_i - f_j||^2 = 2 sum_i d_i||f_i||^2 - 2 sum_ij A_ij f_i^T f_j; equal up to 2.
    LIMITATION: bakes in "edge => same label" (similarity cap); and f = f(X) only, graph never
    enters the predictor, only the loss. Label propagation = purest form, ignores X beyond edges.

A2. Spectral graph theory (the machinery, Background).
    Normalized Laplacian L = I - D^{-1/2} A D^{-1/2} = U Lambda U^T, real symmetric PSD.
    Why this object: convolution defined via translation, translation undefined on irregular
    graph; escape = convolution theorem (conv in space = pointwise mult in Fourier). Need a
    Fourier basis -> orthonormal eigenbasis of a symmetric operator. U = graph Fourier modes
    (low eigenvalue = slow/smooth mode, high = oscillatory), eigenvalues = frequencies, GFT =
    U^T x, inverse = U(.). Roughness x^T L x >= 0 => PSD => spectrum starts at 0.
    KEY BOUND: normalized L eigenvalues in [0,2]. Proof: D^{-1/2}AD^{-1/2} similar to D^{-1}A
    (random walk), rows convex combos => eigenvalues in [-1,1] => L = I - that in [0,2], 2 on
    bipartite. (Unnormalized Delta's top eigenvalue grows with max degree — boundedness is the
    gift of normalizing.)
    COST: forming U is O(N^3); applying U or U^T is O(N^2) dense. Prohibitive at 20k nodes.

A3. Hammond 2011 (spectral graph wavelets). g(L) can be APPLIED without forming U: expand a
    smooth spectral function in Chebyshev polynomials of a rescaled Laplacian; since
    (U Lambda U^T)^k = U Lambda^k U^T, a polynomial in Lambda through U equals the polynomial
    in L applied directly. L sparse => K-order polynomial in L costs K sparse mat-vecs =
    O(K|E|), no eigendecomposition. Chebyshev chosen: bounded |T_k|<=1 and orthogonal on
    [-1,1], three-term recurrence T_k = 2y T_{k-1} - T_{k-2}, T_0=1, T_1=y, stable for smooth
    functions => rescale L into [-1,1].

A4. Bruna 2014 (Spectral networks, ICLR 2014) — BASELINE. First spectral CNN on graphs:
    filter = free diagonal in Fourier domain, g_theta = diag(theta), theta in R^N;
    g_theta * x = U diag(theta) U^T x. FOUR GAPS: (i) O(N) free params (non-parametric,
    over-flexible); (ii) not spatially localized (arbitrary spectral diagonal = global support
    in node domain); (iii) needs O(N^3) eigendecomp + O(N^2) U/U^T multiplies; (iv) basis-
    specific, theta tied to one graph's U, non-transferable.

A5. Defferrard 2016 ChebNet (NIPS 2016) — BASELINE / direct ancestor.
    g_{theta'} * x ~ sum_{k=0}^K theta'_k T_k(L~) x, L~ = (2/lambda_max) L - I in [-1,1].
    K-localized (K-hop), O(K|E|), no eigendecomp, K params, transferable (poly coeffs not U
    coords). GAP GCN reacts to: each filter still K coeffs; K does DOUBLE DUTY = filter
    expressiveness AND receptive-field radius, tangled; on wide-degree graphs a high-K filter
    around a hub reaches a huge neighborhood and can overfit local structure. If you'll get
    receptive field from stacking and expressiveness from nonlinearities, per-layer
    parameterization is richer than necessary.

A6. Skip-gram graph embeddings (BASELINE): DeepWalk (Perozzi 2014) = random walks + skip-gram
    (Mikolov 2013); LINE (Tang 2015), node2vec (Grover 2016) extend walk schemes; Planetoid
    (Yang 2016) injects labels into embedding objective. GAP: multi-stage pipeline (walk gen +
    unsupervised embedding + separate classifier), each optimized separately, embedding never
    sees labels (Planetoid partly fixes but heavy, separately tuned).

A7. Iterative classification ICA (Lu & Getoor 2003; Sen 2008) — BASELINE. Local logistic
    regression on features + bootstrap + iterate relational classifier on aggregate of
    neighbors' CURRENT label estimates. GAP: aggregates label info not learned representations;
    hand-built aggregation + iterate-to-convergence.

A8. Other GNNs (Background / Baselines): Gori 2005 / Scarselli 2009 recurrent contraction-map
    to fixed point (straitjacket); Li 2016 GG-NN gated; Duvenaud 2015 conv-like for molecules
    but DEGREE-SPECIFIC weight matrices (one per degree — explodes on wide degree dist, no
    transfer); Atwood 2016 DCNN O(N^2); Niepert 2016 needs a node ordering for a 1D CNN.
    COMMON GAP: none gives a single weight matrix per layer that scales O(|E|) and handles
    wide degree distributions via normalization alone.

A9. Depth as substitute for filter order (Background). Prevailing NN wisdom: stack simple
    layers + pointwise nonlinearity recovers functions a single wide layer can't; very deep
    nets trainable via residuals (He 2015). => Alternative to high-order filters: keep each
    layer's filter minimal, stack. Also: a layer applied many times ~ power iteration, so a
    spectral radius != 1 compounds with depth -> stability of per-layer operator is first-class.

================================================================================
PART 2 — DESIGN-DECISION -> WHY table (drives reasoning.md derivations + answer.md)
Each row: decision | why this | rejected alternative + its failure mode
================================================================================

D1. Put the graph in the MODEL not the loss (predictor f(X,A)).
    Why: loss-side smoothness assumes edge=>similarity (cap) AND graph never reaches predictor;
    conditioning on A lets gradient from few labels flow along edges to unlabeled nodes and lets
    the model decide what an edge means.
    Rejected: Laplacian regularization f^T Delta f (A1) — similarity cap, f=f(X) only.

D2. Define convolution SPECTRALLY (via convolution theorem), not by sliding a filter.
    Why: translation undefined on irregular graph (no canonical "shift by one"); convolution
    theorem is the only translation-free definition.
    Rejected: spatial sliding-window conv — needs translation; degree-specific aggregation
    (Duvenaud) — param explosion, no transfer; node-ordering 1D CNN (Niepert) — arbitrary order.

D3. Basis = symmetric normalized Laplacian L = I - D^{-1/2}AD^{-1/2}.
    Why: real symmetric => orthonormal eigenbasis (Parseval, inverse = U^T), PSD, bounded
    spectrum [0,2].
    Rejected: raw A (not the right Fourier object); unnormalized Delta = D-A (top eigenvalue
    grows with degree, unbounded spectrum).

D4. Free spectral filter g_theta = diag(theta) -> REJECT (this is the wall after D3).
    Four gaps (A4): O(N) params, not localized, needs eigendecomp + O(N^2), non-transferable.

D5. Chebyshev polynomial filter (ChebNet, via Hammond).
    Why: g_theta(Lambda) ~ sum theta'_k T_k(L~); (U Lambda U^T)^k = U Lambda^k U^T telescopes
    U away => apply sum theta'_k T_k(L~) directly on sparse L~; K params, K-localized (K-hop),
    O(K|E|), no eigendecomp, transferable. DERIVE telescoping + recurrence inline.
    Rejected: free diagonal (D4).

D6. Rescale Laplacian to [-1,1]: L~ = (2/lambda_max) L - I.
    Why: Chebyshev bounded/orthogonal only on [-1,1]; outside, recurrence blows up like y^k.
    Rejected: un-rescaled L — numerical blowup.

D7. K = 1 (linear filter per layer).
    Why: in ChebNet K does double duty (expressiveness AND receptive field). Untangle: depth
    supplies hops (k one-hop layers => k-hop receptive field), nonlinearities supply
    expressiveness; so each layer needs only first-order filter. Bonus regularization: fewer
    params per layer fights overfitting large 1-hop neighborhoods of hubs.
    Rejected: K>1 — tangles the two, over-parameterized per layer, hub overfitting.

D8. Approximate lambda_max ~ 2.
    Why: spectrum in [0,2] (D3) makes 2 a safe guess; trainable downstream weights absorb the
    scale error (operator scale is a free gauge). Avoids a per-graph eigensolver call.
    Then L~ = L - I = -(D^{-1/2}AD^{-1/2}); g*x ~ theta'_0 x - theta'_1 D^{-1/2}AD^{-1/2} x.
    Rejected: exact lambda_max via eigsh — unnecessary per-graph cost.

D9. Symmetric normalization D^{-1/2}AD^{-1/2}, not random-walk D^{-1}A.
    Why: (a) spectral story needs a SYMMETRIC operator (orthonormal eigenbasis, real spectrum);
    D^{-1}A non-symmetric, breaks it. (b) symmetric weights edge (i,j) by 1/sqrt(d_i d_j),
    not 1/d_i => not mere averaging, richer dynamics, down-weights high-degree neighbors
    (1/sqrt(d_j) on far end).
    Rejected: D^{-1}A random-walk (averaging, non-symmetric); raw A (degree-dependent rescale,
    activations blow up/crush layer to layer — why normalize at all).

D10. Tie theta = theta'_0 = -theta'_1 (single parameter).
    Why: fewer params (regularization), fewer matmuls; g*x ~ theta (I + D^{-1/2}AD^{-1/2}) x =
    "yourself plus normalized neighbors."
    Rejected: two free coeffs — more params/matmuls, no real gain given D7/depth.

D11. Renormalization trick: Ahat = D~^{-1/2} A~ D~^{-1/2}, A~=A+I, D~_ii = sum_j A~_ij.
    Why (LOAD-BEARING for depth): I + D^{-1/2}AD^{-1/2} = 2I - L has eigenvalues in [0,2],
    top ~2; stacking an operator of spectral radius ~2 explodes signals ~2^depth (and crushes
    near-0 components) => exploding/vanishing with depth, hostile to the very depth strategy D7.
    Cause: self-loop (the +I) added AFTER normalizing, so its mass isn't in the degree norm;
    node over-weighted, pushing top eigenvalue 1->2. Fix: fold self-loop in BEFORE normalizing.
    A~=A+I, normalize with D~ => Ahat is the normalized adjacency of the self-looped graph =>
    spectrum back in [-1,1], radius ~1, stacks safely; self-loop now weighted 1/sqrt(d~_i d~_j)
    consistently with neighbors.
    Rejected: I + D^{-1/2}AD^{-1/2} (radius 2, depth-unstable).

D12. Multi-channel + layer rule. Theta in R^{C×F}; Z = Ahat X Theta (O(|E|FC), sparse-dense).
    H^{(l+1)} = sigma(Ahat H^{(l)} W^{(l)}), sigma=ReLU, H^0 = X. Stack 2 => 2-hop receptive
    field. Z = softmax(Ahat ReLU(Ahat X W0) W1). Loss = CE over labeled nodes only; outputs
    for all N nodes so features/grad flow across edges to unlabeled. No lambda, no f^T Delta f.

D13. WL connection (Appendix A). Node-wise: h_i^{(l+1)} = sigma( sum_{j in N_i} (1/c_ij)
    h_j^{(l)} W ), c_ij = sqrt(d~_i d~_j). Compare 1-WL: h_i <- hash(sum_{j in N_i} h_j).
    GCN = differentiable, parameterized, normalized generalization of one 1-WL step: hash ->
    sigma(.W), unit weight -> 1/c_ij. The constant c_ij = sqrt(d~_i d~_j) is FORCED (not free):
    it's the only c that makes the WL sum equal the symmetric-normalized operator from D9/D11.
    Two motivations (spectral + WL) land on the SAME operator. Prediction (testable w/o
    training): untrained random-weight GCN on featureless X=I, small community graph (karate
    club, 34 nodes, 4 communities) already yields community-structured embeddings (3 layers,
    Glorot W, tanh, 2D). Aggregation does the work; training sharpens.

D14. Training hygiene (answer.md): Adam lr 0.01, dropout 0.5, L2 5e-4 on first layer, 16 hidden
    units, Glorot init, row-normalized features, early stop on val loss (window 10), 200 epochs,
    full-batch over whole graph, sparse Ahat => memory O(|E|).
    (Residual variant for very deep: H^{(l+1)} = sigma(Ahat H^l W^l) + H^l, He 2015 — Appendix B.)

================================================================================
PART 3 — reasoning.md ORDER (insight-before-method, all derivations inline)
================================================================================
Each item: lead with pain/constraint, let the formula DROP OUT.
1. concrete failure: one graph, ~20 labels/class, must use words AND citations.
2. everyone puts graph in loss -> DERIVE the quadratic-form identity -> see the two caps
   (similarity + f=f(X)) -> resolution: f(X,A), graph in model. (D1)
3. survey GNNs (recurrent/contraction, degree-specific Duvenaud, Atwood O(N^2), Niepert order)
   -> none clean -> go spectral. (D2/A8)
4. WHY spectral: translation undefined -> convolution theorem -> need Fourier basis. (D2)
5. WHICH basis: symmetric normalized L, derive [0,2] bound, contrast unnormalized Delta. (D3)
6. free spectral filter (Bruna) -> hit FOUR walls. (D4) DEAD END.
7. filter as function of spectrum -> Chebyshev (Hammond); DERIVE telescoping (U Lambda U^T)^k
   = U Lambda^k U^T and recurrence; rescale to [-1,1] why. (D5/D6) -> ChebNet, good, could stop.
8. but K double-duty bugs me -> untangle via depth + nonlinearity -> K=1. (D7)
9. drop lambda_max ~ 2 (gauge absorbed) -> theta'_0 x - theta'_1 D^{-1/2}AD^{-1/2} x. (D8)
10. pause on normalization: sym vs random-walk vs raw A. (D9)
11. tie coeffs -> theta (I + D^{-1/2}AD^{-1/2}). (D10)
12. WALL: spectral radius ~2, depth blows up -> diagnose cause (+I after norm) -> renormalization
    trick -> Ahat, radius ~1. (D11) AHA: stability restored.
13. multi-channel, layer rule, 2-layer model, masked loss. (D12)
14. AHA #2: node-wise reading = generalized 1-WL; c_ij forced; untrained-embedding prediction.
    (D13) — emerges from the math, not announced.
15. recap why-each-simplification-stands paragraph (already in V3, keep).
16. real code (tkipf-grounded), comments tie back, causal-chain recap. (D12/D14)

DEAD ENDS to live: free spectral filter (4 walls); raw-A propagation (degree blowup);
radius-2 operator destabilizing depth.
AHA moments: telescoping kills U; renormalization pulls radius to 1; WL = same operator.

================================================================================
PART 4 — answer.md: open with the method (GCN), problem, key-idea-with-why bullets (D5-D11),
final model (D12), WL reading (D13), training (D14), faithful code grounded in tkipf/gcn (TF)
+ pygcn (PyTorch). In-frame: no citation header, no arXiv links in code. Naming "tkipf/gcn"
for code grounding is ALLOWED in answer.md only (NOT in context.md scaffold).

NO HINDSIGHT anywhere: no GAT, GraphSAGE, oversmoothing, SGC, PPNP, anything post-2016.
