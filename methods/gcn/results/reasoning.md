Okay, let me start from the thing that actually bothers me, which is a concrete failure mode. I have one graph — a citation network, say — with N nodes, each carrying a sparse bag-of-words feature vector X_i, an adjacency matrix A, and labels for maybe twenty nodes per class. A few hundred labels out of tens of thousands of nodes. I need to label the rest, and I have to use both the words in each document and the citation links, because the links carry information the words don't: two papers can cite each other across a subfield boundary, and that edge tells me something even when their bags of words look completely different.

So how does everyone currently fold the graph in? They put it in the loss. The standard move is L = L_0 + λ·L_reg, where L_0 is the supervised loss on the labeled nodes and L_reg = (1/2)Σ_{i,j} A_{ij} ‖f(X_i) − f(X_j)‖² = f(X)ᵀ Δ f(X), with Δ = D − A the unnormalized Laplacian. Let me make sure I see why those two forms are equal, because that equality is the whole content of the prior. Expand f(X)ᵀΔf(X) = f(X)ᵀDf(X) − f(X)ᵀAf(X) = Σ_i d_i ‖f_i‖² − Σ_{i,j} A_{ij} f_iᵀf_j; and Σ_{i,j} A_{ij}‖f_i − f_j‖² = Σ_{i,j} A_{ij}(‖f_i‖² − 2f_iᵀf_j + ‖f_j‖²) = 2Σ_i d_i‖f_i‖² − 2Σ_{i,j}A_{ij}f_iᵀf_j (using Σ_j A_{ij} = d_i and symmetry), so the ordered double sum is exactly 2 f(X)ᵀΔf(X), and the half-sum is the quadratic form. That penalty is small precisely when every pair of adjacent nodes gets nearly the same output. It is literally encoding "an edge means same label." And f itself — the predictor — is a function of X alone; the graph only ever appears as a smoothness whip in the loss. That's two problems in one. First, the similarity assumption is a modeling cap: a citation edge might link two different-topic papers, and I'm forcing them toward the same prediction. Second, the structure never enters the model, so the architecture itself can't carry feature or gradient information from a labeled node, across edges, to an unlabeled one. Label propagation (Zhu et al. 2003) is the purest version of this — it solves for a harmonic function where each unlabeled node is the degree-weighted average of its neighbors, in closed form, but it doesn't even use X beyond the edges. Manifold regularization (Belkin et al. 2006) and deep semi-supervised embedding (Weston et al. 2012) are the same f(X)ᵀΔf(X) idea bolted onto richer predictors. Same cap.

Both of those problems point the same direction: the graph wants to be inside the predictor, not in the loss. If the predictor were f(X, A) — conditioned on the adjacency directly — then gradient from the few labeled nodes would flow through the architecture, along the edges, to the unlabeled ones, and I would not be forced to assume edges mean similarity; the model could decide for itself what to do with each edge. So the question I actually have to answer is what f(X, A) should look like. I want it to be a real neural network with stacked layers, because depth has paid off everywhere else, but "neural network that reads an adjacency matrix" is not yet a concrete architecture — I have to find the operation that consumes A.

What's already out there for "neural network on a graph"? Let me think through them, because I want to inherit, not reinvent. The recurrent graph-NN line (Gori 2005, Scarselli 2009) applies a contraction map until node states reach a fixed point — clunky, and the contraction requirement is a straitjacket; Li et al. 2016 loosen it with gated updates but it's still a recurrent thing, with the propagation run to convergence rather than a fixed shallow stack. Duvenaud et al. 2015 do a convolution-like propagation for molecules, but they learn a separate weight matrix per node degree — and my graphs have wide degree distributions, dozens of distinct degrees, so that's a parameter explosion and it won't transfer to a graph with different degrees. Atwood & Towsley 2016 is O(N²), dead for 20,000 nodes. Niepert 2016 needs me to impose a node ordering to feed a 1D CNN, which is arbitrary — there's no canonical order on graph nodes. None of these is a clean, single-weight-matrix-per-layer, scales-with-edges convolution. So I think the right place to look is the spectral line, where "convolution on a graph" has an actual definition rather than a hand-built aggregation rule.

But wait — why do I even need to go spectral to define a convolution? Why not just write down "slide a filter across the graph" directly? Because sliding is exactly what I can't do. An ordinary convolution is built on *translation*: take a little filter, shift it by one position, multiply-and-sum, shift again. On a regular grid "shift by one" is unambiguous. On a graph it is meaningless — node 7 has three neighbors, node 8 has nine hundred, there is no canonical "the next node over." Translation is undefined, so the spatial definition of convolution doesn't transfer. The only way I know to define convolution *without* translation is the convolution theorem: in ordinary signal processing, convolving two signals in the spatial domain equals multiplying them pointwise in the Fourier domain. If I can build a Fourier domain on the graph, I can define convolution as "transform, multiply by a filter, transform back," and never mention translation at all. The spectral route is the one that survives the loss of a shift operator.

So I need a graph Fourier basis. The object that supplies one is the symmetric normalized Laplacian L = I_N − D^{−1/2} A D^{−1/2}. Why this matrix and not, say, A itself? Because L is real and symmetric, so it diagonalizes with an *orthonormal* eigenbasis L = UΛUᵀ, U orthogonal, Λ diagonal and real — and an orthonormal basis is exactly what a Fourier basis must be (Parseval, invertibility by Uᵀ rather than an inverse I'd have to compute). The eigenvectors are the graph's Fourier modes: the eigenvector with eigenvalue near 0 varies slowly across edges (it's the would-be DC / smooth mode), the eigenvector with the largest eigenvalue oscillates hardest from node to node. The eigenvalues are the frequencies. The graph Fourier transform of a node signal x ∈ R^N is Uᵀx; the inverse is U(·). And the roughness of a signal is xᵀLx = Σ_{i,j}(1/2)A_{ij}(x_i/√d_i − x_j/√d_j)² ≥ 0, which is why L is positive semidefinite and why its spectrum starts at 0. One more fact I'll lean on hard later: for the *normalized* Laplacian the eigenvalues are bounded, λ ∈ [0, 2]. Quick check on the upper bound: D^{−1/2}AD^{−1/2} is similar to the random-walk matrix D^{−1}A, so the random-walk operator has spectral radius at most 1 while the similar symmetric matrix has real eigenvalues; those eigenvalues therefore lie in [−1, 1], and L = I − D^{−1/2}AD^{−1/2} has eigenvalues in [0, 2], with 2 attained on bipartite components. That boundedness is a gift the *unnormalized* Δ = D − A doesn't give me — Δ's top eigenvalue grows with the maximum degree — and it's the first hint that normalizing is not cosmetic.

Now, with a Fourier basis, define spectral convolution. A filter is anything diagonal in the spectral domain — multiply each Fourier coefficient by its own gain. With a free filter g_θ = diag(θ), θ ∈ R^N,

  g_θ ⋆ x = U g_θ Uᵀ x = U diag(θ) Uᵀ x.

This is exactly Bruna et al. 2014's spectral CNN. And I immediately hit a wall — four of them, actually. One: θ has N free parameters, one per eigenvalue. It's non-parametric, it doesn't localize in space (a generic diagonal in the spectral domain is a filter with global support in the node domain — flipping one Fourier coefficient touches every node), and it can't transfer to another graph because θ is defined relative to *this* graph's U. Two: I need U, which means an O(N³) eigendecomposition of L. Three: even with U in hand, applying U and Uᵀ to a vector is a dense O(N²) multiply. Four: there's no notion of "this filter looks at a node's neighborhood" — locality, the thing that makes ordinary CNN filters cheap and meaningful, is simply absent. On a graph with 20,000 nodes this is all dead on arrival. Free spectral filters are out.

So how do I get a filter that's a *function of the spectrum* without ever computing the spectrum? Stop treating g_θ as a free diagonal and instead write it as g_θ(Λ), a smooth function of the eigenvalues, then approximate that function by a polynomial. This is the Hammond et al. 2011 idea. Suppose

  g_{θ'}(Λ) ≈ Σ_{k=0}^{K} θ'_k T_k(Λ̃),

a polynomial of order K in the eigenvalues, where T_k are Chebyshev polynomials and Λ̃ = (2/λ_max)Λ − I_N rescales the eigenvalues from [0, λ_max] into [−1, 1]. Why Chebyshev specifically, and why rescale to [−1, 1]? Because Chebyshev polynomials are orthogonal and bounded (|T_k| ≤ 1) precisely on [−1, 1], which is where they give near-minimax, numerically stable approximations of smooth functions; feed them eigenvalues outside that interval and the recurrence T_k = 2yT_{k−1} − T_{k−2} blows up like y^k. So the rescaling Λ̃ isn't decoration — it's what keeps the polynomial expansion stable. λ_max is the largest eigenvalue of L (a single quantity, cheap to get with one sparse eigensolver call, no full decomposition).

Now push this back through the U's. (UΛUᵀ)^k = UΛ^k Uᵀ — the U's telescope: U Λ Uᵀ · U Λ Uᵀ = U Λ (UᵀU) Λ Uᵀ = U Λ² Uᵀ, and by induction any power, hence any polynomial. So

  g_{θ'} ⋆ x = U (Σ_k θ'_k T_k(Λ̃)) Uᵀ x = Σ_k θ'_k U T_k(Λ̃) Uᵀ x = Σ_k θ'_k T_k(L̃) x,

with L̃ = (2/λ_max)L − I_N. The U's vanished. I never form them. I only ever apply L̃ — sparse, a nonzero only where there's an edge — to a vector, K times, via the recurrence T_k(L̃)x = 2 L̃ T_{k−1}(L̃)x − T_{k−2}(L̃)x, T_0 = I, T_1 = L̃. That's K sparse matrix-vector products, cost O(K|E|), linear in edges, no eigendecomposition. And it's K-localized in exactly the right sense: a K-th order polynomial in L̃ has a nonzero at (i,j) only if there's a walk of length ≤ K from i to j, so the filter at a node reaches only its K-hop neighborhood. All four of Bruna's walls fall at once — parameters drop from N to K+1 coefficients, locality is back and is exactly K hops, no eigendecomposition, and θ' is graph-independent because it's coefficients of a polynomial in L, not coordinates in a fixed U. This is Defferrard et al. 2016's ChebNet, and it's a genuinely good convolution. I could stop here and stack ChebNet layers.

But let me look harder at what ChebNet costs me per layer. Each filter still has the K+1 free coefficients θ'_0…θ'_K, so with C input and F output channels that's K+1 weight matrices per layer. And here's the structural thing that bugs me: in ChebNet, K does double duty. It sets the filter's expressiveness (how wild a function of the spectrum it can be) *and* it sets the receptive field (how many hops the layer reaches). Those are tangled together. On a graph with a wide degree distribution, a K=3 filter centered on a high-degree hub reaches the hub's entire 3-hop neighborhood — which can be a huge fraction of the graph — and a filter with that much reach and that many parameters can overfit the idiosyncratic local structure around that one hub. I want to *untangle* expressiveness from receptive field.

And there's a clean way to untangle them, because I'm going to *stack* layers. Receptive field composes: if one layer reaches one hop and I stack k of them, the composition reaches k hops. So depth can supply the receptive field that K supplied in ChebNet — and depth is the move that's worked everywhere else (He et al. 2015 made very deep nets trainable). Meanwhile, the *expressiveness* I lose by making each layer's spectral filter low-order, I get back from the point-wise nonlinearities between layers: a stack of linear-spectral-filter-then-nonlinearity is not itself a low-order polynomial filter, it's a rich nonlinear function on the graph, and crucially one I'm *not* forced to write as an explicit Chebyshev expansion. So: take K as small as it goes — K = 1 — making each layer's spectral filter *linear* in L, and let depth and nonlinearity do the rest. There's a regularization argument folded in here too: K=1 means two parameters per filter instead of K+1, and on graphs where a single layer's 1-hop neighborhood is already large (hubs), fewer parameters per layer is exactly what fights overfitting the local neighborhood.

Set K = 1. The expansion becomes g_{θ'} ⋆ x ≈ θ'_0 T_0(L̃) x + θ'_1 T_1(L̃) x = θ'_0 x + θ'_1 L̃ x. Now L̃ = (2/λ_max)L − I_N still carries λ_max, and I'd rather not run an eigensolver per graph just to get the top eigenvalue. Cheap approximation: I just established λ ∈ [0, 2], so λ_max ≈ 2 is a reasonable guess, and the trainable weights downstream will simply rescale themselves to absorb whatever error the approximation introduces. The scale of the operator is a free gauge that gradient descent fixes. Set λ_max = 2: then L̃ = L − I_N = (I_N − D^{−1/2}AD^{−1/2}) − I_N = −D^{−1/2}AD^{−1/2}. So

  g_{θ'} ⋆ x ≈ θ'_0 x + θ'_1 (L − I_N) x = θ'_0 x − θ'_1 D^{−1/2} A D^{−1/2} x,

two free parameters: θ'_0 multiplying the node's own signal, θ'_1 multiplying the normalized sum over neighbors. That's "keep yourself, mix in your neighbors," with two knobs, and the filter is shared over the whole graph — no per-node, no per-degree matrices, exactly what Duvenaud's degree-specific weights failed to give me.

Now let me pause on that neighbor-mixing operator D^{−1/2}AD^{−1/2}, because I have a choice of normalization and it matters. The obvious alternative is the random-walk normalization D^{−1}A, whose row i is (1/d_i)Σ_j A_{ij} — literally the *average* of node i's neighbors. Why not just average? Two reasons. First, the spectral story I built everything on requires a *symmetric* operator: D^{−1}A is not symmetric, it does not have an orthonormal eigenbasis, and the Fourier picture (real spectrum, U orthogonal, the convolution theorem) quietly breaks. D^{−1/2}AD^{−1/2} is symmetric — it's I − L, it shares L's orthonormal eigenbasis, the whole derivation stays coherent. Second, even setting the spectral story aside, mere averaging is a weaker operation: with D^{−1}A every node just sees the mean of its neighbors, and the symmetric form D^{−1/2}AD^{−1/2} — which weights the edge (i,j) by 1/√(d_i d_j) rather than 1/d_i — does *not* reduce to averaging, so it produces richer node-to-node dynamics. It down-weights edges to high-degree neighbors more aggressively (a 1/√d_j factor on the far end), which is sensible: a citation to a paper that's cited by everyone carries less information than a citation to an obscure one. So symmetric normalization, both for spectral coherence and for being more than an average.

And one more sanity check on *why I normalize at all* rather than using raw A. If I propagated with the bare adjacency, x → Ax sums neighbors with no rescaling, and the operator's eigenvalues scale with the degrees (the top one grows like the max degree). Applied once it rescales feature magnitudes by something degree-dependent; applied layer after layer it blows the activations up or crushes them, node-by-node, depending on local degree. Normalization is what keeps the feature scale stable across propagation. That's the same instinct that's about to bite me one level deeper.

Can I make it leaner still? Two parameters per filter, doubled across input/output channels, two matmuls per layer. To fight overfitting and cut operations I'll tie them: set θ = θ'_0 = −θ'_1, a single parameter. Then

  g_θ ⋆ x ≈ θ (I_N + D^{−1/2} A D^{−1/2}) x.

One parameter, one operator I_N + D^{−1/2}AD^{−1/2}: "yourself plus your normalized neighbors," a self-loop of weight one added to the normalized adjacency. It's the leanest thing that still both keeps the node and mixes neighbors. But now I have to check stability, because I'm going to apply this operator once per layer and stack many layers — spectrally that's close to a power iteration with this matrix, so its largest eigenvalue governs whether activations grow or decay under composition. What is its spectrum? L = I − D^{−1/2}AD^{−1/2} has eigenvalues in [0, 2], so D^{−1/2}AD^{−1/2} = I − L has eigenvalues in [−1, 1], so I + D^{−1/2}AD^{−1/2} = 2I − L has eigenvalues in [0, 2]. The top eigenvalue is ≈ 2, and that's exactly the wrong number.

Let me make this concrete rather than trust the algebra blindly, because the conclusion — that depth becomes unusable — is strong enough that I want to see it. Take a small graph: a triangle on nodes {0,1,2} with a tail node 3 hanging off node 2, so degrees are (2,2,3,1). Form S = D^{−1/2}AD^{−1/2} and op = I + S. Its eigenvalues come out [0.271, 0.500, 1.229, 2.000] — the upper bound is attained, exactly 2, not merely "≈ 2." Now take a random signal x and apply op repeatedly, tracking ‖op^L x‖: L=1 gives 0.92, L=4 gives 6.8, L=8 gives 109. That is the 2^L growth I feared, made visible — eight layers already amplifies by ~120×, and the components living near the small eigenvalues (0.27, 0.5) get crushed relative to the top mode at the same time. So the operator I landed on is numerically hostile to exactly the depth I argued is my whole strategy. Wall.

Let me think about *why* the radius is 2 and how to pull it back to 1. The "+1" came from the identity I_N — I bolted a self-loop of weight one onto the graph *after* normalizing the neighbor part. So the self-loop's mass isn't accounted for in the degree normalization: every node now receives its neighbors' normalized contributions *plus* a full unit of its own, but the D^{−1/2} factors were computed from the degrees of the self-loop-free graph. The node is over-weighted relative to how it's normalized, and that excess is exactly what pushes the top eigenvalue from 1 up to 2. The fix that suggests itself: fold the self-loop into the adjacency *before* normalizing, so the degrees used for normalization already include it. Define Ã = A + I_N — add a self-loop to every node — and D̃_ii = Σ_j Ã_ij = d_i + 1 — the degree in the self-looped graph — and symmetric-normalize *that*:

  I_N + D^{−1/2}AD^{−1/2}  →  D̃^{−1/2} Ã D̃^{−1/2}.

Call it the renormalization trick. The reason I expect this to help: D̃^{−1/2}ÃD̃^{−1/2} is *itself* the symmetric normalized adjacency of a graph — the graph-with-self-loops — so by the very same argument as before its eigenvalues should live in [−1, 1] (it equals I − L̃_self where L̃_self = I − D̃^{−1/2}ÃD̃^{−1/2} is a proper normalized Laplacian, spectrum in [0, 2]), which would put the spectral radius at 1. But this is precisely the kind of "by the same argument" step I just caught the bound-at-2 case quietly violating in the bad direction (there the +1 fell *outside* the normalization), so I should not trust it until I see it. Same triangle-with-a-tail graph: now degrees become d̃ = (3,3,4,2) after adding self-loops, and the eigenvalues of D̃^{−1/2}ÃD̃^{−1/2} come out [−0.148, 0.000, 0.564, 1.000]. Spectral radius exactly 1 — the top eigenvalue sits at 1, not 2. Applying it eight times to that same random signal: the norm goes 0.43 → 0.41 → 0.41, dead flat instead of exploding to 109. So folding the self-loop in before normalizing is what pulls the radius from 2 down to 1, and stacking is now safe.

And it kept the spirit of "yourself plus your normalized neighbors" — the self-loop is still there, just normalized *consistently* with the rest, so the node and its neighbors sit on the same footing. Let me read off the entries to be sure that's what happened: on this graph Â[0,1] = 0.333, and 1/√(d̃_0 d̃_1) = 1/√(3·3) = 0.333 — match; the self-loop entry Â[2,2] = 0.25, and 1/√(d̃_2 d̃_2) = 1/d̃_2 = 1/4 = 0.25 — match. So every entry of Â is 1/√(d̃_i d̃_j) over the self-looped graph, the self-loop included, all from the same D̃. This is the operator I'll commit to. Name it Â = D̃^{−1/2}ÃD̃^{−1/2}.

Now generalize from a scalar signal x to the real case: X ∈ R^{N×C} with C input channels per node, and I want F output feature maps. The single parameter θ becomes a weight matrix Θ ∈ R^{C×F}, and the convolved output is

  Z = Â X Θ = D̃^{−1/2} Ã D̃^{−1/2} X Θ,  Z ∈ R^{N×F}.

Cost: Â is sparse with O(|E|) nonzeros, ÂX is a sparse-times-dense product — implement it as exactly that, never densify Â — so the whole layer is O(|E|FC), linear in edges. Scalability requirement met. Stick a nonlinearity on it and stack: the layer-wise propagation rule is

  H^{(l+1)} = σ( Â H^{(l)} W^{(l)} ),  H^{(0)} = X,

σ = ReLU, W^{(l)} the per-layer weights. This is exactly the K=1 Chebyshev filter, stacked: each layer is one localized propagation, and the depth supplies the receptive field. For the classifier I'll use two layers, which after composition gives each node a 2-hop receptive field. Precompute Â once (just a preprocessing step on A), then

  Z = f(X, A) = softmax( Â · ReLU( Â X W^{(0)} ) · W^{(1)} ),

with W^{(0)} ∈ R^{C×H} mapping C features to H hidden units and W^{(1)} ∈ R^{H×F} mapping to F classes, softmax row-wise. The loss is cross-entropy over the labeled nodes only — and crucially *only* the labeled ones, even though Z is computed for all N nodes at once:

  L = − Σ_{l∈Y_L} Σ_{f=1}^{F} Y_{lf} ln Z_{lf}.

The unlabeled nodes still participate — their features flow through Â into the labeled nodes' representations and the gradient flows back out to them — but they contribute no supervised term. That's the whole semi-supervised mechanism, and the graph is now entirely inside the model: no λ, no f(X)ᵀΔf(X), no edges-mean-similarity assumption. Exactly the resolution I wanted at the start.

Let me write the propagation rule for one node, in vector form, by reading off what Â H W does at node i. Â_ij = 1/√(d̃_i d̃_j) for i, j adjacent in the self-looped graph. So

  h_i^{(l+1)} = σ( Σ_{j ∈ N_i ∪ {i}} (1/c_ij) h_j^{(l)} W^{(l)} ),  with c_ij = √(d̃_i d̃_j).

Compare that to the 1-dimensional Weisfeiler–Lehman graph-isomorphism algorithm, the classical thing for assigning canonical node colorings: WL-1 repeats h_i ← hash( h_i, multiset{h_j : j ∈ N_i} ) until the coloring stabilizes — each node hashes its own current color together with the aggregate of its neighbors' colors, over and over, and two graphs that produce different stable colorings are provably non-isomorphic. My layer is *that*, with two substitutions: replace the hash with a differentiable, parameterized, normalized map σ(Σ_j (1/c_ij) h_j W), and use the normalization constant c_ij = √(d̃_i d̃_j) that the entry-check above already pinned down — Â_ij was exactly 1/√(d̃_i d̃_j), so 1/c_ij is precisely the weight my operator puts on edge (i,j). WL-1 uses an unweighted aggregate (c_ij = 1), which is the raw-adjacency aggregation I already rejected as degree-unstable; my operator is the same neighborhood-aggregation skeleton with the degree-symmetric weight swapped in for the constant 1. So the same operator I built from the spectral side — Chebyshev, K=1, λ_max≈2, tie the coefficients, renormalize — reads, node by node, as a smooth trainable version of one 1-WL relabeling step: a network that does WL-style neighborhood aggregation but *learns* what to aggregate instead of hashing.

That second reading suggests a check I can actually run before trusting the supervised classifier. If the propagation operator alone really carries graph structure, then even with no features (X = I_N) and *untrained* random weights, nodes that play the same structural role should land near each other after a couple of Â-propagation layers; if they don't, the operator isn't doing what I think and stacking it under a classifier won't save me. Concretely: two triangles {0,1,2} and {3,4,5} joined by a single bridge edge (2,3) — two obvious communities. Set X = I_6, draw W^{(0)} (6×8) and W^{(1)} (8×2) at random, push through H^{(1)} = tanh(Â X W^{(0)}), Z = Â H^{(1)} W^{(1)}, and look at the 2-D rows. Community-A nodes come out clustered (0 and 1 essentially identical at (1.03, 0.63), node 2 nearby) and community-B nodes cluster separately around (0.5–0.7, 1.15); the mean within-community distance is 0.16 against 0.66 across — a 4× gap — with zero training and no features. So the operator itself organizes nodes by their structural role; the supervised weights only have to sharpen what propagation already exposes. Two unrelated motivations — spectral convolution and WL relabeling — landed on the same operator, and the operator passes the structure-preservation check on its own.

Let me also pin down why each simplification is allowed to stand, since I made several and any one could be the weak link. Going spectral at all: forced, because translation is undefined on a graph and the convolution theorem is the only translation-free definition of convolution. Symmetric normalized Laplacian as the basis: forced, because it's the symmetric PSD operator with an orthonormal eigenbasis and a bounded [0,2] spectrum. Chebyshev over a free diagonal: kills O(N) params, gives K-locality, removes the eigendecomposition, transfers across graphs. Rescaling to [−1,1]: required for the Chebyshev recurrence to stay bounded. K=1: untangles expressiveness from receptive field — depth supplies the hops, nonlinearity supplies the expressiveness — and cuts parameters so wide-degree neighborhoods don't overfit. λ_max≈2: the [0,2] bound makes 2 a safe guess and trainable weights absorb the error. Symmetric vs random-walk normalization: symmetric keeps the operator in the spectral framework and is more than a neighbor-average. Tying θ'_0 = −θ'_1: one parameter, fewer matmuls, more regularization. The renormalization trick is the one that's load-bearing for *depth* — and it's the one I bothered to check numerically rather than wave through, because the radius-2 version had already shown me that a self-loop added in the wrong place ruins the spectrum: without renormalization the operator amplified a signal ~120× over eight applications, with it the norm stayed flat and the radius was exactly 1, so I can stack freely (and if I go very deep, residual connections H^{(l+1)} = σ(Â H^{(l)} W^{(l)}) + H^{(l)} carry the previous layer through, the same trick that made deep nets trainable elsewhere). Each step is a deliberate move from rich-but-expensive ChebNet toward the cheapest operator that still propagates and still stacks.

Now to code, and it should slot straight into the bare harness I started with: the sparse graph-support slot becomes the Â operator I just derived; the layer slot becomes dropout, a feature projection, sparse propagation, summation over supports, and an activation; the masked full-batch loss is untouched. Everything hinges on three things: build Â once with sparse SciPy ops from the raw A, make each layer a sparse-dense matmul of the graph operator against a dense activation times a weight, and train full-batch with the masked loss so all-node outputs supervise on labeled-node targets only.

```python
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

def sparse_to_tuple(sparse_mx):
    """Convert a scipy sparse matrix, or a list of them, to TensorFlow sparse tuples."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        return coords, mx.data, mx.shape
    return [to_tuple(mx) for mx in sparse_mx] if isinstance(sparse_mx, list) else to_tuple(sparse_mx)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    return sparse_to_tuple(sp.diags(r_inv).dot(features))

def glorot(shape, name=None):
    """Glorot & Bengio uniform initialization."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range,
                                dtype=tf.float32)
    return tf.Variable(initial, name=name)

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for TensorFlow SparseTensor inputs."""
    random_tensor = keep_prob + tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    dropped = tf.sparse_retain(x, dropout_mask)
    return dropped * (1. / keep_prob)

def normalize_adj(adj):
    """Symmetric normalization D^{-1/2} A D^{-1/2}, kept sparse throughout.
    Symmetric (not random-walk D^{-1}A) so the operator stays in the spectral
    framework and is more than a neighbor-average; edge (i,j) gets 1/sqrt(d_i d_j)."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))                  # degrees
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^{-1/2} diagonal
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat).transpose().dot(d_mat).tocoo()

def preprocess_adj(adj):
    # Renormalization: add self-loops first, then normalize with the new degrees.
    return sparse_to_tuple(normalize_adj(adj + sp.eye(adj.shape[0])))

def dot(x, y, sparse=False):
    return tf.sparse_tensor_dense_matmul(x, y) if sparse else tf.matmul(x, y)

class GraphLayer:
    """One layer: act(sum_s support_s @ (dropout(x) @ W_s))."""
    def __init__(self, input_dim, output_dim, support, act=tf.nn.relu,
                 dropout=0., sparse_inputs=False, num_features_nonzero=None):
        self.support = support
        self.act, self.dropout = act, dropout
        self.sparse_inputs = sparse_inputs
        self.num_features_nonzero = num_features_nonzero
        self.weights = [glorot([input_dim, output_dim], name='weights_%d' % i)
                        for i in range(len(support))]

    def __call__(self, x):
        x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero) if self.sparse_inputs \
            else tf.nn.dropout(x, 1 - self.dropout)
        out = []
        for s, W in zip(self.support, self.weights):
            xw = dot(x, W, sparse=self.sparse_inputs)  # X W
            out.append(dot(s, xw, sparse=True))        # Â (X W)
        return self.act(tf.add_n(out))

class GraphModel:
    def __init__(self, placeholders, input_dim, hidden, num_classes):
        support = placeholders['support']              # one sparse placeholder for Â
        self.inputs = placeholders['features']
        dropout = placeholders['dropout']
        self.layers = [
            GraphLayer(input_dim, hidden, support, act=tf.nn.relu, dropout=dropout,
                       sparse_inputs=True,
                       num_features_nonzero=placeholders['num_features_nonzero']),
            GraphLayer(hidden, num_classes, support, act=lambda z: z, dropout=dropout),
        ]
        h = self.layers[0](self.inputs)                # ReLU(Â X W0)
        self.outputs = self.layers[1](h)               # Â h W1, logits

def masked_softmax_cross_entropy(logits, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mask = tf.cast(mask, tf.float32); mask /= tf.reduce_mean(mask)
    return tf.reduce_mean(loss * mask)

features = preprocess_features(features)               # sparse row-normalized X
support = [preprocess_adj(adj)]                        # sparse tuple for Â
num_supports = len(support)

placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32,
                                      shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
}

model = GraphModel(placeholders, input_dim=features[2][1],
                   hidden=16, num_classes=y_train.shape[1])
loss = masked_softmax_cross_entropy(model.outputs,
                                    placeholders['labels'],
                                    placeholders['labels_mask'])
loss += 5e-4 * tf.add_n([tf.nn.l2_loss(W) for W in model.layers[0].weights])
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

feed_dict = {
    placeholders['features']: features,
    placeholders['labels']: y_train,
    placeholders['labels_mask']: train_mask,
    placeholders['num_features_nonzero']: features[1].shape,
    placeholders['dropout']: 0.5,
}
feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
_, train_loss = sess.run([train_op, loss], feed_dict=feed_dict)
# Repeat the full-batch step for up to 200 epochs and early-stop on validation loss.
```
