# node2vec synthesis (grounding notes)

Verified arXiv: 1607.00653 "node2vec: Scalable Feature Learning for Networks" (KDD 2016), Grover & Leskovec (Stanford). Canonical code: aditya-grover/node2vec (snap.stanford.edu/node2vec) — node2vec.py get_alias_edge confirms the α_pq biases exactly.

## Pain point / research question
- Supervised ML on networks needs feature vectors for nodes/edges. Hand-engineered features (centralities, Adamic-Adar) are tedious and don't generalize across tasks. Want to *learn* features unsupervised.
- Spectral/dim-reduction (PCA, IsoMap, Laplacian eigenmaps) require eigendecomposition → don't scale, and bake in assumptions (spectral clustering assumes homophily) → poor/inflexible on diverse networks.
- DeepWalk and LINE (the skip-gram-for-networks line) scale but use a *rigid* notion of neighborhood, so they're insensitive to which network pattern matters: homophily (community membership) vs structural equivalence (same role — hub, bridge — possibly far apart). Real networks mix both. Need a *flexible* neighborhood definition.

## Load-bearing ancestors (grounded)
- word2vec / Skip-gram (Mikolov 2013): learn word vectors that predict context words in a sliding window; distributional hypothesis (similar contexts → similar meaning). Optimized by SGD with negative sampling. Linear text → window neighborhood is natural.
- DeepWalk (Perozzi 2014): treat a network as a "document" by generating node sequences via *uniform* (1st-order) truncated random walks, then run Skip-gram on them. Gap: the uniform walk is a single rigid sampling strategy — no control over BFS-like vs DFS-like exploration.
- LINE (Tang 2015): optimize 1st- and 2nd-order proximity objectives directly (BFS-like, immediate neighbors). Gap: again rigid, context window of 1, no DFS-style macro exploration.
- BFS vs DFS as the two extremes of neighborhood sampling: BFS (immediate neighbors) → captures structural equivalence (microscopic local view, low variance, but explores tiny portion); DFS (increasing distance) → captures homophily/community (macro view), but high variance and far nodes less representative.

## The method (grounded; Section 3 + Algorithm 1 + code)
Objective: skip-gram for graphs. Learn f: V→R^d. For each source u, maximize log Pr(N_S(u) | f(u)). With conditional independence and softmax (symmetric, dot-product) assumptions:
  max_f Σ_u [ −log Z_u + Σ_{n_i∈N_S(u)} f(n_i)·f(u) ],  Z_u = Σ_v exp(f(u)·f(v)).
Z_u approximated by negative sampling; optimized by SGD. The novelty is NOT the objective (it's skip-gram) but the *neighborhood sampling* N_S(u).

**2nd-order biased random walk.** Walk of length l, c_0=u. P(c_i=x | c_{i-1}=v) = π_vx/Z if (v,x)∈E. Just traversed edge (t,v), now at v, deciding next x. Set unnormalized π_vx = α_pq(t,x)·w_vx, where
  α_pq(t,x) = 1/p if d_tx=0 (x==t, return to previous node)
            = 1   if d_tx=1 (x is a common neighbor of t and v)
            = 1/q if d_tx=2 (x is farther from t)
d_tx = shortest-path distance between t and x, ∈{0,1,2} (since v is adjacent to both t and x). Two params p,q necessary and sufficient. Walk is 2nd-order Markov (depends on t and v).

- **Return parameter p**: controls revisiting the just-visited node t. High p (>max(q,1)) → less likely to backtrack within 2 steps, avoids 2-hop redundancy, encourages exploration. Low p (<min(q,1)) → backtracks, keeps walk local near u.
- **In-out parameter q**: differentiates inward vs outward. q>1 → biased toward nodes close to t (within 1-hop locality) → BFS-like, local/microscopic, structural-equivalence affinity. q<1 → biased toward nodes farther from t → DFS-like, outward exploration, homophily/community affinity. (Within the walk framework, so not strictly increasing distance.)
- p,q smoothly interpolate BFS↔DFS, so node2vec generalizes DeepWalk (uniform walk = p=q=1).

**Algorithm.** LearnFeatures(G, d, r, l, k, p, q): precompute modified weights π → G'; for r iterations, for every node u, walk = node2vecWalk(G', u, l); then SGD with context size k over all walks. node2vecWalk: greedily extend [u] by alias-sampling neighbors per π. Three phases (preprocess π / simulate walks / SGD) sequential, each parallelizable, async.

**Efficiency.** Alias sampling → O(1) per sampled node after precompute. Space: O(|E|) for neighbors + O(a²|V|) for the 2nd-order interconnections (a = avg degree). Sample reuse: a walk of length l>k yields k context samples for l−k nodes (Markovian), effective complexity O(l/(k(l−k))) per sample. Start-node bias offset by r walks from every node.

**Edge features.** For node pair (u,v), binary operator ∘ on f(u),f(v) → g(u,v): Average (f_i(u)+f_i(v))/2; Hadamard f_i(u)·f_i(v); Weighted-L1 |f_i(u)−f_i(v)|; Weighted-L2 |f_i(u)−f_i(v)|². Defined for any pair (edge or non-edge) for link prediction. Hadamard works best empirically.

## Design-decision → why (grounded)
- Random walk (vs pure BFS/DFS): BFS/DFS need k|V| separate samplings; random walks reuse samples across source nodes, are O(1) to sample with alias method, and DFS-like exploration becomes tractable. BFS has low variance but tiny coverage; DFS high variance and far-node irrelevance. Biased walk gets a tunable middle.
- 2nd-order (depends on previous node t): a 1st-order walk can only bias by edge weight w_vx, which can't express "go back / stay local / venture out" relative to where you came from. Conditioning on t lets α distinguish return (d=0), stay-in-neighborhood (d=1), explore-outward (d=2). Two params suffice because d_tx∈{0,1,2}.
- α as multiplicative bias on edge weight: keeps weighted-graph behavior (π=w when p=q=1) and is precomputable.
- skip-gram objective unchanged: the contribution is the sampling, so reuse the proven, scalable word2vec machinery (negative sampling, SGD).

## Hyperparameters (grounded)
d=128, r=10 walks/node, l=80 walk length, context size k=10, 1 epoch, negative sampling. p,q tuned by grid {0.25, 0.5, 1, 2, 4} via 10-fold CV on 10% labeled data. Performance saturates around d≈100; r and l have high impact; k smaller impact. Sampling budget K = r·l·|V| held equal across baselines for fairness.

## Evaluation settings (pre-method, no outcomes)
Tasks: multi-label node classification (one-vs-rest logistic on learned features; Macro/Micro-F1) and link prediction (remove 50% edges, learn on rest; score candidate edges via edge feature g(u,v) + classifier; AUC). Datasets: BlogCatalog, PPI (protein-protein), Wikipedia word co-occurrence (POS), Facebook, arXiv collaboration, Erdos-Renyi for scalability. Baselines: DeepWalk, LINE, spectral clustering; for link prediction also Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment. Robustness: noisy/missing edges; scalability to 1M nodes.
