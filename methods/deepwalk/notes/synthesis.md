# deepwalk synthesis (grounding notes)

Verified arXiv: 1403.6652 "DeepWalk: Online Learning of Social Representations" (KDD 2014), Perozzi, Al-Rfou, Skiena (Stony Brook). Canonical code: phanein/deepwalk — graph.py random_walk (uniform), __main__.py uses gensim Word2Vec(sg=1, hs=1) = skip-gram + hierarchical softmax.

## Pain point / research question
- Network classification (multi-label node classification): predict labels of vertices. Traditional: relational classifiers with iterative approximate inference (ICA, Gibbs, label relaxation) on a Markov network — combine labels + edges. Sparsity of graph adjacency makes statistical generalization hard.
- Want: learn *latent* low-dimensional, continuous vertex representations (social representations) from network structure *alone*, unsupervised, that encode community membership, then feed to any simple classifier. Desired properties: adaptable (online, handle small graph changes), community-aware (distance ≈ social similarity), low-dimensional, continuous.

## Load-bearing ancestors (grounded)
- Neural language models / word2vec Skip-gram (Mikolov 2013): learn word vectors; turn the prediction around — use one word to predict its context window (both sides, order-independent). Objective: minimize −log Pr({w_{i−w}..w_{i+w}} \ {w_i} | Φ(w_i)). Optimized by SGD. This is the engine DeepWalk reuses.
- Hierarchical softmax (Morin & Bengio 2005; Mnih & Hinton 2009): to avoid the O(|V|) partition function, put vocabulary on the leaves of a binary tree; Pr(u_k|Φ(v_j)) = Π over tree path = product of binary decisions, each a logistic classifier at an internal node. O(log|V|) per prediction. Huffman coding gives frequent vertices shorter paths.
- Truncated random walks: short random walks capture local community structure (connection to local-community detection / output-sensitive algorithms, Spielman-Teng). The basic tool for extracting local structure.
- Relational classification baselines (Tang & Liu SocDim: modularity / spectral "social dimensions"): prior latent-representation methods for nodes; global view; weaker when labels sparse.

## The unifying observation (the aha)
Power-law connection. In a scale-free (power-law degree) graph, the frequency at which vertices appear in short random walks *also* follows a power law. Word frequency in natural language follows a power law (Zipf). Same statistical shape → the neural-language-model machinery built for the power-law symbol distribution of language can be re-purposed to model community structure in networks. So: treat short random walks as "sentences," vertices as "words," and run a language model.

## The method (grounded; Section 4 + Algorithms 1 & 2)
- Φ: V → R^{|V|×d}, the embedding matrix (free parameters), uniform-initialized.
- Build a binary tree T over V (for hierarchical softmax).
- For γ passes: shuffle V; for each vertex v_i: W_{v_i} = RandomWalk(G, v_i, t) (uniform walk of length t, each step picks a uniform random neighbor of the last vertex; root sampled at v_i); then SkipGram(Φ, W_{v_i}, w).
- SkipGram: for each v_j in the walk, for each context u_k in window [j−w, j+w]: J(Φ) = −log Pr(u_k | Φ(v_j)); Φ ← Φ − α ∂J/∂Φ.
- Hierarchical softmax: Pr(u_k|Φ(v_j)) = Π_{l=1}^{⌈log|V|⌉} Pr(b_l | Φ(v_j)) over the root→leaf path (b_0=root, b_{⌈log|V|⌉}=u_k); each Pr(b_l|·) is a binary logistic classifier at b_l's parent. O(log|V|) vs O(|V|).
- Optimization: SGD, backprop; learning rate α starts at 2.5% and decays linearly with the number of vertices seen. Parameters {Φ, T}, each O(d|V|). Complexity O(dwL log|V|), L = γt.
- Skip-gram relaxations adopted: predict context from center (not center from context); both-sides context; order-independent. Order-independence matches the unordered "nearness" of random-walk neighborhoods.
- Online/streaming: only needs a sample from the walk distribution; no full graph needed; trivially parallelizable (asynchronous SGD, Hogwild-style); incremental updates for small graph changes.

## Design-decision → why (grounded)
- Random walks as the sampling tool: short walks encode local community info; sublinear-time local-structure connection; allow online updates without global recomputation; turn the graph into language-like sequences.
- Uniform (1st-order) walk: simplest; captures local neighborhood; (no return/in-out bias here — that flexibility is absent, which is the rigidity later methods react to). Optional restarts tried, no advantage.
- Skip-gram (not full sequence likelihood Pr(v_i | v_1..v_{i−1})): the full conditional is infeasible as walk length grows; skip-gram's order-independent windowed context is tractable and matches the unordered neighborhood notion.
- Hierarchical softmax (not flat softmax / logistic over |V|): |V| can be millions → O(|V|) normalization infeasible; tree → O(log|V|). Huffman coding (shorter paths for frequent vertices) further speeds it because vertex frequencies are power-law (frequent vertices accessed often).
- γ passes with shuffle: offset start-vertex bias; shuffling speeds SGD convergence.
- Linearly decaying lr from 2.5%: standard SGD annealing.

## Hyperparameters (grounded)
Reported best: γ=80 walks/vertex, window w=10, d=128, walk length t=40 (the parameter-sensitivity study fixes w=10, t=40 and varies d, γ; benefit saturates by γ≈30, d≈ a few hundred). Canonical code defaults: number-walks 10, walk-length 40, representation-size 64, window 10, gensim Word2Vec(sg=1, hs=1, min_count=0). lr 2.5% → linear decay.

## Evaluation settings (pre-method, no outcomes)
Task: multi-label vertex classification. Datasets: BlogCatalog, Flickr, YouTube (social networks). Protocol: learn embeddings unsupervised; train a one-vs-rest logistic regression on a fraction T_R of labeled nodes; report Macro-F1 / Micro-F1 across training ratios. Baselines: SpectralClustering, Modularity (SocDim), EdgeCluster, wvRN, Majority. Robustness studied vs labeled-data fraction and sparsity; scalability to web-scale (YouTube) via parallel implementation.
