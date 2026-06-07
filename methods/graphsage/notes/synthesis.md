# GraphSAGE — Synthesis (pre–Phase-2 understanding)

## The pain point (research question)
Production graphs (Reddit posts, YouTube videos, citation graphs, PPI graphs for new
organisms) are *evolving*: new nodes / new entire graphs appear constantly after training.
We need embeddings for nodes that were **never seen during training**, and to do so *cheaply*
(no re-optimization per new node). Existing node-embedding methods all assume a single fixed
graph with all nodes present at train time — they are **transductive**. Goal: an **inductive**
embedding generator that, given a node's features + local neighborhood, emits a useful
embedding for any node, including ones (and graphs) absent at train time.

Why hard: generalizing to unseen nodes means "aligning" a newly observed subgraph to an
embedding space already optimized on other nodes; the model must recognize *structural
properties* of a neighborhood (local role + global position) from features alone.

## Load-bearing ancestors (verified against primary text)

1. **DeepWalk (Perozzi et al. 2014) / node2vec (Grover & Leskovec 2016) / LINE (Tang 2015).**
   - Core: sample random walks, treat node sequences like sentences, optimize a SkipGram /
     negative-sampling objective so that co-occurring nodes have similar dot products. The
     trainable object is an **embedding lookup table** Z ∈ R^{|V|×d}: one free vector *per node*.
   - Levy & Goldberg (2014) view: this is implicit matrix factorization, Z Zᵀ ≈ M where M holds
     random-walk co-occurrence statistics.
   - Limitation #1 (transductive): the parameters ARE the per-node vectors. A new node has no
     row in Z → you must run more SGD to fit it; cannot generalize to unseen nodes for free.
     Paper reports DeepWalk is ~100–500× slower at test time because of this re-optimization.
   - Limitation #2 (no cross-graph transfer / drift): objective depends only on dot products
     zᵢᵀzⱼ, which is invariant to any orthogonal rotation Q (ZQᵀQZᵀ = ZZᵀ). So embeddings of two
     separately-trained graphs are arbitrarily rotated relative to each other → a classifier
     trained on graph A's embeddings is meaningless on graph B's. Also makes streaming re-training
     drift. (Appendix C derivation; load-bearing for "why we can't just retrain DeepWalk".)
   - Limitation #3: ignores node features entirely.

2. **Planetoid-I (Yang et al. 2016)** — one inductive exception, but it uses graph structure
   only as *training-time regularization*, not at inference; doesn't use neighborhood structure
   to produce the embedding of a new node. So it doesn't really exploit the graph at test time.

3. **GCN (Kipf & Welling 2017).** The closest relative.
   - Propagation rule: H^{(l+1)} = σ( D̂^{-1/2} Â D̂^{-1/2} H^{(l)} W^{(l)} ), Â = A + I (self-loops),
     D̂ = degree of Â. Derived as a first-order (Chebyshev K=1) approximation of a spectral graph
     convolution, with a renormalization trick to keep eigenvalues in a stable range.
   - Key advance over DeepWalk: parameters are *shared weight matrices* W^{(l)} acting on
     **features**, not per-node vectors. So GCN already "learns a function," in spirit.
   - Limitation (the one GraphSAGE attacks): the rule is written as a **full-graph matrix
     multiply** using the entire (normalized) adjacency Â. Training is **transductive** — it needs
     the whole graph Laplacian / adjacency in memory, all nodes participate in every forward pass,
     and the published algorithm trains all node representations simultaneously. It was only ever
     applied to a fixed graph in a semi-supervised setting. No minibatch over nodes, no notion of
     "generate an embedding for a brand-new node from just its neighborhood," and O(|V|)-sized
     dense operations don't scale to graphs with hundreds of thousands of nodes or to unseen graphs.
   - Also: GCN's aggregation is a *fixed* weighted mean — a single linear convolution; limited
     aggregation expressiveness.

4. **Graph neural nets / GGS-NN / structure2vec (Scarselli 2009, Li 2015, Dai 2016)** and
   **graph kernels (Weisfeiler–Lehman, Shervashidze 2011)**: conceptually inspiring (iterative
   neighborhood message passing; WL refinement), but aimed at *whole-graph* classification, not
   per-node inductive embeddings.

5. **PointNet / deep sets (Qi et al. 2017)**: a symmetric (permutation-invariant) function over an
   unordered point set = max-pool over per-point MLP outputs, with a universal-approximation
   guarantee for Hausdorff-continuous symmetric set functions. This is the template for the pool
   aggregator and the engine of the theory.
   **LSTM (Hochreiter & Schmidhuber 1997)** and **Hornik (1991)** universal approximation are also
   used.

## The chain of reasoning (problem → method)

- Reframe the unit of learning: instead of optimizing a vector *per node* (transductive by
  construction), optimize a small set of **functions** (aggregators + shared weight matrices) that
  *map a node's features + neighborhood → embedding*. Then a never-seen node just runs the
  functions. This is the single decisive move; everything else is making it work.

- GCN already shares weights over features, so it points the way — but its full-graph
  Â-multiply is transductive and unscalable. Rewrite the convolution **node-locally**: each node
  aggregates its immediate neighbors' previous-layer representations, then combines with its own.
  Stacking K such layers gives K-hop reach (depth = search radius), exactly like GCN's depth but
  expressed as a local recursion that needs only a node's K-hop neighborhood — not the whole graph.

- Forward pass (Algorithm 1): h⁰_v = x_v; for k=1..K:
  h^k_{N(v)} = AGGREGATE_k({h^{k-1}_u : u∈N(v)});
  h^k_v = σ(W^k · CONCAT(h^{k-1}_v, h^k_{N(v)})); then ℓ2-normalize h^k_v. Output z_v = h^K_v.

- **Concatenate self + neighbor (skip connection), don't fold self into the mean.** GCN's rule
  averages self and neighbors together through one W. Keeping h^{k-1}_v explicit and concatenating
  it preserves the node's own signal across depths (a skip connection between search depths). The
  paper reports this concatenation gives significant gains; the GCN-style variant (no concat) is
  exactly the ablation that does worse.

- **ℓ2-normalize per layer.** Keeps representations on the unit sphere so scale doesn't blow up
  across K layers and so the unsupervised dot-product loss is comparing directions; stabilizes
  training.

- **Aggregators must be permutation-invariant** (a node's neighbors are an unordered set). Three
  candidates:
  - *Mean*: elementwise mean of neighbor vectors. Folding self into the mean (no concat, one shared
    W) recovers an inductive variant of GCN ("GraphSAGE-GCN"), differing from Kipf's exact rule
    only by a normalization constant. Cheap, but a fixed linear aggregator — limited capacity.
  - *Pool* (chosen as best): each neighbor → single-layer MLP σ(W_pool h_u + b), then **elementwise
    max** over neighbors. Symmetric, trainable, high capacity; max lets the model pick out distinct
    salient features of the neighbor set. Inspired by PointNet. Mean-pool vs max-pool: no
    significant empirical difference, max-pool kept.
  - *LSTM*: larger capacity, but NOT permutation-invariant (sequential). Adapt by feeding a *random
    permutation* of neighbors each time so the LSTM is trained over random orderings. Works
    surprisingly well; slowest (~2× pool); roughly tied with pool in accuracy.

- **Neighborhood sampling for fixed batch cost.** Real graphs have hub nodes; using the *full*
  neighbor set makes per-batch memory/time unpredictable, worst case O(|V|). So define N(v) as a
  **fixed-size uniform sample** of neighbors, drawn fresh per layer k. Per-batch cost becomes
  O(∏_{i=1}^K S_i), constant. Found K=2 with S₁·S₂ ≤ 500 (S₁=25, S₂=10) works well; K=2 beats K=1
  by ~10–15%, K>2 gives ~0–5% for 10–100× cost. Sample with replacement when degree < sample size.

- **Minibatch (Algorithm 2): sample backward, compute forward.** To avoid touching the whole
  graph, start from the batch B of target nodes and expand *down* through K hops of sampled
  neighbors to get the support sets B^K ⊇ B^{K-1} ⊇ … ⊇ B^0, then run the forward recursion *up*
  on exactly those nodes. From the target's view, K=2 with sizes S₁,S₂ means S₂ immediate
  neighbors and S₁·S₂ two-hop neighbors.

- **Unsupervised graph-based loss (for task-agnostic embeddings).** Want nearby nodes similar,
  distant nodes distinct, but with NO per-node embedding to optimize — the loss must act on the
  *outputs* z_u of the functions:
  J(z_u) = −log σ(z_uᵀ z_v) − Q · E_{v_n∼P_n} log σ(−z_uᵀ z_{v_n}),
  where v co-occurs with u on a fixed-length random walk, P_n is the negative-sampling
  distribution (degree^{0.75}), Q is #negatives. Same SkipGram/neg-sampling shape as DeepWalk, but
  the z's are *generated from features+neighborhood*, not looked up. Can be swapped for a supervised
  cross-entropy loss on labels.

- **Why this can capture structure despite being feature-based (theory).** Set K=|V|, W=identity,
  hash aggregator, no nonlinearity → Algorithm 1 *is* the Weisfeiler–Lehman vertex-refinement
  isomorphism test. GraphSAGE is a continuous, trainable approximation of WL. Theorem 1: if all
  node features are pairwise > C apart, then for any ε there is a parameter setting such that after
  K=4 iterations the pool-based GraphSAGE approximates every node's clustering coefficient to within
  ε. Proof chain: Lemma 1 builds a continuous bump function g(x)=Σ_v 3|D|ε/(b d_v²+1) − 2ε,
  b=(3|D|−1)/C², that is >0 at any anchor point and ≤−ε beyond distance C; Lemma 2 (Hornik 1991)
  approximates g by a 1-hidden-layer MLP; Lemma 3 uses χ(G⁴)-colorability to assign every node in
  any 2-hop neighborhood a unique one-hot indicator via pool MLPs. Then with identity weights and
  summing aggregators, h³_v carries [own indicator | own adjacency row | sum of neighbors' adjacency
  rows]; bᵀc counts edges among v's neighbors, Σb[i]=d_v, and c_v = 2(bᵀc − d_v)/(d_v(d_v−1)), a
  continuous function approximable by one more MLP layer. Corollary 2: random continuous features ⇒
  pairwise-distinct a.s. ⇒ conditions met. This is why pool > GCN/mean (relies on pool's universal
  symmetric approximation).

## Design-decision → why (with rejected alternatives)

| Decision | Why / alternative rejected |
|---|---|
| Learn aggregator *functions*, not per-node vectors | per-node lookup is transductive by construction; functions generalize to unseen nodes for free |
| Local neighbor recursion instead of full Â multiply | GCN's full-graph Laplacian multiply is transductive & O(|V|); local recursion needs only K-hop neighborhood |
| K stacked layers | depth = hop radius; K=2 best (K=1 too shallow −10–15%, K>2 marginal +0–5% for 10–100× cost) |
| Concatenate self+neighbor (skip) | preserves node's own signal across depths; folding into mean (GCN-style) loses it → measurably worse |
| ℓ2-normalize each layer | prevents scale blow-up over K layers; puts dot-product loss on directions |
| Permutation-invariant aggregator | neighbors are an unordered set; non-symmetric agg gives order-dependent garbage |
| Mean aggregator | cheap GCN-like baseline; but fixed linear → low capacity |
| Pool = MLP + elementwise max | symmetric + trainable + high capacity; max selects salient neighbor features (PointNet); also the only aggregator with the universal-approx guarantee → theory + best results |
| LSTM aggregator on random permutation | high capacity; not symmetric so randomize order; works but ~2× slower than pool |
| max vs mean pooling | no significant difference empirically; keep max |
| fixed-size uniform neighbor sample | full neighborhoods → unpredictable O(|V|) batches (hubs); sampling fixes per-batch cost at O(∏S_i) |
| fresh sample per layer k | decorrelates samples across depths |
| sample with replacement if degree<S | keeps fixed tensor shape |
| backward-sample then forward-aggregate minibatch | computes only needed nodes; avoids whole-graph passes |
| SkipGram/neg-sampling unsupervised loss on z's | nearby-similar/far-distinct without per-node params; reuses DeepWalk's proven objective but on generated embeddings |
| P_n = degree^{0.75}, Q=20 negatives | standard word2vec/DeepWalk negative-sampling recipe |
| ReLU, Adam, h-dim 256, K=2 (S₁=25,S₂=10) | reported working hyperparameters |

## Canonical implementations (grounding for code)
- Official TF: github.com/williamleif/GraphSAGE (aggregators.py = Mean/GCN/MaxPooling/Seq(LSTM);
  prediction.py = BipartiteEdgePredLayer xent neg-sampling loss; neigh_samplers.py = uniform
  shuffle+slice; models.py SampleAndAggregate; supervised_train / unsupervised_train).
- Clean PyTorch reference: github.com/williamleif/graphsage-simple (MeanAggregator with sample,
  Encoder doing concat+W+relu, SupervisedGraphSage). Best to ground the answer code on a clean
  PyTorch rendering of these.
