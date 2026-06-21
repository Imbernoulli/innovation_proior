# Context: unsupervised representation learning and link prediction on graphs (circa 2014-2016)

## Research question

We are handed an undirected, unweighted graph — a citation network, a social network, a
biological interaction network — given as an adjacency matrix `A` over `N` nodes, and often a
matrix of node features `X` (an `N x D` table where row `i` describes node `i`, e.g. the
bag-of-words of a paper). Part of the graph is hidden from us: some real edges have been
removed. The task is **link prediction** — score every candidate node pair so that the held-out
true edges score higher than node pairs that are genuinely unconnected. This is the same
computation that powers friend recommendation, paper recommendation, knowledge-graph
completion, and protein-interaction prediction.

The setting is *unsupervised*: there are no node labels, only the observed connectivity and
(when available) the node features `X`. The broad question is how to learn, purely from this
input, a low-dimensional embedding `z_i` per node together with a way to score candidate node
pairs from those embeddings.

## Background

By this time, two largely separate lines of work bear on the problem.

**Heuristic link-prediction scores.** A long tradition computes a hand-designed similarity
score between two nodes `x` and `y` from the observed topology and ranks candidate links by
it (Liben-Nowell & Kleinberg 2007; Lü & Zhou 2011). Writing `Γ(x)` for the neighbor set of
`x`, the simplest is **common neighbors**, `f_CN(x,y) = |Γ(x) ∩ Γ(y)|` — two nodes are likely
to link if they already share many neighbors. **Jaccard** normalizes it,
`|Γ(x) ∩ Γ(y)| / |Γ(x) ∪ Γ(y)|`. **Preferential attachment** uses the product of degrees,
`|Γ(x)|·|Γ(y)|`, encoding the rich-get-richer mechanism of scale-free networks (Barabási &
Albert 1999). **Adamic-Adar** down-weights high-degree common neighbors,
`Σ_{z ∈ Γ(x) ∩ Γ(y)} 1/log|Γ(z)|`, and **resource allocation** down-weights them harder,
`Σ_z 1/|Γ(z)|`. Going beyond local neighborhoods, the **Katz index**
`Σ_{l=1}^{∞} β^l |walks^{(l)}(x,y)|` sums all walks between `x` and `y` with longer walks
discounted by `β < 1`, and **rooted PageRank** / **SimRank** use random-walk stationary
distributions. Each is a fixed, hand-designed function of the observed topology, effective
when the network's formation mechanism matches the chosen heuristic.

**Latent-feature / embedding methods.** The second line learns a low-dimensional vector `z_i`
per node — *not* hand-designed — by fitting an objective derived from the graph. The classical
instance is **matrix factorization**: approximate the adjacency by a low-rank product,
`Â_{ij} = z_i^T z_j`, and minimize the squared reconstruction error over available pair
labels `Ω`, `L = (1/|Ω|) Σ_{(i,j) ∈ Ω} (A_{ij} − z_i^T z_j)²`. A link between two unseen nodes is then
predicted by the inner product of their fitted vectors. If instead of `A` one factorizes the
graph Laplacian `L` with the objective `Σ_{(i,j) ∈ E} ||z_i − z_j||²`, the nontrivial solution
is the eigenvectors for the smallest nonzero eigenvalues of `L` — the Laplacian-eigenmap /
spectral-clustering embedding (Belkin & Niyogi 2002; von Luxburg 2007).

These embedding methods optimize a single global objective, so a node's final embedding can be
influenced by every other node in its connected component. They are fitted per-node from
connectivity alone; the embedding is a free vector per node ID rather than the output of a map
from features and structure to representations.

**Convolutions on graphs.** Independently, a way to build neural networks that consume both a
graph's structure and its node features had just crystallized. Spectral graph convolution
defines a filter in the Fourier domain of the graph, `g_θ ⋆ x = U g_θ(Λ) U^T x`, where `U` are
the eigenvectors of the normalized Laplacian `L = I − D^{-1/2} A D^{-1/2}` and `Λ` its
eigenvalues. Evaluating this directly is `O(N²)` and needs an eigendecomposition.
Hammond et al. (2011) and Defferrard et al. (2016) approximate `g_θ(Λ)` by a truncated
Chebyshev expansion `Σ_{k=0}^K θ'_k T_k(Λ̃)`, which is `K`-localized (depends only on the
`K`-hop neighborhood) and costs `O(|E|)`. Kipf & Welling (2017) push this to a first-order,
single-parameter form, then stabilize it with a **renormalization trick**: replace
`I + D^{-1/2} A D^{-1/2}` (whose eigenvalues span `[0,2]`) by `D̃^{-1/2} Ã D̃^{-1/2}` with
`Ã = A + I` (self-loops folded in) and `D̃_{ii} = Σ_j Ã_{ij}`. A layer of the resulting
**graph convolutional network** is

```
H^{(l+1)} = σ( D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)} ),     H^{(0)} = X,
```

with a per-layer weight matrix `W^{(l)}` and pointwise nonlinearity `σ`. The propagation
matrix `D̃^{-1/2} Ã D̃^{-1/2}` mixes each node's representation with its neighbors', weighting
the edge `(i,j)` by `1/√(d̃_i d̃_j)`: self-loops let a node retain its own features, and the
*symmetric* normalization (rather than the row-stochastic `D^{-1} A`, which would merely
average neighbors) down-weights high-degree neighbors and keeps the operator symmetric with a
real spectrum. Stacking such layers gives a differentiable map `f(X, A)` from features and
structure to per-node vectors, costing `O(|E| · F · C)` for `C` input and `F` output channels.
This machinery had been used and validated for **node classification**, a supervised,
per-node-label task.

**Latent-variable generative modeling.** A parallel and quite general tool was the
variational auto-encoder (Kingma & Welling 2014; Rezende et al. 2014) for learning latent
representations of i.i.d. data. It posits data generated as `z ∼ p(z)`, `x ∼ p_θ(x|z)`, with
the posterior `p(z|x)` intractable, and introduces a recognition model `q_φ(z|x)` (a
"probabilistic encoder") that approximates it. Learning maximizes the evidence lower bound

```
L(θ,φ;x) = E_{q_φ(z|x)}[ log p_θ(x|z) ] − KL[ q_φ(z|x) || p(z) ],
```

a reconstruction term minus a divergence pulling the posterior toward the prior. The gradient
w.r.t. `φ` is made low-variance and differentiable by the **reparameterization trick**: for a
Gaussian `q_φ(z|x) = N(μ, σ² I)`, write `z = μ + σ ⊙ ε` with `ε ∼ N(0, I)`, so the randomness
sits in `ε` and gradients flow through `μ, σ`. When `q` and the prior are both Gaussian the KL
has the closed form `−½ Σ_j (1 + log σ_j² − μ_j² − σ_j²)`, computable without sampling. This
framework had been applied to images and other vector data, where it assumes i.i.d.
datapoints.

## Baselines

These are the prior methods a new graph-embedding / link-prediction method would be measured
against and would react to.

**DeepWalk (Perozzi, Al-Rfou & Skiena, KDD 2014).** Treat a graph like a corpus: from each
node, sample short truncated **random walks**, read each walk as a "sentence" of node IDs, and
feed these to the **SkipGram** language model, which learns an embedding `Φ(v) ∈ R^d` per node
by maximizing the probability of a node's walk-context,
`min_Φ −log Pr({v_{i−w}, …, v_{i+w}} \ v_i | Φ(v_i))`. The `|V|`-way softmax over node IDs is
made tractable with a **hierarchical softmax** (a Huffman tree, so each prediction is a
product of binary decisions along a root-to-leaf path, `O(log|V|)` instead of `O(|V|)`).
Embeddings are trained by SGD with an annealed learning rate. Nodes that co-occur in walks get
similar vectors, so DeepWalk captures multi-hop neighborhood proximity. It runs in two stages —
walk generation, then a separately optimized SkipGram model — and reads only the graph
structure. LINE (Tang et al. 2015) and node2vec (Grover & Leskovec 2016) sharpen the
walk/objective in the same family.

**Spectral clustering / Laplacian-eigenmap embedding (Tang & Liu 2011).** Embed nodes by the
bottom eigenvectors of the graph Laplacian — the minimizer of `Σ_{(i,j) ∈ E} ||z_i − z_j||²`
subject to an orthonormality constraint — placing connected nodes near each other in a
low-dimensional space, then score a candidate pair by the proximity of its endpoints.
*Core math:* eigen-decomposition of `L`; the embedding is a fixed linear-algebraic function of
the adjacency, computed from structure alone and recomputed per graph.

**Matrix factorization for link prediction (Koren et al. 2009; Ahmed et al. 2013).** Fit
low-rank `z_i` by reconstructing `Â_{ij} = z_i^T z_j` against adjacency entries under squared
loss, then predict unseen links by the inner product. *Core math:* low-rank
approximation of `A` from structure alone; the embedding is a free per-node vector and the
reconstruction target is the raw binary adjacency under a squared loss.

**Graph convolutional networks for node classification (Kipf & Welling 2017).** The
feature-and-structure encoder above, trained *supervised* with a softmax head and a
cross-entropy loss over labeled nodes. *Core math:* the propagation rule
`H^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)})`. It produces per-node representations
from features and structure and is set up to classify nodes against given labels.

## Evaluation settings

The natural yardstick is link prediction on citation networks (Sen et al. 2008): **Cora**,
**Citeseer**, **Pubmed**, each a graph of documents (nodes) with citation links (edges) and a
sparse bag-of-words feature vector per document. The protocol removes a fraction of the edges
to form held-out sets — typically a validation set of about 5% of the links and a test set of
about 10% — keeping all node features, and pairs each held-out true edge with an equal number
of randomly sampled unconnected node pairs (non-edges). A model is scored on how well it ranks
the true held-out edges above the sampled non-edges, reported as **area under the ROC curve
(AUC)** and **average precision (AP)**, averaged over several random initializations on fixed
splits; the validation set is used to choose hyperparameters. A **featureless** variant of the
same protocol drops `X` (replacing it with the identity), to isolate how much the node features
contribute.

## Code framework

The pieces that already exist are: a sparse-matrix data pipeline that hands us the training
adjacency `A` in COO form and a node-feature matrix `X`; the graph-convolution primitive — a
layer that maps node representations to new node representations by mixing each node with its
neighbors through the normalized propagation matrix; an automatic-differentiation library and
a gradient-based optimizer (Adam); and an evaluation routine that, given model-produced node
vectors and a set of positive / negative candidate pairs, computes AUC and AP. What is not
settled is the graph representation learner itself: the model must learn from observed
connectivity, use `X` when available, and expose scores for candidate pairs. The scaffold below
leaves that contribution as a single generic slot.

```python
import torch
import torch.nn as nn


def normalized_adjacency(A):
    """D̃^{-1/2} Ã D̃^{-1/2} with Ã = A + I (renormalization). Already available."""
    ...


class GraphConv(nn.Module):
    """One graph-convolution layer: H' = prop_matrix @ (H W). Already available."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H, A_norm):
        return torch.sparse.mm(A_norm, self.W(H))


class GraphModel(nn.Module):
    """Consumes node features plus graph structure and provides link scores."""

    def __init__(self, in_dim, hidden_dim, emb_dim):
        super().__init__()
        # TODO: the model we will design.
        pass

    def encode(self, X, A_norm):
        # TODO: the representation rule we will design.
        pass

    def score_pairs(self, Z, pair_index):
        # TODO: the candidate-pair scoring rule we will design.
        pass

    def loss(self, Z, A):
        # TODO: the unsupervised objective we will design.
        pass


# existing training loop the model plugs into
def train(model, X, A, A_norm, optimizer, n_iters):
    for _ in range(n_iters):
        optimizer.zero_grad()
        Z = model.encode(X, A_norm)
        loss = model.loss(Z, A)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, X, A_norm, pos_pairs, neg_pairs):
    """Encode, score the positive and negative candidate pairs, return AUC/AP.
    Already available."""
    ...
```
