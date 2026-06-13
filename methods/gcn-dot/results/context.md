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

The precise goal is a method that simultaneously: (1) learns a low-dimensional embedding
`z_i` per node from which a candidate pair can be scored cheaply; (2) **uses node
features when they exist**, because the structure alone often under-determines who should link
to whom, yet the dominant embedding methods of the day ignore features entirely; (3) is a
**single model trained end to end** against one objective, rather than a multi-stage pipeline
where representation learning and the downstream scorer are optimized separately; (4) is
*unsupervised* — it must learn purely from the observed connectivity (and features), with no
node labels; and (5) produces a latent space with usable structure, so that, e.g., nodes of
the same (unobserved) community land near each other. The existing approaches below each hit
a subset of these and miss the rest. Closing that gap is the problem.

## Background

By this time, two largely separate lines of work bear on the problem, and the relevant pain
point is that neither line does what we want.

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
distributions. These are effective when the network's formation mechanism happens to match the
chosen heuristic, but each is one fixed feature of the topology; collectively they capture
only a small slice of the structural patterns a graph can exhibit, and they ignore node
features entirely.

**Latent-feature / embedding methods.** The second line learns a low-dimensional vector `z_i`
per node — *not* hand-designed — by fitting an objective derived from the graph. The classical
instance is **matrix factorization**: approximate the adjacency by a low-rank product,
`Â_{ij} = z_i^T z_j`, and minimize the squared reconstruction error over available pair
labels `Ω`, `L = (1/|Ω|) Σ_{(i,j) ∈ Ω} (A_{ij} − z_i^T z_j)²`. A link between two unseen nodes is then
predicted by the inner product of their fitted vectors. If instead of `A` one factorizes the
graph Laplacian `L` with the objective `Σ_{(i,j) ∈ E} ||z_i − z_j||²`, the nontrivial solution
is the eigenvectors for the smallest nonzero eigenvalues of `L` — the Laplacian-eigenmap /
spectral-clustering embedding (Belkin & Niyogi 2002; von Luxburg 2007).

The diagnostic facts about these embedding methods are load-bearing here. They optimize a
single global objective, so a node's final embedding can be influenced by every other node in
its connected component — that is their strength. But they are still fitted per-node from
connectivity alone. When the observed graph is sparse, disconnected, or missing many links,
the topology underdetermines many candidate pairs; a node's text, attributes, or other
features can carry exactly the information the links leave ambiguous, and these embeddings
have no mechanism to fold in that `X`. They also need a fairly large latent dimension even to
express simple neighborhood heuristics such as common-neighbors (Nickel et al. 2014). The
embedding is learned, but the encoder is still a table of node IDs rather than a learned map
from features and graph structure to representations.

**Convolutions on graphs.** Independently, a way to build neural networks that consume both a
graph's structure and its node features had just crystallized. Spectral graph convolution
defines a filter in the Fourier domain of the graph, `g_θ ⋆ x = U g_θ(Λ) U^T x`, where `U` are
the eigenvectors of the normalized Laplacian `L = I − D^{-1/2} A D^{-1/2}` and `Λ` its
eigenvalues. Evaluating this is `O(N²)` and needs an eigendecomposition, so it does not scale.
Hammond et al. (2011) and Defferrard et al. (2016) approximate `g_θ(Λ)` by a truncated
Chebyshev expansion `Σ_{k=0}^K θ'_k T_k(Λ̃)`, which is `K`-localized (depends only on the
`K`-hop neighborhood) and costs `O(|E|)`. Kipf & Welling (2017) push this to a first-order,
single-parameter form, then stabilize it with a **renormalization trick**: replace
`I + D^{-1/2} A D^{-1/2}` (whose eigenvalues span `[0,2]`, so repeated application explodes or
vanishes) by `D̃^{-1/2} Ã D̃^{-1/2}` with `Ã = A + I` (self-loops folded in) and
`D̃_{ii} = Σ_j Ã_{ij}`. A layer of the resulting **graph convolutional network** is

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
This machinery had been used and validated for **node classification** — a supervised,
per-node-label task. Whether and how it could drive an *unsupervised* objective over node
*pairs* was an open question.

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
framework had been applied to images and other vector data; it assumes i.i.d. datapoints and
says nothing, on its own, about relational data where the "datapoint" is an entire graph and
the thing to reconstruct is its connectivity.

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
similar vectors, so DeepWalk captures multi-hop neighborhood proximity. **Gaps:** it is a
*two-stage pipeline* — walk generation, then a separately optimized SkipGram model — not one
end-to-end objective; and it is purely structural, with **no way to ingest node features** `X`,
so when features carry signal that connectivity lacks, DeepWalk simply cannot use it. LINE
(Tang et al. 2015) and node2vec (Grover & Leskovec 2016) sharpen the walk/objective but
inherit the featureless, multi-stage character.

**Spectral clustering / Laplacian-eigenmap embedding (Tang & Liu 2011).** Embed nodes by the
bottom eigenvectors of the graph Laplacian — the minimizer of `Σ_{(i,j) ∈ E} ||z_i − z_j||²`
subject to an orthonormality constraint — placing connected nodes near each other in a
low-dimensional space, then score a candidate pair by the proximity of its endpoints.
*Core math:* eigen-decomposition of `L`; the embedding is a fixed linear-algebraic function of
the adjacency. **Gaps:** like DeepWalk it is purely structural — no node features — and the
embedding is a non-parametric spectral object recomputed per graph, with no learned map from
features-and-structure to representation, hence no end-to-end coupling to a link-scoring loss.

**Matrix factorization for link prediction (Koren et al. 2009; Ahmed et al. 2013).** Fit
low-rank `z_i` by reconstructing `Â_{ij} = z_i^T z_j` against adjacency entries under squared
loss, then predict unseen links by the inner product. *Core math:* low-rank
approximation of `A`. **Gaps:** again no node features, transductive, and it requires a large
rank to express even simple neighborhood heuristics (Nickel et al. 2014); the reconstruction
target is the raw binary adjacency under a squared loss rather than a probabilistic edge model.

**Graph convolutional networks for node classification (Kipf & Welling 2017).** The
feature-and-structure encoder above, trained *supervised* with a softmax head and a
cross-entropy loss over labeled nodes. *Core math:* the propagation rule
`H^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)})`. **Gap (for the present purpose):** it
is a supervised, per-node-label model. It produces excellent per-node representations from
features and structure, but it was set up to classify nodes against given labels; it does not,
as posed, learn from connectivity alone, nor does it score node *pairs* for the existence of an
edge — the unsupervised, relational objective is left undefined.

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

The pieces that already exist are: a sparse-matrix data pipeline that hands us the (training)
adjacency `A` in COO form and a node-feature matrix `X`; the graph-convolution primitive — a
layer that maps node representations to new node representations by mixing each node with its
neighbors through the normalized propagation matrix; an automatic-differentiation library and
a gradient-based optimizer (Adam); and an evaluation routine that, given per-node embeddings
and a set of positive / negative candidate pairs, computes AUC and AP. What is *not* settled —
and is exactly what the method must supply — is two empty slots: how to turn `(X, A)` into
node embeddings under an **unsupervised** signal, and how to turn a pair of embeddings into an
edge score together with the loss that trains both. Everything below the two `# TODO`s already
exists.

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
    """Maps node features + structure to per-node representations and scores
    node pairs. Both the embedding objective and the pair-scoring rule are the
    contribution — left empty here."""

    def __init__(self, in_dim, hidden_dim, emb_dim):
        super().__init__()
        # TODO: the encoder we will design, built from GraphConv layers,
        #       producing per-node representations from (X, A).
        pass

    def encode(self, X, A_norm):
        # TODO: produce per-node representations Z from features and structure.
        pass

    def score_pairs(self, Z, pair_index):
        # TODO: turn the representations of a batch of node pairs into edge
        #       scores (logits).
        pass

    def loss(self, Z, A):
        # TODO: the unsupervised objective that trains the encoder and the
        #       pair-scoring rule from the observed adjacency.
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
