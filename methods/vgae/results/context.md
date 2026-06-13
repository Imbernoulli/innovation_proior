# Context: unsupervised representation learning on graphs for link prediction (circa 2016)

## Research question

Given a single large undirected graph with only a fraction of its edges observed, predict which
of the unobserved node pairs are actually linked. The setting is one graph, not many: a citation
network, a social network, a protein-interaction network, where we see the nodes, some attributes
attached to each node, and an incomplete adjacency matrix, and we want to score every candidate
pair by how likely it is to be a real edge. The practical target is to learn a function that maps
each node to a vector and a function that scores a pair of vectors, trained only on the structure
that is actually present (plus whatever node attributes we have), so that held-out edges score
higher than held-out non-edges.

Two things make this hard at once. First, the supervision is one-class and lopsided: in a sparse
graph the vast majority of the N^2 possible pairs are non-edges, so any scorer trained naively
will be swamped by the easy negatives and learn to predict "no edge" everywhere. Second, and more
fundamental, the two sources of evidence about a link — the graph topology around the two nodes,
and the attributes (features) of the two nodes — live in different representations and most
existing methods can use only one of them. A method that exploits topology typically throws away
the features; a method that uses features typically ignores the graph. A solution would have to
fold both into a single node representation, in an unsupervised way (we have no link labels beyond
the edges themselves), and do it cheaply enough to run on graphs with thousands to tens of
thousands of nodes. It would also be desirable for the learned representation to be more than a
lookup table — to have some smooth, organized geometry so that "close in the space" means
"likely to be linked," and so that the model is a genuine probabilistic model of the graph rather
than a bag of per-node coordinates.

## Background

The field state at the time has two largely separate lineages that bear on this problem.

**Node-embedding / network-representation learning.** A line of work learns a low-dimensional
vector per node from the graph structure alone, so that downstream tasks (classification, link
prediction) can be run on the vectors with standard machinery. The dominant recent idea, imported
from natural-language modeling, is to treat a graph like a corpus: generate sequences of nodes
and learn embeddings that make co-occurring nodes similar, exactly as word2vec's skip-gram makes
co-occurring words similar. These embeddings encode neighborhood and community structure in a
continuous space of modest dimension. They are trained without labels, which is the right regime
for our problem, but they are computed purely from the adjacency relation — the node attributes,
when they exist, are simply not part of the model.

**Spectral / Laplacian methods.** An older lineage represents a graph through the spectrum of its
Laplacian. The (symmetric normalized) graph Laplacian L = I - D^{-1/2} A D^{-1/2} has eigenvectors
that capture global structure (cluster/community membership), and projecting nodes onto the leading
eigenvectors gives an embedding. The same operator underlies spectral graph convolution: a signal
x on the nodes is filtered in the Fourier domain g_theta * x = U g_theta(Lambda) U^T x, where U
diagonalizes L. This is expensive (an O(N^2) dense eigenbasis multiply, plus the eigendecomposition
itself), and it is again a structure-only view of the graph.

**Neural networks that consume both structure and features.** A more recent development makes the
two sources of evidence finally usable together. Starting from the spectral-convolution view above,
a localized first-order approximation collapses the expensive Fourier machinery into a cheap,
spatially-local operator. Approximating the filter by a first-order Chebyshev polynomial in L and
sharing a single parameter gives g_theta * x ~ theta (I_N + D^{-1/2} A D^{-1/2}) x; the operator
I_N + D^{-1/2} A D^{-1/2} has eigenvalues in [0, 2], so stacking many such layers risks
exploding/vanishing signals. A *renormalization trick* fixes this by folding the self-loop into the
adjacency before normalizing: with Atilde = A + I_N and Dtilde its degree matrix, the operator
I_N + D^{-1/2} A D^{-1/2} is replaced by Dtilde^{-1/2} Atilde Dtilde^{-1/2}, whose spectrum is
well-behaved for deep stacking. The resulting layer-wise propagation rule is

```
H^{(l+1)} = sigma( Dtilde^{-1/2} Atilde Dtilde^{-1/2} H^{(l)} W^{(l)} ),   H^{(0)} = X,
```

so a two-layer network reads Z = Ahat ReLU(Ahat X W0) W1 with Ahat = Dtilde^{-1/2} Atilde Dtilde^{-1/2}.
Each layer mixes a node's own feature vector with a normalized average of its neighbors' vectors and
applies a learned linear map and nonlinearity; after L layers a node's representation has aggregated
its L-hop neighborhood *and* the features along the way. The cost is O(|E|) per layer (a sparse-dense
matmul), linear in the number of edges, and the whole thing is trained by full-batch gradient
descent. This operator (Weisfeiler-Lehman-flavored: it is a differentiable, parameterized version
of 1-dimensional WL neighborhood aggregation) is the first cheap, end-to-end-differentiable way to
turn (X, A) into node representations that depend on both. It has so far been used with a softmax
head and a cross-entropy label loss for *semi-supervised node classification*; for our problem there
are no such labels.

**Amortized variational inference for latent-variable models.** Separately, a framework exists for
fitting a latent-variable generative model p(x) = int p(z) p(x|z) dz when the likelihood p(x|z) is a
neural network and both the marginal likelihood and the true posterior p(z|x) are intractable. The
move is to introduce a *recognition model* (an encoder) q(z|x) that approximates the posterior, and
to optimize the evidence lower bound

```
log p(x) = D_KL( q(z|x) || p(z|x) ) + L,    L = E_q[ log p(x|z) ] - D_KL( q(z|x) || p(z) ),
```

which holds for any q because the KL to the true posterior is non-negative, so L lower-bounds
log p(x). The bound splits cleanly into an expected reconstruction term E_q[log p(x|z)] and a
regularizer D_KL(q(z|x) || p(z)) that pulls the approximate posterior toward a fixed prior p(z).
Maximizing L wrt the encoder by a naive Monte-Carlo (score-function) gradient is too high-variance
to be usable. The *reparameterization trick* removes that variance: instead of sampling z ~ q(z|x)
directly, write z as a deterministic differentiable transform of a parameter-free noise variable,
z = g(eps, x) with eps ~ p(eps); for a diagonal-Gaussian q(z|x) = N(mu, diag(sigma^2)) this is

```
z = mu + sigma . eps,    eps ~ N(0, I),
```

so the expectation becomes a differentiable average over eps and gradients flow into (mu, sigma).
When both q(z|x) and the prior p(z) are Gaussian, the KL term has a closed form and needs no
sampling at all. The standard derivation, for a diagonal Gaussian posterior N(mu, diag(sigma^2))
against a standard-normal prior N(0, I) in J dimensions, evaluates the two Gaussian integrals

```
int q log p(z) dz = -J/2 log(2pi) - 1/2 sum_j ( mu_j^2 + sigma_j^2 ),
int q log q(z) dz = -J/2 log(2pi) - 1/2 sum_j ( 1 + log sigma_j^2 ),
```

and subtracts to give

```
-D_KL( q(z|x) || p(z) ) = 1/2 sum_j ( 1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 ).
```

In the canonical instantiation the encoder is a multilayer perceptron whose two output heads
produce mu and log sigma^2 from a single input vector x; the decoder is another MLP giving a
Bernoulli or Gaussian likelihood for x. This framework is for i.i.d. data — one independent x per
datapoint, a per-datapoint posterior — and the encoder/decoder are dense networks.

## Baselines

These are the prior methods a new link-prediction approach would be measured against on the same
citation networks.

**Spectral clustering (SC) (Tang & Liu, 2011).** Embed nodes via the leading eigenvectors of a
matrix derived from the graph (Laplacian / modularity), then use the embedding for downstream tasks.
To score a candidate link from such an embedding Z, one takes the inner product of the two node
vectors (i.e. an entry of Z Z^T) as the affinity. The embedding is a global spectral summary of the
adjacency. Limitation: it is computed from the graph structure only; there is no mechanism to admit
node attributes, and the eigendecomposition does not scale gracefully. It is also a fixed linear
projection of the adjacency, not a learned, feature-aware representation.

**DeepWalk (DW) (Perozzi, Al-Rfou & Skiena, 2014).** Generate truncated random walks from each
node, treat each walk as a sentence and each node as a word, and run skip-gram (word2vec) on the
resulting "corpus" to learn a vector per node so that nodes that co-occur in walks get similar
vectors. The embeddings capture neighborhood and community membership and work well as unsupervised
features. Limitation: the pipeline is multi-step (walk generation, then a separately optimized
skip-gram objective), so it is not a single end-to-end model; the random-walk objective is a proxy
for structural proximity rather than a generative model of the adjacency; and, like SC, it consumes
only the graph topology — the node attributes never enter. Later refinements (LINE; node2vec, with
its BFS/DFS-biased walks) sharpen the proximity notion but stay within the same structure-only,
multi-step family and likewise leave features out.

**A feature-aware encoder used only for supervised classification.** The graph-convolutional
propagation rule above can turn (X, A) into representations that depend on both structure and
features, and is cheap and end-to-end differentiable. So far it has been wired to a softmax output
and trained against node labels for semi-supervised classification. Limitation for our problem: that
usage is supervised — it needs node labels, which we do not have here — and the representation it
produces is tuned to a label taxonomy rather than to the graph's own connectivity. It has not been
trained in an unsupervised regime where only the edges supervise.

## Evaluation settings

The natural yardsticks are citation networks where nodes are documents, edges are citation links,
and each document carries a sparse bag-of-words feature vector: Cora (~2.7k nodes, ~1.4k-dim
features), Citeseer (~3.3k nodes, ~3.7k-dim features), and Pubmed (~20k nodes). The protocol holds
out a fraction of the edges and trains on the remaining incomplete graph: a validation set of ~5%
of the links and a test set of ~10% of the links are removed, and for each held-out edge an equal
number of randomly sampled non-edges (unconnected node pairs) is drawn. All node features are kept.
A model scores every pair in the held-out sets, and is judged on its ability to rank true edges
above non-edges — area under the ROC curve (AUC) and average precision (AP). The validation set is
used for hyperparameter selection. Both featureless and feature-using variants are of interest
(featureless = replace X with the identity matrix). Weights are initialized with the standard
fan-in/fan-out scheme; training is full-batch.

## Code framework

The harness already in place wraps a `LinkPredictor` with three responsibilities: an `encode` that
maps node features plus the training edges to a per-node representation, a `decode` that scores a
batch of candidate node pairs from those representations, and a `forward` that runs the two in
sequence. The training loop supplies node features `x`, the observed undirected training edges
`edge_index`, and a set of candidate pairs `edge_label_index` (positive edges plus sampled
negatives), and applies a binary edge-classification loss to the returned scores. The graph-
convolution layer that mixes a node's features with its neighbors' is available as a primitive
(`GCNConv`), as are negative sampling and the optimizer. What is *not* settled is the contents of
`encode`/`decode` — the form of the node representation, what (if anything) is regularized, and how
a pair is scored. That is the empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LinkPredictor(nn.Module):
    """Encode nodes from (features, training edges); score candidate node pairs.

    The neighborhood-aggregating GCN layer exists as a primitive; the node
    representation, any regularization on it, and the pair-scoring rule are
    the open design.
    """

    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        # TODO: the encoder we will design -- maps (x, edge_index) to a
        #       per-node representation built from GCN layers.
        pass

    def encode(self, x, edge_index):
        # returns a per-node representation [N, hidden_channels]
        # TODO: build the representation here.
        pass

    def decode(self, z_src, z_dst):
        # returns [num_edges] scores for the given source/destination rows
        # TODO: the pair-scoring rule we will design.
        pass

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        return self.decode(z_src, z_dst)


# existing training loop the predictor plugs into
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    # candidate edges: positives from the graph + sampled negatives
    scores = model(data.x, data.edge_index, data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(scores, data.edge_label)
    loss.backward()
    optimizer.step()
    return float(loss)
```
