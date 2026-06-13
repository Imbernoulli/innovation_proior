# Context: inductive node embedding on large, evolving graphs

## Research question

Low-dimensional vector embeddings of nodes have become the standard way to feed graph-structured
data into downstream machine learning: a node's position and neighborhood are distilled into a
dense vector that can be used for node classification, clustering, and link prediction. Every
mature approach, however, assumes one fixed graph in which *all* nodes are present at training
time. The embedding of a node is, quite literally, one of the parameters fit during training.

Real production systems do not work this way. A content platform sees new posts, videos, and users
every minute; a citation graph grows every year; a biology lab collects a protein-protein
interaction graph for an organism that did not exist in any training set. These systems need an
embedding for a node — or for an *entire new graph* — that was never observed during training, and
they need it cheaply, at inference time, without re-running an optimization.

The precise problem: given a node together with its features and its local graph neighborhood,
produce a useful embedding for it, where "useful" means it supports the same downstream tasks as a
trained embedding would — and where the node, and possibly the whole graph it lives in, may be
unseen at training time. A solution must (i) generalize to unseen nodes and unseen graphs without
per-node re-optimization, (ii) produce embeddings whose space is consistent across graphs (so a
classifier trained on one graph's embeddings applies to another's), (iii) scale to graphs with
hundreds of thousands of nodes and to high-degree hub nodes, and (iv) capture *structural* role and
position, not merely raw features.

## Background

**Node embeddings as dimensionality reduction.** The dominant idea is to compress a node's
high-dimensional neighborhood information into a dense vector and hand it to a downstream model. The
unit of learning is a free vector per node, assembled into a matrix Z ∈ R^{|V|×d}, optimized
directly. Because the parameters *are* the per-node vectors, these methods are inherently
**transductive**: they only ever produce vectors for the nodes seen during training.

**Random-walk / matrix-factorization objectives.** A successful family (random-walk SkipGram
methods, and related spectral-clustering and multidimensional-scaling approaches) trains Z so that
nodes co-occurring on short random walks have high dot-product similarity. Levy & Goldberg (2014)
showed this class is equivalent to an implicit matrix factorization, learning Z with Z Zᵀ ≈ M for
some matrix M of random-walk statistics. A consequence that turns out to matter a great deal: an
objective built only from dot products zᵢᵀzⱼ is invariant under any orthogonal transformation Q,
since (ZQᵀ)(QZᵀ) = ZZᵀ. The embedding space is otherwise unconstrained, so it is free to rotate
arbitrarily during training. Two graphs trained separately therefore land in arbitrarily rotated
spaces; even one graph re-trained as it grows can drift. This is a known, derivable property of the
objective, not an implementation accident — and it means a classifier fit on one graph's embeddings
is essentially random when fed another's.

**Graph convolutions from spectral filtering.** A separate line defines convolution on graphs via
the graph Laplacian's spectrum. Truncating spectral filters to low-order polynomials (e.g.
Chebyshev) localizes them; taking the first-order term and a renormalization yields a simple,
efficient layer-wise rule that operates directly on features and shared weights, rather than on
free per-node vectors. This is conceptually a big step: the learned object becomes a *function of
features* with parameters shared across nodes. As published and used, however, the rule is written
as a single matrix multiplication against the entire (normalized) adjacency of the graph, applied
to all nodes at once, in a fixed-graph semi-supervised setting.

**Iterative neighborhood refinement and graph isomorphism.** Classic combinatorial work refines a
node's label by repeatedly hashing the multiset of its neighbors' labels — the Weisfeiler–Lehman
vertex-refinement test for graph isomorphism. After enough rounds, structurally distinct nodes get
distinct labels (for a broad class of graphs). This gives a principled, purely structural notion of
"aggregate your neighbors, repeat" and a yardstick for how much structure such a scheme can in
principle recover.

**Symmetric functions over unordered sets.** A node's neighbors form a *set* with no canonical
order, so any neighbor-combining operation should be permutation-invariant. Recent point-set work
(PointNet, Qi et al. 2017) showed that pushing each element through a shared per-element network and
then taking an elementwise max is a permutation-invariant operation that can approximate *any*
Hausdorff-continuous symmetric set function to arbitrary precision. Together with the classical
single-hidden-layer universal approximation theorem (Hornik 1991), this provides the machinery to
reason about what a learned set-aggregator can and cannot represent.

**Motivating empirical facts about the prior art.** Re-embedding a new node with a transductive
random-walk method requires fresh rounds of stochastic gradient descent on that node before any
prediction can be made — orders of magnitude slower than a single feed-forward pass. The rotational
invariance above is observed to bite harder when a test set connects weakly back to the training
graph (fewer cross-edges ⇒ worse alignment). And feature data in real graphs is often sparse (in
some protein graphs a large fraction of nodes have no nonzero features), so leveraging neighborhood
information is not optional. These are facts about existing systems and datasets, knowable before
any new method exists.

## Baselines

**Random-walk SkipGram embeddings (DeepWalk; Perozzi et al. 2014; node2vec, Grover & Leskovec
2016; LINE, Tang et al. 2015).** Generate truncated random walks, treat each walk as a sentence,
and train per-node vectors with a SkipGram / negative-sampling objective so co-occurring nodes have
similar dot products. The objective has the form α Σ_{i,j∈A} f(zᵢᵀzⱼ) + β Σ_{i,j∈B} g(zᵢᵀzⱼ) over
positive/negative pairs. Strong on a single fixed graph. Gaps: (1) transductive — the parameters
are the per-node vectors, so an unseen node has no vector and requires additional SGD before it can
be used; (2) the dot-product objective is orthogonally invariant, so embedding spaces of different
graphs are arbitrarily rotated and do not transfer, and re-trained spaces drift; (3) node features
are ignored entirely. node2vec with p=q=1 reduces to DeepWalk.

**Inductive regularization embeddings (Planetoid-I; Yang et al. 2016).** An embedding-based
semi-supervised method that *is* inductive, but uses graph structure only as a regularizer during
training; it does not consult a node's neighborhood at inference time. Gap: it does not use graph
structure to generate the embedding of a new node, so it leaves the core inductive mechanism on the
table.

**First-order spectral graph convolution (GCN; Kipf & Welling 2017).** Layer-wise rule
H^{(l+1)} = σ( D̂^{-1/2} Â D̂^{-1/2} H^{(l)} W^{(l)} ) with Â = A + I (self-loops) and D̂ the degree
matrix of Â; derived as a first-order approximation of a spectral convolution. The learned object
is a function of node features with weights W^{(l)} shared across all nodes — a genuine departure
from per-node lookups. Gaps: the rule is a full-graph operation requiring the entire adjacency /
graph Laplacian in memory; all node representations are trained simultaneously; it has only been
applied transductively to a single fixed graph; there is no minibatch over nodes and no mechanism
to embed a brand-new node from just its local neighborhood; and its neighbor aggregation is a
single fixed (normalized) weighted mean — one linear convolution, with limited aggregation capacity.

**Feature-only classifier.** A logistic-regression model on raw node features that ignores graph
structure entirely. Gap: discards all relational information; a floor that any graph-aware method
must clear, and the natural way to measure how much the graph adds.

## Evaluation settings

The natural inductive yardsticks are tasks where test nodes (or whole test graphs) are held out:

- **Citation graph (Web of Science).** An undirected paper-citation graph (~300K nodes, average
  degree ~9) over several biology fields. Node features are node degree plus a sentence embedding of
  the paper abstract (300-d word vectors). Train on earlier years, predict subject category on a
  later year's papers that were absent during training. Metric: micro/macro-averaged F1.
- **Reddit posts.** A post-to-post graph (~230K nodes, average degree ~492) where two posts are
  linked if the same user commented on both; features are GloVe embeddings of title and comments
  plus post score and comment count; label is the community (subreddit). Train on early days of a
  month, predict community for later-day posts. Metric: F1. (Edges may be down-sampled so no node
  exceeds a fixed degree, to allow dense adjacency lists.)
- **Protein-protein interaction (PPI).** Many graphs, one per human tissue (~2.4K nodes each,
  average degree ~29). Features are positional/motif/immunological gene sets; labels are gene
  ontology functions (multi-label, 121 classes). Train on a set of graphs, test on *entirely
  separate* graphs — the strict cross-graph inductive setting. Metric: F1.

Protocol: predictions are always on nodes not seen during training; for PPI, on entirely unseen
graphs. Embeddings can be evaluated either by training a downstream logistic classifier on
unsupervised embeddings, or end-to-end with a supervised cross-entropy loss. A standard control is
random feature noise injected in place of real features to probe how much purely structural signal
a method captures. Timing (train and inference wall-clock) is part of the comparison, since the
point is cheap inference on new nodes.

## Code framework

The available pieces are an automatic-differentiation framework with an optimizer (Adam), standard
layers (linear, ReLU, an LSTM cell, dropout), a SkipGram-style negative-sampling loss, and a graph
stored as adjacency lists. A minimal node-embedding harness can be sketched with empty slots:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- graph data: features as a lookup, neighbors as adjacency lists ---
# features: callable mapping node-id tensor -> feature/representation tensor
# adj_lists: dict node -> set(neighbor nodes)

def sample_neighbors(adj_lists, nodes, num_sample):
    # TODO
    pass

# --- empty model slot: map a node (and what is observable about it) to a vector ---

class EmbeddingModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # TODO
    def forward(self, nodes):
        # TODO
        pass

# --- losses ---

def embedding_loss(*args, **kwargs):
    # TODO
    pass

def supervised_loss(scores, labels):
    return F.cross_entropy(scores, labels)

# --- training loop ---
# for batch of target nodes:
#   z = EmbeddingModel(batch)
#   loss = embedding_loss(...) or supervised_loss(...)
#   loss.backward(); optimizer.step()
```
