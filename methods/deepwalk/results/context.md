# Context

## Research question

Many tasks on social and information networks reduce to *classifying vertices* — predicting one or more labels for a node (a user's interests, a page's topics) from the network structure. The obstacle is representation: a network is naturally described by its adjacency structure, which is sparse and discrete, and sparse discrete features are hard for ordinary statistical classifiers to generalize from, especially when only a small fraction of nodes are labeled. The question is whether one can *learn*, in an unsupervised way and from the graph structure alone, a low-dimensional, continuous feature vector for every vertex — a "social representation" — that encodes community membership and can then be fed to any off-the-shelf classifier. A good solution would (a) place nodes from the same community close together in the vector space, (b) be low-dimensional and continuous, (c) scale to networks with millions of nodes, and (d) be *online* — updatable from local information as the graph changes, without recomputing everything.

## Background

**Relational classification.** The traditional approach to labeling nodes poses the problem as inference in an undirected Markov network and runs iterative approximate inference — iterative classification, Gibbs sampling, label relaxation — to compute label posteriors from the network and the known labels jointly. These methods entangle feature construction with the specific classification task; they do not produce reusable, task-independent features, and they can struggle to scale.

**Latent social dimensions.** A line of work (modularity-based and spectral "social dimension" methods) extracts latent vertex features by eigendecomposition of a graph matrix (modularity or Laplacian). These give continuous features but require a global computation, do not update incrementally, and tend to weaken when labeled data is scarce.

**Truncated random walks and local structure.** Short random walks from a vertex are a well-studied probe of *local* community structure — they are the basis of output-sensitive algorithms that estimate local community membership in time sublinear in the graph size. Two properties make them attractive as a feature-extraction primitive: the co-occurrences within a short walk encode local community information, and because a walk only uses local information, small changes to the graph can be accommodated without global recomputation.

**Power-law statistics (diagnostic fact).** Many real networks are scale-free: their degree distribution follows a power law. A consequence is that the frequency with which vertices appear in a stream of short random walks *also* follows a power law. Separately, the frequency of words in natural-language text follows a power law (Zipf's law). The two distributions have the same shape — a small number of very frequent symbols and a long tail of rare ones. This is a pre-method fact about the world: the symbol-frequency statistics of random walks on a scale-free graph match those of natural language.

**Neural language models.** Language modeling estimates the probability of a sequence of words. Classical formulations estimate Pr(w_n | w_0,…,w_{n−1}), which becomes infeasible as context grows. The Skip-gram model reframes the task: use a single word to predict the words in a surrounding window (both sides), and drop the requirement to know each context word's exact offset — i.e., maximize the probability of the *set* of context words given the center word, order-independently. This yields, for a center word v_i and window w, the objective `minimize −log Pr({v_{i−w},…,v_{i−1}, v_{i+1},…,v_{i+w}} | Φ(v_i))`, where Φ maps each symbol to a d-dimensional vector; it is optimized by SGD.

**Hierarchical softmax.** A flat softmax (or a logistic classifier) over a vocabulary of size |V| needs an O(|V|) normalization per prediction — infeasible when |V| is in the millions. Hierarchical softmax places the vocabulary at the leaves of a binary tree and factors the probability of a target into a product of binary decisions along the root-to-leaf path, each decision made by a logistic classifier at an internal node, reducing the cost to O(log|V|). Assigning shorter tree paths to more frequent symbols (Huffman coding) further reduces expected cost — which pays off precisely when symbol frequencies follow a power law.

## Baselines

**SpectralClustering / Modularity (social dimensions; Tang & Liu).** Eigendecompose a graph matrix (normalized Laplacian or modularity matrix) to produce latent node features for classification. Gap: global eigendecomposition does not scale and is not incremental, and the features degrade with sparse labels.

**EdgeCluster.** Clusters edges to produce sparse social-dimension features scalable to larger graphs. Gap: still a fixed, global construction tied to the clustering, not a learned continuous representation.

**Collective/relational classifiers (wvRN, majority voting).** Propagate labels through the graph using neighbor label distributions. Gap: not feature learning — they require labels at inference and produce no reusable vertex features.

**Skip-gram / neural language models (Mikolov et al.).** The learner that maps symbols in sequences to continuous vectors via a windowed, order-independent prediction objective, trained by SGD with hierarchical softmax. It operates on sequences of symbols; what it lacks, for networks, is any notion of a "sentence" — there are no sequences in a graph.

## Evaluation settings

- **Task:** multi-label vertex classification — learn embeddings unsupervised, then train a one-vs-rest logistic-regression classifier on a fraction of labeled nodes and predict the rest.
- **Datasets:** BlogCatalog, Flickr, YouTube (social networks of varying size, up to web scale).
- **Metrics:** Macro-F1 and Micro-F1, reported across a range of labeled-data fractions (to probe behavior under label sparsity).
- **Baselines:** SpectralClustering, Modularity, EdgeCluster, wvRN, majority.
- **Other probes:** sensitivity to embedding dimension d and walks-per-vertex γ; robustness to label sparsity; scalability via a parallel implementation.

## Code framework

The primitives that already exist: a graph with adjacency/neighbor lists; the Skip-gram trainer (SGD over windowed symbol contexts, with hierarchical softmax over a binary tree) that consumes sequences of symbols and emits d-dimensional vectors; SGD with a decaying learning rate. The slots below are what a graph feature-learning method would fill in.

```python
import random

class Graph:                          # adjacency lists (exists)
    def neighbors(self, v): ...
    def nodes(self): ...

# --- the Skip-gram language model: maps symbols in sequences to vectors ---
def skipgram(sequences, dim, window):
    # known: for each symbol, predict the symbols within `window`; SGD with
    #        hierarchical softmax (binary tree over the vocabulary); returns Phi
    ...

# --- turning a graph into "sentences" ---
def make_corpus(G, ...):
    # TODO: a graph has no sentences -- how is a corpus of symbol sequences
    #       generated from the graph structure?
    pass

# --- the overall feature learner ---
def learn_representations(G, dim, window):
    corpus = make_corpus(G)           # TODO: fill this in
    return skipgram(corpus, dim=dim, window=window)
```
