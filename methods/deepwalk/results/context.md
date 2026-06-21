# Context

## Research question

Many tasks on social and information networks reduce to *classifying vertices* — predicting one or more labels for a node (a user's interests, a page's topics) from the network structure. A network is naturally described by its adjacency structure, which is sparse and discrete. The question is whether one can *learn*, in an unsupervised way and from the graph structure alone, a low-dimensional, continuous feature vector for every vertex — a "social representation" — that encodes community membership and can then be fed to any off-the-shelf classifier.

## Background

**Relational classification.** The traditional approach to labeling nodes poses the problem as inference in an undirected Markov network and runs iterative approximate inference — iterative classification, Gibbs sampling, label relaxation — to compute label posteriors from the network and the known labels jointly. Feature construction is entangled with the specific classification task.

**Latent social dimensions.** A line of work (modularity-based and spectral "social dimension" methods) extracts latent vertex features by eigendecomposition of a graph matrix (modularity or Laplacian). These give continuous features via a global computation.

**Truncated random walks and local structure.** Short random walks from a vertex are a well-studied probe of *local* community structure — they are the basis of output-sensitive algorithms that estimate local community membership in time sublinear in the graph size. A walk uses only local information (the neighbors of the current node) at each step.

**Power-law statistics (diagnostic fact).** Many real networks are scale-free: their degree distribution follows a power law. A consequence is that the frequency with which vertices appear in a stream of short random walks *also* follows a power law — a small number of very frequent (high-degree hub) vertices and a long tail of rare ones. Separately, the frequency of words in natural-language text follows a power law (Zipf's law). These are pre-method facts about the world.

**Neural language models.** Language modeling estimates the probability of a sequence of words. Classical formulations estimate Pr(w_n | w_0,…,w_{n−1}), which becomes infeasible as context grows. The Skip-gram model reframes the task: use a single word to predict the words in a surrounding window (both sides), and drop the requirement to know each context word's exact offset — i.e., maximize the probability of the *set* of context words given the center word, order-independently. This yields, for a center word v_i and window w, the objective `minimize −log Pr({v_{i−w},…,v_{i−1}, v_{i+1},…,v_{i+w}} | Φ(v_i))`, where Φ maps each symbol to a d-dimensional vector; it is optimized by SGD.

**Hierarchical softmax.** A flat softmax (or a logistic classifier) over a vocabulary of size |V| needs an O(|V|) normalization per prediction. Hierarchical softmax places the vocabulary at the leaves of a binary tree and factors the probability of a target into a product of binary decisions along the root-to-leaf path, each decision made by a logistic classifier at an internal node, reducing the cost to O(log|V|). Assigning shorter tree paths to more frequent symbols (Huffman coding) further reduces expected cost — which pays off precisely when symbol frequencies follow a power law.

## Baselines

**SpectralClustering / Modularity (social dimensions; Tang & Liu).** Eigendecompose a graph matrix (normalized Laplacian or modularity matrix) to produce latent node features for classification.

**EdgeCluster.** Clusters edges to produce sparse social-dimension features scalable to larger graphs.

**Collective/relational classifiers (wvRN, majority voting).** Propagate labels through the graph using neighbor label distributions, using labels at inference.

**Skip-gram / neural language models (Mikolov et al.).** A learner that maps symbols in sequences to continuous vectors via a windowed, order-independent prediction objective, trained by SGD with hierarchical softmax. It operates on sequences of symbols.

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

# --- producing symbol sequences from a graph ---
def make_corpus(G, ...):
    # TODO: produce a corpus of symbol sequences from the graph structure
    pass

# --- the overall feature learner ---
def learn_representations(G, dim, window):
    corpus = make_corpus(G)           # TODO: fill this in
    return skipgram(corpus, dim=dim, window=window)
```
