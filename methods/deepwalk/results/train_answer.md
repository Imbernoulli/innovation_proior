The problem is to learn a low-dimensional, continuous feature vector for every vertex of a network, using only the graph structure and no labels, so that an ordinary classifier can later predict node labels from these vectors. Networks are naturally described by sparse, discrete adjacency information, which standard statistical learners do not generalize from well, especially when only a small fraction of nodes are labeled. Relational-classification methods attack the label-prediction task directly through iterative inference over a Markov network, but they entangle feature construction with the specific classification problem and produce no reusable, task-independent representation. Spectral and modularity-based "social dimension" methods do produce continuous latent features, yet they rely on a global eigendecomposition of a graph matrix, which does not scale to very large networks and cannot be updated incrementally when the graph changes.

A better starting point is to notice that neural language models already solve an analogous problem: they map discrete symbols — words — to dense continuous vectors so that symbols that appear in similar contexts end up close in the vector space. The remaining obstacle is that a language model consumes sequences, while a graph is not a sequence. The key observation is that short random walks on a scale-free network generate a vertex-frequency distribution that follows a power law, just like word frequencies in natural-language text follow Zipf's law. This statistical match licenses treating short random walks as "sentences" and vertices as "words," so the same language-model machinery can be applied to graphs. Random walks are also a natural probe of local community structure: a short walk from a vertex tends to stay inside its community, and each step uses only the neighbors of the current node, so the method can be updated locally and streams naturally.

The method is DeepWalk. It first builds a corpus of short, uniform random walks: for several passes over the vertex set, it shuffles the vertices each pass and generates one truncated walk per vertex, at each step moving to a uniformly chosen neighbor. These walks are then fed to a Skip-gram model. Skip-gram reverses the usual language-modeling direction: it uses a center vertex to predict the set of vertices appearing within a fixed window on both sides, and it ignores the order of context vertices. The order-independence is a good fit for random walks, because co-occurrence inside a short walk is an unordered notion of neighborhood rather than a strict sequence. The objective for a center vertex v and context vertex u is to minimize negative log probability of u given the embedding of v.

At network scale, a flat softmax over the full vertex vocabulary is infeasible, because every prediction would require O(|V|) normalization. DeepWalk instead uses hierarchical softmax: all vertices are placed at the leaves of a binary tree, and the probability of a target vertex factors into a product of binary logistic decisions along the root-to-leaf path, reducing each prediction to O(log|V|). The tree is built with Huffman coding, so frequent hub vertices receive shorter paths. Because the walk corpus has a power-law frequency distribution, this substantially reduces the average prediction cost. The parameters — the vertex embedding matrix and the internal node classifiers — are optimized with plain SGD, using a learning rate that starts around 2.5% and decays linearly as more vertices are processed. The total cost is near-linear in the total walk length and trivially parallelizable, since walks can be processed asynchronously.

```python
import random
from gensim.models import Word2Vec


def random_walk(G, start, length):
    """Uniform truncated random walk."""
    path = [start]
    while len(path) < length:
        nbrs = G.neighbors(path[-1])
        if not nbrs:
            break
        path.append(random.choice(nbrs))
    return path


def build_corpus(G, num_walks, walk_length):
    """Generate gamma passes of short random walks, one per vertex per pass."""
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:
            walks.append([str(n) for n in random_walk(G, v, walk_length)])
    return walks


def deepwalk(G, dim=128, num_walks=80, walk_length=40, window=10, workers=1):
    """DeepWalk: random-walk corpus -> Skip-gram with hierarchical softmax."""
    walks = build_corpus(G, num_walks, walk_length)
    model = Word2Vec(
        walks,
        vector_size=dim,
        window=window,
        sg=1,
        hs=1,
        min_count=0,
        workers=workers,
    )
    return model.wv
```
