# DeepWalk — Online Learning of Social Representations

## Problem

Learn low-dimensional, continuous feature vectors ("social representations") for the vertices of a network, unsupervised and from structure alone, so that an off-the-shelf classifier can use them for multi-label vertex classification — scalably, and online (updatable from local information).

## Key idea

Generalize neural language models from text to graphs. Short truncated random walks are treated as "sentences" and vertices as "words"; a Skip-gram language model with hierarchical softmax then learns vertex embeddings that place co-occurring (same-community) vertices close together. The license for the analogy is statistical: on a scale-free graph the frequency of vertices in short random walks follows a power law, exactly like word frequencies (Zipf) in natural language.

## Method

For each center vertex v_i in a walk and window w, minimize
`−log Pr({v_{i−w},…,v_{i−1}, v_{i+1},…,v_{i+w}} | Φ(v_i))`,
with Φ the |V|×d embedding matrix; Skip-gram uses the center vertex to predict its windowed context on both sides, order-independently.

Algorithm (γ = walks per vertex, t = walk length, w = window, d = dimension):
1. Initialize Φ uniformly; build a binary (Huffman) tree T over V.
2. For γ passes: shuffle V; for each vertex v_i, generate a uniform random walk W_{v_i} of length t (each step → a uniformly random neighbor of the last vertex), then run a Skip-gram update over W_{v_i}.
3. Skip-gram update: for each center v_j and each context u_k in its window, take an SGD step on `J = −log Pr(u_k | Φ(v_j))`.

Hierarchical softmax: vertices sit at the leaves of the tree; for a root-to-leaf path (b_0=root,…,b_{⌈log|V|⌉}=u_k),
`Pr(u_k | Φ(v_j)) = Π_{l=1}^{⌈log|V|⌉} Pr(b_l | Φ(v_j))`,
each factor a binary logistic classifier at an internal node — O(log|V|) instead of O(|V|). Huffman coding gives frequent (hub) vertices shorter paths, which pays off under the power-law frequency distribution.

Optimization: SGD with backprop; learning rate starts at 2.5% and decays linearly with the number of vertices seen. Parameters {Φ, T} are each O(d|V|); total cost O(dwL log|V|) with L = γt; trivially parallelizable (asynchronous SGD) and online. Reported best hyperparameters: γ = 80, w = 10, d = 128, walk length t = 40.

## Code

```python
import random
from gensim.models import Word2Vec

def random_walk(G, start, length):
    """Uniform truncated random walk: each step jumps to a random neighbor."""
    path = [start]
    while len(path) < length:
        nbrs = G.neighbors(path[-1])
        if not nbrs:
            break
        path.append(random.choice(nbrs))
    return path

def build_corpus(G, num_walks, walk_length):
    """gamma passes; shuffle vertices each pass; one walk per vertex."""
    walks, nodes = [], list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:
            walks.append([str(n) for n in random_walk(G, v, walk_length)])
    return walks

def deepwalk(G, dim=128, num_walks=80, walk_length=40, window=10, workers=1):
    """Walks -> Skip-gram (sg=1) with hierarchical softmax (hs=1). Returns Phi."""
    walks = build_corpus(G, num_walks, walk_length)
    model = Word2Vec(walks, vector_size=dim, window=window,
                     sg=1, hs=1, min_count=0, workers=workers)
    return model.wv
```
