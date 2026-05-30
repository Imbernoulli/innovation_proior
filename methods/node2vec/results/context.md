# Context

## Research question

Any supervised learning task on a network — predicting node labels (user interests, protein functions) or predicting links (which two nodes should be connected) — first needs a feature vector for each node, and often for each pair of nodes. Hand-engineering these features (centralities for nodes, Adamic-Adar scores for edges) is tedious, requires expert knowledge, and produces features tuned to one task that fail to transfer to another. The question is whether node (and edge) feature representations can be *learned* in an unsupervised, task-independent way that (a) scales to networks with millions of nodes, and (b) is flexible enough to capture the *different kinds* of node similarity that matter in different networks and tasks. A good solution would map each node to a low-dimensional vector such that nodes that are "similar in the network" land close together — where the operative notion of similarity can be tuned to the network at hand.

## Background

**Two notions of node similarity (diagnostic facts about networks).** Prediction tasks on networks shuttle between two distinct equivalences:
- *Homophily:* nodes that are densely interconnected and belong to the same community should be embedded close together. This is a *macroscopic* property — it is about community membership, which one only sees by exploring a sizable, possibly distant portion of the graph.
- *Structural equivalence:* nodes that play the same *role* — hubs, bridges — should be embedded close together, even if they are far apart in the graph and share no neighbors. This is a *microscopic* property — a node's role is determined by the structure of its immediate neighborhood, and it does *not* require connectivity between the equivalent nodes.

Real networks exhibit a mixture: some node distinctions are community-driven, others role-driven. A representation method that bakes in only one notion (e.g., spectral clustering implicitly assumes homophily through graph cuts) generalizes poorly across diverse networks.

**Neighborhood sampling as local search — BFS vs DFS.** If one fixes a budget of k neighbor nodes to sample for a source u, the two extreme sampling strategies are:
- *Breadth-first sampling (BFS):* sample only immediate neighbors. This gives an accurate microscopic characterization of u's local neighborhood — and because the same near nodes recur across samples, it has low variance — which is exactly what is needed to read off structural roles (hub, bridge). But for any fixed budget it explores only a tiny portion of the graph.
- *Depth-first sampling (DFS):* sample nodes at increasing distance from u. This gives a macroscopic view, reaching distant communities — useful for inferring homophily — but with a fixed budget it has high variance, and very distant sampled nodes may be only weakly representative of u.

So BFS-like neighborhoods favor structural-equivalence embeddings and DFS-like neighborhoods favor homophily embeddings. The two extremes are both rigid; what is missing is a knob that interpolates between them per network.

**The distributional hypothesis and Skip-gram.** In NLP, the distributional hypothesis says words in similar contexts have similar meanings. Skip-gram (word2vec) operationalizes this: learn a vector for each word that maximizes the probability of its context words within a sliding window, optimized by SGD with negative sampling. Because text is linear, the "context window" is unambiguous. The analogy to networks requires turning a (non-linear) graph into sequences of nodes — and the *sampling strategy* that produces those sequences is then the crucial, underdetermined design choice.

**Alias sampling.** A method to draw from a fixed discrete distribution over n outcomes in O(1) time per draw after O(n) preprocessing. Relevant because if per-step transition probabilities along a walk can be precomputed, each step of the walk is O(1).

## Baselines

**DeepWalk (Perozzi et al. 2014).** Treats the network as a "document": generate node sequences by *uniform* (first-order) truncated random walks — at each step move to a uniformly random neighbor — then feed these sequences to Skip-gram (with hierarchical softmax) to learn node vectors. Gap: the uniform walk is a single, rigid sampling strategy with no control over BFS-like vs DFS-like exploration, so the embeddings cannot be steered toward homophily or structural equivalence as the network demands.

**LINE (Tang et al. 2015).** Directly optimizes objectives preserving first-order proximity (adjacent nodes) and second-order proximity (shared-neighbor nodes), essentially a BFS-style, immediate-neighborhood objective with a context of one. Gap: rigid and local; no mechanism for macroscopic, DFS-style exploration.

**Spectral methods (PCA, Laplacian eigenmaps, IsoMap).** Embed nodes via eigendecomposition of a graph matrix (adjacency or Laplacian). Gap: eigendecomposition does not scale to large networks without quality-degrading approximation, and the objective bakes in fixed assumptions (e.g., homophily via graph cuts) that do not generalize across networks.

**Skip-gram / word2vec (Mikolov et al. 2013).** The neighborhood-preserving likelihood objective and the negative-sampling SGD machinery that the network methods reuse. It leaves open, for networks, the definition of a node's "context."

**Link-prediction heuristics (Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment).** Hand-designed scores for whether two nodes should be linked. Gap: fixed, not learned, and limited to the link-prediction task.

## Evaluation settings

- **Tasks:** multi-label node classification (train a one-vs-rest logistic classifier on the learned node features; report Macro-F1 / Micro-F1) and link prediction (hide a fraction of edges, learn features on the rest, score candidate node pairs via an edge feature plus a classifier; report AUC).
- **Datasets:** BlogCatalog, protein-protein interaction (PPI), Wikipedia word co-occurrence (POS tags), Facebook, arXiv collaboration; Erdős–Rényi graphs (constant average degree) up to 1M nodes for scalability.
- **Baselines:** DeepWalk, LINE, spectral clustering; for link prediction also Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment.
- **Protocol:** for fairness across embedding methods, the sampling budget (total samples generated) is held equal. Robustness is probed under noisy/missing edges.

## Code framework

The primitives that already exist: a graph object with weighted edges and neighbor lists; a uniform random-walk routine; the Skip-gram/word2vec trainer (SGD with negative sampling) that consumes sequences of tokens and emits d-dimensional vectors; alias sampling for O(1) draws from a precomputed discrete distribution. The slots below are what a flexible network feature-learning method would fill in.

```python
import numpy as np

class Graph:                       # weighted graph (exists)
    def neighbors(self, node): ...
    def weight(self, u, v): ...

def alias_setup(probs):            # O(n) preprocess for O(1) sampling (exists)
    ...
def alias_draw(J, q):              # O(1) draw (exists)
    ...

class NodeFeatureLearner:
    def __init__(self, G, **params):
        self.G = G
        # TODO: any parameters that control HOW neighborhoods are sampled

    def preprocess_transition_probs(self):
        # TODO: the transition rule for the walk -- what biases the next step?
        #       (uniform? edge weight? something that can be tuned BFS<->DFS?)
        pass

    def walk(self, start, length):
        # TODO: extend a sequence from start by repeatedly sampling the next node
        pass

    def simulate_walks(self, num_walks, length):
        # generate node sequences from every node (offsetting start-node bias)
        walks = []
        # TODO: for each of num_walks rounds, for each node, append self.walk(...)
        return walks

    def learn(self, num_walks, length, dim, context_size):
        walks = self.simulate_walks(num_walks, length)
        # known: Skip-gram with negative sampling over the node sequences
        return skipgram(walks, dim=dim, window=context_size, negative_sampling=True)

# --- edge features from node features ---
def edge_feature(fu, fv):
    # TODO: a binary operator combining two node vectors into a pair vector
    pass
```
