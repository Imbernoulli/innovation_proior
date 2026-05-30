# node2vec — Scalable Feature Learning for Networks

## Problem

Learn unsupervised, task-independent, low-dimensional feature vectors for the nodes (and pairs of nodes) of a network, scalably, in a way flexible enough to capture the different notions of node similarity — homophily (community membership) and structural equivalence (network role) — that matter in different networks and tasks.

## Key idea

Reuse the Skip-gram objective (a node should predict the nodes in its "context"), and make the *context-sampling* flexible. node2vec generates each node's context with a **second-order biased random walk** governed by two parameters, a return parameter p and an in-out parameter q, that smoothly interpolate between BFS-like (local → structural equivalence) and DFS-like (outward → homophily) exploration. Because the bias depends on the previously visited node, the walk can express "direction" relative to where it came from, which a first-order walk cannot. Setting p = q = 1 recovers a uniform random walk (DeepWalk).

## Method

Objective (Skip-gram for graphs), maximized by SGD with negative sampling:
`max_f Σ_u [ −log Z_u + Σ_{n_i∈N_S(u)} f(n_i)·f(u) ]`, with `Z_u = Σ_v exp(f(u)·f(v))`.

Biased walk: just traversed (t, v), now at v choosing next x among v's neighbors. Unnormalized transition `π_vx = α_pq(t,x)·w_vx`, where with `d_tx` = shortest-path distance from t to x (∈ {0,1,2}):
```
α_pq(t,x) = 1/p  if d_tx = 0   (x == t: return to previous node)
          = 1    if d_tx = 1   (x is a neighbor of t: stay local)
          = 1/q  if d_tx = 2   (x is farther from t: move outward)
```
- **Return p:** high p (> max(q,1)) discourages backtracking → exploration; low p (< min(q,1)) keeps the walk near the start.
- **In-out q:** q > 1 biases toward t's locality → BFS-like, structural equivalence; q < 1 biases outward → DFS-like, homophily.

Algorithm: precompute the alias tables for `π` over all directed edges (O(1) sampling per step); run r walks of length l from every node; train Skip-gram with context window k over the walk sequences. Three phases (precompute / walk / SGD) are each parallelizable. Space O(|E|) for neighbors plus O(a²|V|) for second-order interconnections (a = average degree); sample reuse gives effective cost O(l/(k(l−k))) per sample.

Edge features (for link prediction): combine node vectors with a component-wise binary operator — Average `(f_i(u)+f_i(v))/2`, Hadamard `f_i(u)·f_i(v)`, Weighted-L1 `|f_i(u)−f_i(v)|`, Weighted-L2 `|f_i(u)−f_i(v)|²` — defined for any pair (edge or not). Hadamard works best.

Defaults: d = 128, r = 10 walks/node, l = 80, context size k = 10, 1 epoch, negative sampling; p, q tuned by grid search over {0.25, 0.5, 1, 2, 4} via 10-fold CV on 10% labeled data.

## Code

```python
import numpy as np
from gensim.models import Word2Vec

class Node2Vec:
    def __init__(self, G, p, q):
        self.G, self.p, self.q = G, p, q

    def get_alias_edge(self, t, v):                 # arriving at v from t
        probs = []
        for x in sorted(self.G.neighbors(v)):
            w = self.G[v][x]['weight']
            if x == t:                              # d_tx = 0  -> return
                probs.append(w / self.p)
            elif self.G.has_edge(x, t):             # d_tx = 1  -> common neighbor of t
                probs.append(w)
            else:                                   # d_tx = 2  -> outward
                probs.append(w / self.q)
        Z = sum(probs)
        return alias_setup([pr / Z for pr in probs])

    def preprocess_transition_probs(self):
        self.alias_nodes = {}                       # first step: edge weight only
        for u in self.G.nodes():
            probs = [self.G[u][x]['weight'] for x in sorted(self.G.neighbors(u))]
            Z = sum(probs)
            self.alias_nodes[u] = alias_setup([pr / Z for pr in probs])
        self.alias_edges = {}                       # 2nd-order, keyed by (t, v)
        for (a, b) in self.G.edges():
            self.alias_edges[(a, b)] = self.get_alias_edge(a, b)
            self.alias_edges[(b, a)] = self.get_alias_edge(b, a)

    def node2vec_walk(self, l, u):
        walk = [u]
        while len(walk) < l:
            cur = walk[-1]
            nbrs = sorted(self.G.neighbors(cur))
            if not nbrs:
                break
            if len(walk) == 1:
                J, q = self.alias_nodes[cur]
            else:
                J, q = self.alias_edges[(walk[-2], cur)]
            walk.append(nbrs[alias_draw(J, q)])
        return walk

    def simulate_walks(self, r, l):
        walks, nodes = [], list(self.G.nodes())
        for _ in range(r):
            np.random.shuffle(nodes)
            for u in nodes:
                walks.append(self.node2vec_walk(l, u))
        return walks

def learn_features(G, d=128, r=10, l=80, k=10, p=1.0, q=1.0):
    n2v = Node2Vec(G, p, q)
    n2v.preprocess_transition_probs()
    walks = [[str(n) for n in w] for w in n2v.simulate_walks(r, l)]
    model = Word2Vec(walks, vector_size=d, window=k, sg=1, negative=5, min_count=0, epochs=1)
    return model.wv

def edge_feature(fu, fv, op='hadamard'):
    if op == 'average':  return (fu + fv) / 2
    if op == 'hadamard': return fu * fv
    if op == 'l1':       return np.abs(fu - fv)
    if op == 'l2':       return (fu - fv) ** 2
```
