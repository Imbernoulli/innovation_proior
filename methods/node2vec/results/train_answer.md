I propose the canonical method name node2vec, a scalable algorithm for learning feature representations of nodes in networks. The core goal is to map every node of a graph into a low-dimensional vector such that nodes that are similar in the network are close in the embedding space, where the operative notion of similarity can be tuned to the network and task at hand. The method reuses the Skip-gram objective from word2vec: a node should predict the nodes that appear in its context. What distinguishes node2vec from earlier approaches is how it manufactures those contexts from a non-linear graph.

In networks, node similarity splits into two distinct concepts. Homophily says that nodes in the same densely connected community should be embedded close together, even if they are several hops apart. Structural equivalence says that nodes that play the same role, such as hubs or bridges, should be close together even if they are far apart and share no neighbors. A method that bakes in only one of these notions will fail on tasks that reward the other. Prior work fixed a single sampling strategy: DeepWalk uses a uniform random walk, which is one fixed blend of similarities, while LINE uses immediate neighborhoods, which is purely local. node2vec introduces a flexible sampler that can interpolate continuously between these extremes.

The sampler is a second-order biased random walk controlled by two parameters, p and q. Suppose the walk has just traversed the edge from t to v and is now choosing the next node x among the neighbors of v. The choice depends on the shortest-path distance from t to x, which can only be 0, 1, or 2 because v is adjacent to both t and x. If x equals t, the distance is 0 and the walk would backtrack. If x is also a neighbor of t, the distance is 1 and the walk stays local. Otherwise the distance is 2 and the walk moves outward. The unnormalized transition probability is the edge weight between v and x multiplied by a bias alpha_pq(t, x) that equals 1 over p in the backtrack case, 1 in the local case, and 1 over q in the outward case.

This second-order structure is essential because a first-order walk has no sense of direction; it only knows where it is, not where it came from. By conditioning on the previous node, the walk can express stay-near-home versus venture-out behavior. The return parameter p controls how often the walk immediately backtracks. A small p keeps the walk oscillating near the start node, producing very local contexts, while a large p discourages backtracking and encourages exploration. The in-out parameter q is the main dial between breadth-first and depth-first exploration. When q is greater than 1, outward steps are penalized and the walk stays near the previously visited node, giving low-variance local neighborhoods that favor structural equivalence. When q is less than 1, outward steps are rewarded and the walk pushes away from where it came from, reaching distant communities and favoring homophily. Setting p equal to q equal to 1 recovers the uniform random walk used by DeepWalk.

Efficiency comes from precomputing the transition distributions and using alias sampling. For every directed edge from t to v we build an alias table over the neighbors of v according to the biased probabilities; after linear preprocessing in the size of the edge set, each step of each walk is a constant-time draw. To avoid start-node bias, node2vec simulates r walks of length l from every node. These sequences are then fed to Skip-gram with negative sampling, producing a d-dimensional vector for every node. For link prediction, pair features are derived from node features with component-wise operators such as the average, the Hadamard product, weighted L1 distance, or weighted L2 distance; the Hadamard product tends to work best in practice.

For downstream tasks, the hyperparameters p and q are typically tuned by grid search over values like 0.25, 0.5, 1, 2, 4 using a small amount of labeled data. Common defaults are d equal to 128, r equal to 10 walks per node, l equal to 80, and context size k equal to 10. Because the three phases, preprocessing the transition probabilities, simulating the walks, and running Skip-gram stochastic gradient descent, are independent, they can each be parallelized.

The following implementation is the one I actually use. It takes a weighted networkx graph G, precomputes the alias tables for the first step of a walk (edge weight only, since there is no previous node yet) and for every directed edge (the biased second-order table keyed by the (t, v) pair that was just traversed), simulates r walks of length l from every node by drawing from the appropriate table at each step with the O(1) alias-method sampler, and then hands the resulting corpus of walks straight to gensim's Word2Vec running in skip-gram mode (`sg=1`) with negative sampling to produce the node embeddings. A final helper turns a pair of node vectors into a single edge feature for link prediction using the average, Hadamard, or weighted-L1/L2 operator.

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

The `alias_setup`/`alias_draw` pair is the standard O(1)-per-draw alias-method sampler that the preprocessing step feeds: `preprocess_transition_probs` builds one table per node for the very first step of any walk, where there is no previous node to bias against, and one table per directed edge for every subsequent step, where the bias `alpha_pq` kicks in. `node2vec_walk` then does nothing more than look up the correct table, given whether it is at the first step or has a predecessor, and draw the next node in constant time. Running this on a real graph, lowering q below 1 pushes the walks outward across bridges between communities and yields embeddings that reflect homophily, while raising q above 1 keeps the walks local and yields embeddings that reflect structural role instead — exactly the two regimes the bias was built to interpolate between.
