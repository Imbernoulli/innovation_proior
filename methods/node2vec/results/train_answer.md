I propose the canonical method name node2vec, a scalable algorithm for learning feature representations of nodes in networks. The core goal is to map every node of a graph into a low-dimensional vector such that nodes that are similar in the network are close in the embedding space, where the operative notion of similarity can be tuned to the network and task at hand. The method reuses the Skip-gram objective from word2vec: a node should predict the nodes that appear in its context. What distinguishes node2vec from earlier approaches is how it manufactures those contexts from a non-linear graph.

In networks, node similarity splits into two distinct concepts. Homophily says that nodes in the same densely connected community should be embedded close together, even if they are several hops apart. Structural equivalence says that nodes that play the same role, such as hubs or bridges, should be close together even if they are far apart and share no neighbors. A method that bakes in only one of these notions will fail on tasks that reward the other. Prior work fixed a single sampling strategy: DeepWalk uses a uniform random walk, which is one fixed blend of similarities, while LINE uses immediate neighborhoods, which is purely local. node2vec introduces a flexible sampler that can interpolate continuously between these extremes.

The sampler is a second-order biased random walk controlled by two parameters, p and q. Suppose the walk has just traversed the edge from t to v and is now choosing the next node x among the neighbors of v. The choice depends on the shortest-path distance from t to x, which can only be 0, 1, or 2 because v is adjacent to both t and x. If x equals t, the distance is 0 and the walk would backtrack. If x is also a neighbor of t, the distance is 1 and the walk stays local. Otherwise the distance is 2 and the walk moves outward. The unnormalized transition probability is the edge weight between v and x multiplied by a bias alpha_pq(t, x) that equals 1 over p in the backtrack case, 1 in the local case, and 1 over q in the outward case.

This second-order structure is essential because a first-order walk has no sense of direction; it only knows where it is, not where it came from. By conditioning on the previous node, the walk can express stay-near-home versus venture-out behavior. The return parameter p controls how often the walk immediately backtracks. A small p keeps the walk oscillating near the start node, producing very local contexts, while a large p discourages backtracking and encourages exploration. The in-out parameter q is the main dial between breadth-first and depth-first exploration. When q is greater than 1, outward steps are penalized and the walk stays near the previously visited node, giving low-variance local neighborhoods that favor structural equivalence. When q is less than 1, outward steps are rewarded and the walk pushes away from where it came from, reaching distant communities and favoring homophily. Setting p equal to q equal to 1 recovers the uniform random walk used by DeepWalk.

Efficiency comes from precomputing the transition distributions and using alias sampling. For every directed edge from t to v we build an alias table over the neighbors of v according to the biased probabilities; after linear preprocessing in the size of the edge set, each step of each walk is a constant-time draw. To avoid start-node bias, node2vec simulates r walks of length l from every node. These sequences are then fed to Skip-gram with negative sampling, producing a d-dimensional vector for every node. For link prediction, pair features are derived from node features with component-wise operators such as the average, the Hadamard product, weighted L1 distance, or weighted L2 distance; the Hadamard product tends to work best in practice.

For downstream tasks, the hyperparameters p and q are typically tuned by grid search over values like 0.25, 0.5, 1, 2, 4 using a small amount of labeled data. Common defaults are d equal to 128, r equal to 10 walks per node, l equal to 80, and context size k equal to 10. Because the three phases, preprocessing the transition probabilities, simulating the walks, and running Skip-gram stochastic gradient descent, are independent, they can each be parallelized.

The following Python script illustrates the complete pipeline on a tiny graph. It defines an alias sampler, implements the biased second-order walk, simulates walks, and learns embeddings with a simplified Skip-gram-style objective.

```python
import numpy as np
import random

# Tiny graph: two cliques connected by a bridge node.
graph = {
    0: [1, 2, 3, 7],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [0, 1, 2, 4],
    4: [3, 5, 6, 7],
    5: [4, 6, 7],
    6: [4, 5, 7],
    7: [0, 4, 5, 6]
}


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)
    smaller, larger = [], []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(J, q):
    K = len(J)
    kk = int(np.random.randint(K))
    if np.random.rand() < q[kk]:
        return kk
    return J[kk]


def sigmoid(z):
    # Numerically stable sigmoid.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


class Node2Vec:
    def __init__(self, graph, p, q):
        self.graph = graph
        self.p = p
        self.q = q
        self.alias_nodes = {}
        self.alias_edges = {}

    def get_alias_edge(self, t, v):
        nbrs = self.graph[v]
        probs = []
        for x in nbrs:
            if x == t:
                probs.append(1.0 / self.p)
            elif x in self.graph.get(t, []):
                probs.append(1.0)
            else:
                probs.append(1.0 / self.q)
        Z = sum(probs)
        return alias_setup([pr / Z for pr in probs])

    def preprocess_transition_probs(self):
        for u in self.graph:
            nbrs = self.graph[u]
            probs = [1.0 / len(nbrs)] * len(nbrs)
            self.alias_nodes[u] = alias_setup(probs)
        for t in self.graph:
            for v in self.graph[t]:
                self.alias_edges[(t, v)] = self.get_alias_edge(t, v)

    def node2vec_walk(self, length, start):
        walk = [start]
        while len(walk) < length:
            cur = walk[-1]
            nbrs = self.graph[cur]
            if not nbrs:
                break
            if len(walk) == 1:
                J, q = self.alias_nodes[cur]
            else:
                J, q = self.alias_edges[(walk[-2], cur)]
            walk.append(nbrs[alias_draw(J, q)])
        return walk

    def simulate_walks(self, num_walks, walk_length):
        walks = []
        nodes = list(self.graph.keys())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length, node))
        return walks


def train_node2vec(graph, p=1.0, q=1.0, dim=16, num_walks=20,
                   walk_length=10, window=2, lr=0.01, epochs=100):
    n2v = Node2Vec(graph, p, q)
    n2v.preprocess_transition_probs()
    walks = n2v.simulate_walks(num_walks, walk_length)
    nodes = list(graph.keys())
    n_nodes = len(nodes)
    node_to_id = {n: i for i, n in enumerate(nodes)}
    W = np.random.randn(n_nodes, dim) * 0.01
    C = np.random.randn(n_nodes, dim) * 0.01

    for _ in range(epochs):
        np.random.shuffle(walks)
        for walk in walks:
            for i, u in enumerate(walk):
                uid = node_to_id[u]
                lo = max(0, i - window)
                hi = min(len(walk), i + window + 1)
                for j in range(lo, hi):
                    if i == j:
                        continue
                    vid = node_to_id[walk[j]]
                    pos_score = sigmoid(np.dot(W[uid], C[vid]))
                    neg = np.random.randint(0, n_nodes)
                    neg_score = sigmoid(np.dot(W[uid], C[neg]))
                    W[uid] += lr * ((1.0 - pos_score) * C[vid] - neg_score * C[neg])
                    C[vid] += lr * (1.0 - pos_score) * W[uid]
                    C[neg] -= lr * neg_score * W[uid]
    return W, node_to_id


W, mapping = train_node2vec(graph, p=1.0, q=0.5, dim=16)
for node in sorted(graph):
    print(node, W[mapping[node]][:4])
```

This script is intentionally small so that every step of the method is visible: the alias sampler, the second-order transition bias, the walk simulation, and the negative-sampling update. On the two-clique graph, setting q below 1 will tend to push the walk across the bridge and reveal the community structure, while raising q above 1 will keep the walk local and emphasize structural roles. In practice one would use a much larger graph, more walks, and a production Skip-gram implementation, but the mechanics remain identical.
