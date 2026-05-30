I want to learn a vector for every node of a graph, unsupervised, so that downstream classifiers and link predictors can just consume those vectors instead of hand-crafted centralities. The piece I'm sure about is the learning machinery: Skip-gram already does exactly this for words. Give it a word and the words around it in a window, and it learns vectors that make a node predict its context, optimized cheaply with negative-sampling SGD. Formally, for a source u with a context set N_S(u), I want to maximize the log-probability of the context given u's vector, Σ_u log Pr(N_S(u) | f(u)). Assume the context nodes are conditionally independent given f(u), and model each pairwise term as a softmax over the dot product of feature vectors, Pr(n_i | f(u)) = exp(f(n_i)·f(u)) / Σ_v exp(f(v)·f(u)). Then the objective collapses to Σ_u [ −log Z_u + Σ_{n_i ∈ N_S(u)} f(n_i)·f(u) ], with the per-node partition function Z_u that negative sampling lets me approximate. None of this is the hard part — it's lifted straight from word2vec.

The hard part is that text is *linear* and a graph is not. For a word, "context" is unambiguous: the few words to the left and right in the sentence. For a node, what is the context? There's no natural sliding window. I have to *manufacture* sequences of nodes from the graph and call the co-occurring nodes the context. And the moment I say that, I realize the whole method lives or dies on *how* I sample those sequences — that choice is wide open, and different choices will give completely different embeddings.

So let me think about what "similar nodes should be close" even means in a network, because that's what the sampling has to capture. Staring at a graph, there are two genuinely different notions of similarity, and they pull in opposite directions. One is homophily: two nodes are similar if they're in the same densely-connected community — they may be several hops apart, but they belong to the same cluster. The other is structural equivalence: two nodes are similar if they play the same *role* — both are hubs of their respective communities, or both are bridges — even though they might be on opposite sides of the graph and share not a single neighbor. These aren't the same and they aren't even correlated: homophily is about *who you're connected to*, structural equivalence is about *the shape of your local wiring*. Real networks have both, in different proportions, and which one matters depends on the prediction task. That's the thing prior methods miss: DeepWalk samples sequences with a plain uniform random walk — move to a random neighbor each step — and LINE optimizes immediate-neighbor proximity directly. Each is *one fixed* sampling rule, so each captures one fixed blend of these similarities, with no way to dial it.

Let me frame the sampling as a search problem and find the two extremes, because the extremes will tell me what the knob has to span. Fix a budget of k context nodes for a source u. One extreme is breadth-first: sample only u's immediate neighbors. That gives a sharp, low-variance picture of u's local wiring — and because I keep revisiting the same near nodes, the picture is statistically stable — which is precisely what you need to read off a structural role like "hub" or "bridge." But with budget k, BFS sees a tiny patch of the graph. The other extreme is depth-first: sample nodes at ever-increasing distance from u. That reaches far, sketching the macro-structure — which community u sits in — which is what homophily needs. But with a fixed budget DFS is high-variance, and the far-flung nodes it grabs may barely relate to u. So: BFS-like contexts → structural equivalence; DFS-like contexts → homophily. The two extremes are rigid. What I want is a sampling procedure with a dial that smoothly interpolates between BFS-ish and DFS-ish exploration.

Why not literally run BFS or DFS to build the contexts, then? Cost. To respect the budget k for every source I'd have to run the search k|V| times, and DFS in particular has nasty preprocessing overhead. Random walks fix this beautifully: a single walk of length l, by the Markov property, hands me context sets for *many* source nodes at once — a length-l walk gives k context samples for l−k of its nodes — so samples get reused across sources and the effective cost per sample plummets. And if I can precompute the per-step transition distribution, each step is O(1) with alias sampling. So I want to stay inside the random-walk framework and get my BFS↔DFS dial by *biasing* the walk, not by abandoning walks for explicit search.

Now, how do I bias a walk so it can behave like either BFS or DFS? The naive bias is the edge weight: step to neighbor x of the current node v with probability proportional to w_vx. But that's a *first-order* walk — the next step depends only on where I am now, v — and that's fundamentally too weak to express what I need. "BFS-like" means "stay near where I started"; "DFS-like" means "keep venturing outward." Both are statements about *direction of travel*, and direction only exists relative to where I just *came from*. A walk that knows only its current node v has no sense of direction. So I need the walk to remember the previous node t — make it second-order Markov, conditioning the next step on the edge (t,v) I just traversed.

With t in hand, let me see what structure the next step has. I'm at v, having come from t, choosing the next node x among v's neighbors. The key quantity is the shortest-path distance d_tx between t and the candidate x. Because v is adjacent to both t and x, d_tx can only be one of three values:
- d_tx = 0: x is t itself — I'd be stepping straight back to where I came from.
- d_tx = 1: x is also a neighbor of t — it's a node "around" t, in the same local pocket; stepping there keeps me circling t's neighborhood.
- d_tx = 2: x is *not* connected to t — it's strictly farther out; stepping there moves me away.
Three cases, and they map *exactly* onto the behaviors I want to control: backtrack, stay-local, venture-out. So a second-order walk with a per-case multiplier is necessary and — since there are only these three cases — *sufficient* to span BFS↔DFS. I'll set the unnormalized transition π_vx = α_pq(t,x)·w_vx, keeping the edge weight as a base and multiplying by a search bias

  α_pq(t,x) = 1/p  if d_tx = 0  (return to t),
            = 1    if d_tx = 1  (common neighbor of t),
            = 1/q  if d_tx = 2  (move away from t).

Two parameters, p and q, one per controllable direction (the d=1 case is the neutral reference, fixed at 1). Note that when p = q = 1, α ≡ 1 and π_vx = w_vx — the plain uniform/weighted walk — so this *contains* DeepWalk as a special case; I'm generalizing, not replacing.

Let me reason out what each parameter does so I know I've wired the cases correctly. The return parameter p sits on the d=0 (backtrack) case. If p is large — bigger than max(q,1) — then 1/p is small, so I'm reluctant to step right back to t, which avoids redundantly re-sampling the last two-hop region and pushes the walk to move on; moderate exploration. If p is small — below min(q,1) — then 1/p is large, the walk eagerly backtracks, and it ends up oscillating in a tight pocket near the start node u; that keeps the context local. The in-out parameter q sits on the d=2 (venture-out) case and is the real BFS↔DFS dial. If q > 1, then 1/q < 1, so outward steps are discouraged and the walk prefers d=1 nodes that hug t's locality — a local, inward, BFS-like exploration that surfaces structural-equivalence neighborhoods. If q < 1, then 1/q > 1, outward steps are favored, the walk pushes away from t — a DFS-like outward exploration that surfaces homophilous community structure. And it does this *inside* the walk framework, so I keep the cheap sample-reuse and O(1) sampling; the sampled nodes aren't at strictly increasing distance the way textbook DFS demands, but they trend outward, which is what matters. So p and q together let me slide continuously from BFS-ish to DFS-ish, picking up whatever mixture of homophily and structural equivalence a given network rewards.

The efficiency falls out cleanly. Since π depends only on (t, v, x) and the graph is fixed, I precompute, for every directed edge (t,v), the alias structure over v's neighbors; then each walk step is an O(1) alias draw. Storing immediate neighbors is O(|E|); storing the second-order interconnections (which neighbors of v are also neighbors of t) costs O(a²|V|) with a the average degree, small for real networks. To remove the bias from where a walk happens to start, I launch r walks of length l from *every* node. Three phases — precompute transition probabilities, simulate the walks, run Skip-gram SGD over them — each independently parallelizable.

One more thing I actually need, because half the tasks are about *edges*, not nodes: link prediction asks whether a pair (u,v) should be connected, so I need a feature vector for a *pair*. I'll bootstrap it from the node vectors with a simple binary operator that maps f(u), f(v) → g(u,v), defined for *any* pair whether or not an edge exists (the test set has both real and fake edges). Component-wise options: the average (f_i(u)+f_i(v))/2; the Hadamard product f_i(u)·f_i(v); the weighted-L1 |f_i(u)−f_i(v)|; the weighted-L2 |f_i(u)−f_i(v)|². These keep the pair representation at dimension d and let the same learned node vectors serve edge tasks; the symmetric Hadamard product tends to work best, but I'll keep the menu since the right operator can be picked per task.

Let me lay out the pieces, mirroring how the biased walk and its precompute actually go.

```python
import numpy as np

# ---------------------------------------------------------------
# Precompute the 2nd-order transition bias alpha_pq for every
# directed edge (t -> v): for each neighbor x of v, multiply the
# edge weight by 1/p (x == t, backtrack), 1 (x neighbor of t,
# stay local), or 1/q (x farther, venture out). Alias tables give
# O(1) sampling per step.
# ---------------------------------------------------------------
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
            else:                                   # d_tx = 2  -> move outward
                probs.append(w / self.q)
        Z = sum(probs)
        return alias_setup([pr / Z for pr in probs])

    def preprocess_transition_probs(self):
        self.alias_nodes = {}                       # 1st step (no previous node): weight only
        for u in self.G.nodes():
            probs = [self.G[u][x]['weight'] for x in sorted(self.G.neighbors(u))]
            Z = sum(probs)
            self.alias_nodes[u] = alias_setup([pr / Z for pr in probs])
        self.alias_edges = {}                       # subsequent steps: 2nd-order, keyed by (t, v)
        for (a, b) in self.G.edges():
            self.alias_edges[(a, b)] = self.get_alias_edge(a, b)
            self.alias_edges[(b, a)] = self.get_alias_edge(b, a)

    # ---- one biased walk of length l from start node u ----
    def node2vec_walk(self, l, u):
        walk = [u]
        while len(walk) < l:
            cur = walk[-1]
            nbrs = sorted(self.G.neighbors(cur))
            if not nbrs:
                break
            if len(walk) == 1:                      # first step: no previous node yet
                J, q = self.alias_nodes[cur]
            else:                                   # 2nd-order: condition on previous node
                prev = walk[-2]
                J, q = self.alias_edges[(prev, cur)]
            walk.append(nbrs[alias_draw(J, q)])
        return walk

    # ---- r walks from EVERY node (offset start-node bias) ----
    def simulate_walks(self, r, l):
        walks = []
        nodes = list(self.G.nodes())
        for _ in range(r):
            np.random.shuffle(nodes)
            for u in nodes:
                walks.append(self.node2vec_walk(l, u))
        return walks

# ---------------------------------------------------------------
# Full pipeline: precompute -> walks -> Skip-gram (negative sampling).
# p=q=1 recovers a uniform walk. d=128, r=10, l=80, context k=10.
# ---------------------------------------------------------------
def learn_features(G, d=128, r=10, l=80, k=10, p=1.0, q=1.0):
    n2v = Node2Vec(G, p, q)
    n2v.preprocess_transition_probs()
    walks = [[str(n) for n in w] for w in n2v.simulate_walks(r, l)]
    # Skip-gram over node "sentences": each node predicts nodes within window k
    model = Word2Vec(walks, vector_size=d, window=k, sg=1, negative=5,
                     min_count=0, epochs=1)
    return model.wv

# ---------------------------------------------------------------
# Edge features: combine two node vectors for link prediction.
# ---------------------------------------------------------------
def edge_feature(fu, fv, op='hadamard'):
    if op == 'average':   return (fu + fv) / 2
    if op == 'hadamard':  return fu * fv
    if op == 'l1':        return np.abs(fu - fv)
    if op == 'l2':        return (fu - fv) ** 2
```

So the chain: I wanted learned node vectors and the Skip-gram objective handed me the learner for free, leaving exactly one open question — how to define a node's "context" in a non-linear graph. Looking at what node similarity means revealed two incompatible notions, homophily and structural equivalence, and that BFS-like sampling captures the latter while DFS-like sampling captures the former; prior methods fixed a single rigid sampler and so couldn't span both. Wanting the cheapness of random walks but the steerability of BFS/DFS, I biased the walk — and realized "direction" (toward home vs outward) only exists relative to the previous node, forcing a second-order walk; the previous node t splits the next step into exactly three distance cases (0, 1, 2), so two multipliers 1/p and 1/q are necessary and sufficient to dial backtracking, locality, and outward exploration, recovering the uniform walk at p=q=1. Precompute the per-edge alias tables, walk r times from every node, feed the sequences to negative-sampling Skip-gram, and bootstrap edge features from node vectors with a component-wise operator.
