I want to learn a vector for every node of a graph, unsupervised, so that downstream classifiers and link predictors can just consume those vectors instead of hand-crafted centralities. The piece I'm sure about is the learning machinery: Skip-gram already does exactly this for words. Give it a word and the words around it in a window, and it learns vectors that make a node predict its context, optimized cheaply with negative-sampling SGD. Formally, for a source u with a context set N_S(u), I want to maximize the log-probability of the context given u's vector, Σ_u log Pr(N_S(u) | f(u)). Assume the context nodes are conditionally independent given f(u), and model each pairwise term as a softmax over the dot product of feature vectors, Pr(n_i | f(u)) = exp(f(n_i)·f(u)) / Σ_v exp(f(v)·f(u)). Then the objective collapses to Σ_u [ −log Z_u + Σ_{n_i ∈ N_S(u)} f(n_i)·f(u) ], with the per-node partition function Z_u that negative sampling lets me approximate. None of this is the hard part — it's lifted straight from word2vec.

The hard part is that text is *linear* and a graph is not. For a word, "context" is unambiguous: the few words to the left and right in the sentence. For a node, what is the context? There's no natural sliding window. I have to *manufacture* sequences of nodes from the graph and call the co-occurring nodes the context. And the moment I say that, I realize the whole method lives or dies on *how* I sample those sequences — that choice is wide open, and different choices will give completely different embeddings.

So let me think about what "similar nodes should be close" even means in a network, because that's what the sampling has to capture. Staring at a graph, there are two genuinely different notions of similarity, and they pull in opposite directions. One is homophily: two nodes are similar if they're in the same densely-connected community — they may be several hops apart, but they belong to the same cluster. The other is structural equivalence: two nodes are similar if they play the same *role* — both are hubs of their respective communities, or both are bridges — even though they might be on opposite sides of the graph and share not a single neighbor. These aren't the same and they aren't even correlated: homophily is about *who you're connected to*, structural equivalence is about *the shape of your local wiring*. Real networks have both, in different proportions, and which one matters depends on the prediction task. Now look at what the existing samplers do against this split: DeepWalk samples sequences with a plain uniform random walk — move to a random neighbor each step — and LINE optimizes immediate-neighbor proximity directly. Each is *one fixed* sampling rule. A uniform walk has no parameter at all; LINE's objective is hard-wired to immediate neighbors. Neither has a knob, so each captures whatever single blend of these two similarities its fixed rule happens to produce, and on a task that rewards the other blend there is no recourse. If both notions genuinely live in real networks, a sampler I can't steer between them is leaving accuracy on the table — so the thing I actually need to design is the steering.

Let me frame the sampling as a search problem and find the two extremes, because the extremes will tell me what the knob has to span. Fix a budget of k context nodes for a source u. One extreme is breadth-first: sample only u's immediate neighbors. That gives a sharp, low-variance picture of u's local wiring — and because I keep revisiting the same near nodes, the picture is statistically stable — which is precisely what you need to read off a structural role like "hub" or "bridge." But with budget k, BFS sees a tiny patch of the graph. The other extreme is depth-first: sample nodes at ever-increasing distance from u. That reaches far, sketching the macro-structure — which community u sits in — which is what homophily needs. But with a fixed budget DFS is high-variance, and the far-flung nodes it grabs may barely relate to u. So: BFS-like contexts → structural equivalence; DFS-like contexts → homophily. The two extremes are rigid. What I want is a sampling procedure with a dial that smoothly interpolates between BFS-ish and DFS-ish exploration.

Why not literally run BFS or DFS to build the contexts, then? Cost. To respect the budget k for every source I'd have to run the search k|V| times, and DFS in particular has nasty preprocessing overhead. Random walks fix this beautifully: a single walk of length l, by the Markov property, hands me context sets for *many* source nodes at once — a length-l walk gives k context samples for l−k of its nodes — so samples get reused across sources and the effective cost per sample plummets. And if I can precompute the per-step transition distribution, each step is O(1) with alias sampling. So I want to stay inside the random-walk framework and get my BFS↔DFS dial by *biasing* the walk, not by abandoning walks for explicit search.

Now, how do I bias a walk so it can behave like either BFS or DFS? The naive bias is the edge weight: step to neighbor x of the current node v with probability proportional to w_vx. But that's a *first-order* walk — the next step depends only on where I am now, v — and that's fundamentally too weak to express what I need. "BFS-like" means "stay near where I started"; "DFS-like" means "keep venturing outward." Both are statements about *direction of travel*, and direction only exists relative to where I just *came from*. A walk that knows only its current node v has no sense of direction. So I need the walk to remember the previous node t — make it second-order Markov, conditioning the next step on the edge (t,v) I just traversed.

With t in hand, let me see what structure the next step has. I'm at v, having come from t, choosing the next node x among v's neighbors. The candidate x is described relative to *both* t and v: relative to v it's just a neighbor, but relative to t it could be near or far, and "near or far from where I came from" is precisely the directional information I was missing. The natural quantity to read that off is the shortest-path distance d_tx from t to the candidate x. So how many distinct values can d_tx take? I want to claim it's only ever 0, 1, or 2, and the argument is short: x is adjacent to v, and v is adjacent to t, so there is always a path t–v–x of length 2, which means d_tx ≤ 2 no matter what x is. That caps it at three possibilities — 0 (x is t itself), 1 (x and t are directly connected), 2 (x is two hops from t through v and no shorter way). Before I lean on that, I should make sure the bound is actually tight and I'm not miscounting — it's the fact the whole construction rests on. Let me just enumerate. Take a small graph: a home triangle {a,b,c} all mutually connected, c bridging to a far triangle {d,e,f}. I'll loop over every legal configuration — every v, every t adjacent to v, every x adjacent to v — compute the true BFS distance d_tx, and record the maximum:

```
CLAIM1 max d_tx over all valid (t,v,x): 2   d=2 example: (t=d, v=c, x=a)
```

The maximum over the entire graph is 2 — never 3 — so the three-case partition is exhaustive, not an approximation. (The d=2 witness makes sense: arriving at c from d, the neighbor a is two hops from d, the farthest anything can be.) Good, the bound is real. Now the three cases line up with three different things I might want the walk to do: d=0 is stepping straight back to t (backtrack), d=1 is moving to a node in t's immediate pocket (stay local), d=2 is moving to a node strictly farther from t (venture out). Three controllable behaviors, three cases — and since one of them can be the fixed reference, I need exactly two free multipliers to set the other two. So I'll set the unnormalized transition π_vx = α_pq(t,x)·w_vx, keeping the edge weight as a base and multiplying by a search bias

  α_pq(t,x) = 1/p  if d_tx = 0  (return to t),
            = 1    if d_tx = 1  (common neighbor of t),
            = 1/q  if d_tx = 2  (move away from t).

Two parameters, p and q, one per controllable direction (the d=1 case is the neutral reference, fixed at 1). Note that when p = q = 1, α ≡ 1 and π_vx = w_vx — the plain uniform/weighted walk — so this *contains* DeepWalk as a special case; I'm generalizing, not replacing.

Now I'm worried I've talked myself into a story without checking that these two knobs actually move the walk the way I'm imagining. The claims "q < 1 ventures out," "q > 1 hugs the locality," "small p backtracks," "p=q=1 is uniform" are all just words at this point. Let me make the transition distribution concrete on the same small graph and read the numbers. I sit at v = c having arrived from t = b. With unit weights, c's three neighbors are: b (d=0, the way back), a (d=1, the other corner of the home triangle), and d (d=2, the bridge outward). Normalizing α over those three for a few parameter settings:

```
q=0.5  outward favored (DFS-ish):   P(c->a)=0.250  P(c->b)=0.250  P(c->d)=0.500
q=2    local favored (BFS-ish):     P(c->a)=0.400  P(c->b)=0.400  P(c->d)=0.200
p=q=1  uniform (DeepWalk):          P(c->a)=0.333  P(c->b)=0.333  P(c->d)=0.333
```

This is exactly the behavior I was hoping for, and now I can see it rather than assert it. At q = 0.5 the outward node d carries half the mass — the walk is twice as likely to push across the bridge into new territory as to do anything else, which is the DFS-ish, community-spanning behavior homophily wants. At q = 2 that same outward step drops to 0.20 while the two home-triangle nodes a and b absorb 0.40 each — the walk now prefers to stay in the local pocket, the BFS-ish behavior that surfaces structural roles. And p = q = 1 gives a flat 1/3 each, numerically recovering the uniform walk — so the special-case claim isn't just algebra, the distribution really is the DeepWalk one. Then for p alone, holding q = 1, the return probability P(c → b) as p sweeps:

```
p=0.25: P(c->b)=0.667    p=1: P(c->b)=0.333    p=4: P(c->b)=0.111
```

Small p doubles the chance of bouncing straight back to t, so the walk oscillates in a tight pocket near where it started, keeping the context local; large p cuts backtracking to a ninth, so the walk is pushed onward and explores. That matches the role I assigned p. One honest caveat the numbers don't let me overstate: the outward bias makes individual *steps* more likely to increase distance, but it's not textbook DFS — the walk can still double back later, so the sampled nodes only *trend* outward rather than marching at strictly increasing distance. For getting a macroscopic, homophily-flavored context cheaply, that trend is what matters. So p and q together let me slide continuously from BFS-ish to DFS-ish, picking up whatever mixture of homophily and structural equivalence a given network rewards, while staying inside the walk framework with its cheap sample-reuse and O(1) sampling.

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

So the chain: I wanted learned node vectors and the Skip-gram objective handed me the learner for free, leaving exactly one open question — how to define a node's "context" in a non-linear graph. Looking at what node similarity means revealed two incompatible notions, homophily and structural equivalence, and that BFS-like sampling captures the latter while DFS-like sampling captures the former; prior methods fixed a single rigid sampler and so couldn't span both. Wanting the cheapness of random walks but the steerability of BFS/DFS, I biased the walk — and realized "direction" (toward home vs outward) only exists relative to the previous node, forcing a second-order walk. With the previous node t in hand, the t–v–x path of length 2 bounds the next step's distance from t at d ≤ 2, which I confirmed by enumeration is tight (max d_tx = 2), giving exactly three distance cases (0, 1, 2); fixing d=1 as the reference leaves two multipliers 1/p and 1/q to dial backtracking and outward exploration. Tracing the actual transition distribution on a small graph showed those knobs do what I claimed — q=0.5 sends half the mass outward, q=2 cuts it to a fifth, small p makes the walk bounce back, and p=q=1 flattens to the uniform DeepWalk walk. Precompute the per-edge alias tables, walk r times from every node, feed the sequences to negative-sampling Skip-gram, and bootstrap edge features from node vectors with a component-wise operator.
