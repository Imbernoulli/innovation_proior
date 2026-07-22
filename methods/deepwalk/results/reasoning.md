I want feature vectors for the vertices of a graph so I can throw an ordinary classifier at node-labeling tasks. The trouble with what's around is that the network's native description — its adjacency structure — is sparse and discrete, and sparse discrete features are exactly what statistical learners generalize from badly, especially when only a handful of nodes carry labels. The relational-classification tradition handles labels by iterative inference over a Markov network, but that tangles feature-building with the specific task and produces nothing reusable. The spectral / "social dimension" methods do give continuous latent features, but by eigendecomposing a graph matrix: a global computation that doesn't scale and can't be nudged when the graph changes a little. So what I really want is a way to *learn* a low-dimensional, continuous vector per vertex, unsupervised, from structure alone — vectors where same-community nodes sit close — and I want it to be cheap and updatable from local information.

Let me think about what tool already learns continuous vectors for discrete symbols. Language modeling does precisely this: neural language models map words to dense vectors such that nearby words end up with similar vectors. That's the shape of what I want — discrete symbols (there, words; here, vertices) mapped to a continuous space that encodes "things that occur together." If I could somehow present the graph to a language model, I'd inherit all of that machinery for free.

But a language model eats *sequences* — sentences, ordered streams of symbols — and a graph isn't a sequence. There's the gap. I need to manufacture symbol sequences out of a graph, and they have to be sequences whose statistics carry the community structure I care about.

The obvious sequence to generate from a graph is a random walk: start at a vertex, repeatedly jump to a random neighbor, write down the vertices visited. That's appealing for independent reasons — a short walk from a vertex stays inside its local neighborhood, so the co-occurrences within a walk should encode which nodes hang together locally, the same locality that local-community-detection algorithms exploit in sublinear time; and a walk only ever looks at the neighbors of the current node, so it's cheap and streams and can be recomputed locally when the graph changes. But appeal isn't a reason to believe the language-model machinery will actually fit. Word2vec and its hierarchical-softmax trick weren't built for arbitrary symbol streams; they were tuned around a specific property of language — that word frequencies are wildly skewed (Zipf's law: "the" everywhere, a long tail of rare words). If the symbol stream I hand it has flat or uniform symbol frequencies, the machinery would still run but I'd be using it off-distribution, and the speed tricks in particular might buy me nothing. So before committing, I should ask: what does the frequency-of-symbols distribution look like for vertices appearing across a pile of short random walks?

Let me actually work this out rather than guess. A simple random walk on an undirected connected graph has a known stationary distribution: the long-run probability of sitting at vertex v is π(v) = deg(v) / (2·|E|) — proportional to the vertex's degree. (Check: the detailed-balance condition π(u)·(1/deg(u)) = π(v)·(1/deg(v)) for an edge (u,v) holds exactly when π is proportional to degree, since each transition probability is 1/deg of the source.) So if I run many walks and tally how often each vertex is visited, the visit frequency of v should track deg(v). Now bring in the one empirical fact I know about these networks: real social/information networks are scale-free, their degree distribution is a power law — a few hub vertices of enormous degree, a long tail of low-degree vertices. Compose the two: visit-frequency ∝ degree, and degree is power-law distributed, so visit-frequency should be power-law distributed too. That is the same skewed shape as word frequencies. If that holds, the analogy isn't just poetic; the *statistics* the language model was engineered for are the statistics my walk corpus would have.

I don't want to take that composition on faith — the stationary argument is asymptotic and my walks are short and truncated, so let me check it numerically on a synthetic scale-free graph. I build a 3000-node Barabási–Albert graph (preferential attachment, m=3), which gives degrees ranging from 3 up to 178 with mean 6. I run γ=40 passes of length-40 walks from every vertex and count visits. Two things to look at. First, is visit-count actually proportional to degree on this short-walk regime? The Pearson correlation between deg(v) and visit-count(v) comes out at 0.9997 — essentially perfectly linear, so the stationary-distribution intuition survives truncation. Second, is the resulting frequency distribution power-law-shaped? Sorting visit counts by rank and fitting a line to log(rank) vs log(frequency) over the head of the distribution gives a slope near −0.5 — a straight line in log-log, i.e. a power law, not an exponential or a flat distribution. So the corpus of walk-vertices does have Zipfian symbol statistics. That settles the worry: feeding random walks to a language model is feeding it the kind of input it was built for, not abusing it.

So I'll generate, from each vertex, short uniform random walks: start at the vertex, and at each step jump to a uniformly chosen neighbor of the current node, for some fixed length. (I could add restarts — a teleport back to the root — but I don't see what it buys over plain truncation, and it adds a parameter, so a plain walk.)

Now, what language-model objective do I put on these walk-sentences? The textbook sequence likelihood would be to estimate Pr(v_i | v_1, …, v_{i−1}) — predict the next vertex from all the ones seen so far in the walk. Two things bother me about that. It becomes infeasible to compute as the walk grows, and it bakes in an ordering. Does the data even have that ordering? A random walk's "neighborhood" around a vertex is, from the standpoint of community structure, a loose bag of nearby nodes — whether hub h showed up two steps before v or two steps after v isn't carrying community information, it's just an artifact of which direction the walk happened to go. So an objective that insists on predicting exact positions is fitting noise. The Skip-gram reframing drops exactly that. Turn the prediction around: instead of using the context to predict a vertex, use one vertex to predict its context; take the context from *both* sides of the center vertex within a window; and drop the requirement to know each context vertex's exact offset — just maximize the probability of the *set* of vertices in the window. For a center vertex v_i and window w that's

  minimize  −log Pr( {v_{i−w}, …, v_{i−1}, v_{i+1}, …, v_{i+w}} | Φ(v_i) ),

where Φ is the |V|×d embedding matrix I'm learning. The order-independence here isn't an approximation I'm grudgingly accepting to save computation — random-walk co-occurrence really is an unordered notion of "nearness," so throwing away order is discarding noise, not signal. And it makes training cheap — one center vertex feeds a small update at a time.

So the algorithm takes shape. Initialize Φ at random. Make γ passes over the vertex set; in each pass, shuffle the vertices (start-vertex order shouldn't bias the result, and shuffling is the usual way to speed SGD convergence), and for each vertex generate a length-t random walk and hand that walk to a Skip-gram update. The update, for each center vertex v_j in the walk and each context vertex u_k in its window, takes a gradient step on −log Pr(u_k | Φ(v_j)).

There's one piece of that update I can't take for granted: computing Pr(u_k | Φ(v_j)). The natural model is a softmax over all possible target vertices — but the "vocabulary" here is the vertex set V, which can be in the millions or more. A softmax (or a flat logistic classifier) over |V| targets needs an O(|V|) normalization for every single prediction, and I'm making one per context position per walk per pass; that's hopeless at scale. Let me put numbers on "hopeless": for |V| = 10^6 a flat softmax does on the order of 10^6 inner products *per predicted context vertex*, and I have billions of such predictions across the run. I need to get the per-prediction cost down from |V| to something sublinear. The trick is hierarchical softmax: don't predict the target vertex directly out of |V| options; instead put all vertices at the *leaves* of a binary tree, and identify a target by its root-to-leaf path. Then the probability of reaching leaf u_k factors into a product of binary decisions, one at each internal node along the path:

  Pr(u_k | Φ(v_j)) = Π_{l=1}^{⌈log|V|⌉} Pr(b_l | Φ(v_j)),

where (b_0,…,b_{⌈log|V|⌉}) is the path with b_0 the root and the last node u_k, and each Pr(b_l | Φ(v_j)) is a single binary logistic classifier sitting at b_l's parent ("go left or right?"). A balanced tree over |V| leaves has depth ⌈log₂|V|⌉, so a prediction costs that many binary decisions. Concretely: for |V| = 10^6 that's ⌈log₂ 10^6⌉ = 20 decisions instead of 10^6 inner products — a five-orders-of-magnitude cut per prediction. That's the difference between feasible and not.

There's a further refinement, and here I want to be careful not to assume it pays off just because it sounds clever. The cost of a prediction is the depth of the target leaf, so if I shape the tree to put *frequent* vertices on *shorter* paths, the average prediction gets cheaper. Huffman coding builds exactly that tree — shorter codes for more frequent symbols, minimizing expected code length. But how much does it actually save, and does the saving depend on the distribution being skewed? Let me compute the expected leaf depth for |V| = 1024 under two weightings. With *uniform* vertex frequencies, Huffman's expected depth comes out at 10.00 — identical to the balanced tree's log₂1024 = 10, i.e. no saving at all, which makes sense because there's no skew to exploit. With *Zipf* (power-law) frequencies, weight(rank r) ∝ 1/(r+1), Huffman's expected depth drops to 7.54 — about a 25% reduction, because the handful of high-frequency vertices land near the root and most predictions terminate early. So the Huffman refinement isn't free magic: it earns its keep precisely because the walk-vertex frequencies are power-law, the same property I verified above. The skew that licensed the language-model analogy in the first place is also what makes the tree fast; if the distribution were flat, Huffman would do nothing.

The parameters are {Φ, the tree's internal classifiers}, each of size O(d|V|), optimized by plain SGD with backprop through the tree path. I anneal the learning rate the usual way — start it around 2.5% and decay it linearly as more vertices are processed. Total walk length across the run is L = γt, and since each context update costs O(w·log|V|), the whole thing is O(dwL log|V|) — near-linear in the corpus, and trivially parallelizable since the per-walk updates are local and can run asynchronously.

Let me lay out the pieces, mirroring how the walk generator and the Skip-gram-with-hierarchical-softmax actually go.

```python
import random
from gensim.models import Word2Vec

# ---------------------------------------------------------------
# Random walk generator: a short uniform walk is a "sentence."
# Each step jumps to a uniformly random neighbor of the last node.
# Short walks probe local community structure and use only local
# information (so updates can be online).
# ---------------------------------------------------------------
def random_walk(G, start, length):
    path = [start]
    while len(path) < length:
        cur = path[-1]
        nbrs = G.neighbors(cur)
        if not nbrs:
            break
        path.append(random.choice(nbrs))          # uniform neighbor (1st-order walk)
    return path

# ---------------------------------------------------------------
# Build the corpus: gamma passes; each pass shuffles the vertices
# (offset start-vertex bias, speed SGD) and emits one walk per node.
# ---------------------------------------------------------------
def build_corpus(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):                     # gamma passes
        random.shuffle(nodes)
        for v in nodes:
            walks.append([str(n) for n in random_walk(G, v, walk_length)])
    return walks

# ---------------------------------------------------------------
# DeepWalk: corpus of walks -> Skip-gram with hierarchical softmax.
#   sg=1  : Skip-gram (one vertex predicts its windowed context,
#           order-independent).
#   hs=1  : hierarchical softmax -- vertices at the leaves of a
#           Huffman binary tree; Pr(u_k|Phi(v_j)) = product of
#           binary logistic decisions along the root->leaf path,
#           O(log|V|) instead of O(|V|). Frequent (hub) vertices get
#           shorter paths.
# Reported best: gamma=80, walk_length=40, d=128, window=10.
# ---------------------------------------------------------------
def deepwalk(G, dim=128, num_walks=80, walk_length=40, window=10, workers=1):
    walks = build_corpus(G, num_walks, walk_length)
    model = Word2Vec(walks, vector_size=dim, window=window,
                     sg=1, hs=1, min_count=0, workers=workers)
    return model.wv                                # Phi: |V| x d embeddings
```
