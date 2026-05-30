I want feature vectors for the vertices of a graph so I can throw an ordinary classifier at node-labeling tasks. The trouble with what's around is that the network's native description — its adjacency structure — is sparse and discrete, and sparse discrete features are exactly what statistical learners generalize from badly, especially when only a handful of nodes carry labels. The relational-classification tradition handles labels by iterative inference over a Markov network, but that tangles feature-building with the specific task and produces nothing reusable. The spectral / "social dimension" methods do give continuous latent features, but by eigendecomposing a graph matrix: a global computation that doesn't scale and can't be nudged when the graph changes a little. So what I really want is a way to *learn* a low-dimensional, continuous vector per vertex, unsupervised, from structure alone — vectors where same-community nodes sit close — and I want it to be cheap and updatable from local information.

Let me think about what tool already learns continuous vectors for discrete symbols. Language modeling does precisely this: neural language models map words to dense vectors such that nearby words end up with similar vectors. That's the shape of what I want — discrete symbols (there, words; here, vertices) mapped to a continuous space that encodes "things that occur together." If I could somehow present the graph to a language model, I'd inherit all of that machinery for free.

But a language model eats *sequences* — sentences, ordered streams of symbols — and a graph isn't a sequence. There's the gap. I need to manufacture symbol sequences out of a graph, and they have to be sequences whose statistics carry the community structure I care about.

Here's the connection that makes me think this can actually work, and it's a statistical one. Take a scale-free network — degree distribution following a power law, which most real social networks do. Now run a bunch of short random walks on it and count how often each vertex shows up across the walks. That frequency distribution is itself a power law: a few high-degree hub vertices appear constantly, a long tail of low-degree vertices rarely. Now recall what word frequencies look like in natural-language text: Zipf's law, a power law — a few words like "the" everywhere, a long tail of rare words. The two distributions have the *same shape*. The symbol-frequency statistics of random walks on a scale-free graph match the symbol-frequency statistics of language. That's the license: the neural-language-model machinery, which was engineered around exactly this power-law symbol distribution, should transfer to model community structure in a graph if I feed it the right sequences. So treat short random walks as "sentences" and vertices as "words," and run a language model on the corpus of walks.

Why random walks specifically as the sequence generator? Because a short random walk from a vertex is a probe of its *local community* — the co-occurrences inside a short walk encode which nodes hang together locally, and this is the same locality that local-community-detection algorithms exploit in sublinear time. And because a walk only ever uses local information — the neighbors of the current node — I get two bonuses I wanted: I can update representations from local changes without recomputing globally, and the whole thing streams. So I'll generate, from each vertex, short uniform random walks: start at the vertex, and at each step jump to a uniformly chosen neighbor of the current node, for some fixed length. (I could add restarts — a teleport back to the root — but there's no real gain, so a plain walk.)

Now, what language-model objective do I put on these walk-sentences? The textbook sequence likelihood would be to estimate Pr(v_i | v_1, …, v_{i−1}) — predict the next vertex from all the ones seen so far in the walk. But as the walk grows this conditional becomes infeasible to compute, and it bakes in an ordering that I don't think the data even has: the "neighborhood" a random walk reveals around a vertex isn't an ordered thing, it's a loose bag of nearby nodes. The Skip-gram reframing fixes both problems and fits the graph setting better than it even fit language. Turn the prediction around: instead of using the context to predict a vertex, use one vertex to predict its context; take the context from *both* sides of the center vertex within a window; and drop the requirement to know each context vertex's exact offset — just maximize the probability of the *set* of vertices in the window, order-independently. For a center vertex v_i and window w that's

  minimize  −log Pr( {v_{i−w}, …, v_{i−1}, v_{i+1}, …, v_{i+w}} | Φ(v_i) ),

where Φ is the |V|×d embedding matrix I'm learning. The order-independence is the part that genuinely suits this problem: random-walk co-occurrence is an unordered notion of "nearness," so dropping order isn't an approximation I'm tolerating, it's the right modeling choice. And it makes training cheap — one center vertex feeds a small update at a time.

So the algorithm takes shape. Initialize Φ at random. Make γ passes over the vertex set; in each pass, shuffle the vertices (start-vertex order shouldn't bias the result, and shuffling is the usual way to speed SGD convergence), and for each vertex generate a length-t random walk and hand that walk to a Skip-gram update. The update, for each center vertex v_j in the walk and each context vertex u_k in its window, takes a gradient step on −log Pr(u_k | Φ(v_j)).

There's one piece of that update I can't take for granted: computing Pr(u_k | Φ(v_j)). The natural model is a softmax over all possible target vertices — but the "vocabulary" here is the vertex set V, which can be in the millions or more. A softmax (or a flat logistic classifier) over |V| targets needs an O(|V|) normalization for every single prediction, and I'm making one per context position per walk per pass; that's hopeless at scale. I need to evaluate the conditional in sublinear-in-|V| time. The trick is hierarchical softmax: don't predict the target vertex directly out of |V| options; instead put all vertices at the *leaves* of a binary tree, and identify a target by its root-to-leaf path. Then the probability of reaching leaf u_k factors into a product of binary decisions, one at each internal node along the path:

  Pr(u_k | Φ(v_j)) = Π_{l=1}^{⌈log|V|⌉} Pr(b_l | Φ(v_j)),

where (b_0,…,b_{⌈log|V|⌉}) is the path with b_0 the root and the last node u_k, and each Pr(b_l | Φ(v_j)) is a single binary logistic classifier sitting at b_l's parent ("go left or right?"). The path has length ⌈log|V|⌉, so a prediction costs O(log|V|) instead of O(|V|). That's the difference between feasible and not.

And there's a refinement that's almost free and synergizes with the power-law fact I started from. The cost of a prediction is proportional to the depth of the target leaf, so if I put the *frequent* vertices on *shorter* paths near the root, the average prediction is cheaper. Building the tree with Huffman coding does exactly that — shorter codes for more frequent symbols. Because the vertex frequencies in my walk corpus are power-law (the same property that justified the whole approach), a handful of hub vertices dominate the predictions, and putting them near the root cuts the expected work substantially. The power law shows up twice: once as the reason language-model machinery applies at all, and again as the reason Huffman-coded hierarchical softmax is fast.

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

So the chain: I needed continuous, community-aware vertex vectors and noticed that neural language models already learn exactly that kind of vector for discrete symbols — but they consume sequences and a graph has none. The power-law match between random-walk vertex frequencies on a scale-free graph and word frequencies in language licensed treating short uniform random walks as sentences and vertices as words, so the graph could be fed to a language model; random walks were the right sequence generator because they probe local community structure with only local information, giving online, near-linear behavior. Skip-gram's center-predicts-windowed-context, order-independent objective fit the unordered "nearness" of walk neighborhoods (and dodged the infeasible full-sequence likelihood), and hierarchical softmax over a Huffman tree made the per-prediction cost O(log|V|) instead of O(|V|) — with the same power law that justified the analogy also making the Huffman tree fast. γ shuffled passes of length-t walks, SGD with a linearly decaying rate, and the result is the embedding matrix Φ.
