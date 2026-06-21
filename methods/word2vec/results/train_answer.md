I want one dense, low-dimensional vector per word, learned from very large corpora — billions of tokens, vocabularies in the millions — cheaply enough to finish in about a day on commodity hardware. "Good" means more than just placing similar words near each other; I want the space to encode many simultaneous kinds of similarity as *linear* structure, so that the offset $\mathrm{vec}(b) - \mathrm{vec}(a)$ carried over to $\mathrm{vec}(c)$ lands near the answer of the analogy $a:b::c:?$. The clue I keep returning to is that vectors from a recurrent language model already show this: $\mathrm{vec}(\text{King}) - \mathrm{vec}(\text{Man}) + \mathrm{vec}(\text{Woman})$ is closest to $\mathrm{vec}(\text{Queen})$, and the same "feminine" offset, "plural" offset, and "capital-of" offset recur across many pairs. So crisp linear regularities are latent and learnable. The trouble is that they only sharpen when both the data and the dimensionality grow *together* — push only one and accuracy plateaus — and every method that produces these vectors is a neural language model too slow to reach that scale. Nobody has trained on more than a few hundred million words, and dimensionality is stuck at 50–100. This is not a modeling-quality problem; it is a cost problem.

The cost concentrates in two places. Writing $Q$ for the parameters touched per training word, Bengio's feedforward model embeds the previous $N$ words ($N\!\cdot\!D$), runs a non-linear $\tanh$ hidden layer of width $H$ over the dense projection ($N\!\cdot\!D\!\cdot\!H$), and applies a softmax over the whole vocabulary $V$ ($H\!\cdot\!V$), giving $Q = N\!\cdot\!D + N\!\cdot\!D\!\cdot\!H + H\!\cdot\!V$; the recurrent model trades the projection term for a dense $H\!\cdot\!H$ recurrence but keeps the same $H\!\cdot\!V$ output. With $V$ in the millions and $H$ in the hundreds, $H\!\cdot\!V$ alone is hundreds of millions of multiply-adds per training word. The two adversaries are therefore the **non-linear hidden layer** and the **vocabulary-sized softmax**. Hierarchical softmax and NCE-style sampling were already on the table as partial cures for the second, but they were always wrapped around a full language model carrying the first; and the precedent of learning vectors with a single-hidden-layer network and freezing them showed only that the embedding job can be decoupled, not that it can be made cheap enough.

I propose Word2Vec — two minimal log-linear architectures, CBOW and Skip-gram, paired with an output layer that is logarithmic or constant in the vocabulary. The first move is to rip out the non-linear hidden layer entirely and make the model log-linear, so that the score of a word given its context is a bare inner product. I do not want a good language model; I want a linearly-structured space, and the regularities I am after are themselves linear offsets — a model that is linear in the word vectors is not merely adequate for that but better suited to it, and the capacity I lose I buy back many times over by training on far more data with far larger vectors. The two architectures differ only in which direction they predict. CBOW averages the input vectors of the surrounding words into a single vector $h_t = |C_t|^{-1}\sum_{j\in C_t} v_{w_j}$ and predicts the center word by maximizing $\frac{1}{T}\sum_t \log p(w_t \mid h_t)$; averaging discards word order, but for learning which words keep similar company that is fine, even helpful, since every context position then reinforces the same target through shared parameters. Skip-gram instead uses the center word's input vector to predict each neighbor independently, maximizing $\frac{1}{T}\sum_t \sum_{-c\le j\le c,\,j\ne 0} \log p(w_{t+j}\mid w_t)$; it is more expensive per center word but manufactures far more (input, target) pairs, which I expect to help rare words and semantic structure since each word is trained as a predictor of many neighbors. For cheap distance-weighting I draw, for each center word, a window radius $R$ uniformly in $[1,c]$ and use only the $R$ words on each side: a word at distance 1 is included for every draw, a word at distance $c$ only when $R=c$, so nearer words are sampled into training more often with zero extra parameters — far cleaner than attaching explicit distance weights.

With the hidden layer gone, the softmax is now the *entire* cost, so everything hinges on replacing it. The full output layer for either architecture is
$$p(w_O\mid h) = \frac{\exp(v'_{w_O}\!\cdot h)}{\sum_{w=1}^{V}\exp(v'_w\!\cdot h)},$$
with $h = h_t$ for CBOW and $h = v_{w_I}$ for Skip-gram, and its gradient touches every one of the $V$ output vectors — $10^5$ to $10^7$ inner products per prediction. I attack it two ways. The first keeps a genuine normalized probability but factors the normalization into a binary tree with the $V$ words at the leaves: predicting a word is walking root-to-leaf, each inner node a logistic decision. With path bits $\mathrm{code}_j \in \{0,1\}$ (taking $\mathrm{code}_j = 0$ as the positive branch) and inner-node vectors $v'_{\mathrm{point}_j}$,
$$p(w\mid h) = \prod_j \sigma\big((1 - 2\,\mathrm{code}_j)\, v'_{\mathrm{point}_j}\!\cdot h\big).$$
Because $\sigma(x) + \sigma(-x) = 1$, the two children at every node split the incoming probability mass, so the leaf masses sum to exactly 1 over the whole vocabulary — it is a real softmax, merely factored. Computing $p$ and its gradient now walks one path of length $\approx \log_2 V$ instead of summing over $V$, turning $\sim\!10^6$ into $\sim\!20$ for a million-word vocabulary. The tree shape is mine to choose, and since word frequencies are wildly skewed I build a **Huffman tree**, giving frequent words short paths so the expected path length drops to about $\log_2$ of the unigram perplexity. The bookkeeping cost is that the output side becomes one vector per inner node ($V-1$ of them), but each is just the logistic-regression weight of a path decision, and the per-node update is clean: with label $1 - \mathrm{code}_j$ and prediction $f = \sigma(v'_{\mathrm{point}_j}\!\cdot h)$, set $g = (1 - \mathrm{code}_j - f)\,\alpha$, accumulate the input-side gradient $g\,v'_{\mathrm{point}_j}$ from the *old* node vector, then update $v'_{\mathrm{point}_j} \mathrel{+}= g\,h$.

The second replacement abandons normalization altogether, because I do not need a calibrated distribution — I need words that genuinely co-occur to point together and random pairs not to. That is discrimination, not density estimation. Starting from NCE, which mixes $k$ noise samples per real datum and trains a data-vs-noise classifier whose Bayes-optimal posterior is $P(D{=}1\mid w,c) = p_d(c\mid w) / (p_d(c\mid w) + k\,p_n(c))$, I substitute the unnormalized score $\exp(v'_c\!\cdot v_w)$ for the data term and then ask the irreverent question: since I only want vectors, what if I throw away the part tying this back to a softmax and simply set $k\,p_n(c) = 1$? The posterior collapses, because $x/(x+1)$ with $x = e^z$ is exactly $\sigma(z)$, to $P(D{=}1\mid w,c) = \sigma(v'_c\!\cdot v_w)$ — no partition function, no $p_n$ values in the denominator, no tree. The per-pair objective becomes
$$\log\sigma(v'_{w_O}\!\cdot v_{w_I}) + \sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n(w)}\big[\log\sigma(-v'_{w_i}\!\cdot v_{w_I})\big],$$
which pushes the true output word to align with the input vector and $k$ random words to anti-align. I call this **negative sampling**. I am honest that setting $k\,p_n = 1$ breaks NCE's guarantee that this recovers the softmax likelihood; I traded that guarantee for radical simplicity, and the payoff is that I now need only *samples* from the noise distribution, never its numeric values. The number of negatives $k$ directly sets the cost multiplier — one extra inner product and update per negative — so I use $k = 5\text{–}20$ for small corpora and $2\text{–}5$ for large. For the noise distribution itself, uniform leaves frequent words rarely pushed away from anything so their geometry stays mushy, while raw unigram over-samples a handful of ultra-frequent words so hard that rare words never appear as negatives; the compromise is the three-quarters power, $P_n(w) \propto U(w)^{3/4}$, which boosts rare words' relative weight (a small mass $m$ becomes the proportionally larger $m^{3/4}$) while still letting common words dominate more than uniform would. The negative-sampling update mirrors the hierarchical one over a flat list: with label $\in\{1,0\}$ and $f = \sigma(v_{w_I}\!\cdot v'_{\mathrm{target}})$, set $g = (\text{label} - f)\,\alpha$, update $v'_{\mathrm{target}} \mathrel{+}= g\,v_{w_I}$, and accumulate the input gradient $g\,v'_{\mathrm{target}}$.

Two further pieces make the geometry come out right. Each word keeps *two* vectors, an input vector $v_w$ (the embedding I ship) and an output vector $v'_w$ used only in the prediction layer, because sharing a single vector would make the self-prediction score $v_w\!\cdot v_w = \|v_w\|^2$ always large and positive — yet a word rarely sits next to itself, so the model would fight to shrink the norms and corrupt the very geometry I am building. With separate tables the score $v'_w\!\cdot v_w$ has no built-in tendency to be large, and the pathology dissolves. And because the token stream is dominated by "the", "a", "in" — which appear next to everything and so teach almost nothing, while their vectors stop moving after a few million examples — I thin frequent words before forming windows, discarding each token with probability $P(\text{discard } w) = 1 - \sqrt{t/f(w)}$, where $f(w)$ is the relative frequency: at or below the threshold $t$ a word is kept always, above it the discard probability climbs toward 1 monotonically so frequencies are never reordered. This both saves updates on near-empty pairs and lets the window reach across more content words, so rare-word neighborhoods come out cleaner. The remaining choices matter at this scale: SGD with the learning rate starting at $0.025$ and decaying *linearly* by raw tokens consumed; input vectors initialized to $U[-0.5/D,\,0.5/D]$ — small so initial dot products land in $\sigma$'s near-linear region with healthy gradients, scaled by $1/D$ so they do not blow up with dimension — and output vectors initialized to exactly **zero**, so every initial $\sigma$ is $0.5$ and the data, not the initialization, writes the geometry; and $\sigma$ precomputed on a grid over $[-6,6]$ and clamped outside, where it is within $0.0025$ of its asymptote, saving an $\exp$ on every one of the billions of updates. The pieces interlock in the training loop: thin each sentence, slide over positions, draw the dynamic radius, build $h$ (the context average for CBOW, the center vector for Skip-gram), run the swappable cheap output layer, and scatter the accumulated gradient back; ship the input vectors $\mathrm{syn0}$.

```python
import numpy as np

MAX_EXP = 6.0
def build_sigmoid_table(size=1000):
    xs = (np.arange(size) / size * 2 - 1) * MAX_EXP
    return 1.0 / (1.0 + np.exp(-xs)), size
EXP_TABLE, EXP_SIZE = build_sigmoid_table()
def sigm(x):
    if x >= MAX_EXP:  return 1.0
    if x <= -MAX_EXP: return 0.0
    return EXP_TABLE[int((x + MAX_EXP) * (EXP_SIZE / MAX_EXP / 2))]

# ---- frequent-word subsampling: discard with prob 1 - sqrt(t/f) -------------
def subsample_keep(count, train_words, t, rng):
    if t <= 0: return True
    fr = count / train_words
    if fr <= 0: return True
    keep_p = (np.sqrt(fr / t) + 1.0) * (t / fr)      # C trainer's keep form
    return keep_p >= 1.0 or rng.random() < keep_p

# ---- Huffman tree: short codes for frequent words --------------------------
def build_huffman(counts):
    # Counts are in descending vocabulary order, as in the C trainer.
    V = len(counts)
    count = list(counts) + [10**15] * (V - 1)        # internal nodes start "infinite"
    binary = [0] * (2 * V - 1); parent = [-1] * (2 * V - 1)
    pos1, pos2 = V - 1, V
    def take_min():
        nonlocal pos1, pos2
        if pos1 >= 0 and count[pos1] < count[pos2]:
            idx = pos1; pos1 -= 1
        else:
            idx = pos2; pos2 += 1
        return idx
    for a in range(V - 1):                            # merge two smallest each step
        min1, min2 = take_min(), take_min()
        count[V + a] = count[min1] + count[min2]
        parent[min1] = parent[min2] = V + a; binary[min2] = 1
    root = 2 * V - 2
    codes, points = [], []
    for a in range(V):
        code, path, b = [], [], a
        while True:                                   # climb leaf -> root
            code.append(binary[b]); path.append(b); b = parent[b]
            if b == root: break
        codes.append(list(reversed(code)))
        inner_nodes = [root] + list(reversed(path[1:]))
        points.append([n - V for n in inner_nodes])    # root first; one point per code bit
    return codes, points

# ---- the two cheap output layers -------------------------------------------
def hs_update(h, word, syn1, codes, points, alpha):
    grad = np.zeros_like(h)
    for code_bit, node in zip(codes[word], points[word]):
        f = sigm(np.dot(h, syn1[node]))
        g = (1 - code_bit - f) * alpha               # (label - prediction)*alpha, label = 1-code
        grad += g * syn1[node]
        syn1[node] += g * h
    return grad

def neg_update(h, word, syn1neg, unigram_table, k, alpha, rng):
    grad = np.zeros_like(h)
    for d in range(k + 1):
        if d == 0:
            target, label = word, 1                  # positive pair
        else:
            target = unigram_table[rng.integers(len(unigram_table))]  # ~ U(w)^{3/4}
            if target == word: continue
            label = 0
        f = sigm(np.dot(h, syn1neg[target]))
        g = (label - f) * alpha
        grad += g * syn1neg[target]
        syn1neg[target] += g * h
    return grad

def build_unigram_table(counts, size=10**8, power=0.75):
    w = np.power(np.asarray(counts, dtype=np.float64), power); w /= w.sum()  # U(w)^{3/4}/Z
    table, i, cum = np.empty(size, dtype=np.int64), 0, w[0]
    for a in range(size):
        table[a] = i
        if a / size > cum and i < len(w) - 1: i += 1; cum += w[i]
    return table

def predict(h, target, syn1, syn1neg, codes, points, table, k, hs, neg, alpha, rng):
    grad = np.zeros_like(h)
    if hs:  grad += hs_update(h, target, syn1, codes, points, alpha)
    if neg: grad += neg_update(h, target, syn1neg, table, k, alpha, rng)
    return grad

# ---- fill the generic scaffold slots ---------------------------------------
class TrainingState:
    def __init__(self, counts, V, D, window, k, arch, hs, neg, sample, table_size, rng):
        self.counts = np.asarray(counts, dtype=np.float64)
        self.train_words = float(self.counts.sum())
        self.window, self.k, self.arch = window, k, arch
        self.hs, self.neg, self.sample = hs, neg, sample
        self.syn0 = (rng.random((V, D)) - 0.5) / D     # syn0: input vectors kept as embeddings
        self.syn1 = np.zeros((V - 1, D)) if hs else None      # syn1: HS inner-node vectors
        self.syn1neg = np.zeros((V, D)) if neg else None      # syn1neg: NEG output vectors
        self.codes, self.points = build_huffman(self.counts) if hs else (None, None)
        self.table = build_unigram_table(self.counts, size=table_size, power=0.75) if neg else None

def configure_training(counts, V, D, options, rng):
    return TrainingState(counts, V, D,
                         window=options.get("window", 5),
                         k=options.get("k", 5),
                         arch=options.get("arch", "skipgram"),
                         hs=options.get("hs", False),
                         neg=options.get("neg", True),
                         sample=options.get("sample", 1e-3),
                         table_size=options.get("table_size", 10**8),
                         rng=rng)

def prepare_sentence(sentence, state, rng):
    return [w for w in sentence
            if subsample_keep(state.counts[w], state.train_words, state.sample, rng)]

def window_context(sentence, pos, window, rng):
    b = int(rng.integers(window)) if window > 0 else 0  # dynamic radius = window - b
    return [sentence[i] for i in range(pos - window + b, pos + window + 1 - b)
            if 0 <= i < len(sentence) and i != pos]

def training_step(sentence, pos, state, alpha, rng):
    center = sentence[pos]
    ctx = window_context(sentence, pos, state.window, rng)
    if not ctx: return
    if state.arch == "cbow":
        h = state.syn0[ctx].mean(axis=0)                # average context -> predict center
        grad = predict(h, center, state.syn1, state.syn1neg, state.codes,
                       state.points, state.table, state.k, state.hs, state.neg, alpha, rng)
        for w_ctx in ctx:                               # C-style scatter, no /len(ctx)
            state.syn0[w_ctx] += grad
    else:                                               # skip-gram: center -> each context
        for w_ctx in ctx:
            grad = predict(state.syn0[center], w_ctx, state.syn1, state.syn1neg,
                           state.codes, state.points, state.table, state.k,
                           state.hs, state.neg, alpha, rng)
            state.syn0[center] += grad

def linear_alpha(alpha0, seen, total):
    return max(alpha0 * (1 - seen / total), alpha0 * 1e-4)

def train(corpus, counts, V, D=100, window=5, k=5, arch="skipgram",
          hs=False, neg=True, alpha0=0.025, sample=1e-3, table_size=10**8,
          epochs=5, seed=1):
    rng = np.random.default_rng(seed)
    state = configure_training(counts, V, D,
                               {"window": window, "k": k, "arch": arch, "hs": hs,
                                "neg": neg, "sample": sample, "table_size": table_size},
                               rng)
    total, seen = epochs * state.train_words, 0
    for _ in range(epochs):
        for sentence in corpus:
            raw_len = len(sentence)
            alpha = linear_alpha(alpha0, seen, total)
            sent = prepare_sentence(sentence, state, rng)
            for pos, _ in enumerate(sent):
                training_step(sent, pos, state, alpha, rng)
            seen += raw_len
    return state.syn0   # the learned word embeddings
```
