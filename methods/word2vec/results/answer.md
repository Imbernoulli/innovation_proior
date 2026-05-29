# Word2Vec — CBOW and Skip-gram

## Problem

Learn one dense, low-dimensional vector per word from very large corpora (billions of tokens, vocabularies in the millions), cheaply enough to finish in about a day on commodity hardware. "Good" vectors must do two things: place similar words near each other, and encode many kinds of similarity as **linear** structure, so that `vec(b) − vec(a) + vec(c)` lands near the answer to the analogy `a : b :: c : ?`. The blocker is cost: prior neural language models pay for a non-linear hidden layer and a vocabulary-sized softmax, capping the data and dimensionality at which the linear regularities can emerge.

## Key idea

Throw out everything that does not help produce a linearly-structured space. Remove the non-linear hidden layer to make the model **log-linear** (the score of a word given its context is a bare inner product), and replace the full-vocabulary softmax with an output layer whose cost is logarithmic or constant in the vocabulary. With per-word cost slashed, pour the saving into orders of magnitude more data and larger vectors; the working bet is that scale and linear structure matter more for the embeddings than hidden-layer language-model capacity.

Two architectures, both log-linear:

- **CBOW (Continuous Bag-of-Words):** average the input vectors of the context words (order is discarded; parameters shared across positions) and predict the center word. With hierarchical softmax the per-word cost is `Q = N·D + D·log₂V`; with negative sampling the output term becomes `D·(k+1)`.
- **Skip-gram:** use the center word's input vector to predict each surrounding word independently. With hierarchical softmax the per-word cost is `Q = C·(D + D·log₂V)`; with negative sampling replace `D·log₂V` by `D·(k+1)`. It manufactures more training pairs per word, which is useful for rare words and semantic neighborhoods.

A **dynamic window** gives cheap distance-weighting: for each center word draw a radius `R` uniformly in `[1, c]` and use only the `R` words on each side, so nearer words are sampled into training more often, with zero extra parameters.

Each word has **two** vectors: an input vector `v_w` (the embedding that is kept) and an output vector `v'_w` used only in the prediction layer. Keeping them separate avoids the `v_w·v_w = ‖v_w‖²` self-prediction pathology that a single shared vector would create.

## The objective and the two cheap output layers

**CBOW objective.** For position `t`, select a context set `C_t`, average its input vectors into `h_t = |C_t|^{-1}Σ_{j∈C_t} v_{w_j}`, and maximize:

```
(1/T) Σ_{t=1}^{T} log p(w_t | h_t)
```

**Skip-gram objective** (averaged over the corpus of length `T`, window `c`):

```
(1/T) Σ_{t=1}^{T}  Σ_{−c ≤ j ≤ c, j ≠ 0}  log p(w_{t+j} | w_t)
```

The full output layer for either architecture is:

```
p(w_O | h) = exp(v'_{w_O}·h) / Σ_{w=1}^{V} exp(v'_w·h)
```

where `h = h_t` for CBOW and `h = v_{w_I}` for Skip-gram. This costs `O(V)` per gradient. Two replacements:

**1. Hierarchical softmax (HS).** Put the `V` words at the leaves of a binary tree. For word `w`, let `point_j` be the inner node on the root-to-leaf path and let `code_j ∈ {0,1}` be the branch bit used by the trainer, with `code_j = 0` treated as the positive branch. For input vector `h`:

```
p(w | h) = Π_j σ((1 − 2·code_j) · v'_{point_j}·h)
```

Because `σ(x) + σ(−x) = 1`, the two children at every internal node split the incoming probability mass, so the leaf probabilities sum to 1. Cost of `p` and its gradient is the path length `≈ log₂V`. A **Huffman tree** (short paths for frequent words) lowers the expected path length to about `log₂(unigram perplexity)`. Parameters: one input vector per word, one output vector per inner node (`V−1` of them). Per-node gradient with label `= 1 − code_j`: `g = (1 − code_j − σ(v'_point_j·h))·α`; accumulate input gradient `g·v'_point_j` using the old node vector, then update `v'_point_j += g·h`.

**2. Negative sampling (NEG).** Start from NCE's data-vs-noise posterior `P(D=1|w,c) = p_d(c|w) / (p_d(c|w) + k·p_n(c))`. Substituting the unnormalized score `exp(v'_c·v_w)` for the data term gives `exp(v'_c·v_w) / (exp(v'_c·v_w) + k·p_n(c))`; then — since only the vectors matter, not a calibrated likelihood — set `k·p_n(c) = 1`, collapsing it to `σ(v'_c·v_w)`. Each positive term `log p(w_O | w_I)` becomes:

```
log σ(v'_{w_O}·v_{w_I}) + Σ_{i=1}^{k} E_{w_i ∼ P_n(w)} [ log σ(−v'_{w_i}·v_{w_I}) ]
```

Align the true output word, anti-align `k` random words. Needs only **samples** from `P_n`, not its numeric values. Noise distribution: `P_n(w) = U(w)^{3/4}/Z`, a flattened unigram distribution between uniform and raw unigram sampling. `k = 5–20` for small data, `2–5` for large. Per-sample gradient with `label ∈ {1, 0}`: `g = (label − σ(v_{w_I}·v'_target))·α`; update `v'_target += g·v_{w_I}`, accumulate input gradient `g·v'_target`.

**Subsampling of frequent words.** The idealized discard rule is `P(discard w) = 1 − √(t/f(w))`, where `f(w)` is the relative frequency. It leaves words at or below the threshold effectively untouched and thins increasingly frequent words. The C trainer uses the keep-probability `min(1, (√(f/t)+1)·t/f) = min(1, √(t/f) + t/f)`, a slightly less aggressive form with the same leading `√(t/f)` behavior for very frequent words.

**Optimization details.** SGD; learning rate starts at `0.025` and decays linearly by raw corpus tokens consumed; input vectors initialized to `U[−0.5/D, 0.5/D]` (small, so dot products stay in σ's linear region; scaled by `1/D`), output vectors initialized to **zero** (so initial `σ = 0.5`, no spurious push); σ precomputed on a `[−6, 6]` table, clamped outside.

## Code

Grounded in the canonical C trainer `word2vec.c` (`syn0` = input vectors / kept embeddings; `syn1` = HS inner-node vectors; `syn1neg` = NEG output vectors). Rendered in Python for clarity, faithful to that structure.

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

**Defaults** (canonical trainer): `D = 100`, `window = 5`, `negative = 5`, `sample = 1e-3`, `alpha = 0.025` with linear decay, `iter = 5`, `min_count = 5`. The shipped embeddings are the input vectors `syn0`.
