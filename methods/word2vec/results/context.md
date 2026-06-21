## Research question

How can we learn high-quality continuous vector representations of words — one dense, low-dimensional vector per word — from *very large* corpora (billions of tokens, vocabularies of millions of words), cheaply enough to finish in about a day on ordinary hardware?

"High quality" means two things. First, similar words should land near each other in the vector space. Second, and more demanding, the space should encode *multiple, simultaneous degrees of similarity* as **linear** structure: there is a "make-it-plural" direction, a "capital-of-country" direction, a "comparative-adjective" direction, and these should be consistent enough that simple vector arithmetic answers analogies — the offset `vec(b) − vec(a)` applied to `vec(c)` should land near the word that is to `c` as `b` is to `a`. If that holds, a single embedding becomes a reusable building block for downstream NLP (tagging, translation, retrieval, sentiment).

## Background

**Distributed representations.** The foundational idea (Rumelhart, Hinton & Williams 1986; Hinton et al. 1986) is that a concept is represented by a *pattern of activity* over many units rather than by one dedicated symbol. For words this means a dense vector instead of a one-hot index, so that statistical strength is shared across similar words — the central cure for the sparsity of symbolic/n-gram models, where every word is an island.

**Neural language models.** Bengio, Ducharme & Vincent (2003) made this concrete for language modeling. Each of the previous `N` words is mapped through a shared embedding matrix to a `D`-dim vector; the `N` vectors are concatenated into a projection layer of size `N·D`; a `tanh` hidden layer of size `H` follows; and a softmax over the whole vocabulary `V` produces `p(next word | context)`. The embedding matrix and the language model are learned jointly by backprop. This both beat n-grams and produced, as a by-product, the word vectors people wanted. The per-example cost is `Q = N·D + N·D·H + H·V`, dominated by the output term `H·V` and the projection→hidden term `N·D·H`.

**Recurrent language models** (Mikolov et al. 2010, 2011) drop the fixed context length: a recurrent hidden state of size `H` is fed back through a `H·H` matrix, with cost `Q = H·H + H·V`. These models gave the best vectors at the time but took weeks of CPU to train on a few hundred million words.

**Two ways to defuse the softmax `V` term.**
- *Hierarchical softmax* (Morin & Bengio 2005; Mnih & Hinton 2009). Arrange the `V` words as leaves of a binary tree. The probability of a word is a product of binary decisions along the root→leaf path, each decision a logistic unit attached to an inner node. Computing one word's probability and its gradient costs the path length, on average `≈ log₂ V`, instead of `V`. The tree's shape sets the per-word path lengths, so it is a design choice that affects cost.
- *Unnormalized / sampling-based training.* Rather than normalize, train the model to *rank* observed text above corrupted text. Collobert & Weston (2008) use a margin/hinge ranking loss on real vs. noise windows. Mnih & Hinton's log-bilinear model is a stripped-down scorer `score(w, context) = (Σ context vectors)·v'_w` with no hidden non-linearity; with diagonal weighting it is a compact architecture.

**Noise-Contrastive Estimation (NCE)** (Gutmann & Hyvärinen 2010; applied to language models by Mnih & Teh 2012) is the principled member of that family. To fit an *unnormalized* model `p_θ` without ever computing the partition function `Z`, NCE turns density estimation into a binary classification problem: mix the real data with `k` "noise" samples per datum drawn from a known distribution `p_n`, and train a logistic-regression classifier to tell data from noise. Bayes' rule gives the optimal classifier as
```
P(data | u) = p_d(u) / ( p_d(u) + k·p_n(u) ),
```
and substituting the model density for `p_d` (with `Z` treated as just another free parameter, since the classifier only needs ratios) lets you learn `θ` by maximizing the classifier's log-likelihood. As the model improves it provably comes to approximate the true (normalized) softmax, and the estimator needs the actual numeric values of `p_n(u)`.

**Linear regularities and scaling behavior.** Two observations about *existing* vectors were already on the table. (1) Linear regularities: in vectors from a recurrent model, `vec("King") − vec("Man") + vec("Woman")` is closest to `vec("Queen")`, and this works for many syntactic and semantic relations (Mikolov, Yih & Zweig 2013). (2) Scaling behavior: accuracy on these regularities keeps climbing with both more data and more dimensions, and the two must grow *together* — adding only dimensions, or only data, saturates quickly.

## Baselines

The main prior architectures were:

- **Feedforward NNLM (Bengio et al. 2003).** Concatenate `N` context embeddings → `tanh` hidden → full softmax. Learns vectors jointly with an LM. Per-word cost: `Q = N·D + N·D·H + H·V`.

- **Recurrent NNLM (Mikolov et al. 2010).** Recurrent state, `Q = H·H + H·V`, unbounded context. Produces strong vectors, especially syntactic.

- **Two-step approach (Mikolov 2007, 2009).** First learn word vectors with a *single-hidden-layer* network, then train an n-gram NNLM on top of the frozen vectors. Word vectors can be learned by a simpler model on their own, decoupled from building a full LM.

- **Log-bilinear / hierarchical log-bilinear model (Mnih & Hinton 2007, 2009).** A bilinear scorer with no hidden non-linearity; the hierarchical version (HLBL) uses a tree softmax, and a diagonal-weight version is compact. Tree construction is a design choice.

- **Collobert & Weston / SENNA (2008, 2011).** Vectors learned with a margin ranking loss (real window scored above a corrupted one), avoiding normalization. Produces good syntactic structure in the vectors.

- **HLBL with NCE (Mnih & Teh 2012).** Trains an unnormalized neural LM by NCE, sidestepping the softmax normalization. Requires the noise distribution's numeric probabilities and is framed around recovering the LM likelihood.

## Evaluation settings

- **Analogy / word-relationship task.** A question presents paired words sharing a relation: `a : b :: c : d`. Predict `d` by computing `x = vec(b) − vec(a) + vec(c)` and returning the vocabulary word whose vector has highest cosine similarity to `x` (input words `a`, `b`, `c` excluded from the search). Scored by *exact-match* accuracy; synonyms count as wrong. The set is split into **semantic** relations (capital-of-country, currency, city-in-state, man/woman) and **syntactic** relations (plural, comparative/superlative, tense, adjective→adverb, nationality), reported separately and combined. An earlier syntactic-only relatedness set (Mikolov, Yih & Zweig 2013) is also available.
- **Corpora.** Large news corpora and curated LDC-style text collections. Standard preprocessing: restrict the vocabulary to the most frequent words / drop words below a minimum count.
- **Microsoft Research Sentence Completion Challenge** (Zweig & Burges 2011): sentences with one missing word, choosing the best completion from a small candidate set.
- **Protocol.** SGD with backprop; learning rate decayed linearly over training. Cost compared via the per-token operation count `Q` and wall-clock time. Published vectors (SENNA, Turian, Mnih, Mikolov RNNLM, Huang) serve as external comparison points.

## Code framework

The available scaffold is a vocabulary with counts, a sentence stream, input-side vectors to learn, auxiliary state for the prediction layer, a logistic primitive, and an SGD loop with linear learning-rate decay. The design work fits into three generic slots: configure the training state, prepare a sentence before local updates, and perform one local SGD update.

```python
import numpy as np

MAX_EXP = 6.0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Vocab:
    """Word counts and indices from a corpus; supports a min-count cutoff."""
    def __init__(self, token_stream, min_count=5):
        raise NotImplementedError  # TODO: count words, drop rare ones, assign integer ids
    def __len__(self):
        raise NotImplementedError  # TODO
    @property
    def counts(self):
        raise NotImplementedError  # TODO: array of corpus counts per word id

def sentences(path, vocab):
    """Yield lists of integer word ids in corpus order."""
    raise NotImplementedError  # TODO

def init_input_vectors(V, D, rng):
    return (rng.random((V, D)) - 0.5) / D

def linear_alpha(alpha0, seen, total):
    return max(alpha0 * (1.0 - seen / total), alpha0 * 1e-4)

class TrainingState:
    def __init__(self, V, D, counts, options, rng):
        self.V = V
        self.D = D
        self.counts = np.asarray(counts, dtype=np.float64)
        self.input_vectors = init_input_vectors(V, D, rng)
        self.auxiliary = None
        self.options = options

def configure_training(counts, V, D, options, rng):
    raise NotImplementedError  # TODO

def prepare_sentence(sentence, state, rng):
    raise NotImplementedError  # TODO

def training_step(sentence, position, state, alpha, rng):
    raise NotImplementedError  # TODO

def train(corpus, counts, V, D=100, alpha0=0.025, epochs=5, options=None, seed=1):
    rng = np.random.default_rng(seed)
    state = configure_training(counts, V, D, options or {}, rng)
    total = epochs * sum(counts)
    seen = 0
    for _ in range(epochs):
        for sentence in corpus:
            raw_len = len(sentence)
            alpha = linear_alpha(alpha0, seen, total)
            sentence = prepare_sentence(sentence, state, rng)
            for i, _ in enumerate(sentence):
                training_step(sentence, i, state, alpha, rng)
            seen += raw_len
    return state.input_vectors
```
