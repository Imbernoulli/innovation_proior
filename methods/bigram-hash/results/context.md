## Research question

When a neural network consumes text, every discrete token has to be turned into a dense
real-valued vector before any continuous loss can be optimized. The standard way to do this is a
learned embedding table: one trainable `d`-dimensional row per token in the vocabulary, looked up by
the token's integer id. This is simple and works, but it scales badly. For large corpora the
vocabulary easily reaches hundreds of thousands of entries, and the embedding table becomes the bulk
of the model's parameters — Google's pre-trained word2vec vectors hold a 3-million-entry vocabulary
at `d = 300`, close to a billion parameters in the embedding alone. The problem becomes far worse the
moment you want to embed not just single tokens but token *pairs* or longer `n`-grams as first-class
features, because the number of distinct `n`-grams explodes combinatorially: a full table indexed by
ordered pairs of a 50k vocabulary would need on the order of `50000^2 ≈ 2.5` billion rows, which is
simply not storable.

So the precise goal is a token-representation layer that (1) keeps the parameter cost bounded and
roughly independent of the true size of the feature space, even when that space is enormous because
it includes `n`-grams; (2) needs no dictionary built ahead of time and can handle a feature space too
large or too dynamic to enumerate, as in online learning; (3) does not collapse useful distinctions
when distinct features are forced to share storage; and (4) drops into existing training as a plain
trainable layer optimized by ordinary gradient descent. The existing options below each get some of
this and miss the rest. Closing that gap is the problem.

## Background

By this time, distributed word representations are the standard front end for text models. The
distributional intuition — tokens that occur in similar contexts get similar vectors — is realized by
methods like word2vec and GloVe, and the resulting embeddings carry fine semantic structure when `d`
is large. But large `d` times a large vocabulary is exactly what makes the embedding layer dominate
the parameter budget, and several facts about the field frame the tension:

- **Zipf's law.** In any natural-language corpus a small subset of the vocabulary accounts for most of
  the text, and the overwhelming majority of distinct tokens appear only a handful of times. So most
  rows of a standard embedding table are updated very rarely and carry little signal, yet they still
  cost `d` parameters each. This is the structural waste a large table pays.
- **`n`-gram features help but cost combinatorially.** Bag-of-words is order-blind; adding `n`-grams
  (especially bigrams) as extra features injects cheap local word-order information and is observed to
  raise text-classification accuracy by a few points. But the count of distinct `n`-grams dwarfs the
  unigram vocabulary, so a literal `n`-gram embedding table is infeasible and forces some kind of
  approximation.
- **Dictionary pruning is a blunt instrument.** The usual fixes — keep only the `K` most frequent
  tokens, drop stop words, prune by entropy or by vector norm after training — each risk removing
  features that matter for a specific task (a rare medical term; even a "stop word" like *and* in a
  corpus about logic), and some, like online training, cannot prune at all because the vocabulary is
  not known in advance.
- **Lossy compression of the vectors** (product quantization and related codebook methods) shrinks the
  stored table by replacing each vector with an approximation built from a shared set of centroids,
  but it is applied as post-hoc compression rather than as the trained representation itself, and it
  leaves the dictionary-construction problem untouched.

The load-bearing background concept is **feature hashing**, also called the hashing trick (Weinberger,
Dasgupta, Attenberg, Langford & Smola, ICML 2009). To map a high-dimensional sparse input `x ∈ R^n`
into a bounded space `R^m` with `m ≪ n`, fix a hash function `h: {1,…,n} → {1,…,m}` and define the
hashed feature map coordinate-wise by summing every original coordinate that lands in a given bucket,

```
phi_i(x) = sum_{ j : h(j) = i } x_j .
```

No dictionary is needed — the hash is computed on the fly — sparsity is preserved, and there is no
projection matrix to store. The cost is **collisions**: when `m ≪ n`, many distinct features `j` map
to the same bucket `i` and become indistinguishable. Weinberger et al. add one decisive refinement: a
second, independent **sign hash** `xi: {1,…,n} → {±1}`, giving the *signed* hashed feature map

```
phi_i^{(h,xi)}(x) = sum_{ j : h(j) = i } xi(j) x_j .
```

They prove this signed kernel is **unbiased** — the hashed inner product equals the true inner product
in expectation, `E[⟨x, x'⟩_phi] = ⟨x, x'⟩`, with variance `O(1/m)` for unit vectors — and give
tail bounds showing the hashed length concentrates on the true length with high probability when the
input mass is spread out, and that interference between independently hashed subspaces is negligible.
The unsigned sum is biased because colliding features add up coherently in one direction; the random
sign makes colliding contributions cancel in expectation. The bucket count `m` controls the collision
variance directly, and the concentration guarantees require `m` to grow with the desired accuracy and
failure probability rather than with the original feature dimension.

A relevant empirical anchor: a memory-efficient text classifier of the period represents a sentence as
a bag of word and `n`-gram features, looks each up in a trainable table, averages them into a hidden
vector, and applies a linear classifier with softmax. To keep the `n`-gram table bounded it maps the
`n`-grams through the hashing trick into a fixed number of bins — on the order of `10^7` bins for
bigrams, `10^8` for general `n`-grams — with an embedding dimension as small as 10. Adding bigram
features this way is reported to improve accuracy by 1-4%. This establishes that hashing `n`-grams
into a fixed bin table and adding their looked-up vectors to a token representation is a workable, cheap
source of extra signal.

## Baselines

The prior representation layers in this design space.

**Standard learned embedding table.** One trainable `d`-vector per token, stored as a `K × d` matrix
`E`, looked up by integer id: the token's representation is the row `E[id(w)]`. It is exact (every
token has its own vector, no collisions) and trains by ordinary backprop. **Gap:** the parameter count
is `K · d` and grows linearly with the vocabulary; under Zipf most rows are rarely updated yet still
paid for; it requires a dictionary fixed before training; and it cannot represent an `n`-gram feature
space whose size is `K^n` because the table would not fit.

**Feature hashing / the hashing trick (Weinberger et al. 2009; used for `n`-grams in fastText, Joulin,
Grave, Bojanowski & Mikolov 2016).** Assign each token or `n`-gram to one of `B` buckets by a hash
function and give each bucket its own embedding row; the feature's representation is the row its hash
selects (with the signed variant summing `xi(j) x_j` over colliding features for an unbiased kernel).
Memory is `B · d` regardless of how many distinct features exist, no dictionary is needed, and the
combinatorial `n`-gram space becomes storable by choosing `B`. **Gap:** with `B` necessarily far below
the true feature count, collisions are guaranteed; two distinct, possibly both-important features that
land in the same bucket are handed *the same* vector and the model cannot tell them apart. The bucket
assignment is a fixed hash with a discrete codomain, so there is no gradient by which training could
move an important feature out of a bad collision; the only lever is to enlarge `B`, paying memory back.

**Multi-prototype / co-occurrence-hashed embeddings (Reisinger & Mooney 2010; Huang et al. 2012;
Argerich et al. 2016; Bai et al. 2009).** Give a token several vectors (for several senses), or build
the embedding from hashed word co-occurrence statistics, or merge frequently co-occurring words into a
shared feature to cut dimensionality. **Gap:** these either add parameters (multiple prototypes per
word) or rely on a fixed, untrained co-occurrence-based assignment, and none of them gives a single
bounded-memory, dictionary-free, end-to-end-trainable layer that also keeps colliding features
separable.

**Subword / character-level inputs (Zhang et al. 2015; Xiao & Cho 2016; Conneau et al. 2016).** Sidestep
the vocabulary entirely by feeding characters or character `n`-grams and letting the model build word
meaning internally. **Gap:** this pushes representation-learning cost into a deeper, slower model and
typically needs more data and compute; it changes the architecture rather than offering a drop-in
bounded embedding layer.

## Evaluation settings

The natural yardsticks already in use for token-representation layers:

- **Text-classification benchmark suite (Zhang et al. 2015 protocol).** Seven balanced datasets —
  AG's News, DBPedia, Yelp Review Polarity, Yelp Review Full, Yahoo! Answers, Amazon Review Full,
  and Amazon Review Polarity — spanning topic classification, ontology classification, sentiment, and news
  categorization, with train sizes from ~120k to ~3.6M documents. Metric: test classification
  accuracy. Preprocessing limited to punctuation removal; documents turned into `n`-gram sequences;
  the document representation formed by summing the token-level embeddings; report accuracy at a fixed
  parameter budget so an embedding layer is judged at equal capacity.
- **Causal language-model pretraining of a GPT-style decoder.** A fixed Transformer decoder is trained
  to predict the next token on a large web corpus tokenized by a fixed byte-level BPE tokenizer, with
  the standard learned token embedding plus learned absolute position embedding and the input embedding
  tied to the output projection. Metric: validation cross-entropy (and downstream perplexity such as
  WikiText-2 and LAMBADA). The model, optimizer, training loop, tokenizer, and Transformer blocks are
  held fixed so that only the embedding layer varies; parameter accounting excludes position-embedding
  parameters so that an intervention cannot win by quietly adding capacity through positions.
- Protocol in both settings: identical training elsewhere across embedding variants; the embedding
  layer is the only thing that changes; comparisons are read at matched parameter budgets.

## Code framework

The representation plugs into the existing training harness as one embedding module. The Transformer
blocks, optimizer, loss, data pipeline, and training loop are fixed. The unsettled part is how a token id
becomes the vector stream consumed by the decoder. The harness already expects a `TokenEmbedding` module
with `forward(idx)`, the tied-head weight hook, the position-parameter accounting hook, and an optional
layer hook that may return an extra tensor to add before a block. The empty slots below are deliberately
generic: they mark only where a representation can add state and computation, not what that
representation will be.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


class TokenEmbedding(nn.Module):
    """Maps token ids to vectors for the fixed decoder.

    The standard pieces already exist: a learned token table, a learned absolute
    position table, dropout, and weight tying to the output projection.
    """

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # learned token table
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # learned absolute positions
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        # TODO: any additional trainable state the representation will need.

    def forward(self, idx):
        # idx: (B, T) token ids  ->  (B, T, n_embd)
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        # TODO: compute the token representation.
        return self.drop(tok_emb + pos_emb)

    def get_value_embed(self, layer_idx):
        # TODO: return an extra layer input signal if the design needs one.
        return None

    def get_lm_head_weight(self):
        return self.wte.weight            # tied output projection (fixed convention)

    def get_num_pos_params(self):
        return self.wpe.weight.numel()    # excluded from the reported parameter count


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = TokenEmbedding(config)
        self.transformer = nn.ModuleDict(dict(
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.get_lm_head_weight()

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        for i, block in enumerate(self.transformer.h):
            extra = getattr(self.embedding, "get_value_embed", lambda _: None)(i)
            if extra is not None:
                x = x + extra
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None if targets is None else F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
```

The downstream blocks and loss computation are fixed; `TokenEmbedding` is the module that will be filled.
