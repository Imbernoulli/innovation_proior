# Context: reducing a variable-size set of token vectors to one fixed-size vector

## Research question

A growing family of models produces, for each item, not a single vector but a *variable-size
collection* of D-dimensional vectors. A transformer encoder emits one contextual vector per
input token, so a sentence of n tokens becomes `{h_1, ..., h_n}`, `h_i ∈ R^D`, with n differing
from item to item. A per-variable tokenizer for a multi-channel input emits one group of patch
vectors per input variable, so a location with V variables becomes a set of V vectors. In every
such case the downstream consumer — a similarity comparison, a classifier, the next backbone
block — needs **one fixed-size vector per item**. The open question is the reduction itself: a
function `g({h_1, ..., h_n}) -> R^D` that turns the collection into a single vector.

The reduction is not free to be anything. The collection has **no canonical order** at the
point of aggregation. Sequence order has already been handled upstream by the contextual
encoder, and the variables at a spatial location have no privileged ranking, so the reduction
should not assign meaning to the arbitrary order in which the vectors are presented. It must
accept **any set size** n, including sizes never seen at training time. Its raw output should
**not be dominated by the set size**: a 50-element set and a 5-element set should not land at
systematically different raw scales before any optional normalization, or dot products,
Euclidean distances, and shared fixed-scale heads will be confounded by length rather than
content. A useful reference reducer should also add **no learnable parameters**, so any
learned aggregator can be judged against a transparent baseline. Several reductions are already
used in practice, each justified by intuition rather than by a statement of what it can and
cannot represent; the problem is to compare those choices under the same constraints.

## Background

**Composing token vectors into a single representation has a long pre-transformer history.**
The bag-of-embeddings sentence vector — average the static word vectors of a sentence (e.g.
the GloVe vectors of Pennington, Socher & Manning 2014, or word2vec) — was the standard cheap
sentence representation. It is order-free and size-agnostic by construction, but the
underlying word vectors are static: the same word gets the same vector regardless of context.

**Contextual token vectors arrived with the transformer encoder.** A pre-trained bidirectional
transformer (Devlin, Chang, Lee & Toutanova 2018, BERT, built on the transformer of Vaswani
et al. 2017) maps an input sequence to one contextual vector per token, `T_i ∈ R^H`, plus a
designated special classification token whose final hidden state `C ∈ R^H` is called the
"aggregate sequence representation for classification." Two facts about that designated token
are load-bearing here. First, it is *one* vector asked to summarize the whole sequence.
Second, it is trained on the pre-training objective (next-sentence prediction), and the
authors note explicitly that this vector "is not a meaningful sentence representation without
fine-tuning, since it was trained with [that objective]." So off the shelf, neither the
special token nor a naive average of the contextual token outputs gives a vector that behaves
well under a similarity metric — a phenomenon repeatedly observed when people fed single
sentences through such a network and read off either the special-token output or the averaged
outputs.

**The choice of reduction had already been studied in the recurrent-encoder setting.**
Conneau, Kiela, Schwenk, Barrault & Bordes (2017, InferSent) build a sentence encoder from a
bidirectional LSTM that produces a hidden state `h_t` per token, then combine the variable
number of `{h_t}` into a fixed vector in one of two ways: **max pooling**, taking the maximum
value over each dimension across the tokens (after Collobert & Weston 2008), or by uniformly
averaging the hidden states. They report that for their BiLSTM, max pooling worked
better than mean. They also study a **self-attentive** reduction, `u = Σ_i α_i h_i` with
weights `α_i = softmax(h̄_i^T u_w)` produced by a learned context query vector `u_w` — a
content-dependent weighted average. The lesson carried from this line is that the pooling
reduction materially changes representation quality, and that which one wins is architecture-
and task-dependent rather than settled.

**A transformer sentence encoder fixed the reduction differently.** Cer et al. (2018,
Universal Sentence Encoder) take the contextual word representations from a transformer
encoder and convert them to a fixed-length vector "by computing the element-wise sum of the
representations at each word position," and then **divide by the square root of the sentence
length** so that, in their words, "the differences between short sentences are not dominated
by sentence length effects." Their alternative encoder, a deep averaging network, instead
**averages** the word and bigram embeddings and feeds the average through a feed-forward
network. So both a plain average and a sum-then-divide-by-√n appear in the literature as the
reduction, with the √n normalization motivated by a concern about how aggregate magnitude
scales with set size.

**The recurring tension.** Across all of these, the reduction is a *symmetric* operation over
the set — sum, averaging, max, or a content-weighted average — adopted because it is order-free
and handles variable sizes, but justified case by case. Sum-based reductions raise a magnitude
question (the sum grows with n); single-token reductions raise a bottleneck-and-objective
question (one vector, trained for the wrong task); learned reductions raise a parameter
question. What is missing at this point is a clear account of how the fixed, symmetric choices
trade off set-size behavior, information retention, smoothness, and parameter count.

## Baselines

These are the reductions already available for the same fixed-vector slot.

**Special-token / pick-one-token reduction.** Designate one position (the special
classification token) and use its final hidden vector as the item representation (Devlin et
al. 2018). Core idea: let the network learn to funnel a summary of the whole sequence into one
designated slot through self-attention. Limitation: that single vector must carry everything,
and it is shaped by whatever objective trained it; the originators state it "is not a
meaningful sentence representation without fine-tuning," so off the shelf it underperforms even
averaged static word vectors under cosine similarity. It also presupposes such a designated
token exists, which a per-variable tokenizer at a spatial location does not provide.

**Max pooling over the set.** Take, for each of the D dimensions, the maximum value across the
set members (Collobert & Weston 2008; used in InferSent, Conneau et al. 2017): `g({h_i})_d =
max_i h_{i,d}`. Core idea: keep, per feature, the single strongest activation, on the
intuition that the most salient member should drive that feature. It is order-free and
fixed-dimensional, though the expected extreme can still drift as the set grows. Limitation:
it reads off only the extreme member per dimension and discards the
contribution of every other member, so when the set's information is spread across many
comparably-informative members rather than concentrated in an outlier, most of the set is
thrown away; it is also non-smooth (per dimension, gradient flows to one member). Whether it
beats the average is reported to depend on the architecture — it won for the InferSent BiLSTM
but is not a universal default.

**Sum pooling over the set.** Add the member vectors, `g({h_i}) = Σ_i h_i` (the element-wise
sum used as the reduction in Cer et al. 2018 before normalization). Core idea: the simplest
order-free combination, with the count implicitly retained. Limitation: the magnitude of the
output scales with the number of members, so sets of different sizes land at systematically
different scales — irrelevant to a pure cosine comparison after L2 normalization, but
problematic for dot products, Euclidean distances, fixed-scale heads, and any consumer that
uses the unnormalized vector norm, where length then masquerades as content.

**Content-weighted (attentive) pooling.** Replace uniform combination with a learned weighted
average, `u = Σ_i α_i h_i` with weights from a learned query (self-attentive InferSent,
Conneau et al. 2017; more generally a learnable query attending over the set members). Core
idea: let the model decide how much each member contributes. It is order-free, size-agnostic,
and more expressive. Limitation: it introduces learnable parameters, so it is no longer a
purely fixed baseline; it can overfit on small data; and it gives up the transparency of a
fixed reduction.

**Averaged static word vectors.** Average of context-free word vectors (GloVe, Pennington et al.
2014; word2vec). Core idea: the original bag-of-embeddings item vector. Order-free and
size-agnostic. Limitation: the per-element vectors are static, carrying no contextual
information, so the representation cannot reflect how a member's meaning depends on the rest of
the set.

## Evaluation settings

The yardsticks already in use for fixed-size item representations:

- **Semantic Textual Similarity (STS).** SemEval STS 2012–2016, the STS benchmark (Cer et al.
  2017), and SICK-Relatedness: pairs of sentences with human relatedness labels. Each item is
  encoded to one vector; the predicted similarity is the cosine similarity of the two vectors.
  Metric: Spearman rank correlation between predicted cosine similarity and the gold labels.
  Cosine removes a global scale factor from each vector, so this setting mainly tests whether
  the reduction gives comparable directions across items of different lengths.
- **Transfer classification (SentEval toolkit, Conneau & Kiela 2018).** Encode each item to
  one vector, then train a logistic-regression classifier on the fixed vectors: movie-review
  sentiment (MR), product-review sentiment (CR), subjectivity (SUBJ), opinion polarity (MPQA),
  Stanford Sentiment Treebank (SST), question-type classification (TREC), paraphrase (MRPC).
  Metric: classification accuracy, by 10-fold cross-validation. Here a trained head can
  re-weight dimensions, so the requirement on the pooled vector is weaker than under cosine.
- **Training data for the encoder.** Natural-language-inference corpora — SNLI (Bowman et al.
  2015) and MultiNLI (Williams et al. 2018), sentence pairs labeled entailment / contradiction
  / neutral — are the standard supervision for fitting sentence encoders, after the finding
  that NLI is a good signal for general-purpose sentence representations.
- **The weather variable-aggregation setting.**
  A per-variable tokenizer emits, at each of L spatial patches, one D-dimensional embedding per
  meteorological variable, giving a set of V variable vectors per location (here V = 48,
  L = 512, D = 1024); the reduction must return one D-vector per location, `[B, V, L, D] ->
  [B, L, D]`. The model is fine-tuned from pretrained weights on ERA5 reanalysis at 5.625°,
  optimizer and schedule fixed, evaluated by latitude-weighted RMSE (errors weighted by the
  cosine of latitude) on geopotential at 500 hPa (3-day), temperature at 850 hPa (5-day), and
  10 m wind speed (7-day). Every spatial location receives the same variable list, so this
  axis is not ragged in the usual padded-batch sense.

## Code framework

The reduction plugs into an existing pipeline that already produces the set of token/variable
vectors and already consumes one pooled vector per item. Everything around the reduction
exists; the reduction itself is the single empty slot. The encoder that emits the per-member
contextual vectors, the batching that pads variable-size sets to a common length and carries an
attention mask marking real vs. pad positions, and the downstream that takes the pooled vector
are all in place. What goes in the slot — how to turn the set of member vectors into one vector
— is exactly what is to be decided.

```python
import torch
import torch.nn as nn


class SetReducer(nn.Module):
    """Reduce a variable-size set of D-dimensional member vectors to ONE D-dimensional
    vector per item. The members are produced upstream (e.g. transformer token outputs,
    or per-variable patch embeddings); a downstream consumer takes the single output
    vector. The reduction is the empty slot below.

    Inputs at forward time:
      tokens: [B, N, D]  -- B items, up to N members each, D-dim member vectors
      mask:   [B, N]     -- 1.0 for a real member, 0.0 for a padding slot
                            (all-ones when every set is full, e.g. a fixed-size set)
    Output:
      [B, D]             -- one vector per item
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # TODO: the reduction we will design

    def forward(self, tokens, mask):
        # TODO: combine the member vectors of each set into one vector per item
        raise NotImplementedError


class VariableAggregator(nn.Module):
    """Reduce a set of per-variable tokens at each spatial location to one token.

    Input:  x : [B, V, L, D]  -- V variable tokens at each of L spatial locations
    Output:     [B, L, D]     -- one token per spatial location
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # TODO: the per-location reduction we will design

    def forward(self, x):
        # TODO: produce one token per spatial location, returning [B, L, D]
        raise NotImplementedError


# existing surrounding pipeline the reducer plugs into
def encode_item(encoder, batch, reducer):
    tokens, mask = encoder(batch)          # upstream: set of member vectors + mask
    pooled = reducer(tokens, mask)         # the reduction (slot above)
    return pooled                          # downstream consumes one vector per item
```

The surrounding code supplies the set of member vectors and the mask; `forward` is where the
reduction rule will live.
