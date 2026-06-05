## Research question

Pre-trained Transformer encoders set the state of the art on sentence-pair regression
and classification — semantic textual similarity (STS), natural language inference,
paraphrase detection — by feeding *both* sentences into the network together (a
cross-encoder) and reading a prediction off the joint representation. That works
beautifully for scoring one pair at a time, but it is catastrophic the moment the task
is "find the most similar pair (or nearest neighbor) in a large collection." Because
the score is only defined on a *pair* jointly encoded, finding the best match in a
collection of n = 10,000 sentences needs n·(n−1)/2 ≈ 50 million forward passes — on the
order of 65 hours on a single modern GPU; clustering or large-scale semantic search is
simply infeasible. The goal: derive a *fixed-size embedding for a single sentence* such
that semantically similar sentences are close under a cheap vector similarity (e.g.
cosine), so that the expensive joint encoding is replaced by one encode per sentence
plus near-instant vector comparisons — while keeping the accuracy of the strong
pre-trained encoder. The crux is that the encoder was never trained to produce a
*standalone* sentence vector that behaves well under cosine similarity.

## Background

**The cross-encoder and its scaling wall.** A pre-trained Transformer takes the two
sentences concatenated with a `[SEP]` separator, runs full multi-head self-attention
across all tokens of both sentences over its layers, and passes the result to a small
regression/classification head. The two sentences interact through attention at every
layer, which is exactly why it is accurate — and exactly why it does not factorize:
there is no per-sentence vector you can precompute and reuse. Any retrieval/clustering
workload that is quadratic in the number of *pairs* inherits the full forward-pass cost.

**Naive single-sentence embeddings from a pre-trained encoder, and the diagnostic that
they are bad.** The obvious workaround is to push a single sentence through the encoder
and read off a fixed vector — either the output of the special classification token, or
the mean of the token output vectors (offered by popular "encoder-as-a-service" tooling).
The pre-method finding that motivates everything here: these vectors are poor for
similarity. Measured by Spearman correlation between cosine similarity and human STS
labels, averaging the encoder outputs lands well below averaging GloVe word vectors,
and the classification-token vector is worse still. So out of the box the encoder maps
sentences to a space that is unsuitable for cosine similarity — the representation
exists but its geometry is wrong for the metric.

**Why cosine geometry must be trained in.** Cosine similarity treats every dimension
equally and only sees the angle between two vectors. A representation can carry the
information needed for a *downstream classifier* (which can re-weight dimensions) yet be
useless under cosine, where no re-weighting is allowed. So a sentence embedding intended
for cosine retrieval has to be *trained* so that semantic closeness corresponds to small
angle directly. This is the gap a fine-tuning objective must close.

**Siamese and triplet networks (Bromley et al. 1993; Schroff et al. 2015).** The
classical recipe for learning a metric-friendly embedding: pass each input through the
*same* network (tied weights), producing comparable vectors in one shared space, and
train with an objective defined on the resulting vectors — a classifier over a pair, a
regression onto a similarity score, or a triplet margin that pulls an anchor toward a
positive and away from a negative. The tied-weight (siamese) structure is what
guarantees the two embeddings are directly comparable. This is the structural tool the
method will adopt and place on top of the pre-trained encoder.

## Baselines

**The cross-encoder (the accurate but unscalable reference).** Pre-trained Transformer
fed both sentences jointly, with a regression head; state of the art on the STS
benchmark. Core idea: full cross-attention between the sentences. Gap: produces no
reusable per-sentence vector, so retrieval/clustering over a collection is quadratic in
pairs and infeasible at scale.

**Averaged / `[CLS]` pre-trained-encoder embeddings.** Mean of token outputs, or the
classification-token output, used as a sentence vector. Core idea: reuse the pre-trained
encoder with no extra training. Gap: the resulting geometry is poor under cosine
similarity — worse than averaged GloVe on STS.

**Averaged word embeddings (GloVe — Pennington et al. 2014; fastText).** Mean of static
word vectors. Core idea: cheap, fixed, compositional-by-averaging sentence vector. Gap:
no contextualization; a strong but shallow baseline that the pre-trained encoder's
naive embeddings fail to beat on STS.

**InferSent (Conneau et al. 2017).** A BiLSTM with max-pooling trained *from scratch* on
SNLI + MultiNLI in a siamese setup, feeding the classifier the concatenation
`(u, v, |u−v|, u·v)`. Core idea: supervised NLI data yields good general sentence
embeddings; the siamese structure makes them comparable. Gap: trained from random
initialization (no large-scale pre-trained encoder underneath), so it leaves the
pre-trained encoder's knowledge on the table.

**Universal Sentence Encoder (Cer et al. 2018).** A Transformer trained on multiple
tasks and augmented with SNLI supervision to produce sentence embeddings. Core idea:
multitask training of a dedicated sentence encoder. Gap: again a purpose-built encoder
rather than a light fine-tune of an existing pre-trained model; broad but not always
strongest on STS.

**Poly-encoders (Humeau et al. 2019).** Compute a score between m learned context
vectors and precomputed candidate embeddings via attention, reducing cross-encoder cost
for retrieval. Core idea: partial precomputation for ranking. Gaps: the score function
is not symmetric and the overhead is still too large for clustering, which needs O(n²)
symmetric similarity computations over plain vectors.

## Evaluation settings

Training data for general-purpose embeddings: SNLI (570k pairs labeled
entailment/contradiction/neutral) combined with MultiNLI (430k pairs). Task-specific
fine-tuning data: the STS benchmark train split (5,749 pairs; dev 1,500; test 1,379),
the Argument Facet Similarity corpus (~6,000 argument pairs, 0–5 scale), and the
Wikipedia-section triplet dataset (Dor et al. 2018; ~1.8M train / ~223k test triplets).
Unsupervised STS evaluation: SemEval STS 2012–2016, the STS benchmark, and SICK-R, with
gold relatedness labels — scored by Spearman rank correlation between cosine similarity
of sentence embeddings and gold labels (Pearson is avoided as ill-suited for STS).
Transfer evaluation: the SentEval toolkit (MR, CR, SUBJ, MPQA, SST, TREC, MRPC), which
trains a logistic-regression classifier on the embeddings via 10-fold cross-validation.
Computational-efficiency setting: sentences-per-second throughput on CPU and GPU,
including a length-sorted "smart batching" that pads only to the longest element in a
mini-batch. No outcome numbers are part of these settings.

## Code framework

The harness wraps a pre-trained Transformer encoder, adds a pooling step to turn token
outputs into one fixed vector, and provides siamese training loops that share encoder
weights across the inputs of a pair/triplet. The pooling choice, the training objective
that shapes the embedding space, and the inference similarity are the empty slots.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Encoder(nn.Module):
    """Pre-trained Transformer; returns token output vectors [B, L, H]."""
    def forward(self, input_ids, attention_mask):
        pass

def pool(token_embeddings, attention_mask):
    # TODO: reduce token outputs to one fixed-size sentence vector.
    pass

def embed(encoder, batch):
    tok = encoder(batch.input_ids, batch.attention_mask)
    return pool(tok, batch.attention_mask)

class PairObjective(nn.Module):
    """Trains a metric-friendly sentence embedding from labeled sentence pairs."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # shared (tied) weights across both inputs
    def forward(self, batch_a, batch_b, labels):
        u = embed(self.encoder, batch_a)
        v = embed(self.encoder, batch_b)
        # TODO: design the loss that makes cosine(u, v) track semantic similarity.
        pass

def similarity(encoder, sent_a, sent_b):
    # TODO: cheap vector comparison used at inference / retrieval time.
    pass

def train(objective, pairs, epochs=1):
    opt = torch.optim.Adam(objective.parameters(), lr=2e-5)
    for _ in range(epochs):
        for batch_a, batch_b, labels in pairs:
            loss = objective(batch_a, batch_b, labels)
            loss.backward(); opt.step(); opt.zero_grad()
```
