## Research question

Pre-trained Transformer encoders achieve strong performance on sentence-pair regression
and classification — semantic textual similarity (STS), natural language inference,
paraphrase detection — by feeding *both* sentences into the network together (a
cross-encoder) and reading a prediction off the joint representation. The question is
how to produce fixed-size sentence embeddings from a pre-trained encoder that support
efficient similarity search and clustering over large collections.

## Background

**The cross-encoder.** A pre-trained Transformer takes the two
sentences concatenated with a `[SEP]` separator, runs full multi-head self-attention
across all tokens of both sentences over its layers, and passes the result to a small
regression/classification head. The two sentences interact through attention at every
layer. Finding the best match in a collection of n = 10,000 sentences requires
n·(n−1)/2 = 49,995,000 forward passes, on the order of 65 hours on a V100 GPU.

**Single-sentence embeddings from a pre-trained encoder.** A single sentence can be
pushed through the encoder and a fixed vector read off — either the output of the
special classification token, or the mean of the token output vectors (offered by
popular "encoder-as-a-service" tooling). Measured by Spearman correlation between
cosine similarity and human STS labels, averaging the encoder outputs lands well below
averaging GloVe word vectors, and the classification-token vector performs worse still.

**Cosine similarity and vector geometry.** Cosine similarity treats every dimension
equally and only sees the angle between two vectors. A representation can carry
information needed for a *downstream classifier* (which can re-weight dimensions) yet
produce different results under cosine, where no re-weighting is allowed. Pre-training
objectives for Transformers do not explicitly target the cosine geometry of resulting
sentence representations.

**Siamese and triplet networks (Bromley et al. 1993; Schroff et al. 2015).** The
classical recipe for learning a metric-friendly embedding: pass each input through the
*same* network (tied weights), producing comparable vectors in one shared space, and
train with an objective defined on the resulting vectors — a classifier over a pair, a
regression onto a similarity score, or a triplet margin that pulls an anchor toward a
positive and away from a negative under a vector distance such as Euclidean distance. The tied-weight (siamese) structure makes the two
embeddings land in one shared space, so vectors produced for different inputs are
directly comparable.

## Baselines

**The cross-encoder (the accurate reference).** Pre-trained Transformer
fed both sentences jointly, with a regression head; state of the art on the STS
benchmark. Core idea: full cross-attention between the sentences.

**Averaged / `[CLS]` pre-trained-encoder embeddings.** Mean of token outputs, or the
classification-token output, used as a sentence vector. Core idea: reuse the pre-trained
encoder with no extra training.

**Averaged word embeddings (GloVe — Pennington et al. 2014; fastText).** Mean of static
word vectors. Core idea: cheap, fixed, compositional-by-averaging sentence vector.

**InferSent (Conneau et al. 2017).** A BiLSTM with max-pooling trained *from scratch* on
SNLI + MultiNLI in a siamese setup, feeding the classifier the concatenation
`(u, v, |u−v|, u·v)`. Core idea: supervised NLI data yields general sentence
embeddings; the siamese structure makes them comparable.

**Universal Sentence Encoder (Cer et al. 2018).** A Transformer trained on multiple
tasks and augmented with SNLI supervision to produce sentence embeddings. Core idea:
multitask training of a dedicated sentence encoder.

**Poly-encoders (Humeau et al. 2019).** Compute a score between m learned context
vectors and precomputed candidate embeddings via attention, reducing cross-encoder cost
for retrieval. Core idea: partial precomputation for ranking.

## Evaluation settings

Training data for general-purpose embeddings: SNLI (570k pairs labeled
entailment/contradiction/neutral) combined with MultiNLI (430k pairs). Task-specific
fine-tuning data: the STS benchmark train split (5,749 pairs; dev 1,500; test 1,379),
the Argument Facet Similarity corpus (~6,000 argument pairs, 0–5 scale), and the
Wikipedia-section triplet dataset (Dor et al. 2018; ~1.8M train / ~223k test triplets).
STS labels are stored on a 0–5 scale; pairwise regression training can normalize them to
0–1 before comparing them with a cosine score.
Unsupervised STS evaluation: SemEval STS 2012–2016, the STS benchmark, and SICK-R, with
gold relatedness labels — scored by Spearman rank correlation between cosine similarity
of sentence embeddings and gold labels (Pearson is avoided as ill-suited for STS).
Transfer evaluation: the SentEval toolkit (MR, CR, SUBJ, MPQA, SST, TREC, MRPC), which
trains a logistic-regression classifier on the embeddings via 10-fold cross-validation.
Computational-efficiency setting: sentences-per-second throughput on CPU and GPU,
including a length-sorted "smart batching" that pads only to the longest element in a
mini-batch.

## Code framework

The harness wraps a pre-trained Transformer encoder, adds a pooling step to turn token
outputs into one fixed vector, and provides siamese training loops that share encoder
weights across the inputs of a pair/triplet. The pooling choice, the training objective,
and the inference similarity are the empty slots.

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
        # TODO: design a categorical or graded pairwise objective over u and v.
        pass

class TripletObjective(nn.Module):
    """Trains from anchor / positive / negative examples."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, anchor, positive, negative):
        a = embed(self.encoder, anchor)
        p = embed(self.encoder, positive)
        n = embed(self.encoder, negative)
        # TODO: enforce that positive is closer to anchor than negative.
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
