# Sentence-BERT (SBERT)

## Problem

Pre-trained Transformer cross-encoders are state of the art on sentence-pair tasks but
require feeding both sentences in jointly, so there is no reusable per-sentence vector:
finding the most similar pair among n=10,000 sentences needs ~50M forward passes
(~65 hours on one GPU), making clustering and semantic search infeasible. Naive
single-sentence embeddings (mean of token outputs, or the classification-token output)
have poor cosine geometry — worse than averaged GloVe on STS. We want a fixed-size
single-sentence embedding whose cosine similarity tracks semantic similarity, computed
once per sentence.

## Key idea

Fine-tune the pre-trained encoder in a **siamese / triplet** structure (tied weights, so
both sentence vectors share one space), add a **pooling** step (mean pooling by default)
to produce a fixed vector, and train an objective that shapes the embedding *geometry*
so semantic closeness = small angle. At inference, discard the training head and compare
precomputed embeddings by cosine similarity. This turns 65 hours of pairwise encoding
into ~10,000 encodes (seconds) plus near-free vector comparisons.

## Objectives (chosen by the available data)

- **Classification (NLI data):** concatenate `[u ; v ; |u−v|]` and apply a softmax
  classifier, a trainable linear map from the 3n-dimensional concatenation to `k`
  labels; train with cross-entropy.
  The element-wise difference `|u−v|` is the key term — it shapes coordinate-wise
  distance. (`u·v` is optional and not used by default.)
- **Regression (graded STS data):** train `cosine(u, v)` directly toward the target
  similarity value with MSE; the 0–5 STS labels are normalized to 0–1 before the loss.
- **Triplet (anchor/positive/negative data):** `max(‖s_a−s_p‖ − ‖s_a−s_n‖ + ε, 0)` with
  Euclidean distance and margin `ε = 1`.

Default pooling is MEAN. Training is short: fine-tune one epoch on SNLI+MultiNLI with
the softmax objective, batch size 16, Adam lr 2e-5, linear warmup over 10% of steps;
optionally continue on STS with the regression objective. At inference, the
concatenation/heads are irrelevant — only `u`, `v`, and cosine are used.

## Code

```python
import math
import torch, torch.nn as nn, torch.nn.functional as F

def mean_pool(token_embeddings, attention_mask):
    mask   = attention_mask.unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def embed(encoder, batch):
    tok = encoder(batch.input_ids, batch.attention_mask)   # [B, L, H]
    return mean_pool(tok, batch.attention_mask)            # [B, H]

class SoftmaxObjective(nn.Module):
    def __init__(self, encoder, sent_dim, num_labels=3,
                 use_rep=True, use_diff=True, use_mul=False):
        super().__init__()
        self.encoder = encoder                              # tied weights for both sentences
        self.use_rep, self.use_diff, self.use_mul = use_rep, use_diff, use_mul
        n = (2 if use_rep else 0) + (1 if use_diff else 0) + (1 if use_mul else 0)
        self.classifier = nn.Linear(n * sent_dim, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
    def forward(self, a, b, labels):
        u, v = embed(self.encoder, a), embed(self.encoder, b)
        feats = []
        if self.use_rep:  feats += [u, v]
        if self.use_diff: feats += [torch.abs(u - v)]
        if self.use_mul:  feats += [u * v]
        return self.loss_fct(self.classifier(torch.cat(feats, 1)), labels.view(-1))

class CosineRegressionObjective(nn.Module):
    def __init__(self, encoder):
        super().__init__(); self.encoder = encoder; self.loss_fct = nn.MSELoss()
    def forward(self, a, b, target):                        # STS target normalized to 0..1
        u, v = embed(self.encoder, a), embed(self.encoder, b)
        return self.loss_fct(F.cosine_similarity(u, v), target.view(-1).float())

class TripletObjective(nn.Module):
    def __init__(self, encoder, margin=1.0):
        super().__init__(); self.encoder = encoder; self.margin = margin
    def forward(self, anc, pos, neg):
        a, p, n = (embed(self.encoder, anc), embed(self.encoder, pos), embed(self.encoder, neg))
        d_ap = F.pairwise_distance(a, p, p=2)
        d_an = F.pairwise_distance(a, n, p=2)
        return F.relu(d_ap - d_an + self.margin).mean()

# --- training ---
def linear_warmup_decay(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

def train(objective, data, epochs=1):
    total_steps = epochs * len(data)
    warmup_steps = math.ceil(0.1 * total_steps)
    opt = torch.optim.Adam(objective.parameters(), lr=2e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: linear_warmup_decay(step, total_steps, warmup_steps)
    )
    for _ in range(epochs):
        for batch in data:
            loss = objective(*batch)
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()

# --- inference: encode once, compare by cosine ---
@torch.no_grad()
def encode_bank(encoder, batches):
    return torch.cat([embed(encoder, b) for b in batches])         # [N, H], precomputed once

def most_similar(query_vec, bank):
    return F.cosine_similarity(query_vec.unsqueeze(0), bank).argmax()
```
