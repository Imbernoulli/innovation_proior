# Contriever: Unsupervised Dense Retrieval with Contrastive Learning

## Problem

Train a dense retriever — a bi-encoder that maps queries and documents into a shared vector space
so relevance is a dot product — **without any labeled (query, relevant-document) pairs**, and reach
or beat the unsupervised lexical baseline BM25, including zero-shot across domains and in languages
where no retrieval supervision exists. Supervised dense retrievers (DPR, ANCE) need large human-
labeled sets and transfer poorly out of domain; BM25 needs no labels but is blind to synonymy and
cross-script matches (the lexical gap).

## Key idea

Use the one label-free signal in raw text — **document identity** — and learn by contrastive
discrimination: build a positive pair from a single document and train the encoder to score it
above pairs drawn from other documents, via the InfoNCE loss

  L(q, k₊) = −log [ exp(s(q,k₊)/τ) / ( exp(s(q,k₊)/τ) + Σ_{i=1..K} exp(s(q,k_i)/τ) ) ],
  s(q, k) = ⟨f_θ(q), f_θ(k)⟩,

with τ a temperature (≈0.05). Three choices make it a strong retriever:

- **Positive pairs by independent random cropping.** Sample two independent contiguous token spans
  (length 5%–50% of a ~256-token document) as the pair, then apply 10% token deletion.
  Unlike the inverse Cloze task (which uses a span and its *complement*),
  independent crops are symmetric and *can overlap*, so the model learns both exact lexical matching
  (BM25's strength) and semantic matching when spans do not overlap.
- **Many negatives via MoCo.** Keep a queue of key vectors from recent batches (up to ~131,072) as
  negatives, so the number of negatives is decoupled from batch size. Encode keys with a momentum
  encoder whose weights are an EMA of the query encoder, θ_k ← m·θ_k + (1−m)·θ_q (m≈0.9995), so the
  queued vectors stay mutually consistent as the model trains. Gradients flow only through the query
  side; keys are detached.
- **Shared single-vector bi-encoder.** One BERT-base encoder for both query and document (more
  robust zero-shot than DPR's two towers), **mean pooling** over last-layer token states (not
  [CLS]), scored by dot product (with optional L2-normalization). Single vectors let the document
  index be precomputed and searched with FAISS, which is what makes retrieval over millions of
  documents tractable (a cross-encoder cannot).

Training: AdamW, lr ≈5e-5, batch ≈2048, ~500k steps, on a mix of Wikipedia and CCNet; no labels
anywhere. A multilingual variant initializes from mBERT and samples languages uniformly. The model
can then be fine-tuned on MS MARCO for further gains, but the contrastive stage alone is fully
unsupervised.

## Code

```python
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class Contriever(transformers.BertModel):
    """Bi-encoder tower: BERT -> one vector via mean pooling. Shared by query and key."""
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)

    def forward(self, input_ids, attention_mask, normalize=False):
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        last = out["last_hidden_state"]
        last = last.masked_fill(~attention_mask[..., None].bool(), 0.0)
        emb = last.sum(dim=1) / attention_mask.sum(dim=1)[..., None]   # mean pooling
        if normalize:                                 # optional L2 normalize before scoring
            emb = F.normalize(emb, dim=-1)
        return emb


def token_delete(tokens, p=0.10):
    if p <= 0.0 or len(tokens) <= 1:
        return tokens
    kept = [tok for tok in tokens if random.random() > p]
    return kept if kept else [tokens[random.randrange(len(tokens))]]


def build_positive_pair(tokens, low=0.05, high=0.5, delete_prob=0.10):
    """Two independent crops of one document = a label-free positive pair."""
    n = len(tokens)
    if n == 0:
        return [], []

    def crop():
        length = max(1, min(n, int(round(n * random.uniform(low, high)))))
        start = random.randint(0, n - length)
        return token_delete(tokens[start:start + length], delete_prob)
    return crop(), crop()


class MoCo(nn.Module):
    """InfoNCE with a momentum-encoder queue."""
    def __init__(self, opt):
        super().__init__()
        self.temperature = opt.temperature           # ~0.05
        self.momentum = opt.momentum                 # ~0.9995
        self.queue_size = opt.queue_size             # up to ~131072
        self.label_smoothing = opt.label_smoothing   # optional, mild regularizer
        self.norm = opt.normalize                    # L2-normalize embeddings (optional)
        self.encoder_q = Contriever.from_pretrained(opt.model_id)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        self.register_buffer("queue", F.normalize(torch.randn(opt.dim, self.queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data = pk.data * self.momentum + pq.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        bsz = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % bsz == 0
        self.queue[:, ptr:ptr + bsz] = keys.T
        self.queue_ptr[0] = (ptr + bsz) % self.queue_size

    def forward(self, q_tokens, q_mask, k_tokens, k_mask):
        q = self.encoder_q(q_tokens, q_mask, normalize=self.norm)
        with torch.no_grad():
            self._momentum_update()
            k = self.encoder_k(k_tokens, k_mask, normalize=self.norm)
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)          # positive at column 0
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)  # InfoNCE
        self._dequeue_and_enqueue(k)
        return loss


def train(trainer, loader, opt):
    optim = torch.optim.AdamW((p for p in trainer.parameters() if p.requires_grad), lr=opt.lr)
    for batch in loader:                              # batches mix Wikipedia + CCNet
        loss = trainer(batch["q_tokens"], batch["q_mask"], batch["k_tokens"], batch["k_mask"])
        loss.backward(); optim.step(); optim.zero_grad()
```
