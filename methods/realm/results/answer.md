# REALM (Retrieval-Augmented Language Model pre-training)

## Problem

Pre-trained LMs store world knowledge implicitly in their weights, which is opaque, capacity-bounded (more facts ⇒ bigger network), and unmodular (cannot revise/inspect facts without retraining). REALM augments LM *pre-training* with a learned textual knowledge retriever, so knowledge lives explicitly in a corpus (Wikipedia) the model retrieves from before each prediction — and the retriever is trained end-to-end from the language-modeling signal, with gradients flowing into the document index.

## Key idea: retrieve-then-predict as a latent-variable model

Decompose p(y|x) into retrieve then predict, treating the document z as a latent variable and marginalizing:

  p(y | x) = Σ_{z ∈ Z} p(y | z, x) · p(z | x).

- **Knowledge retriever** p(z|x) = softmax_z f(x,z), with relevance f(x,z) = Embed_input(x)ᵀ Embed_doc(z), each embedding a BERT [CLS] vector linearly projected to dimension d (Embed_doc reads title + body).
- **Knowledge-augmented encoder** p(y|z,x): a *separate* Transformer over the joined (x, z), enabling cross-attention. Pre-training (MLM): p(y|z,x) = ∏_j p(y_j|z,x), p(y_j|z,x) ∝ exp(w_jᵀ BERT_MASK(j)(join(x, z_body))). Fine-tuning (Open-QA span extraction): p(y|z,x) ∝ Σ_{s∈S(z,y)} exp(MLP([h_START(s); h_END(s)])).

Train by maximizing log p(y|x) for both pre-training (masked salient spans) and fine-tuning (Open-QA), with SGD/Adam.

## Why the latent objective trains the retriever (gradient)

The gradient w.r.t. retriever parameters θ is

  ∇ log p(y|x) = Σ_z [ p(z|y,x) − p(z|x) ] ∇ f(x,z)
            = Σ_z [ p(y|z,x)/p(y|x) − 1 ] p(z|x) ∇ f(x,z).

So each document's score is pushed up iff p(y|z,x) > p(y|x) — i.e. iff document z predicts the correct answer better than the retriever's *average* document. Helpful retrievals are reinforced; useless ones suppressed. In the limit where one z* gives perfect prediction and others give zero, this reduces to ∇ log p(z*|x) — ordinary supervised retrieval of the gold document.

## Scaling to millions of documents

- **Top-k truncation.** Approximate Σ_{z∈Z} by the top-k documents under p(z|x) (k ≈ 8 pre-train, ≈ 5 fine-tune), valid because the distribution is peaked.
- **MIPS.** Ranking by p(z|x) = ranking by the inner product f(x,z), so top-k selection is Maximum Inner Product Search over precomputed Embed_doc(z), sub-linear in corpus size.
- **Asynchronous index refresh.** The index depends on θ, so it goes stale after each update. Key insight: the index is used *only to select* the top-k; exact scores and gradients are recomputed on those few with the current θ. So a parallel index-builder re-embeds + re-indexes the whole corpus every ~500 steps in the background while the trainer never blocks. (Fine-tuning builds the index once and freezes Embed_doc; Embed_input is still trained.)

## Inductive biases (keep the latent retriever from collapsing)

- **Salient span masking** — mask named entities/dates (via an NER tagger + date regex), the spans that actually need world knowledge.
- **Null document** z_∅ added to the top-k — a credit sink when no retrieval is needed.
- **Prohibit trivial retrieval** — when X = Z, exclude the document the masked sentence came from, else the retriever degenerates to exact string match.
- **ICT warm start** — pre-train the retriever with the Inverse Cloze Task to avoid the cold-start cycle (random retrieval → ignored → no retriever gradient); warm-start the encoder with BERT-base (12L, 768h, 12 heads).

## Setup

Knowledge corpus: Dec-2018 Wikipedia, ~288-wordpiece chunks → ~13M documents. Pre-train 200k steps, batch 512, lr 3e-5, marginalize over 8 docs incl. null. Open-QA fine-tuning/eval (NaturalQuestions-Open, WebQuestions, CuratedTrec), exact match, top-5 retrieval. Swapping in a newer Wikipedia updates answers without retraining.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Retriever(nn.Module):
    def __init__(self, bert_q, bert_d, dim):
        super().__init__(); self.bert_q, self.bert_d = bert_q, bert_d
        self.W_input = nn.Linear(bert_q.config.hidden_size, dim, bias=False)
        self.W_doc   = nn.Linear(bert_d.config.hidden_size, dim, bias=False)
    def embed_input(self, x): return self.W_input(self.bert_q(**x).last_hidden_state[:, 0])
    def embed_doc(self, z):   return self.W_doc(self.bert_d(**z).last_hidden_state[:, 0])
    def score(self, q, d):    return q @ d.t()          # f(x,z) = inner product

class KnowledgeAugmentedEncoder(nn.Module):             # separate Transformer over join(x,z)
    def __init__(self, bert):
        super().__init__(); self.bert = bert
        self.mlm_head = nn.Linear(bert.config.hidden_size, bert.config.vocab_size)
    def mlm_logprob(self, x, z, masked_pos, y):
        h = self.bert(**join(x, z)).last_hidden_state
        return F.log_softmax(self.mlm_head(h[masked_pos]), -1).gather(-1, y).sum()

class MIPSIndex:                                        # only SELECTS the top-k
    def build(self, doc_embeddings): self.E = doc_embeddings
    def search(self, q, k):          return topk_inner_product(self.E, q, k)

def marginal_logprob(x, y, retriever, predictor, index, docs, k, null_doc):
    q = retriever.embed_input(x)
    cand = [docs[i] for i in index.search(q, k) if not is_trivial(i, x)] + [null_doc]
    d = retriever.embed_doc(batch(cand))                # recompute with CURRENT theta
    log_pz  = F.log_softmax(retriever.score(q, d), -1)              # exact log p(z|x)
    log_pyz = torch.stack([predictor.logprob(x, z, y) for z in cand])
    return torch.logsumexp(log_pz + log_pyz, 0)         # log sum_z p(z|x) p(y|z,x)

def maybe_refresh_index(retriever, index, corpus, step, every=500):
    if step % every == 0:                               # async builder, trainer never blocks
        index.build(embed_all_docs(retriever, corpus))  # ~13M docs

def train(retriever, predictor, index, corpus, docs, data, steps, k, null_doc):
    opt = torch.optim.Adam(list(retriever.parameters()) + list(predictor.parameters()), lr=3e-5)
    for step, (x, y) in enumerate(data):                # x: salient-span-masked sentence
        loss = -marginal_logprob(x, y, retriever, predictor, index, docs, k, null_doc)
        opt.zero_grad(); loss.backward(); opt.step()
        maybe_refresh_index(retriever, index, corpus, step)
```

REALM is the first to backpropagate a language-modeling signal into a learned, MIPS-indexed retriever during pre-training, making world knowledge explicit, retrievable, and editable — with a provable performance-based retriever gradient and an asynchronous-refresh trick that lets a stale index coexist with exact gradients.
