# Dense Passage Retrieval (DPR)

## Problem

Open-domain QA decomposes into retrieve-then-read. The reader is strong when given the right passage, so end-to-end accuracy is bottlenecked by the retriever: can we find, from ~21M Wikipedia passages, the small top-k set that contains the answer — accurately, and fast enough to serve in real time? Sparse BM25 is fast but semantically blind ("bad guy" != "villain"). Dense retrieval should fix that, but the practical question is whether it can be trained from ordinary question-passage supervision instead of relying on expensive self-supervised retriever pretraining.

## Key idea

Learn two independent BERT encoders — one for questions, one for passages — that map text to a single `[CLS]` vector (d = 768), with similarity defined as the plain inner product:

  sim(q, p) = E_Q(q)ᵀ E_P(p).

Independence makes the score *decomposable*: passage vectors are precomputed once and indexed for maximum inner-product search (FAISS), so query time is one question encoding plus a sub-linear MIPS lookup. Train only on question–passage pairs (no extra pretraining); the decisive ingredient is the choice of negatives.

## Training objective

For a question with positive passage p⁺ and negatives p⁻₁…p⁻ₙ, minimize the negative log-likelihood of the positive under a softmax over candidates:

  L = −log [ exp(sim(q, p⁺)) / ( exp(sim(q, p⁺)) + Σⱼ exp(sim(q, p⁻ⱼ)) ) ].

**In-batch negatives.** With B questions per batch, stack vectors into **Q**, **P** ∈ ℝ^{B×d}; then **S** = **Q P**ᵀ ∈ ℝ^{B×B} scores every question against every passage. Row i's positive is the diagonal S_{ii}; the other B−1 entries are free "gold" negatives: other questions' positive passages reused as negatives. Add **one BM25 hard negative** per question: a high-BM25 passage that matches many question tokens but lacks the answer. Append those hard-negative vectors after the B positives, so the score matrix is B×2B, the target for row i remains column i, and every appended BM25 passage is a negative for every question in the batch.

Best recipe: BERT-base (uncased) dual encoder, batch size 128, one BM25 negative per question, Adam at lr 1e-5 with linear warmup, dropout 0.1, up to 40 epochs for large datasets and 100 for small datasets.

## Serving

Encode all ~21M passages (disjoint 100-word blocks, title prepended via `[SEP]`) offline with E_P; build a FAISS HNSW index configured for inner-product search. At query time: v_q = E_Q(q), then top-k MIPS.

## Reader (turning retrieval into answers)

Over the top-k (≤100) retrieved passages, a BERT reader cross-encodes (question, passage) — affordable because k is tiny. With passage token representations **P**ᵢ ∈ ℝ^{L×h} and learnable **w**_start, **w**_end, **w**_selected ∈ ℝ^h:

  P_start,i(s) = softmax(**P**ᵢ **w**_start)_s,  P_end,i(t) = softmax(**P**ᵢ **w**_end)_t,
  P_selected(i) = softmax(**P̂**ᵀ **w**_selected)_i,  **P̂** = [P₁^[CLS],…,P_k^[CLS]] ∈ ℝ^{h×k}.

Span score = P_start,i(s)·P_end,i(t); answer = best span in the highest-selection passage. Train by sampling 1 positive + (m̃−1) negatives (m̃ = 24) from the retriever's top 100, maximizing the marginal log-likelihood of correct spans plus the positive-passage selection likelihood. The expensive cross-attention reader is applied only after retrieval, over the bounded top-k candidate set.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import BertModel
import faiss

class BertEncoder(nn.Module):
    def __init__(self, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
    def forward(self, input_ids, attn_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        return out.last_hidden_state[:, 0, :]            # [CLS] -> (B, 768)

def dot_product_scores(q_vecs, p_vecs):
    return torch.matmul(q_vecs, p_vecs.transpose(0, 1))  # Q P^T

def biencoder_nll(q_vecs, pos_vecs, hard_vecs=None):
    # positives occupy columns 0..B-1; one BM25 hard negative per question appends after them
    p_vecs = pos_vecs if hard_vecs is None else torch.cat([pos_vecs, hard_vecs], dim=0)
    target = torch.arange(q_vecs.size(0), device=q_vecs.device)
    scores = dot_product_scores(q_vecs, p_vecs)          # (B, B + #hard)
    log_p  = F.log_softmax(scores, dim=1)
    return F.nll_loss(log_p, target)                     # diagonal positive columns

def train_step(encoder_q, encoder_p, q_ids, q_mask, pos_ids, pos_mask, hard_ids, hard_mask, opt):
    q = encoder_q(q_ids, q_mask)
    pos = encoder_p(pos_ids, pos_mask)
    hard = encoder_p(hard_ids, hard_mask)                # BM25 negative passage for each question
    loss = biencoder_nll(q, pos, hard)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

# --- offline index + online MIPS ---
class PassageIndex:
    def __init__(self, dim):
        self.index = faiss.IndexHNSWFlat(dim, 512, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 128
    def add(self, vecs):     self.index.add(vecs)        # all 21M, encoded offline
    def search(self, q, k):  return self.index.search(q, k)

# --- extractive reader ---
class Reader(nn.Module):
    def __init__(self, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        h = self.bert.config.hidden_size
        self.qa_outputs    = nn.Linear(h, 2)             # start/end
        self.qa_classifier = nn.Linear(h, 1)             # passage selection
    def forward(self, pair_ids, pair_mask):              # rows are [CLS] question [SEP] passage [SEP]
        seq = self.bert(input_ids=pair_ids, attention_mask=pair_mask).last_hidden_state
        start, end = self.qa_outputs(seq).split(1, dim=-1)
        select     = self.qa_classifier(seq[:, 0, :])
        return start.squeeze(-1), end.squeeze(-1), select.squeeze(-1)
```

DPR is a dual-encoder dense retriever trained with softmax NLL over other questions' positives plus one BM25 hard negative per question, then served by a FAISS inner-product index over offline passage vectors.
