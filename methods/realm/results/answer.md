REALM (Retrieval-Augmented Language Model pre-training)

**Problem**

Pre-trained LMs store world knowledge implicitly in their weights, which is opaque, capacity-bounded (more facts means a bigger network), and unmodular (facts cannot be revised or inspected directly). REALM augments LM *pre-training* with a learned textual knowledge retriever, so knowledge lives explicitly in a corpus (Wikipedia) the model retrieves from before each prediction. The masked-LM loss trains the retriever end-to-end on the retrieved candidates; the MIPS index only proposes candidates and is refreshed as the document encoder changes.

**Retrieve-then-predict as a latent-variable model**

Decompose p(y|x) into retrieve then predict, treating the document z as a latent variable and marginalizing:

  p(y | x) = Σ_{z ∈ Z} p(y | z, x) · p(z | x).

- **Knowledge retriever** p(z|x) = softmax_z f(x,z), with relevance f(x,z) = Embed_input(x)^T Embed_doc(z), each embedding a BERT [CLS] vector linearly projected to dimension d. Embed_doc reads title + body.
- **Knowledge-augmented encoder** p(y|z,x): a *separate* Transformer over the joined input and document body, enabling cross-attention. Pre-training (MLM): p(y|z,x) = product_j p(y_j|z,x), p(y_j|z,x) proportional to exp(w_j^T BERT_MASK(j)(join(x, z_body))). Fine-tuning (Open-QA span extraction): p(y|z,x) proportional to sum_{s in S(z,y)} exp(MLP([h_START(s); h_END(s)])).

Train by maximizing log p(y|x) for both pre-training (masked salient spans) and fine-tuning (Open-QA), with SGD/Adam.

**Why the latent objective trains the retriever**

The gradient w.r.t. retriever parameters θ is

  ∇ log p(y|x) = Σ_z [ p(z|y,x) − p(z|x) ] ∇ f(x,z)
            = Σ_z [ p(y|z,x)/p(y|x) − 1 ] p(z|x) ∇ f(x,z).

So each document's score is pushed up iff p(y|z,x) > p(y|x), meaning document z predicts the correct answer better than the retriever's average document. Helpful retrievals are reinforced; useless ones suppressed. In the limit where one z* gives perfect prediction and others give zero, this reduces to ∇ log p(z*|x), ordinary supervised retrieval of the gold document.

**Scaling to millions of documents**

- **Top-k truncation.** Approximate Σ_{z∈Z} by a small retrieved candidate set under p(z|x): 8 total candidates during pre-training, including the null document, and top-5 candidates at Open-QA inference.
- **MIPS.** Ranking by p(z|x) = ranking by the inner product f(x,z), so top-k selection is Maximum Inner Product Search over precomputed Embed_doc(z), sub-linear in corpus size.
- **Asynchronous index refresh.** The index depends on θ, so it goes stale after each update. The index is used only to select the candidate set; scores, the candidate-set softmax, and gradients are recomputed on those retrieved documents with the current θ. A parallel index-builder re-embeds and re-indexes the whole corpus about every 500 steps while the trainer keeps running. Fine-tuning builds the index once and freezes Embed_doc; Embed_input is still trained.

**Inductive biases**

- **Salient span masking** — mask named entities/dates (via an NER tagger + date regex), the spans that actually need world knowledge.
- **Null document** z_null added to the candidate set -- a credit sink when no retrieval is needed.
- **Prohibit trivial retrieval** -- when X = Z, exclude the document the masked sentence came from, else the retriever degenerates to exact string match.
- **ICT warm start** -- pre-train the retriever with the Inverse Cloze Task to avoid the cold-start cycle (random retrieval, ignored retrieval, no retriever gradient); warm-start the encoder with BERT-base (12L, 768h, 12 heads).

**Setup**

Knowledge corpus: December 20, 2018 Wikipedia, chunks of up to 288 BERT wordpieces, just over 13M documents. Pre-training: 200k steps, batch 512, learning rate 3e-5, 8 marginalized candidates including the null document. Open-QA fine-tuning/evaluation: NaturalQuestions-Open, WebQuestions, CuratedTrec, exact match, top-5 retrieval. Because retrieval reads the corpus at inference time, changing the corpus can change predictions without changing the model weights, though common facts can still remain in the encoder parameters.

**Code**

```python
import copy
import torch, torch.nn as nn, torch.nn.functional as F

class Retriever(nn.Module):
    def __init__(self, bert_q, bert_d, dim):
        super().__init__(); self.bert_q, self.bert_d = bert_q, bert_d
        self.W_input = nn.Linear(bert_q.config.hidden_size, dim, bias=False)
        self.W_doc   = nn.Linear(bert_d.config.hidden_size, dim, bias=False)
    def embed_input(self, x): return self.W_input(self.bert_q(**x).last_hidden_state[:, 0])
    def embed_doc(self, z):   return self.W_doc(self.bert_d(**join_title_body(z)).last_hidden_state[:, 0])
    def score(self, q, d):    return q @ d.t()          # f(x,z) = inner product

class KnowledgeAugmentedEncoder(nn.Module):             # separate Transformer over join(x,z)
    def __init__(self, bert):
        super().__init__(); self.bert = bert
        self.mlm_head = nn.Linear(bert.config.hidden_size, bert.config.vocab_size)
        self.span_mlp = nn.Linear(2 * bert.config.hidden_size, 1)
    def mlm_logprob(self, x, z, target):
        h = self.bert(**join_input_and_body(x, z)).last_hidden_state
        logits = self.mlm_head(h[target.masked_pos])
        return F.log_softmax(logits, -1).gather(-1, target.tokens).sum()
    def span_logprob(self, x, z, target):
        h = self.bert(**join_input_and_body(x, z)).last_hidden_state
        scores = [self.span_mlp(torch.cat([h[a], h[b]])) for a, b in target.matching_spans(z)]
        return torch.logsumexp(torch.stack(scores), 0)
    def logprob(self, x, z, target, task="mlm"):
        return self.mlm_logprob(x, z, target) if task == "mlm" else self.span_logprob(x, z, target)

class MIPSIndex:                                        # only SELECTS the top-k
    def build(self, doc_embeddings): self.E = doc_embeddings
    def search(self, q, k):          return topk_inner_product(self.E, q, k)
    def replace(self, other):        self.E = other.E

def marginal_logprob(x, y, retriever, predictor, index, docs, K, null_doc, task="mlm"):
    q = retriever.embed_input(x)
    ids = index.search(q.detach(), K + 8)               # fetch slack, then drop trivial self-docs
    cand = [docs[i] for i in ids if not is_trivial(i, x)][:K - 1] + [null_doc]
    d = retriever.embed_doc(batch(cand))                # recompute with CURRENT theta
    log_pz  = F.log_softmax(retriever.score(q, d), -1).squeeze(0)   # top-K softmax p(z|x)
    log_pyz = torch.stack([predictor.logprob(x, z, y, task) for z in cand])
    return torch.logsumexp(log_pz + log_pyz, 0)         # log sum_z p(z|x) p(y|z,x)

class AsyncIndexRefresh:
    def __init__(self, every=500):
        self.every, self.pending = every, None
    def maybe_refresh(self, retriever, index, corpus, step):
        if self.pending is not None and self.pending.done():
            index.replace(self.pending.result())         # install finished background index
            self.pending = None
        if self.pending is None and step % self.every == 0:
            theta = copy.deepcopy(retriever).eval()      # snapshot theta' for index builder
            self.pending = launch_background(lambda: build_mips_index(theta, corpus))

def train(retriever, predictor, index, corpus, docs, data, steps, k, null_doc):
    refresher = AsyncIndexRefresh(every=500)
    opt = torch.optim.Adam(list(retriever.parameters()) + list(predictor.parameters()), lr=3e-5)
    for step, (x, y) in enumerate(data):                # x: salient-span-masked sentence
        loss = -marginal_logprob(x, y, retriever, predictor, index, docs, k, null_doc)
        opt.zero_grad(); loss.backward(); opt.step()
        refresher.maybe_refresh(retriever, index, corpus, step)
```

REALM sends the language-modeling gradient into a learned retriever during pre-training, making world knowledge explicit, retrievable, and editable. The MIPS index selects candidates, and asynchronous refresh lets that stale selector coexist with fresh gradients on the retrieved documents.
