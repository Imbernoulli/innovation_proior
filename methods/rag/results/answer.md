# Retrieval-Augmented Generation (RAG)

## Problem

A closed-book pre-trained language model stores all its knowledge in its weights: it cannot be cheaply updated, cannot cite a source, and hallucinates. Prior retrieval-augmented hybrids (ORQA, REALM) fixed this only for *extractive* open-domain QA with a masked-LM reader. RAG brings a non-parametric memory to a *general-purpose* pre-trained seq2seq generator and trains the retriever end-to-end with no document-level supervision.

## Key idea

Pair a parametric memory (a pre-trained encoder–decoder, BART-large, ~400M params) with a non-parametric memory (a dense vector index of Wikipedia accessed by a pre-trained DPR bi-encoder). Treat the retrieved document z as a **latent variable** and marginalize it to define p(y|x), then maximize the marginal likelihood of the observed output — this trains the retriever purely from the downstream generation signal.

- **Retriever** (DPR bi-encoder): p_η(z|x) ∝ exp(d(z)ᵀ q(x)), with d, q both BERT-base. Top-k retrieval is MIPS over the precomputed index (FAISS + HNSW).
- **Generator** (BART-large): p_θ(y_i | x, z, y_<i); the document is fed in by concatenating x and z.

## Two marginalizations

**RAG-Sequence** — one document for the whole output (sum *outside* the product):

  p(y|x) ≈ Σ_{z∈top-k} p_η(z|x) · ∏_i p_θ(y_i | x, z, y_<i).

**RAG-Token** — a different document allowed per token (sum *inside* the product):

  p(y|x) ≈ ∏_i Σ_{z∈top-k} p_η(z|x) · p_θ(y_i | x, z, y_<i).

For a length-one output (classification, e.g. FEVER), the product collapses and the two models are identical.

## Training

Minimize the negative marginal log-likelihood Σ_j −log p(y_j|x_j) with Adam, no supervision on which z to retrieve. The **document encoder and index are kept fixed** (only the query encoder and BART are fine-tuned), avoiding REALM-style periodic re-indexing — this is found unnecessary for strong performance. Trainable params ≈ 110M (query encoder) + 406M (BART) ≈ a few hundred million, far fewer than a comparable closed-book model.

## Decoding

- **RAG-Token** defines a proper marginalized transition p'_θ(y_i|x,y_<i) = Σ_z p_η(z|x) p_θ(y_i|x,z,y_<i) → plug into standard beam search.
- **RAG-Sequence** does not factor per token: run beam search per document to get candidate set Y, then score each y under every document (extra forward passes for ones missing from a document's beam), weight by p_η(z|x), and sum — "thorough decoding." Approximating p_θ(y|x,z)≈0 for y absent from z's beam gives cheaper "fast decoding" for long outputs.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class DenseRetriever:                  # fixed doc encoder + index; trainable query encoder
    def __init__(self, query_encoder, doc_index):
        self.q_enc, self.index = query_encoder, doc_index   # FAISS MIPS over fixed doc vectors
    def retrieve(self, x, k):
        q = self.q_enc(x)
        docs, doc_scores = self.index.search(q, k)          # raw q(x)·d(z) scores
        return docs, doc_scores

class Seq2SeqGenerator(nn.Module):     # BART-large; doc enters by concatenation
    def __init__(self, bart): super().__init__(); self.bart = bart
    def forward(self, x, z, y):
        return self.bart(input=concat(x, z), labels=y).logits   # (n_docs, T, V)

def ragtoken_logprob(seq_logits, doc_scores, n_docs):           # marginalize per token
    seq_lp = F.log_softmax(seq_logits, -1).view(
        seq_logits.shape[0]//n_docs, n_docs, -1, seq_logits.size(-1))
    doc_lp = F.log_softmax(doc_scores, 1)
    return torch.logsumexp(seq_lp + doc_lp.unsqueeze(-1).unsqueeze(-1), dim=1)

def ragsequence_nll(seq_logits, target, doc_scores, n_docs):    # marginalize per sequence
    seq_lp = F.log_softmax(seq_logits, -1).view(
        seq_logits.shape[0]//n_docs, n_docs, -1, seq_logits.size(-1))
    doc_lp = F.log_softmax(doc_scores, 1).unsqueeze(-1).unsqueeze(-1)
    seq_lp = torch.cat([seq_lp[:, :, :1, :],                    # fold doc log-prob into ONE token
                        seq_lp[:, :, 1:2, :] + doc_lp,
                        seq_lp[:, :, 2:, :]], dim=2)
    ll = gather_target(seq_lp, target).sum(2)                   # sum over tokens -> (B, K)
    return -ll.logsumexp(1)                                     # logsumexp over docs

def train(retriever, generator, data, n_docs):
    opt = torch.optim.Adam(list(retriever.q_enc.parameters()) + list(generator.parameters()))
    for x, y in data:
        docs, doc_scores = retriever.retrieve(x, n_docs)
        loss = ragsequence_nll(generator(x, docs, y), y, doc_scores, n_docs).mean()
        opt.zero_grad(); loss.backward(); opt.step()
```

RAG augments a pre-trained generator with a swappable, raw-text non-parametric memory, training retrieval and generation jointly through latent-variable marginalization — yielding a single architecture that is editable, source-grounded, and applies across knowledge-intensive QA, generation, and classification.
