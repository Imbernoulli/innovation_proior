# Context

## Research question

Pre-trained language models — BERT, RoBERTa, T5 — store a surprising amount of world knowledge: BERT will fill "The ___ is the currency of the United Kingdom" with "pound." But that knowledge lives *implicitly*, smeared across millions of weights. Three problems follow. It is opaque — you cannot tell what the network knows or where it is stored. It is bounded by capacity — to hold more facts you must train an ever-larger network, which becomes prohibitively slow and expensive. And it is unmodular — you cannot revise, inspect, or swap out a fact without retraining.

The question: can language-model pre-training be augmented so that world knowledge is captured *explicitly and modularly* — stored in an external textual corpus the model learns to *retrieve* from, rather than memorized in parameters? Concretely, before each prediction the model would consult a learned retriever, pull documents from a corpus (e.g. all of Wikipedia), and attend over them to inform its prediction. The crux is *learning the retriever end-to-end from the language-modeling signal alone*, with no labels for which document is the right one — and doing so when the retriever must score *millions* of candidate documents at every training step, with gradients flowing back through that retrieval. The pre-trained model would then transfer to a knowledge-intensive downstream task (open-domain QA) where the explicit-knowledge advantage should show.

## Background

**Masked language model pre-training.** The dominant pre-training recipe (BERT) is the masked language model (MLM): take unlabeled text, randomly mask some tokens, train the model to predict the masked tokens from the surrounding context. To fill "The ___ is the currency of the UK" the model must encode syntax/semantics *and* world knowledge. The pre-trained encoder is then fine-tuned on a downstream task. This is the substrate to augment.

**Open-domain QA as the knowledge yardstick.** Open-domain QA (Open-QA): given a question, output the answer string, with *no* pre-identified evidence document — unlike reading comprehension (SQuAD), where the gold passage is handed to the model. Open-QA forces a system to retain or retrieve knowledge from millions of documents, so it is the natural test of whether knowledge is accessible. Two paradigms exist. *Retrieval-based*: retrieve candidate documents from a textual corpus, then extract the answer span (DrQA, ORQA). *Generation-based*: a seq2seq model encodes the question and decodes the answer token-by-token from parametric knowledge (T5, GPT-2).

**Dense inner-product retrieval and MIPS.** A retriever can score a (query, document) pair by the inner product of dense embeddings, f(x, z) = Embed_input(x)ᵀ Embed_doc(z); the retrieval distribution is a softmax over relevance scores. Because the ordering by p(z|x) matches the ordering by inner product, finding the top-k reduces to Maximum Inner Product Search (MIPS), which scales sub-linearly in the corpus size with the right index. This requires precomputing Embed_doc(z) for every document and building a search index — fast, but the index is only valid for the *current* doc-encoder parameters.

**Latent-variable view of retrieve-then-predict.** If "which document was retrieved" is unobserved, the document z is a latent variable and the output likelihood marginalizes over it: p(y|x) = Σ_z p(y|z,x) p(z|x). Maximizing this marginal trains the retriever without document labels — a retrieval that raises the probability of the correct y is implicitly rewarded. ORQA (Lee et al., 2019) is built exactly this way for Open-QA and trains by maximizing the marginal likelihood, with the retriever warm-started by the Inverse Cloze Task (ICT) — given a sentence, predict the passage it came from — and a *fixed* document index.

**Memory- and retrieval-augmented networks (prior art and its gaps).** Adding a discrete retrieval step to neural nets has a long history — key-value memory networks, DrQA — but these used *non-learned* (e.g. TF-IDF/BM25) retrievers for the large-corpus step and never applied retrieval to *pre-training*. The kNN-LM (Khandelwal et al., 2020) retrieves similar LM contexts to improve memorization, but it was not fine-tuned for downstream tasks: a kNN over labeled examples cannot transfer, because at fine-tuning time the LM examples carrying the world knowledge are not labeled for the target task. The opening is a retriever that (a) is *learned*, (b) operates during *pre-training*, (c) backpropagates into the index, and (d) retrieves raw text so it transfers.

**Diagnostic facts about MLM spans.** Not every masked token needs world knowledge — many ("of", function words, locally-predictable tokens) need only local context. Spans that *do* need world knowledge are typically *salient* — named entities ("United Kingdom") and dates ("July 1969"). This distinction is a pre-method fact that will matter for which masking forces the retriever to be useful.

## Baselines

- **BERT-style MLM (parametric only).** Knowledge implicit in weights. **Gap:** opaque, capacity-bounded, unmodular — the very problems being attacked.

- **T5 / generation-based Open-QA (Roberts et al., 2020, concurrent).** A large seq2seq model generates the answer from parameters. Scales knowledge by scaling parameters — Base, Large, up to 11B. **Gap:** knowledge is implicit and uninterpretable; requires enormous parameter counts; cannot be edited without retraining.

- **Heuristic-retrieval + reading (DrQA, HardEM, GraphRetriever, PathRetriever).** Retrieve ~20–80 documents with a *non-learned* sparse retriever (TF-IDF/BM25) or entity linking, optionally re-rank with a learned model, then read. **Gap:** the initial heuristic retrieval caps coverage — relevant documents missed by BM25 are never recoverable; the retriever is not optimized for the end task.

- **ORQA (Lee et al., 2019).** The closest prior: a latent-variable Open-QA model with a learned MIPS retriever, trained by marginal-likelihood maximization, retriever warm-started with ICT. **Gap:** the retriever is only *heuristically* pre-trained (ICT) and the document index is *fixed* during training — gradients never flow into the doc encoder, so the retrieval is never refined by a language-modeling signal; there is no knowledge-rich pre-training stage tailored to make retrieval useful.

## Evaluation settings

- **Knowledge corpus Z.** December 20, 2018 English Wikipedia snapshot, greedily split into chunks of up to 288 BERT wordpieces, giving just over **13 million** retrieval candidates. Each document is encoded by the doc encoder; a MIPS index is built over the embeddings.
- **Pre-training corpus X.** Either Wikipedia (same as Z) or CC-News (an English-news reproduction), to test single-corpus vs. separate-corpus pre-training.
- **Downstream task / datasets.** Open-QA on NaturalQuestions-Open (short-answer questions, ≤5 tokens), WebQuestions, and CuratedTrec (answers as regex matching all variants). Predicted answers scored by **exact match** against references.
- **Protocol knobs (pre-method facts about the setup).** Top-k documents marginalized per example (small k, e.g. 5–8); a BERT-base encoder (12 layers, 768 hidden, 12 heads) backbone; retriever warm-start objective (ICT). At inference the small-k retrieval makes the full model runnable on a single 12 GB GPU. (Retrieval-based competitors typically retrieve 20–80 documents.)

## Code framework

The pieces that already exist: BERT-style Transformer encoders, an MLM head, a MIPS index library, and an SGD/Adam optimizer. The method must define the dense retriever's scoring, the knowledge-augmented predictor that conditions on a retrieved document, the marginal likelihood that ties them together, and — the hard systems part — how to keep a MIPS index over millions of documents usable while the doc encoder is being trained.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# --- Dense retriever: embed query and document, score by inner product ---
class Retriever(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # TODO: a Transformer encoder + projection for the query embedding
        # TODO: a Transformer encoder + projection for the doc embedding
        pass
    def score(self, x, z):
        # f(x, z) = embed_input(x) . embed_doc(z)
        pass
    def doc_embed(self, z):
        # used offline to build the index
        pass

# --- Predictor conditioned on a retrieved document (p(y | z, x)) ---
class KnowledgeAugmentedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: a (separate) Transformer over the joined (x, z); a head for the task
        pass
    def forward(self, x, z, y):
        # MLM-style token prediction (pre-train) or answer span (fine-tune)
        pass

# --- Index over millions of doc embeddings; valid only for current doc encoder ---
class MIPSIndex:
    def build(self, doc_embeddings): pass      # precompute + index ALL documents
    def search(self, query_vec, k):  pass      # approximate top-k by inner product

# --- The contribution: marginal likelihood + keeping the index usable while training ---
def marginal_logprob(x, y, retriever, predictor, index, k):
    # p(y|x) = sum_z p(y|z,x) p(z|x), approximated over the top-k retrieved z
    # TODO: retrieve top-k, score them, combine p(z|x) with p(y|z,x)
    pass

def maybe_refresh_index(retriever, index, corpus, step):
    # the doc embeddings drift as the retriever trains -> the index goes stale
    # TODO: how / how often to rebuild it without stalling training?
    pass

def train(retriever, predictor, index, corpus, data, steps, k):
    opt = torch.optim.Adam(...)   # BERT default optimizer
    for step, (x, y) in enumerate(data):
        loss = -marginal_logprob(x, y, retriever, predictor, index, k)
        opt.zero_grad(); loss.backward(); opt.step()
        maybe_refresh_index(retriever, index, corpus, step)
```
