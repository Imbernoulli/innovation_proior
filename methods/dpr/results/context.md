# Context

## Research question

Open-domain question answering asks a system to answer a factoid question — "Who first voiced Meg on Family Guy?", "Where was the 8th Dalai Lama born?" — against a corpus so large (English Wikipedia is ~21 million 100-word passages; the open web is orders larger) that no reader model can attend over all of it. The standard decomposition is two-stage: a *retriever* that selects a small candidate set of passages, then a *reader* that reads those candidates and extracts an answer span. Modern reading-comprehension models score above 80% exact match on SQuAD v1.1 when handed the gold paragraph; in the open-domain pipeline, where the reader relies on a retriever to find the paragraph, end-to-end exact match is far lower. The reader's accuracy is upper-bounded by whether the right passage is in the top-k it receives.

So the precise question is: how do we retrieve, from tens of millions of passages, the small set (k ≈ 20–100) that contains the answer — accurately and fast enough to serve queries in real time? "Fast enough" is a hard constraint: with M ~ 10⁷ passages, the retriever cannot run a fresh neural network over every (question, passage) pair at query time. The scoring function must let everything about the passages be pre-computed offline, reducing the online step to something a sub-linear index can serve in milliseconds.

## Background

**The sparse-retrieval status quo.** Open-domain QA retrieval has, for decades, meant TF-IDF or BM25 over an inverted index. Each passage and question is represented as a high-dimensional, sparse vector of weighted term counts; relevance is a weighted overlap of terms. BM25 (Robertson & Zaragoza, 2009) is the workhorse — it scores a passage by a saturating, length-normalized sum over the query terms it contains. The inverted index makes this extremely fast and the method needs no training. It excels at exact lexical match: a rare, selective keyword or phrase ("Thoros of Myr") is found precisely.

**Dense representations.** A dense, low-dimensional embedding of question and passage maps text composed of different tokens to nearby vectors when the meaning is close. Dense vector retrieval has a long history going back to Latent Semantic Analysis (Deerwester et al., 1990). Discriminatively trained dense encoders — learning embeddings from labeled (query, document) pairs — became popular for web search, ad relevance, cross-lingual retrieval, and entity linking (Huang et al., 2013; Gillick et al., 2019; Yih et al., 2011). Dense encodings are *learnable* — the embedding function can be tuned to the task — and they are *decomposable* if the question and passage are encoded independently, so passage vectors can be precomputed and indexed.

**The efficiency primitive: maximum inner-product search.** If similarity is the inner product of a question vector and a passage vector, then retrieval is exactly maximum inner-product search (MIPS) over the precomputed passage vectors. MIPS over millions–billions of vectors is a solved systems problem: in-memory data structures and quantization/graph indices (e.g. FAISS; Johnson et al., 2017) serve approximate nearest-neighbor / MIPS queries sub-linearly. Given good encoders, the serving infrastructure already exists.

**The current state of dense retrieval for QA.** The prevailing view is that learning a good dense retriever needs an enormous number of labeled question–context pairs. A recent pretraining-heavy approach (ORQA; Lee et al., 2019) reported a dense retriever matching or surpassing BM25 for open-domain QA, using an auxiliary self-supervised objective — the Inverse Cloze Task — to pretrain the retriever before fine-tuning.

**The encoder substrate.** BERT (Devlin et al., 2018) is a standard pretrained Transformer; taking the representation at the `[CLS]` token gives a fixed-size sentence/passage embedding (768-d for BERT-base). The dual-encoder / "Siamese" architecture (Bromley et al., 1994) — two towers producing comparable embeddings — is the natural shape for independently encoding questions and passages.

## Baselines

- **BM25 / TF-IDF sparse retrieval.** Question and passage as sparse weighted term vectors; relevance via inverted-index term overlap with BM25's saturation and length normalization. No training, very fast, strong on exact lexical match.

- **ORQA (Lee et al., 2019).** A dense retriever that matches or beats BM25 on open-domain QA. It introduces the Inverse Cloze Task (ICT): mask a sentence out of a passage and pretrain the retriever to predict which passage block the sentence came from, as a surrogate for the question→passage matching signal. The question encoder and reader are then fine-tuned jointly on (question, answer) pairs; the passage encoder is fixed after ICT pretraining.

- **REALM (Guu et al., 2020), concurrent.** Tunes the passage encoder as well, which forces periodic re-indexing of the whole corpus during training (the passage vectors drift, so the MIPS index is rebuilt) via asynchronous re-indexing during pretraining.

- **Iterative / phrase-level dense retrieval (Das et al., 2019; Seo et al., 2019).** Reformulate the question vector iteratively, or directly index answer-phrase vectors and skip passage retrieval.

- **Cross-attention rerankers (Wang et al., 2019; Nogueira et al., 2019).** A full BERT cross-attention model over a concatenated (question, passage) pair scores relevance with high accuracy. It is non-decomposable — it runs a network over each pair — so it is used as a reranker over a small candidate set.

## Evaluation settings

- **Corpus.** English Wikipedia dump (Dec. 20, 2018), cleaned with the DrQA preprocessing (drop tables, info-boxes, lists, disambiguation pages), split into **disjoint 100-word passages** as the retrieval unit; each passage prepended with its article title and a `[SEP]` token. This yields **21,015,324 passages**.
- **QA datasets.** Natural Questions (real Google queries, Wikipedia-span answers), TriviaQA (trivia Q/A scraped from the web), WebQuestions (Google Suggest queries, Freebase-entity answers), CuratedTREC (TREC + web sources, intended for unstructured-corpus open QA), and SQuAD v1.1 (reading-comprehension questions written against a given paragraph — included for comparability). Same train/dev/test splits as prior work.
- **Positive-passage construction.** Where datasets give only (question, answer): run BM25 with the question and answer together, use the highest-ranked passage that contains the answer as the positive, and discard the question if no top-100 passage contains the answer. Where gold contexts exist (SQuAD, NQ): match the gold paragraph to the corresponding 100-word passage in the candidate pool.
- **Retriever metric.** *Top-k retrieval accuracy* — the fraction of questions for which the top-k retrieved passages contain a span answering the question — reported in isolation (typically k from 1 to 100). End-to-end metric: **exact match** of the predicted answer span.
- **Efficiency protocol.** Offline passage encoding + index build, then per-query latency and throughput against the index. MIPS index: FAISS HNSW over precomputed passage vectors, with graph/search-depth settings controlling the latency/recall tradeoff.

## Code framework

The available pieces are a pretrained Transformer encoder that can be pooled to a fixed vector, an Adam optimizer with linear warmup, an ANN/MIPS index library, a BM25 system for constructing positives and hard negatives, and a small cross-attention reader scaffold for candidate passages. The empty slots are the parts left unimplemented below.

```python
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import BertModel

# --- Encoder: wrap a pretrained Transformer, pool to a fixed vector ---
class TextEncoder(nn.Module):
    def __init__(self, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)

    def forward(self, input_ids, attn_mask):
        # TODO: run BERT, pool the token sequence down to one fixed-size vector
        pass

# --- How do we score a (question, passage) pair so passages are precomputable? ---
def similarity(q_vecs, p_vecs):
    # TODO: a *decomposable* score between already-computed q and p vectors
    pass

# --- Training objective over a batch of (question, positive-passage[, negatives]) ---
def retrieval_loss(q_vecs, p_vecs, positive_idx):
    # TODO: turn "the right passage should score highest" into a differentiable loss
    pass

def build_batch(examples, tokenizer):
    # TODO: assemble the question / passage token tensors for a batch
    pass

# --- Offline indexing + online retrieval (serving) ---
class PassageIndex:
    def __init__(self, dim): ...      # an ANN/MIPS index over passage vectors (e.g. FAISS)
    def add(self, passage_vecs): pass # encode all M passages offline, add to index
    def search(self, q_vec, k): pass  # top-k by the similarity above

# --- Reader over the retrieved candidates (extractive span QA) ---
class Reader(nn.Module):
    def __init__(self, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        # TODO: heads for span start/end and passage selection

    def forward(self, pair_ids, pair_mask):
        # TODO: run BERT over [question, passage] pairs; score spans and passage selection
        pass

# --- Training loop scaffold ---
def train_retriever(encoder_q, encoder_p, data, steps):
    opt = torch.optim.Adam(...)   # optimizer and schedule placeholder
    for batch in data:
        # TODO: encode, score, loss, step
        pass
```
