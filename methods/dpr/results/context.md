# Context

## Research question

Open-domain question answering asks a system to answer a factoid question — "Who first voiced Meg on Family Guy?", "Where was the 8th Dalai Lama born?" — against a corpus so large (English Wikipedia is ~21 million 100-word passages; the open web is orders larger) that no reader model can possibly attend over all of it. The standard decomposition is two-stage: a *retriever* that selects a small candidate set of passages, then a *reader* that reads those candidates and extracts an answer span. The reader has gotten very good — modern reading-comprehension models score above 80% exact match on SQuAD v1.1 when handed the gold paragraph. But plug that same reader into the open-domain pipeline, where it must rely on a retriever to find the paragraph, and exact match collapses to under 40%. The bottleneck is unambiguous: it is the retriever. Everything the reader could do is upper-bounded by whether the right passage is even in the top-k it receives.

So the precise question is: how do we retrieve, from tens of millions of passages, the small set (k ≈ 20–100) that actually contains the answer — accurately and fast enough to serve queries in real time? "Fast enough" is a hard constraint, not a nicety: with M ~ 10⁷ passages, the retriever cannot run a fresh neural network over every (question, passage) pair at query time. Whatever scoring function we use must let us pre-compute everything about the passages offline and reduce the online step to something a sub-linear index can serve in milliseconds.

## Background

**The sparse-retrieval status quo.** Open-domain QA retrieval has, for decades, meant TF-IDF or BM25 over an inverted index. Each passage and question is represented as a high-dimensional, sparse vector of weighted term counts; relevance is a weighted overlap of terms. BM25 (Robertson & Zaragoza, 2009) is the workhorse — it scores a passage by a saturating, length-normalized sum over the query terms it contains. The inverted index makes this extremely fast and the method needs no training. Its strength is exact lexical match: a rare, selective keyword or phrase ("Thoros of Myr") is found precisely. Its weakness is the flip side: it is blind to meaning. The question "Who is the bad guy in Lord of the Rings?" should match "Sala Baker is best known for portraying the villain Sauron…", but "bad guy" and "villain" share no tokens, so a term-matching system cannot bridge them. Synonymy and paraphrase fall through the cracks.

**Dense representations as the complement.** A dense, low-dimensional embedding of question and passage is complementary by design: synonyms and paraphrases composed of entirely different tokens can be mapped to nearby vectors. Dense vector retrieval has a long history going back to Latent Semantic Analysis (Deerwester et al., 1990). Discriminatively trained dense encoders — learning embeddings from labeled (query, document) pairs — became popular for web search, ad relevance, cross-lingual retrieval, and entity linking (Huang et al., 2013; Gillick et al., 2019; Yih et al., 2011). The appeal is that dense encodings are *learnable* — the embedding function can be tuned to the task — and they are *decomposable* if the question and passage are encoded independently, so passage vectors can be precomputed and indexed.

**The efficiency primitive: maximum inner-product search.** If similarity is the inner product of a question vector and a passage vector, then retrieval is exactly maximum inner-product search (MIPS) over the precomputed passage vectors. MIPS over millions–billions of vectors is a solved systems problem: in-memory data structures and quantization/graph indices (e.g. FAISS; Johnson et al., 2017) serve approximate nearest-neighbor / MIPS queries sub-linearly. So *if* we can learn good encoders, the serving infrastructure already exists.

**The diagnostic that set the bar.** The prevailing wisdom, however, was that learning a good dense retriever needs an enormous number of labeled question–context pairs, and in fact dense retrieval had *never* been shown to beat BM25 for open-domain QA — until a pretraining-heavy approach (ORQA; Lee et al., 2019) crossed that line using an auxiliary self-supervised objective (the Inverse Cloze Task) to pretrain the retriever before fine-tuning. So the field's two beliefs at this moment were: (1) dense *can* beat sparse, but (2) only with expensive extra pretraining. That second belief is the thing to test.

**The encoder substrate.** BERT (Devlin et al., 2018) is now a standard pretrained Transformer; taking the representation at the `[CLS]` token gives a fixed-size sentence/passage embedding (768-d for BERT-base). The dual-encoder / "Siamese" architecture (Bromley et al., 1994) — two towers producing comparable embeddings — is the natural shape for independently encoding questions and passages.

## Baselines

- **BM25 / TF-IDF sparse retrieval.** Question and passage as sparse weighted term vectors; relevance via inverted-index term overlap with BM25's saturation and length normalization. No training, very fast, exact-match-strong. **Gap:** semantically blind — misses synonym/paraphrase matches with no lexical overlap. This is the bar a dense retriever must clear, and historically had not.

- **ORQA (Lee et al., 2019).** The first dense retriever to beat BM25 on open-domain QA. It introduces the Inverse Cloze Task (ICT): mask a sentence out of a passage and pretrain the retriever to predict which passage block the sentence came from, as a surrogate for the question→passage matching signal. The question encoder and reader are then fine-tuned jointly on (question, answer) pairs. **Gaps:** (1) ICT pretraining is computationally intensive, and it is not clear that a held-out sentence is a faithful surrogate for a real question; (2) the *passage/context* encoder is fixed after pretraining and never fine-tuned on question–answer pairs, so the passage representations may be suboptimal for the actual task.

- **REALM (Guu et al., 2020), concurrent.** Pushes the joint idea further by also tuning the passage encoder, which forces periodic re-indexing of the whole corpus during training (the passage vectors drift, so the MIPS index must be rebuilt). **Gap:** heavy machinery — asynchronous re-indexing during pretraining is expensive and complex.

- **Iterative / phrase-level dense retrieval (Das et al., 2019; Seo et al., 2019).** Reformulate the question vector iteratively, or directly index answer-phrase vectors and skip passage retrieval. **Gap:** more complex pipelines that have not cleanly beaten the simple retrieve-then-read paradigm on accuracy.

- **Cross-attention rerankers (Wang et al., 2019; Nogueira et al., 2019).** A full BERT cross-attention model over a concatenated (question, passage) pair scores relevance with high accuracy. **Gap:** non-decomposable — it must run a network over every pair at query time, so it cannot be used for first-stage retrieval over millions of passages; only as a reranker over a small candidate set.

## Evaluation settings

- **Corpus.** English Wikipedia dump (Dec. 20, 2018), cleaned with the DrQA preprocessing (drop tables, info-boxes, lists, disambiguation pages), split into **disjoint 100-word passages** as the retrieval unit; each passage prepended with its article title and a `[SEP]` token. This yields **21,015,324 passages**.
- **QA datasets.** Natural Questions (real Google queries, Wikipedia-span answers), TriviaQA (trivia Q/A scraped from the web), WebQuestions (Google Suggest queries, Freebase-entity answers), CuratedTREC (TREC + web sources, intended for unstructured-corpus open QA), and SQuAD v1.1 (reading-comprehension questions written against a given paragraph — included for comparability though many questions lack standalone context). Same train/dev/test splits as prior work.
- **Positive-passage construction.** Where datasets give only (question, answer): use the highest-ranked BM25 passage that contains the answer as the positive; discard the question if no top-100 passage contains the answer. Where gold contexts exist (SQuAD, NQ): match the gold paragraph to the corresponding 100-word passage in the candidate pool.
- **Retriever metric.** *Top-k retrieval accuracy* — the fraction of questions for which the top-k retrieved passages contain a span answering the question — reported in isolation (typically k from 1 to 100). End-to-end metric: **exact match** of the predicted answer span.
- **Efficiency protocol.** Offline passage encoding + index build (the costly step), then per-query latency and throughput against the index. MIPS index: an approximate ANN library (HNSW/flat) over the precomputed passage vectors.

## Code framework

The pieces that already exist before the method: a pretrained Transformer encoder we can pool to a fixed vector, an Adam optimizer with linear warmup, an ANN/MIPS index library, and a BM25 system for building positives/negatives. The method has to fill in the encoder pooling, the similarity, the contrastive training loss, and the retrieve-then-read glue.

```python
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import BertModel, BertTokenizer

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
    # TODO: assemble question / passage token tensors; where do negatives come from?
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

    def forward(self, passage_ids, passage_mask):
        # TODO: per-token start/end logits + a passage-selection score
        pass

# --- Training loop scaffold ---
def train_retriever(encoder_q, encoder_p, data, steps):
    opt = torch.optim.Adam(...)   # lr ~1e-5, linear warmup
    for batch in data:
        # TODO: encode, score, loss, step
        pass
```
