# Context

## Research question

Autoregressive language modeling factorizes text as p(x₁,…,x_n) = ∏ᵢ p(xᵢ | x_<i), and the dominant way to improve a neural LM has been scale — more parameters, more data, more compute. Scaling Transformers from ~100M to >100B parameters predictably improves downstream performance. This scaling brings two distinct benefits: more *computation* at train and inference time, and more *memorization* of the training data baked into the weights. The bigger a model, the more of its capacity is spent storing facts and phrasings it has seen, and the more expensive each forward pass.

The question: can a model be given direct access to a large text database at prediction time, so that the "memorization" benefit comes from *retrieval* rather than from parameters? The setting of interest is large-scale: the database should be allowed to grow toward *trillions* of tokens (existing retrieval-LM work used ≤ a few billion), and incorporating the retrieved text should cost time *linear* in the amount retrieved. With retrieval, a model's knowledge would also be inspectable and updatable — swap the database — rather than frozen in weights.

## Background

**Scale and memorization.** Large LMs (GPT-2, GPT-3, Jurassic-1) improve steeply with size, and scaling laws make the improvement predictable. Large models also *memorize* and can reproduce verbatim chunks of their training data; that memorization is part of why scale helps.

**Test-set leakage.** Large training sets overlap with evaluation sets, and this leakage inflates measured performance. For a retrieval-augmented model the issue is direct: it can *access the training data at evaluation time* and copy a leaked chunk. An evaluation of a retrieval LM therefore controls for the overlap between an evaluation chunk and the training database.

**Sparse vs. dense retrieval.** Classical text retrieval is sparse term matching — TF-IDF, BM25 over an inverted index — and topic models like LDA. With deep learning, retrieval moved to *dense* learned representations of a network's activations. The relevant lineage:
- **Continuous cache** (Grave et al., 2016) and **kNN-LM** (Khandelwal et al., 2020): at inference, retrieve tokens whose stored activation resembles the current one and *interpolate* the LM's next-token distribution with a distribution computed from the retrieved tokens. kNN-LM extended the store to all of Wikipedia and improved Wikitext103. These methods do not modify the model — they interpolate output distributions — and retrieve at the granularity of *individual tokens*.
- **SPALM** (Yogatama et al., 2021): adds a gating network to post-process retrieved data.
- **DPR** (Karpukhin et al., 2020): trains two BERT encoders (query, key) with a contrastive loss to align a question with its answer passages — dense *passage*-level retrieval, trained in isolation from the downstream task, primarily for open-domain QA.
- **ORQA** (Lee et al., 2019): an inverse-cloze-task pretrained passage retriever.
- **REALM** (Guu et al., 2020): trains the retriever *end-to-end* against the LM cross-entropy, which requires searching the database during training and *periodically re-embedding/re-indexing* the whole corpus. REALM *prepends* the retrieved document to the prompt.
- **RAG** (Lewis et al., 2020) and **FiD** (Izacard & Grave, 2020): build on DPR with encoder-decoder Transformers; FiD encodes retrieved passages separately and fuses them in the decoder. Both target QA.

**Reusable primitives.** A pre-trained, *frozen* BERT can embed a span of text into a fixed vector — averaging token embeddings over time gives a key. Approximate nearest-neighbour libraries (e.g. SCaNN) retrieve top-k by distance in O(log T) for a T-element database, so a multi-trillion-token store is queryable in ~10ms. The Transformer's encoder-decoder cross-attention (Vaswani et al., 2017) is the standard differentiable way to let a decoder attend over encoded auxiliary content. RMSNorm and relative positional encodings are available drop-in alternatives to the original Transformer's LayerNorm and absolute positions.

## Baselines

- **Purely-parametric Transformer LM (GPT-style).** Decoder-only, scaled to 100M–100B+ params. Knowledge is added by adding parameters, and is stored in the weights.

- **kNN-LM / continuous cache.** Frozen retriever; interpolate the LM's token distribution with a retrieval-derived distribution at inference. Token-level retrieval.

- **REALM / RAG / FiD / EMDR.** End-to-end-trained dense retrievers + readers/generators, state-of-the-art on QA. Training searches and periodically re-indexes the database; built for QA single-shot retrieval.

- **SPALM.** Frozen retrieval + a gating post-processor.

## Evaluation settings

- **Data.** MassiveText (multi-lingual; >5T tokens; Web 55%, Books 25%, News 10%, Wikipedia 5%, GitHub 5%), tokenized with SentencePiece, vocabulary 128,000. Training retrieves from a 600B-token database matching the training mix; evaluation retrieves from the full union (~1.75T tokens, books sub-sampled to 4%).
- **Leakage control.** Deduplicate: remove training documents with ≥0.8 13-gram Jaccard (MinHash) similarity to any validation/test document; remove Wikitext103 val/test articles from training Wikipedia. Because a retrieval LM can access the training database at evaluation time, a reported metric accounts for residual overlap between an evaluation chunk and its nearest training chunk.
- **Benchmarks.** Language modeling: Wikitext103, the Pile, C4, Curation Corpus (bits-per-byte). Downstream: open-domain QA on Natural Questions (exact match), using the DPR/FiD-provided retrieved passages.
- **Model scales.** Baseline retrieval-free Transformers at 132M, 368M, 1.3B, 7.0B parameters (embeddings excluded), with matched retrieval-augmented counterparts.

## Code framework

The pieces already available are a decoder-only Transformer LM stack (attention + feed-forward residual blocks, RMSNorm, relative positions), a frozen pre-trained BERT embedder, an approximate-nearest-neighbour index, and the standard encoder-decoder cross-attention operator. What remains is to settle how the external store is queried and how its contents enter the decoder.

```python
import torch, torch.nn as nn

class FrozenEmbedder:
    def embed(self, tokens): pass            # frozen BERT, mean-pooled over time

class ANNIndex:
    def build(self, keys, values): pass      # values are raw text spans
    def search(self, query_vec, k): pass     # approximate top-k by L2 distance, O(log T)

def causal_attn(H): ...
def ffw(H): ...
def cross_attn(Q, KVs): ...

def split_segments(tokens, segment_len):
    # TODO
    pass

def fetch_external_text(segments, embedder, index, k):
    # TODO
    pass

class ExternalTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, external_tokens, segment_acts):
        # TODO
        pass

def local_memory_update(H, encoded_external, segment_len):
    # TODO
    pass

class MemoryAugmentedDecoder(nn.Module):
    def __init__(self, L, access_layers):
        super().__init__()
        # TODO
        pass
    def forward(self, X, neighbours):
        # TODO
        pass

def loss(X, neighbours, model):
    # autoregressive NLL with retrieval-augmented conditioning
    pass
```
