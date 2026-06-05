# Context

## Research question

Autoregressive language modeling factorizes text as p(x₁,…,x_n) = ∏ᵢ p(xᵢ | x_<i), and the dominant way to make a neural LM better has been brute scale — more parameters, more data, more compute. Scaling Transformers from ~100M to >100B parameters predictably improves downstream performance. But that scaling conflates two distinct benefits: (1) more *computation* at train and inference time, and (2) more *memorization* of the training data baked into the weights. The bigger a model, the more of its capacity is spent simply storing facts and phrasings it has seen — and serving that giant model is expensive at every forward pass.

The question: can these two be *decoupled*? Specifically, can a model be given a massive-scale memory — direct access to a large text database at prediction time — so that the "memorization" benefit comes from *retrieval* rather than from parameters, *without* significantly increasing the model's computation? If so, a semi-parametric LM could match a far larger purely-parametric one while doing much less compute, and its knowledge would be inspectable and updatable (swap the database) rather than frozen in weights. Two hard constraints make this nontrivial at the scale that matters: the database must be allowed to grow to *trillions* of tokens (existing retrieval-LM work used ≤ a few billion), and incorporating the retrieved text must cost *time linear in the amount retrieved* — anything quadratic kills the whole point.

## Background

**Scale and memorization.** Large LMs (GPT-2, GPT-3, Jurassic-1) improve steeply with size, and scaling laws make the improvement predictable. But large models are shown to *memorize* and even reproduce verbatim chunks of their training data. That memorization is part of why scale helps — which is precisely the part one might offload to an external store.

**Test-set leakage.** Large training sets overlap with evaluation sets, and this leakage inflates measured performance — an acute problem for retrieval-augmented models, which can *directly access the training data at evaluation time* and copy a leaked chunk. Any honest evaluation of a retrieval LM must control for the overlap between an evaluation chunk and the training database.

**Sparse vs. dense retrieval.** Classical text retrieval is sparse term matching — TF-IDF, BM25 over an inverted index — and topic models like LDA. With deep learning, retrieval moved to *dense* learned representations of a network's activations. The relevant lineage:
- **Continuous cache** (Grave et al., 2016) and **kNN-LM** (Khandelwal et al., 2020): at inference, retrieve tokens whose stored activation resembles the current one and *interpolate* the LM's next-token distribution with a distribution computed from the retrieved tokens. kNN-LM extended the store to all of Wikipedia and improved Wikitext103. Crucially, these **do not modify the model** — they interpolate output distributions, so the network never *reasons* over the retrieved text; it only mixes probabilities. They also retrieve at the granularity of *individual tokens*.
- **SPALM** (Yogatama et al., 2021): adds a gating network to post-process retrieved data, but most of the network is still untouched by retrieval.
- **DPR** (Karpukhin et al., 2020): trains two BERT encoders (query, key) with a contrastive loss to align a question with its answer passages — dense *passage*-level retrieval, trained in isolation from the downstream task, primarily for open-domain QA.
- **ORQA** (Lee et al., 2019): an inverse-cloze-task pretrained passage retriever.
- **REALM** (Guu et al., 2020): trains the retriever *end-to-end* against the LM cross-entropy, which requires searching the database during training and *periodically re-embedding/re-indexing* the whole corpus — expensive, and it severely limits the scale (corpus size) at which it can operate. REALM *prepends* the retrieved document to the prompt.
- **RAG** (Lewis et al., 2020) and **FiD** (Izacard & Grave, 2020): build on DPR with encoder-decoder Transformers; FiD encodes retrieved passages separately and fuses them in the decoder. Both target QA, not arbitrary-text language modeling.

**Why none of these solve the stated problem.** The end-to-end trained retrievers (REALM, RAG, FiD, EMDR) can't scale the database past a few billion tokens because training has to search and periodically re-index it. The frozen-retriever interpolation methods (kNN-LM, continuous cache) scale better but never let the network *reason* over retrieved text and work at the token level. And all of these are built around QA-style single-shot retrieval on a prompt — none is designed to model *arbitrary long text sequences*, where different parts of a 2000-token sequence need *different* retrievals.

**Reusable primitives.** A pre-trained, *frozen* BERT can embed a span of text into a fixed vector — averaging token embeddings over time gives a key. Approximate nearest-neighbour libraries (e.g. SCaNN) retrieve top-k by distance in O(log T) for a T-element database, so a multi-trillion-token store is queryable in ~10ms. The Transformer's encoder-decoder cross-attention (Vaswani et al., 2017) is the standard differentiable way to let a decoder attend over encoded auxiliary content. RMSNorm and relative positional encodings are available drop-in improvements over the original Transformer's LayerNorm and absolute positions.

## Baselines

- **Purely-parametric Transformer LM (GPT-style).** Decoder-only, scaled to 100M–100B+ params. **Gap:** the only way to add knowledge is more parameters → more compute per token at train and inference; knowledge frozen in weights, uninspectable, unupdatable.

- **kNN-LM / continuous cache.** Frozen retriever; interpolate the LM's token distribution with a retrieval-derived distribution at inference. **Gap:** the network cannot reason over retrieved text (only output probabilities are mixed); token-level retrieval; no benefit during training.

- **REALM / RAG / FiD / EMDR.** End-to-end-trained dense retrievers + readers/generators, state-of-the-art on QA. **Gap:** training must search and periodically re-index the database, capping it at ≤ a few billion tokens; built for QA single-shot retrieval, not arbitrary-length sequence modeling with per-segment retrieval.

- **SPALM.** Frozen retrieval + a gating post-processor. **Gap:** most of the network is still untouched by retrieval.

## Evaluation settings

- **Data.** MassiveText (multi-lingual; >5T tokens; Web 55%, Books 25%, News 10%, Wikipedia 5%, GitHub 5%), tokenized with SentencePiece, vocabulary 128,000. Training retrieves from a 600B-token database matching the training mix; evaluation retrieves from the full union (~1.75T tokens, books sub-sampled to 4%).
- **Leakage control.** Deduplicate: remove training documents with ≥0.8 13-gram Jaccard (MinHash) similarity to any validation/test document; remove Wikitext103 val/test articles from training Wikipedia. A *filtered* evaluation reports bits-per-byte restricted to evaluation chunks whose maximal contiguous token overlap with the nearest training chunk is below a threshold α — so performance on genuinely novel text is separable from leakage exploitation.
- **Benchmarks.** Language modeling: Wikitext103, the Pile, C4, Curation Corpus (bits-per-byte). Downstream: open-domain QA on Natural Questions (exact match), using the DPR/FiD-provided retrieved passages.
- **Model scales.** Baseline retrieval-free Transformers at 132M, 368M, 1.3B, 7.0B parameters (embeddings excluded), with matched retrieval-augmented counterparts.

## Code framework

The pieces that already exist: a decoder-only Transformer LM stack (attention + feed-forward residual blocks, RMSNorm, relative positions), a frozen pre-trained BERT embedder, an approximate-nearest-neighbour index, and the standard encoder-decoder cross-attention operator. The method must define how a long sequence is segmented for retrieval, how neighbours are fetched and encoded, and — the core new operator — how retrieved content is fused into an autoregressive decoder at linear cost without breaking causality.

```python
import torch, torch.nn as nn

# --- Frozen embedder + ANN index (already exist) ---
class FrozenEmbedder:
    def embed(self, tokens): pass            # frozen BERT, mean-pooled over time -> key vector

class ANNIndex:
    def build(self, keys, values): pass      # values = raw text chunks (+ their continuations)
    def search(self, query_vec, k): pass     # approximate top-k by L2 distance, O(log T)

# --- Standard residual operators (already exist) ---
def attn(H): ...                             # causal self-attention residual
def ffw(H): ...                              # feed-forward residual

# --- How is a long sequence prepared for retrieval? ---
def split_into_chunks(X, m):
    # TODO: split the n-token sequence into l = n/m contiguous chunks
    pass

def retrieve_neighbours(chunks, embedder, index, k):
    # TODO: for each chunk, fetch k nearest neighbours (+ continuations); which chunk
    #       may a position attend to, to stay autoregressive?
    pass

# --- The contribution: encode neighbours and fuse them into the decoder ---
class NeighbourEncoder(nn.Module):
    def __init__(self): super().__init__()   # a (bidirectional) encoder over retrieved text
    def forward(self, neighbours, chunk_acts):
        # TODO: encode each neighbour; condition on the retrieving chunk's activations
        pass

def chunked_fusion(H, E):
    # TODO: the new operator that lets decoder positions attend to the right chunk's
    #       encoded neighbours, at cost LINEAR in retrieved data and preserving causality
    pass

class RetrievalLM(nn.Module):
    def __init__(self, L, fuse_layers):      # interleave fusion blocks among LM blocks
        super().__init__()
        # TODO: embedding, L decoder layers (some with chunked_fusion), read-out head
        pass
    def forward(self, X, neighbours):
        # TODO: run decoder; encode neighbours once; fuse at the designated layers
        pass

def loss(X, neighbours, model):
    # autoregressive NLL where each token may use neighbours of EARLIER chunks only
    pass
```
