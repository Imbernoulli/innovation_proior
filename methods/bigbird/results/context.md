# Context

## Research question

Transformer encoders built on self-attention have displaced recurrent models across natural-language tasks, and pretraining them on large corpora transfers well to downstream tasks. Each token attends to every other token: the full attention map is an `n × n` matrix of query–key inner products, so both compute and memory grow as `O(n²)` in the sequence length `n`. On the accelerators and model sizes in common use this caps a single sequence at roughly 512 tokens.

Many problems put their signal beyond that window: answering a question against a long passage, classifying or summarizing a long document, or modeling a genomic sequence whose functional context spans thousands of bases. The question is how to make attention scale **linearly** in `n` — using far fewer than `n²` inner products. Two guarantees are known for full attention: that the encoder is a universal approximator of continuous sequence-to-sequence functions, and that the full encoder–decoder is Turing complete.

## Background

**Self-attention as the unit of computation.** A single attention head maps a sequence `X = (x_1,…,x_n) ∈ R^{n×d}` through learned projections `Q(x)=xW_Q`, `K(x)=xW_K`, `V(x)=xW_V` and a scoring function `σ` (softmax) to outputs `Attn(X)_i = Σ_j σ(Q(x_i)K(x_j)^T/√m) V(x_j)`. Multi-head attention runs `H` such heads in parallel; an encoder layer wraps attention with a residual connection and a two-layer feed-forward network, also residual. Because every output position sums over all `n` keys, the score matrix is dense `n × n`.

**Attention as a graph.** The set of inner products a mechanism evaluates can be read as the arcs of a directed graph `D` on vertices `[n]`: an arc `i → j` means query `i` attends to key `j`, and `N(i)` is the out-neighborhood that `i` sums over. The dense map is the complete digraph; its adjacency matrix is all-ones. Reducing the number of inner products is then the question of which arcs to keep — a graph-sparsification problem — and graph theory supplies tools to reason about it. Two graph properties relate a sparse graph to a complete one:

- *Short paths / spectral gap.* Erdős–Rényi random graphs with `Θ̃(n)` edges have shortest paths of length `O(log n)` between any two nodes, and they approximate the complete graph spectrally — the second eigenvalue of the adjacency matrix is well separated from the first. A large spectral gap means random walks mix rapidly, the graph-theoretic way of saying information can travel between any pair of nodes in a few hops.
- *Locality / clustering.* Empirical analysis of trained attention (Clark et al., 2019) finds that neighboring inner products carry a large share of the signal; linguistic structure and biological sequences both display strong locality of reference. The clustering coefficient measures this: it is high when the graph is full of near-cliques. Plain random graphs have low clustering. The small-world model of Watts and Strogatz (1998) — start from a ring lattice in which each node connects to its `w` nearest neighbors, then rewire a small fraction of edges at random — attains high clustering *and* short paths at once.

**The expressivity proofs.** The universal-approximation argument for dense transformers (Yun et al., 2019) proceeds by approximating a continuous function with a piecewise-constant one on a fine grid, then using attention to build a *contextual mapping* — a unique numeric code for each (sequence, position) pair — which feed-forward layers decode to the target; the code is produced by a selective-shift construction over the columns. The Turing-completeness argument (Pérez et al., 2019) simulates a machine with an encoder–decoder under arbitrary precision. Both proofs are written for full attention.

## Baselines

- **Full self-attention (Vaswani et al., 2017; BERT, Devlin et al., 2018).** Every token attends to every token; `σ` is softmax over all `n` keys. The same architecture reaches the top across many benchmarks, and is backed by the expressivity guarantees above. Cost is `O(n²)` in time and memory.

- **Sliding-window-only attention (e.g. Wang et al., 2019).** Restrict each query to a fixed band of neighbors. Linear and locality-respecting; information crosses the sequence by stacking many layers.

- **Retrieve-then-read pipelines (SpanBERT, ORQA, REALM, RAG).** Select a small relevant subset of context to feed a standard transformer, possibly iterating, sidestepping length.

- **Recurrence / compression for left-to-right LMs (Transformer-XL, Dai et al., 2019; adaptive and compressive memories, Sukhbaatar et al., 2019; Rae et al., 2019).** Carry or compress past activations across segments. Built around a causal direction for autoregressive language modeling.

- **Sparse / hashed attention patterns.** Child et al. (2019) factorize attention into strided patterns for `O(n√n)`. Kitaev et al. (2019, Reformer) use locality-sensitive hashing to attend to near neighbors at `O(n log n)`. Others use binary partitions (Ye et al., 2019) or block sparsity (Qiu et al., 2019).

- **Window + few global tokens (Longformer, Beltagy et al., 2020; ETC, Ainslie et al., 2020).** Combine a local window with a handful of tokens that attend globally; ETC introduced the idea of dedicated global tokens to encode structure. Offered as engineering recipes.

## Evaluation settings

The natural yardstick is the masked-language-model pretraining setup used for bidirectional encoders, followed by fine-tuning, with attention measured at the standard length (512) and, for any candidate whose cost is genuinely linear, at longer lengths. Pretraining corpora are the large general-purpose text collections (book and web corpora) used for BERT-style models. Downstream protocols are the established encoder benchmarks: extractive and multi-hop question answering, natural-language inference, and long-document classification and summarization, plus genomics tasks such as promoter-region and chromatin-profile prediction from DNA. Metrics are the per-task standards (masked-LM accuracy; exact-match / F1 for QA; accuracy for NLI; ROUGE for summarization). Hardware is the GPU/TPU accelerators whose memory budget creates the `O(n²)` bottleneck in the first place; throughput and maximum feasible sequence length are part of the setting.

## Code framework

The pieces that exist before the mechanism: a token embedding, a place to add position information, a stack of identical encoder layers each wrapping an attention step and a feed-forward step with residual connections, and the standard projection-and-softmax machinery for a single dense head. The open slot is the attention pattern itself — which keys each query is allowed to see — and how to compute it efficiently on hardware that dislikes fine-grained sparsity.

```python
import math
import torch
import torch.nn as nn


def scaled_dot_product(q, k, v, scale):
    # standard dense head: every query sees every key
    scores = (q @ k.transpose(-1, -2)) * scale
    return torch.softmax(scores, dim=-1) @ v


class AttentionPattern(nn.Module):
    """The unresolved slot: which keys each query attends to,
    and how to evaluate only those inner products efficiently."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        # TODO: choose which keys each query attends to so cost is sub-quadratic.
        pass


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_size):
        super().__init__()
        self.attn = AttentionPattern(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size), nn.ReLU(),
            nn.Linear(ffn_size, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x
```
