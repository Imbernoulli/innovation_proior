# Context: inference-time sparse attention for long-context LLMs

## Research question

A pretrained causal language model spends most of its long-context latency inside attention.
For a query token `q_t`, full softmax attention scores `q_t` against every preceding key
`k_1..k_t` and forms a weighted sum of the values — `O(t)` work per query, `O(t^2)` over the
sequence. Theoretical breakdowns put attention at roughly 70-80% of total latency when
decoding 64k-length contexts, so the quadratic term dominates exactly where long context is
most useful (repository-level code, long documents, multi-turn agents, long chain-of-thought).

The goal is a sparse attention module that **drops into a frozen, already-trained LLM at
inference time** and preserves long-context task quality under a **fixed sparsity budget** — a
hard cap on the fraction of query-key pairs that may receive nonzero attention — with no
retraining, no fine-tuning, and no architectural surgery beyond replacing the attention
forward. The setting is parallel-forward: every forward processes the full prefix in one shot
and the same module is replayed at every generation step, so the importance signal must be
computable from the queries and keys present in the current forward, not from a mutated cache.

What makes this hard is a tension. Softmax attention is empirically sparse — for a given
query only a small fraction of keys carry almost all the mass — so in principle one could
attend to just those keys and pay far less. But *which* keys matter is query-dependent and
changes from prompt to prompt, so the important set cannot be fixed ahead of time; it has to
be chosen from the current query. And choosing it by scoring every individual `(q, k)` pair is
itself `O(t^2)` — the same cost the sparsity was supposed to remove. A solution has to find the
right keys for each query *without* first computing the dense score it is trying to avoid, and
it has to do so in a way that maps onto how GPUs actually move and multiply data.

## Background

**Attention and its cost.** For an input of length `t`, attention is
`o_t = sum_i alpha_{t,i} v_i / sum_j alpha_{t,j}` with `alpha_{t,i} = exp(q_t . k_i / sqrt(d_k))`.
During prefill and training the batched matmuls are compute-bound; during single-token decode
the work is small but the whole KV cache must be loaded, so decode is memory-bandwidth bound.
A method that reduces work is only useful if the reduction survives the memory system — the
gap between *theoretical* FLOP savings and *realized* latency savings is the central practical
trap in this area.

**Arithmetic intensity and why granularity matters.** Each GPU has a critical
compute-to-memory-access ratio; an operation above it is compute-bound, below it
memory-bound. Modern accelerators deliver far higher throughput for **contiguous block
accesses** than for random index-based reads, and Tensor Cores want dense block-shaped
operands. This is why high-performance attention is built blockwise (FlashAttention tiles K/V
into contiguous blocks loaded once into SRAM). Any sparsity scheme that selects scattered
individual tokens forces non-contiguous gathers from the KV cache, falls back to low hardware
utilization, and gives up much of the speedup it claimed on paper.

**Attention is sparse, dynamic, and spatially clustered.** Three empirical facts about
existing full-attention models, all knowable from a frozen model, set up the design space.
(1) *Sparse:* in a 128k-context attention matrix, the top 4k columns recall ~96.8% of the
total attention mass — each token effectively attends to a small subset (Jiang et al. 2024,
MInference). (2) *Dynamic:* the sparse set is highly context-dependent — reusing the top-4k
columns found on one 128k prompt on a different prompt drops recall to ~83.7% — so a single
fixed mask cannot capture it; the choice must depend on the current query. (3) *Spatially
clustered:* the nonzero attention entries are not scattered uniformly — the distance between a
nonzero entry and its nearest nonzero neighbor concentrates around 5, i.e. the important
entries arrive in contiguous two-dimensional **blocks** ("block-sparse" pattern). Neighboring
keys tend to share similar importance, so importance is approximately a block-level, not a
token-level, property. These three facts pull in a consistent direction: pick keys per query
(because dynamic), but at block granularity (because clustered and because contiguous reads are
fast).

**Cheap block summaries.** A recurring idea in long-context work is to summarize a contiguous
run of keys by a single representative so that relevance can be judged per block instead of per
token. Representatives have been built as a learned low-dimensional projection, as the few
highest-"representative-score" tokens in the block, or — a parameter-free option that needs no
training and so applies directly to a frozen model — as a simple average of the block's key
vectors, which is competitive with the learned and top-token variants (Xiao et al. 2024,
InfLLM, including its mean-representative ablation).

**Grouped-query attention.** Modern LLMs share each KV head across a group of query heads
(GQA; Ainslie et al. 2023) to cut the KV-cache memory traffic that bottlenecks decode. This
changes the accounting for any per-head sparse selection: the KV that must be loaded for a GQA
group is the *union* of the blocks selected by all query heads in that group, so a method that
lets every head pick independently can reduce arithmetic while leaving memory traffic almost
unchanged.

## Baselines

**Sliding window.** Each query attends only to the most recent `w` keys. Cheap, contiguous,
and content-blind. **Gap:** anything older than `w` is invisible, so it cannot answer a query
whose evidence sits far back in the context.

**Attention sink + window (StreamingLLM; Xiao et al., ICLR 2024).** Empirically, a few initial
tokens absorb a large, persistent share of attention mass (softmax must put weight somewhere
even when no past key is relevant, and the first tokens become that "sink"). Keeping the KV of
the first ~4 sink tokens plus a recent window recovers most of the quality of windowed
attention and lets the model stream past its training length. **Gap:** the pattern is static —
sinks plus a recent window — so it still cannot route attention to a relevant fact lying in the
*middle* of a long context; its mask does not depend on what the query is asking for.

**Global + window + random blocks (BigBird; Zaheer et al., NeurIPS 2020).** Combine a few
global tokens that attend to and are attended by everyone, a local window, and a handful of
random blocks per query; the random links make the attention graph an expander, giving
theoretical full-rank/connectivity guarantees. **Gap:** still a *fixed*, content-independent
pattern; the random blocks are not chosen for the query, so on a specific retrieval the
relevant block is hit only by chance unless the budget is large.

**Query-aware blockwise selection (Quest; Tang et al., 2024).** Split the KV cache into pages;
for each page keep the channelwise min and max of its keys; given `q`, upper-bound the page's
best possible logit by `sum_i max(q_i * min_i, q_i * max_i)` (this is `>=` `q . k` for every key
`k` in the page regardless of the sign of each `q_i`); select the top-`K` pages by this bound
and run exact attention on them. Content-adaptive and contiguous. **Gap:** each attention head
selects its pages independently, so under GQA the KV that must be loaded for a group is the
union over the group's heads — memory traffic does not fall in step with the arithmetic.

**Block memory with learned/representative units (InfLLM; Xiao et al., 2024).** Organize past
KV into fixed blocks; summarize each block by representative tokens (top-`r_k` by representative
score) or by a mean of its keys; score a block's relevance to the current tokens by a dot
product against the representative; load the top-`k_m` blocks plus sink and local window.
Parameter-free and training-free. **Gap:** it is positioned as a memory-lookup mechanism for
extending context, and uses a few discrete representative tokens whose selection is a
non-differentiable heuristic — the relevance signal is an approximation whose recall depends on
how well a handful of representatives stand in for the whole block.

The shared shape across these is: the cheap static patterns are content-blind, and the
content-adaptive ones either price their selection per head (so GQA memory does not drop) or
lean on token-granular / heuristic representatives. None of them, under a frozen GQA model in
parallel-forward, both chooses blocks from the current query *and* keeps the chosen set
consistent enough to be a contiguous, hardware-friendly read.

## Evaluation settings

- **Backbone:** an instruction-tuned 1.5B causal LLM (Qwen2.5-1.5B-Instruct; 12 query heads,
  2 KV heads — GQA). The module is monkey-patched into every attention layer; the harness
  replicates GQA so the module sees 12 heads on both Q and K/V.
- **Inputs:** `q, k, v` of shape `(B, H, N, D)` in fp16/bf16, `is_causal=True`. Output in the
  same shape and dtype.
- **Density accounting:** after each forward the module reports `last_density` = fraction of
  `(q, k)` pairs receiving nonzero attention, with the causal denominator `N(N+1)/2`. The
  harness averages `last_density` across all attention layers and aborts the run if the mean
  exceeds the budget plus a small slack; a missing, NaN, infinite, negative, or `>1` report is
  a harness error.
- **Budget:** `density_budget = 0.25`; only a reference dense baseline (reporting density 1.0)
  may exceed it.
- **Tasks (the natural yardsticks that exist already):** synthetic Needle-In-A-Haystack
  retrieval at 8K context (retrieval accuracy); LongBench Qasper (single-doc scientific-paper
  QA, F1); LongBench MultiFieldQA-EN (long-document multi-field QA, F1). These need
  instruction-following, hence the instruction-tuned backbone.
- **Hardware/constraints:** single A100 80GB, FP16 only, no Triton kernels — pure PyTorch ops
  (or `torch.nn.attention.flex_attention` if available). Branching on `is_causal` and on `N`
  is allowed.

## Code framework

The substrate is a single attention module that the harness loads in place of the dense
forward, one instance per attention layer. The constructor records the head geometry and the
density budget; the forward receives `q, k, v` in `(B, H, N, D)`, an `is_causal` flag, and an
optional softmax `scale`, and must return the attention output in the same shape and dtype and
set `self.last_density`. Everything about *which* keys each query is allowed to see — and how
that decision is made cheaply enough to be worth it — is the empty slot.

```python
import math
import torch
import torch.nn as nn


class SparseAttention(nn.Module):
    """Drop-in replacement for the dense attention forward, one per layer.
    Must return the attention output in the same (B, H, N, D) shape/dtype and
    report self.last_density = fraction of (q, k) pairs given nonzero attention
    (causal denominator N(N+1)/2 when is_causal)."""

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.density_budget = density_budget
        self.last_density = None

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        scale = scale if scale is not None else 1.0 / math.sqrt(D)

        # TODO: decide, under the density budget, which (query, key) pairs each
        #       query is allowed to attend to, cheaply enough to be worthwhile;
        #       build the corresponding keep-mask, report self.last_density, and
        #       return the masked-softmax attention output.
        keep = None   # the keep decision we will design
        pass
```
