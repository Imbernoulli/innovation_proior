## Research question

Serving a large language model cheaply means batching many requests, but in batched
autoregressive decoding the key-value (KV) cache — the attention keys and values cached
from every past token so they are not recomputed — becomes the dominant cost. Its size is
`b × (l_prompt + l_gen) × d` per layer (batch × sequence length × hidden size), so it grows
linearly with both batch and context. For a 540B model at batch 512 and context 2048 the KV
cache alone reaches several terabytes — multiple times the size of the model weights. Worse,
decoding is memory-bound: to generate each single token the GPU must stream the *entire* KV
cache from device memory (HBM) into on-chip SRAM, and while that load happens the compute
cores sit idle. So the KV cache is the new bottleneck in both memory and speed.

The precise goal is to shrink the number of bytes per cached KV element — the natural lever
is quantization, reducing each element from 16-bit float to a few-bit integer — while
keeping generation quality essentially intact, and doing it under three hard constraints.
First, **no training or calibration**: the method must be plug-and-play on an already-trained
model. Second, it must respect the **streaming nature** of the cache: keys and values for new
tokens arrive one at a time during decode and are appended, so any quantization scheme has to
work incrementally on a cache whose length keeps growing, which rules out expensive
optimization-based post-training quantization that would have to re-solve per step. Third, it
must be **hardware-friendly** — the quantize/dequantize overhead has to be small enough that
the memory savings translate into real throughput. Existing simple recipes hold quality at
4-bit but fall apart at the extreme low-bit (2-bit) end; closing the gap to genuinely low bit
width without retraining is the problem.

## Background

By this time stochastic LLM serving is built on a few well-understood pieces, and several
prior attacks on KV cache size already exist, none of which simply reduce the bytes per
element without side effects:

- **Reducing the number of KV heads** — multi-query attention (Shazeer 2019) and grouped-query
  attention (Ainslie et al. 2023) share one or a few KV heads across many query heads, cutting
  cache size, but they require training the model from scratch or fine-tuning it.
- **Evicting tokens** — heavy-hitter / importance-based eviction such as H2O (Zhang et al. 2023)
  and Scissorhands (Liu et al. 2023) keep only a small set of tokens that carry most of the
  attention mass, and StreamingLLM (Xiao et al. 2023) keeps only a few initial "attention sink"
  tokens plus a recent window. These drop tokens rather than compress them.
- **System-level** — offloading the cache to CPU/disk (FlexGen, Sheng et al. 2023) or paging it
  with virtual-memory techniques (vLLM / PagedAttention, Kwon et al. 2023) manage where the
  cache lives, orthogonal to how many bits each element takes.

The lever that reduces bytes-per-element directly is **quantization**. The standard tool is
**group-wise asymmetric round-to-nearest (RTN)**: pick `g` contiguous elements as a group, and
for that group set the zero-point `z = min(group)` and scale `s = (max(group) − min(group)) /
(2^B − 1)`; then `Q(x) = round((x − z) / s)` clamped to `[0, 2^B − 1]`, and dequantization is
`x' = Q(x)·s + z`. A smaller group means tighter per-group ranges and less error, at the cost
of storing more `(z, s)` metadata. FlexGen used exactly this — 4-bit, group size 64 — to
compress both weights and KV cache with negligible loss, dequantizing back to FP16 before the
matmul; like prior KV work it grouped both key and value **per-token** (groups of contiguous
elements within a single token's vector), because a freshly generated token forms a complete
per-token group that can be appended along the token dimension, which fits streaming naturally.
Optimization-based PTQ like GPTQ (Frantar et al. 2022), which solves a per-layer reconstruction
problem, is accurate for weights but too expensive to redo on a cache that grows every step.

Two empirical facts about LLM internals are load-bearing here. **(1) Activation outliers live
in a few fixed channels.** It is well documented (LLM.int8, Dettmers et al. 2022; SmoothQuant,
Xiao et al. 2022) that a small number of fixed feature channels in transformer activations
carry magnitudes on the order of 100× the rest, and that an outlier channel tends to be an
outlier for every token. Any quantization group that spans many channels can therefore have
its range dominated by a few large coordinates, which makes normal coordinates share a coarse
step size. **(2) Attention is highly sparse.** A query attends strongly to only a handful of
past tokens; the vast majority of attention weights are near zero (analyzed e.g. by Tian et
al. 2023, exploited by H2O). Measured on a 13B model, attention sparsity is around 84%.

A diagnostic study of plain group-wise RTN on the KV cache (Llama-2-13B, group size 32, CoQA
and TruthfulQA, fake-quant: quantize then dequantize inside the attention layer) sets up the
problem with three observations. First, the streaming-friendly per-token default is stable at
4-bit but degrades sharply when pushed to 2-bit. Second, changing the grouping axis is not a
minor implementation choice: some axis choices create catastrophic failures. Third, raw tensor
reconstruction error can be misleading because cached tensors affect the model only through
attention logits and weighted value mixtures. These facts do not yet prescribe a cache policy;
they say the grouping axis and the downstream attention use are load-bearing.

## Baselines

These are the prior approaches a new low-bit KV quantizer would be compared against and react
to.

**FlexGen group-wise KV quantization (Sheng et al., ICML 2023).** Apply 4-bit fine-grained
group-wise asymmetric RTN, group size 64, to both weights and KV cache, grouping the cache
along the hidden dimension within each token — i.e. **per-token** quantization of both key and
value. Dequantize to FP16 before each matmul; the goal is compression and reduced I/O, not
integer matmul. This is the natural streaming-friendly choice because each new token is a
complete per-token group appended along the token axis. **Limitation:** it is comfortable at
4-bit but is not designed for the extreme low-bit regime; pushed to 2-bit, uniform per-token
quantization of both caches loses substantial quality, and the scheme treats key and value
identically despite their very different element distributions.

**SmoothQuant (Xiao et al., ICML 2023).** A post-training quantization that migrates
quantization difficulty from activations to weights via an equivalent per-channel scaling
transform, so that activations (which carry the fixed-channel outliers) become easier to
quantize. It can take the KV cache to 8-bit with minor loss. **Limitation:** it faces a
significant accuracy drop when scaled to 4-bit or below, and the scaling-transform machinery
is aimed at general activation/weight quantization rather than the streaming KV cache.

**Weight-only quantization — GPTQ (Frantar et al. 2022), AWQ (Lin et al. 2023), SqueezeLLM
(Kim et al. 2023).** GPTQ uses approximate second-order (Hessian) information to quantize
weights to 3-4 bit accurately; AWQ scales weights in an activation-aware way to protect the
channels that matter; SqueezeLLM uses sensitivity-based non-uniform quantization with a
dense-and-sparse split. **Limitation (for this problem):** they quantize the *weights*, not the
running KV cache, so they leave the KV-cache bottleneck untouched; they are orthogonal and
combinable, not a solution to KV size. Their activation-aware results remain useful background
for why quantization error can concentrate around a small subset of channels.

**Token-eviction methods — H2O (Zhang et al. 2023), StreamingLLM (Xiao et al. 2023).** Keep
only the high-attention "heavy hitter" tokens (H2O) or a few initial sink tokens plus a recent
window (StreamingLLM), discarding the rest of the cache. **Limitation:** they reduce size by
*dropping* tokens, so information in evicted tokens is gone — a different and lossier tradeoff
than compressing every token's representation; again orthogonal to bit-width reduction.

## Evaluation settings

The natural yardsticks already in use for KV-cache compression:

- **Models** spanning attention variants: multi-head attention models (the Llama / Llama-2
  family, Mistral-7B) and a multi-query-attention model (Falcon-7B, which already has a single
  KV head and is thus already heavily compressed). Implemented on the Hugging Face Transformers
  codebase.
- **Normal-context generation tasks** from the LM-Eval harness: CoQA (exact-match accuracy),
  TruthfulQA (BLEU), and GSM8K (exact-match accuracy) as a hard multi-step-reasoning task. Closed
  multiple-choice tasks that take a single decode step and read logits directly are unsuitable,
  since they do not stress a compressed *generation* cache.
- **Long-context tasks** from LongBench, chosen across subgroups: single-document QA (Qasper),
  summarization (QMSum, MultiNews), few-shot learning (TREC, TriviaQA, SAMSum), and code
  completion (LCC, RepoBench-P), plus multi-hop QA (HotpotQA) and passage retrieval. Maximum
  sequence length 8192 for Mistral, 4096 for the others.
- **Needle-in-a-haystack (NIAH)** passkey/phrase retrieval inside long filler text (Paul Graham
  essays as background), following the passkey-retrieval template, to test long-context retrieval
  after quantizing the cache.
- **Efficiency**: ShareGPT-derived workloads (average prompt length ~161, output ~338), increasing
  batch size until out-of-memory on a single A100-80GB, measuring peak memory and throughput.
- **Quantization quality protocol**: "fake quantization" — quantize the KV tensors to low
  precision and immediately dequantize inside the attention layer — to isolate the quantization
  scheme from any kernel/packing details; group size held fixed across configurations for fair
  comparison; when a per-channel scheme leaves a token count not divisible by the group, pad so
  all tokens are still quantized.

## Code framework

The quantizer plugs into a fixed tensor-level decode-replay harness. The harness owns the model
and the deterministic greedy decoding loop; at each decode step after prefill it snapshots the
real KV tensors, hands them to a quantizer object to be (de)quantized, restores the result into
the cache, and advances. The quantizer is the only thing being designed. The KV tensors have
shape `[batch, heads, seq_len, head_dim]`. What already exists is the generic group-wise
asymmetric RTN primitive over a caller-chosen contiguous axis and a memory-accounting hook. The
empty slot is the cache compression policy.

```python
import math
import torch

FP_BITS = 16.0  # the unquantized reference precision per element


class AdaptiveKVQuantizer:
    """The slot the harness fills. The harness supplies real KV tensors of shape
    [batch, heads, seq_len, head_dim] and calls quantize_key / quantize_value at
    each decode step, plus estimate_bits for memory accounting. The (de)quantized
    tensor is restored into the cache and decoding continues."""

    def __init__(self):
        self.bits = 4
        self.group_size = 32          # elements per quantization group (given)

    def reset_request(self, request_meta: dict, budget_state: dict):
        self.bits = 4

    def _group_minmax_quantdequant(self, data: torch.Tensor, bits: int, group_size: int):
        """Generic group-wise asymmetric RTN over the LAST dim, returned dequantized
        (fake-quant): split the last axis into groups, per group z=min,
        s=(max-min)/(2^bits-1), q=round((x-z)/s) clamped, x'=q*s+z."""
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, int(2 ** bits) - 1)
        trailing = data.shape[-1]
        g = trailing if int(group_size) <= 0 else int(group_size)
        padded = math.ceil(trailing / g) * g
        work = torch.nn.functional.pad(data, (0, padded - trailing)) if padded != trailing else data
        grouped = work.reshape(*work.shape[:-1], padded // g, g)
        gmin = grouped.amin(dim=-1, keepdim=True)
        gmax = grouped.amax(dim=-1, keepdim=True)
        scale = (gmax - gmin).clamp(min=1e-5) / max_int
        q = torch.round((grouped - gmin) / scale).clamp(0, max_int)
        return q.mul(scale).add(gmin).reshape(*work.shape[:-1], padded)[..., :trailing]

    def quantize_key(self, layer_id, key_states, cache_meta):
        # TODO: implement the cache compression policy for this tensor.
        pass

    def quantize_value(self, layer_id, value_states, cache_meta):
        # TODO: implement the cache compression policy for this tensor.
        pass

    def estimate_bits(self, layer_id, kv_kind, seq_len, head_dim, cache_meta):
        # TODO: average effective bits per cached element under the policy above.
        pass
```

A concrete cache policy fills these stubs with its effective-bits accounting.
