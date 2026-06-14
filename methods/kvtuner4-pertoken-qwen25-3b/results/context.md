# Context: low-bit KV cache quantization for LLM decoding (circa 2024-2025)

## Research question

During autoregressive decoding, a transformer keeps a key/value cache so that each
new token attends to all previous tokens without recomputing their projections.
The cache size grows linearly with batch size and sequence length, so in long
context and large-batch serving the KV cache — not the model weights — becomes the
memory and bandwidth bottleneck: at every decode step the GPU must stream the whole
cache from main memory to SRAM while the compute units sit nearly idle. Quantizing
the cache to low bit-width is the most deployable way to shrink it, but the precise
problem is sharper than "compress the cache." Round-to-nearest INT8 KV quantization
is essentially lossless; INT4 is usually fine; but at INT2 — and, on some models,
already at INT4 *key* quantization — accuracy collapses. The goal is a KV-cache
quantizer that pushes the *effective* bit-width well below 4 while staying nearly
lossless on hard generation tasks (multi-step math, long-context retrieval, code),
and that does so in a way an existing inference stack can actually run: it must be
hardware-friendly (no per-token or per-head precision differences inside a layer
that break FlashAttention / vLLM paged caches), must add no online control-flow
overhead at decode time, and must adapt to the fact that different models — and
different layers within a model — tolerate quantization very differently. A method
that achieves the average bit-width budget on paper but corrupts a few critical
attention computations, or that needs an online critical-token search at every
step, does not solve the problem.

## Background

**The KV cache and where it hurts.** For input hidden state `x_i^l` at layer `l`,
the query/key/value projections give `q_i^l = W_q^l x_i^l`, `k_i^l = W_k^l x_i^l`,
`v_i^l = W_v^l x_i^l`. Attention at step `i` is
`a_i^l = softmax(q_i^l K^l^T / sqrt(D))`, output `o_i^l = a_i^l V^l`, where
`K^l, V^l` are all keys/values up to step `i`. These tensors are cached and reused.
For OPT-175B at batch 512, prompt 512, 32 generated tokens, the KV cache is ~1.2 TB
— 3.8x the weights — and the per-token decode latency is dominated by streaming it.

**Round-to-nearest is the only streaming-friendly quantizer.** Because KV tensors
arrive one token at a time, optimization-based post-training quantizers (GPTQ-style)
are too costly to run online; the practical choice is uniform round-to-nearest. For
`X in R^{S x D}`, group-wise asymmetric quantization-dequantization is
`Q(X) = round((X - z)/s)`, `X_hat = Q(X)·s + z`, with zero-point `z = min(X)` and
scale `s = (max(X) - min(X)) / (2^B - 1)`. The tensor can be grouped and quantized
along either the **token** dimension (each token row shares a scale/zero-point —
"per-token") or the **channel** dimension (each feature column shares them —
"per-channel"). Per-token is natural for streaming because a freshly generated
token's row can be quantized and appended without touching earlier rows; per-channel
needs a full block of tokens before its column statistics are known.

**Key cache carries channel outliers; value cache does not.** Visualizing KV
magnitudes (Liu et al. 2024, KIVI; Hooper et al. 2024, KVQuant) shows that a few
fixed *channels* of the key cache have persistently large magnitude — the same
activation-outlier-channel phenomenon documented for LLM activations generally
(Dettmers et al. 2022, LLM.int8(); Lin et al. 2024, AWQ; Xiao et al. 2023,
SmoothQuant) — whereas the value cache has no clear outlier structure. Because a
per-channel scale shares its dynamic range only within a column, per-channel key
quantization confines an outlier's wide range to its own channel and leaves the
other channels at fine resolution; per-token key quantization lets one outlier
channel inflate the scale of an entire token row. Measured on Llama-2-13B, per-token
key quantization gives roughly 5x the relative attention-score error of per-channel.

**Value cache wants per-token, for a different reason.** The attention output is a
sparse weighted sum of value rows, `[A X_V]_{i*} = sum_j A_{ij} [X_V]_{j*}`, and
attention is highly sparse (a handful of tokens carry almost all the weight, ~84%
sparsity on Llama-2-13B). Per-token value quantization confines each token's error
to that token, so corrupting unimportant tokens barely moves the output; per-channel
value quantization spreads error across all tokens and inflates the output error
~15x. So the established wisdom is: key per-channel, value per-token.

**Two-dimensional error accumulation.** A transformer is sequential in two
directions at once: a quantization error in layer `l` feeds the input of layer
`l+1`, and an error in the token generated at step `i` feeds every layer at step
`i+1`. So the per-token, per-layer error `e_i^l` depends on all earlier layers
`1..l-1` at step `i` and all layers at all earlier steps `1..i-1`. A single
token's, single layer's quantization error is negligible; accumulated over a deep
model and a long generation it is not, and it produces *token flipping*. A
documented example on Llama-2-13B-chat with 15-shot GSM8K: full-precision and INT4
KV produce the identical correct derivation `20 - 4 - 4 = 12 -> 60%`, but INT2 KV
flips a single sign so the model writes `20 + 4 + 4 = 28 -> 14%` and the final
answer is wrong. Minor errors that would be invisible in perplexity become
catastrophic in multi-step reasoning.

**Quantization error correlates with attention pattern.** Measuring relative
attention-output error `e_o` per layer under per-token-asym quantization on
Llama-3.1-8B (GSM8K): dropping key precision 8->4 and 4->2 multiplies the average
attention-score error by about 13.9x and 4.6x. The heads that blow up are the
*non-sparse retrieval heads*; the sparse "streaming" heads that concentrate on a
few tokens stay robust. Equal-memory comparisons make the key/value asymmetry
explicit: with per-token-asym on Llama-3.1-8B, K4V2 attention-output error
(`e_o = 0.453`) is far below K2V4 (`e_o = 0.892`) — spending the bits on keys beats
spending them on values at the same budget. Word-perplexity across many models tells
the same story: K8V4 ≈ KV8 and K4V2 ≈ KV4, while K4V8 and K2V4 degrade sharply — a
decline appears when key precision drops, not when value precision drops.

**Sensitivity is an inherent, prompt-independent model property.** The layer-wise
error distribution is stable across input prompts: a layer that is sensitive on one
prompt is sensitive on others, and the same families (e.g. Qwen2.5-7B,
Mistral-7B-v0.3) show the same layer profiles. There is no clean depth heuristic —
sensitive and insensitive layers are interleaved with no monotone-in-depth pattern,
and the most sensitive layer even changes when the quantization mode changes
(per-token-asym vs per-channel-asym). Models differ sharply too: most tolerate INT2
only with damage, but Qwen2.5-7B and Qwen2.5-Math-7B are sensitive already at INT4
key quantization.

## Baselines

**Uniform low-bit round-to-nearest (per-token-asym / per-channel-asym).** Quantize
every layer's KV to the same target bit-width with one quantization mode. INT8 is
near-lossless; INT4 per-token is usually fine; INT2 collapses, and on sensitive
models INT4 key already collapses. Gap: a single global bit-width ignores that
layers differ by more than an order of magnitude in sensitivity and that key and
value differ in importance, so a uniform budget either overpays (8-bit everywhere)
or breaks the few layers that cannot take the cut.

**KIVI (Liu et al., ICML 2024).** Tuning-free 2-bit KV quantization that quantizes
the **key cache per-channel** and the **value cache per-token** with group size
`G = 32`, motivated by the outlier analysis above. To make per-channel quantization
work in a streaming cache it splits each tensor into a *grouped* part (every `G`
tokens quantized) and a *residual* part (the most recent `R <= 128` tokens kept in
fp16); attention is computed as `A = concat(A_grouped from Q(K_g), A_residual from
fp K_r)`, and a recent token is folded into the quantized block only once the
residual queue fills. The fp16 sliding window is shown to be crucial on hard tasks
like GSM8K. Gap: KIVI is **uniform precision across all layers** and assumes the
static prefix/recent blocks are the important ones — an assumption that fails on the
non-sparse retrieval heads of sensitive models (where low-precision key quantization
still shifts the attention distribution), and per-channel key quantization needs
specially designed GPU operators and careful cache management.

**KVQuant (Hooper et al., NeurIPS 2024).** Per-channel key quantization applied
*before* RoPE (since the outlier channels are more consistent pre-RoPE), a
sensitivity-weighted non-uniform datatype (nuqX) derived offline on a calibration
set, and per-vector dense-and-sparse isolation of ~1% numerical outliers stored in a
separate sparse representation, reaching ~3-bit with <0.1 perplexity loss via custom
CUDA kernels. Gap: it targets a single shared precision per tensor rather than
allocating different bit-widths to different layers by sensitivity, and the
pre-RoPE/non-uniform/sparse-outlier machinery needs bespoke kernels that are hard to
fuse with FlashAttention and vLLM paged caches.

**Online fine-grained mixed precision (QAQ, Dong et al. 2024; MiKV, Yang et al.
2024; ZipCache, He et al. 2024).** Dynamically identify critical KV entries at decode
time and keep them at higher precision while quantizing the rest aggressively,
improving accuracy at a given average budget. Gap: intra-layer per-token (or
per-page) precision differences cannot be fused with FlashAttention or vLLM paged
attention, and the online critical-token identification adds control-flow overhead
that does not fit static-graph acceleration (torch.compile); the extra per-step
logic erodes the throughput the quantization was meant to buy.

## Evaluation settings

These are the yardsticks already in use, with no method-specific outcomes.

- **Models.** Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, and the Qwen2.5
  series — Qwen2.5-{3B, 7B, 14B, 32B, Math-7B}-Instruct and the AWQ-quantized
  Qwen2.5-3B — chosen to span scales from on-device (3B) to large (32B) and to
  include a math-specialized model. The on-device target is `Qwen2.5-3B-Instruct`
  (36 transformer layers).
- **Quantization modes compared.** Per-token-asym (key and value both quantized
  along the token dimension) and the KIVI-style mode (key per-channel with group
  32 and residual 32, value per-token). Both use asymmetric round-to-nearest.
- **Tasks.** General: CEVAL, MMLU, TriviaQA, RACE, TruthfulQA. Math/science/logic:
  GSM8K at {0, 4, 8, 16}-shot, multi-round GSM8K, GPQA. Plus wikitext word
  perplexity via lm-evaluation-harness. The natural long-context yardsticks are
  LongBench-style QA / retrieval / code-completion, needle-in-a-haystack retrieval,
  and GSM8K final-answer accuracy.
- **Search machinery (for the offline component).** A black-box multi-objective
  optimizer — a decomposition-based or NSGA-II-style evolutionary search driven
  through Optuna — with DBSCAN for clustering. The accuracy objective is computed
  on the first 200 GSM8K 4-shot prompts.
- **Efficiency metric.** Effective KV bits per cached element, taken at a fixed
  reference span (e.g. 4096 tokens) so it is hardware-independent, and the
  compression ratio `16 / effective_kv_bits` relative to an FP16 (16-bit) cache.
- **Calibration design.** A calibration setup that lets quantization error
  *accumulate* across layers during prefill (using dequantized KV for prefill
  self-attention) and that uses long, hard generations (math reasoning) so that
  small errors flip intermediate tokens and thereby separate good from bad
  precision assignments. The first 20 GSM8K zero-shot prompts are a common probe
  set for measuring per-layer attention-output error.

## Code framework

The quantizer plugs into a fixed tensor-level replay harness over Hugging Face
`Transformers`. The harness owns data loading, deterministic greedy decoding, the
`DynamicCache` snapshot/restore, scoring, and result emission. At each decode step
after prefill it hands the editable quantizer the real per-layer key/value tensors
of shape `[batch, heads, seq_len, head_dim]`, replaces the cache with the quantized
version, and advances generation. What the quantizer actually does to those tensors
— the integer grid, the grouping axis, any per-layer bit choices, any residual
window, the memory accounting — is exactly the open slot. The base integer
quantization primitive (round-to-nearest with a scale and zero-point over a group)
already exists; nothing about *which bits go where, along which axis, in which
layers* is settled.

```python
import math
import torch

FP_BITS = float(torch.finfo(torch.float16).bits)  # 16: the uncompressed reference


class AdaptiveKVQuantizer:
    """Editable KV-cache quantizer. The fixed harness supplies real key/value
    tensors [batch, heads, seq_len, head_dim] from a DynamicCache and calls these
    hooks; the returned tensor must keep the input shape. The actual quantization
    algorithm — bit allocation, grouping axis, residual policy, accounting —
    lives here."""

    def reset_request(self, request_meta: dict, budget_state: dict):
        # per-example reset hook; nothing stateful needed by a static policy
        return None

    def needs_prefill_qkv_observer(self) -> bool:
        return False

    def observe_prefill_qkv(self, layer_id, query_states, key_states,
                            value_states, attention_meta):
        return None

    def query_observation_position(self) -> str:
        return "post_rope"

    def _quantize_group(self, tensor, bits, axis, group_size, residual_length):
        # Generic round-to-nearest group quantize/dequantize over the chosen axis.
        # The integer grid, scale, zero-point, grouping and residual handling
        # are the open slot:
        # TODO: the quantization rule we will design.
        pass

    def quantize_key(self, layer_id, key_states, cache_meta):
        # TODO: how (and at what precision, along which axis, with what grouping
        #       and residual) the key cache of THIS layer is quantized.
        pass

    def quantize_value(self, layer_id, value_states, cache_meta):
        # TODO: same decision for the value cache of THIS layer.
        pass

    def estimate_bits(self, layer_id, kv_kind, seq_len, head_dim, cache_meta) -> float:
        # TODO: the effective bits-per-element this policy spends at (layer, kind).
        pass
```

The harness fills `quantize_key` / `quantize_value` with the real tensors and reads
back `estimate_bits` for the efficiency term; `_quantize_group` is the shared
arithmetic the two quantize hooks call. The bit choices, the axis, the per-layer
policy, and the accounting are what the method will define.
