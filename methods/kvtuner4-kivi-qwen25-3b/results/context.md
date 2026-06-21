# Context: low-bit KV cache quantization for LLM decoding

## Research question

During autoregressive decoding, a transformer keeps a key/value cache so each new
token can attend to all previous tokens without recomputing their projections. The
cache size grows linearly with batch size and sequence length, so long-context and
large-batch serving can become memory-bandwidth bound: each decode step streams a
large cache from GPU memory while the compute units do little work. Quantizing the
cache reduces this memory footprint. Round-to-nearest INT8 KV quantization is usually
near lossless, INT4 is often acceptable, and INT2 frequently corrupts generation;
some model families are already sensitive when the key cache drops to INT4.

The research question is how to design a KV-cache quantizer that pushes effective
bit width below a uniform 8-bit cache while preserving generation quality on hard
tasks such as multi-step math and long-context retrieval, for the 36-layer
Qwen2.5-3B-Instruct model.

## Background

For hidden state `x_i^l` at layer `l`, the attention projections produce
`q_i^l = W_q^l x_i^l`, `k_i^l = W_k^l x_i^l`, and `v_i^l = W_v^l x_i^l`. The decode
attention distribution and output are

`a_i^l = softmax(q_i^l (K^l)^T / sqrt(D))`, `o_i^l = a_i^l V^l`,

where `K^l` and `V^l` contain all keys and values through the current step. Those
tensors are cached and reused. For OPT-175B at batch size 512, prompt length 512,
and 32 generated tokens, the KV cache is about 1.2 TB, roughly 3.8 times the model
weights, and per-token latency is dominated by loading it.

Because the cache is updated online, optimization-heavy post-training quantizers are
not a natural fit. The practical primitive is group-wise round-to-nearest
quantization. For `X in R^{S x D}`,

`Q(X) = round((X - z) / s)`, `X_hat = Q(X) * s + z`,

with `z = min(X)` and `s = (max(X) - min(X)) / (2^B - 1)` for the chosen group.
Groups can run along the token dimension, where each token row gets its own scale
and zero-point, or along the channel dimension, where each feature column gets them.
Per-token grouping is simple for a streaming cache because a new token row can be
quantized and appended immediately. Per-channel grouping needs a block of tokens
before the column statistics are known, so it requires extra cache management.

The key and value caches fail differently under low-bit quantization. The key cache
has persistent high-magnitude channels, the same broad activation-outlier pattern
reported in low-bit LLM work. A per-token key scale lets one large channel inflate
the scale for an entire token row, reducing resolution for the other channels. A
per-channel key scale confines that wide range to its own column; KIVI reports much
lower attention-score error for key per-channel than key per-token on Llama-2-13B.

Values have a different structure. The attention output is a weighted sum over value
rows, `[A X_V]_{i*} = sum_j A_{ij} [X_V]_{j*}`, and attention is usually sparse: a
small number of tokens carry most of the weight. Per-token value quantization keeps
each token's error local, so errors on near-zero-weight tokens barely move the
output. Per-channel value quantization spreads error across all tokens and can make
the output error far larger; KIVI reports about 15x worse output error for that
choice.

Low-bit errors also accumulate in two directions. A layer's quantization error feeds
the next layer at the same decode step, and a token generated with error feeds all
layers at later steps. A small local error can therefore cross an argmax boundary
later in the generation. The canonical example is a GSM8K chain in which full
precision and a 4-bit cache keep `20 - 4 - 4 = 12`, while a 2-bit cache flips the
sign and continues from `20 + 4 + 4 = 28`, producing the wrong final answer.

Layer sensitivity is not well described by depth alone. Measurements on Llama,
Mistral, and Qwen models show interleaved sensitive and insensitive layers, and the
identity of the most sensitive layers can change when the quantization mode changes
from per-token to per-channel. The same model family can also be much more sensitive
than another one: Qwen2.5-7B and Qwen2.5-Math-7B show severe degradation when key
precision drops to INT4 in settings where other models mainly fail at INT2.

## Baselines

**Uniform round-to-nearest.** Quantize every layer's keys and values to the same
precision with the same grouping mode. It is simple and fast.

**KIVI.** Quantizes keys per-channel and values per-token, with group size 32 and a
recent fp16 residual window so per-channel key statistics can be formed in a
streaming setting. The original experiments use residual length 128 in many settings
and also report residual length 32 as a faster comparable setting.

**KVQuant.** Quantizes keys before RoPE so channel outliers remain consistent, uses
an offline sensitivity-weighted non-uniform datatype, and stores a small fraction of
numerical outliers in a sparse full-precision representation. It reaches very low
effective precision with custom CUDA kernels.

**Online fine-grained mixed precision.** Methods such as QAQ, MiKV, and ZipCache
identify important tokens, pages, or entries at decode time and keep them at higher
precision, adjusting the effective bit budget per step.

## Evaluation settings

The relevant measurements are final task accuracy, attention-output error, and
effective KV bits per cached element. The per-element errors, computed by a simulated
quantize-then-dequantize pass, are the relative key error
`e_k = mean(|K - K_hat| / |K|)`, the relative value error `e_v`, the absolute
attention-score error `e_a = mean(|a - a_hat|)`, and the relative attention-output
error `e_o = mean(|o - o_hat| / |o|)`, with
`a_hat = softmax(q K_hat^T / sqrt(D))` and `o_hat = a_hat V_hat`. Useful model
families include Llama-3.1-8B, Mistral-7B-Instruct-v0.3, and the Qwen2.5 series,
including the 36-layer Qwen2.5-3B-Instruct target. Useful task families include GSM8K
with few-shot chain-of-thought prompting, multi-turn GSM8K, GPQA, general
QA/classification suites such as MMLU/CEVAL/TriviaQA/RACE/TruthfulQA, wikitext
word-perplexity, and long-context retrieval/code tasks. A tensor-level replay harness
evaluates a quantizer by greedy decode: at each step after prefill it snapshots the
real KV tensors, runs them through the quantizer, restores the quantized cache, and
advances; it reports a quality score plus an `effective_kv_bits` at a 4096-token
reference span and `kv_compression_ratio = 16 / effective_kv_bits` against an FP16 KV
reference, so efficiency is hardware-independent.

## Code framework

The editable implementation slot is a tensor-level cache quantizer inside a fixed
Hugging Face style generation harness. The harness owns tokenization, generation,
cache snapshot/update, scoring, and result emission. At each cache update it supplies
the real per-layer key/value tensors of shape `[batch, heads, seq_len, head_dim]`.
The open decisions are the bit width for each layer's key and value tensors, the
grouping axis, the group size, whether any residual fp16 window is retained, and the
memory accounting.

```python
class EditableKVQuantizer:
    def reset_request(self, request_meta: dict, budget_state: dict):
        return None

    def quantize_key(self, layer_id, key_states, cache_meta):
        # choose key precision, grouping axis, group size, and residual policy
        raise NotImplementedError

    def quantize_value(self, layer_id, value_states, cache_meta):
        # choose value precision, grouping axis, group size, and residual policy
        raise NotImplementedError

    def estimate_bits(self, layer_id, kv_kind, seq_len, head_dim, cache_meta) -> float:
        # report effective bits per cached element for the chosen policy
        raise NotImplementedError
```
