## Research question

Decoder-only LLM inference in the long-context large-batch regime is bottlenecked by the key/value cache. Every past token leaves a key vector and a value vector in every layer, the cache grows linearly in batch size and sequence length, and each decode step streams the whole cache from HBM while the compute cores idle. Quantization of the running K/V tensors is the one lever that reduces bytes-per-element without dropping tokens or retraining. The design object is the **quantization policy** — bit width, grouping axis, asymmetric ranges and zero-points, residual window, and any prefill-time observation — that preserves benchmark output quality while shrinking the effective KV footprint. Everything else (the model, the decode replay loop, the workloads, the scoring) is fixed.

## Prior art / Background / Baselines

Most quantization methods assume a static tensor; the KV cache is a streaming object that grows one token at a time during decode. That constraint rules out much of the PTQ literature and defines the baseline family.

- **Optimization-based PTQ (GPTQ, OBQ).** Solves a per-layer reconstruction problem with second-order error feedback. Gap: re-solving an optimization over the whole cache every decode step is too expensive; only cheap, local primitives are viable.
- **Group-wise round-to-nearest (FlexGen).** Per group takes min/max, sets zero-point and scale, and rounds each element into `[0, 2^B − 1]`, running on a single token without calibration. Gap: uniform low-bit RTN is adequate at 4-bit but collapses at 2-bit, and it spends the same number of bits everywhere regardless of the tensor region.
- **Outlier-aware quantization (LLM.int8(), SmoothQuant).** Notes that transformer activations carry persistent large-magnitude channels that propagate into the key cache. Gap: the diagnosis points to sensitive axes, but it does not yet yield a streaming cache quantization policy.

These baselines are quantization policies on a common tensor-level interface; the contribution is the policy, not a backend or repository.

## Fixed substrate / Code framework

A deterministic greedy-decode replay harness over Hugging Face `Transformers` is frozen and must not be touched. It loads `Qwen/Qwen2.5-3B-Instruct` (36 layers), runs the benchmark prompts, and after each prefill step snapshots the real KV tensors, hands them to the editable quantizer, restores the quantized cache, and advances generation. Prefill is lossless; only the cached copy is quantized. The harness owns dataset loading (LongBench-E, NeedleBench/NIAH, GSM8K), prompt templates, generation limits, the parser, the score definitions, and a 4096-token reference-span efficiency accountant (`estimate_policy_efficiency`). It also exposes a query-observer hook that captures post-RoPE query states per layer for policies that want prefill statistics.

## Editable interface

Only the `AdaptiveKVQuantizer` class in `custom_quant_eval.py` (lines 41–172) is editable. The harness calls, per example: `reset_request(request_meta, budget_state)`; then `needs_prefill_qkv_observer()` and, if true, `observe_prefill_qkv(layer_id, query_states, key_states, value_states, attention_meta)` with `query_observation_position()` selecting pre/post-RoPE; then per decode step and layer, `quantize_key(layer_id, key_states, cache_meta)` and `quantize_value(...)`; and for the hardware-independent footprint, `estimate_bits(layer_id, kv_kind, seq_len, head_dim, cache_meta)`. `key_states` and `value_states` have shape `[batch, heads, seq_len, head_dim]`, and a `quantize_*` return must be a tensor of the identical shape (fake-quant: quantize-then-dequantize back to the model dtype). There is no algorithm enum or backend selector — grouping, asymmetric ranges, zero-points, per-layer presets, residual retention, and memory accounting all live inside this class.

The starting point is the scaffold default: a single global 4-bit policy, key per-channel, value per-token, group size 32, and a tail residual window. Each baseline replaces this class wholesale.

```python
# EDITABLE region of custom_quant_eval.py (lines 41-172) -- default fill
class AdaptiveKVQuantizer:
    """Editable KV-cache quantizer. The fixed harness supplies real key/value
    tensors and calls this class for the actual tensor algorithm."""

    def __init__(self):
        self.bits = 4
        self.key_group_size = 32
        self.value_group_size = 32
        self.key_residual_length = 128
        self.value_residual_length = 128

    def reset_request(self, request_meta: dict, budget_state: dict):
        self.bits = min(4, int(budget_state.get("budget_bits", 4)))
        workload = str(request_meta.get("workload", ""))
        residual = 128 if workload.startswith("longbench_") else 32
        self.key_residual_length = residual
        self.value_residual_length = residual

    def needs_prefill_qkv_observer(self) -> bool:
        return False                                   # no prefill statistics by default

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        return None

    def query_observation_position(self) -> str:
        return "post_rope"

    def _residual_keep_length(self, seq_len, residual_length, residual_policy="tail"):
        residual_length = max(0, min(seq_len, int(residual_length)))
        if residual_length == 0 or residual_policy in {"none", ""}:
            return 0
        if residual_policy == "block_modulo":          # sawtooth: leftover that doesn't fill a block
            return seq_len % residual_length
        if residual_policy == "tail":                  # fixed sliding window of recent rows
            return residual_length
        raise ValueError(f"Unsupported residual_policy={residual_policy}")

    def _minmax_quantize_last_dim(self, data, bits, group_size):
        # group-wise asymmetric round-to-nearest over the LAST dim, returned dequantized
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, int(2 ** int(bits)) - 1)
        trailing = data.shape[-1]
        group_size = trailing if int(group_size) <= 0 else int(group_size)
        padded = math.ceil(trailing / group_size) * group_size
        work = data
        if padded != trailing:
            work = torch.nn.functional.pad(work, (0, padded - trailing))
        grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
        gmin = grouped.amin(dim=-1, keepdim=True)
        gmax = grouped.amax(dim=-1, keepdim=True)
        scale = (gmax - gmin).clamp(min=1e-5) / max_int
        quant = torch.round((grouped - gmin) / scale).clamp(0, max_int)
        dequant = quant.mul(scale).add(gmin)
        return dequant.reshape(*work.shape[:-1], padded)[..., :trailing]

    def _quantize_grouped_minmax(self, layer_tensor, *, axis, bits, group_size,
                                 residual_length, residual_policy="tail"):
        work = layer_tensor.float().clone()
        batch, heads, seq_len, head_dim = work.shape
        residual = self._residual_keep_length(seq_len, residual_length, residual_policy)
        quant_end = seq_len - residual
        if quant_end <= 0 or bits >= FP_BITS - 0.5:
            return work.to(layer_tensor.dtype), FP_BITS
        quant_slice = work[:, :, :quant_end, :]
        if axis == "channel":                          # per-channel: group G TOKENS of a channel
            quant_len = quant_slice.shape[-2]
            group_size = quant_len if int(group_size) <= 0 else int(group_size)
            usable = quant_len - (quant_len % group_size)
            main = quant_slice[:, :, :usable, :]
            tail = quant_slice[:, :, usable:, :]
            if usable > 0:
                main = main.transpose(2, 3).reshape(batch, heads, head_dim, usable // group_size, group_size)
                main = self._minmax_quantize_last_dim(main, bits, group_size)
                work[:, :, :usable, :] = main.reshape(batch, heads, head_dim, usable).transpose(2, 3)
            if tail.numel() > 0:
                work[:, :, usable:quant_end, :] = tail
            fp_tokens = residual + (quant_len - usable)
            avg_bits = (usable * bits + fp_tokens * FP_BITS) / max(seq_len, 1)
        else:                                          # per-token: group flattened features per token
            flat = quant_slice.transpose(1, 2).reshape(batch, quant_slice.shape[-2], heads * head_dim)
            flat = self._minmax_quantize_last_dim(flat, bits, group_size)
            work[:, :, :quant_end, :] = flat.reshape(batch, quant_slice.shape[-2], heads, head_dim).transpose(1, 2)
            avg_bits = (quant_end * bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(layer_tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id, key_states, cache_meta):
        return self._quantize_grouped_minmax(
            key_states, axis="channel", bits=self.bits,
            group_size=self.key_group_size, residual_length=self.key_residual_length,
            residual_policy="tail")

    def quantize_value(self, layer_id, value_states, cache_meta):
        return self._quantize_grouped_minmax(
            value_states, axis="token", bits=self.bits,
            group_size=self.value_group_size, residual_length=self.value_residual_length,
            residual_policy="tail")

    def estimate_bits(self, layer_id, kv_kind, seq_len, head_dim, cache_meta):
        residual = self.key_residual_length if kv_kind == "key" else self.value_residual_length
        residual = self._residual_keep_length(seq_len, residual, "tail")
        quant_tokens = max(0, seq_len - residual)
        return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
```

(`FP_BITS = 16` and `math`/`torch` are provided by the surrounding module.)

## Evaluation settings

Five workloads on the shared public text-benchmark protocol, all scored 0–100, single seed `{42}`: `longbench_hotpotqa` (LongBench-E `hotpotqa_e`, QA F1), `longbench_passage_retrieval` (`passage_retrieval_en_e`, retrieval score), `longbench_repobench` (`repobench-p_e`, code similarity), `needlebench_niah` (RULER/NeedleBench-style exact-phrase retrieval over public essay text), and `gsm8k` (`openai/gsm8k` main test, exact-answer accuracy after numeric normalization). The parser reads, per workload, a `final_score` (quality), an `effective_kv_bits` (the quantizer's average bits per cached element), a `kv_compression_ratio` (`16 / effective_kv_bits`, FP16 reference), and a diagnostic `runtime_seconds` that does not enter the score. Each workload score is a weighted mean of quality (weight 6, bounded-power vs the best current baseline) and KV efficiency (weight 4, bounded-power on compression ratio with 4× as the reference and 8× as the bound); the task score is the geometric mean across the five workloads. `effective_kv_bits` is computed at a fixed 4096-token reference span, so efficiency is hardware-independent.
