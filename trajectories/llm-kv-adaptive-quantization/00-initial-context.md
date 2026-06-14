## Research question

Decoder-only LLM inference is bottlenecked, in the long-context large-batch regime, by the
key/value cache rather than the weights: every past token leaves a key vector and a value vector in
every layer, the cache grows linearly in batch times sequence length, and each decode step streams
the whole thing out of HBM while the compute cores idle. The one lever that shrinks bytes-per-element
without dropping tokens or retraining is **quantization** of the running K/V tensors. The single
thing being designed here is the **quantization policy** — the bit allocation, the grouping axis, the
asymmetric ranges and zero-points, the full-precision residual window, and the optional prefill-time
observation — that preserves benchmark output quality while reducing the effective KV footprint.
Everything else (the model, the decode replay loop, the workloads, the scoring) is fixed.

## Prior art before the first rung

The cache is a *streaming* object: keys and values arrive one token at a time during decode and get
appended to a tensor whose length keeps growing. That single fact forecloses most of the
quantization literature and sets up the family the first rung reacts to.

- **Optimization-based PTQ (GPTQ, Frantar et al. 2022; OBQ/OBC, Frantar & Alistarh 2022).** Solve a
  per-layer reconstruction problem with second-order error feedback — accurate, but they re-solve an
  optimization over a whole weight matrix. Gap: the cache changes every decode step, so re-fitting
  anything per step is hopeless; only a cheap, local primitive survives.
- **Group-wise round-to-nearest (FlexGen, Sheng et al. 2023).** The streaming-feasible primitive:
  per group take min/max, set zero-point `z = min`, scale `s = (max − min)/(2^B − 1)`, round
  `(x − z)/s` into `[0, 2^B − 1]`, dequantize `q·s + z`. No calibration, runs on a single token.
  FlexGen applies it uniformly, per-token, 4-bit. Gap: uniform low-bit RTN is fine at 4-bit and falls
  off a cliff at 2-bit, and it spends bits flat — it has nothing to say about *where* (which axis,
  which layer) the precision should go.
- **Outlier-aware quantization (LLM.int8(), Dettmers et al. 2022; SmoothQuant, Xiao et al. 2023).**
  Transformer activations carry persistent large-magnitude channels; the key cache `X·W_K` inherits
  them, the value cache does not. Gap: the diagnosis is about *which axis* confines outlier error,
  not yet a streaming cache policy.

The four baselines below are quantization policies on a *common tensor-level interface*; the
contribution is always the policy, never a backend or a paper repository.

## The fixed substrate

A deterministic greedy-decode replay harness over Hugging Face `Transformers` is frozen and must not
be touched. It loads `Qwen/Qwen2.5-3B-Instruct` (36 layers), runs the benchmark prompts, and at each
decode step after prefill it **snapshots the real KV tensors, hands them to the editable quantizer,
restores the quantized cache, and advances generation**. Prefill itself is lossless — the exact
prompt K/V flow forward; only the cached copy is quantized. The harness owns: dataset loading
(LongBench-E, NeedleBench/NIAH, GSM8K), prompt templates, generation limits, the parser, the score
definitions, and a 4096-token reference-span efficiency accountant (`estimate_policy_efficiency`).
It also exposes, for a policy that wants prefill statistics, a query-observer hook that captures
post-RoPE query states per layer.

## The editable interface

Exactly one region is editable — the `AdaptiveKVQuantizer` class in `custom_quant_eval.py` (lines
41–172). Every method on the ladder is a fill of this same contract. The harness calls, per example:
`reset_request(request_meta, budget_state)`; then `needs_prefill_qkv_observer() -> bool` and (if true)
`observe_prefill_qkv(layer_id, query_states, key_states, value_states, attention_meta)` with
`query_observation_position() -> str` selecting pre/post-RoPE; then per decode step, per layer,
`quantize_key(layer_id, key_states, cache_meta) -> tensor | (tensor, avg_bits)` and
`quantize_value(...)`; and for the hardware-independent footprint,
`estimate_bits(layer_id, kv_kind, seq_len, head_dim, cache_meta) -> float`. `key_states` and
`value_states` have shape `[batch, heads, seq_len, head_dim]`, and a `quantize_*` return **must be a
tensor of the identical shape** (it is fake-quant: quantize-then-dequantize back to the model's
dtype). There is no algorithm enum and no backend selector — grouping, asymmetric ranges,
zero-points, per-layer presets, residual retention, and memory accounting all live inside this class.

The starting point is the scaffold default: a single global 4-bit policy, key per-channel, value
per-token, group size 32, a tail residual window. Each baseline replaces this class wholesale.

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

Five workloads on the shared public text-benchmark protocol, all scored 0–100, single seed `{42}`:
`longbench_hotpotqa` (LongBench-E `hotpotqa_e`, QA F1), `longbench_passage_retrieval`
(`passage_retrieval_en_e`, retrieval score), `longbench_repobench` (`repobench-p_e`, code
similarity), `needlebench_niah` (RULER/NeedleBench-style exact-phrase retrieval over public essay
text), and `gsm8k` (`openai/gsm8k` main test, exact-answer accuracy after numeric normalization).
The parser reads, per workload, a `final_score` (quality), an `effective_kv_bits` (the quantizer's
average bits per cached element), a `kv_compression_ratio` (`16 / effective_kv_bits`, FP16 reference),
and a diagnostic `runtime_seconds` that does **not** enter the score. Each workload score is a
weighted mean of quality (weight 6, bounded-power vs the best current baseline) and KV efficiency
(weight 4, bounded-power on compression ratio with 4× as the reference and 8× as the bound); the task
score is the geometric mean across the five workloads. `effective_kv_bits` is computed at a fixed
4096-token reference span, so efficiency is hardware-independent.
