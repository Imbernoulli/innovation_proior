# KVTuner, distilled (KIVI-mode layer-wise mixed-precision KV quantization)

KVTuner is a sensitivity-aware, layer-wise, *mixed-precision* KV-cache quantizer. Instead of
quantizing every transformer layer's key/value cache to the same bit-width, it assigns each
layer one coarse precision pair `(P_k^l, P_v^l)`, searches that table offline, and loads it at
inference — so every token inside a layer keeps uniform precision (kernel-friendly), the bit
budget is concentrated on the key (which matters more than the value), and there is zero
online decision-making cost. This entry is the KIVI-mode configuration for Qwen2.5-3B-Instruct:
keys per-channel, values per-token, group size 32, residual length 32.

## Problem it solves

Low-bit KV-cache quantization is the most deployable way to shrink the decode-time memory
bottleneck, but below ~4 bits it degrades generation through *catastrophic token flips* (a
single flipped operator/digit derails a reasoning chain), driven by error that accumulates
two-dimensionally across layers and decode steps. The degradation is sharply uneven across
key vs. value, across quantization axis, and across layers — and layer sensitivity has no
clean depth heuristic. The goal: reach a low average bit-width while staying nearly lossless,
keep each layer hardware-friendly, and add zero online overhead.

## Key ideas

1. **Key cache is more important than value cache.** At equal memory, K4V2 beats K2V4 (e.g.
   relative attention-output error `e_o` 0.453 vs 0.892 on Llama-3.1-8B). Spend bits on keys.
2. **Why (the lemma).** Quantizing keys shifts the attention *distribution*, and the shift
   rides entirely on `qΔK`; only a head with a dominating key cancels it. So the fix for a
   sensitive (non-sparse) head is to raise *key* precision, not value precision (value error
   is applied after the weights are chosen and cannot repair a shifted distribution).
3. **Axis recipe.** Keys have persistent channel outliers → per-channel key quant confines
   the error to the outlier's channel (~5× better attention-score error than per-token).
   Values enter attention only as a sparse weighted sum → per-token value quant keeps the few
   important rows clean (~15× better attention-output error than per-channel). Per-channel
   keys are made streaming via a grouped part + an FP residual sliding window.
4. **Layer sensitivity is inherent and prompt-independent**, with no depth shortcut → measure
   it offline once and assign per-layer precision pairs accordingly.
5. **Coarse-grained per layer, mixed across layers, fully offline** → fuses with kernels,
   adapts where it matters, costs nothing online.

## The lemma (only theorem)

*Only attention heads with sparse and concentrated patterns are consistently robust to
low-precision KV cache quantization.*

For query `q ∈ R^{1×D}` and key cache `K ∈ R^{D×S}`, the error-free score is
`a_i = exp(qK_i) / Σ_j exp(qK_j)`. Asymmetric uniform key error `ΔK ∼ N(0, σ²)` with
`σ = (max K − min K)/(2^B − 1)` grows geometrically as bits `B` drop. The corrupted score
factorizes as

```
â_i = exp(qK_i)·exp(qΔK_i) / Σ_j exp(qK_j)·exp(qΔK_j)
    = exp(qK_i) / Σ_j exp(qK_j)·[ exp(qΔK_j)/exp(qΔK_i) ].
```

The distribution is exactly unchanged only in the degenerate Case 1 — all projected errors
identical, `qΔK_i = qΔK_j` for all `i,j`, generically false for independent errors. The useful
case is Case 2 and it is *approximate*: if one key `i` dominates, `exp(qK_i) ≫ exp(qK_j)` for
`j ≠ i`, then dividing by `exp(qK_i)`,

```
â_i = exp(qΔK_i) / Σ_j exp(qΔK_j)·[exp(qK_j)/exp(qK_i)] ≈ exp(qΔK_i)/exp(qΔK_i) = 1,
```

with dominated `â_j ≈ 0` — the distribution barely moves regardless of error size (the
robustness tightens as one key's mass grows). So a sparse/concentrated head is robust; a
spread retrieval head is fragile, and the remedy is to shrink `qΔK` by raising key precision
in sensitive layers. ∎

## The search

Configuration `P ∈ S^L` assigns each layer a pair `(P_k^l, P_v^l)` from `{2,4,8}²` (9 per
layer). Multi-objective problem:

```
min_P ( f_m(P), f_a(P) )   s.t.   f_m(P) ≤ M,  f_a(P) ≤ ΔA,
f_m(P) = Σ(P)/(2L)  (average equivalent bits),
f_a(P) = A_LLM(KV_half) − A_LLM(KV_P)  (accuracy loss vs FP16 KV, a black box).
```

Naive space `9^L` (≈ 3.4×10³⁰ at L=32) is pruned in two levels:

- **Intra-layer Pareto pruning.** Per layer, keep only the Pareto frontier in (equivalent
  bits, `e_o`). Because key > value, most layers reduce to the key-first set
  {KV8, K8V4, KV4, K4V2, KV2} (5 pairs) → `S^L → S_p^L` (`5^32 ≈ 2.3×10²²`). Layer exceptions
  (e.g. K4V8/K2V4 in the per-channel-key mode) are kept where the frontier differs.
- **Inter-layer clustering.** Partition layers by their pruned candidate set, then DBSCAN
  (`eps=0.05`, `min_samples=2`) on the `e_o` sensitivity vectors → `L` (28–64) collapses to
  `G` (4–8) groups, one pair per group: `S_p^L → S_p^G` (`5^6 = 15625`).

Then a black-box multi-objective evolutionary search (Optuna with an MOEA/D / NSGA-style
sampler) over the per-group pairs: maximize accuracy on the first 200 GSM8K 4-shot prompts,
minimize equivalent bits, ~200 iterations, soft constraints at 4-bit and 6-bit. **Calibration
design:** use *dequantized* KV in prefill self-attention (turns on the cross-layer error
accumulation) and a hard math dataset (so a flipped token shows up in the final answer), to
amplify and distinguish the failure mode. The Pareto-optimal table is loaded at inference.

## Quantizer (signed-asymmetric round-to-nearest)

For a group `g` of size `group_size`, signed grid `q_max = 2^{B−1}−1`, `q_min = −2^{B−1}`
(span `q_max − q_min = 2^B − 1`, the same range formula on a signed integer grid):

```
scale = clamp(max(g) − min(g), 1e-5) / (q_max − q_min)
zeros = round(min(g)/scale) − q_min
quant = clamp(round(g/scale − zeros), q_min, q_max)
dequant = (quant + zeros)·scale
```

KIVI mode: key axis = per-channel (transpose last two dims, group over a token window),
value axis = per-token (group over head_dim within each token), group size 32, residual
length 32 (keep the most-recent `seq_len mod 32` tokens in FP). Effective bits per element
`= (quant_tokens·B + residual_tokens·16)/seq_len`.

## Searched preset (KIVI mode, Qwen2.5-3B-Instruct, 4-bit search family)

36 layers, key-dominant: K4V8 in layers 0,1; K2V4 in 2,4; K2V2 in 12,28; K4V4 in 34,35;
K4V2 in all other 28 layers. Key bits sum 136, value bits sum 92, so the preset equivalent
precision is `(136 + 92)/(2·36) ≈ 3.17` bits before the sequence-dependent residual overhead.
"KVTuner4" names the 4-bit search family, not a realized exact 4-bit width.

## Working code

```python
import math
import torch

FP_BITS = 16.0  # FP16 KV reference footprint


class KVTunerKIVIQuantizer:
    """KVTuner layer-wise mixed-precision KV quantizer, KIVI mode, Qwen2.5-3B-Instruct.
    Key per-channel, value per-token, group size 32, residual 32. The per-layer
    (key bits, value bits) table is the offline Pareto/MOO search result; online it is a
    pure lookup with no decision-making overhead. The signed-asymmetric arithmetic matches
    the FlexibleVanillaQuantizedCache / VanillaQuantizer asymmetric path."""

    _PRESET = {
        0: {"key": 4, "value": 8}, 1: {"key": 4, "value": 8}, 2: {"key": 2, "value": 4},
        3: {"key": 4, "value": 2}, 4: {"key": 2, "value": 4}, 5: {"key": 4, "value": 2},
        6: {"key": 4, "value": 2}, 7: {"key": 4, "value": 2}, 8: {"key": 4, "value": 2},
        9: {"key": 4, "value": 2}, 10: {"key": 4, "value": 2}, 11: {"key": 4, "value": 2},
        12: {"key": 2, "value": 2}, 13: {"key": 4, "value": 2}, 14: {"key": 4, "value": 2},
        15: {"key": 4, "value": 2}, 16: {"key": 4, "value": 2}, 17: {"key": 4, "value": 2},
        18: {"key": 4, "value": 2}, 19: {"key": 4, "value": 2}, 20: {"key": 4, "value": 2},
        21: {"key": 4, "value": 2}, 22: {"key": 4, "value": 2}, 23: {"key": 4, "value": 2},
        24: {"key": 4, "value": 2}, 25: {"key": 4, "value": 2}, 26: {"key": 4, "value": 2},
        27: {"key": 4, "value": 2}, 28: {"key": 2, "value": 2}, 29: {"key": 4, "value": 2},
        30: {"key": 4, "value": 2}, 31: {"key": 4, "value": 2}, 32: {"key": 4, "value": 2},
        33: {"key": 4, "value": 2}, 34: {"key": 4, "value": 4}, 35: {"key": 4, "value": 4},
    }

    def reset_request(self, request_meta, budget_state):
        return None

    def needs_prefill_qkv_observer(self):
        return False

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        return None

    def query_observation_position(self):
        return "post_rope"

    def _residual_keep_length(self, seq_len, residual_length):
        residual_length = max(0, min(seq_len, int(residual_length)))
        return seq_len % residual_length if residual_length else 0

    def _signed_asymmetric(self, tensor, bits, axis, group_size, residual_length):
        work = tensor.float().clone()
        _, _, seq_len, _ = work.shape
        residual = self._residual_keep_length(seq_len, residual_length)
        quant_end = seq_len - residual
        if quant_end <= 0 or bits >= FP_BITS - 0.5:
            return work.to(tensor.dtype), FP_BITS
        quant_slice = work[:, :, :quant_end, :]
        shaped = quant_slice.transpose(-2, -1).contiguous() if axis == 1 else quant_slice
        group_size = shaped.shape[-1] if int(group_size) == -1 else int(group_size)
        original_shape = shaped.shape
        trailing = shaped.shape[-1]
        padded = math.ceil(trailing / group_size) * group_size
        shaped = torch.nn.functional.pad(shaped, (0, padded - trailing)) if padded != trailing else shaped
        rows = shaped.reshape(-1, group_size)
        q_max, q_min = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))   # q_max - q_min = 2**bits - 1
        max_vals = rows.max(dim=1).values
        min_vals = rows.min(dim=1).values
        scale = (max_vals - min_vals).clamp(min=1e-5) / (q_max - q_min)
        zeros = (min_vals / scale).round() - q_min
        quant = torch.round(rows / scale.unsqueeze(1) - zeros.unsqueeze(1)).clamp(q_min, q_max)
        dequant = (quant + zeros.unsqueeze(1)) * scale.unsqueeze(1)
        dequant = dequant.reshape(*original_shape[:-1], padded)[..., :trailing]
        if axis == 1:
            dequant = dequant.transpose(-2, -1).contiguous()
        work[:, :, :quant_end, :] = dequant
        avg_bits = (quant_end * bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id, key_states, cache_meta):
        return self._signed_asymmetric(key_states, self._PRESET[layer_id]["key"],
                                       axis=1, group_size=32, residual_length=32)

    def quantize_value(self, layer_id, value_states, cache_meta):
        return self._signed_asymmetric(value_states, self._PRESET[layer_id]["value"],
                                       axis=0, group_size=32, residual_length=32)

    def estimate_bits(self, layer_id, kv_kind, seq_len, head_dim, cache_meta):
        residual = self._residual_keep_length(seq_len, 32)
        quant_tokens = max(0, seq_len - residual)
        bits = self._PRESET[layer_id][kv_kind]
        return float((quant_tokens * bits + residual * FP_BITS) / max(seq_len, 1))
```

The offline search machinery (per-layer `e_o` measurement, Pareto pruning, DBSCAN clustering,
the multi-objective evolutionary search) produces `_PRESET`; the inference-time quantizer above
is the entire online cost.
