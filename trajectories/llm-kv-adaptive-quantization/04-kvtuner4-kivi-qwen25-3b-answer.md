**Problem (from step 3).** The per-token layer-wise search won GSM8K (43.52) and compression (4.40×)
but, being GSM8K-calibrated, sacrificed the pure long-context retrieval workloads it never optimized:
NIAH collapsed to 53.74 (floor 65.31), passage retrieval to 56.63, RepoBench to 43.74. The cause is
*per-token keys* — the lossiest axis for the key cache's persistent channel outliers, exactly what a
needle-retrieval head depends on.

**Key idea (KVTuner, KIVI mode).** Keep the whole layer-wise machinery — accumulation story, key-first
ladder, offline Pareto+DBSCAN-reduced multi-objective search, accumulation-amplified GSM8K oracle —
but run it on top of **per-channel keys** (KIVI mode) instead of per-token keys. Per-channel grouping
confines each fixed-channel outlier to its own group's scale, cutting key error several-fold and the
attention-*score* error ~5× — the fidelity needle retrieval needs — so the searched table spends its
bits on an already-faithful key representation. Values stay per-token (mixer + sparsity).

**Why it works / streaming.** This is the strongest within-layer axis (per-channel keys, step 1) fused
with the strongest across-layer allocation (the searched table, step 3). Per-channel keys do not
stream, so the grouped-plus-residual structure returns: group `G = 32` tokens along the token axis
(transpose, `axis=1`), keep the most-recent `seq_len mod 32` tokens FP (`residual_length = 32`, a
sliding window load-bearing on local reasoning). Signed-asymmetric grid identical to step 3
(`scale = range/(2^B − 1)`, zero-point cancels). No prefill observer (static table).

**Scaffold edit / hyperparameters.** `needs_prefill_qkv_observer() -> False`. The searched Qwen2.5-3B
KIVI-mode `_PRESET`: mostly K4V2; layers 0/1 K4V8; layers 2/4 K2V4; layers 12/28 K2V2; layers 34/35
K4V4. Key `axis=1` `group_size=32` `residual=32`; value `axis=0` `group_size=32` `residual=32`.
Nominal `(136 + 92)/(2·36) ≈ 3.17` bits; `estimate_bits` with the `mod 32` residual at the 4096 span
reports **3.166667** bits → **~5.05× compression** (the highest of any rung).

**What it must clear.** Recover NIAH from 53.74 toward ~65, RepoBench toward the high-40s, passage
retrieval toward the high-50s, while GSM8K holds in the mid-30s (above floor 31.84 and SQuat 31.77,
below the per-token 43.52) — at ~5× compression, the best efficiency on the ladder. The open risk is
HotpotQA dipping from the per-token 39.47 under the more aggressive 3.17-bit budget.

```python
class AdaptiveKVQuantizer:
    """KVTuner FlexibleVanillaQuantizedCache with official KIVI-style preset."""

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

    def reset_request(self, request_meta: dict, budget_state: dict):
        return None

    def needs_prefill_qkv_observer(self) -> bool:
        return False

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        return None

    def query_observation_position(self) -> str:
        return "post_rope"

    def _residual_keep_length(self, seq_len: int, residual_length: int) -> int:
        residual_length = max(0, min(seq_len, int(residual_length)))
        return seq_len % residual_length if residual_length else 0

    def _signed_asymmetric(self, tensor: torch.Tensor, bits: int, axis: int, group_size: int, residual_length: int) -> tuple[torch.Tensor, float]:
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
        q_max, q_min = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
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

    def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._signed_asymmetric(key_states, self._PRESET[layer_id]["key"], axis=1, group_size=32, residual_length=32)

    def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._signed_asymmetric(value_states, self._PRESET[layer_id]["value"], axis=0, group_size=32, residual_length=32)

    def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
        residual = self._residual_keep_length(seq_len, 32)
        quant_tokens = max(0, seq_len - residual)
        bits = self._PRESET[layer_id][kv_kind]
        return float((quant_tokens * bits + residual * FP_BITS) / max(seq_len, 1))
```
