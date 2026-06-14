# KVTuner, distilled

KVTuner is sensitivity-aware, layer-wise, mixed-precision KV cache quantization.
Instead of quantizing every layer's keys and values to one global bit-width, it
assigns each transformer layer its own coarse-grained precision pair `(P_k, P_v)`
(e.g. K8V4, K4V2) — coarse-grained meaning the whole low-bit KV inside a layer
shares one pair, so the layer stays uniform precision and fuses with FlashAttention
and vLLM. The per-layer table is found by a multi-objective search ahead of time and
loaded at inference with zero online overhead, because layer-wise quantization
sensitivity is an inherent, prompt-independent property of the model.

## Problem it solves

Low-bit KV cache quantization shrinks the decode-time memory/bandwidth bottleneck,
but accuracy collapses below 4-bit (and at 4-bit *key* on sensitive models) because
quantization error accumulates in two dimensions — across layers and across decode
steps — and a single flipped token poisons the rest of a multi-step generation.
Uniform bit-widths cannot place precision where accumulation happens; online
fine-grained per-token/per-page methods can place it but break kernel fusion and add
decode-time control-flow overhead. KVTuner targets a near-lossless sub-4-bit
*effective* budget that an existing inference stack can run unchanged.

## Key ideas

1. **Key cache matters more than value cache.** Quantization noise distorts the
   *attention distribution* through the key term `q ΔK` inside the softmax; value
   error only reweights the output afterward. Expanding the noisy score,
   `â_i = exp(q K_i) / [ sum_j exp(q K_j)·( exp(q ΔK_j)/exp(q ΔK_i) ) ]`, the
   distribution is preserved (Lemma) only for **sparse, concentrated heads** (a
   dominating key drives `â_i → 1`); non-sparse retrieval heads shift. The lever
   against this is *key* precision. Values, by contrast, tolerate aggressive
   quantization because attention is sparse and per-token value quantization
   confines each token's error to that token. So at equal memory, K4V2 beats K2V4
   (`e_o ≈ 0.453` vs `0.892` on Llama-3.1-8B per-token-asym), and the per-layer
   candidate menu collapses to the key-first ladder `{KV8, K8V4, KV4, K4V2, KV2}`.

2. **Sensitivity is layer-wise and prompt-independent, with no depth heuristic.**
   So the precision assignment can be searched once ahead of time and loaded at
   inference.

3. **Coarse-grained, mixed-across-layers, decided ahead of time.** Coarse inside a
   layer (kernel-friendly), mixed across layers (matches the order-of-magnitude
   variation in sensitivity), decided offline (no runtime cost) — the gap left by
   uniform (coarse, not mixed) and online fine-grained (mixed, but runtime-costly)
   methods.

## The search

Discrete multi-objective optimization over per-layer pairs:

```
min_P (f_m(P), f_a(P))   s.t.  f_m(P) <= M,  f_a(P) <= ΔA
P in S^L : (P_k^l, P_v^l) per layer
f_m(P) = sum(P) / (2L)                         # average equivalent bits
f_a(P) = A_LLM(KV_half) - A_LLM(KV_P)          # accuracy loss vs fp16 KV
```

Raw space is `9^L` (`{2,4,8}×{2,4,8}` per layer; a 32-layer model: `9^32 ≈ 3.4e30`).
Two-level reduction:

- **Intra-layer Pareto pruning (lossless under the local criterion).** For each
  layer, keep only the pairs on the Pareto frontier of (equivalent bits, relative
  attention-output error `e_o`). Most layers reduce to the key-first 5 pairs.
  `9^L → ~5^L` (e.g. `5^32 ≈ 2.3e22`).
- **Inter-layer clustering (pragmatic collapse).** Partition layers by pruned
  candidate set, then cluster by sensitivity (`e_o` signature) with DBSCAN
  (`eps=0.05`, `min_samples=2`); tie each cluster to one pair. Tying layers can
  exclude assignments, so this is not lossless — it is justified by the observed
  stability of sensitivity profiles. `L` (28-64) → `G` groups (4-8); `5^L → 5^G`
  (e.g. `5^6 = 15625`).

Then a black-box multi-objective evolutionary search (a decomposition-based /
NSGA-II-style sampler driven through Optuna): maximize accuracy on the first 200
GSM8K 4-shot prompts, minimize equivalent bits, soft constraints around 4-bit and
6-bit. **Calibration** uses *dequantized* KV for prefill self-attention so error
accumulates across layers, on hard math generations where a flipped intermediate
token produces a large, measurable final-answer swing — amplifying the signal that
separates good assignments. The output is a per-layer `(P_k, P_v)` table loaded at
inference. (This accumulate-through-prefill behavior is a calibration device for
the search; at run time the cached KV is quantized while the current prefill states
pass through at full precision.)

## Deployable per-token quantization

Both keys and values are quantized along the **token** dimension (axis 0), so a new
token's row is quantized and appended without per-channel regrouping or special
operators — the simplest streaming-friendly mode. The per-token key error (larger
than per-channel, since keys have channel outliers) is paid back by the search,
which raises *key* bits on the layers that need them. Signed-asymmetric
round-to-nearest over a group (per-token = whole `head_dim` as one group, no
residual window):

```
q_max = 2^(B-1) - 1,   q_min = -2^(B-1)              # q_max - q_min = 2^B - 1
scale = clamp(max(g) - min(g), 1e-5) / (q_max - q_min)   # = range / (2^B - 1)
zeros = round(min(g)/scale) - q_min                  # maps group min onto q_min
quant = clamp( round(g/scale - zeros), q_min, q_max )
dequant = (quant + zeros) * scale                    # zero-point cancels back out
```

The zero-point lets the asymmetric KV range use the full signed grid;
`residual_length = 0` because the fp16 residual window is a per-channel streaming
artifact, unneeded per-token.

Searched table for **Qwen2.5-3B-Instruct, per-token mode** (36 layers; sparse and
non-monotone in depth — the reason a depth heuristic fails): L0 = K8V4; L18, L27,
L29 = K8V8; L10, L19, L24, L26, L33 = K4V4; all other 27 layers = K4V2. The bit
sums are 160 key + 102 value over 36 layers, so the equivalent precision is
`(160 + 102)/(2·36) = 3.6389` bits. (A different searched assignment for the same
model/mode at a strict 4.00-bit budget also exists — K8V2 at L0, more K8V4 layers;
the table below is the ~3.64-bit one used here.)

## Working code

```python
import math
import torch

FP_BITS = float(torch.finfo(torch.float16).bits)  # 16: uncompressed reference


class AdaptiveKVQuantizer:
    """KVTuner signed-asymmetric quantizer with a searched per-layer mixed-precision
    table (Qwen2.5-3B-Instruct, per-token mode, ~3.64 equivalent bits)."""

    _PRESET = {
        0: {"key": 8, "value": 4}, 1: {"key": 4, "value": 2}, 2: {"key": 4, "value": 2},
        3: {"key": 4, "value": 2}, 4: {"key": 4, "value": 2}, 5: {"key": 4, "value": 2},
        6: {"key": 4, "value": 2}, 7: {"key": 4, "value": 2}, 8: {"key": 4, "value": 2},
        9: {"key": 4, "value": 2}, 10: {"key": 4, "value": 4}, 11: {"key": 4, "value": 2},
        12: {"key": 4, "value": 2}, 13: {"key": 4, "value": 2}, 14: {"key": 4, "value": 2},
        15: {"key": 4, "value": 2}, 16: {"key": 4, "value": 2}, 17: {"key": 4, "value": 2},
        18: {"key": 8, "value": 8}, 19: {"key": 4, "value": 4}, 20: {"key": 4, "value": 2},
        21: {"key": 4, "value": 2}, 22: {"key": 4, "value": 2}, 23: {"key": 4, "value": 2},
        24: {"key": 4, "value": 4}, 25: {"key": 4, "value": 2}, 26: {"key": 4, "value": 4},
        27: {"key": 8, "value": 8}, 28: {"key": 4, "value": 2}, 29: {"key": 8, "value": 8},
        30: {"key": 4, "value": 2}, 31: {"key": 4, "value": 2}, 32: {"key": 4, "value": 2},
        33: {"key": 4, "value": 4}, 34: {"key": 4, "value": 2}, 35: {"key": 4, "value": 2},
    }

    def reset_request(self, request_meta: dict, budget_state: dict):
        return None

    def needs_prefill_qkv_observer(self) -> bool:
        return False

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        return None

    def query_observation_position(self) -> str:
        return "post_rope"

    def _signed_asymmetric(self, tensor, bits, axis, group_size, residual_length):
        work = tensor.float().clone()
        _, _, seq_len, _ = work.shape
        residual = max(0, min(seq_len, int(residual_length)))
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
        q_max, q_min = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))   # q_max - q_min = 2^bits - 1
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
        return self._signed_asymmetric(key_states, self._PRESET[layer_id]["key"], axis=0, group_size=-1, residual_length=0)

    def quantize_value(self, layer_id, value_states, cache_meta):
        return self._signed_asymmetric(value_states, self._PRESET[layer_id]["value"], axis=0, group_size=-1, residual_length=0)

    def estimate_bits(self, layer_id, kv_kind, seq_len, head_dim, cache_meta):
        return float(self._PRESET[layer_id][kv_kind])
```

This is the same signed-asymmetric arithmetic as the `VanillaQuantizer` asymmetric
path in the KVTuner `FlexibleVanillaQuantizedCache`, run per-token
(`axis=0, group_size=-1, residual_length=0`) with the searched per-layer table.

## Relation to prior methods

- **Uniform round-to-nearest (per-token/per-channel-asym):** one global bit-width;
  KVTuner keeps the same arithmetic but allocates bits per layer by sensitivity.
- **KIVI:** key per-channel + value per-token + fp16 residual window, *uniform*
  across layers. KVTuner adds the layer-wise mixed-precision search on top; its
  per-token variant trades KIVI's per-channel key accuracy for streaming simplicity
  and recovers the loss by spending key bits where the search says.
- **KVQuant:** per-channel pre-RoPE key + non-uniform datatype + sparse outliers
  (one target precision); KVTuner instead varies the *integer* bit-width per layer
  with a deployable uniform grid.
- **QAQ / MiKV / ZipCache:** online critical-token mixed precision; KVTuner moves
  the decision offline and coarse-grained so it fuses with FlashAttention / vLLM.
