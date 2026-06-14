# KIVI, distilled

KIVI is a tuning-free, plug-and-play low-bit quantizer for the LLM key-value (KV) cache. Its
one idea is an **asymmetric axis choice**: quantize the **key cache per-channel** and the
**value cache per-token**, with group-wise asymmetric round-to-nearest and a small
full-precision residual window on each cache to handle streaming and protect locally-relevant
tokens. The configuration here is **K4/V4** — 4-bit keys and 4-bit values — the conservative
member of the family for settings where the cache should absorb less quantization noise.

## Problem it solves

In batched LLM decoding the KV cache (`b × (l_prompt + l_gen) × d` per layer) is the dominant
memory cost and, because every generated token requires streaming the whole cache from HBM to
SRAM, the dominant speed cost. Quantizing each element from FP16 to a few-bit integer reduces
both — but it must be done with **no training or calibration**, **incrementally on a streaming
cache** (new keys/values arrive one token at a time), and **hardware-friendly**. Simple uniform
group-wise quant holds quality at 4-bit but collapses at 2-bit; KIVI reaches the low-bit regime
by quantizing key and value along different axes.

## Key idea and why each axis

- **Key → per-channel.** Key activations carry outliers in a few *fixed* channels (~100×
  normal), persistent across tokens. Per-channel grouping puts each channel in its own group
  with its own `(min, scale)`, confining the outlier channel's huge range to itself instead of
  inflating the scale of normal channels (which is what per-token grouping does, since a
  per-token group spans the outlier channel). Measured: ~5× smaller attention-score error
  per-channel than per-token for keys.
- **Value → per-token.** Value has no channel outliers, so raw reconstruction error is
  comparable either way — but the value is a *mixer*: the output row
  `[A·X_V]_{i*} = Σ_j A_{ij} [X_V]_{j*}` is a sparse weighted sum of value rows (attention is
  ~84% sparse). Per-token quant confines each token's error to its row; the sparse mix ignores
  the unimportant rows, so their error never reaches the output. Per-channel quant smears
  unimportant-token error through the shared channel scale into the important rows. Measured:
  ~15× smaller output error `‖(A X_V − A X'_V)/(A X_V)‖_F` per-token than per-channel.

## Quantization primitive

Group-wise asymmetric round-to-nearest over `B` bits, per group of `G` contiguous elements:

```
z = min(group),   s = (max(group) - min(group)) / (2^B - 1)
Q(x) = clamp(round((x - z) / s), 0, 2^B - 1)
x'   = Q(x) * s + z          # dequantized (fake-quant) value
```

Asymmetric (zero-point at the min) because KV distributions are not zero-centered; this uses
the whole `[0, 2^B - 1]` codebook.

## Streaming split (grouped vs. residual)

Per-token value quant is streaming-natural (a new token is a complete group). Per-channel key
quant groups *along the token axis*, which grows one token at a time, so a complete group of `G`
tokens cannot be formed until enough tokens arrive. KIVI splits each cache:

- **Grouped part** `X_{·g}`: a token-count divisible by the group, stored quantized.
- **Residual part** `X_{·r}`: kept in full precision (FP16).
  - **Key residual** = `l mod R` (the leftover that does not complete an `R`-block); flushed and
    quantized when it reaches `R`, then reset — a sawtooth FP window averaging ~`R/2`.
  - **Value residual** = a fixed sliding tail of the last `R` rows; on overflow the oldest row
    is popped, quantized per-token, and appended.

Attention is computed separably and concatenated:
`A = Concat([t_Q·Q(X_{K_g})^T, t_Q·X_{K_r}^T])`, then `t_O = A_g·Q(X_{V_g}) + A_r·X_{V_r}`.
During **prefill** the exact FP keys/values are passed to the next layer; only the quantized
cache is retained. The FP residual window also keeps the locally-relevant recent tokens lossless.

## Defaults and why

- **Group size `G = 32`.** Smaller groups → tighter ranges, lower error, but more stored
  `(min, scale)` metadata. 32 is the error-vs-metadata default used here.
- **Residual length `R = 128`** (multiple of `G`). A reasonably large `R` protects recent-token
  locality while staying small against a long sequence. The FP residual is ≤ `R` tokens against
  a sequence ≫ `R`, so its memory cost is negligible.
- **Bits `B = 4`** here (K4/V4). The general method spans `B = 2` and `B = 4`; 4-bit is used
  where there is little cache redundancy to spend.

## Effective bits

A quantized token costs `B` data bits/element, an FP residual token costs 16; this is the
tensor-replay harness estimate and does not include packed-kernel scale/min metadata:
`avg_bits = (quant_tokens·B + residual·16) / seq_len`, with `quant_tokens = seq_len − residual`.
Key residual `= seq_len mod R`; value residual `= min(R, seq_len)`, which equals `R` once the
sequence is at least `R` tokens. At a 4096-token reference, `R = 128`, `B = 4`: key = 4.0 bits
(`4096 mod 128 = 0`); value `= (3968·4 + 128·16)/4096 = 4.375` bits; averaged across same-sized
key and value elements, the KV cache is `(4.0 + 4.375)/2 = 4.1875` bits per element.

## Algorithm (prefill / decode)

```
Prefill(X):                                # X: [l_prompt, d]
  X_K, X_V = X W_K, X W_V
  X_Vg, X_Vr = X_V[:max(l_prompt-R, 0)], X_V[-min(R, l_prompt):]  # value: FP tail
  Q(X_Vg) = GroupQuant(X_Vg, groups=channels_within_each_token)
  Q(X_Kg), X_Kr = KeyQuant(X_K)                          # key: per-channel quant grouped, FP leftover
  cache <- Q(X_Kg), X_Kr, Q(X_Vg), X_Vr
  return X_K, X_V                                        # exact tensors forwarded

Decode(cache, t):                          # t: one new token
  t_Q, t_K, t_V = t W_Q, t W_K, t W_V
  X_Kr <- Concat([X_Kr, t_K]); X_Vr <- Concat([X_Vr, t_V])
  if len(X_Kr) == R:                                     # key residual full: flush a whole block
     Q(X_Kr), _ = KeyQuant(X_Kr); Q(X_Kg) <- Concat([Q(X_Kg), Q(X_Kr)]); X_Kr <- empty
  if len(X_Vr) > R:                                      # value residual overflow: quantize oldest
     Q(X_Vr') = GroupQuant(X_Vr[:-R], groups=channels_within_each_token); Q(X_Vg) <- Concat([Q(X_Vg), Q(X_Vr')]); X_Vr <- X_Vr[-R:]
  A = softmax( Concat([t_Q Q(X_Kg)^T, t_Q X_Kr^T]) / sqrt(d_head) )
  t_O = A[:-len(X_Vr)] Q(X_Vg) + A[-len(X_Vr):] X_Vr
  return t_O

KeyQuant(X_K):                             # X_K: [l, d]
  r = l mod R; X_Kg, X_Kr = X_K[:l-r], X_K[l-r:]
  return GroupQuant(X_Kg, groups=tokens_within_each_channel), X_Kr
```

## Working code (tensor-level fake-quant)

Filling the `AdaptiveKVQuantizer` slot of the tensor-replay harness; KV tensors are
`[batch, heads, seq_len, head_dim]`. Per-channel key quant transposes so the token axis is last;
per-token value quant groups the flattened per-token feature axis. The grouped part is
quantize-then-dequantize, the residual is kept exact.

```python
import math
import torch

FP_BITS = 16.0


class AdaptiveKVQuantizer:
    """KIVI K4/V4: key per-channel, value per-token, with full-precision residual windows."""

    def __init__(self):
        self.bits = 4
        self.group_size = 32
        self.key_residual_length = 128
        self.value_residual_length = 128

    def reset_request(self, request_meta: dict, budget_state: dict):
        self.bits = 4

    def needs_prefill_qkv_observer(self) -> bool:
        return False                       # tuning-free

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        return None

    def query_observation_position(self) -> str:
        return "post_rope"

    def _residual_keep_length(self, seq_len: int, residual_length: int, residual_policy: str) -> int:
        residual_length = max(0, int(residual_length))
        if residual_policy == "block_modulo":          # key: leftover not completing an R-block
            return seq_len % residual_length if residual_length else 0
        if residual_policy == "tail":                  # value: sliding tail of R rows
            return min(seq_len, residual_length)
        return 0

    def _minmax_last_dim(self, data: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
        # group-wise asymmetric round-to-nearest over the LAST dim, returned dequantized
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, int(2 ** bits) - 1)                  # largest integer code
        trailing = data.shape[-1]
        group_size = trailing if int(group_size) <= 0 else int(group_size)
        padded = math.ceil(trailing / group_size) * group_size
        work = torch.nn.functional.pad(data, (0, padded - trailing)) if padded != trailing else data
        grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
        gmin = grouped.amin(dim=-1, keepdim=True)                       # z = min
        gmax = grouped.amax(dim=-1, keepdim=True)
        scale = (gmax - gmin).clamp(min=1e-5) / max_int                 # s = (max-min)/(2^B-1)
        q = torch.round((grouped - gmin) / scale).clamp(0, max_int)
        return q.mul(scale).add(gmin).reshape(*work.shape[:-1], padded)[..., :trailing]

    def _quantize(self, tensor: torch.Tensor, axis: str, residual_policy: str) -> tuple[torch.Tensor, float]:
        work = tensor.float().clone()
        batch, heads, seq_len, head_dim = work.shape
        residual_length = self.key_residual_length if residual_policy == "block_modulo" else self.value_residual_length
        residual = self._residual_keep_length(seq_len, residual_length, residual_policy)
        quant_end = seq_len - residual
        if quant_end <= 0:
            return work.to(tensor.dtype), FP_BITS
        quant_slice = work[:, :, :quant_end, :]
        if axis == "channel":                                          # per-channel: group G tokens / channel
            usable = quant_slice.shape[-2] - (quant_slice.shape[-2] % self.group_size)
            if usable > 0:
                main = quant_slice[:, :, :usable, :].transpose(2, 3)    # token axis -> last
                main = main.reshape(batch, heads, head_dim, usable // self.group_size, self.group_size)
                main = self._minmax_last_dim(main, self.bits, self.group_size)
                work[:, :, :usable, :] = main.reshape(batch, heads, head_dim, usable).transpose(2, 3)
            fp_tokens = residual + (quant_slice.shape[-2] - usable)     # partial channel-group stays FP
            avg_bits = (usable * self.bits + fp_tokens * FP_BITS) / max(seq_len, 1)
        else:                                                          # per-token: group flattened token features
            flat = quant_slice.transpose(1, 2).reshape(batch, quant_slice.shape[-2], heads * head_dim)
            flat = self._minmax_last_dim(flat, self.bits, self.group_size)
            work[:, :, :quant_end, :] = flat.reshape(batch, quant_slice.shape[-2], heads, head_dim).transpose(1, 2)
            avg_bits = (quant_end * self.bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize(key_states, "channel", "block_modulo")

    def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize(value_states, "token", "tail")

    def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
        policy = "block_modulo" if kv_kind == "key" else "tail"
        residual_length = self.key_residual_length if kv_kind == "key" else self.value_residual_length
        residual = self._residual_keep_length(seq_len, residual_length, policy)
        quant_tokens = max(0, seq_len - residual)
        return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
```

The packed/CUDA implementation (`triton_quantize_and_pack_along_last_dim`, fused dequant-matmul)
is the production form of the same scheme: it transposes the key so the token axis becomes the
last dim and groups tokens within each channel, quantizes the value along each token's feature
axis, packs `32 // B` integers per `int32`, and computes attention over the quantized-grouped
and FP-residual pieces via a mixed-precision matmul. The fake-quant code above is the
dequantized equivalent used by the tensor-replay harness.
