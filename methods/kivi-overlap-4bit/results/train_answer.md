When I serve a large language model cheaply, the only way to amortize the cost is to batch many requests together — but the instant I do that, the key-value cache becomes the thing that eats me alive. Every past token leaves behind a key vector and a value vector in every layer so I never recompute attention over the history, and that cache is $b \times (l_{\text{prompt}} + l_{\text{gen}}) \times d$ per layer: it grows linearly with the batch and linearly with the sequence. For a 540B model at batch 512 and context 2048 that is several terabytes, multiple times the size of the weights themselves. And the pain is twofold, because decoding is memory-bound: to produce a single token the GPU must stream the *entire* cache out of HBM into SRAM, and while that load happens the compute cores sit idle. So the cache is the bottleneck in both memory and speed, and the obvious lever is to spend fewer bytes per cached element — quantize the keys and values from 16-bit float down to a few-bit integer. The trouble is the constraints. I cannot retrain or even calibrate; the model is already trained and I want something plug-and-play. The cache is a *streaming* object: during decode the keys and values arrive one token at a time and get appended, so whatever I do has to work incrementally on a thing whose length keeps growing. That immediately rules out accurate optimization-based post-training quantizers like GPTQ — they solve a per-layer reconstruction problem, and I cannot afford to re-solve anything every decode step. What survives is the cheap, local thing: group-wise round-to-nearest, which is exactly what FlexGen does at 4-bit, group size 64, on both weights and the cache. And the natural plan — take that recipe and just push the bits down — dies precisely where I need it to live. FlexGen groups the cache the streaming-natural way, *per-token*: a freshly generated token is a complete group along the hidden dimension that I can append along the token axis without disturbing anything before it. At 4-bit, per-token for both key and value, the numbers barely move (66.5 versus a 16-bit 66.4 on CoQA). Drop to 2-bit, still per-token for both, and it falls off a cliff to 53. The eviction methods (H2O, StreamingLLM) and the head-sharing methods (MQA, GQA) do not help me here either — they drop tokens or require retraining; they reduce size by a different, lossier mechanism, orthogonal to bit-width. So uniform low-bit quant is the right lever but the wrong policy, and nobody has explained *why* the cache resists going low-bit. I have to look at what the elements actually are.

What I propose is KIVI, and its single load-bearing idea is an asymmetric axis choice: quantize the **key cache per-channel** and the **value cache per-token**, with group-wise asymmetric round-to-nearest, plus a small full-precision residual window on each cache that makes the streaming work and protects the locally-relevant recent tokens. The configuration I write down here is the conservative member of the family, **K4/V4** — 4-bit keys and 4-bit values — used where the cache has little redundancy to spend. The two axis choices come from two completely unrelated failure modes, and that is the whole insight. Look first at the key cache: a few *fixed* channels light up with magnitudes on the order of $100\times$ the rest, and the same channels are large for essentially every token — vertical stripes. This is the well-documented activation-outlier structure, and the key cache, being $X W_K$, an activation, inherits it; the value cache shows no comparable channel-locked outliers. Now think about what an outlier channel does to a per-token group. A per-token group slices across channels within one token, so it *contains* the outlier channel; the group's max is set by that one huge element, the scale $s = (\max - \min)/(2^B - 1)$ blows up, and every normal element sharing the group is quantized with that gigantic step. At 4-bit there are 16 levels and the normal elements barely survive; at 2-bit there are only 4 levels and one outlier throws the rest away. The fix is to group along the **channel** dimension: put each channel's elements (across tokens) into their own group with their own min and scale, so the outlier channel sits alone, its huge range confined to its own group, and the normal channels each get a tight scale because their groups never see the outlier. The diagnostics confirm it — relative reconstruction error on the key drops from $\sim 13.7$ per-token to $\sim 4.6$ per-channel, and what actually matters, the attention-score error $\mathrm{Softmax}(t_Q X_K^\top)$, drops from $\sim 47$ to $\sim 9.6$, roughly a $5\times$ improvement. The key wants per-channel for outlier confinement.

The value is where the reasoning almost trips, and where the second idea lives. The value has no channel outliers, so the outlier argument says nothing, and indeed the raw reconstruction error is comparable either way ($\sim 4.6$ per-token, $\sim 3.7$ per-channel, the per-channel one even slightly smaller). By reconstruction error I would happily quantize the value per-channel. But at 2-bit, per-channel value quantization *collapses the entire model* to single-digit accuracy regardless of how the key is quantized. So reconstruction error of the value is simply the wrong diagnostic, and the reason is *where the value enters the computation*. The value is not a thing I reconstruct — it is a thing I *mix*. The output row is
$$[A X_V]_{i*} = \sum_j A_{ij}\,[X_V]_{j*},$$
a weighted sum over tokens $j$ of value *rows*, with attention scores as weights. The object I must protect is the mixed output $A X_V$, not the value entries in isolation, and there per-token is $\sim 3.6$ while per-channel is $\sim 49.9$, about $15\times$ worse. Why? Attention is sparse — a query attends strongly to only a handful of tokens, $\sim 84\%$ of the weights are near zero — so the output is effectively a combination of a *few* important tokens' value rows. With **per-token** quantization each token's error lives entirely inside its own row; the unimportant rows carry whatever error they carry but are multiplied by $\sim 0$ and contribute nothing, while the few important rows are each cleanly quantized within themselves. The sparse mixer discards exactly the rows whose error would hurt. With **per-channel** quantization a single channel-group spans the important *and* unimportant tokens together, so the group's min/max — hence its scale — is set partly by tokens the output never uses, and that error leaks through the shared channel scale into the important rows, where there is no sparsity to suppress it. So the value must be per-token for mixer-error confinement — the reason is the mixing geometry, not the element distribution. Key per-channel, value per-token, for two unrelated reasons.

The quantization primitive underneath both is group-wise asymmetric round-to-nearest over $B$ bits, per group of $G$ contiguous elements: set the zero-point and scale
$$z = \min(\text{group}), \qquad s = \frac{\max(\text{group}) - \min(\text{group})}{2^B - 1},$$
then $Q(x) = \mathrm{clamp}(\mathrm{round}((x - z)/s),\, 0,\, 2^B - 1)$ and dequantize $x' = Q(x)\cdot s + z$. I make it *asymmetric*, with the zero-point at the min rather than a symmetric range around zero, because KV distributions are not zero-centered — the keys especially are skewed by their outlier channels — so an asymmetric $[\min, \max]$ range uses the whole $[0, 2^B-1]$ codebook where a symmetric range would waste half of it; and the scale is floored away from zero so a constant group never divides by zero.

The hard part is making per-channel key quantization *stream*. Per-token value quant is trivial — a new token's value vector is a complete group the instant it arrives, quantize it and append along the token axis. But per-channel key quant groups *along the token axis*, and during decode tokens arrive one at a time, so I cannot form a complete group of $G$ tokens until $G$ tokens have shown up, and the running count is almost never a clean multiple. The resolution is to split each cache into a grouped quantized part whose token count is divisible by the group, and a full-precision residual tail. I pick a block length $R$ that is a multiple of $G$ and flush in whole blocks rather than one group at a time. For the **key** the residual is $r = l \bmod R$, the leftover that does not complete an $R$-block; the grouped prefix $X_{K_g} = X_K[:l-r]$ is a whole number of $R$-blocks, hence a whole number of $G$-token channel groups, and the residual stays FP16. When the residual reaches $R$ tokens I quantize that whole block per-channel, concatenate it onto the grouped part, and reset — so the key's FP window sawtooths from 0 up to $R-1$ and averages about $R/2$. For the **value** I keep a deliberate sliding tail of the last $R$ rows: on overflow I pop the oldest row, quantize it per-token, and append it. The two residuals therefore behave differently for the same $R$ — the key residual exists because of group divisibility and so resets, the value residual is a steady window I hold on purpose — but in both cases attention still works because the dot product is linear and separable. I compute the logits against the quantized grouped keys and the FP residual keys in two matmuls and concatenate before the softmax,
$$A = \mathrm{Concat}\!\left(\big[\, t_Q\, Q(X_{K_g})^\top,\ t_Q\, X_{K_r}^\top \,\big]\right),$$
and split the output the same way, $t_O = A_g\, Q(X_{V_g}) + A_r\, X_{V_r}$. The residual does double duty: it makes per-channel key quant *possible* by isolating the not-yet-groupable partial group, and it keeps the locally-relevant recent tokens exact — on a hard multi-step generation task each next token depends sharply on the last few, so protecting the nearest cache positions is cheap insurance, costing at most $R$ FP tokens against a sequence far longer than $R$. During prefill I forward the *exact* full-precision keys and values to the next layer's attention and only retain the quantized cache in memory, so the prompt encoding stays lossless and all quantization effects are confined to the decode path where the memory pressure actually bites.

The constants follow from the structure rather than being pulled from the air. The group size $G = 32$ trades off two competing pressures: smaller groups give tighter per-group ranges and lower error, but every group costs a stored $(\min, \text{scale})$ pair, and on a long sequence that metadata-per-element is what determines real compression — 32 is fine enough to bound outlier damage and coarse enough to amortize the metadata. The residual length $R = 128$ is a multiple of $G$ (four groups at a time) so the flushed key block is a whole number of groups, and it is large enough to protect a meaningful recent window while staying negligible against a long sequence; $R = 32$ would be the aggressive setting. The bit width is $B = 4$ here: the method spans $B = 2$ and $B = 4$, and 4-bit is the conservative choice for models that leave less slack — a multi-query-attention model like Falcon already has a single KV head and so a heavily compressed cache with little redundancy to spend on quantization noise. For accounting, a quantized token costs $B$ data bits per element and an FP residual token costs 16, so the average is $(\text{quant\_tokens}\cdot B + \text{residual}\cdot 16)/\text{seq\_len}$ with $\text{quant\_tokens} = \text{seq\_len} - \text{residual}$; the key residual is $\text{seq\_len} \bmod R$ and the value residual is $\min(R, \text{seq\_len})$. At a 4096-token reference with $R = 128$, $B = 4$, the key is exactly 4.0 bits ($4096 \bmod 128 = 0$) and the value is $(3968\cdot 4 + 128\cdot 16)/4096 = 4.375$ bits, averaging $(4.0 + 4.375)/2 = 4.1875$ bits per cached KV element.

For the tensor-replay harness I do not need packed integer storage or fused kernels — it scores quality — so I implement the fake-quant form: take the real key/value tensor of shape $[\text{batch}, \text{heads}, \text{seq\_len}, \text{head\_dim}]$, quantize-then-dequantize the grouped part along the right axis, leave the residual untouched, and return a tensor of the same shape equal to what the model would see after dequantization. The one primitive is group-wise asymmetric RTN over the last dimension; for the value I transpose so the last axis holds a token's features and group that, and for the key I transpose so the last axis becomes the token axis, group $G$ tokens of one channel, and transpose back, trimming to the largest multiple of $G$ so I never form a partial channel-group.

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
