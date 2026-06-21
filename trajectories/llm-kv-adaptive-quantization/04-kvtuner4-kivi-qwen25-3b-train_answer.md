The per-token layer-wise search did the two things I predicted and the one I feared. GSM8K jumped to 43.52 — a huge move off the floor's 31.84 and SQuat's 31.77 — because the search was calibrated to amplify the accumulation that flips an arithmetic token and put K8V8 on layers 18, 27, 29; HotpotQA rose to 39.47, the best of any rung; and the efficiency term climbed to 3.6389 bits, 4.40× compression. But the pure long-context retrieval workloads, which the GSM8K-calibrated search never optimized, fell off: passage retrieval dropped to 56.63, RepoBench to 43.74, and NIAH *collapsed* to 53.74 from the floor's 65.31. The NIAH cliff is the tell. A needle-in-a-haystack task lives or dies on the attention distribution finding the one relevant key across a long context, and per-token key quantization is the lossiest possible choice for keys — a per-token group is a slice across channels that swallows the key cache's persistent channel outliers, blowing up the scale and crushing the resolution of the normal channels that carry the retrieval signal. The per-token search bought GSM8K and compression by accepting larger key error everywhere, then leaned on the table to spend key bits back only where the *math* calibration said it mattered; retrieval needed key fidelity on layers the math calibration never lit up. The diagnosis is confirmed: keep the layer-wise mixed precision, but stop paying the per-token key penalty.

I propose KVTuner in KIVI mode: re-run the identical search machinery — the two-dimensional accumulation story, the key-first ladder, the offline prompt-independent table, the Pareto-plus-DBSCAN reduction of $9^L$ to $5^G$, the accumulation-amplified GSM8K oracle — but on top of **per-channel keys** instead of per-token keys. None of the search machinery was the problem; the problem was the quantization *mode* the table sits on. With keys grouped along the channel dimension (the floor's axis), the channel outliers are confined to their own groups, so the layer-wise table spends its bits on a key representation that is already several times more faithful per bit, and I expect the retrieval workloads to come back without surrendering the GSM8K and compression wins. This is the natural last move: the strongest within-layer idea (per-channel keys, from the floor) fused with the strongest across-layer idea (the searched mixed-precision table, from the per-token rung).

Per-channel keys are the right axis for the same structural reason as in the floor. The key cache $X \cdot W_K$ is an activation, and trained activations carry persistent large-magnitude outliers in a small number of fixed channels — the same channel is huge across essentially every token. Per-token quantization shares one scale across all channels of a token, so one giant outlier channel blows up that token's range and every well-behaved channel is quantized against a range it does not need; per-channel quantization shares the scale within a channel across a window of tokens, so the outlier channel gets its own wide range and the normal channels get tight ranges that fit them. Measured at low bits the relative key error is several times smaller per-channel, and on the attention *score* $\text{softmax}(qK^\top)$ — which is what the model consumes and exactly what NIAH depends on — the gap is roughly fivefold. That fivefold attention-score fidelity is precisely what per-token keys threw away. Values stay per-token for the same reason the floor kept them there: the value is a mixer $o = \sum_j a_j v_j$, attention is sparse, per-token confines each token's error to its row where the unattended rows are weighted to $\approx 0$, while per-channel value quant smears unimportant-token error through shared channels into the important rows.

The streaming consequence the per-token rung got to ignore returns and is load-bearing. Per-token value quant streams trivially, but per-channel key quant does *not* — a channel's scale needs a whole window of tokens before it is defined and tokens arrive one at a time — so I am back to the grouped-plus-residual structure: quantize the key cache in complete groups of $G = 32$ tokens along the token axis, keep the most recent partial group in full precision, flush it when it fills. With the residual length tied to the group, the kept FP count is $\text{seq\_len} \bmod 32$ — a small sliding window aligned to the group boundary. That FP residual window, which the per-token mode set to zero, comes back and is load-bearing on exactly the tasks that broke: the freshest tokens are the locally relevant ones, and keeping them exact protects the local reasoning the per-token mode left fully quantized. So the mode switch is not just an axis flag — it carries the per-channel grouping, the transpose so the token axis becomes the group axis (`axis=1`), and the residual sliding window.

The mechanistic reason the *key* is the layer-allocation lever survives the mode change. Take one query $q$ and key cache $K$; the clean score is $a_i = \exp(qK_i)/\sum_j \exp(qK_j)$. Quantize with $\Delta K \sim \mathcal{N}(0, \sigma^2)$, $\sigma = (\max K - \min K)/(2^B - 1)$; the corrupted score $\hat{a}_i = \exp(q(K_i + \Delta K_i))/\sum_j \exp(q(K_j + \Delta K_j))$, factoring the exponential and dividing by $\exp(q\Delta K_i)$ gives $\hat{a}_i = \exp(qK_i)/[\sum_j \exp(qK_j) \cdot (\exp(q\Delta K_j)/\exp(q\Delta K_i))]$. If a single key dominates the softmax, $\hat{a}_i \approx \exp(q\Delta K_i)/\exp(q\Delta K_i) = 1$ and the rest 0 — the distribution survives any key error. So sparse, concentrated heads are robust; spread-out retrieval heads where no key dominates are fragile, the error ratios do not cancel, and the softmax reshuffles. NIAH is the purest such retrieval head — one needle key among thousands, no domination — which is precisely why per-token key error wrecked it. The corruption rides entirely on $q\Delta K$, so the remedy is key precision in fragile layers; the value never enters the softmax ($o = \sum_j a_j v_j$, value error applied after the weights are set), so value bits cannot repair a shifted distribution. That is why the ladder is key-first, and per-channel keys are the same lever from the other side: they shrink $\sigma$ for the key by confining outliers, before the table spends a single bit.

So the offline pipeline is identical to the per-token rung's, just run in KIVI mode: measure per-layer attention-output error for the nine pairs under per-channel key quantization, Pareto-prune each layer to its key-first candidate set, partition by candidate set and DBSCAN-cluster by sensitivity into groups, run the multi-objective black-box search over the per-group pairs against the accumulation-amplified GSM8K oracle and the memory objective, read off the table. The pruning sometimes keeps different exceptions in this mode — once per-channel makes the key cheap and accurate, a few layers can actually prefer lowering the key further and raising the value (K2V4, K4V8) where per-token never would — and the per-layer pruning respects that. The arithmetic underneath is the signed-asymmetric grid again, identical to the per-token rung: $q_{\max} = 2^{B-1} - 1$, $q_{\min} = -2^{B-1}$, $\text{scale} = \text{clamp}(\max - \min, 10^{-5})/(2^B - 1)$, $\text{zeros} = \text{round}(\min/\text{scale}) - q_{\min}$, $\text{quant} = \text{clamp}(\text{round}(g/\text{scale} - \text{zeros}), q_{\min}, q_{\max})$, $\text{dequant} = (\text{quant} + \text{zeros})\cdot\text{scale}$. The only differences from the per-token form are the per-key axis (transpose so head_dim becomes the row index and the group runs over a token window, `axis=1`), the group size ($32$, not the whole head_dim), and the residual ($32$, the sliding window, not $0$).

The searched table for Qwen2.5-3B in KIVI mode at the 4-bit family is dominated by K4V2 — most layers run a 4-bit per-channel key and a 2-bit value — with the exceptions the search insists on: the earliest layers 0 and 1 get K4V8, two early layers 2 and 4 drop to K2V4, a couple of layers 12 and 28 go to K2V2, and the last two layers 34 and 35 get K4V4. The shape is what the lemma predicted — bits concentrated on the key, value pushed to 2-bit wherever the heads are sparse enough to tolerate it, with a handful of sensitive layers protected — but the specific table differs from the per-token rung's because per-channel keys change which pairs are Pareto-optimal per layer. Summing gives key total 136 and value total 92 over 36 layers, so the nominal precision is $(136 + 92)/(2\cdot36) \approx 3.17$ bits per element before the residual overhead — even lower than the per-token rung's 3.6389, because per-channel keys are accurate enough at 4 bits that the search could push more layers' values to 2-bit without breaking accumulation. The harness's `estimate_bits` folds in the $\text{seq\_len} \bmod 32$ residual at the 4096 reference span, landing the reported effective bits at 3.166667 — the table's nominal 3.17 with the residual at a 4096-span multiple of 32 contributing nothing, $\approx 5.05\times$ compression, the highest of any rung. No prefill observer is needed (the table is static), so this rung returns `needs_prefill_qkv_observer() -> False`; the intelligence is in the offline table plus the per-channel grouping.

The bar this endpoint must clear, against the three measured rungs below it: the efficiency win should be the largest, 3.166667 effective bits at $\approx 5.05\times$, well past the per-token rung's 4.40×, the floor's 3.82×, and SQuat's 4.0×. On quality the claim is that per-channel keys recover the retrieval workloads the per-token table sacrificed without giving back GSM8K — NIAH should climb from 53.74 toward the floor's per-channel level near 65.31, RepoBench toward the high-40s, passage retrieval toward the high-50s, while GSM8K holds in the mid-30s (above the floor's 31.84 and SQuat's 31.77, below the per-token 43.52, because the more aggressive 3.17-bit budget gives back a little). The one genuine uncertainty is whether the lower budget costs HotpotQA relative to the per-token rung's 39.47. If NIAH, RepoBench, and passage retrieval all recover toward or past the floor while GSM8K stays in the mid-30s and compression jumps to $\approx 5\times$, then fusing per-channel keys with the searched layer-wise table is the strongest point on the ladder — the across-layer and within-layer ideas finally compounding instead of trading off.

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
