SQuat did exactly what its bound promised and nothing more. At 4 bits and 4× compression it landed 36.53 on HotpotQA, 59.96 on passage retrieval, 46.48 on RepoBench, 65.31 on NIAH, and 31.77 on GSM8K — essentially tracking the KIVI floor on quality while edging efficiency from 3.82× to 4.0×. The query-subspace orthogonality protected the attention distribution within each layer at the same bits, but GSM8K refused to move (31.77 vs 31.84) and NIAH was identical to the floor. The confirmed lesson is that orthogonality fixes *which directions* I quantize within a layer, but both rungs so far apply one global precision to all 36 layers, and the failure that actually pins quality down — the long greedy GSM8K chain — is an accumulation problem no within-layer trick can reach. The next thing to break is that uniformity across layers.

The GSM8K failure mode is the whole clue. At full precision and at 4-bit KV the model writes the identical correct chain; push the cache to 2-bit uniformly and the first sentences are nearly identical, then a single token flips — a minus becomes a plus — and now it writes "20 + 4 + 4 = 28 ... 14%," final answer wrong. One flipped token poisons all downstream arithmetic; this is not smooth degradation I can average away but a discrete catastrophe triggered by one error at the wrong place. The reason a tiny quantization error becomes a flipped token is that the transformer is sequential in *two* directions at once: the quantized output of layer $l$ at step $i$ is the input of layer $l+1$ at step $i$, and the token generated at step $i$ is the input of every layer at step $i+1$. So the error at layer $l$, step $i$ is a function of every earlier layer this step and every layer at every earlier step, $e_i^l = f(e_i^{1:l-1}, e_{i-1}^{1:L}, \dots)$. A single token-and-layer error is negligible; accumulated over depth and over a long generation it crosses the threshold where the next-token argmax flips. That is why SQuat could not save GSM8K with a within-layer fix at uniform bits — it had no way to put *more* bits on the specific layers where the accumulation compounds, because a uniform policy does not know which layers those are. The lever has to be a bit-*allocation* across layers.

I propose KVTuner in per-token mode: allocate a per-layer precision pair $(P_k, P_v)$, searched offline, and read it off as a static table at inference. Two derivations set the shape of the allocation. First, key-versus-value. Take one query $q$ attending over a key cache $K$; the clean score on token $i$ is $a_i = \exp(qK_i)/\sum_j \exp(qK_j)$. Quantize the keys with asymmetric uniform error $\Delta K \sim \mathcal{N}(0, \sigma^2)$, $\sigma = (\max K - \min K)/(2^B - 1)$ growing geometrically as $B$ drops. The corrupted score is $\hat{a}_i = \exp(q(K_i + \Delta K_i))/\sum_j \exp(q(K_j + \Delta K_j))$; the exponential of a sum factors, and dividing top and bottom by $\exp(q\Delta K_i)$ gives $\hat{a}_i = \exp(qK_i)/[\sum_j \exp(qK_j) \cdot (\exp(q\Delta K_j)/\exp(q\Delta K_i))]$. If a single key $i$ dominates the softmax, pulling $\exp(qK_i)$ out instead gives $\hat{a}_i \approx \exp(q\Delta K_i)/\exp(q\Delta K_i) = 1$ and the dominated tokens get 0 — the distribution survives. So the heads *robust* to low-bit keys are exactly the sparse, concentrated ones with a dominating key; the fragile ones are the spread-out retrieval heads where no key dominates and the noise reshuffles the whole softmax. The corruption rides on $q\Delta K$, so the lever is *key* precision in fragile layers — the same term SQuat attacked by steering the residual orthogonal to $q$, here attacked by raising the bits. And value precision cannot repair it: the output $o = \sum_j a_j v_j$ applies value error only *after* the distribution is set. Measurements agree — K4V2 attention-output error $\approx 0.45$ versus K2V4 $\approx 0.89$ — so the candidate pairs collapse to a *key-first ladder* {KV8, K8V4, KV4, K4V2, KV2} with key $\ge$ value, rather than the full nine-way $\{2,4,8\}^2$.

Second, the layer axis, which I cannot derive but take as an observed property: the layer-wise sensitivity profile is *stable across prompts* — a layer sensitive on one GSM8K prompt is sensitive on the others — and there is no clean depth heuristic, sensitive and insensitive layers are interleaved. Because sensitivity is inherent rather than prompt-dependent, I can decide the per-layer precision *once, offline*, and load it at inference with zero runtime cost. That is exactly why I do *not* reach for fine-grained online methods that bump precision per token: an intra-layer per-token precision difference cannot fuse with FlashAttention or a paged cache, and the online critical-token search adds branchy control flow that wrecks static-graph acceleration — the throughput I am buying with quantization gets eaten by the machinery that decides it. I want every low-bit token in a layer to share one precision pair (coarse-grained, fuses cleanly) but let different layers use different pairs, chosen by measured sensitivity, decided ahead of time.

The decision is a two-objective optimization $\min_P (f_m(P), f_a(P))$ over assignments $P \in \mathcal{S}^L$, where $f_m(P) = \Sigma(P)/(2L)$ is the average equivalent bits and $f_a(P) = A(\text{KV}_{\text{half}}) - A(\text{KV}_P)$ is the accuracy drop, a non-differentiable expensive black box I drive with an evolutionary search using model accuracy as the oracle. But the space is $9^L \approx 3.4\times10^{30}$ for 32 layers, which no search touches, so I shrink it before searching by attacking the two factors separately. The base $9$: for a single layer plot the nine pairs in (equivalent bits, attention-output error) and keep only the Pareto frontier — by the key-first analysis the dominated pairs are exactly the value-heavy ones (K2V4, K4V8), so most layers collapse to the five-pair ladder, $9^L \to \sim 5^L$. The exponent $L$: collapse layers into groups that share a decision — partition layers by their pruned candidate set, then within each partition cluster by sensitivity (the vector of attention-output errors) using DBSCAN with a small radius, so I need not pre-commit a cluster count and outlier layers can sit alone. A model's 28–64 layers fold to $\sim$4–8 groups, $5^L \to 5^G \approx 5^6 \approx 15625$, a space an evolutionary optimizer can cover.

One subtlety in how I *score* an assignment, because the danger I am fighting is accumulation: if I calibrate with isolated, non-accumulating per-layer error I under-weight exactly the layers whose errors compound. So I deliberately let the error accumulate during calibration — use the *dequantized* KV for prefill self-attention so error propagates layer-to-layer, and calibrate on long hard generations (math reasoning) where a small error flips an intermediate token and produces a large measurable final-answer swing. Concretely, maximize accuracy on a couple hundred GSM8K few-shot prompts as one objective, minimize equivalent bits as the other, over the reduced grouped space. The output is a table of per-layer $(P_k, P_v)$, and that table is the entire online cost.

For this deployable rung on Qwen2.5-3B both keys and values are quantized along the token dimension — per-token mode, every token's row getting its own scale and zero-point — because that drops straight into a streaming cache with no per-channel regrouping. The cost of per-token keys versus the floor's per-channel keys is exactly the larger key error I derived, but that is the cost the layer-wise search is built to pay back by spending extra key bits on the layers where per-token key error would otherwise flip tokens. The arithmetic uses a *signed* asymmetric grid: $q_{\max} = 2^{B-1} - 1$, $q_{\min} = -2^{B-1}$, so $q_{\max} - q_{\min} = 2^B - 1$ and $\text{scale} = \text{clamp}(\max - \min, 10^{-5})/(2^B - 1)$, $\text{zeros} = \text{round}(\min/\text{scale}) - q_{\min}$, $\text{quant} = \text{clamp}(\text{round}(g/\text{scale} - \text{zeros}), q_{\min}, q_{\max})$, $\text{dequant} = (\text{quant} + \text{zeros})\cdot\text{scale}$, the zero-point cancelling on the way out. In per-token mode the group is the whole `head_dim` (one group per token row, `group_size = -1`) and `residual_length = 0` — the FP sliding window was a *per-channel* streaming artifact, since you need a block of tokens before a channel's stats are known, and per-token needs none of it, so every cached token is quantized at its layer's bits and the bit accounting is trivially clean: `estimate_bits` just reports the layer's bit count.

The searched table on Qwen2.5-3B at the $\approx 4$-bit budget is sparse and non-monotone in depth, which is the whole reason I needed the search and could not use a depth heuristic. Most layers come back K4V2 (the key-first workhorse); layer 0 gets extra key precision K8V4; the layers the search flags as carrying the most accumulation — 18, 27, 29 — get full K8V8; a scattering — 10, 19, 24, 26, 33 — get K4V4; everything else K4V2. Tallying gives 160 key bits and 102 value bits over 36 layers, so the equivalent precision is $(160 + 102)/(2\cdot36) = 3.6389$ bits — *under* 4 even though the menu was built from a 4-bit budget, because the key-first pruning let most layers drop values to 2-bit, $\approx 4.40\times$ compression above the floor's 3.82× and SQuat's 4.0×. Reading that dictionary off is the entire learned artifact; no prefill observer is needed (the policy is static), so unlike SQuat this rung returns `needs_prefill_qkv_observer() -> False` — the intelligence moved offline into the table, not a runtime hook.

The bet is sharp and asymmetric. The search was calibrated to amplify the GSM8K accumulation, so GSM8K is where I expect the biggest move, well above the floor's 31.84 and SQuat's 31.77, because the search put K8V8 on exactly the layers where the chain flips; HotpotQA should hold or improve; the efficiency half should rise on every workload. The risk sits on the workloads the search did *not* calibrate against — passage retrieval and NIAH are pure long-context retrieval, and per-token keys are inherently lossier than per-channel keys, so if the 3.6389-bit table starved a layer those tasks needed, passage retrieval and NIAH could dip below the floor even as GSM8K and efficiency climb. If that is what I see — GSM8K and compression up, NIAH and passage retrieval down because per-token keys plus an aggressive sub-4-bit table cost the retrieval workloads — then the diagnosis for the strongest rung is already written: keep the layer-wise search, but pay back the per-token key penalty by running the search in the *KIVI per-channel key mode* so retrieval recovers without giving up the mixed-precision win.

```python
class AdaptiveKVQuantizer:
    """KVTuner FlexibleVanillaQuantizedCache with official per-token preset."""

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

    def _signed_asymmetric(self, tensor: torch.Tensor, bits: int, axis: int, group_size: int, residual_length: int) -> tuple[torch.Tensor, float]:
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
        return self._signed_asymmetric(key_states, self._PRESET[layer_id]["key"], axis=0, group_size=-1, residual_length=0)

    def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._signed_asymmetric(value_states, self._PRESET[layer_id]["value"], axis=0, group_size=-1, residual_length=0)

    def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
        return float(self._PRESET[layer_id][kv_kind])
```
