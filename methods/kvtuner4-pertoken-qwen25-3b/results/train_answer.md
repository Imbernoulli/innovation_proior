In long-context, large-batch autoregressive decoding the key/value cache, not the weights, is the bottleneck: at every step the GPU streams the entire cache from main memory to SRAM while the compute units idle, and for a deep model over a long sequence the cache dwarfs the parameters. Shrinking it by quantizing to low bit-width is the most deployable fix, but the precise problem is sharper than "compress the cache." The only quantizer I can afford online is uniform round-to-nearest, because keys and values arrive one token at a time and an optimization-based post-training quantizer would be far too costly to run every decode step; over a group it is $Q(X) = \mathrm{round}((X - z)/s)$, $\hat X = Q(X)\cdot s + z$, with $z = \min(X)$ and $s = (\max(X) - \min(X))/(2^B - 1)$. At INT8 this is essentially lossless, at INT4 usually fine, but at INT2 — and on sensitive models such as Qwen2.5 already at INT4 *key* quantization — accuracy collapses. The naive plan, pick one target bit-width and apply it uniformly to every layer's keys and values, keeps hitting the same wall: the average bit-width looks great on paper while the generation is wrong in a way perplexity barely registers.

The failure mode is the whole clue. On a 15-shot GSM8K prompt, full precision and INT4 produce the identical correct derivation, "$20 - 4 - 4 = 12 \to 60\%$," but INT2 flips a single sign so the model writes "$20 + 4 + 4 = 28 \to 14\%$" and the answer is wrong. One flipped token poisons all downstream arithmetic, so this is not a smooth degradation I can average away; it is a discrete catastrophe triggered by a single error in the wrong place. The reason a tiny per-element error becomes a flipped token is that a transformer is sequential in two directions at once: the quantization error of layer $l$ feeds the input of layer $l+1$, and the error in the token generated at step $i$ feeds every layer at step $i+1$. So the per-token, per-layer error is not local — $e_i^l = f_e(e_i^{1:l-1}, e_{i-1}^{1:L}, \dots, e_1^{1:L})$ — and although a single token-layer error is negligible, accumulated over depth and over a long generation it crosses the threshold where the argmax of the next-token distribution flips. That two-dimensional error accumulation, not the per-element error, is the real adversary, and uniform quantization has nothing to say about *where* in the network the bits should go to stop it; it spends them flat. The fine-grained alternatives that identify critical tokens on the fly and bump their precision per step do place bits intelligently, but a per-token or per-page precision difference *inside* a layer cannot be fused with FlashAttention or a vLLM paged cache, and the online critical-token search adds branchy control flow that wrecks static-graph acceleration — the throughput the quantization was meant to buy gets eaten by the machinery that decides it. KIVI fixes the axis question (key per-channel, value per-token, with an fp16 residual window) but stays uniform precision across all layers; KVQuant reaches ~3 bits but with pre-RoPE, non-uniform, sparse-outlier machinery that needs bespoke kernels. None allocate the integer bit-width per layer, at zero online cost, in a form an existing stack runs unchanged.

I propose KVTuner: sensitivity-aware, layer-wise, mixed-precision KV cache quantization. Instead of one global bit-width, each transformer layer gets its own coarse-grained precision pair $(P_k^l, P_v^l)$ — coarse-grained meaning the whole low-bit KV inside a layer shares one pair, so the layer stays uniform precision and fuses cleanly with FlashAttention and vLLM — and the per-layer table is found by a multi-objective search ahead of time and loaded at inference with zero online overhead. Two facts make this the right shape. The first is that key precision matters more than value precision, and I can derive it rather than just measure it. Take one query $q$ attending over keys $K$ with $S$ tokens; the noise-free score is $a_i = \exp(qK_i)/\sum_j \exp(qK_j)$, and quantizing the keys adds $\Delta K \sim \mathcal{N}(0,\sigma^2)$ with $\sigma = (\max K - \min K)/(2^B-1)$ — note $\sigma$ scales geometrically in the bit cut, not linearly. The noisy score factors as $\hat a_i = \exp(qK_i)\exp(q\Delta K_i)/\sum_j \exp(qK_j)\exp(q\Delta K_j)$, which I rewrite as
$$\hat a_i = \frac{\exp(qK_i)}{\sum_j \exp(qK_j)\,\big(\exp(q\Delta K_j)/\exp(q\Delta K_i)\big)}.$$
This equals $a_i$ only when the ratio $\exp(q\Delta K_j)/\exp(q\Delta K_i)$ stops mattering. It is generically never identically one (the errors are independent draws), so the surviving case is a *dominating* key: if for all $j\neq i$, $\exp(qK_j)/\exp(qK_i)\approx 0$, then factoring $\exp(qK_i)$ out of the denominator instead leaves only the $j=i$ term, $\hat a_i \approx \exp(q\Delta K_i)/\exp(q\Delta K_i) = 1$, and every dominated token gets $\hat a_j = 0$. So the heads robust to low-precision key quantization are exactly the sparse, concentrated ones with a dominating key; the fragile ones are the spread-out, non-sparse retrieval heads where no key dominates and the noise $q\Delta K_j$ reshuffles the whole softmax. The lever against attention-distribution shift is therefore *key* precision in the fragile layers. Values behave oppositely for a separate reason: the output $o_i = \sum_j a_{ij} v_j$ is a weighted sum over a highly sparse attention pattern, so per-token value quantization confines each token's error to that token and butchering the near-zero-weight tokens barely moves the output, whereas per-channel value quantization would smear error across the important tokens (measured ~15x worse). Holding memory fixed, key error enters the softmax nonlinearly while value error only reweights the output after the distribution is set, so K4V2 beats K2V4 at equal budget ($e_o \approx 0.453$ vs $0.892$ on Llama-3.1-8B per-token-asym). That collapses the per-layer menu to a key-first ladder $\{\text{KV8}, \text{K8V4}, \text{KV4}, \text{K4V2}, \text{KV2}\}$ instead of the full nine-way $\{2,4,8\}\times\{2,4,8\}$.

The second fact is observed, not derived: layer-wise sensitivity is stable across input prompts, the same within a model family, and follows no clean depth heuristic — sensitive and insensitive layers interleave, and the most sensitive layer even moves when the quantization mode changes. Stability across prompts is what licenses everything: because sensitivity is an inherent property of the model, the per-layer precision can be decided once before deployment and loaded at inference with no runtime decision, sidestepping the online fine-grained methods entirely. That makes the assignment a discrete multi-objective optimization,
$$\min_P \big(f_m(P), f_a(P)\big)\quad\text{s.t.}\quad f_m(P)\le M,\;\; f_a(P)\le \Delta A,$$
with $P\in S^L$ giving each layer a pair, $f_m(P)=\mathrm{sum}(P)/(2L)$ the average equivalent bits over all keys and values, and $f_a(P)=A_{\mathrm{LLM}}(\mathrm{KV}_{\text{half}})-A_{\mathrm{LLM}}(\mathrm{KV}_P)$ the accuracy drop versus the fp16-KV model. The raw space $9^L$ is hopeless — $9^{32}\approx 3.4\times10^{30}$ — so I shrink it in two stages before searching. Stage one attacks the base: for each layer, plot the nine pairs in (equivalent bits, relative attention-output error $e_o$) and keep only the Pareto frontier; any pair dominated on both axes can never be the right local choice, so deleting it loses nothing under the local criterion, and most layers collapse to exactly the key-first five because value-first pairs like K2V4 and K4V8 are dominated. That takes $9^L\to{\sim}5^L$ ($5^{32}\approx 2.3\times10^{22}$), still hopeless. Stage two attacks the exponent: partition layers by their pruned candidate set (same qualitative response) and within each partition cluster by sensitivity vector with DBSCAN ($\mathrm{eps}=0.05$, $\mathrm{min\_samples}=2$, chosen so I need not pre-commit to a cluster count), then tie each cluster to one pair. Unlike the Pareto deletion this is not lossless — tying layers can exclude assignments where two clustered layers would want different pairs — but it is justified by the observed stability and similarity of sensitivity profiles, and it is where the real reduction lives: 28–64 layers fold to 4–8 groups, $5^L\to 5^G$ ($5^6=15625$), a space an evolutionary optimizer can actually cover. A black-box multi-objective search — a decomposition-based / NSGA-II-style sampler driven through Optuna — then maximizes accuracy on the first 200 GSM8K 4-shot prompts and minimizes equivalent bits under soft constraints around 4-bit and 6-bit. The one subtlety is calibration: since the danger is *accumulated* error, I deliberately use *dequantized* KV for the prefill self-attention so error propagates layer-to-layer, and calibrate on long math generations where a flipped intermediate token produces a large, measurable final-answer swing — amplifying the signal that separates good assignments from bad. This accumulate-through-prefill trick is a calibration device only; at run time the cached KV is quantized while the current prefill states pass through at full precision.

The deployable realization is the per-token mode, where both keys and values are quantized along the token dimension (axis 0) so a freshly generated token's row is quantized and appended without per-channel regrouping or special operators — the simplest streaming-friendly form. Per-token keys carry larger error than per-channel (keys have outlier channels that inflate a token-row scale), but that is exactly the cost the layer-wise search pays back by spending extra key bits where they are needed. The arithmetic is a signed-asymmetric round-to-nearest over a group; for $B$ bits I use $q_{\max}=2^{B-1}-1$ and $q_{\min}=-2^{B-1}$, so $q_{\max}-q_{\min}=2^B-1$ and the scale comes out consistent with the background formula:
$$\mathrm{scale}=\frac{\mathrm{clamp}\big(\max(g)-\min(g),\,10^{-5}\big)}{q_{\max}-q_{\min}}=\frac{\text{range}}{2^B-1},\qquad \mathrm{zeros}=\mathrm{round}\!\big(\min(g)/\mathrm{scale}\big)-q_{\min},$$
$$\mathrm{quant}=\mathrm{clamp}\big(\mathrm{round}(g/\mathrm{scale}-\mathrm{zeros}),\,q_{\min},\,q_{\max}\big),\qquad \mathrm{dequant}=(\mathrm{quant}+\mathrm{zeros})\cdot \mathrm{scale}.$$
The clamp at $10^{-5}$ is the divide-by-zero floor for a constant group; the zero-point maps the group minimum onto $q_{\min}$ and cancels on the way back out ($g/\mathrm{scale}-\mathrm{zeros}+\mathrm{zeros}$ times $\mathrm{scale}$ returns $g$), which is why I carry it: an asymmetric KV range uses the full signed grid instead of wasting codes, and the asymmetry is exactly why a symmetric $|\max|$-only scale would be worse. In per-token mode the group is the whole `head_dim` (one group per token row) and $\mathrm{residual\_length}=0$ — the fp16 residual sliding window is a per-channel streaming artifact (you need a block of tokens before a channel's statistics are known), so per-token needs none of it and every cached token is quantized at its layer's bits. For Qwen2.5-3B-Instruct (36 layers) in per-token mode the search returns a sparse, non-monotone-in-depth table — the very reason a depth heuristic fails: layer 0 gets K8V4, layers 18, 27, 29 get full K8V8, layers 10, 19, 24, 26, 33 get K4V4, and the other 27 layers stay K4V2. The bit sums are 160 key and 102 value over 36 layers, so the equivalent precision is $(160+102)/(2\cdot 36)=3.6389$ bits — under 4 even though the menu was built from a 4-bit budget, because key-first pruning let most layers drop their values to 2-bit. That dictionary, loaded into the quantizer, is the entire learned artifact and the entire online cost.

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
