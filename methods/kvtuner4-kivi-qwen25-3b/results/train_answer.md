During autoregressive decoding a transformer keeps every past key and value so each new token can attend to the whole prefix without recomputing it, and that cache grows linearly in batch size times sequence length. In the long-context, large-batch regime it is the cache, not the weights, that pins decoding to memory bandwidth — for a 175B model at batch 512, prompt 512, and 32 generated tokens the KV cache exceeds a terabyte, several times the weight footprint, and each step is dominated by streaming it. The obvious lever is to store keys and values in fewer bits, but the cache is a *streaming* structure: a new key row and value row are appended every step, so anything that needs the whole tensor at once to solve a little optimization (a GPTQ-style fit) is dead on arrival. The only primitive cheap enough to run online on a growing tensor is group-wise round-to-nearest with a per-group scale and zero-point: take a group, read its min and max, map onto an integer grid, round, store integers plus scale and offset, dequantize on the way into attention. For a $B$-bit group, $z=\min(X)$ and $s=(\max X-\min X)/(2^{B}-1)$, so each dropped bit roughly doubles the noise. That part is forced; the freedom is *where* to spend bits.

What makes the problem sharp is that low-bit KV does not fail gently. At 8 bits quantization is usually benign, at 4 bits often acceptable, and at 2 bits the model stops degrading smoothly and instead *flips* a critical token: a fifteen-shot GSM8K prompt that at full precision computes $20-4-4=12$ is generated under uniform 2-bit KV as $20+4+4=28$, and the entire remaining derivation is wrong, while the same prompt at 4-bit reproduces the full-precision answer token for token. So the damage is a rare catastrophic flip, not uniform blurring, which means the error has to be understood as something that *accumulates*: the quantized cache at layer $l$, step $i$ is the input to attention at every later layer and every later step, so its error depends on every earlier layer's error and every earlier step's error — a two-dimensional cascade. A single element error is negligible; the compounding over a long generation flips a token. The existing low-bit quantizers all miss the consequence of this. Uniform round-to-nearest applies one precision and grouping mode to every layer, overpaying for robust layers and breaking the few that need more. KIVI fixes the right *axes* (keys per-channel, values per-token, group 32, fp16 residual window) and is tuning-free, but its precision is uniform across layers. KVQuant reaches very low precision with a non-uniform datatype, pre-RoPE key quantization, and a sparse outlier path, but at the cost of custom kernels and offline calibration that is hard to fuse with paged caches. The fine-grained online schemes (QAQ, MiKV, ZipCache) adapt per token at decode time, but an intra-layer precision *difference* cannot be fused with fused-attention or a paged cache, and the online critical-token logic adds per-step control flow that breaks static-graph acceleration. They buy accuracy with a deployment tax. A uniform budget pretends every layer contributes equally to the cascade; the layer-wise measurements say the opposite.

I propose KVTuner: a sensitivity-aware, layer-wise, *mixed-precision* KV-cache quantizer that keeps each layer's cache a single uniform format (kernel-friendly) but lets different layers use different precision pairs, chosen by their measured sensitivity offline so there is zero online decision cost. Three discovered facts fix its shape. First the axis recipe. Real key tensors carry persistent channel outliers — the same fixed channels are large across all tokens — and per-token quantization shares one scale across all channels of a token, so a single giant channel inflates the range and starves the well-behaved channels of resolution; per-channel quantization gives the outlier its own column and confines its error there, measuring about $5\times$ smaller attention-score error. Values show no such outlier structure, so per-channel and per-token give similar *reconstruction* error — but values never enter attention raw, only as the output $o_i=\sum_j a_{ij}V_j$, and attention is highly sparse (around 84%), so a few rows carry the mass. Per-token value quantization keeps each row's error local, multiplied by that token's near-zero weight on the unimportant rows; per-channel value quantization smears error across the token dimension and lands it on exactly the high-weight rows, giving about $15\times$ worse output error. So keys go per-channel, values per-token. Per-token is trivially streaming; per-channel is not, because a channel scale needs a whole window of tokens, so the key cache is split into a grouped part already quantized in complete groups of $G$ and a residual part of the most recent $r$ tokens kept in full precision (with $l-r$ divisible by $G$); a newly arrived key joins the residual, and once it fills a group it is quantized and concatenated. That fp16 residual sliding window is load-bearing on hard tasks because the freshly generated tokens being reasoned over stay exact.

Second, the key cache matters more than the value cache, and not just empirically (at equal memory, K4V2 gives relative output error $0.453$ versus $0.892$ for K2V4 on Llama-3.1-8B). The mechanism is a small lemma: only attention heads with sparse, concentrated patterns are consistently robust to low-precision key quantization. For a query $q\in\mathbb{R}^{1\times D}$ and key cache $K\in\mathbb{R}^{D\times S}$ the clean score is $a_i=\exp(qK_i)/\sum_j\exp(qK_j)$. Asymmetric uniform key error $\Delta K\sim\mathcal{N}(0,\sigma^2)$ with $\sigma=(\max K-\min K)/(2^{B}-1)$ grows geometrically as $B$ drops, and because the exponential factorizes,
$$\hat a_i=\frac{\exp(qK_i)\,\exp(q\Delta K_i)}{\sum_j \exp(qK_j)\,\exp(q\Delta K_j)}=\frac{\exp(qK_i)}{\sum_j \exp(qK_j)\,\big[\exp(q\Delta K_j)/\exp(q\Delta K_i)\big]}.$$
The distribution is exactly unchanged only in the degenerate case where every $q\Delta K_i=q\Delta K_j$, generically false for independent errors. The useful case is approximate: if one key $i$ dominates, $\exp(qK_i)\gg\exp(qK_j)$ for $j\neq i$, then dividing numerator and denominator by $\exp(qK_i)$,
$$\hat a_i=\frac{\exp(q\Delta K_i)}{\sum_j \exp(q\Delta K_j)\,[\exp(qK_j)/\exp(qK_i)]}\approx\frac{\exp(q\Delta K_i)}{\exp(q\Delta K_i)}=1,$$
with the dominated $\hat a_j\approx 0$ — the distribution barely moves no matter how large the key error, because the dominating token's noise cancels against itself. A sparse head is robust; a spread retrieval head with no dominating term shifts and can re-order which keys look critical. The corruption rides entirely on $q\Delta K$, so the remedy is to raise *key* precision in sensitive layers; value precision cannot help, because value error $\sum_j a_{ij}\Delta V_j$ is applied after the weights are chosen and cannot repair a distribution that has already shifted.

Third, layer sensitivity is real, inherent, and prompt-independent: the layer-wise error profile is a property of the trained model, stable across prompts, with no depth shortcut — sensitive and insensitive layers interleave, and the most sensitive layer even moves (e.g. from 29 under per-token-asym to 11 and 13 in the per-channel-key mode) when the quantization mode changes. So sensitivity must be *measured*, once, offline. Putting these together, finding the per-layer table is a discrete multi-objective black-box optimization. A configuration $P\in\mathcal{S}^L$ assigns each layer a pair $(P_k^l,P_v^l)$ from $\{2,4,8\}^2$, and I minimize memory and accuracy loss jointly,
$$\min_P\big(f_m(P),\,f_a(P)\big)\quad\text{s.t.}\quad f_m(P)\le M,\ f_a(P)\le\Delta A,$$
with $f_m(P)=\Sigma(P)/(2L)$ the average equivalent bits and $f_a(P)=A_{\text{LLM}}(\mathrm{KV}_{\text{half}})-A_{\text{LLM}}(\mathrm{KV}_P)$ the real model accuracy drop versus FP16 KV — a non-differentiable, expensive black box that captures the nonlinear cascade. The naive space $9^{L}$ ($\approx 3.4\times10^{30}$ at $L=32$) is hopeless, so I shrink it using the structure already found, attacking both factors. Intra-layer Pareto pruning keeps, per layer, only the frontier in (equivalent bits, output error $e_o$); because the key dominates, most layers collapse to the key-first ladder $\{\text{KV8},\text{K8V4},\text{KV4},\text{K4V2},\text{KV2}\}$ (a pair like K4V8 is dominated by K8V4 at equal bits), taking $9^{L}\to 5^{L}$, with the genuine per-layer exceptions (some layers prefer K4V8 or K2V4 once per-channel makes the key cheap) kept. Inter-layer clustering then partitions layers by their pruned candidate set and runs DBSCAN ($\mathrm{eps}=0.05$, $\mathrm{min\_samples}=2$) on the $e_o$ sensitivity vectors, since I do not know the number of distinct regimes in advance and a uniquely sensitive layer should be allowed to sit alone; this collapses $L$ (28–64) to $G$ (4–8) groups sharing one pair, reaching $\mathcal{S}_p^G$ ($\approx 5^6=15625$). A black-box multi-objective evolutionary search (MOEA/D or NSGA-style) then optimizes over the per-group pairs, maximizing accuracy on the first 200 GSM8K few-shot prompts and minimizing equivalent bits over a couple hundred iterations under soft constraints near 4 and 6 bits. The calibration design is deliberate: use the *dequantized* KV in prefill self-attention so the cross-layer accumulation is switched on, and a hard math dataset so a flipped intermediate token shows up in the final answer — only an accumulation-amplified oracle can distinguish a pair that is quietly fine from one that quietly flips tokens.

For the shipped case — KIVI mode (key per-channel, value per-token, group 32, residual 32) on Qwen2.5-3B-Instruct, searched in the 4-bit family — the resulting 36-layer table is key-dominant: K4V8 in layers 0 and 1, K2V4 in 2 and 4, K2V2 in 12 and 28, K4V4 in 34 and 35, and K4V2 in the other 28 layers. Key bits sum to 136 and value bits to 92, so the preset equivalent precision is $(136+92)/(2\cdot 36)\approx 3.17$ bits per element before the residual overhead — well under 4, while staying nearly lossless on the math task because the bits land where accumulation would otherwise flip a token ("KVTuner4" names the 4-bit search family, not a realized exact 4-bit width). The table feeds a signed-asymmetric round-to-nearest quantizer: for a group $g$, the signed grid is $q_{\max}=2^{B-1}-1$, $q_{\min}=-2^{B-1}$ (span $2^{B}-1$), the scale is $\mathrm{clamp}(\max g-\min g,\,10^{-5})/(q_{\max}-q_{\min})$, the zero-point is $\mathrm{round}(\min g/\text{scale})-q_{\min}$, the quantized value is $\mathrm{clamp}(\mathrm{round}(g/\text{scale}-\text{zeros}),q_{\min},q_{\max})$, and dequantization is $(\text{quant}+\text{zeros})\cdot\text{scale}$, with the $10^{-5}$ floor stopping a constant group from giving a zero scale. The axis is handled by transposing before grouping (per-channel transposes so head\_dim are the rows and groups run over a token window; per-token leaves it), the residual keeps the most recent $\text{seq\_len}\bmod 32$ tokens in FP, and the effective bits per element is the token-weighted average $(\text{quant\_tokens}\cdot B+\text{residual\_tokens}\cdot 16)/\text{seq\_len}$. The offline search produces `_PRESET`; the inference-time quantizer below is the entire online cost.

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
