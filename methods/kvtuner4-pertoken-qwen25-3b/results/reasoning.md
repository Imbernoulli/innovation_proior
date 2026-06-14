Let me start from what actually breaks when I quantize the KV cache. In long-context, large-batch decoding the cache is the bottleneck — at every step the GPU streams the whole thing from main memory while the compute cores idle, and for a big model with a long sequence the cache dwarfs the weights. So I want it small. The only quantizer I can afford online is round-to-nearest, because keys and values arrive one token at a time and an optimization-based post-training quantizer would be far too expensive to run every decode step. Round-to-nearest over a group is `Q(X) = round((X - z)/s)`, dequantize `X_hat = Q(X)·s + z`, with `z = min(X)` and `s = (max(X) - min(X))/(2^B - 1)`. At INT8 this is essentially free of error; at INT4 it's usually fine; at INT2 it falls apart. The naive plan is: pick a target bit-width, apply it uniformly to every layer's keys and values, done. And I keep hitting the same wall — the average bit-width on paper looks great, but the *generation* is wrong in a way perplexity barely registers.

Let me look hard at exactly how it goes wrong, because the failure mode is the whole clue. Take a 15-shot GSM8K prompt on a chat model. At full precision and at INT4 the model writes the identical correct chain: "20 - 4 - 4 = 12 ... 60%." At INT2 the first few sentences are nearly identical, and then a single token flips — a minus sign becomes a plus — and now it writes "20 + 4 + 4 = 28 ... 28/20 = 14%," and the final answer is just wrong. One flipped token poisons the entire downstream arithmetic. So this isn't a smooth degradation I can average away; it's a discrete catastrophe triggered by a single error at the wrong place. Why does a tiny quantization error in one tensor become a flipped token? Because a transformer is sequential in *two* directions at once. The output of layer `l`, with its quantization error, is the input of layer `l+1`; and the token I generate at step `i`, with its error, becomes the input of every layer at step `i+1`. So the error at layer `l`, step `i` isn't local — it's a function of every earlier layer at this step and every layer at every earlier step: `e_i^l = f_e(e_i^{1:l-1}, e_{i-1}^{1:L}, ..., e_1^{1:L})`. A single token-layer error is negligible; accumulated over depth and over a long generation it crosses the threshold where the argmax of the next-token distribution flips. That's the real adversary: two-dimensional error accumulation, not the per-element error. Uniform low-bit quantization has nothing to say about *where* in the network the bits should go to stop accumulation; it spends them flat.

So the question sharpens: if I have a fixed average-bit budget, where do I spend the precision? Two axes to think about — key versus value, and which layers. Let me take key-versus-value first, because there's a clean asymmetry I can actually derive rather than just measure. Picture one query token `q` attending over a key cache `K` with `S` tokens. The error-free score on token `i` is `a_i = exp(q K_i) / sum_j exp(q K_j)`. Now quantize the keys; the asymmetric uniform quantization error per element is roughly Gaussian, `ΔK ~ N(0, σ²)` with `σ = (max K - min K)/(2^B - 1)`. Notice that already — `σ` scales like `1/(2^B - 1)`, so dropping `B` blows the noise up roughly geometrically in the bit cut, not linearly. With key error the score becomes

  `â_i = exp(q(K_i + ΔK_i)) / sum_j exp(q(K_j + ΔK_j))`.

The exponential of a sum factors, `exp(q(K_i + ΔK_i)) = exp(q K_i) exp(q ΔK_i)`, so

  `â_i = exp(q K_i) exp(q ΔK_i) / sum_j exp(q K_j) exp(q ΔK_j)`.

I want to know when `â_i = a_i` — when does the noise leave the attention distribution unchanged? Divide numerator and denominator by `exp(q ΔK_i)`:

  `â_i = exp(q K_i) / [ sum_j exp(q K_j) · ( exp(q ΔK_j) / exp(q ΔK_i) ) ]`.

Compare that to `a_i = exp(q K_i) / sum_j exp(q K_j)`. They're equal exactly when the bracketed denominator equals `sum_j exp(q K_j)`, i.e. when the factor `exp(q ΔK_j)/exp(q ΔK_i)` doesn't matter. Two ways that can happen. One: that ratio is identically 1, meaning `q ΔK_i = q ΔK_j` for every pair — every key's error projects onto `q` by the same amount. The errors are independent draws, so generically this never holds; I can throw it out. Two — and this is the interesting one — there's a *dominating* key. Suppose token `i` is the one the query overwhelmingly attends to: for all `j ≠ i`, `exp(q K_i) >> exp(q K_j)`, equivalently `exp(q K_j)/exp(q K_i) ≈ 0`. Let me rewrite `â_i` by pulling `exp(q K_i)` out of the denominator instead, factoring it from each term:

  `â_i = exp(q ΔK_i) / [ sum_j exp(q ΔK_j) · ( exp(q K_j) / exp(q K_i) ) ]`.

Every term in that sum with `j ≠ i` is multiplied by `exp(q K_j)/exp(q K_i) ≈ 0`, so only the `j = i` term survives, and it's `exp(q ΔK_i) · 1`. So

  `â_i ≈ exp(q ΔK_i) / exp(q ΔK_i) = 1`,

and every dominated token gets `â_j = 0`. The distribution is unchanged. So the heads that are *robust* to low-precision key quantization are exactly the ones with a sparse, concentrated attention pattern — a few dominating keys. The ones that are fragile are the spread-out, non-sparse retrieval heads, where no single key dominates and the noise `q ΔK_j` reshuffles the whole softmax. That's a real result, not a hunch: only sparse/concentrated heads are consistently robust to low-bit key quantization. And it tells me the concrete lever to fight attention shift — raise the *key* precision (shrink `q ΔK` through a smaller `σ`) in the fragile layers, rather than touching the values.

But wait — that argument was entirely about keys. Is key precision actually more valuable than value precision, or did I just analyze the side I happened to start with? Let me think about what value quantization even does to the output. The attention output is `o_i = sum_j a_{ij} v_j` — a weighted sum of value rows, weights being the attention scores. Two things differ between key and value here. First, the structure of the noise: it's documented that the key cache has a few *channels* with persistently huge magnitude — the same outlier-channel phenomenon known from LLM activations generally — while the value cache shows no such channel structure. A wide-ranging outlier channel is poison for a *per-token* scale, because one fat channel inflates the scale of the entire token row and crushes the resolution of all the other channels in that row; a *per-channel* scale would confine that outlier to its own column. So keys "want" per-channel and the per-token key error is large (measured ~5x the per-channel attention-score error on Llama-2-13B). Second, and this is why values behave oppositely: attention is highly sparse — a handful of tokens carry almost all the weight. So `o_i` is effectively the combination of a few important value rows. If I quantize values *per-token*, each token's error stays inside that token, so butchering the unimportant tokens — the ones with near-zero `a_{ij}` — barely moves the output, while a *per-channel* value scale would smear error across all tokens including the important ones (measured ~15x worse output error). So the inherited wisdom is clean: keys per-channel, values per-token.

Now hold the *memory* fixed and compare spending it on keys versus values. The danger I derived is attention-distribution shift, and that shift is driven by key error through `q ΔK` in the softmax; value error only ever enters as a linear reweighting of the output *after* the distribution is set. So I expect keeping keys high and values low to win. And it does — relative attention-output error on a real model under per-token-asym: K4V2 lands around `e_o = 0.453` while K2V4 is around `0.892`, roughly double, at the same bits. Word-perplexity says the same thing from the other side: K8V4 ≈ KV8 and K4V2 ≈ KV4, but K4V8 and K2V4 degrade sharply — quality drops when key precision is cut, not when value precision is. So at a fixed budget the key-first allocation is right, and that gives me a small, ordered menu of pairs to even consider per layer: KV8, K8V4, KV4, K4V2, KV2 — a "key ≥ value" ladder — rather than the full nine-way `{2,4,8} × {2,4,8}`.

So I have the key-versus-value axis. The layer axis is the other half, and here I can't derive it — I have to take it as an observed property of the model. The layer-wise sensitivity profile is stable across input prompts: a layer that is sensitive on one GSM8K prompt is sensitive on the others, the same model family shows the same profile, and there's no clean depth heuristic — sensitive and insensitive layers are interleaved, and the *most* sensitive layer even moves when I switch quantization mode (per-token vs per-channel). Models differ wildly too: most only really break at INT2 key, but the Qwen2.5-7B and Qwen2.5-Math models break already at INT4 key. The crucial word is "stable across prompts." If sensitivity were prompt-dependent I'd be forced into an online, per-request decision. Because it's an inherent model property, I can decide the per-layer precision once, before deployment, and just load the decision at inference with zero runtime cost.

That reframes the whole thing. The fine-grained crowd — methods that identify critical tokens on the fly and bump their precision per step — are solving a harder problem than I need to. And they pay for it: a per-token or per-page precision difference *inside* a layer can't be fused with FlashAttention or a vLLM paged cache, and the online critical-token search adds branchy control flow that wrecks static-graph acceleration. The throughput I was trying to buy with quantization gets eaten by the machinery that decides the quantization. I don't want any of that. I want every low-bit token in a given layer to share one precision pair — coarse-grained, one `(P_k, P_v)` per layer, like K8V4 — so the layer stays uniform precision and fuses cleanly with existing kernels, and I want the pair *chosen* per layer rather than fixed globally, because the sensitivity varies by more than an order of magnitude across layers. Coarse inside a layer, mixed across layers, decided ahead of time.

How do I make that decision? It's an optimization with two objectives in tension: minimize the quantized KV memory and minimize the accuracy loss, subject to a memory cap `M` and an accuracy-loss cap `ΔA`:

  `min_P (f_m(P), f_a(P))  s.t.  f_m(P) ≤ M,  f_a(P) ≤ ΔA`,

where `P ∈ S^L` assigns a pair `(P_k^l, P_v^l)` to each of the `L` layers, `f_m(P) = sum(P)/(2L)` is the average equivalent bits over all keys and values, and `f_a(P) = A_LLM(KV_half) - A_LLM(KV_P)` is the accuracy drop versus the fp16-KV model. This is a discrete multi-objective combinatorial problem; I'd solve it with an evolutionary search over the layer assignments, treating the model's accuracy under an assignment as a black box — it's not differentiable in the bit choices. Fine in principle. Now count the search space. Nine pairs per layer, `L` layers, so `9^L`. For a 32-layer model that's `9^32 ≈ 3.4 × 10^30`. Completely intractable — no evolutionary search samples its way through `10^30` usefully. Wall. I can't search this raw.

So I have to shrink the space before I search. I'll attack the two factors of `9^L` separately — first the base `9`, then the exponent `L`. The base first: for a *single* layer, do I really need all nine pairs as candidates? Plot each of the nine pairs for that layer as a point in (equivalent bits, relative attention-output error `e_o`) and keep only the Pareto frontier — the pairs that aren't dominated by some other pair that is both cheaper and more accurate. Any pair that's strictly worse on both axes can never be the right choice for that layer under *this* local error criterion, so dropping it loses nothing relative to that criterion. And here's the payoff of the key-versus-value analysis: when I do this, most layers' frontier collapses to exactly the key-first ladder I already derived — `{KV8, K8V4, KV4, K4V2, KV2}`, five pairs instead of nine — precisely because the value-first pairs like K2V4 and K4V8 are dominated (worse `e_o` at the same or higher cost). Occasionally a layer keeps a different set (the very first layer sometimes prefers K4V8 over K8V4; sensitive Qwen layers sometimes keep K8V2 over KV4 because uniform 4-bit key is already too lossy there). So intra-layer Pareto pruning takes `9^L` down to about `5^L`, e.g. `5^32 ≈ 2.3 × 10^22`. Better, still hopeless — I've only fixed the base.

Now the exponent. `L` is the killer; I need to collapse layers into a much smaller number of *groups* that share one decision. The handle is that I already have, per layer, both its pruned candidate set and its sensitivity signature — the vector of attention-output errors the pruned pairs produce. Layers with the same pruned candidate set are responding to quantization the same qualitatively; among those, layers whose sensitivity vectors are close are responding the same quantitatively. So: partition layers by their pruned candidate set, then within each partition cluster by sensitivity. I want a clustering that doesn't make me pre-commit to a number of clusters and that just merges whatever is densely similar — DBSCAN fits, with a small radius (`eps = 0.05`) and `min_samples = 2`. Layers in a cluster get tied to a single precision pair, so the search is now over *group* assignments. This step is not lossless the way the local Pareto deletion was — tying layers together can exclude assignments where two clustered layers would have wanted different pairs. It's a pragmatic collapse justified by the observed stability and similarity of sensitivity profiles, and it's where the real reduction lives: a model's 28-64 layers fold to roughly 4-8 groups, so `5^L` becomes `5^G` — `5^6 = 15625` for the 32-layer example, a space an evolutionary optimizer can actually cover. And the clustering is doing real work, not just bookkeeping: the layers the per-prompt error plots flag as highly sensitive land in different groups from the insensitive ones.

One subtlety in *how* I score an assignment during the search, because if I get the calibration wrong I'll rank the pairs by the wrong signal. The whole danger I'm guarding against is *accumulated* error — the two-dimensional kind that flips a token deep in a long generation. If I calibrate with isolated, non-accumulating per-layer error, I'll under-weight exactly the layers whose errors compound. So I deliberately let the error accumulate during calibration: use the *dequantized* KV for the prefill self-attention so the error propagates layer-to-layer, and calibrate on long, hard generations — math reasoning — where a small error flips an intermediate token and produces a large, measurable final-answer swing. That amplifies the signal that distinguishes a good assignment from a bad one. Concretely I run the black-box search under an accuracy objective on the first couple hundred GSM8K few-shot prompts and a memory objective, with soft equivalent-precision constraints (around 4-bit and 6-bit). A decomposition-based multi-objective evolutionary method is the natural choice for the Pareto-front search; in the concrete implementation the same grouped black-box problem comes out through a constrained NSGA-II sampler driving the study — the sampler name matters less than the shape: maximize task accuracy, minimize equivalent bits, over the reduced discrete space. The output is a table: for each layer, the chosen `(P_k, P_v)`. That table is the entire online cost — load it, quantize each layer at its assigned bits, no per-step decision at all.

Now let me make this concrete for the deployable per-token mode and the on-device Qwen2.5-3B with its 36 layers. In the per-token mode both keys and values are quantized along the token dimension — every token's row gets its own scale and zero-point — because that's the mode that drops straight into a streaming cache with no per-channel regrouping and no special operators. The cost of per-token keys (vs per-channel) is exactly the larger key error I derived — but that's the cost the layer-wise search is built to pay back: it spends extra *key bits* on the layers where the per-token key error would otherwise flip tokens. That's the point of having a per-token variant alongside a KIVI-style one — it narrows the gap between the simple-but-lossier per-token mode and the accurate-but-heavier per-channel KIVI mode by putting the bits where the search says they're needed.

Let me nail the integer arithmetic, because the deployable form uses a *signed* asymmetric grid, not the unsigned background formula, and I want the scale to come out consistent. For `B` bits I use the signed range `q_max = 2^(B-1) - 1`, `q_min = -2^(B-1)`. For a group `g` (in per-token mode the group is the whole `head_dim` of one token row), I take its min and max and set

  `scale = clamp(max(g) - min(g), min=1e-5) / (q_max - q_min)`.

Check the denominator: `q_max - q_min = (2^(B-1) - 1) - (-(2^(B-1))) = 2^B - 1`. So `scale = range/(2^B - 1)` — exactly the round-to-nearest scale from the background, just expressed over a signed grid. The clamp at `1e-5` is the divide-by-zero floor for a dead (constant) group. The zero-point that maps the group's minimum onto the bottom of the signed grid is

  `zeros = round(min(g)/scale) - q_min`,

so that `round(min(g)/scale) - zeros = q_min`. Then quantize and clamp to the grid, and dequantize:

  `quant = clamp( round(g/scale - zeros), q_min, q_max )`,
  `dequant = (quant + zeros) · scale`.

Sanity-check that `dequant ≈ g`: ignoring rounding and clipping, `g/scale - zeros` rounds to roughly `g/scale - zeros`, and `(g/scale - zeros + zeros)·scale = g`. Good — the zero-point cancels on the way back out, which is the whole point of carrying it: it lets an asymmetric range (KV ranges are not symmetric about zero) use the full signed grid instead of wasting codes, and the asymmetry is exactly why I don't use a symmetric `|max|`-only scale. In per-token mode the group size is the full `head_dim` (one group per token row) and there's no residual window — the residual fp16 sliding window is a *per-channel* streaming artifact (you need a block of tokens before a channel's stats are known); per-token quantization needs none of it, so `residual_length = 0` and every cached token is quantized at its layer's bits, which also keeps the memory accounting trivially clean. One thing I won't conflate: the accumulate-error-through-prefill trick is a *calibration* device for the search; the deployed inference path quantizes the cached KV but lets the current prefill states pass through at full precision (no forced prefill quantization), so the search-time amplifier and the run-time setting are different.

For the actual per-layer assignment on Qwen2.5-3B in per-token mode at the ~4-bit budget, the search returns a sparse, non-monotone-in-depth table — which is the whole reason I needed the search and couldn't have used a depth heuristic. Most layers come back K4V2 (key 4-bit, value 2-bit — the key-first ladder's workhorse). Layer 0 gets extra key precision, K8V4. A few layers the search flags as carrying the most accumulation, layers 18, 27 and 29, get full K8V8. And a scattering — layers 10, 19, 24, 26, 33 — get K4V4, keeping value precision up where it matters. Everything else stays K4V2. Tallying it: 160 key bits and 102 value bits over 36 layers, so the equivalent precision is `(160 + 102)/(2·36) = 3.6389` bits — under 4, even though the menu was built from a 4-bit budget, because the key-first pruning let most layers drop their values to 2-bit. Reading that dictionary off is the entire learned artifact.

So now I fill in the harness. The quantizer needs: the per-layer preset table; a single grouped signed-asymmetric quantize/dequantize routine that both `quantize_key` and `quantize_value` call with the layer's assigned bits, axis 0, group size -1 (whole `head_dim`), residual 0; and an `estimate_bits` that just reports the layer's bit count since there's no residual to average. No prefill observer is needed — the policy is static, loaded from the offline search — so the observer hook is a no-op and the query observation position is the default post-RoPE. The quantize routine works in fp32 for stability, reshapes each token row into one group, computes the per-group min/max, scale, and zero-point, rounds onto the signed grid, dequantizes, writes the dequantized tensor back, and reports the layer's effective bits; if the requested bits are already full precision (≥ the fp reference) it returns the tensor untouched at 16 bits.

```python
import math
import torch

FP_BITS = float(torch.finfo(torch.float16).bits)  # 16: uncompressed reference


class AdaptiveKVQuantizer:
    """Sensitivity-aware layer-wise mixed-precision KV quantizer.

    Loads an offline-searched per-layer (P_k, P_v) table and quantizes each
    layer's keys/values at its assigned precision with a signed-asymmetric
    round-to-nearest grid, per-token (axis 0), one group per token row."""

    # Offline-searched per-layer pairs for Qwen2.5-3B-Instruct, per-token mode:
    # key-first ladder, sparse and non-monotone in depth; ~3.64 equivalent bits.
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
        return None  # static policy: nothing to reset per example

    def needs_prefill_qkv_observer(self) -> bool:
        return False  # no online observation; the table is searched ahead of time

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        return None

    def query_observation_position(self) -> str:
        return "post_rope"

    def _signed_asymmetric(self, tensor, bits, axis, group_size, residual_length):
        # signed-asymmetric round-to-nearest over groups; per-token here = whole
        # head_dim as one group, no residual window
        work = tensor.float().clone()
        _, _, seq_len, _ = work.shape
        residual = max(0, min(seq_len, int(residual_length)))   # 0 in per-token mode
        quant_end = seq_len - residual
        if quant_end <= 0 or bits >= FP_BITS - 0.5:             # full precision: pass through
            return work.to(tensor.dtype), FP_BITS
        quant_slice = work[:, :, :quant_end, :]
        # axis 0 = per-token (last dim is the group); axis 1 would transpose to per-channel
        shaped = quant_slice.transpose(-2, -1).contiguous() if axis == 1 else quant_slice
        group_size = shaped.shape[-1] if int(group_size) == -1 else int(group_size)
        original_shape = shaped.shape
        trailing = shaped.shape[-1]
        padded = math.ceil(trailing / group_size) * group_size
        shaped = torch.nn.functional.pad(shaped, (0, padded - trailing)) if padded != trailing else shaped
        rows = shaped.reshape(-1, group_size)                  # one row per group
        q_max, q_min = 2 ** (bits - 1) - 1, -(2 ** (bits - 1)) # signed grid; q_max - q_min = 2^bits - 1
        max_vals = rows.max(dim=1).values
        min_vals = rows.min(dim=1).values
        scale = (max_vals - min_vals).clamp(min=1e-5) / (q_max - q_min)   # = range / (2^bits - 1)
        zeros = (min_vals / scale).round() - q_min            # maps group min onto q_min
        quant = torch.round(rows / scale.unsqueeze(1) - zeros.unsqueeze(1)).clamp(q_min, q_max)
        dequant = (quant + zeros.unsqueeze(1)) * scale.unsqueeze(1)       # zero-point cancels back out
        dequant = dequant.reshape(*original_shape[:-1], padded)[..., :trailing]
        if axis == 1:
            dequant = dequant.transpose(-2, -1).contiguous()
        work[:, :, :quant_end, :] = dequant
        avg_bits = (quant_end * bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id, key_states, cache_meta):
        # this layer's assigned key precision; per-token (axis 0), one group/row, no residual
        return self._signed_asymmetric(key_states, self._PRESET[layer_id]["key"], axis=0, group_size=-1, residual_length=0)

    def quantize_value(self, layer_id, value_states, cache_meta):
        return self._signed_asymmetric(value_states, self._PRESET[layer_id]["value"], axis=0, group_size=-1, residual_length=0)

    def estimate_bits(self, layer_id, kv_kind, seq_len, head_dim, cache_meta):
        return float(self._PRESET[layer_id][kv_kind])   # no residual to average in per-token mode
```

Let me retrace the causal chain. The thing that actually breaks low-bit KV cache isn't per-element error, it's two-dimensional error accumulation across layers and decode steps that flips a single token and poisons everything downstream — and uniform quantization can't address it because it spends bits flat. The fix has to be a bit-allocation. Expanding the softmax with key noise shows the attention distribution is preserved only for sparse/concentrated heads, so the danger is attention-distribution shift driven by key error `q ΔK`, and the lever against it is *key* precision in the fragile layers; combined with the attention-sparsity reason values tolerate per-token quantization, this makes a key-first precision ladder and says, at equal budget, K4V2 beats K2V4. Layer-wise sensitivity is an inherent, prompt-independent property with no depth heuristic, which means the per-layer precision can be decided once before deployment and loaded with zero online cost — sidestepping the fine-grained online methods that can't fuse with FlashAttention/vLLM. That makes it a discrete multi-objective optimization over per-layer pairs, whose `9^L` space is intractable, so I shrink it in two stages: intra-layer Pareto pruning collapses the nine pairs to the key-first five (lossless under the local error criterion), and inter-layer similarity clustering collapses the `L` layers to a handful of groups (a pragmatic collapse on the observed sensitivity stability), taking `9^L` down to `5^G`. A black-box multi-objective search under an accumulation-amplifying math-reasoning calibration then returns the per-layer table. Realized in the deployable per-token signed-asymmetric form — one group per token row, no residual, scale `= range/(2^B - 1)`, a zero-point that cancels on dequant — the searched table for Qwen2.5-3B comes out mostly K4V2 with a few K8V8 / K8V4 / K4V4 layers (about 3.64 equivalent bits), and that table loaded into the quantizer is the whole method.
