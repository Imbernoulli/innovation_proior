Let me start from what actually hurts in long-context decode, because average reconstruction error does not explain why low-bit KV cache quantization wrecks generation. Every past key and value tensor has to be cached so the next token can attend to the whole prefix without recomputing it, and that cache grows linearly in batch size times sequence length. In the long-context, large-batch regime it is the cache, not the weights, that pins me to the memory and the bandwidth — for a 175B model at batch 512, prompt 512, 32 generated tokens, the KV cache is over a terabyte, several times the size of the weights, and each decode step is dominated by streaming it. So I want to store the running key/value in fewer bits. The constraint that makes this hard is that the cache is a streaming structure: a new key row and a new value row are appended at every single decode step, and I have to quantize them as they arrive. Anything that needs to look at the whole tensor at once and solve a little optimization — a GPTQ-style fit — is dead on arrival here, because the tensor I'd be fitting changes every step. The only primitive cheap enough to run online on a growing tensor is group-wise round-to-nearest with a per-group scale and zero-point: pick a group of numbers, take its min and max, map the range onto an integer grid, round, store the integers plus the scale and offset, and dequantize on the way into attention. For a `B`-bit group, `z = min(X)`, `s = (max(X) − min(X))/(2^B − 1)`, so each bit I drop roughly doubles the quantization noise. That part is forced; the freedom I have is *where* I spend bits.

And the first thing I notice is that low-bit KV does not fail gently. At 8 bits it's usually benign; at 4 bits often acceptable; at 2 bits the model stops degrading smoothly and instead *flips* a critical token. Concretely: a fifteen-shot GSM8K prompt that at full precision computes "20 − 4 − 4 = 12" gets generated under uniform 2-bit KV as "20 + 4 + 4 = 28", and from that single flipped operator the whole rest of the derivation is wrong — final answer 14 instead of 60. The same prompt at 4-bit reproduces the full-precision answer token for token. So the damage is not uniform blurring of the output; it is a rare, catastrophic token flip on a sensitive step. That tells me the error has to be understood as something that *accumulates*. The quantized cache at layer `l`, step `i` is the input to attention at every later step and every later layer, so the error there depends on every earlier layer's error and every earlier step's error — a two-dimensional accumulation, `e_i^l = f_e(e_i^{1:l−1}, e_{i−1}^{1:L}, …, e_1^{1:L})`. A single token-and-layer error is negligible; the compounding over a long generation is what flips a token. Which reframes the goal: I don't need every element quantized equally well, I need to *not* let error build on the parts of the computation where a flip is catastrophic, and I can afford to be sloppy everywhere else. A uniform bit budget pretends every layer contributes equally to that cascade — the layer-wise measurements say the opposite. So the whole question becomes: where is the cache most error-tolerant, and where is it not?

Two sub-questions inside that. First, given a fixed bit budget for one layer, should I spend it on the key cache or the value cache? Second, along which axis should I quantize each — per token (one scale/zero per row across channels) or per channel (one scale/zero per channel across a window of tokens)? Let me take the axis question first, because there's a clean empirical handle on it. If I look at the actual key tensors of a trained model, a handful of *fixed* channels carry magnitudes far larger than the rest, and they're the same channels across tokens — persistent channel outliers, the same activation-outlier phenomenon people already found in weight and activation quantization. Now think about what per-token quantization does to that. Per-token shares one scale and zero-point across all channels of a single token. One giant outlier channel blows up the min/max range of that token, so every *other*, well-behaved channel in the token gets quantized against a range it doesn't need — its few available levels are wasted spanning the outlier's reach. Per-channel quantization instead shares the scale within a channel, across a window of tokens, so the outlier channel gets its own wide range and the normal channels get tight ranges that fit them. The error from an outlier is confined to its own channel instead of contaminating the whole token.

Before I lean on that, let me actually measure the gap, because the argument is plausible but the size matters. I take a 64-token by 16-channel key block, normal channels at magnitude ~0.3, and plant a persistent outlier in channel 5 by adding 8 to it across all tokens. Then I quantize-dequantize it both ways and compare the relative error `|K − K_hat|/|K|`:

```
bits=4   per-token rel err 0.1741   per-channel rel err 0.0312   ratio 5.58x
bits=8   per-token rel err 0.0101   per-channel rel err 0.0018   ratio 5.71x
```

So with one planted outlier channel, per-channel cuts the key reconstruction error by roughly 5-6×, and the ratio is stable across bit-widths. That matches the order of magnitude reported on real models (~2.5× on the raw key error, and a larger ~5× gap on the attention *score* `softmax(qK^T/√D)`, which is what the model actually consumes — the channel-confined error perturbs the logits even less than it perturbs the raw key). The mechanism and the magnitude both hold up, so per-channel for keys.

Does the same argument carry to values? The value tensor shows no such persistent-outlier pattern — its channels are roughly comparable in magnitude — so on the face of it per-channel and per-token should give similar *reconstruction* error for values, and indeed they do if I only look at `|V − V_hat|`. But values never enter attention raw; they enter as the *output* `o_i = Σ_j a_{ij} V_j`, a weighted sum of value rows with the attention weights, and attention is highly sparse — a small handful of tokens carry almost all the weight (measured sparsity around 84%). So the right error to look at is on the output, not the raw value. Let me check which axis that favors. I take a 64-token by 16-channel value block with a sparse attention vector (mass 0.5/0.3/0.15/0.05 on four tokens), quantize both ways, and compare the relative *output* error:

```
bits=2   per-token out-err 0.3702   per-channel out-err 0.5419   ratio 1.46x
bits=4   per-token out-err 0.0549   per-channel out-err 0.0790   ratio 1.44x
```

Per-token wins on the output even though the raw value errors are comparable, and the direction is exactly what the sparsity argument predicts: per-token keeps each row's error local, so an unimportant token's error is multiplied by its near-zero weight and barely reaches the output, while per-channel shares a scale across tokens within a channel and leaks the many unimportant tokens' error into the same channel of the few important rows the output actually sums with large weight. My toy only shows ~1.45×, not the ~15× reported on real models — which I'd expect, since my synthetic value block has mild channel structure and a not-very-peaked attention vector, and the real gap grows with how concentrated the attention is and how heavy-tailed the per-channel value ranges are. The sign is what I trust here, and the sign is unambiguous: per-token for values. So the axis recipe at low bits is keys per-channel, values per-token.

There's an implementation wrinkle I should pin down now because it shapes the data structure. Per-token quantization is trivially streaming — a new token's row just gets its own scale/zero and is appended. Per-channel is not, because a channel's scale needs a whole *window* of tokens before it's defined, and tokens arrive one at a time. The fix is to group the key cache every `G` tokens and quantize a group only once it's full: split the key cache into a *grouped* part `X_{Kg} = X_K[:l−r]` that's already quantized in complete groups of `G`, and a *residual* part `X_{Kr} = X_K[l−r:]` of the most recent `r` tokens kept in full precision, where `l−r` is divisible by `G`. Each newly arrived key joins the residual; once the residual fills a group, quantize it and concatenate it onto the grouped part, and reset. Values get the same treatment with a per-token residual queue: keep the most recent tokens in FP, pop and per-token-quantize the oldest as the queue advances. And crucially, during prefill I pass the *exact* keys and values forward to the next layers and only keep the quantized cache in storage — I don't want prefill itself to start the accumulation early. That FP residual sliding window — the most recent tokens always exact — turns out to be load-bearing on hard tasks like GSM8K, because the freshly generated tokens are the ones currently being reasoned over. With `G = 32` and the residual length a multiple of `G` (32 on math tasks), this is the streaming-friendly per-channel-key / per-token-value scheme. That's a strong tuning-free baseline — but it applies one precision *uniformly* across all layers, and that uniformity is where I suspect it's leaving accuracy on the table.

Back to the first sub-question: at a fixed memory budget, key or value? I have a hint from the axis discussion — keys carry the structure attention is most sensitive to — but let me get it sharply. Fix the budget at "one of the pair is 4-bit, the other 2-bit" and compare K4V2 against K2V4. Measured relative attention-output error on Llama-3.1-8B: K4V2 gives 0.453, K2V4 gives 0.892 — roughly double the error when I starve the key instead of the value. And in word-perplexity it's the same story: K8V4 tracks full KV8, K4V2 tracks KV4, but the moment I invert it — K4V8, K2V4 — perplexity blows up. So the key cache is generally more important than the value cache; spend the bits on the key. Some models are even more lopsided — Qwen2.5-7B and the Math-7B are sensitive even to 4-bit *key* quant, while still tolerating 2-bit value comfortably.

But "generally more important" is an empirical summary, and I distrust empirical summaries I can't explain. *Why* is the key more important, mechanistically? Let me try to derive it. Take one query token `q ∈ R^{1×D}` and the key cache `K ∈ R^{D×S}`, with `K_i` the `i`-th key vector and `qK_i` the scalar logit. The error-free attention score on token `i` is `a_i = exp(qK_i) / Σ_j exp(qK_j)`. Now quantize the keys: the asymmetric uniform quantization error is approximately `ΔK ∼ N(0, σ²)` with `σ = (max K − min K)/(2^B − 1)`, so as the bit-width `B` drops the noise `σ` grows geometrically — fewer bits, bigger `qΔK`. The corrupted score is

  â_i = exp(q(K_i + ΔK_i)) / Σ_j exp(q(K_j + ΔK_j)).

The exponential splits multiplicatively, `exp(q(K_i+ΔK_i)) = exp(qK_i)·exp(qΔK_i)`, so

  â_i = exp(qK_i)·exp(qΔK_i) / Σ_j exp(qK_j)·exp(qΔK_j).

I want to compare this to `a_i`, so let me pull `exp(qΔK_i)` out of the numerator by dividing top and bottom by it:

  â_i = exp(qK_i) / Σ_j exp(qK_j)·[ exp(qΔK_j) / exp(qΔK_i) ].

There the corruption sits, fully exposed: every denominator term is reweighted by the error ratio `exp(qΔK_j)/exp(qΔK_i)`. When does the whole distribution survive that reweighting unchanged? One way is exact: if every error ratio is exactly 1 — `qΔK_i = qΔK_j` for all `i, j`, every key's error sharing the same inner product with the query — the denominator is untouched and `â = a`. But the errors are independent draws from a noise distribution, so generically that doesn't hold; it's a measure-zero coincidence, not something I can rely on. The other, approximate way is the one that might actually fire: suppose a single key `i` dominates the softmax, `exp(qK_i) ≫ exp(qK_j)` for all `j ≠ i`. Divide the corrupted score's numerator and denominator by `exp(qK_i)`:

  â_i = exp(qΔK_i) / Σ_j exp(qΔK_j)·[ exp(qK_j) / exp(qK_i) ].

For `j ≠ i`, `exp(qK_j)/exp(qK_i) ≈ 0`, and for `j = i` it's 1. So the sum collapses to its `j = i` term, the denominator is `≈ exp(qΔK_i)`, and

  â_i ≈ exp(qΔK_i) / exp(qΔK_i) = 1,

with all the dominated `â_j ≈ 0`. On paper the corrupted distribution stays a near-one-hot on `i` no matter how big the key errors are, because the dominating token's own noise factor cancels against itself and the rest are negligible.

That's a clean derivation, but it's exactly the kind of "it cancels" claim I should not trust until I've watched it happen, so let me put it on a concrete example with deliberately *brutal* noise. Eight-dimensional query, five keys, and I push one key (index 2) up so it dominates the softmax. The clean distribution is `[0, 0, 1, 0, 0]` — a one-hot on key 2. The key range gives `σ ≈ 1.9` at 2 bits, which is enormous relative to the keys themselves. Adding fresh `N(0, σ²)` key noise across three trials:

```
clean dist      [0.   0.   1.   0.   0.]   argmax 2
trial0 corrupted [0.   0.   1.   0.   0.]   L1 shift 0.0000
trial1 corrupted [0.006 0.027 0.943 0.003 0.021]   L1 shift 0.1133
trial2 corrupted [0.   0.   1.   0.   0.]   L1 shift 0.0000
```

Even with noise this violent the dominating key holds: the distribution barely moves (worst trial shifts ~0.11 in L1) and the argmax never flips. The cancellation is real, not just algebra. Now I should check the *other* side of the claim — that a spread head, with no dominating key, is genuinely fragile, because if both kinds of head were robust the conclusion would be empty. Six comparable keys, clean distribution `[0.059, 0.104, 0.130, 0.021, 0.421, 0.266]` — diffuse, top mass only 0.42 — and the same kind of 2-bit-scale noise:

```
spread head  mean L1 shift over 6 trials  0.918
clean argmax 4   argmax flips among 6 trials  1
```

The spread head moves by ~0.9 in L1 on average and its argmax flips in 1 of 6 trials. So the two cases come apart exactly as the derivation said: a sparse/concentrated head is robust to key quantization because the dominating key's noise self-cancels; a diffuse retrieval head reading from many comparable keys has no dominating term, the error ratios don't cancel, and the distribution shifts enough to re-order which keys look critical. And the derivation tells me the *remedy* precisely — the corruption rides entirely on `qΔK`, the query dotted into the key error, so to fight attention shift in a sensitive (non-sparse) head I should shrink `qΔK` by raising the *key* precision there. Not the value precision — the value cache doesn't enter the softmax at all; value error is applied *after* the weights are chosen, as a linear perturbation `Σ_j a_{ij} ΔV_j`, so it can't repair a distribution that has already shifted. That is why the key is more important, and it's why the lever for accuracy is key bits in the sensitive layers.

Now, *which* layers are sensitive? The sparse-vs-retrieval distinction is per head, but I need a per-layer knob if I want something hardware-friendly, so let me look at the layer-wise picture. If I measure, for each layer, the attention-score error and the relative attention-output error `e_o` under a fixed quantization, two things jump out. First, the sensitive layers stay sensitive across *different prompts* — the layer-wise error profile is a property of the trained model, not of the input. That's enormously useful: it means I can characterize a model's sensitivity *once*, offline, and trust it at inference time on inputs I've never seen. Second, there is no tidy depth rule. It's tempting to assume "the first few and last few layers are the important ones, keep those high-precision," the way some KV-eviction methods statically protect prefix and recent tokens. But the data refuses to cooperate: sensitive and insensitive layers are interleaved, and the identity of the most sensitive layer even *changes* with the quantization mode — under per-token-asym the most sensitive layer of one model is layer 29, while under the per-channel key mode the most sensitive layers move to 11 and 13. So I cannot hard-code a heuristic — I have to *measure* per-layer sensitivity and let the measurement decide.

Let me take stock of what's forced so far. Error accumulates two-dimensionally and a flip is catastrophic, so I must protect the sensitive parts; the key is the lever, per-channel for keys and per-token for values; layer sensitivity is real, inherent, prompt-independent, and not guessable from depth. The existing low-bit quantizers — KIVI, KVQuant, the per-token-asym baseline — all pick *one* target precision (and axis) and apply it the *same* way to every layer. KIVI is tuning-free and elegant but uniform; KVQuant adds non-uniform datatypes, pre-RoPE key quant, and sparse outlier storage but still targets one precision per layer, at the cost of custom kernels and offline calibration that's hard to fuse with paged-cache backends. The fine-grained online methods (QAQ, MiKV, ZipCache) do adapt — per token, on the fly — but an intra-layer per-token precision *difference* can't be fused with fused-attention or a vLLM paged cache (the layer's cache is no longer one uniform format), and the online critical-token logic adds control flow that breaks static-graph acceleration. So they buy accuracy with a deployment tax I refuse to pay.

What's left in the gap none of them occupy: keep each layer's cache a *single uniform format* (so it fuses with the kernels — that means coarse-grained, the whole low-bit KV in a layer shares one precision pair like K8V4 or K4V2, not per-token bits), but let *different layers* use *different* pairs, chosen by their measured sensitivity. And do the choosing *offline*, since sensitivity is prompt-independent, so there is exactly zero decision-making cost at inference — at serving time I just load a table that says "layer 5: key 4-bit value 2-bit; layer 13: key 8-bit value 4-bit; …" and quantize accordingly. Coarse-grained for the hardware, mixed-across-layers for the accuracy, offline for the zero overhead. The problem is now: how do I *find* that table?

Let me formalize it as an optimization. A configuration `P ∈ S^L` assigns each of the `L` layers a precision pair `(P_k^l, P_v^l)`. I have two things I want to minimize at once: memory and accuracy loss. Memory is clean — the average equivalent bits across all KV is `f_m(P) = Σ(P) / (2L)` (sum of all the per-layer key and value bit-widths, divided by `2L` because each layer contributes a key and a value). Accuracy loss is `f_a(P) = A_LLM(KV_half) − A_LLM(KV_P)` — the drop in actual model accuracy when I quantize with configuration `P` versus keeping KV in 16-bit. So it's a multi-objective problem,

  min_P ( f_m(P), f_a(P) )  s.t.  f_m(P) ≤ M,  f_a(P) ≤ ΔA,

with hardware-driven memory caps `M` and accuracy-loss tolerances `ΔA`. The objective `f_a` is the wall: it's the *real* model accuracy on real data, a black box — non-differentiable, expensive to evaluate (a forward pass over a calibration set per configuration), and it captures the nonlinear two-dimensional error accumulation that no closed form gives me. So this is a discrete, multi-objective, black-box combinatorial search; I'll drive it with a black-box multi-objective optimizer (an evolutionary scheme like MOEA/D, or an NSGA-style sampler) treating accuracy under each `P` as the oracle.

But before I reach for the optimizer, let me size the search space, because that decides whether the optimizer is even usable. The candidate per-layer pairs are `{2,4,8} × {2,4,8}` — 9 options per layer. With `L` layers that's `9^L`. For a 32-layer model that's `9^32 ≈ 3.4 × 10^30` configurations. No black-box search touches that. I have to *shrink* the space first, and the only honest way to shrink it without throwing away good solutions is to use the structure I already discovered. Two reductions, attacking the two factors of `9^L` separately — the 9 (pairs per layer) and the `L` (the exponent).

The 9 first. Within a single layer, I'm choosing a pair on a (memory, error) trade-off, and most of the 9 pairs are simply dominated. Take the relative attention-output error `e_o` for each of the 9 pairs in a layer, plot it against the pair's equivalent bits, and keep only the Pareto frontier — the pairs for which no other pair is both cheaper and more accurate. Because the key is more important than the value, the dominated pairs are exactly the ones that waste bits on the value: in most layers the surviving frontier is the *key-first* ladder {KV8, K8V4, KV4, K4V2, KV2} — five pairs that step the key down only after the value is already minimal at each memory level. A pair like K4V8 (cheap key, expensive value) is dominated by K8V4 (same bits, but the bits are on the key where they help) and gets pruned. Five pairs instead of nine per layer takes me from `9^L` to about `5^L` — for 32 layers, `5^32 ≈ 2.3 × 10^22`. Better, still impossible. (I keep the few layer exceptions where the frontier genuinely differs — the very first layer sometimes prefers K4V8, and in the per-channel key mode several layers actually prefer K4V8 or K2V4 over K8V4 or K4V2, because once per-channel makes the key cheap and accurate, lowering the key further and raising the value can win. The pruning is per-layer and respects these.)

Now the exponent `L`. This is where prompt-independence pays off again: if two layers have the *same* sensitivity profile, they should get the same precision pair, so I shouldn't search them independently — I should tie them together. So cluster the layers. First partition layers by *which pruned candidate set* they ended up with (layers with different Pareto frontiers are qualitatively different and shouldn't be merged). Then, within a partition, cluster layers by their sensitivity, using the relative attention-output errors `e_o` of the pruned pairs as each layer's sensitivity feature vector. I want a clustering that doesn't force me to pre-specify the number of clusters — I don't know in advance how many distinct sensitivity regimes a given model has — and that's robust to outlier layers (a single uniquely-sensitive layer should be allowed to sit alone). Density-based clustering fits: DBSCAN, which groups points denser than a threshold and leaves true outliers ungrouped, with a neighborhood radius `eps = 0.05` on the error vectors and `min_samples = 2`. This isn't a theorem that no optimum is lost — it's a pragmatic approximation backed by the observed stability of layer sensitivity. It collapses the `L` layers (28 to 64 across the models I care about) into `G` groups (4 to 8), and every layer in a group shares one precision pair. Now the space is `S_p^G` — about `5^6 = 15625` for a 6-group model. *That* a black-box multi-objective search can chew through in a couple hundred iterations. The two-level pruning is what makes the search tractable, and because both levels are derived from the sensitivity structure rather than from blind enumeration, I expect it to keep the good solutions rather than discard them — though that's an expectation I'd want to confirm against an unpruned search on a small model, not something the pruning proves.

One subtlety in *how I evaluate* a configuration during the search, because the whole point is the accumulation. If I just simulate quantize→dequantize on cached tensors without letting errors propagate (the way I measured per-layer `e_o`), I get a clean per-layer error signal — good for the Pareto pruning — but it *under-represents* the thing that actually kills accuracy, the compounding across layers and steps. So for the calibration that drives the final accuracy oracle `f_a`, I deliberately do the opposite: use the *dequantized* KV cache for self-attention during prefill (not the exact tensors), so that each layer's quantization error feeds into the next and the two-dimensional accumulation is switched on and amplified. And I pick a calibration dataset where accumulation actually bites: long-context generation and hard mathematical reasoning, where a single flipped intermediate token derails the whole answer (the GSM8K "20−4−4" → "20+4+4" failure is exactly this). Specifically, maximize accuracy on the first 200 GSM8K few-shot prompts as the first objective, minimize equivalent bits as the second, run a couple hundred search iterations under soft constraints around 4-bit and 6-bit equivalent precision. A calibration that amplifies the failure mode is what lets the search *distinguish* a pair that's quietly fine from one that quietly flips tokens — the differences would be invisible on an easy dataset evaluated without accumulation.

So the offline pipeline is: measure per-layer `e_o` for the 9 pairs under the chosen quantization mode; Pareto-prune each layer to its key-first candidate set; partition by candidate set and DBSCAN-cluster by `e_o` sensitivity into groups; run the multi-objective black-box search over the per-group pairs with the accumulation-amplified GSM8K accuracy oracle and the memory objective; read off the Pareto-optimal table of per-layer precision pairs. And then at inference the table is just *loaded* — a dict from layer id to `(key bits, value bits)` — with no online computation whatsoever. The contribution is the policy, and the policy compiles down to a lookup table plus the streaming round-to-nearest quantizer I already have.

Let me make that concrete for the case I'll ship: the KIVI quantization mode (key per-channel, value per-token, group 32, residual 32) on Qwen2.5-3B-Instruct, searched in the 4-bit family. The searched table for this model and mode is dominated by K4V2 — most of the 36 layers can run a 4-bit key and a 2-bit value — with a few exceptions the search insists on: the earliest layers (0, 1) get K4V8, two early layers (2, 4) drop to K2V4, a couple of layers (12, 28) go to K2V2, and the last two layers (34, 35) get K4V4. The shape is what the lemma led me to expect — bits concentrated on the key, value pushed to 2-bit wherever the layer's heads are sparse/concentrated enough to tolerate it, with a handful of sensitive layers protected. Now I want to check the equivalent precision rather than assert it. Summing over the 36 layers: the key bits are `2·4 + 2 + 4 + 2 + 4·22 + 2·2 + 4·2` worth of entries — let me just total the table directly. Key bits sum to 136, value bits sum to 92, so the preset equivalent precision is `(136 + 92)/(2·36) = 228/72 = 3.1667` bits per cached element before the sequence-dependent residual overhead — well under 4. That residual overhead is worth a sanity check too, because the harness reports `effective_kv_bits` at a 4096-token reference span: with `residual_length = 32`, the kept count is `4096 mod 32 = 0`, so at the reference span the residual contributes nothing and the effective bits equal the nominal bits exactly (a 4-bit layer reports 4.0000, a 2-bit layer 2.0000); it's only at non-multiples like 4097 that the residual nudges a 4-bit layer to 4.0029. So the ~3.17 figure is the honest equivalent precision at the reference span, and it's staying nearly lossless on the math task because the bits are spent where accumulation would otherwise flip a token. ("KVTuner4" names the 4-bit search family, not a promise that the realized precision is exactly 4.)

Now the actual quantizer the table feeds into. The round-to-nearest primitive in signed-integer form: for a group `g` (a row of `group_size` values), the grid is `q_max = 2^{B−1} − 1`, `q_min = −2^{B−1}`, whose span `q_max − q_min = 2^B − 1` — the same denominator as the unsigned background formula, so the scale is identical, just realized on a signed grid with an integer zero-point. The scale is `clamp(max(g) − min(g), 1e-5) / (q_max − q_min)`, the zero-point is `round(min(g)/scale) − q_min`, the quantized value is `clamp(round(g/scale − zeros), q_min, q_max)`, and dequantization is `(quant + zeros)·scale`. The `1e-5` floor on the range stops a constant group from giving a zero scale and a divide-by-zero. The axis is handled by transposing before grouping: for the key (per-channel) I transpose the last two dims so head_dim becomes the row index and the grouping runs over a window of tokens within a channel; for the value (per-token) I leave it and group along head_dim within each token. The residual is the most recent `seq_len mod residual_length` tokens kept in FP (with `residual_length = group_size = 32` the kept count stays below a group and aligns to the group boundary, exactly the streaming sliding window) — those are not quantized, the rest are, and the effective bits per element is the token-weighted average `(quant_tokens·B + residual_tokens·16) / seq_len`.

Putting the searched table and the quantizer together, filling the empty policy slot:

```python
import math
import torch

FP_BITS = 16.0  # FP16 KV reference footprint


class KVTunerKIVIPolicy:
    """KVTuner layer-wise mixed-precision policy in KIVI mode for Qwen2.5-3B-Instruct.
    Key per-channel (axis transpose), value per-token, group size 32, residual 32.
    The per-layer (key bits, value bits) table is the offline Pareto/MOO search result;
    online it is just a lookup -- zero decision-making overhead."""

    # Offline-searched layer-wise KV precision pairs (KIVI mode, 4-bit search family).
    # Bits are concentrated on the key (key > value); value drops to 2-bit wherever the
    # layer's heads are sparse enough to tolerate it; sensitive early/late layers are
    # protected with higher precision. (136 key bits + 92 value bits)/(2*36) ~= 3.17.
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
        return False  # sensitivity is searched offline; nothing to observe online

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        return None

    def query_observation_position(self):
        return "post_rope"

    def _residual_keep_length(self, seq_len, residual_length):
        # Streaming sliding window: keep the most-recent (seq_len mod R) tokens in FP,
        # so the residual stays below a group and aligns to the group boundary.
        residual_length = max(0, min(seq_len, int(residual_length)))
        return seq_len % residual_length if residual_length else 0

    def _signed_asymmetric(self, tensor, bits, axis, group_size, residual_length):
        # Signed round-to-nearest asym RTN; axis=1 -> per-channel (transpose), axis=0 -> per-token.
        work = tensor.float().clone()
        _, _, seq_len, _ = work.shape
        residual = self._residual_keep_length(seq_len, residual_length)
        quant_end = seq_len - residual
        if quant_end <= 0 or bits >= FP_BITS - 0.5:
            return work.to(tensor.dtype), FP_BITS          # whole slice stays FP
        quant_slice = work[:, :, :quant_end, :]
        # per-channel: transpose so head_dim are rows and groups run over a token window
        shaped = quant_slice.transpose(-2, -1).contiguous() if axis == 1 else quant_slice
        group_size = shaped.shape[-1] if int(group_size) == -1 else int(group_size)
        original_shape = shaped.shape
        trailing = shaped.shape[-1]
        padded = math.ceil(trailing / group_size) * group_size
        shaped = torch.nn.functional.pad(shaped, (0, padded - trailing)) if padded != trailing else shaped
        rows = shaped.reshape(-1, group_size)              # one group per row
        q_max, q_min = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))   # signed grid, span 2^B-1
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
        # key per-channel (axis=1), searched key bits for this layer
        return self._signed_asymmetric(key_states, self._PRESET[layer_id]["key"],
                                       axis=1, group_size=32, residual_length=32)

    def quantize_value(self, layer_id, value_states, cache_meta):
        # value per-token (axis=0), searched value bits for this layer
        return self._signed_asymmetric(value_states, self._PRESET[layer_id]["value"],
                                       axis=0, group_size=32, residual_length=32)

    def estimate_bits(self, layer_id, kv_kind, seq_len, head_dim, cache_meta):
        residual = self._residual_keep_length(seq_len, 32)
        quant_tokens = max(0, seq_len - residual)
        bits = self._PRESET[layer_id][kv_kind]
        return float((quant_tokens * bits + residual * FP_BITS) / max(seq_len, 1))
```

The causal chain, end to end. Low-bit KV decode fails by *catastrophic token flips*, not gentle blurring, because quantization error accumulates two-dimensionally across layers and steps — so the goal becomes protecting the sensitive parts of the computation rather than quantizing everything equally. The axis recipe falls out of the data, and I checked both halves on small examples: keys carry persistent channel outliers, so per-channel key quant confines an outlier's error to its own channel (~5.6× smaller key error in my planted-outlier block, matching the ~5× attention-score gap on real models), while values enter attention only as a sparse weighted sum, so per-token value quant keeps the few important rows clean (per-token beat per-channel on output error in my sparse-attention toy, the same direction as the ~15× real-model gap); per-channel keys need a streaming grouped+residual structure with an FP sliding window. Deriving the attention score under key noise shows the corruption rides entirely on `qΔK`; tracing it on a five-key example, a dominating key holds its near-one-hot distribution even under 2-bit-scale noise (L1 shift ≤ 0.11, no argmax flip) while a diffuse six-key head shifts ~0.9 and flips its argmax — so only sparse/concentrated heads are robust, the key is the lever (raise key bits to shrink `qΔK`), and the value never enters the softmax. Layer sensitivity is real, inherent, prompt-independent, and not guessable from depth, so it must be measured, not heuristically assigned. That makes the right design coarse-grained per layer (one uniform format per layer, so it fuses with the kernels) but mixed across layers (each layer its searched pair) and fully offline (zero online cost, since sensitivity doesn't depend on the prompt) — the gap the uniform quantizers and the fine-grained online quantizers both miss. Finding the table is a discrete multi-objective black-box optimization of (memory, accuracy-loss); the `9^L` space is cut by intra-layer Pareto pruning to the key-first set and by inter-layer DBSCAN clustering on `e_o` sensitivity down to `S_p^G`, which the evolutionary search optimizes against an accumulation-amplified hard-math accuracy oracle. The result is a per-layer precision table — key-dominant, value pushed to 2-bit where heads are sparse, sensitive layers protected — whose bits I totalled to 3.17-bit equivalent precision at the 4096-token reference span (where the residual contributes exactly nothing), loaded at inference and fed into a signed-asymmetric round-to-nearest quantizer running keys per-channel and values per-token with group 32 and an FP residual window, staying nearly lossless on the task accumulation would otherwise wreck.
