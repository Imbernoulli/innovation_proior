Q6_K reached +0.16% on the 70B — essentially fp16 quality — but it did so by giving *every* tensor 6.56 bits, and the family is now pinned end to end: Q4_K_S at +1.57% / 4.5 bpw, Q6_K at +0.16% / 6.56 bpw, Q5_K between. The obvious thing is to pick one width and live with its point on the frontier. But Q6_K planted a doubt: the +1.57% gap of Q4_K_S almost certainly is not spread evenly across the model — it is concentrated in a few tensors that the 4-bit code mangles, while the vast majority are perfectly happy at 4 bits. If so, spending 6.56 bits everywhere to fix a few sensitive tensors is paying full price across the whole model for a localized problem. The question stops being "what width?" and becomes "what width *where*?"

I propose **mixed k-quant allocation** (Q4_K_M). This is not a new per-block code at all — Q4_K, Q5_K, Q6_K stay exactly as they are. The innovation is the *allocation policy*: a per-tensor, per-layer rule deciding which k-quant width each tensor gets, so that the bulk of the model runs at 4 bits and extra bits are spent only on the tensors that actually need them, landing at an *average* bpw barely above 4.5 but a perplexity much closer to the high-precision anchor. To build the policy I lean on the real structure of a transformer layer, because its weight matrices are not interchangeable in how much quantization error damages the output. The $\texttt{attn\_v}$ matrix (the value projection) feeds directly into the attention-weighted sum that becomes the layer's output — a small relative error on V is a small error on every attended output, so it is leverage-heavy and a prime candidate for extra bits. The $\texttt{ffn\_down}$ matrix (the second FFN projection, writing the FFN's whole contribution back into the residual stream) multiplies the heavy-tailed, outlier-prone hidden activations that follow the nonlinearity — exactly the regime where coarse quantization hurts — so it is the second tensor to protect. There is also *positional* structure: the first and last few layers tend to be more sensitive than the middle, the early layers because everything downstream depends on them and the late layers because their errors do not get averaged out.

That gives a concrete rule. I encode "this layer deserves more bits" as a predicate over the layer index,

$$\texttt{use\_more\_bits}(i, n) = \big(i < n/8\big) \;\lor\; \big(i \ge 7n/8\big) \;\lor\; \big((i - n/8)\bmod 3 = 2\big),$$

whose first two clauses cover the boundary eighths and whose third sprinkles an upgraded layer every third one through the middle, so the deep middle is not uniformly starved but gets a regular fraction of promotions. Wiring this into the allocation: for a "medium" 4-bit mix most tensors stay Q4_K, and the promotions fire on $\texttt{attn\_v}$ and $\texttt{ffn\_down}$. For Q4_K_M (and Q5_K_M) the layers where $\texttt{use\_more\_bits}$ fires jump to Q6_K on both tensors; for Q4_K_S the first few value matrices and the first eighth of $\texttt{ffn\_down}$ layers are lifted to Q5_K. There are architecture-specific guardrails on top: in the 70B model eight heads share the value tensor, making $\texttt{attn\_v.weight}$ roughly $8\times$ smaller than $\texttt{attn\_q.weight}$, so it can be lifted to Q5_K for a near-negligible size increase; Falcon has its own $\texttt{ffn\_down}$ split because its sensitivity differs; and expert-heavy (8-expert) architectures bump the shared value projection more aggressively since it is cheap to protect.

The result is a model that is *mostly* Q4_K with a minority of Q5_K/Q6_K tensors sprinkled exactly where the error is most costly. Because the promoted tensors are a small fraction of the weights, the average bits-per-weight creeps up only slightly above Q4_K_S, while the perplexity drops substantially toward the Q6_K anchor — the extra bits land on the tensors responsible for most of Q4_K_S's gap. Mechanically this all lives one level above the quantizers, in a function $\texttt{llama\_tensor\_get\_type}$ that, given a tensor's name, category, layer index, and the target ftype, returns the GGML type to use; the per-block routines are untouched. I am also clear about the deeper limit this *cannot* reach, because no bit-allocation policy can. Every k-quant, at every width, still chooses its integer levels to minimize the error in the *weights*, with a weight whose importance is a magnitude-derived proxy — $\overline{x} + |x_i|$ in the plain Q4_K fit, $x_i^2$ in the symmetric scale fit. But a weight's true importance is how much it moves the *output*, which depends on the *activations* it multiplies, not on the weight's magnitude. That is baked into what every sub-block scale optimizes, and it is the next thing to attack.

```c
    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
```

```c
    } else if (category_is_attn_v(category)) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        if (qs.model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs.i_attention_wv;
```

```c
    } else if (category == tensor_category::FFN_DOWN) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
```
