The mixed allocation worked — Q4_K_M cut the 70B gap to +1.20% from Q4_K_S's +1.57% for about 2 GiB, and on the 8B it pulled q4_K_S's ΔPPL of 0.269 down to 0.175 at q4_K_M — so bits should go non-uniformly across *tensors*. But it cannot reach the thing underneath, and that thing is more fundamental than which tensor gets which width. Look at what every k-quant is actually optimizing, all the way down: $\texttt{make\_qkx2\_quants}$, $\texttt{make\_qx\_quants}$, every sub-block chooses its scale and levels to minimize $\sum_i w_i\,(\text{recon}_i - x_i)^2$, and without activation data the weight $w_i$ is derived from the weight values themselves — $\overline{x} + |x_i|$ in the plain Q4_K fit, $x_i^2$ in the symmetric scale fit. So the quantizer is minimizing a magnitude-weighted error in the *weights*, using each weight's own magnitude as a stand-in for how much it matters. But a weight contributes to the output as $w_i\cdot a_i$, where $a_i$ is the activation it multiplies; its real influence is governed by $w_i\cdot a_i$, not $w_i$ alone. A large weight almost always multiplied by a near-zero activation contributes nothing — getting it wrong costs nothing — while a modest weight consistently multiplied by a large activation is expensive to get wrong. The magnitude proxy is blind to the activation: it lavishes precision on big-but-unused weights and skimps on modest-but-always-used ones. No allocation policy fixes this, because it lives inside the per-sub-block objective.

I propose **importance-matrix (imatrix) quantization**: weight each weight's quantization error by how big the activations it multiplies typically are. Write the layer output for input $a$ as $y = \sum_i w_i a_i$; quantizing $w \to \hat w$ gives output error $\sum_i (w_i - \hat w_i) a_i$. Squared and averaged over a representative set of inputs, the dominant separable part of the expected squared output error weights each weight's reconstruction error $(w_i - \hat w_i)^2$ by $E[a_i^2]$ — the *mean squared activation* on coordinate $i$. That is the signal I want, and it lives in the *data*, not the weights, so I get it from a separate calibration pass: before quantizing, run the fp16 model over a calibration corpus, hook every matmul's input, accumulate $\sum_t a_{i,t}^2$ per input channel into an "importance matrix," and write it out. That per-channel statistic is then handed to the quantizer as the per-weight importance $\texttt{quant\_weights}$.

Folding it in is clean because the hook already exists — the search routines all take a per-entry weight array; I have just been filling it with weight-derived values. When an importance vector $\texttt{qw}$ is supplied I replace that with

$$w_i = \texttt{qw}_i\cdot\sqrt{\sigma^2 + x_i^2},$$

where $\texttt{qw}_i$ is the activation-importance from calibration and $\sigma^2$ is the block's activation-variance proxy, $\sigma^2 = 2\,\Sigma x^2 / \texttt{QK\_K}$. The $\sqrt{\sigma^2 + x_i^2}$ factor is a magnitude-aware floor, so that even a low-importance but large weight is not completely ignored and stays representable; the product is importance times that regularizer, and it is exactly the substitution that turns weight-error minimization into an estimate of output-error minimization. Two further consequences are threaded through. First, the scale+min search becomes $\texttt{make\_qkx3\_quants}$ — structurally identical to $\texttt{make\_qkx2}$ (bracket from min-max, sweep the inverse-scale, solve weighted least-squares for (scale, min) at each candidate, keep the lowest weighted error) but with a tighter sweep and the importance weights threaded through every sum. Only the weights change, and that change is the whole point. Second, the super-block scale quantization should also respect importance, since a sub-block full of important weights needs its scale represented well; so the eight scales and eight mins are quantized by $\texttt{make\_qp\_quants}$ using the per-sub-block importance sums $\texttt{sw}[j] = \sum_l \texttt{weights}[l]$ as their weights, rather than the plain min-max of the reference path.

The beauty and the catch sit together. The *format* — $\texttt{block\_q4\_K}$, 4.5 bpw — is byte-for-byte identical to plain Q4_K; nothing about the stored bits changes, only *which* integers get stored, because the objective they minimize is now activation-aware. So this is a pure quality gain at *zero* in-model bpw cost after the calibration pass — finally a move straight down the frontier with no rightward step, with the gain concentrated where the activation distribution is most skewed, the same $\texttt{ffn\_down}$-type tensors the mixed allocation flagged, but fixed *inside* the quantizer rather than by spending bits. The catch is that it needs the calibration corpus, and $E[a_i^2]$ is a first-order approximation that ignores cross-channel activation correlations, so it can under-protect tensors with strong cross-channel structure; but to first order it is the right signal, and Wikitext-style calibration generalizes well for a base model. Most importantly, it unlocks what the magnitude proxy could not: at 2 bits the magnitude-weighted error is hopeless, but once the quantizer knows which coordinates matter, it can sacrifice the unimportant ones wholesale and keep the model coherent at bit-widths that were previously unusable — the door the finale walks through.

```c
static float make_qkx3_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights ? weights[0] : x[0]*x[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max <= min) {
        memset(L, 0, n);
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff*diff;
        float w = weights ? weights[i] : x[i]*x[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l  += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff*diff;
                float w = weights ? weights[i] : x[i]*x[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}
```

```c
static void quantize_row_q4_K_impl(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    assert(n_per_row % QK_K == 0);
    const int64_t nb = n_per_row / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K/32];
    uint8_t Lm[QK_K/32];
    float   weights[32];
    float   sw[QK_K/32];
    float   mins[QK_K/32];
    float   scales[QK_K/32];

    for (int i = 0; i < nb; i++) {

        float sum_x2 = 0;
        for (int l = 0; l < QK_K; ++l) sum_x2 += x[l] * x[l];
        float sigma2 = 2*sum_x2/QK_K;
        float av_x = sqrtf(sigma2);

        for (int j = 0; j < QK_K/32; ++j) {
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*i + 32*j;
                for (int l = 0; l < 32; ++l) weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);
            } else {
                for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            }
            float sumw = 0;
            for (int l = 0; l < 32; ++l) sumw += weights[l];
            sw[j] = sumw;
            scales[j] = make_qkx3_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        float d_block = make_qp_quants(QK_K/32, 63, scales, Ls, sw);
        float m_block = make_qp_quants(QK_K/32, 63, mins,   Lm, sw);
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = Ls[j];
            uint8_t lm = Lm[j];
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(d_block);
        y[i].dmin = GGML_FP32_TO_FP16(m_block);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;

    }
}
```
