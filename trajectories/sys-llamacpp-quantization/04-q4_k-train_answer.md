Q5_0 cut the 70B gap to +1.26% at 5.5 bpw, decisively beating Q4_1's +2.37%, so resolution is the stronger lever. But three rungs in, the pattern is undeniable and it points the wrong way: Q4_0 at 4.5 bpw, Q4_1 at 5.0, Q5_0 at 5.5 — each one better, every one bigger. The only knob any of them found for quality was *add bits*, and I have already traced why. Every format pays a fixed fp16 scale (Q4_1 two of them) per 32-weight block, and at these widths that per-block metadata is a fat slice of the budget — one fp16 over 32 weights is 0.5 bits/weight of pure overhead, exactly the half-bit gap between Q4_0's "4-bit" label and its real 4.5. If I want lower perplexity at *lower* bpw, I have to attack that overhead, not keep feeding it. The trap is that small blocks adapt better but cost more scales, while big blocks dilute the scale but fit a heterogeneous range badly.

I propose **Q4_K**, the first k-quant: decouple the *granularity* of the scale from its *storage cost*. Keep small 32-weight sub-blocks, each with its own scale and its own min so the step adapts finely — but group eight of them into a **256-weight super-block** that stores only *one* pair of fp16 numbers, a super-scale $d$ and a super-min $\texttt{dmin}$. Each sub-block's scale and min are then stored *relative* to those, as small 6-bit integers. The fine-grained adaptation survives; the expensive fp16 is paid once per 256 weights instead of once per 32. Counting the budget makes the payoff concrete: 256 weights at 4 bits is $256\cdot 4 = 1024$ bits = 128 bytes of quants; the eight sub-block scales and eight sub-block mins quantize to 6 bits each, packed into 12 bytes total; the super-block carries two fp16 = 4 bytes. That is $128 + 12 + 4 = 144$ bytes over 256 weights, i.e. **4.5 bits per weight** — the same nominal width as Q4_0, but now with *eight* independently fitted affine scale+min pairs per 256 weights where Q4_0 had one coarse symmetric scale per 32. This is Q4_1's affine adaptation and Q5_0-like fine granularity at Q4_0's bits-per-weight, and that is the frontier finally moving down-and-left.

Two things make it work. The first is the two-level quantization. There are now two rounding stages: the weights round to 4-bit indices using their sub-block's own (scale, min), and then the sub-block's real-valued (scale, min) round to 6-bit indices using the super-block's $(d, \texttt{dmin})$. I quantize the eight scales against $d$ and the eight mins against $\texttt{dmin}$, packing the 6-bit results into 12 bytes with a bit-interleaving layout — the first four scales and mins occupy the low 6 bits of their bytes, the last four are split across the high 2 bits of earlier bytes, and the dequantizer's $\texttt{get\_scale\_min\_k4}$ reverses it. The risk here is that quantizing a sub-block's scale to 6 bits injects a *second* rounding error into the scales, but 6 bits over a super-block's eight scales is ample resolution for their dynamic range, so the net is strongly positive.

The second, and the part that lifts this above merely "cheaper metadata," is how each sub-block's scale and min are *chosen*. Q4_0 and Q4_1 used a min-max heuristic that minimizes nothing in particular; it just guarantees no clipping. Now that I am fitting a (scale, min) pair per 32-weight sub-block, I can *search* for the pair that minimizes the reconstruction error, and make it a *weighted* error so the larger-magnitude weights — which dominate the block's contribution — count more. I set each entry's weight to roughly its own magnitude plus the block's average, $w_i \approx \sqrt{\overline{x^2}} + |x_i|$, so small entries are not ignored entirely, and minimize

$$\sum_i w_i\,(\text{scale}\cdot L_i + \text{min} - x_i)^2$$

over the integer levels $L_i$ and the continuous (scale, min). This is $\texttt{make\_qkx2\_quants}$. It brackets from the min-max initial scale, then sweeps the inverse-scale over a small grid, $\texttt{iscale} = (\texttt{rmin} + \texttt{rdelta}\cdot\texttt{is} + \texttt{nmax})/(\text{max}-\text{min})$ for $\texttt{is}$ from 0 to $\texttt{nstep}$, and for each candidate re-rounds the levels and solves the weighted least-squares for the optimal (scale, min) given those fixed levels. With the sums $\Sigma w$, $\Sigma wL$, $\Sigma wL^2$, $\Sigma wx$, $\Sigma wLx$ and determinant $D = \Sigma w\cdot\Sigma wL^2 - (\Sigma wL)^2$, the closed form is $\text{scale} = (\Sigma w\cdot\Sigma wLx - \Sigma wx\cdot\Sigma wL)/D$ and $\text{min} = (\Sigma wL^2\cdot\Sigma wx - \Sigma wL\cdot\Sigma wLx)/D$; a positive $\text{min}$ is clamped to 0 and the scale refit alone, because reconstruction subtracts the min and a wrong-signed offset is invalid. The candidate with the lowest weighted error wins. This is a genuine optimizer over (scale, min, levels), not a heuristic anchor, and the magnitude weighting means precision is spent where the block's energy actually is.

So the super-block routine: for each of the eight sub-blocks set the per-entry weights to $\overline{x} + |x_i|$ and call the weighted scale+min search to get that sub-block's scale, min, and 4-bit levels; collect the eight scales and eight mins; quantize those sixteen numbers to 6-bit against a shared $d$ and $\texttt{dmin}$; pack the 6-bit indices into the 12-byte field and the 4-bit weight levels into 128 bytes. Dequant reverses it — unpack a sub-block's 6-bit scale and min, multiply by $d$ and $\texttt{dmin}$ to recover the real (scale, min), and reconstruct each weight, with the stored min being the nonnegative subtracted offset.

```c
// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");
```

```c
static float make_qkx2_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
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
            float w = weights[i];
            sum_l += w*l;
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
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
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
void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

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
