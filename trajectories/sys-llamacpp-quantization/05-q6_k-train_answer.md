Q4_K vindicated the super-block: +1.57% at 4.5 bpw on the 70B, beating Q4_0 at the same bpw and almost catching Q5_0 a full bit cheaper. The hierarchy is the right skeleton. But Q4_K is still a 4-bit code, and +1.57% is not nothing — for the parts of a model that are most sensitive, or for a user with the memory to spare who wants quality as close to fp16 as possible, I want the *other* end of the frontier: the highest-quality k-quant, the one whose perplexity gap is in the noise. The super-block machinery should build that cleanly by turning up the per-weight resolution while keeping the cheap shared metadata.

I propose **Q6_K**: push the weight code to 6 bits — 64 levels — inside the same 256-weight super-block. At 64 levels the half-step round-off floor is a quarter of the 4-bit floor, so the per-weight reconstruction error nearly vanishes, and a few design choices fall out of that. First, the sub-block geometry. At 4 bits I used eight sub-blocks of 32; at 6 bits the weights are already finely resolved, so the dominant residual is no longer the weight rounding but how well each sub-block's *scale* tracks the local magnitude. To track it more tightly I shrink the sub-block to **sixteen sub-blocks of 16 weights**, sixteen scales each fit to a shorter, more homogeneous run. Second, I can *drop the offset*. The min in Q4_1 and Q4_K existed because at 16 levels, wasting half the range on an empty side was catastrophic; at 64 levels that argument weakens — there is enough resolution that a symmetric code centered at zero is fine — so dropping the min halves the per-sub-block metadata and simplifies the arithmetic to $x \approx \text{scale}\cdot q$. Q6_K therefore goes back to a *symmetric* per-sub-block code, like Q5_0 but with the super-block's shared-scale trick: each sub-block has a signed scale, no min. Third, the scale quantization. With no min to store I only encode sixteen sub-block scales, and I quantize each to a *signed 8-bit* value against a single fp16 super-scale $d$ — eight bits, more than Q4_K's six, because at this quality level I do not want scale quantization to be the bottleneck, and sixteen one-byte scales over 256 weights is cheap. Counting: $256\cdot 6 = 192$ bytes of quants (128 bytes of low-nibbles $\texttt{ql}$ plus 64 bytes of high-2-bits $\texttt{qh}$), 16 bytes of int8 scales, and a 2-byte fp16 $d$ is 210 bytes over 256 weights, i.e. **6.5625 bits per weight**.

The genuinely different piece from Q4_K is the per-sub-block fit. Because the code is symmetric with no min, this is a *symmetric* scale optimization, not the affine least-squares of $\texttt{make\_qkx2}$. I want the scale $s$ minimizing $\sum_i w_i\,(s\cdot L_i - x_i)^2$ over integer levels $L_i \in [-32, 31]$ (6-bit signed). The routine $\texttt{make\_qx\_quants}$ with $\texttt{rmse\_type}=1$ does this: anchor the inverse-scale to the absmax, $\texttt{iscale} = -\texttt{nmax}/\text{max}$ with $\texttt{nmax}=32$, take initial levels, then for fixed levels compute $\texttt{sumlx} = \sum_i w_i x_i l_i$ and $\texttt{suml2} = \sum_i w_i l_i^2$ and set the closed-form least-squares scale $\text{scale} = \texttt{sumlx}/\texttt{suml2}$. It then sweeps $\texttt{iscale} = -(\texttt{nmax} + 0.1\cdot\texttt{is})/\text{max}$ over $\texttt{is} \in [-9, 9]$, re-rounding each time, and keeps the candidate with the largest $\text{scale}\cdot\texttt{sumlx}$, which is equivalently the lowest weighted error. The $\texttt{rmse\_type}=1$ setting makes the per-weight weight $w_i = x_i^2$ — magnitude-squared weighting, so the largest weights dominate the scale fit, the same spirit as Q4_K's $|x|$-weighting but for the symmetric form.

The super-block then takes the sixteen sub-block scales, finds the one of largest magnitude, and sets the super-scale so that the extreme maps to the $-128$ int8 level: $\texttt{iscale} = -128/\text{max\_scale}$, $d = 1/\texttt{iscale}$, with each sub-block scale quantized via $\texttt{nearest\_int}(\texttt{iscale}\cdot\text{scale})$ clamped to 127. Then every weight is re-rounded with its *quantized* scale $d\cdot s_q$: $l = \text{round}(x/(d\cdot s_q))$, clamped to $[-32, 31]$, stored as $l + 32$ (unsigned $0..63$). The packing splits each 6-bit value into a low nibble (into $\texttt{ql}$, two per byte) and a high 2 bits (into $\texttt{qh}$, four per byte), interleaved across the 128-weight half-super-block in the layout the dequantizer reverses.

The expected gap to fp16 is essentially negligible — well under half a percent — because at 6 bits with sixteen tightly-fit symmetric sub-scales there is almost no resolution left to lose. But this is deliberately *not* a move down-and-left on the frontier: at 6.56 bpw Q6_K is the opposite corner, the near-lossless *anchor*. With Q4_K pinned at 4.5 bpw / +1.57% and Q6_K at 6.56 bpw / near-zero, the family is now pinned end to end, with a graded set between (Q5_K). And the anchor surfaces the next lever: within one model some tensors need this precision while most are perfectly fine at Q4_K, which means the right move is not one width for the whole model but a *mixture*.

```c
// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");
```

```c
static float make_qx_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, int rmse_type,
        const float * GGML_RESTRICT qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 0; i < n; ++i) {
#else
    for (int i = 0; i < n; ++i) {
#endif
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}
```

```c
void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * GGML_RESTRICT ql = y[i].ql;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}
```
