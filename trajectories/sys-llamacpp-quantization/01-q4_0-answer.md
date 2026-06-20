**Problem (from the baseline).** Fp16 weights are too large to fit or stream on commodity hardware; 8-bit
quantization only halves the cost. The target is ~4 bits/weight with the least perplexity loss. A single
scale over a whole row wastes resolution on outliers, so the format must work per *block* of weights.

**Key idea — Q4_0.** Symmetric, zero-offset, round-to-nearest with one scale per 32-weight block. Find the
absolute-max entry in the block; set the scale `d = max/-8` so the largest-magnitude weight maps to the
extreme negative 4-bit level (never clipped); round every weight to its nearest of 16 levels, `q = round(x/d)`,
and pack two 4-bit values per byte. Reconstruction is a single multiply, `x ≈ d·q`.

**Why it works.** Blocking lets each 32-weight run adapt its step size to its own local magnitude instead of
inheriting the row's outliers. Anchoring the absmax to the range edge guarantees the most important
(largest) weight is never clipped. Cost: 32 nibbles (128 bits) + one fp16 scale (16 bits) = **4.5 bpw**.
The soft spot is built in: a symmetric zero-offset code wastes levels on blocks whose values are skewed away
from zero, and round-to-nearest minimizes weight error, not output error — both targets for later rungs.

**Change / code.** The per-row Q4_0 reference quantizer (`block_q4_0` = one fp16 `d` + 16 packed nibbles).

```c
#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");
```

```c
void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}
```

