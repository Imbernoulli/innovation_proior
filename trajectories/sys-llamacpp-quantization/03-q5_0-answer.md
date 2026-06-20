**Problem (from step 2).** Q4_1 cut perplexity by spending a half-bit on a per-block offset (5.0 bpw, +2.37% on
70B). But on blocks where the offset already fits, the residual error is *round-off* from having only 16 levels —
an offset can't shrink it. The rival use of bits is resolution, not offset.

**Key idea — Q5_0.** A symmetric 5-bit code: 32 levels instead of 16, which halves the spacing `(max)/16` between
adjacent levels and so halves the half-step rounding error. Keep Q4_0's cheap metadata — one fp16 scale per block,
no offset, anchor `d = max/-16` so the absmax maps to the −16 end — and pour the extra bit entirely into levels.

**Why it works.** Five bits don't pack into a byte, so each 5-bit value is split: its low 4 bits join the nibble
array (two per byte, as in Q4_0) and its single high bit goes into a separate 32-bit field `qh`, one bit per
weight. Reconstruction ORs the `qh` bit back onto the nibble. This isolates the resolution lever from the offset
lever — Q5_0 (5.5 bpw, one scale) vs Q4_1 (5.0 bpw, scale+min) measures which buys more quality per bit. Cost:
16 nibble-bytes + 4 `qh`-bytes + one fp16 = 176 bits → **5.5 bpw**. The unspent worry: like Q4_0/Q4_1 it pays a
fixed fp16 per 32-weight block — the per-block metadata overhead is still untouched.

**Change / code.** The per-row Q5_0 reference quantizer (`block_q5_0` adds a 4-byte `qh` for the fifth bits).

```c
#define QK5_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_half) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");
```

```c
void quantize_row_q5_0_ref(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK5_0;

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

        const float d  = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        uint32_t qh = 0;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(qh));
    }
}
```

