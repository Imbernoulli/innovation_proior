**Problem (from step 1).** Q4_0's symmetric, zero-centered code (PPL 3.5550, +3.61% on 70B) wastes half its 16
levels on skewed blocks — runs whose values cluster away from zero get only ~8 usable levels because the rest
straddle an empty range.

**Key idea — Q4_1.** Make the code *affine* instead of symmetric: reconstruct `x ≈ d·q + m`, where `m` is the
block's own minimum. The 16 unsigned levels now tile the block's *actual* [min, max] range wherever it sits,
instead of being pinned around zero. Scan the block for both `min` and `max`; step `d = (max−min)/15`; quantize
`q = round((x−min)/d)` into [0,15].

**Why it works.** On a skewed block the affine code lays all 16 levels across the populated interval rather than
discarding the half that covers values the block doesn't contain; on a centered block it is close to the symmetric
layout while still paying for an explicit offset. So it targets exactly the asymmetric blocks transformer
matrices are full of. Cost: it must store a *second* fp16 per block (the offset `m` alongside the scale `d`), so
32 nibbles + 2×fp16 = 160 bits → **5.0 bpw**, half a bit more than Q4_0. The open question it leaves: that
half-bit bought a better offset — could it instead buy more *levels* or cheaper *shared* metadata?

**Change / code.** The per-row Q4_1 reference quantizer (`block_q4_1` carries both `d` and `m`).

```c
#define QK4_1 32
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_half) + QK4_1 / 2, "wrong q4_1 block size/padding");
```

```c
void quantize_row_q4_1_ref(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k) {
    const int qk = QK4_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + 0    + j] - min)*id;
            const float x1 = (x[i*qk + qk/2 + j] - min)*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}
```
