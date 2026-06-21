Q4_0 landed at perplexity 3.5550 on the 70B model against fp16's 3.4313 — a +3.61% gap at 4.5 bits per weight — and that is the worst gap of any of the 4-bit-and-up formats I can put next to it. I know where the bleeding is, because it is the assumption I built in: the code is symmetric and zero-centered. A 32-weight block is not, in general, symmetric around zero. Plenty of blocks are skewed — values clustered in, say, $[-0.02, +0.31]$, mostly positive with a thin negative tail — and a code centered at zero spends half its 16 levels on the empty $[-0.31, 0]$ side and crams the entire populated mass into the other eight. Half the resolution is thrown at values that do not exist.

I propose **Q4_1**: stop pretending the block is centered and let it carry its own offset. The reconstruction becomes *affine*, $x \approx d\cdot q + m$, where $m$ is the block's actual minimum, so the 16 integer levels no longer have to straddle zero — they tile the block's *real* range, from its min to its max, wherever that range happens to sit. A block in $[-0.02, +0.31]$ now gets all 16 levels laid across that interval instead of half of them wasted. This is the move from a symmetric (absmax) quantizer to an asymmetric (affine) one, and for skewed data it is strictly spending more of its levels on populated values.

The scale and offset follow directly from wanting level 0 at the minimum and level 15 at the maximum, evenly spaced. So I scan the block for *both* $\text{min}$ and $\text{max}$ — not the absmax, since I need both ends now — and set the step to $d = (\text{max} - \text{min})/15$, fifteen intervals between sixteen levels, with offset $m = \text{min}$. To quantize a weight I subtract the min to bring it to $[0, \text{max}-\text{min}]$, divide by the step, and round to nearest into $[0,15]$: $q = \text{round}((x - \text{min})/d)$, clamped at 15. Reconstruction inverts it, $x \approx d\cdot q + \text{min}$. Note that the code is now *unsigned* 4-bit — levels $0..15$ with no sign bit — because the sign information has been absorbed into where $\text{min}$ sits. The rounding is also cleaner than Q4_0's: there is no implicit shift to undo, so $(x-\text{min})\cdot id$ already lands in $[0,15]$ and round-to-nearest is just $+0.5$ then truncate, clamped at 15.

This is not free, and counting the cost is the part that frames the rung. Q4_0 stored one fp16 per block; Q4_1 must store *two* — the scale $d$ and the min $m$ — because the offset is genuinely per-block, different blocks sitting in different ranges, and a global offset would defeat the purpose. So per block of 32: 16 bytes of nibbles plus two fp16 is $128 + 32 = 160$ bits over 32 weights, i.e. **5.0 bits per weight**, half a bit more than Q4_0. I am buying the asymmetry fix with an extra fp16 per block. The bet is that never wasting levels on empty range outweighs that half-bit, and I lean yes because transformer matrices are full of skewed blocks — biased channels, gated activations feeding asymmetric distributions — where Q4_0's wasted-levels problem is severe. But I am explicit that this only counts as progress if the perplexity drop is large enough to justify the step *right* on the cost axis, and it leaves a sharp open question: that half-bit bought a better offset — could the same half-bit instead buy more *levels*, or cheaper *shared* metadata? Those are the next two rungs.

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
