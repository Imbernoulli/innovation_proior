An fp16 LLaMA-class model is too large to fit or to stream on the hardware people actually own: a 70B checkpoint is ~128 GiB, and every token has to drag all of those weights through the memory bus. Eight-bit-per-weight quantization already solves half the problem essentially losslessly — store each weight as an int8 with one shared scale per block, reconstruct as $x \approx d\cdot q$ — but halving is not enough. To make a large model fit on a single consumer card I need to get down near 4 bits per weight, and that is precisely where the simple recipe begins to bleed. The question that opens the ladder is therefore narrow and concrete: at 4 bits, what is the *least* perplexity I have to give up, and what is the cheapest format that gives it up?

I propose **Q4_0**: a symmetric, zero-offset, round-to-nearest code with one fp16 scale per 32-weight block. The first design decision is to quantize per *block* rather than per row. A full attention or FFN row of several thousand weights is not magnitude-homogeneous — a handful of large entries sit on top of a long tail of small ones — and a single row-wide scale would let those one or two outliers set the step size while the entire bulk gets rounded to mush. So I chop the row into contiguous runs of 32 and give each run its own scale, short enough that the magnitude inside it is roughly uniform, long enough that one scale amortizes over 32 weights.

Within a block I use 16 integer levels (4 bits) laid out *symmetrically* around zero, because the weights are roughly zero-centered: reconstruction is the single multiply $x \approx d\cdot q$ with no offset to store. To pin the scale I look at the entry of largest absolute value and force it to land at the extreme of the integer range, so the most important weight in the block — the one that dominates its contribution — is never clipped, clipping it being the single most damaging rounding error available. With a signed 4-bit range of $-8..+7$ I anchor that absmax entry to $-8$, giving $d = \text{max}/{-8}$ where $\text{max}$ is the *signed* value at the position of largest magnitude. Anchoring to $-8$ rather than $+7$ deliberately uses the one extra negative slot the range carries, so whatever the sign of the extreme weight, it gets the full-resolution end. Each weight then becomes $q = \text{round}(x \cdot id)$ with $id = 1/d$; concretely I shift by $+8$ and round in one step, $\texttt{(int)(x}\cdot\texttt{id + 8.5)}$, clamped to 15 so nothing overflows the nibble. Two 4-bit values pack into one byte.

Counting the true cost is the honest part, because "4-bit" is a lie if the metadata is ignored. Per block of 32: 16 bytes of nibbles ($128$ bits) plus one fp16 scale ($16$ bits) is $144$ bits over 32 weights, i.e. **4.5 bits per weight**. That extra half-bit is the scale overhead, and at block size 32 it is unavoidable — it is the price of letting each block adapt. A larger block would dilute the fp16 but break the magnitude-homogeneity assumption inside the block, so 32 is the compromise. I am clear about the soft spot I am building in: this is round-to-nearest under *unweighted* squared error, which minimizes the error in the weights rather than in what the layer outputs, and a symmetric zero-offset code wastes half its 16 levels on any block whose values are skewed away from zero. Both are real, and both are the targets of later rungs. But symmetric round-to-nearest is the right *first* rung — the minimal real format, trivial to dequantize (one multiply, no add) — and it fixes the baseline perplexity at ~4.5 bpw that everything else must beat.

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
