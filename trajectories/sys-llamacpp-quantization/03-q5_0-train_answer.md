Q4_1 brought the 70B gap down to +2.37% from Q4_0's +3.61%, but it did it by spending half a bit — 5.0 bpw against 4.5 — on a second fp16 per block, and that whole half-bit went into a per-block *offset*. The open question from that rung was whether the same half-bit, spent differently, buys more, and there is an obvious rival use: resolution. On the blocks where Q4_1's offset already places the 16 levels well, the residual error is no longer asymmetry — it is round-off. The level spacing is $(\text{max}-\text{min})/15$, and every weight is rounded to within half a step of its true value; that half-step is the irreducible noise floor of a 16-level code, and an offset cannot shrink it. To push the floor down I do not need a better-placed code, I need a *finer* one.

I propose **Q5_0**: a symmetric 5-bit code, 32 levels instead of 16, which halves the spacing and so halves the half-step rounding error. I deliberately go back to Q4_0's symmetric form — anchor the absmax, reconstruct $x \approx d\cdot q$, no offset — rather than keeping Q4_1's affine min and *also* adding a bit, because I want to isolate the resolution effect and keep the metadata cheap. A symmetric 5-bit code stores just one fp16 scale per block, the same as Q4_0, so the entire extra bit goes into levels. With 5 bits the signed range is $-16..+15$, so the scale anchors the extreme to the $-16$ end, $d = \text{max}/{-16}$, and the largest-magnitude weight still maps cleanly to the edge, never clipped. Pitting Q5_0 at 5.5 bpw against Q4_1 at 5.0 is then a clean test of which lever — more levels or a per-block offset — gives more perplexity per bit.

The reason this is a distinct *format* and not a trivial tweak is the packing. Four bits pack two-per-byte and everything stays byte-aligned; five bits do not divide a byte. The clean trick is to split each 5-bit value into a low nibble (4 bits) and a single high bit. The low nibbles pack exactly like Q4_0 — two per byte, 16 bytes for 32 weights — and the 32 leftover high bits are gathered into a separate 32-bit field $\texttt{qh}$, one bit per weight. Reconstruction reads the nibble, ORs in the corresponding bit from $\texttt{qh}$ shifted into position, subtracts 16, and multiplies by $d$.

The quantize step has to do that bit-splitting carefully. The block is processed in two halves of 16, as in Q4_0; for each of the 16 nibble positions I take a low-half weight and a high-half weight, scale each by $id = 1/d$, and round-and-bias into $[0,31]$ with $+16.5$ clamped to 31 — round-to-nearest into the 5-bit unsigned range, the same structure as Q4_0's $+8.5$ but for 32 levels. Each quantized value $\texttt{xi}$ is now $0..31$. Its low four bits $\texttt{xi \& 0x0F}$ go into the nibble (low-half value in the low nibble, high-half value in the high nibble), and its fifth bit $(\texttt{xi \& 0x10}) \gg 4$ goes into $\texttt{qh}$ at the weight's global position — bit $j$ for the low-half weight, bit $j + 16$ for the high-half — after which the 32-bit $\texttt{qh}$ is copied into the block.

Counting: per block of 32, 16 bytes of nibbles plus 4 bytes of $\texttt{qh}$ plus one fp16 scale is $128 + 32 + 16 = 176$ bits over 32 weights, i.e. **5.5 bits per weight**. The round-off argument says this should land clearly below Q4_1, since 32 levels genuinely halve the per-weight rounding error and on a near-symmetric block — which the symmetric code handles natively — that error is the dominant residual. But I am also clear-eyed about the cost-axis story, because it is now stark: Q4_0 at 4.5, Q4_1 at 5.0, Q5_0 at 5.5 — three formats marching *up* the bits axis, each better than the last, none cheaper. The real prize is lower perplexity at *lower* bpw, and none of these three reach toward it; they all answer "how do I spend more bits well." Every one of them also pays a fixed fp16 (or two) per 32-weight block, and that per-block overhead is a fat slice of the budget at these widths. Sharing the scale metadata across a *larger* group while keeping a fine per-sub-block step is the move that would break the pattern — but that is the next fight. First, measure resolution against offset.

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
