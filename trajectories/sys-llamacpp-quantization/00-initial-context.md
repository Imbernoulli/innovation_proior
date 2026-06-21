## Research question

A released LLaMA-class transformer stores weights as fp16: an 8B model is ~15 GiB, a 70B model ~128 GiB. That exceeds commodity RAM and GPUs, and at inference the memory bandwidth to stream fp16 weights through every matmul caps throughput. The only variable that simultaneously shrinks the file, the RAM footprint, and the bytes moved per token is the number of bits per weight. The task is to design a **weight quantization scheme**—the rule that maps a row of fp16 weights to a compact integer representation and back—pushed to the lowest bits-per-weight at which Wikitext-2 perplexity stays near the fp16 baseline.

Everything else is fixed: the checkpoint, the engine, the dequantization-into-matmul path, and the perplexity metric. A scheme is better when it reaches lower perplexity at the same or lower bpw.

## Prior art / Background / Baselines

Weight quantization stores each weight as a low-bit integer index plus shared metadata so the original value can be reconstructed as `x ≈ scale · q`. A block of contiguous weights (e.g. 32) shares one scale, because a single row contains a few large-magnitude outliers; per-block scaling keeps the step size matched to the local bulk instead of being dominated by the extremes. The honest cost is **bits-per-weight**: stored integers plus scales and any other metadata, divided by weight count.

Baseline: **absmax round-to-nearest with one scale per block.** For each block, set scale = max|weight| / max integer level and round every weight to the nearest integer. It stores `n` small integers plus one fp16 scale per block, and dequantization is one multiply. At 8 bits it is nearly lossless; at 4 bits it leaves most quantization levels unused for the non-outlier weights, so perplexity degrades noticeably.

## Fixed substrate / Code framework

- Model: a released LLaMA-class base checkpoint.
- Engine: llama.cpp; inference dequantizes packed weights inside the matmul.
- Metric: perplexity on Wikitext-2 test, reported relative to the same engine's fp16 result.
- Hardware: commodity CPU/GPU; formats must unpack cheaply with byte-aligned blocks and simple arithmetic.

The quantizer contract is a single routine that consumes a row of fp16/fp32 weights and writes packed blocks:

```c
// Baseline block: 32 weights, one fp16 scale, 32 x 4-bit indices
struct block_q4_0 {
    ggml_half d;        // scale
    uint8_t qs[16];     // 32 4-bit weights packed byte-wise
};

void quantize_row(const float * src, void * dst, int n_per_row);
```

## Editable interface

The only free variable is the per-row quantization routine and the block layout it emits. Its body and the `block` struct it fills can be changed; the engine's dequantizer must match. The starting fill is absmax round-to-nearest with one fp16 scale per 32-weight block.

## Evaluation settings

- Model & corpus: LLaMA-class checkpoint; Wikitext-2 test set. Reference fp16 perplexities include Llama-2 70B at 3.4313 and Llama-3 8B at ~6.233.
- Primary metric: perplexity (lower is better), reported as `ΔPPL = PPL(quant) − PPL(fp16)`. Kullback–Leibler divergence of quantized logits from fp16 logits (KLD, lower is better) is used as a finer probe of distribution shift.
- Cost axis: bits-per-weight and on-disk size (GiB), counting all metadata. An improvement must move down-and-left on the quality-vs-size frontier: lower perplexity at the same or lower bpw.
- Hardware: single consumer GPU or CPU; packed format is unpacked inside the matmul.
