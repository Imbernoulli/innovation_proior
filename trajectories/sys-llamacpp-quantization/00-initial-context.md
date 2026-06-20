## Research question

A trained LLaMA-class transformer ships its weights as 16-bit floats. An 8B model is then ~15 GiB; a
70B model is ~128 GiB. That does not fit on the commodity hardware people actually own — a laptop, a
single consumer GPU, a phone — and even when it fits, the memory bandwidth to stream 16-bit weights
through every matmul is the wall that caps tokens/second. The one lever that moves both numbers at once
is the **number of bits used to store each weight**: halve the bits and you roughly halve the file, the
RAM footprint, and the bytes moved per token. So the single thing being designed here is the **weight
quantization scheme** — the exact rule that maps a row of fp16 weights to a compact integer
representation and back — pushed to the **lowest bits-per-weight at which the model's quality barely
moves**.

Everything else about the problem is fixed. The model is fixed (a released LLaMA-class checkpoint). The
quality metric is fixed: **perplexity (PPL)** of the quantized model on a fixed held-out corpus
(Wikitext-2 test), where lower is better, reported against the fp16 model's perplexity as the
no-quantization reference. The inference engine, the dequantization-into-matmul path, and the
evaluation harness are fixed. The only free variable is the quantization format: how the bits are
budgeted within a row, what is stored alongside the integers (scale, offset, codebook), and how the
quantizer chooses the integers given the floats. A scheme is "better" when it reaches a **lower
perplexity at the same or fewer bits-per-weight** — i.e. it sits lower and to the left on the
quality-vs-size frontier.

## Background

Weight quantization replaces each fp16 weight with a low-bit integer index plus a small amount of
shared metadata, so that the original value can be approximately reconstructed at inference time as
`x ≈ scale · q (+ offset)`. The reconstruction (dequantization) happens on the fly inside the matmul:
the engine reads the packed integers, expands them, and multiplies. Two facts about transformer weights
shape every scheme:

- **Weights are not uniform across a row.** A row of an attention or FFN weight matrix has a few
  large-magnitude entries and many small ones; a single scale fitted to the whole row wastes most of
  its dynamic range on the outliers and quantizes the bulk coarsely. This is why quantization is done
  per **block** — a contiguous run of weights (e.g. 32) shares one scale, so each block can adapt its
  step size to its own magnitude. Larger blocks amortize the metadata over more weights (fewer bits per
  weight) but track the local magnitude worse; smaller blocks track better but spend more bits on
  scales. The block size is the first tension every scheme negotiates.

- **Not every weight matters equally to the output.** A weight that is multiplied by activations that
  are almost always near zero contributes little to the layer's output regardless of how it is rounded;
  a weight multiplied by large, frequent activations contributes a lot. A quantizer that minimizes the
  raw weight reconstruction error spends its bits as if all weights were equally important — which they
  are not.

Quality is judged by **perplexity** on a fixed corpus: run the quantized model over the text, average
the negative log-likelihood of each true next token, exponentiate. It is monotonically sensitive to
small degradations in the output distribution, so it is the standard yardstick for "how much did
quantization hurt." Because perplexity depends on the engine's exact arithmetic, the numbers are only
meaningful **relative to the same engine's fp16 perplexity** on the same corpus, never across projects.
The companion measure is **bits-per-weight (bpw)**: total stored bits (integers + all metadata) divided
by weight count — the honest cost, because a "4-bit" scheme that also stores a 16-bit scale per 32
weights actually costs 4.5 bpw, not 4.

## Prior art before the first rung

The baseline the ladder climbs out of is the simplest thing that shrinks a row: **absmax round-to-nearest
with one scale per block.** Take a block of 32 fp16 weights, find the entry of largest magnitude, set a
single scale so that this extreme value maps to the end of the integer range, and round every weight to
the nearest integer level. Reconstruction is `x ≈ scale · q`. The format stores 32 small integers plus
one fp16 scale per block. This is the established starting point for on-device LLM inference: it is
trivial to implement, the dequantization is a single multiply, and at 8 bits per integer it is nearly
lossless. The open problem is everything *below* 8 bits. As the integer width drops to 4 bits — the
regime that actually makes an 8B model fit in a few GiB — this simplest scheme starts to bleed
perplexity, and the question becomes how to spend each bit so that the quality loss at a given
bits-per-weight is as small as possible. That frontier — lowest perplexity at the fewest bits-per-weight,
on a fixed model and a fixed corpus — is what the ladder below pushes down.

## Evaluation settings

- **Model & corpus.** A released LLaMA-class base checkpoint, evaluated by perplexity on the Wikitext-2
  test set — the convention used for judging quantization quality loss. Two checkpoints anchor the
  numbers below: a Llama-2 70B run (fp16 PPL 3.4313) and a Llama-3 8B run (fp16 PPL ≈ 6.233).
- **Primary metric.** Perplexity (lower is better), read against the fp16 model's perplexity on the
  same corpus and engine. The reported companion is `ΔPPL = PPL(quant) − PPL(fp16)` and, for the finer
  runs, the Kullback–Leibler divergence of the quantized logits from the fp16 logits (KLD, lower is
  better) — a more sensitive probe of distribution shift than perplexity alone.
- **Cost axis.** Bits-per-weight and on-disk model size (GiB), counting all stored metadata. An
  improvement must reach a **lower perplexity at the same or fewer bits-per-weight** than the rung
  below — moving down-and-left on the quality-vs-size frontier — not merely a lower perplexity bought
  with more bits.
- **Hardware.** Commodity: a single consumer GPU or CPU. Inference dequantizes the packed format inside
  the matmul; the scheme must be cheap to unpack, so formats are designed around byte-aligned blocks and
  simple reconstruction arithmetic.

## The editable interface

Exactly one thing is replaced at each rung: the per-row quantization routine and the block layout it
writes. The fixed substrate is a function that consumes a row of `n_per_row` fp32/fp16 weights and emits
a packed buffer of quantized blocks; the engine's matching dequantization reads that buffer back during
the matmul. The contract is the signature below — the body, and the `block` struct it fills, are the
free variable.

In llama.cpp this contract appears as a family of row quantizers such as `quantize_q4_0`, `quantize_q4_K`,
and `quantize_iq2_xxs`: each consumes source weights plus an optional `quant_weights` importance vector, writes
packed blocks whose layout is defined in `ggml-common.h`, and is paired with a dequantizer used by the matmul
path. The baseline fill is absmax round-to-nearest with one fp16 scale per 32-weight block.

Each rung on the ladder replaces exactly this routine and its block struct, and is scored by the perplexity
its format yields at its measured bits-per-weight.
