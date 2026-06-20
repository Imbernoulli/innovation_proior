## Research question

A pretrained large language model already exists — say a LLaMA / Llama-2 of 7B to 70B parameters,
trained at full cost — and the task is to make it cheaper to *serve* without touching its weights as
learned. Concretely: **post-training quantization (PTQ) of a fixed pretrained LLM to low bit-width
with the smallest possible loss in quality**, where quality is measured by **WikiText-2 perplexity**
(lower is better) and by **zero-shot accuracy** averaged over standard commonsense-reasoning tasks
(higher is better). The model architecture, its trained weights, and the evaluation harness are all
held fixed. The single free variable is the **quantization method**: how the FP16 tensors are mapped
to a low-bit integer grid, and what preprocessing is allowed before that mapping.

"Post-training" is the binding constraint and what makes this hard. We are not allowed to retrain the
network or run gradient descent over its weights on the training corpus; at most we get a tiny
**calibration set** (a few hundred sequences) and a few hours of single-GPU compute to fit
quantization parameters. Everything else about the model is frozen. The bit-width budget is the lever
we push down on: 8-bit is nearly free, but every bit below that costs accuracy, and the goal of each
rung on this ladder is to give back fewer points of perplexity / accuracy at the *same* bit-width than
the previous method did.

Two distinct axes of difficulty appear as the budget tightens, and the ladder climbs through both:

- **Weight-only quantization (WxA16).** Quantize the weight matrices to 3–4 bits and keep activations
  in FP16. This already halves-or-quarters the memory footprint and speeds up *batch-1 generation*,
  which is memory-bandwidth-bound (the bottleneck is reading weights from DRAM, not the arithmetic).
- **Weight-and-activation quantization (WxAx).** Quantize *both* operands so the matmul itself runs on
  integer tensor cores (INT8, and eventually INT4). This is what speeds up *compute-bound,
  large-batch* serving — but it forces us to quantize the activations, which carry brutal outliers
  that weight-only methods never had to confront.

The metric the ladder is ranked on is the WikiText-2 perplexity (and the zero-shot average) at a
stated bit-width; each rung is a real published method that lowered the quality loss at its target
bit-width, told as the discovery that got us there.

## Prior art before the first rung

The baseline every rung climbs out of is the simplest possible quantizer: **round-to-nearest (RTN)**.
Pick a per-channel (or per-group) scale from the max magnitude of the tensor, divide, round to the
integer grid, clamp. No calibration, no reconstruction, no search — just rounding. RTN is the thing
that "already scales": it costs nothing and it works fine at 8 bits, where the rounding error is small
relative to the signal. The trouble is the low-bit regime. At 4-bit weight-only with a fine group size
(g128) RTN is still tolerable, but at **3-bit per-channel** the rounding error swamps the smallest
weights and perplexity blows up: round-to-nearest on a 3-bit per-channel LLaMA-7B gives WikiText
perplexity **25.54**, against an FP16 reference of about **5.68** — an enormous, unusable gap. And the
moment we try to quantize *activations* to 4 bits, RTN is not merely bad but catastrophic: W4A4 RTN on
Llama-2-7B produces perplexity on the order of **2×10³** (effectively a broken model), because a
handful of activation channels carry values ~100× larger than the rest and a single per-tensor scale
cannot represent both them and the bulk.

So the field starts here, with a method that is free and correct in spirit but loses far too much at
the bit-widths that actually save memory and compute. Every rung below is a named method that closes a
specific part of that gap:

1. the second-order error compensation that fixes weight rounding (GPTQ),
2. the activation-aware scaling that protects the weight channels that matter (AWQ),
3. the offline migration of activation outliers into the weights that first makes W8A8 work
   (SmoothQuant),
4. the computational-invariance rotations that finally make true 4-bit *activations* work (QuaRot),
5. and the *learned* rotations that squeeze the last variance out of the rotation idea (SpinQuant).

## The fixed substrate

The model is a fixed pretrained Transformer LLM (the headline numbers below are on the LLaMA / Llama-2
family at 7B, 13B, and 70B). Quantization is uniform integer quantization on a per-channel or
group-wise grid; the calibration set is a few hundred sequences (e.g. from WikiText-2 or the Pile),
used only to fit scales / Hessians / rotations, never for backprop on the model's own weights.
Evaluation is WikiText-2 perplexity at sequence length 2048 and a fixed suite of zero-shot reasoning
tasks. The bit-width and the granularity (per-channel vs group-of-128, written *g128*) are stated
explicitly for every number, because they are not comparable across settings — a g128 number is
always easier than the per-channel number at the same nominal bits. This frozen harness is the
scaffold; each rung is a single named quantization method dropped into it.

## Evaluation settings

The ranking metric is **WikiText-2 perplexity (lower is better)** at a stated weight/activation
bit-width, with **zero-shot accuracy (higher is better)** as the secondary metric where the source
table reports it. Every number below is copied from the named method's own paper table or repository
README — none is re-run by us — and every number is labeled with its exact bit-width and grouping
(per-channel vs g128, weight-only WxA16 vs weight+activation WxAx). Because the ladder spans two
regimes, the rungs are *not* all on one comparable axis: rungs 1–3 are weight-only (the perplexities
are directly comparable within a bit-width/grouping), rung 4 is the W8A8 activation-quantization
breakthrough (ranked on zero-shot accuracy at INT8), and rungs 5–6 are the true 4-bit-activation
(W4A4KV4) regime, anchored by the QuaRot Table 1 multi-method comparison so SmoothQuant, QuaRot, and
SpinQuant are read off the *same* table at the *same* W4A4 setting. The bit-width and grouping are
stated in each feedback file so no two incomparable numbers are silently compared.
