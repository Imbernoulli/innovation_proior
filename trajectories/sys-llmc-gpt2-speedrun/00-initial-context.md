## Research question

Reproduce **GPT-2 (124M)** pretraining to its published quality — a validation loss of about **3.29** on the
FineWeb/OpenWebText-style stream, roughly matching the original GPT-2 small (and clearing GPT-2's HellaSwag
accuracy of 29.4) — but write the whole training stack in **C/CUDA**, with no PyTorch and no Python in the
training loop. The model is fixed: 12 layers, 12 heads, 768 channels, 1024 context, 124M parameters, AdamW,
the same data and the same token budget (~10B tokens, ~18,865 steps at the standard 0.5M-token batch). The
target loss is fixed. The **only** free variable is the *systems / kernel implementation*: how the forward and
backward passes are mapped onto the GPU. The thing being optimized is **tokens/sec** and **Model FLOPs
Utilization (MFU)** on a fixed GPU — i.e. minimize the wall-clock time and dollar cost to reach 3.29, on the
same hardware, without changing the math the model computes.

This matters because the arithmetic of a 124M transformer is trivially defined but the gap between a correct
implementation and a *fast* one is enormous: a naive but correct CUDA port and an expert-tuned one can differ
by more than an order of magnitude in throughput on identical silicon, which is the difference between a 90-minute
$20 run and a multi-day one. Every rung below is a distinct, nameable systems innovation that converts the same
fixed computation into more tokens/sec on the same GPU.

## Background

A GPT-2 forward pass is a stack of twelve identical transformer blocks, each: LayerNorm → QKV projection
(a matmul to 3C) → multi-head causal self-attention (QKᵀ, a causal softmax over the T×T scores, then the
weighted sum with V) → output projection → residual add → LayerNorm → an MLP (a matmul up to 4C, a GELU
nonlinearity, a matmul back down to C) → residual add. The backward pass mirrors it. At the end, a single big
matmul projects the C-dim hidden state to the V=50257-dim vocabulary logits, on which a softmax + cross-entropy
loss is taken. AdamW then updates the weights. The FLOP count is dominated by the matmuls; the per-token model
FLOPs for a forward+backward pass is the standard `6·N + 12·L·T·C` estimate (N parameters, L layers, T context,
C channels), and dividing achieved FLOPs/sec by the GPU's peak tensor-core FLOPs/sec gives MFU.

The facts that make this a *systems* problem, all knowable before any kernel is written:

- **Modern GPUs are dominated by their tensor cores and their memory system, not their FP32 ALUs.** On an A100,
  generic FP32 math runs at ~19.5 TFLOPS, but the tensor cores deliver ~312 TFLOPS in BF16/FP16 — a >15× gap.
  A kernel that does its multiplies in FP32 on the CUDA cores leaves almost all of the chip idle. The mfu.h
  table in any such stack encodes this: Ampere datacenter peak is recorded as 312 (TF32/BF16/FP16) vs the FP32
  lane.
- **Most non-matmul layers are memory-bound, not compute-bound.** LayerNorm, GELU, residual adds, the softmax,
  and the elementwise AdamW update move far more bytes than they do flops; their speed is set by how close they
  get to the GPU's HBM bandwidth and by how few times each byte is read/written. Reading a tensor from HBM,
  writing it, then reading it back in the next kernel is the default waste.
- **The attention score matrix is O(T²) in memory.** Materializing the full (B, NH, T, T) pre-softmax and
  post-softmax tensors to HBM, as the textbook three-pass (QKᵀ → softmax → ·V) does, is a large bandwidth and
  capacity cost that grows quadratically with context.
- **The vocabulary projection is the single largest activation.** The logits tensor is (B, T, V) with V≈50k;
  materializing it, plus its softmax and its gradient, is the biggest single buffer in the whole pass.
- **BF16 has 8 mantissa bits.** Storing weights and doing the optimizer update naively in BF16 loses small
  updates (a tiny `learning_rate·m̂/√v̂` added to a large weight rounds away), so half-precision training has a
  known stability hazard that has to be handled, not ignored.
- **One GPU is a throughput ceiling.** A 0.5M-token batch over 10B tokens is ~18,865 optimizer steps; on a
  single device that is a long wall-clock even at high MFU. Multiple GPUs can share the batch, but only if the
  gradients can be summed across devices cheaply, and the optimizer state can be made to fit.

The reference CPU implementation (a ~1000-line single-file C program) establishes that the math is simple and
correct; it trains at well under a thousand tokens/sec and exists only to be matched bit-for-bit by the GPU code.

## Baselines

The ladder climbs out of a *correct but slow* CUDA reproduction and the library-and-textbook techniques that
were on the table for speeding GPU transformer training in 2024.

- **Naive one-thread-per-output CUDA port (the FP32 reference, `train_gpt2_fp32.cu`).** Every CPU loop is
  transcribed directly into a kernel: the matmul launches one thread per output element, each thread looping
  over the contraction dimension C and reading a full row of weights and a full row of inputs from global
  memory. It is correct and it agrees with the CPU and PyTorch references to floating-point tolerance. Its
  cost: it runs entirely in FP32 on the CUDA cores (tensor cores untouched), its memory accesses are
  uncoalesced and redundant, and every layer round-trips through HBM. The matmul-forward development file
  records that the hand-rolled naive kernel is left far behind by a library call — the naive attention path,
  likewise, is ~20× slower than a cuBLAS-plus-custom-softmax version. This is the slow, honest starting line.
- **cuBLAS / cuBLASLt as the matmul oracle.** The vendor BLAS gives near-peak tensor-core GEMMs from a single
  call; the development notes mark "version 2 calls cuBLAS, very fast" and "version 3 calls cuBLASLt, should be
  even faster." It is the obvious upper bound for the matmuls and the unit of measurement for any hand-written
  kernel ("you are at 80% of cuBLAS"). What it does *not* do on its own: pick the precision, fuse the
  surrounding elementwise ops, handle attention's causal softmax, or shrink the giant logits buffer.
- **Mixed-precision training (the AMP/Apex lineage).** Running matmuls in FP16/BF16 to use the tensor cores,
  while keeping a master copy of weights and the reductions in FP32, was the established way to roughly double
  effective throughput. The known failure it must avoid is silent loss of small updates in the low-precision
  weights and overflow in FP16; BF16's wider exponent removes the overflow problem but not the rounding one.
- **The textbook three-pass attention vs. the streaming-softmax idea.** Standard attention computes the full
  T×T score matrix, runs a softmax over it (two passes: a max for stability, then the normalized exponentials),
  and multiplies by V — materializing the O(T²) scores. An online/streaming formulation of the softmax that
  keeps a running max and running sum in one pass over the row was known; fusing that with the matmuls so the
  T×T matrix never hits HBM was the direction the field was moving (the cuDNN library was beginning to ship a
  fused attention primitive).
- **Data-parallel SGD with all-reduce, and the ZeRO memory-sharding line.** To use more than one GPU, the
  standard recipe is data parallelism: each GPU runs the same model on a different slice of the batch and the
  gradients are averaged with an all-reduce before the optimizer step. The ZeRO line of work pointed out that
  in plain data parallelism every GPU redundantly stores the full optimizer state (the FP32 master weights plus
  AdamW's two moments — about 12 bytes per parameter), and that this redundancy can be sharded across the data-
  parallel group. These were known designs; wiring them into a single-file C/CUDA trainer, and making the
  communication cheap enough not to erase the speedup, was the open work.

## Evaluation settings

The yardstick is throughput on a fixed GPU at fixed model quality. Primary metrics: **tokens/sec** and **MFU**
(achieved FLOPs/sec ÷ the GPU's peak tensor-core FLOPs/sec; higher is better), measured per optimizer step and
smoothed with a bias-corrected exponential moving average inside the trainer. The fixed correctness bar is the
GPT-2 (124M) reproduction: validation loss ≈ 3.29 over ~10B tokens (~18,865 steps) at a ~0.5M-token batch,
context length 1024, which is checked against the bit-exact CPU/PyTorch reference at small scale before any
speed claim. Each kernel in the development tree (`dev/cuda/`) is benchmarked in isolation across launch
configurations (block sizes), reporting `time … ms`, bandwidth (GB/s), and per-token nanoseconds, against the
CPU reference for correctness. The reference hardware for the end-to-end record is a node of A100 80GB SXM GPUs
(peak BF16 ≈ 312 TFLOPS/GPU); single-GPU throughput is also reported on an A100 40GB PCIe.

## Code framework

The training stack is one C/CUDA file holding the GPT-2 model, the AdamW optimizer, the data loader, and the
step loop; each layer's forward and backward is a function that launches CUDA kernels, and the kernels
themselves are developed and benchmarked one-per-file in a `dev/cuda/` scratch tree before the fastest variant
is hardcoded into the trainer. The frozen, never-edited substrate is the *math* — the layer definitions, the
AdamW update rule, the loss, and the fixed model/data/token budget — verified bit-for-bit against a CPU
reference. The free slot is the *implementation* of each layer's kernels and of the multi-device step loop.

```c
// The fixed substrate: a transformer block's forward, expressed as layer calls.
// Each call launches CUDA kernels. The MATH is fixed and checked against the CPU reference;
// the kernel IMPLEMENTATIONS behind these calls are the editable surface.

typedef float floatX;  // the activation/weight datatype — TODO: which precision do we run in?

void encoder_forward(floatX* out, int* inp, floatX* wte, floatX* wpe, int B, int T, int C);
void layernorm_forward(floatX* out, floatX* mean, floatX* rstd, floatX* inp, floatX* w, floatX* b, int B, int T, int C);

// the projection matmuls (QKV, attention output, MLP up, MLP down, final logits)
void matmul_forward(floatX* out, floatX* inp, floatX* weight, floatX* bias, int B, int T, int C, int OC) {
    // TODO: how is this matrix multiply implemented on the GPU?
}

// multi-head causal self-attention
void attention_forward(floatX* out, floatX* qkv, int B, int T, int C, int NH) {
    // TODO: how are QK^T, the causal softmax, and the (·V) computed and laid out in memory?
}

void gelu_forward(floatX* out, floatX* inp, int N);
void residual_forward(floatX* out, floatX* inp1, floatX* inp2, int N);

// final loss: project hidden state to V-dim logits, softmax + cross-entropy
void classifier_forward(floatX* losses, floatX* logits, int* targets, int B, int T, int V) {
    // TODO: how much of the (B, T, V) logits / softmax / gradient is materialized?
}

// the optimizer step over all parameters
void adamw_update(floatX* params, floatX* grads, float* m, float* v, size_t num_params, /* hypers */);

// the training step loop
void train_step(GPT2* model, int* inputs, int* targets) {
    gpt2_forward(model, inputs, targets);
    gpt2_backward(model);
    // TODO: across how many GPUs, and how are gradients and optimizer state shared between them?
    gpt2_update(model /* , hypers */);
}
```

The starting point that fills these stubs is the naive FP32 port: `floatX = float`, every matmul a
one-thread-per-output kernel looping over C, attention via a materialized T×T score matrix and a textbook
two-pass softmax, the full (B,T,V) logits materialized for the loss, AdamW over the whole FP32 parameter
buffer, on a single GPU. Each rung below replaces one of these implementation choices with a faster one while
leaving the computed result — and the target loss — unchanged.
