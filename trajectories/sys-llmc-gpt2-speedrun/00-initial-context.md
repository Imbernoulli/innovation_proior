## Research question

Reproduce **GPT-2 (124M)** pretraining to its published quality — validation loss ≈ **3.29** on a FineWeb/OpenWebText-style stream and HellaSwag accuracy ≈ **29.4** — but write the whole training stack in **C/CUDA**, with no PyTorch and no Python in the training loop. The model is fixed: 12 layers, 12 heads, 768 channels, 1024 context, 124M parameters, AdamW, the same data, and the same token budget (~10B tokens, ~18,865 steps at a ~0.5M-token batch). The target loss is fixed. The only free variable is the *systems / kernel implementation*: how the forward and backward passes are mapped onto the GPU. The objective is **tokens/sec** and **Model FLOPs Utilization (MFU)** on a fixed GPU — i.e., minimize the wall-clock time and dollar cost to reach 3.29 without changing the math the model computes.

## Prior art / Background / Baselines

A GPT-2 forward pass is a stack of twelve identical transformer blocks, each: LayerNorm → QKV projection → multi-head causal self-attention → output projection → residual add → LayerNorm → MLP up-project → GELU → MLP down-project → residual add. The backward pass mirrors this. A final classifier matmul projects the C-dim hidden state to V=50257 logits, on which softmax + cross-entropy is applied. AdamW then updates the weights. The matmuls dominate the FLOPs; the standard per-token forward+backward estimate is `6·N + 12·L·T·C` (N parameters, L layers, T context, C channels). MFU is achieved FLOPs/sec divided by the GPU's peak tensor-core FLOPs/sec.

The systems facts known before any kernel is written:

- **Modern GPUs have tensor cores and CUDA cores with very different throughput.** On an A100, generic FP32 throughput is ~19.5 TFLOPS while tensor-core BF16/FP16 throughput is ~312 TFLOPS.
- **Most non-matmul layers are memory-bound.** LayerNorm, GELU, residual adds, softmax, and elementwise AdamW updates move more bytes than they compute; their speed is set by HBM bandwidth and by how many times each byte is read/written.
- **The attention score matrix is O(T²) in memory.** Materializing the full `(B, NH, T, T)` pre-softmax and post-softmax tensors, as a textbook three-pass attention does, costs bandwidth and capacity that grow quadratically with context.
- **The vocabulary projection is the single largest activation.** The `(B, T, V)` logits tensor, with V≈50k, plus its softmax and gradient, is the largest buffer in the pass.
- **BF16 has only 8 mantissa bits.** Storing weights and applying optimizer updates in BF16 rounds away small updates compared to FP32.
- **One GPU is a throughput ceiling.** A 0.5M-token batch over 10B tokens is ~18,865 optimizer steps; multiple GPUs can share the batch by summing gradients and distributing optimizer state.

The reference CPU implementation (~1000-line single-file C) establishes that the math is simple and correct; it trains at well under a thousand tokens/sec and exists only to be matched bit-for-bit by the GPU code.

Baselines:

- **Naive one-thread-per-output FP32 CUDA port.** Core idea: transcribe each CPU loop directly into a kernel that computes one output element by looping over the contraction dimension. It runs entirely in FP32 on the CUDA cores with each input row re-read once per output channel.
- **Vendor BLAS libraries for GEMMs.** Core idea: call optimized vendor routines for matrix multiplications to get near-peak tensor-core utilization. The library selects tiled, shared-memory-staged kernels and dispatches to the tensor cores.
- **Mixed-precision training.** Core idea: run matmuls in FP16/BF16 on tensor cores while keeping master weights and reductions in FP32. BF16 avoids FP16 overflow due to its wider exponent range.
- **Textbook three-pass attention and streaming/online softmax.** Core idea: standard attention materializes the full T×T score matrix and applies a stable two-pass softmax, while online softmax keeps a running max and sum in one pass to compute the softmax without a separate pass over the scores.
- **Data-parallel SGD with all-reduce and ZeRO-style optimizer-state sharding.** Core idea: each GPU processes a different batch slice, gradients are averaged via all-reduce, and the redundant FP32 optimizer state is sharded across the data-parallel group.

## Fixed substrate / Code framework

The training stack is one C/CUDA file holding the GPT-2 model, the AdamW optimizer, the data loader, and the step loop. Each layer's forward and backward is a function that launches CUDA kernels, and the kernels are developed and benchmarked one-per-file in a `dev/cuda/` scratch tree before the fastest variant is hardcoded into the trainer.

The frozen substrate is the *math*: the layer definitions, the AdamW update rule, the loss, the fixed model/data/token budget, and the bit-for-bit CPU reference. The free slot is the *implementation* of each layer's kernels and of the multi-device step loop.

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

The starting point that fills these stubs is the naive FP32 port: `floatX = float`, every matmul a one-thread-per-output kernel looping over C, attention via a materialized T×T score matrix and a textbook two-pass softmax, the full `(B,T,V)` logits materialized for the loss, AdamW over the whole FP32 parameter buffer, on a single GPU.

## Editable interface

What can be changed: the kernel implementations behind each layer call (precision, tiling, fusion, and memory layout), the attention algorithm (whether the T×T score matrix is materialized, how the softmax is computed, and how the QKᵀ and ·V matmuls are arranged), the classifier/loss implementation (how much of the `(B,T,V)` logits/softmax/gradient is materialized), and the multi-device step loop (number of GPUs, how gradients are reduced, and where optimizer state lives).

What cannot be changed: the model architecture, the computed activations/gradients/loss up to floating-point tolerance, the training data, the token budget, the target validation loss, and the AdamW update rule.

## Evaluation settings

The yardstick is throughput on a fixed GPU at fixed model quality. Primary metrics: **tokens/sec** and **MFU** (achieved FLOPs/sec ÷ the GPU's peak tensor-core FLOPs/sec), measured per optimizer step and smoothed with a bias-corrected exponential moving average inside the trainer.

The fixed correctness bar is the GPT-2 (124M) reproduction: validation loss ≈ 3.29 over ~10B tokens (~18,865 steps) at a ~0.5M-token batch, context length 1024, checked against the bit-exact CPU/PyTorch reference at small scale before any speed claim.

Each kernel in the development tree (`dev/cuda/`) is benchmarked in isolation across launch configurations (block sizes), reporting time in ms, bandwidth in GB/s, and per-token nanoseconds, against the CPU reference for correctness. The reference hardware for the end-to-end record is a node of A100 80GB SXM GPUs (peak BF16 ≈ 312 TFLOPS/GPU); single-GPU throughput is also reported on an A100 40GB PCIe.
