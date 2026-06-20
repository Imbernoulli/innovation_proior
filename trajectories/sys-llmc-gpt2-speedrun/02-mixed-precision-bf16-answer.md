**Problem (from step 1).** With cuBLAS GEMMs in place, every tensor is still FP32: the matmuls run the slower
TF32 tensor-core path, and the memory-bound layers (LayerNorm, GELU, residual, softmax, AdamW) move 4 bytes per
element when they could move 2. Storage precision is leaving ~2× on the table everywhere.

**Key idea.** Store activations, weights, and gradients in **BF16** (`floatX = __nv_bfloat16`), point cuBLASLt
at `CUDA_R_16BF`. BF16 keeps FP32's 8-bit exponent range, so no loss scaler is needed (unlike FP16). But BF16's
7-bit mantissa would silently round away the tiny AdamW update (~1e-4) against an O(1) weight, so keep an
**FP32 master copy** of the weights and the moments `m`, `v`, do the optimizer arithmetic in FP32, and cast the
updated weight back to BF16 with **stochastic rounding** so the mantissa loss is unbiased noise, not drift.

**Why it works.** Half-precision storage roughly doubles matmul throughput and nearly halves the bytes (and time)
of every memory-bound layer; the activation footprint halves too. Stability is preserved by doing all the
small-number accumulation (moments, master weights, the update) in FP32 and only exposing BF16 to the
tensor-core path; stochastic rounding makes the live BF16 weight equal the FP32 weight in expectation. The
validation-loss target is held (checked against the reference). Still open: BF16 elementwise kernels load one
2-byte element per transaction (wasted bandwidth), attention still materializes T×T, the classifier still
materializes (B,T,V), single GPU.

**Change / code.** The per-parameter AdamW update over FP32 master weights with stochastic rounding to BF16
(`llmc/adamw.cuh`):

```c
template <typename Tp, typename Tg>
__device__ void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory,
                             float* m_memory, float* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2,
                             float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }
    // promote the BF16 gradient to FP32 (folding in the grad-unscale factor)
    float grad = grad_scale * (float)grads_memory[idx];
    float m = m_memory[idx];
    float v = v_memory[idx];
    m = lerp(grad, m, beta1);          m_memory[idx] = m;   // m <- b1*m + (1-b1)*grad
    v = lerp(grad * grad, v, beta2);   v_memory[idx] = v;   // v <- b2*v + (1-b2)*grad^2
    m /= beta1_correction;  // m_hat
    v /= beta2_correction;  // v_hat
    // read the old param as FP32 from the master copy if we keep one
    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
    // AdamW step: decoupled weight decay applied to the weight, not the gradient
    float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));
    // cast down to the BF16 live weight with stochastic rounding (unbiased in expectation)
    stochastic_rounding(param, &params_memory[idx], seed);
    // keep the exact FP32 result in the master copy for the next update
    if (master_params_memory != NULL) { master_params_memory[idx] = param; }
}

// the two-flop EMA / linear-interpolation form used above
__device__ float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}
```

The precision is selected at the BLAS layer, keeping the same GEMM wrapper from step 1
(`llmc/cublas_common.h`):

```c
#if defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#elif defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#else // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif
```
