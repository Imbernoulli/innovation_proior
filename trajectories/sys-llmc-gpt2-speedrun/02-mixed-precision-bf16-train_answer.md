With the projections routed through cuBLAS — the matmul-forward kernel ladder ordering naive(1) $\to$ cuBLAS(2, "very fast") $\to$ cuBLASLt(3, "even faster"), and the cuBLAS attention path recorded at ~20× over the naive port — the GEMMs are on the tensor cores but running the TF32 path, because every tensor in the model is still a 4-byte float. Two costs follow. The tensor cores do BF16/FP16 matmuls at roughly twice the effective rate of TF32, since the reduced-mantissa half-precision path packs more multiply-accumulates per cycle. And every activation, weight, and gradient I touch is 4 bytes wide, so the memory-bound layers — LayerNorm, GELU, residual, softmax, the AdamW update — whose wall-clock is set by bytes crossing HBM, move twice the bytes they need to. Halving the element width is therefore a double win: faster matmuls *and* nearly half the time of every memory-bound layer, plus half the activation footprint I will want later for bigger batches.

I propose **mixed-precision training in BF16 with FP32 master weights and stochastic rounding**. Which 16-bit float matters. FP16 has a 10-bit mantissa and a 5-bit exponent; BF16 has a 7-bit mantissa and an 8-bit exponent — the *same* exponent range as FP32. That range is the whole ballgame for stability. With FP16's 5-bit exponent, gradients and activations routinely underflow to zero or overflow to inf, which is precisely why FP16 recipes need a *loss scaler*: multiply the loss by a large constant before backprop to push small gradients into FP16's range, divide it back out, and dynamically adjust the scale on overflow — a fragile subsystem. BF16, with FP32's exponent range, simply does not have that problem; the magnitudes that occur in training are all representable, so I drop the loss scaler entirely. The price is the 7-bit mantissa, ~2–3 significant decimal digits, and I have to make sure that price does not break the optimizer. So: `floatX = __nv_bfloat16` by default (FP16/FP32 selectable by a compile flag), with cuBLASLt's data type pointed at `CUDA_R_16BF`, so all GEMMs and all activation buffers are BF16.

The hazard is the optimizer update. In steady state AdamW changes a weight $w$ (an $O(1)$ number) by $-\,\text{lr}\cdot(\hat m/\sqrt{\hat v} + \lambda w)$, which at a learning rate of a few times $10^{-4}$ is itself about $10^{-4}$ in magnitude — four orders smaller than the weight. If $w$ is stored in BF16, then near magnitude 1 the spacing between representable BF16 values is about $2^{-7}\approx 8\times10^{-3}$. An update of $10^{-4}$ is far smaller than that spacing, so $w + 10^{-4}$ rounded to nearest BF16 is just $w$ again: the update vanishes. Round-to-nearest into BF16 makes the vast majority of optimizer steps silent no-ops for any weight whose update is below its local spacing — which is most weights, most of the time. Training stalls. This is the classic mixed-precision trap, and it is why storing everything in half precision works for activations and matmul math but not for the *weights*.

The fix is a master copy: keep a full FP32 version of every weight, do the AdamW arithmetic in FP32 against it so the $10^{-4}$ update accumulates faithfully across steps, and only *cast down* to BF16 for the next forward/backward. The moments $m$ and $v$ must be FP32 too — they accumulate over thousands of steps and would degrade in BF16. So per parameter I hold a BF16 live weight (2 bytes), a BF16 gradient (2 bytes), and FP32 master $+\,m\,+\,v$ (12 bytes); for a 124M model that optimizer state is small relative to the activations, and it buys correctness.

The cast-down admits one refinement over plain rounding. Round-to-nearest systematically discards the low mantissa bits the same way every step, so the BF16 weight becomes a biased, quantized shadow of the true weight, and the bias accumulates. *Stochastic rounding* instead rounds the FP32 value up or down to the two neighboring BF16 values with probability proportional to closeness, so that in expectation the BF16 weight equals the FP32 weight: the mantissa loss becomes unbiased noise that averages out over steps rather than drift that compounds. So the cast from master to live weight uses stochastic rounding, seeded per tensor.

The per-parameter update makes all of this explicit. Each thread handles one parameter. It reads the BF16 gradient and promotes it to FP32 with the gradient-unscale factor folded in; updates the FP32 moments via the lerp form of the EMA, $m \leftarrow \beta_1 m + (1-\beta_1)\,\text{grad}$ and $v \leftarrow \beta_2 v + (1-\beta_2)\,\text{grad}^2$, written as two-flop linear interpolations to save an operation; bias-corrects them to $\hat m, \hat v$; reads the *old* parameter from the master copy (or the live param if none); applies the AdamW step with decoupled weight decay — $\lambda\,w_{\text{old}}$, the "W" applied directly to the weight, not through the gradient — then writes the result *stochastically rounded* into the BF16 live parameter and writes the exact FP32 result back into the master copy. The structure that matters: FP32 everywhere the small numbers accumulate (moments, master weight, the update arithmetic), BF16 only for the live weight and gradient that feed the tensor cores, and stochastic rounding on the one lossy cast.

Nothing in this changes the math to within tolerance. The forward now runs in BF16, so the logits differ in the low bits, but the master-weight + stochastic-rounding scheme keeps the optimizer trajectory faithful, so the loss curve and the final validation loss are unchanged within training noise — verified the same way as the GEMM swap, by running a forward and a handful of AdamW steps against the reference and confirming the losses track to mixed-precision tolerance. The 3.29 bar holds; throughput roughly doubles on the matmuls and nearly halves the time of every memory-bound layer.

What is still slow: the elementwise kernels now move BF16 one element at a time through scalar loads — a 16-bit load leaves most of the memory-transaction width unused — so they are bandwidth-bound *and* under-utilizing each transaction. Attention is still the materialized-$T\times T$ path, now in BF16 but still $O(T^2)$ in HBM. The classifier still materializes the full $(B,T,V)$ logits. And it is one GPU. The next thing screaming is those memory-bound kernels loading 2 bytes at a time. The per-parameter AdamW update over FP32 master weights with stochastic rounding to BF16:

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

The precision is selected at the BLAS layer, keeping the same GEMM wrapper from the previous step:

```c
#if defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#elif defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#else // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif
```
