**Problem (from step 3).** Each memory-bound kernel is now fast, but adjacent ones still pass data through HBM:
the residual-add kernel writes the residual sum to global memory and the LayerNorm kernel immediately reads it
back — a full (B,T,C) round-trip whose only purpose is to be normalized. Two LayerNorm seams per layer × 12
layers × every step makes this a large, pure-waste traffic.

**Key idea.** Data can only pass *between* kernel launches via global memory, so **fuse** residual-add and
LayerNorm into one kernel: read `inp1` and `inp2`, form `res = inp1 + inp2`, accumulate `Σres` and `Σres²` in
the same pass (single-pass variance `var = mean(res²) − mean(res)²`), write `res` out (it's the residual stream
+ backward), then normalize using `res` *still staged in shared memory* — never reading the sum back from HBM.
Assign a warp (not one thread) per token so the C-row is loaded coalesced with 128-bit packets; do the
reductions in FP32.

**Why it works.** Fusing deletes one full (B,T,C) write and one full (B,T,C) read (the residual round-trip)
versus running the two kernels separately — a direct cut to a bandwidth-bound op's traffic, repeated at every
LayerNorm seam. Warp-per-row + shared-memory staging avoids the naive one-thread-per-row "uncoalesced access
pattern [that] leads to terrible performance," and FP32 reductions keep the statistics accurate over 768 BF16
values. Bit-faithful to `LayerNorm(inp1+inp2)`, so the 3.29 target holds. Still open: attention still
materializes the O(T²) score matrix (the largest structural memory cost), the classifier still materializes
(B,T,V).

**Change / code.** The fused residual+LayerNorm kernel: warp-per-row, weights/bias and the residual row staged
in shared memory, FP32 reductions, 128-bit I/O (`dev/cuda/fused_residual_forward.cu`):

```c
__global__ void fused_residual_forward_kernel5(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C) {
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);
    // stage LayerNorm weight/bias (reused across all rows) and this row's residual in shared memory
    extern __shared__ char params[];
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias   = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_res    = reinterpret_cast<x128*>(params) + ((2 + threadIdx.y) * C / x128::size);

    int sidx = (threadIdx.x + WarpSize * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WarpSize * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size]   = load128(bias + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx > N) return;
    residual += C * idx; normed += C * idx; inp1 += C * idx; inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    // pass 1: residual sum, write it out, stage it in shared mem, accumulate Sum(res) in FP32
    for(int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) { out[k] = (floatX)((float)in1[k] + (float)in2[k]); sum += (float)out[k]; }
        store128cs(residual + c, out);          // residual stream, needed downstream
        s_res[c / x128::size] = out;            // keep on-chip for the normalize
    }
    sum = warpReduceSum(sum);
    float m = sum / C;
    float v = 0.f;
    // pass 2 reads res from SHARED MEMORY, not HBM
    for(int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        for(int k = 0; k < x128::size; ++k) { v += ((float)res[k] - m) * ((float)res[k] - m); }
    }
    v = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);
    // pass 3: normalize + affine, again reading res from shared memory
    for(int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m);
            out[k] = n * (float)w[k] + (float)b[k];
        }
        store128cs(normed + c, out);
    }
    if(threadIdx.x == 0) { mean[idx] = m; rstd[idx] = s; }   // cache stats for backward
}
```
