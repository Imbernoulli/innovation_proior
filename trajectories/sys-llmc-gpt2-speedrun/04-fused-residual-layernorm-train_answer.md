Every memory-bound kernel now moves 128-bit packets at close to peak bandwidth (the development ladder running naive port $\to$ 128-bit packed reads $\to$ coalesced warp-per-row), but I keep paying for the same byte twice because of where the kernel boundaries fall. Look at the seam between a sublayer and the next: the attention (or MLP) output projection produces a tensor, then a residual-add kernel reads that tensor and the block's input and writes their sum back to HBM, then the LayerNorm kernel reads that sum *back* from HBM, computes its statistics, normalizes, and writes the result. The residual sum is written out and immediately read back — a full $(B,T,C)$ round-trip whose only purpose is to be normalized. In a 12-layer model with two LayerNorms per layer, that is a great many useless round-trips. The reason it exists is purely that residual-add and LayerNorm are two separate launches, and the only way data passes *between* kernel launches is global memory: no register or shared-memory state survives a kernel boundary.

I propose the **fused residual + LayerNorm kernel**. If I want to skip the round-trip, the two operations must live in *one* kernel that reads the two inputs, computes the residual sum, and — without writing that sum out and reading it back — immediately computes its LayerNorm, writing out both the residual (needed unchanged for the next block's residual stream and for the backward pass) and the normalized output (needed by the next matmul). One read of the inputs, the add and the normalize done while the data is still on-chip, two writes; the intermediate sum never makes a round-trip.

The fusion is well-posed even though LayerNorm is not pure elementwise — it has a reduction over the channel dimension $C$, so I must see all $C$ values of a row before normalizing any of them. So the kernel: (1) reads the $C$ values of `inp1[row]` and `inp2[row]`; (2) forms $\text{res} = \text{inp1} + \text{inp2}$ while accumulating $\sum \text{res}$ in the same pass; (3) writes `res` out (it is the residual stream, needed downstream) and stages it in shared memory; (4) computes $m = \text{mean}(\text{res})$ and, in a second pass reading `res` *from shared memory*, the variance $v$, then $\text{rstd} = 1/\sqrt{v+\varepsilon}$; (5) a third pass, again reading `res` from shared memory, writes $(\text{res} - m)\cdot\text{rstd}\cdot\text{weight} + \text{bias}$. The point is that between producing `res` and consuming it for the normalize, `res` stays on-chip. I also stage the LayerNorm `weight` and `bias` — reused across all rows in the block — into shared memory so every row's normalize hits on-chip memory rather than HBM.

The design choice that decides whether this is fast is the thread-to-row mapping. The naive fusion — one thread per row, looping over $C$ — is correct but has a terrible access pattern: adjacent threads handle adjacent *rows*, so within a warp the $C$-loop strides across rows and the loads are uncoalesced (the development file is blunt that this "uncoalesced access pattern leads to terrible performance"). The fix is to put a *group of threads* on each row — a warp cooperates on one token's $C$-vector, each lane handling a strip of channels, and the mean/variance reductions become warp reductions. With $C=768$ and 128-bit packed loads, a warp of 32 threads each loading a couple of `x128` packets covers the row in a coalesced sweep. So the layout is: blocks of a few warps, each warp owning one row, packed 128-bit I/O, warp reductions for the moments, shared-memory staging of the residual row and the weight/bias. The precision discipline from the vectorization rung carries through: the loads/stores are BF16 (`x128`), but the moments and the normalize arithmetic are FP32 — summing 768 BF16 values in BF16 would corrupt the statistics.

Quantitatively, run separately the residual-add reads $2(B{\cdot}T{\cdot}C)$ and writes $(B{\cdot}T{\cdot}C)$, and the LayerNorm reads $(B{\cdot}T{\cdot}C)$ and writes $(B{\cdot}T{\cdot}C)$. Fused, I read $2(B{\cdot}T{\cdot}C)$ once and write $(B{\cdot}T{\cdot}C)$ for the residual plus $(B{\cdot}T{\cdot}C)$ for the normalized output — so I have removed one full $(B{\cdot}T{\cdot}C)$ write *and* one full $(B{\cdot}T{\cdot}C)$ read (the round-trip of the residual sum) relative to running the kernels separately. For a bandwidth-bound op that is a direct, sizable cut, and there are two such seams per layer (attention's residual+LN and the MLP's residual+LN), twelve layers, every step — it compounds.

This is the general principle the rest of the kernel work leans on: any time a memory-bound kernel's output is the immediate input of the next memory-bound kernel, the boundary between them is a forced HBM round-trip, and fusing them deletes it. Residual+LayerNorm is the cleanest and most frequent instance, so it is the one to do first; the same logic will later fold bias and GELU into the matmul that produces their input, and fold the classifier's softmax/cross-entropy/gradient into one pass. Correctness is unchanged — the fused kernel computes exactly $\text{LayerNorm}(\text{inp1} + \text{inp2})$ and emits the same residual sum, to FP32-internal tolerance, so it matches the reference and 3.29 holds.

What still hurts: attention is *still* the materialized-$T\times T$ textbook path, now the single largest structural memory cost in the model, with two big batched matmuls ($QK^\top$ and $\cdot V$) bridged by a softmax that writes and reads the entire $O(T^2)$ score matrix through HBM. That is next — and it is not a simple two-kernel fusion, because the softmax in the middle has its own reduction over $T$; I will need the streaming-softmax idea to fuse it. The fused kernel: warp-per-row, weight/bias and the residual row staged in shared memory, FP32 reductions, 128-bit I/O:

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
