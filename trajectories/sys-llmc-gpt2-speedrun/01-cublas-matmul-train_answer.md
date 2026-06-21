The starting point is a correct but painfully slow CUDA reproduction of GPT-2: every CPU loop transcribed into a kernel, agreeing with the CPU/PyTorch reference to floating-point tolerance and training at well under a thousand tokens/sec. The single worst offender is the matmul, and that is exactly where I have to look first, because the matmuls *are* the model — the QKV projection, the attention-output projection, the two MLP matmuls, and the giant logits projection are where essentially all the FLOPs live.

My naive matmul launches one thread per output element of the $(B{\cdot}T, OC)$ result. Each thread walks the contraction dimension $C$, and for each step reads one element of the input row and one element of the weight row from global memory, multiplies, and accumulates into a register: `for (int i = 0; i < C; i++) val += inp_bt[i] * wrow[i];`. The memory traffic is catastrophic. For a single output tile, thread $t_0$ reads input row $bt$ and weight row $oc$; the neighboring thread $t_1$ computes a *different* output $(bt, oc{+}1)$ and re-reads the *same* input row $bt$ entirely, plus weight row $oc{+}1$. Every input row is re-read $OC$ times from HBM; every weight row is re-read $B{\cdot}T$ times. The arithmetic intensity — flops per byte moved — is on the floor. A matmul is the one operation that *should* be compute-bound, with each loaded value reused $O(\text{tile-width})$ times out of fast on-chip memory; I have instead turned it into a bandwidth-bound disaster that touches global memory on every multiply. Worse, the accumulation is FP32 on the CUDA cores, so the tensor cores — the part of an A100 that does ~312 TFLOPS — sit completely idle while I grind out ~19.5 TFLOPS of FP32 that I cannot even reach because I am memory-starved.

A good matmul would tile the output, stage tiles of input and weight into shared memory, have every thread cooperatively load a tile once and reuse it many times, double-buffer to hide loads behind compute, and feed the multiplies through the warp-level matrix-multiply-accumulate (mma) instructions on the tensor cores. That is a multi-week project of K-loop handling, bank-conflict avoidance, fragment swizzles, and edge cases — and people write entire libraries (CUTLASS) on doing it well. The honest move is not to hand-roll it. NVIDIA ships a tensor-core GEMM that is the product of years of exactly that tuning: cuBLAS and its newer extension cuBLASLt.

So I propose to route every projection through **cuBLASLt tensor-core GEMMs**. A single `cublasLtMatmul` call dispatches a heuristically-selected, expertly-tiled, shared-memory-staged, double-buffered kernel that runs at a large fraction of the chip's peak and feeds the multiplies through the tensor cores. The one correctness obligation is the data layout: BLAS is column-major and my tensors are row-major. My activation is $(B,T,C)$ treated as $(B{\cdot}T, C)$ row-major, my weight is $(OC, C)$ row-major, and the matmul I want is $\text{out}[bt, oc] = \sum_c \text{inp}[bt, c]\cdot\text{weight}[oc, c]$, i.e. $\text{inp}\cdot\text{weight}^\top$. I state the transpose flags (`CUBLASLT_MATMUL_DESC_TRANSA`) and pick the column-major leading dimensions so the row-major storage lines up, then check the output against the CPU reference to floating-point tolerance.

Why cuBLASLt rather than plain `cublasGemmEx`? Plain cuBLAS does the multiply and nothing else; cuBLASLt exposes the same tensor-core GEMM through a richer descriptor — explicit operation and layout objects, a workspace, and an *epilogue* hook. Right now I only need "an even faster GEMM," but the epilogue is the optionality I am buying: a transformer matmul is almost always followed by a bias add, and the MLP up-projection by a GELU, and today those are separate memory-bound kernels each re-reading the matmul output from HBM. cuBLASLt can fold a bias add (and later a GELU) directly into the GEMM's epilogue, writing the result once. I will not wire up the fused GELU yet — the first job is to stop hand-rolling the GEMM — but choosing cuBLASLt now means the matmul infrastructure can absorb those fusions later instead of being a plain `cublasGemmEx` I would have to rip out.

There is one precision choice to pin down even while storage is still FP32. The GEMM descriptor takes a *compute type* separate from the data type. With FP32 inputs I can ask for the TF32 tensor-core path (`CUBLAS_COMPUTE_32F` with the TF32 math mode), which runs the multiplies through the tensor cores at reduced mantissa while accumulating in FP32 — a large speedup over true FP32 on the CUDA cores, at a precision cost that training tolerates. So even before I switch the *storage* to half precision, routing through cuBLAS already lights up the tensor cores that my hand-written kernel left dark. That is why this is a large step rather than a small one: I simultaneously get an expertly-tiled, double-buffered kernel for free *and* move the dominant FLOPs off the FP32 lane onto the tensor cores.

What I am deliberately not fixing yet, so the next pain is clear: storage is still FP32, so I am at the TF32 rate, not the BF16 rate, and I move 4 bytes per element everywhere — twice the HBM traffic of half precision. The non-matmul layers (LayerNorm, GELU, residual, softmax) are still my own memory-bound kernels and now stick out more, because the matmuls just got dramatically faster and Amdahl's law starts pointing at everything *around* them. Attention is still the materialized-$T\times T$ textbook path, and it is all on one GPU. But the single biggest lever is now pulled. The change is a single `matmul` wrapper around `cublasLtMatmul`, with transpose flags, an optional fused bias, and an algorithm chosen by the cuBLASLt heuristic:

```c
void matmul_cublaslt(floatX* d, const floatX* a, const floatX* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false)
{
    bool has_bias = (bias != NULL);
    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts (row-major storage expressed as column-major leading dims)
    cublasLtMatrixLayout_t ALayout, BLayout, CLayout, DLayout;
    if (transA) { cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k)); }
    else        { cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m)); }
    if (transB) { cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n)); }
    else        { cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k)); }
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, CUBLAS_LOWP, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));

    cublasLtEpilogue_t epilogue = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    if (has_bias) {
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    // let cuBLASLt pick the algorithm (cached internally), then call the matmul
    cublasLtMatmulPreference_t preference; cublasLtMatmulHeuristicResult_t heuristic; int returnedResults = 0;
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));
    // ... destroy descriptors/layouts ...
}
```
