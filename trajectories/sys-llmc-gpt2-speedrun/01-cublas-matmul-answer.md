**Problem (from step 0).** The naive port's matmul launches one thread per output element, each looping over
the contraction dimension C and reading input/weight rows straight from global memory. Every input row is
re-read OC times, every weight row B·T times: a matmul (the one operation that should be compute-bound and
tensor-core-fed) is turned into a bandwidth-bound FP32 grind that leaves the tensor cores idle. The matmuls are
where essentially all the model's FLOPs live, so this is the first thing to fix.

**Key idea.** Don't hand-roll the GEMM — route every projection (QKV, attention-output, the two MLP matmuls,
the logits projection) through the vendor tensor-core GEMM, **cuBLASLt**. One library call dispatches a
heuristically-selected, expertly-tiled, shared-memory-staged, double-buffered kernel that runs near the chip's
peak and feeds the multiplies through the tensor cores (TF32 path even while storage is still FP32). Use
cuBLASLt rather than plain cuBLAS so the matmul wrapper can later fold bias and GELU into the GEMM epilogue.

**Why it works.** A matmul reuses each loaded value O(tile-width) times out of shared memory and runs on the
tensor cores; the library does exactly that tiling and uses the ~312-TFLOPS lane that the naive FP32 kernel
never touches. The only correctness work is the row-major↔column-major layout: set the transpose flags
(`CUBLASLT_MATMUL_DESC_TRANSA`) and leading dimensions so `out = inp · weightᵀ`, then check against the CPU
reference. The non-matmul layers, attention, the giant classifier, and single-GPU/FP32 storage are untouched —
the next bottlenecks.

**Change / code.** A single `matmul` wrapper around `cublasLtMatmul`, with transpose flags, an optional fused
bias, and an algorithm chosen by the cuBLASLt heuristic (`llmc/matmul.cuh`):

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
