**Problem (from step 6).** The MLP up-projection GEMM writes its (B·T, 4C) output — the largest activation in
the block — to HBM, then a standalone GELU kernel reads it back, applies the nonlinearity, and writes it out
again. A full round-trip of the biggest tensor in the block, just to apply a pointwise function the GEMM already
had on-chip in its epilogue.

**Key idea.** Fold the GELU (and the bias) into the GEMM's **cuBLASLt epilogue**: `CUBLASLT_EPILOGUE_GELU_AUX_BIAS`
applies bias-then-GELU in the matmul kernel itself, writing the post-GELU result to the main output *and* the
pre-GELU values to an **auxiliary buffer** (needed because GELU-backward depends on the pre-activation). The
backward direction is fused symmetrically with `CUBLASLT_EPILOGUE_DGELU`, which folds the GELU derivative into
the backward GEMM using that saved aux tensor.

**Why it works.** The GELU rides on the matmul kernel that runs anyway, while the data is already in registers,
so it is essentially free; the (B·T·4C) pre-GELU tensor (~3 GB in BF16, the block's largest) no longer
round-trips through HBM just to be transformed, forward (`GELU_AUX_BIAS`) or backward (`DGELU`). This was set up
in step 1 by choosing cuBLASLt for its epilogue hook. Identical GELU math, so 3.29 holds. (Saving the pre-GELU
aux for backward keeps the largest activation resident until backprop — a memory cost the recompute knob will
later trade away.) Still open: the classifier materializes the entire (B,T,V) logits, V≈50k — the largest buffer
in the pass.

**Change / code.** The epilogue-selection logic in the matmul wrapper that fuses GELU/bias forward and the GELU
gradient backward (`llmc/matmul.cuh`):

```c
void matmul_cublaslt(floatX* d, const floatX* a, const floatX* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false)
{
    bool has_bias = (bias != NULL);
    bool has_gelu = (pre_gelu != NULL);
    // ... descriptor, layouts, transpose flags (as in step 1) ...

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue;
    if (has_gelu) {
        int64_t gelu_ld = m;
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        if (backward) {
            assert(!has_bias); // we shouldn't have any backward matmuls that use both GELU and bias
            epilogue = CUBLASLT_EPILOGUE_DGELU;                  // backward: GELU gradient fused in
        } else {
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS // forward: bias -> GELU, save pre-GELU
                                : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if (has_bias) {
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        cublasDataType_t bias_data_type = (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP;
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }
    // ... heuristic + cublasLtMatmul (as in step 1) ...
}
```
