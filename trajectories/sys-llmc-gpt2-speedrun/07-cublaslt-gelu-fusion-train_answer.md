Attention no longer materializes its score matrix, so the remaining HBM waste in a transformer block is back on the matmul side, at the MLP up-projection seam. That matmul takes the $(B{\cdot}T, C)$ hidden state to $(B{\cdot}T, 4C)$; the cuBLASLt GEMM computes it and writes the $(B{\cdot}T, 4C)$ result to HBM, then a separate GELU kernel reads that tensor back, applies the nonlinearity, and writes it out again for the down-projection to read. That intermediate — the pre-GELU, `fch` — is $4\times$ the channel width, $(B{\cdot}T{\cdot}4C)$ elements, the *largest* activation tensor inside the block. And it makes a full HBM round-trip purely to have a pointwise function applied: the GEMM writes it, the GELU reads it, the GELU writes it.

This is the same disease as the residual+LayerNorm seam — two adjacent ops passing a tensor through HBM when one already had it on-chip — but with a twist that makes it more wasteful: the GELU's input is *produced by the GEMM*. At the moment the GEMM finishes computing a tile of its $(B{\cdot}T, 4C)$ output, that tile is sitting in registers/shared memory in the GEMM's epilogue, about to be written out. If the GELU could be applied *right there*, in the epilogue, before the write, the pre-GELU tensor would be transformed on-chip and only the post-GELU result would need to touch HBM. The separate-kernel arrangement throws this away: the GEMM writes the untransformed output, then a second kernel pays a full read to apply a function the first kernel could have applied for free.

I set this up deliberately two rungs back. When I chose cuBLASLt over plain cuBLAS, the reason was its *epilogue* hook — the ability to fold a post-GEMM operation into the matmul kernel itself. So I propose the **cuBLASLt fused GELU/bias matmul epilogue**. cuBLASLt's epilogue supports exactly bias-add and GELU (and bias+GELU together): `CUBLASLT_EPILOGUE_GELU_AUX` applies GELU in the epilogue, `CUBLASLT_EPILOGUE_GELU_AUX_BIAS` applies bias-then-GELU. The MLP up-projection is precisely a matmul followed by a bias and then a GELU, which is the case `GELU_AUX_BIAS` was built for, so I wire the up-projection's `matmul` call to request the GELU epilogue instead of running a standalone GELU kernel afterward.

The "AUX" in the epilogue name is load-bearing for the backward pass. The GELU's *backward* needs the GELU's *input* — $\mathrm{d}\,\text{GELU}/\mathrm{d}x$ is a function of $x$, the pre-GELU value — so to backprop through the GELU I need the pre-activation, not just the post-activation output. If the GEMM applies GELU in its epilogue and only writes the post-GELU result, I have lost the pre-GELU value I need for backward. cuBLASLt handles this with an *auxiliary* output: the `GELU_AUX` epilogue writes the post-GELU result to the main output `d` *and* the pre-GELU values to a separate auxiliary buffer (pointed to by `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`, with its own leading dimension). So in the forward pass a single fused GEMM yields both the GELU output and the saved pre-GELU input, no separate GELU kernel, the pre-GELU available for backward.

The backward direction is fused symmetrically. Backprop through the down-projection must push the gradient through the GELU, which means computing $\mathrm{d}\,\text{GELU}$ and multiplying it into the incoming gradient — and cuBLASLt has a `CUBLASLT_EPILOGUE_DGELU` epilogue that folds the GELU's derivative directly into the backward matmul, reading the saved pre-GELU aux tensor. So the GELU is fused into the GEMM in *both* directions: forward via `GELU_AUX_BIAS` (output + saved pre-GELU), backward via `DGELU` (gradient through GELU, using the saved pre-GELU). No standalone GELU or GELU-backward kernel. The wrapper already had the hook: I pass a `pre_gelu` pointer to `matmul`, and if non-null the epilogue is set to the right variant, the aux pointer set to `pre_gelu`, the enum chosen by whether there is also a bias and whether it is a forward or backward matmul.

Why this is a real win and not a rounding error: the fused-out tensor is $(B{\cdot}T{\cdot}4C)$ — at $B{\cdot}T \approx 0.5$M tokens and $C=768$ that is ~1.5B elements, ~3 GB in BF16, the single biggest activation in the block. Eliminating a standalone kernel that reads 3 GB and writes 3 GB, twice per layer (forward GELU and backward dGELU), across twelve layers, every step, is a substantial chunk of the step's HBM traffic. And because the work happens in the GEMM's epilogue while the data is already resident, the GELU is essentially free — it rides on the matmul kernel that was going to run anyway.

There is a memory-vs-compute tension I am now aware of but leave for later: saving the pre-GELU aux tensor for backward costs storing that $(B{\cdot}T{\cdot}4C)$ tensor through to the backward pass — the largest saved activation. The fused epilogue makes the *forward* cheap but does not reduce how much I must *keep* until backward. If memory ever gets tight there is a different trade available — do not save the pre-GELU, and instead *recompute* the GELU forward during backward from the $(B{\cdot}T{\cdot}C)$ hidden state, trading a little recomputation for a large activation-memory saving, exactly the knob I will want when I push the batch size up. For now the fusion is purely a speed win.

Correctness: the GELU is the same tanh-approximation GELU (`GELU_SCALING_FACTOR = sqrtf(2/π)`) applied at the same point; whether it runs as a standalone kernel or in the GEMM epilogue, the numbers are identical to tolerance, so 3.29 holds. After this the matmul-side fusions are done — bias and GELU folded into the GEMMs both ways. The remaining glaring structural cost is the one I keep deferring: the classifier still materializes the entire $(B,T,V)$ logits tensor, with $V\approx 50$k the single largest buffer in the whole pass, plus its softmax and its gradient. That is the next target. The epilogue-selection logic in the matmul wrapper that fuses GELU/bias forward and the GELU gradient backward:

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
