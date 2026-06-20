**Problem (from step 7).** The final logits tensor (B, T, V) with V≈50k is the single largest buffer in the
pass (~50 GB BF16 at 0.5M tokens). The textbook classifier makes it cross HBM several times: GEMM writes logits
→ softmax writes (B,T,V) probs → cross-entropy reads probs → backward writes a (B,T,V) gradient. Almost all of
that traffic is waste — the loss needs only two scalars per row, and the gradient can be made from the logits
plus those scalars.

**Key idea.** Fuse **softmax + cross-entropy + logit gradient** into one kernel that makes a single pass over
the logits. Per row: an online-softmax pass for the stable normalizer (max, sum); read the target logit, emit
the scalar loss `−log(prob[ix])`; then compute `dlogits[v] = (prob[v] − [v==ix])·dloss` and write it **in place
over the logits**. The (B,T,V) probability tensor is never materialized; no separate gradient buffer is
allocated. This is well-posed because **`dloss` is a known constant** (`1/(B·T)` for mean cross-entropy), not a
backpropagated quantity, so the backward can be computed in the forward pass.

**Why it works.** It collapses ~4 full passes over the largest tensor into one in-place pass — the biggest single
memory saving in the model, and it removes the probabilities buffer from the memory budget entirely. Reverse
block order keeps rows hot in L2 from the preceding logits GEMM; a streaming store on the overwritten logits
preserves cache for still-needed ones. A `__syncthreads()` between the loss read and the in-place gradient write
avoids a race (overwriting a logit before its loss is read could yield `exp(90+)` = inf). FP32-internal math
holds 3.29. Remaining lever is no longer single-GPU kernels: 10B tokens ≈ 18,865 steps is a long wall-clock on
one device — next, more GPUs.

**Change / code.** The fused softmax/cross-entropy/gradient kernel, single pass, in-place gradient
(`llmc/fused_classifier.cuh`):

```c
// Fused Classifier: forwards the cross-entropy loss; never materializes the full normalized logits, only at
// the target label; also kicks off the backward pass, because everything is already loaded.
template <bool WriteDLogits, bool WriteProbs>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
    fused_classifier_kernel5(floatX* logits, float* losses, floatX* probs,
                             const float dloss, const int* targets,
                             int B, int T, int V, int P, std::bool_constant<WriteDLogits>) {
    int64_t idx = gridDim.x - (blockIdx.x+1); // reverse order for cache hits on matmul data
    int ix = targets[idx];

    // online softmax over the V logits of this row (max + sum), nothing materialized
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

    // the loss: -log(prob at the target), from the normalizer only (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] -= logf(prob);
    }
    // barrier: finish the loss read before overwriting logits with gradients (else exp(90+) -> inf)
    __syncthreads();

    // gradient in place: dlogits[v] = (prob[v] - [v==ix]) * dloss, computed from logits + (max,sum)
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
        x128 packed_logits_vec = load128(logits_vec + i * x128::size); // 2nd read; kept in cache from softmax
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob = expf((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
            packed_probs[k] = (floatX)prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        if (WriteDLogits) {
            // reduce cache persistence for the overwritten logits (won't be read again)
            store128cs(logits + idx * P + i * x128::size, packed_logits_vec);
        }
        if (WriteProbs) { store128(probs + idx * P + i * x128::size, packed_probs); }
    }
    // ... handle the V % x128::size tail elements identically ...
}
```
