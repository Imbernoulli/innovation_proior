With the matmuls fused (bias and GELU folded into the GEMM epilogues both directions), the biggest single tensor in the whole pass is not in the transformer blocks at all — it is at the very end, the logits. The final projection takes the $(B{\cdot}T, C)$ hidden state to $(B{\cdot}T, V)$ logits with $V = 50257$, and that $(B, T, V)$ tensor dwarfs everything else: at $B{\cdot}T \approx 0.5$M tokens it is ~25B elements, ~50 GB in BF16. The textbook classifier then makes this enormous tensor cross HBM several times — the GEMM writes the logits; a softmax kernel reads them, finds the row max, computes the exponentials and the row sum, and writes the $(B,T,V)$ probabilities; a cross-entropy kernel reads the probabilities at the target index for the loss; and the backward writes a $(B,T,V)$ gradient $\text{dlogits} = (\text{probs} - \text{onehot})\cdot\text{dloss}$. The largest tensor in the model is read and written several times over, and it is pure waste, because I do not actually *need* most of those numbers.

Stare at what the loss and its gradient actually require. The cross-entropy loss for a token whose target is index $ix$ is $-\log(\text{softmax}(\text{logits})[ix]) = -(\text{logits}[ix] - \text{logsumexp}(\text{logits}))$. To compute it I need exactly two scalars per row — the logit at the target index and the row's log-sum-exp (equivalently the max and the sum-of-exp for the stable form) — not the full vector of probabilities. And the gradient $\text{dlogits}[v] = (\text{softmax}(\text{logits})[v] - [v = ix])\cdot\text{dloss}$ is a full $(B,T,V)$ tensor, yes, but to *produce* it I only need, per element, that element's softmax probability $\exp(\text{logits}[v] - \max)/\text{sum}$ — and I already have $\max$ and $\text{sum}$ from the loss. So the gradient at element $v$ can be computed from $\text{logits}[v]$ and the two per-row scalars, in a single pass, without ever materializing a separate probability tensor.

I propose the **fused classifier**: fold the softmax, the cross-entropy, and the logit gradient into *one kernel* that makes a single pass over the logits. Per row: (1) compute the stable softmax normalizer — the max and the sum-of-exp — in one online pass over the $V$ logits (the same running-max-and-rescale algorithm from the attention softmax, now applied over the vocabulary dimension, which is large); (2) read the logit at the target index, compute the loss $-\log(\text{prob}[ix])$ from the normalizer, and write the one scalar loss; (3) make a second pass over the row computing $\text{dlogits}[v] = (\text{prob}[v] - [v = ix])\cdot\text{dloss}$ and writing it *back into the logits buffer in place*. The $(B,T,V)$ probability tensor is never materialized — only the per-row $(\max, \text{sum})$ scalars are kept on-chip — and the gradient overwrites the logits, so no separate gradient buffer is allocated either.

There is a precondition for fusing the *backward* into the *forward* that I have to check, because it looks too good: computing $\text{dlogits}$ requires $\text{dloss}$, the gradient of the loss scaled by the batch — normally a backward-pass quantity, not available during the forward. But what *is* $\text{dloss}$? For mean cross-entropy over the batch it is just the constant $1/(B{\cdot}T)$ — the loss is the mean of the per-token losses, so each token's loss has gradient $1/(B{\cdot}T)$. That is known *before* the forward even runs; it is not data-dependent. (For fine-tuning where prompt tokens are masked it is $1/(B{\cdot}T)$ or zero per token, also known in advance.) So I hand the kernel $\text{dloss}$ up front and let it compute the gradient in the same pass that computes the loss. The fusion is well-posed precisely because the loss gradient is a known constant, not something that must be backpropagated.

The cache discipline from the Packed128 rung pays off heavily here because the logits are read more than once within the kernel — the first pass reads the whole row for the max and sum, the second reads each logit again to compute its gradient. If the row falls out of cache between the passes I re-read from HBM, so two tricks: process the rows in *reverse* block order (`idx = gridDim.x - (blockIdx.x+1)`) so the rows still hot in L2 from the preceding logits-GEMM are hit first; and on the second pass write the gradient with a streaming/evict hint (`store128cs`), because once a logit has been overwritten by its gradient it will not be read again and should not hold cache the still-needed logits want. The kernel comment is explicit that the second read is "when we reduce cache persistence," maximizing the chance the logits stay hot between the softmax pass and the gradient pass.

The in-place overwrite creates one correctness hazard worth naming because it bites. The loss reads $\text{logits}[ix]$ to compute $-\log(\text{prob})$, and the gradient pass *overwrites* the logits with gradients in the $[-1, 1]$ range. If those race — if a thread overwrites a logit before the loss thread reads it — and the softmax offset (the max) was large, you can end up computing $\exp(90+)$ and getting infinities in the loss. The fix is a `__syncthreads()` between the loss read and the gradient write, so the loss is fully computed from the original logits before any are overwritten. With that barrier the in-place trick is safe. A template flag selects whether the kernel actually writes the gradient (training) or the probabilities (inference/debug).

Quantitatively this is the single largest memory saving in the model: the $(B,T,V)$ tensor at ~50 GB BF16 is the biggest thing in the pass, and collapsing softmax+loss+gradient from ~four full passes over it into one in-place pass removes the bulk of the tail's HBM traffic *and* the entire probabilities buffer from the memory budget — which, like the GELU recompute idea, will matter when I push the batch size up. Correctness: the fused kernel computes exactly the stable softmax, the same cross-entropy loss, and the same $(\text{softmax} - \text{onehot})\cdot\text{dloss}$ gradient, in FP32 internally, so it matches the reference to tolerance and 3.29 holds.

After this, every major structural memory cost in a single forward+backward has been addressed: matmuls are cuBLAS with fused bias/GELU epilogues, elementwise layers are vectorized and fused, attention is flash, the classifier is fused. The model runs near its kernel-level peak on *one* GPU. The remaining lever is no longer about a single GPU's kernels — a 10B-token run is ~18,865 optimizer steps, and on one device that is a long wall-clock no matter how good the kernels are. The next move has to be more GPUs. The fused softmax/cross-entropy/gradient kernel, single pass, in-place gradient:

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
