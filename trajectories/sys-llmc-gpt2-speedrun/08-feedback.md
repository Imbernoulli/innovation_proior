What this record changed: fused the final softmax, cross-entropy loss, and logit gradient into a single
`fused_classifier` kernel that makes one pass over the (B,T,V) logits, never materializes the full normalized
probabilities (only the target entry is needed for the loss), and writes the gradient `(prob − onehot)·dloss`
in place over the logits. Enabled by `dloss` being a known constant (`1/(B·T)`) rather than a backpropagated
quantity.

Measured / documented numbers (the repo's own statements):

| record | figure | source |
|---|---|---|
| what the fusion avoids materializing | "Never materializes the full normalized logits, only at the target label" | `llmc/fused_classifier.cuh` (header comment) |
| why backward can be fused into forward | `dloss` is "just a constant 1/batch_size tensor … known in advance" | `dev/cuda/classifier_fused.cu` (header comment) |
| in-place race fix | `__syncthreads()` prevents `exp(90+)` infinities when logits are overwritten by gradients | `llmc/fused_classifier.cuh` (kernel comment) |
| cache discipline | reverse block order for L2 hits on matmul data; streaming store on overwritten logits | `llmc/fused_classifier.cuh` (`fused_classifier_kernel5`) |
| per-kernel timing | `block_size … | time … ms` per launch config, versions 1–4 | `dev/cuda/classifier_fused.cu` harness |

The fused classifier is the production loss path in the mainline trainer; `dev/cuda/classifier_fused.cu`
develops and benchmarks four versions of it (printing `time … ms` per block-size and checking against the CPU
reference). The repo documents the *mechanism* (never materializing the (B,T,V) probs, in-place gradient) rather
than an isolated end-to-end tokens/sec delta for this fusion alone; the per-rung evidence is the kernel itself
and the structural argument — the largest tensor in the pass (~50 GB BF16) collapses from several full HBM
passes to one in-place pass, also removing the probabilities buffer from the memory budget. The fused result is
the same stable softmax / cross-entropy / `(softmax−onehot)·dloss` gradient (FP32-internal), so the 3.29 target
is held.
