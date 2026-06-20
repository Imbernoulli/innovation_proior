Measured result — AWQ (Lin et al. 2023), activation-aware per-channel weight scaling + group quant.
Metric: WikiText perplexity (**lower is better**); weight-only, group size 128 (g128) throughout, so
these are matched against RTN-g128 (not the per-channel numbers of earlier rungs).

| model | setting | RTN-g128 | AWQ | FP16 | source |
|---|---|---|---|---|---|
| Llama-2-7B | **INT3-g128** weight-only | 6.66 | **6.24** | ~5.47 | AWQ Table 4 (arXiv:2306.00978) |
| Llama-2-7B | **INT4-g128** weight-only | ~5.73 | **5.60** | ~5.47 | AWQ Table 4 |

At the matched g128 grouping, AWQ improves Llama-2-7B INT3-g128 from RTN-g128's **6.66** to **6.24**,
and at the easier INT4-g128 reaches **5.60** — within ~0.13 perplexity of FP16 (~5.47). It does this
with no second-order machinery: one searched per-channel equivalence scale per layer, chosen by a small
α-grid against output MSE, then folded into the surrounding ops for zero runtime overhead — so it is
lighter than GPTQ and does not risk regressing onto the calibration distribution. With weight-only 4-bit
now nearly lossless, the remaining quantization difficulty is entirely on the side these methods left in
FP16: the **activations**, whose per-channel outliers the next rung must confront to make integer-GEMM
W8A8 work at all.
