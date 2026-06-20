Measured result — GPTQ (Frantar et al. 2022), second-order layerwise reconstruction. Metric: WikiText
perplexity (**lower is better**); weight-only, bit-width and grouping stated per row.

| model | setting | RTN | GPTQ | FP16 | source |
|---|---|---|---|---|---|
| LLaMA-7B | 3-bit weight-only, **per-channel** | 25.54 | **8.07** | 5.68 | GPTQ README / paper (arXiv:2210.17323) |

GPTQ takes 3-bit per-channel LLaMA-7B from RTN's catastrophic **25.54** down to **8.07** against an FP16
reference of **5.68** — a >3× reduction in the perplexity gap, achieved post-training in a few GPU-hours
on a small calibration set, with no retraining. The OBS output-reconstruction (round a column, then
compensate the downstream weights via the Hessian H = 2XXᵀ) recovers most of the loss that independent
RTN rounding threw away, and the three scaling fixes (shared column order, lazy batch updates, Cholesky
reformulation) make it run on models up to 175B parameters. Remaining gap to FP16: ~2.4 perplexity at
3-bit per-channel — the room the next rung attacks by protecting the *salient* weight channels rather
than treating all weights symmetrically.
