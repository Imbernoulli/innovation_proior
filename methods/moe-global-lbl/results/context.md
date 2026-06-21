# Context: global-batch load-balancing loss (Demons in the Detail / Qwen3)

## Research question

The Switch Transformer and GShard introduce an auxiliary load-balancing loss
`L_LBL = α · N · Σ_i f_i P_i` that penalizes router imbalance during training.
The question is how to apply this penalty in a way that steers expert utilization
toward balance across the full training corpus.

## What is measured

`L_CE` (perplexity), `L_imb = ½ Σ_i |f_i − 1/N|`, fitness `r = −(L_CE + L_imb)`.

## The substrate

The tiny MoE (`N=8`, top-`K=2`, two MoE layers, `d=64`, latent-topic next-token task). In this
single-process reproduction the training batch *is* the global batch, so the global-vs-micro
distinction is reproduced by computing the Switch rung's `f_i` on 4 micro-splits of the batch and
this rung's `f_i` on the full batch.

## Prior art

Qiu et al. 2025, "Demons in the Detail: On Implementing Load Balancing Loss for Training
Specialized MoE Models" (arXiv:2501.11873) and the Qwen global-load-balance blog examine how
the granularity at which `f_i` is measured affects router behavior. DeepSeek auxiliary-loss-free
balancing (Wang et al. 2024, arXiv:2408.15664) introduces a per-expert selection bias
`b_i` added only to the top-K selection scores (not the gate weights), updated per step as
`b_i ← b_i + u · sign(c̄ − c_i)`, `u ≈ 1e-3`, with no auxiliary gradient — a count-level
controller that runs alongside the differentiable penalty.
