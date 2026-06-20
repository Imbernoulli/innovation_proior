# Context: global-batch load-balancing loss (Demons in the Detail / Qwen3)

## Research question

The Switch/GShard auxiliary loss `α·N·Σ_i f_i P_i` cuts MoE load imbalance, but it computes the
hard count `f_i` on the **micro-batch** — the small slice of tokens one device sees in one forward.
Forcing every micro-batch toward a uniform expert distribution over-constrains the router: a
code-heavy or topic-heavy slice *should* lean on a few experts, and penalizing that skew buys
balance by flattening specialization. The question: *can the same penalty be made to respect
specialization by changing only the scope over which `f_i` is measured?*

## The method in one line

Keep the penalty form, but compute the frequency `f_i` over the **global batch** (all micro-batches
synchronized) instead of the local micro-batch:

```
L_LBL = α · N · Σ_i f_i^{global} · P_i,    α ≈ 1e-2.
```

Balance becomes a property the whole corpus must satisfy, not every tiny slice — so any single
micro-batch is free to be as specialized as its content demands, as long as usage evens out
globally. Optionally pair it with DeepSeek's auxiliary-loss-free bias (`b_i ← b_i + u·sign(c̄ − c_i)`,
`u≈1e-3`, used only for top-K selection, no aux gradient) as a complementary count-level controller.

## What is measured

`L_CE` (perplexity), `L_imb = ½ Σ_i |f_i − 1/N|`, fitness `r = −(L_CE + L_imb)`.

## The substrate

The tiny MoE (`N=8`, top-`K=2`, two MoE layers, `d=64`, latent-topic next-token task). In this
single-process reproduction the training batch *is* the global batch, so the global-vs-micro
distinction is reproduced by computing the Switch rung's `f_i` on 4 micro-splits of the batch and
this rung's `f_i` on the full batch.

## Prior art and known limitation

Qiu et al. 2025, "Demons in the Detail: On Implementing Load Balancing Loss for Training
Specialized MoE Models" (arXiv:2501.11873) and the Qwen global-load-balance blog; DeepSeek
auxiliary-loss-free balancing (Wang et al. 2024, arXiv:2408.15664). Known limitation, which
motivates the ShinkaEvolve endpoint: global-batch LBL equalizes *average* usage but treats a
nearly-dead expert the same as a slightly-cold one — the rescue of badly under-used experts is only
as strong as the smooth `f·P` gradient there, which is weak exactly where it is needed most.
