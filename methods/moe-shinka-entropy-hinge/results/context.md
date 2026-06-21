# Context: MoE load-balancing with global-batch expert-usage loss

## Research question

Global-batch load-balancing loss (Qiu et al. 2025) balances MoE expert usage while preserving
specialization, by measuring the token frequency `f_i` over the whole corpus rather than each
micro-batch. The question: *can the load-balancing objective be improved to better handle expert
usage distribution across the full range of utilization levels?*

## Background

Mixture-of-Experts (MoE) models route each token to a subset of `K` out of `N_E` experts. The
router assigns per-expert probabilities `P_i` and selects the top-`K` experts. Training requires a
balancing loss to prevent expert collapse (where a few experts dominate and others are never used).

**Switch Transformer loss (Fedus et al. 2021).** Per-micro-batch penalty:

```
L_switch = N_E · Σ_i f_i · P_i
```

where `f_i` is the fraction of tokens routed to expert `i` in the current micro-batch and `P_i` is
the mean router probability for expert `i`. Both `f_i` and `P_i` are measured over the micro-batch.

**Global-batch LBL (Qiu et al. 2025).** Computes `f_i` over the entire corpus (global batch) rather
than the current micro-batch:

```
L_global = N_E · (1/L) Σ_ℓ Σ_i f_{ℓ,i} · P_{ℓ,i}
```

where `f_{ℓ,i}` is the global-batch token frequency for expert `i` in layer `ℓ`, and `P_{ℓ,i}` is
the mean router probability for that expert in that layer. The global-batch frequency is updated as
an exponential moving average over training. This approach reduces interference with learned
specialization compared to micro-batch measurements.

## What is measured

`L_CE` (cross-entropy / perplexity), `L_imb = ½ Σ_i |f_i − 1/N|`, fitness `r = −(L_CE + L_imb)`.

## The substrate and scale

Evaluation uses a small MoE (`N=8` experts, top-`K=2`, two MoE layers, latent-topic next-token
task) that reproduces mechanism and ordering. The global-batch LBL reference is from Qiu et al.
2025, validated on large-scale MoE pretraining with `N_E=64` experts, ~2.10B FineWeb tokens.
