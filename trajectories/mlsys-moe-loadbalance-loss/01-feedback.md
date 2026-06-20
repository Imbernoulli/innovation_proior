Measured result — `no-balance` (control, `balance_loss → 0`). Small-scale reproduction: tiny MoE
(`N=8` experts, top-`K=2`, 2 MoE layers, `d=64`), synthetic latent-topic next-token task, 1200
training steps, AdamW. `L_CE`, perplexity, and `L_imb` are the mean over 20 fresh held-out batches
of the trained model; `r = −(L_CE + L_imb)`. Seed 1234.

| Variant | `L_CE` | perplexity | `L_imb` | fitness `r` |
|---|---|---|---|---|
| no-balance (control) | 3.7280 | 41.594 | 0.1286 | −3.8566 |

Notes: with no balancing term the router is visibly skewed — `L_imb = 0.1286`, where `0` is uniform
and the ceiling at `N=8` is `1 − 1/8 = 0.875`. The cross-entropy is unremarkable (perplexity 41.6),
confirming the diagnosis: collapse barely hurts the LM loss, which is exactly why the LM loss does
nothing to prevent it. This is the honest floor — the imbalance every later rung must cut and the
fitness `r` every later rung must beat. (Reference scale: ShinkaEvolve's own eval used a
556M/82M-active MoE, `N=64` top-8, ~2.10B FineWeb tokens, arXiv:2509.19349; this is a deliberately
small reproduction of the mechanism.)
