# Context: MoE routing without a balancing loss (the collapse control)

## Research question

A Mixture-of-Experts layer routes each token, via a small softmax router, to its top-`K` of `N`
expert FFNs. The router is trained only by the language-model cross-entropy. That objective has no
term that rewards spreading tokens across experts, so nothing stops the router from sending almost
all traffic to a few experts: those experts get more gradient and more effective capacity, the
cross-entropy drops, and the remaining experts wither. This is **router collapse**. The question
this control answers is the baseline one — *what actually happens to the token allocation, and to
the model quality, when no balancing loss is added at all?*

## What is measured

- `L_CE`: held-out language-model cross-entropy (perplexity `= exp(L_CE)`).
- `L_imb = ½ Σ_i |f_i − 1/N|`: L1 deviation of the token allocation `f_i` (fraction of routed
  (token, slot) assignments per expert) from uniform. `0` is perfectly balanced; it grows toward
  `1 − 1/N` as the router collapses.
- Fitness `r = −(L_CE + L_imb)` (ShinkaEvolve objective, arXiv:2509.19349).

## The substrate

A tiny Transformer-style MoE: `N=8` experts, top-`K=2`, two MoE layers, `d=64`, trained on a
synthetic latent-topic next-token task (each sequence drawn from one of eight latent topics, so
there is genuine structure for experts to specialize on). The editable surface is one function: the
scalar load-balancing loss added to `L_CE`. The control returns zero.

## Prior art

This rung predates all balancing losses: it is the naive MoE, the thing Switch/GShard
(arXiv:2101.03961, arXiv:2006.16668) added an auxiliary loss to *fix*. It is included as the
honest floor — the imbalance every later rung must cut, and the fitness `r` every later rung must
beat.
