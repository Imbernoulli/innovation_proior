# Context: MoE routing without a balancing loss (the collapse control)

## Research question

A Mixture-of-Experts layer routes each token, via a small softmax router, to its top-`K` of `N`
expert FFNs. The router is trained only by the language-model cross-entropy, with no additional term
that rewards spreading tokens across experts. The question is: what does the token allocation look
like, and what is the model quality, when no load-balancing loss is added?

## What is measured

- `L_CE`: held-out language-model cross-entropy (perplexity `= exp(L_CE)`).
- `L_imb = ½ Σ_i |f_i − 1/N|`: L1 deviation of the token allocation `f_i` (fraction of routed
  (token, slot) assignments per expert) from uniform. `0` is perfectly balanced; it grows toward
  `1 − 1/N` as the router concentrates on fewer experts.
- Fitness `r = −(L_CE + L_imb)` (ShinkaEvolve objective, arXiv:2509.19349).

## The substrate

A tiny Transformer-style MoE: `N=8` experts, top-`K=2`, two MoE layers, `d=64`, trained on a
synthetic latent-topic next-token task (each sequence drawn from one of eight latent topics, so
there is genuine structure for experts to specialize on). The editable surface is one function: the
scalar load-balancing loss added to `L_CE`. The control returns zero.

## Prior art

Switch Transformer (arXiv:2101.03961) and GShard (arXiv:2006.16668) add an auxiliary loss to the
cross-entropy objective to encourage spreading of token assignments across experts. The present
setting uses no such auxiliary term; the router is trained by cross-entropy alone.
