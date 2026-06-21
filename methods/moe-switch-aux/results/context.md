# Context: the Switch/GShard auxiliary load-balancing loss

## Research question

A Mixture-of-Experts layer routes each token, via a small softmax router, to its top-`K` of `N`
expert FFNs. When the router is trained only on language-model cross-entropy, token assignments
concentrate on a small subset of experts. The question is: can an auxiliary penalty term added to
the cross-entropy objective encourage a more even spread of token assignments across experts?

## What is measured

`L_CE` (perplexity), `L_imb = ½ Σ_i |f_i − 1/N|`, and fitness `r = −(L_CE + L_imb)`.

## The substrate

The tiny MoE of the control rung (`N=8`, top-`K=2`, two MoE layers, `d=64`, latent-topic
next-token task). Only the balancing-loss function changes.

## Prior art

Lepikhin et al. 2020 (GShard, arXiv:2006.16668) and Fedus et al. 2021 (Switch Transformer,
arXiv:2101.03961) introduced auxiliary losses to encourage load balancing in MoE models.
Switch Transformer adds a differentiable term to the cross-entropy that uses both the fraction
of tokens dispatched to each expert and the mean router softmax probability mass on each expert.
