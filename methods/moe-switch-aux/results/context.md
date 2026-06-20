# Context: the Switch/GShard auxiliary load-balancing loss

## Research question

An MoE router trained only on language-model cross-entropy collapses — a few of the `N` experts
soak up the traffic and the rest die. The control with no balancing loss exhibits exactly this:
large load imbalance, dead experts. The question here is the classic fix: *can a single
differentiable penalty, computed from the router's own statistics, push the router back toward
using all experts without dropping tokens?*

## The method in one line

Add to the cross-entropy the GShard/Switch auxiliary loss

```
L_aux = α · N · Σ_i f_i · P_i,    α ≈ 1e-2,
```

where `f_i` is the fraction of (token, slot) assignments dispatched to expert `i` (a hard count,
non-differentiable) and `P_i` is the mean router softmax mass on expert `i` (differentiable). The
`f_i` acts as a per-expert weight; minimizing `Σ f_i P_i` pulls probability mass off the over-used
experts. The factor `N` makes the uniform optimum scale-free (`Σ f_i P_i = 1/N` when balanced), and
`α` sets the trade-off against CE. The hard counts `f_i` are computed on the **micro-batch**.

## What is measured

`L_CE` (perplexity), `L_imb = ½ Σ_i |f_i − 1/N|`, and fitness `r = −(L_CE + L_imb)`.

## The substrate

The tiny MoE of the control rung (`N=8`, top-`K=2`, two MoE layers, `d=64`, latent-topic
next-token task). Only the balancing-loss function changes.

## Prior art and known limitation

Lepikhin et al. 2020 (GShard, arXiv:2006.16668) and Fedus et al. 2021 (Switch Transformer,
arXiv:2101.03961) introduced this loss. Its known weakness, which motivates the global-batch
variant: `f_i` is a **micro-batch** statistic — a small, noisy sample — so forcing every
micro-batch toward uniform penalizes the legitimate topic skew of that slice, buying balance partly
by suppressing specialization.
