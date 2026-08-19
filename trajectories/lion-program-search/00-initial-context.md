## Research question

Essentially every state-of-the-art deep model — large language models, vision transformers,
multimodal contrastive models, diffusion models — is trained with one of a tiny handful of
hand-designed first-order optimizers, overwhelmingly AdamW (Adam with decoupled weight decay) or
Adafactor. These were invented by human intuition and have stuck for years. Two questions follow.
First, are the human-designed update rules actually optimal, or is there a better one nobody has
stumbled onto by hand? Second — since the space of possible update rules is enormous and human
search is slow — can the optimizer itself be *discovered automatically* in a way that generalizes
from cheap small-scale search experiments up to the real, expensive, state-of-the-art training
regime, without exceeding the memory an optimizer like Adam already uses?

## Background

**Adam** (Kingma & Ba 2014) maintains an exponential moving average of the gradient
mₜ = β₁mₜ₋₁ + (1−β₁)gₜ and of the squared gradient vₜ = β₂vₜ₋₁ + (1−β₂)gₜ², and steps with
mₜ/(√vₜ + ε) — a per-coordinate learning rate set by the gradient's second moment, with bias
correction on both moments. **AdamW** (Loshchilov & Hutter 2019) decouples weight decay from the
gradient-based update: instead of folding the L2 penalty into g (which then gets divided by √v
and distorted), it shrinks the weights directly, θ ← θ − η(update + λθ). Both keep *two* extra
buffers (m and v), doubling the optimizer's memory relative to the parameters.

**signSGD** (Bernstein et al. 2018), foreshadowed by Rprop (Riedmiller & Braun 1993), steps with
sign(g): every coordinate moves by the same magnitude, only the direction comes from the
gradient. This is communication-efficient (one bit per coordinate), provably robust to
heterogeneous gradient scales, and tends to work well at large batch sizes where the sign of the
averaged gradient is reliable. Its momentum variant signs an EMA of the gradient — one constant,
one role. **NAdam** (Dozat 2016) folds the freshly-updated first moment together with the current
gradient when computing the step (Nesterov-style look-ahead inside Adam), but keeps Adam's
coupled second moment and does not decouple a tracking rate from an application rate.

On the discovery side: **Learning to optimize** (Andrychowicz et al. 2016; Metz et al. 2019,
2022) parameterizes the update rule as a small neural network trained to output updates on a
handful of small tasks — a black box that gives no handle on *why* it does what it does and no
obvious transfer to a training regime orders of magnitude bigger than what it was trained on.
**Symbolic / program search for optimizers** (Bello et al. 2017, "Neural Optimizer Search") is
the more promising thread: it uses RL or Monte Carlo sampling over expression trees built from a
*fixed* set of operands (the gradient g and a bias-corrected EMA of it, m) and a *fixed, bounded*
set of unary/binary math operators. This keeps the search tractable, but the operand set is fixed
— the search can shape how g and m are *combined*, but it cannot change how m itself is tracked
or how many buffers exist. The two named optimizers this search surfaced are **PowerSign**
(update = αf(t)·sign(g)·sign(m)·g, default α=e, f(t)=1) and **AddSign**
(update = (α + f(t)·sign(g)·sign(m))·g, default α=1, f(t)=1). **AutoML-Zero** (Real et al. 2020)
is the ambitious extreme in the other direction: search every component of an ML algorithm as a
linear program of primitive operations, via regularized evolution, evaluated on toy tasks — full
programs, but only ever validated on toy scale.

The relevant phenomena to keep in mind for any search-based approach: the space of programs is
large and sparse, so undirected sampling is unlikely to find anything good; and there is a gap
between proxy tasks (minutes on one chip) and target tasks (>10⁴× more compute), a
*meta-overfitting* phenomenon where search fitness on the proxy may not transfer to larger tasks.

## Pre-existing, measured facts (the bar to clear)

AdamW, thoroughly tuned (peak learning rate and decoupled weight decay swept on a log scale, same
protocol every entrant below gets), training ViT-S/16 and ViT-B/16 from scratch on ImageNet with
RandAugment + Mixup, 300 epochs, batch 4,096, evaluated on three held-out sets (top-1 accuracy,
averaged over three runs):

| Model | ImageNet | ReaL | V2 |
|---|---|---|---|
| ViT-S/16 | 78.89 | 84.61 | 66.73 |
| ViT-B/16 | 80.12 | 85.46 | 68.14 |

This is the number every candidate rule has to beat, on every column, at both scales, to count as
a real improvement rather than noise.

## Code framework

The substrate is a program-search harness over optimizer programs, plus the ordinary training
loop the optimizer plugs into. An optimizer is represented as a `train` function with the *same
input/output signature as AdamW* — inputs are the weight `w`, gradient `g`, learning-rate value
`lr`, and a bounded number of extra state variables; output is the `update` and the new state —
so any discovered program has memory no larger than Adam.

```python
import numpy as np

def interp(x, y, a):
    # linear interpolation primitive available to the search
    return (1.0 - a) * x + a * y

# The optimizer program: this `train` body is the open slot the search fills in.
# Signature fixed to match AdamW (two extra state vars v1, v2, both init 0),
# so the discovered optimizer's memory footprint <= Adam's.
def train(w, g, v1, v2, lr):
    # TODO: a sequence of assignment statements over {w, g, v1, v2, lr} and
    #       primitive math functions (interp, sign, sqrt, abs, clip, ...).
    #       Must return (update, v1, v2).
    update = ...  # TODO
    return update, v1, v2

# Outer training loop (already known):
# for w, g, lr in training_stream:
#     update, v1, v2 = train(w, g, v1, v2, lr)
#     w = w - update

# Search harness scaffold (already-known evolutionary machinery, available if I choose to run it):
def mutate(program):
    # insert / delete / modify one statement; new constants ~ N(0, 1)
    pass

def abstract_execute(program):
    # infer types/shapes (reject invalid), hash semantics (dedup cache),
    # mark redundant statements
    pass

def regularized_evolution(proxy_tasks, meta_validation_tasks):
    # tournament selection, warm-start population, restarts;
    # fitness = proxy performance; select by meta-validation to fight meta-overfitting;
    # then simplify the winner (remove redundant / low-impact statements,
    # rewrite to mathematically-equivalent simpler form)
    pass
```

The proxy task, if I do run this machinery: a 3-layer, 96-hidden-unit, 3-head ViT on 10% of
ImageNet for 30k steps, batch 64, 64×64 images, patch 16 — completes in under 20 minutes on one
chip, versus days on hundreds of chips for the target scale above.

## Evaluation settings

Every candidate rule below is plugged into the same training recipe — ViT-S/16 and ViT-B/16
trained from scratch on ImageNet, RandAugment + Mixup, 300 epochs, batch 4,096 — with its own
peak learning rate and weight decay tuned on a log scale, and reported as top-1 accuracy on
ImageNet, ImageNet-ReaL, and ImageNet-V2, averaged over three runs. Higher is better on all three
columns, at both model scales.
