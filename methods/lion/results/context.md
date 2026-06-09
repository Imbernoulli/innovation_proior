# Context

## Research question

Essentially every state-of-the-art deep model — large language models, vision transformers, multimodal contrastive models, diffusion models — is trained with one of a tiny handful of hand-designed first-order optimizers, overwhelmingly AdamW (Adam with decoupled weight decay) or Adafactor. These were invented by human intuition and have stuck for years. Two questions follow. First, are the human-designed update rules actually optimal, or is there a better one nobody has stumbled onto by hand? Second — since the space of possible update rules is enormous and human search is slow — can the optimizer itself be *discovered automatically* in a way that actually generalizes from cheap small-scale search experiments up to the real, expensive, state-of-the-art training regime? A solution has to confront a brutal generalization gap: a candidate rule can be evaluated quickly on a small proxy model, but the rules that look best on the proxy frequently fall apart at scale. And it should respect the practical constraint that an optimizer carrying more per-parameter state than Adam (which stores two moment buffers) is a hard sell.

## Background

The dominant optimizers all reshape the raw gradient using accumulated statistics. **Adam** (Kingma & Ba 2014) maintains an exponential moving average of the gradient mₜ = β₁mₜ₋₁ + (1−β₁)gₜ and of the squared gradient vₜ = β₂vₜ₋₁ + (1−β₂)gₜ², and steps with mₜ/(√vₜ + ε) — a per-coordinate learning rate set by the gradient's second moment, with bias correction on both moments. **AdamW** (Loshchilov & Hutter 2019) decouples weight decay from the gradient-based update: instead of folding the L2 penalty into g (which then gets divided by √v and distorted), it shrinks the weights directly, θ ← θ − η(update + λθ). This decoupling matters because under Adam the naive L2 term is scaled inconsistently across coordinates. Both keep *two* extra buffers (m and v), doubling the optimizer's memory relative to the parameters. Adafactor (Shazeer & Stern 2018) factorizes the second-moment matrix to save memory.

A different and older idea throws away gradient *magnitude* entirely. **signSGD** (Bernstein et al. 2018), foreshadowed by Rprop (Riedmiller & Braun 1993), steps with sign(g): every coordinate moves by the same magnitude, only the direction comes from the gradient. This is communication-efficient (one bit per coordinate), provably robust to heterogeneous gradient scales, and tends to work well at large batch sizes where the sign of the averaged gradient is reliable. Its momentum variant signs an EMA of the gradient.

On the discovery side, two threads exist. **Learning to optimize** (Andrychowicz et al. 2016; Metz et al. 2019, 2022) parameterizes the update rule as a small neural network trained to output updates. These black-box optimizers are trained on a few small tasks and struggle to generalize to large models trained for many steps — the learned network overfits the training regime. **Symbolic / program search for optimizers** (Bello et al. 2017, "Neural Optimizer Search") uses RL or Monte Carlo sampling over expression trees built from a fixed set of operands (gradient, momentum) and operators; to keep search tractable they fix the operands and bound the tree size, which crucially means they *cannot* change how momentum is tracked or how it feeds the update — and the optimizers they found (PowerSign, AddSign) did not reach state of the art. **AutoML-Zero** (Real et al. 2020) is the ambitious extreme: search every component of an ML algorithm as a linear program of primitive operations, via regularized evolution, evaluated on toy tasks.

The relevant phenomena to keep in mind: the search space of programs is infinite and extremely sparse (almost every program is useless — a random search over 2M programs on a cheap proxy produces nothing that beats AdamW); and there is a large, noisy gap between proxy tasks (minutes on one chip) and target tasks (>10⁴× more compute), so "good on proxy" routinely fails to transfer — a *meta-overfitting* phenomenon where search fitness keeps rising while held-out larger-task performance declines.

## Baselines

**AdamW (Loshchilov & Hutter 2019).** mₜ = β₁mₜ₋₁ + (1−β₁)gₜ; vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²; θ ← θ − η(mₜ/(√vₜ + ε) + λθ), with bias correction. Strong, near-universal default. Limitations a successor would target: two moment buffers (memory), extra hyperparameters (ε and, for Adafactor, factorization knobs), and the fact that it was never *searched* — possibly leaving better rules undiscovered.

**signSGD / signSGD-momentum (Bernstein et al. 2018).** θ ← θ − η·sign(g) (or sign of a gradient EMA). Uniform step magnitude, memory-light, large-batch friendly. Gap: its momentum rule is fixed and simple — the momentum buffer it signs is just a single EMA of g, with one constant governing the whole rule.

**NAdam (Dozat 2016).** Folds the freshly-updated first moment together with the current gradient when computing the step (Nesterov-style look-ahead inside Adam). Combines g and m for the update but keeps Adam's coupled second moment.

**Neural Optimizer Search — PowerSign / AddSign (Bello et al. 2017).** RL over fixed-operand expression trees produced sign-based update rules. Gap: restricted search space (can't alter momentum tracking), and the results did not generalize to real large-scale tasks.

**AutoML-Zero (Real et al. 2020).** Regularized evolution over full ML programs on toy tasks. Establishes the evolutionary-program-search machinery but does not target optimizers that transfer to state-of-the-art training.

## Evaluation settings

The yardsticks against which a discovered optimizer would be measured, all pre-existing: image classification on ImageNet and JFT-300M with ViT (S/16, B/16, …), ResNet-50, MLP-Mixer and hybrid architectures (top-1 accuracy, pre-training compute); vision-language contrastive learning (CLIP/BASIC-style, zero-shot and fine-tuning ImageNet accuracy); diffusion image generation (FID); autoregressive and masked language modeling and downstream NLG/NLU fine-tuning (perplexity, exact match, accuracy across the GLUE/SuperGLUE-style suite). For the *search* itself, the proxy tasks are small (e.g. a 3-layer, 96-hidden-unit, 3-head ViT on 10% of ImageNet for 30k steps, batch 64, 64×64 images, patch 16), and meta-validation tasks scale those up to detect meta-overfitting. Standard augmentations (RandAugment, Mixup) and the usual learning-rate schedules, gradient/update clipping are in place.

## Code framework

The substrate is a program-search harness over optimizer programs, plus the ordinary training loop the optimizer plugs into. An optimizer is represented as a `train` function with the *same input/output signature as AdamW* — inputs are the weight `w`, gradient `g`, learning-rate value `lr`, and a bounded number of extra state variables; output is the `update` and the new state — so any discovered program has memory no larger than Adam.

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

# Search harness scaffold (already-known evolutionary machinery):
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
