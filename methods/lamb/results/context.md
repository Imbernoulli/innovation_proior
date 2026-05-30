# Context: large-batch optimization for deep networks

## Research question

Training a state-of-the-art deep network is dominated by wall-clock time: a large language model or a deep image classifier can take days on a small cluster of accelerators, because stochastic gradient descent is inherently sequential — each parameter update waits on the previous one. The one lever that turns more hardware into less time without changing the model is the **mini-batch size**: a batch of size `b` can be split across many devices, each computing part of the gradient in parallel, and the partial gradients summed. The promise is near-linear speedup — double the devices, double the batch, halve the number of steps.

The problem is that this promise does not hold naively. At a fixed number of training epochs, the number of optimizer updates `T = (#examples × epochs) / b` falls *linearly* as `b` grows, so each step must accomplish proportionally more or training stalls. Pushing harder by raising the learning rate works only up to a point: past some batch-size ceiling, training becomes unstable, diverges, or converges to a solution that generalizes worse. The goal is an optimizer that lets the batch size scale to tens of thousands of examples — to the memory limit of a large accelerator pod — **without per-batch-size hand-tuning and without losing final accuracy**, so that the days-long training of large language and vision models collapses to hours or minutes.

## Background

**Why a larger batch should help, and by how much.** A mini-batch gradient `g = (1/b) Σ ∇ℓ(x, s)` is an unbiased estimate of the true gradient `∇f`. Its variance scales as `1/b`: averaging over `b` independent samples reduces the per-coordinate variance of the estimate by a factor of `b`. A lower-variance gradient can tolerate a larger step, so the conventional wisdom is to raise the learning rate as the batch grows. Two scaling rules circulate. The variance argument suggests **square-root scaling** — multiply the learning rate by `√b` when the batch grows by `b` (Krizhevsky, 2014) — because the standard deviation of the estimate falls as `1/√b`. Empirically, Krizhevsky (2014) and Goyal et al. (2017) found **linear scaling** — learning rate ∝ `b` — works better up to a moderate batch-size ceiling.

**The degradation diagnostic — what breaks at large batch.** Several observations about *existing* training set up the problem. Keskar et al. (2016) reported that very large batches drive the optimizer toward "sharp" minima that generalize worse, opening a **generalization gap** between large-batch and small-batch training. Hoffer et al. (2017) found this gap could be partly closed by training longer, confirming that the gap is tied to taking *too few* updates. Goyal et al. (2017) showed that linear scaling is *harmful during the initial phase* of training — the early, large steps destabilize the network — and required a hand-tuned **warmup** that ramps the learning rate up slowly before the regular schedule, after which a batch of 8192 could match small-batch accuracy on a deep image classifier. But Shallue et al. (2018) measured these scaling heuristics across many tasks and found they *do not transfer*: the batch-size ceiling and the right scaling exponent differ from problem to problem. So a fixed scaling rule plus warmup is fragile, and a single global learning rate that is correct for the network as a whole is the thing being strained.

**The asynchronous alternative and why it lost.** Before large synchronous batches became feasible, the route to parallel SGD was asynchronous distributed training (Dean et al., 2012; Recht et al., 2011), where workers update a shared parameter server without locking. The staleness of the gradients (a worker computes a gradient at parameters that have since moved) limits how far this parallelizes and degrades the solution. As accelerators grew able to hold large batches and compute their gradients in parallel, synchronous large-batch SGD became the cleaner alternative — no staleness, exact gradients — provided the optimization could be stabilized.

**Per-layer geometry.** A neural network is a stack of layers, and the parameters of different layers live on very different scales — a normalization layer's gain, an embedding matrix, and a deep weight matrix can differ by orders of magnitude in their norm `‖x^(i)‖` and in the norm of the step an optimizer wants to take `‖u^(i)‖`. A single global learning rate `η` applies the *same* multiplier to every layer. If `η` is large enough to make progress on a layer whose update-to-weight ratio is small, it is far too large for a layer whose ratio is large, and that layer's parameters move a large fraction of their own norm in one step — the instability. This per-layer mismatch is the structural reason a single global learning rate fails at large batch.

**Nonconvex first-order guarantees.** The yardstick for an optimizer on a smooth nonconvex objective `f` is how fast the expected gradient norm `E‖∇f(x)‖²` shrinks toward zero (a stationary point). For SGD with a large batch `b = T` and an appropriate constant step, Ghadimi & Lan (2013) give
`E‖∇f(x_a)‖² ≤ O((f(x_1) − f*)·L_∞ / T + ‖σ‖² / T)`,
where `x_a` is a uniformly random iterate, `σ` collects the per-coordinate gradient standard deviations, and `L_∞ = max_i L_i` is the **largest** per-layer smoothness constant. The dependence on `L_∞` is pessimistic: a single badly-conditioned layer sets the rate for the whole network. An optimizer whose rate depended on the *average* smoothness `L_avg = (1/h) Σ_i L_i` instead would be provably better whenever curvature is uneven across layers.

## Baselines

**SGD with momentum.** The workhorse. Heavy-ball momentum `m_t = β₁ m_{t-1} + (1−β₁) g_t`, update `x_{t+1} = x_t − η m_t` (with optional `λ x_t` weight decay folded into `g`). Momentum reduces the variance of the step at the cost of a little bias. Limitation: a single global `η`, tuned painstakingly, and the nonconvex rate carries the worst-layer `L_∞`. Past a batch-size ceiling it needs the Goyal warmup and still stalls.

**Adam and AdamW.** Adam keeps a first-moment EMA `m_t = β₁ m_{t-1} + (1−β₁) g_t` and a second-moment EMA `v_t = β₂ v_{t-1} + (1−β₂) g_t²`, debiases them `m̂_t = m_t/(1−β₁^t)`, `v̂_t = v_t/(1−β₂^t)`, and steps `x_{t+1} = x_t − η · m̂_t/(√v̂_t + ε)`. The per-coordinate division by `√v̂` is an adaptive, dimensionwise rescaling that makes Adam robust on ill-conditioned objectives and strong on attention/language models. AdamW decouples weight decay from the gradient, applying `−η λ x_t` directly rather than adding `λ x_t` to `g` before the moment EMAs, which fixes an interaction between L2 regularization and the adaptive denominator. Limitation: Adam is still **globally** scaled — one `η` across all layers — so it inherits the large-batch instability; it tends to plateau below momentum-SGD's accuracy on deep image classifiers, and on large language models it stops scaling past a moderate batch and diverges at the largest batches.

**Linear / square-root learning-rate scaling with warmup.** Not an optimizer but the standard recipe wrapped around SGD: scale `η` with batch size (linearly per Goyal et al. 2017, or as `√b` per the variance argument) and ramp `η` up over the first few epochs. Lets a deep image classifier reach a batch of 8192. Limitation (Shallue et al., 2018): the recipe is problem-specific and breaks beyond its tuned ceiling.

**Layerwise-trust-ratio momentum (You et al., 2017).** The key reaction point. Instead of one global `η`, give every layer `i` its own effective learning rate set by the ratio of its weight norm to its update norm. With momentum `m_t` as the base step,
`x_{t+1}^(i) = x_t^(i) − η_t · (φ(‖x_t^(i)‖) / ‖m_t^(i)‖) · m_t^(i)`,
where `φ` is a (typically clipped-identity) scaling function. The multiplier `φ(‖x^(i)‖)/‖m^(i)‖` makes the per-layer step proportional to the weight norm, decoupling the global `η` from each layer's geometry; with `φ(z)=z` the multiplier is `‖x^(i)‖/‖m^(i)‖`, readable as a per-layer estimate of `1/L_i`. This trained a deep image classifier at a batch of 32k in minutes. Limitations: (1) **no convergence theory** — the adaptation is a heuristic; (2) the gains **do not transfer to attention/language models**, where this momentum-based layerwise scheme trains poorly and diverges at the largest batches — precisely the regime where per-coordinate (Adam-style) adaptivity is known to matter.

**signSGD (Bernstein et al., 2018).** Steps along `sign(g_t)` coordinatewise. Its nonconvex analysis introduces a useful device: comparing a sign/normalized method against SGD through *density* quantities — how concentrated the gradient is versus the curvature and the noise — and a bound on the probability that a stochastic gradient's sign disagrees with the true gradient's sign, `P(sign g_j ≠ sign ∇f_j) ≤ σ_j / (√b · |∇f_j|)`, which follows from the bounded gradient variance. This is the analytic template for bounding a normalized, layerwise update.

## Evaluation settings

The natural yardsticks already exist before any new optimizer. **Language model pre-training**: a deep bidirectional transformer pre-trained on a concatenation of Wikipedia (~2.5B words) and BooksCorpus (~800M words), in two phases — most epochs at sequence length 128, a final tenth at sequence length 512 — with downstream quality measured by the **F1 score on the SQuAD-v1** question-answering task. The standard schedule runs ~1M iterations at batch 512 over ~3 days on 16 accelerator chips, with a polynomially decaying learning rate. **Image classification**: ResNet-50 on ImageNet, **top-1 / top-5 validation accuracy** over a fixed 90-epoch budget, with the Goyal et al. (2017) recipe (5-epoch warmup, ×0.1 learning-rate drops at epochs 30/60/80) as the baseline schedule; small-scale checks on CIFAR-10 (a 9-layer residual net) and MNIST (LeNet). Hardware is accelerator pods whose memory caps the per-step batch (the largest sequence-512 batch fits ~32k). Optimizers are compared at matched epochs across a batch-size sweep (512 → 32k), reporting accuracy/F1, number of steps, and wall-clock time. Baseline optimizers are tuned by grid search over learning rate (and weight decay where applicable). A practical caveat observed at large batch: validation *loss* is unreliable — a lower loss can accompany lower accuracy — so accuracy/F1 is the reported metric.

## Code framework

The primitives that already exist: a data pipeline yielding mini-batches, autodiff giving per-parameter gradients, and a base optimizer abstraction that maintains per-parameter state and applies an update. The training loop is standard. What does not yet exist is the optimizer that adapts the step per layer for the large-batch regime — that is the single empty slot below.

```python
import torch
from torch.optim import Optimizer

class LargeBatchOptimizer(Optimizer):
    """Per-parameter optimizer for the large-batch regime.

    Maintains base-optimizer moment state per parameter, forms a per-parameter
    update from it, then rescales each parameter-group ("layer") step using that
    layer's own geometry so a single global learning rate is decoupled from
    per-layer scale. The body is the contribution.
    """
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                # TODO: initialize / update base-optimizer moment state from g
                # TODO: form the per-parameter update direction from the state
                # TODO: rescale this layer's step from its own geometry
                # TODO: apply the update to p
                pass
        return closure() if closure is not None else None


def sqrt_lr_scaling(base_lr, base_batch, batch):
    # TODO: scale the global learning rate with batch size
    pass

def linear_epoch_warmup(step, warmup_steps, target_lr):
    # TODO: ramp the learning rate from 0 to target over warmup_steps
    pass


# training loop (pre-existing)
def train(model, data, optimizer, scheduler, steps):
    for step in range(steps):
        x, y = next(data)
        loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        # TODO: set optimizer learning rate from scheduler (warmup + decay)
        optimizer.step()
```
