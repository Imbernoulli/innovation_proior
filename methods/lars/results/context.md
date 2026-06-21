## Research question

Training a large convolutional network on a dataset like ImageNet takes a long time on a single device, and the obvious way to speed it up is to throw more hardware at it: data-parallel synchronous stochastic gradient descent (SGD), where the global mini-batch of `B` examples is split across `N` worker nodes, each node computes the gradient on its `B/N`-example chunk, and the partial gradients are summed before a single weight update. To keep each worker busy enough to be efficient, its chunk has to be reasonably large, so adding workers to go faster forces the **global batch size `B` up**. The promise is near-linear speedup — more nodes, larger batch, fewer (but heavier) steps, less wall-clock time.

The question is: how can the learning-rate schedule be set — possibly per-layer — so that synchronous data-parallel SGD continues to converge at very large batch sizes (into the range where a large pod of accelerators is fully utilized) without accuracy loss relative to the small-batch baseline?

## Background

**Synchronous data-parallel SGD.** At step `t` a mini-batch of `B` samples `x_i` is drawn; the loss gradients `∇L(x_i, w)` are averaged and the weights are stepped:
`w_{t+1} = w_t − λ · (1/B) Σ_{i=1}^B ∇L(x_i, w_t)`.
The sum is trivially parallelized across `N` units (each handles `B/N` samples), which is exactly why scaling out the hardware scales up `B`. Momentum is the standard addition: a running velocity `v_{t+1} = m·v_t + (gradient term)`, `w_{t+1} = w_t − v_{t+1}`, with weight decay `β` adding `β·w` to the gradient. A global learning-rate schedule (e.g. polynomial decay `λ_t = λ_0·(1 − t/T)^p`) is applied on top.

**The linear learning-rate scaling rule.** Krizhevsky (2014) argued that when you grow the batch by `k`, you should grow the learning rate by `k` and leave momentum and weight decay unchanged. The reasoning: two consecutive updates with batch `B` and rate `λ`,
`w_{t+2} = w_t − λ·(1/B)( Σ_i ∇L(x_i, w_t) + Σ_j ∇L(x_j, w_{t+1}) )`,
are well approximated by one update with batch `2B` and rate `2λ`,
`w_{t+1} = w_t − 2λ·(1/2B) Σ_i ∇L(x_i, w_t)`,
**provided** `∇L(x_j, w_{t+1}) ≈ ∇L(x_j, w_t)` — i.e. the weights barely move within the step. A variance-based alternative, **square-root scaling** (rate ∝ `√B`, since the standard error of the mini-batch gradient falls as `1/√B`), is more conservative; in practice linear scaling worked better up to a moderate ceiling.

**The role of normalization.** Linear scaling was observed to work much better for networks with Batch Normalization (Inception at `B=6400`, ResNet-152 at `B=5K`). BN stabilizes the activations across a layer's batch, which widens the range of learning rates a network tolerates.

**Warmup.** Goyal et al. (2017) found the linear-scaling rule can be harmful in the initial phase. Their fix is a learning-rate **warmup**: start from a small, "safe" rate and ramp it up over the first few epochs to the target rate, then hand off to the regular decay schedule. With linear scaling plus warmup, ResNet-50 reached `B=8K` matching the small-batch baseline. Linear scaling + warmup became the state-of-the-art recipe for large-batch training.

**Per-layer weight and gradient statistics.** A diagnostic measurement one can make after the first training iteration is the ratio of the L2 norm of a layer's weights to the L2 norm of its gradient, `‖w‖ / ‖∇L(w)‖`, recorded layer by layer. These ratios vary across the network — different convolutional weight tensors, fully connected weight tensors, and bias terms can take substantially different values. The ratios are also different at different points in training.

**Sharp-minima hypothesis (Keskar et al., 2016).** One proposed explanation for large-batch accuracy loss is that large batches converge to sharp minimizers of the training function that generalize poorly, producing a wider train–test loss gap. Measuring the train–test loss gap at small vs. large batch is therefore a diagnostic for whether under-optimization or a generalization gap is the dominant effect.

## Baselines

**SGD with momentum + linear LR scaling.** The workhorse for large-batch training. Heavy-ball velocity `v_{t+1} = m·v_t + λ·(g_t + β·w_t)`, `w_{t+1} = w_t − v_{t+1}`, with `λ` scaled linearly with `B`.

**Linear scaling + warmup (Goyal et al., 2017).** The recipe wrapped around momentum SGD: scale `λ` linearly with `B` and ramp it from a small value over the first few epochs before the regular decay. Reaches `B=8K` on a deep residual classifier at baseline accuracy.

**Square-root scaling + Ghost BN (Hoffer et al., 2017).** A more conservative `λ ∝ √B` with a batch-normalization variant computed over small "ghost" sub-batches, used to reach `B=8K`.

**Per-coordinate adaptive optimizers (Adam — Kingma & Ba, 2014; RMSProp — Tieleman & Hinton, 2012).** Maintain per-weight gradient statistics and rescale each coordinate's step by them (Adam: `m̂/√v̂`; RMSProp: `g/√(EMA of g²)`). They adapt the rate to each weight's gradient history.

**Block-diagonal / diagonal rescaling (Lafond et al., 2017).** Preconditioning that rescales the gradient by a block-diagonal matrix, one block per group of parameters, as a cheap approximation to second-order information.

## Evaluation settings

The natural yardsticks already exist. **Image classification on ImageNet** (Deng et al., 2009): a deep convolutional classifier (an AlexNet-style network, and a 50-layer residual network, He et al., 2016), trained with SGD + momentum 0.9, weight decay, and a polynomial-decay (power 2) learning-rate schedule over a fixed epoch budget (≈90–128 epochs), reporting **top-1 (and top-5) validation accuracy**, averaged over the last few epochs. The baseline is a small global batch (256–512). The comparison sweeps batch size upward (`4K`, `8K`, `16K`, `32K`) at matched epochs, recording the best accuracy and the learning rate that achieves it; very large batches are emulated by accumulating gradients over sub-batches when they exceed device memory. A diagnostic protocol also exists: measure the train–test loss gap to test the sharp-minima hypothesis, and measure per-layer `‖w‖/‖∇L‖` after the first iteration. Baselines are tuned by sweeping the base learning rate. A practical caveat: at very large batch, accuracy (not loss alone) is the metric to trust.

## Code framework

The primitives that already exist: a data pipeline yielding mini-batches, autodiff producing per-parameter gradients, a base optimizer abstraction holding per-parameter momentum state and applying an update with a global learning rate, and a learning-rate scheduler (warmup + polynomial decay). The training loop is standard. The optimizer slot below is where the direction sent into the ordinary momentum step can be replaced by a large-batch-aware direction.

```python
import torch
from torch.optim import Optimizer

class LargeBatchOptimizer(Optimizer):
    """Momentum-SGD optimizer for the large-batch regime.

    Holds per-parameter momentum and applies a step with a global learning
    rate. The open slot constructs the direction that will enter momentum,
    including any weight decay and any large-batch-specific adjustment.
    """
    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, dampening=0.0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        dampening=dampening, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']                 # global learning rate (from scheduler)
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # TODO: construct the direction that enters momentum.
                # It may use p, grad, and weight_decay.
                adjusted_grad = grad
                pass
                grad = adjusted_grad

                # plain momentum-SGD with the global learning rate
                if momentum != 0:
                    state = self.state[p]
                    buf = state.get('momentum_buffer')
                    if buf is None:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)
                    grad = grad.add(buf, alpha=momentum) if nesterov else buf
                p.add_(grad, alpha=-lr)
        return loss
```
