## Research question

Training a large convolutional network on a dataset like ImageNet takes a long time on a single device, and the obvious way to speed it up is to throw more hardware at it: data-parallel synchronous stochastic gradient descent (SGD), where the global mini-batch of `B` examples is split across `N` worker nodes, each node computes the gradient on its `B/N`-example chunk, and the partial gradients are summed before a single weight update. To keep each worker busy enough to be efficient, its chunk has to be reasonably large, so adding workers to go faster forces the **global batch size `B` up**. The promise is near-linear speedup — more nodes, larger batch, fewer (but heavier) steps, less wall-clock time.

The trouble is that this does not come for free. At a fixed number of epochs, growing `B` by a factor `k` means `k` times fewer weight updates, so each step has to make proportionally more progress or training stalls. And empirically, simply enlarging the batch tends to **lower final model accuracy**. The precise problem is: find a training procedure that lets the global batch grow to many thousands of examples — into the range where a large pod of accelerators is fully utilized — **without diverging during optimization and without losing accuracy relative to the small-batch baseline**, and ideally without re-tuning a pile of hyper-parameters for every new batch size.

## Background

**Synchronous data-parallel SGD.** At step `t` a mini-batch of `B` samples `x_i` is drawn; the loss gradients `∇L(x_i, w)` are averaged and the weights are stepped:
`w_{t+1} = w_t − λ · (1/B) Σ_{i=1}^B ∇L(x_i, w_t)`.
The sum is trivially parallelized across `N` units (each handles `B/N` samples), which is exactly why scaling out the hardware scales up `B`. Momentum is the standard addition: a running velocity `v_{t+1} = m·v_t + (gradient term)`, `w_{t+1} = w_t − v_{t+1}`, with weight decay `β` adding `β·w` to the gradient. A global learning-rate schedule (e.g. polynomial decay `λ_t = λ_0·(1 − t/T)^p`) is applied on top.

**The linear learning-rate scaling rule.** Krizhevsky (2014) argued that when you grow the batch by `k`, you should grow the learning rate by `k` and leave momentum and weight decay unchanged. The reasoning: two consecutive updates with batch `B` and rate `λ`,
`w_{t+2} = w_t − λ·(1/B)( Σ_i ∇L(x_i, w_t) + Σ_j ∇L(x_j, w_{t+1}) )`,
are well approximated by one update with batch `2B` and rate `2λ`,
`w_{t+1} = w_t − 2λ·(1/2B) Σ_i ∇L(x_i, w_t)`,
**provided** `∇L(x_j, w_{t+1}) ≈ ∇L(x_j, w_t)` — i.e. the weights barely move within the step. That assumption holds while steps are small and breaks once the steps get large. A variance-based alternative, **square-root scaling** (rate ∝ `√B`, since the standard error of the mini-batch gradient falls as `1/√B`), is more conservative; in practice linear scaling worked better up to a moderate ceiling. With linear scaling, AlexNet trained at `B=1K` with ~1% accuracy loss, but scaling past `B≈2K` diverged for the required large rates.

**The role of normalization.** Linear scaling was observed to work much better for networks with Batch Normalization (Inception at `B=6400`, ResNet-152 at `B=5K`). BN stabilizes the activations across a layer's batch, which widens the range of learning rates a network tolerates and pushes the divergence ceiling higher.

**Warmup.** Goyal et al. (2017) found the linear-scaling rule is harmful precisely in the **initial phase** — the early large steps destabilize the network. Their fix is a learning-rate **warmup**: start from a small, "safe" rate and ramp it up over the first few epochs to the target rate, then hand off to the regular decay schedule. With linear scaling plus warmup, ResNet-50 reached `B=8K` matching the small-batch baseline. Linear scaling + warmup became the state-of-the-art recipe for large-batch training.

**Motivating diagnostic — what actually breaks.** Several observations about existing training set up the problem. First, applying linear scaling + warmup to AlexNet on ImageNet, scaling stalls past `B=2K`: training diverges for rates above a threshold even with warmup, and the best attainable accuracy degrades sharply as `B` grows (a `B=256` baseline near 58% falls to the low-50s at `B=4K` and mid-40s at `B=8K`). Replacing the network's Local Response Normalization with Batch Normalization recovers much of this: large rates become usable, the good-rate interval widens, and the `B=8K` gap shrinks markedly, but a residual gap remains. Second, that residual gap is checked against the sharp-minima generalization-gap hypothesis (Keskar et al., 2016, who reported large batches converge to sharp minimizers that generalize worse): measuring the train–test loss gap at small vs. large batch shows no significant difference, so the residual loss is **not** a generalization gap; it is under-optimization. Third, and the key measurement: the ratio of the L2 norm of a layer's weights to the norm of its gradient, `‖w‖ / ‖∇L(w)‖`, recorded layer by layer after one iteration, **varies by orders of magnitude across the network**: about `5.76` for the first convolutional weight tensor and about `1345` for a later fully connected weight tensor, with weights and biases differing within the same layer. The measured ratios are especially important during the earliest iterations, when the network is most sensitive to an oversized global step. (Hoffer et al., 2017, separately tried a less aggressive square-root scaling with a "Ghost" form of BN to reach `B=8K`, but accuracy stayed well below baseline.)

**Why a single global rate is the wrong knob.** Standard SGD applies the *same* rate `λ` to every layer. When `λ` is large, for a layer whose weight-to-gradient ratio is small the update `‖λ·∇L(w)‖` can exceed `‖w‖`; that layer's parameters move more than their own magnitude in a single step, which is the instability. The same `λ` that is safe for the worst-conditioned layer is far too small for a layer with a large ratio, whose weights then barely move. So the worst layer sets the divergence ceiling on `λ`, and everything else is under-served. This is the structural reason a single global learning rate fails at large batch, and it reframes what warmup is doing: starting `λ` small enough to survive the earliest, most sensitive steps, then delaying the target large-batch rate until the layer-wise ratios have had time to settle.

## Baselines

**SGD with momentum + linear LR scaling.** The workhorse for large-batch training. Heavy-ball velocity `v_{t+1} = m·v_t + λ·(g_t + β·w_t)`, `w_{t+1} = w_t − v_{t+1}`, with `λ` scaled linearly with `B`. Core idea above. Limitation: a single global `λ`; the worst-conditioned layer caps `λ`; diverges past a batch-size ceiling (≈`2K` for AlexNet) during the initial phase.

**Linear scaling + warmup (Goyal et al., 2017).** The recipe wrapped around momentum SGD: scale `λ` linearly with `B` and ramp it from a small value over the first few epochs before the regular decay. Reaches `B=8K` on a deep residual classifier at baseline accuracy. Limitation: still a single global rate, so for some networks (e.g. AlexNet without BN) it diverges well before `B=8K`; the warmup length and target rate need hand-tuning per network/batch.

**Square-root scaling + Ghost BN (Hoffer et al., 2017).** A more conservative `λ ∝ √B` with a batch-normalization variant computed over small "ghost" sub-batches, used to reach `B=8K`. Limitation: accuracy stayed substantially below the small-batch baseline.

**Per-coordinate adaptive optimizers (Adam — Kingma & Ba, 2014; RMSProp — Tieleman & Hinton, 2012).** Maintain per-weight gradient statistics and rescale each coordinate's step by them (Adam: `m̂/√v̂`; RMSProp: `g/√(EMA of g²)`). They adapt the rate to each weight's gradient history. Limitation in this setting: the adaptation is **per individual weight**, which is noisy and does not directly control how large a step is relative to the *layer's* weight magnitude; the step size is set by gradient statistics, not by the weight norm, so the per-layer "fraction of its own size that a layer moves" — the quantity that governs divergence at large batch — is not what is being regulated.

**Block-diagonal / diagonal rescaling (Lafond et al., 2017).** Preconditioning that rescales the gradient by a block-diagonal matrix, one block per group of parameters, as a cheap approximation to second-order information. Relevant because a per-layer rescaling of the update is a special case of block-diagonal rescaling with one block per layer. Limitation: framed generally; does not prescribe the specific per-layer scale that fixes the large-batch divergence.

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
