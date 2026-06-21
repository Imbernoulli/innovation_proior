# Context

## Research question

Modern neural networks are heavily overparametrized — convolutional and fully-connected models carry millions of weights, far more than the task seems to need. This is expensive: memory, compute, and energy at both training and inference time, which matters acutely on resource-limited devices. The standard belief is that a much smaller subnetwork could match the full network's accuracy, and compression-based arguments also suggest a route to tighter generalization. So the goal is to learn a sparse network — keep only a small fraction of the weights non-zero — without losing the accuracy of the dense reference model.

Formally, given a dataset and a desired sparsity level, the problem is the constrained minimization

```
min_w  L(w; D) = (1/n) Σ_i ℓ(w; (x_i, y_i))
s.t.   w ∈ R^m,  ‖w‖_0 ≤ κ,
```

where ℓ is a standard loss (e.g. cross-entropy), m is the total number of parameters, ‖·‖_0 is the L0 "count of non-zeros", and κ is the number of weights allowed to remain. The L0 constraint is combinatorial, so the practical question is: by what *criterion* do we decide which connections to keep, and *when* in the training pipeline do we apply it?

## Background

The pruning literature splits into two families.

**Penalty / projection methods.** One can add sparsity-enforcing penalties to the objective (L1/L0-style terms; classical weight-decay-with-thresholding). A more recent variant treats the constrained problem directly with a stochastic projected-gradient descent where the projection step *is* the pruning.

**Saliency methods.** These treat pruning as selectively removing redundant connections, ranked by a saliency score s_j for connection j. Two criteria dominate:
- *Magnitude*: s_j = |w_j|. Small weights are deemed unimportant.
- *Hessian / curvature* (Optimal Brain Damage / Optimal Brain Surgeon): start from the Taylor expansion of the loss around a trained minimum,
  ```
  δL = (∂L/∂w)ᵀ δw + ½ δwᵀ H δw + O(‖δw‖³),
  ```
  assume convergence so the first-order term vanishes, and read off s_j = w_j² H_jj / 2 (diagonal) or s_j = w_j² / (2 [H⁻¹]_jj) (full).

Both dominant criteria depend on the **scale of the weights**, which ties them to a trained network — magnitude or curvature scores computed on random initial weights carry the distributional properties of the initialization. Pruning has classically been applied as a post-training step, embedded in **prune–retrain cycles**: train, score, remove, retrain to recover, repeat.

Two further building blocks from the field are relevant:
- **Weight initialization theory** (LeCun; Glorot/Xavier variance scaling). With a fixed-variance Gaussian init, the variance of the forward signal (and of gradients) drifts layer to layer, so gradient magnitudes pick up architecture-specific scaling. Variance-scaling initializations are designed precisely so the signal variance is preserved through the layers.
- **Influence functions** (Koh & Liang, 2017): perturb an *input* example and measure the resulting change in loss to gauge that example's importance.

## Baselines

- **Magnitude-based pruning (Han et al., 2015; Guo et al., 2016).** Train the dense net, threshold weights by |w|, retrain. Iterating gives high sparsity at good accuracy.
- **Optimal Brain Damage / Optimal Brain Surgeon (LeCun et al., 1990; Hassibi & Stork, 1993).** Curvature-based saliency w_j² H_jj / 2 (or full-Hessian variant). Applied at a trained minimum, where the first-order Taylor term can be dropped.
- **Early sensitivity criteria (Mozer & Smolensky, 1988; Karnin, 1990).** Identify elements whose removal least degrades performance — their saliency is essentially −∂L/∂w (or w.r.t. neuron activity), folded into the learning process.
- **Direct constrained optimization via projected gradient (Carreira-Perpiñán & Idelbayev, 2018).** Optimize the L0-constrained objective with projection-as-pruning.

## Evaluation settings

The natural yardsticks at the time:
- **Datasets.** MNIST (28×28 grayscale digits), CIFAR-10 (32×32 color, 10 classes), and Tiny-ImageNet (downscaled ImageNet subset). Variants of MNIST/Fashion-MNIST (including inverted versions) are useful for probing whether retained connections are task-relevant.
- **Architectures.** Fully-connected nets (LeNet-300-100), convolutional nets (LeNet-5, AlexNet-style, VGG-style), wide residual networks, and recurrent nets (RNN/LSTM/GRU) — chosen to test robustness across convolutional, residual, and recurrent structure.
- **Metric / protocol.** Classification error/accuracy of the sparse network versus the dense reference, swept over sparsity levels κ (e.g. retaining 10%, 5%, 1% of weights). Compute cost is a secondary axis: pretraining passes, scoring passes, pruning steps, and retraining passes all matter because standard methods interleave pruning with optimization.

## Code framework

The primitives that already exist: an autodiff framework with trainable layers, an SGD/momentum optimizer, a cross-entropy loss, a data pipeline yielding mini-batches, and standard variance-scaling weight initializers. What does *not* yet exist is the criterion that decides which connections to keep and when. The scaffold below leaves that as one empty slot.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def variance_scaling_init(net):
    # Glorot/Xavier or He-style init so signal variance is preserved across layers.
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

def compute_keep_mask(net, batch, keep_fraction):
    # TODO: decide which connections to keep, returning a binary mask
    #       that retains the top-(keep_fraction) connections.
    pass

def apply_mask(net, masks):
    # TODO: fix the chosen connections to zero for the rest of training.
    pass

def train(net, loader, masks, epochs, lr):
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(net(x), y)
            loss.backward()
            # TODO: keep pruned connections at zero (no gradient flow / re-zero)
            opt.step()
```
