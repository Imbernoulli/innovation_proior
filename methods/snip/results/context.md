# Context

## Research question

Modern neural networks are heavily overparametrized — convolutional and fully-connected models carry millions of weights, far more than the task seems to need. This is expensive: memory, compute, and energy at both training and inference time, which matters acutely on resource-limited devices. The standard belief is that a much smaller subnetwork could match the full network's accuracy, and that finding it would also tighten generalization. So the goal is to learn a sparse network — keep only a small fraction of the weights non-zero — without losing the accuracy of the dense reference model.

Formally, given a dataset and a desired sparsity level, the problem is the constrained minimization

```
min_w  L(w; D) = (1/n) Σ_i ℓ(w; (x_i, y_i))
s.t.   w ∈ R^m,  ‖w‖_0 ≤ κ,
```

where ℓ is a standard loss (e.g. cross-entropy), m is the total number of parameters, ‖·‖_0 is the L0 "count of non-zeros", and κ is the number of weights allowed to remain. The L0 constraint is combinatorial, so the practical question is: by what *criterion* do we decide which connections to keep, and *when* in the training pipeline do we apply it? A good criterion would (1) be cheap, (2) not require pretraining or hand-tuned prune/retrain schedules, and (3) be robust across architectures.

## Background

The pruning literature splits into two families.

**Penalty / projection methods.** One can add sparsity-enforcing penalties to the objective (L1/L0-style terms; classical weight-decay-with-thresholding). A more recent variant treats the constrained problem directly with a stochastic projected-gradient descent where the projection step *is* the pruning. In practice these tend to be inferior to saliency-based methods in achieved sparsity and need heavily tuned hyperparameters.

**Saliency methods.** These treat pruning as selectively removing redundant connections, ranked by a saliency score s_j for connection j. Two criteria dominate:
- *Magnitude*: s_j = |w_j|. Small weights are deemed unimportant. Simple and effective, but highly heuristic — the weights to be pruned depend on learning rate and architecture (e.g. normalization layers rescale weights), and it must be applied many times.
- *Hessian / curvature* (Optimal Brain Damage / Optimal Brain Surgeon): start from the Taylor expansion of the loss around a trained minimum,
  ```
  δL = (∂L/∂w)ᵀ δw + ½ δwᵀ H δw + O(‖δw‖³),
  ```
  assume convergence so the first-order term vanishes, and read off s_j = w_j² H_jj / 2 (diagonal) or s_j = w_j² / (2 [H⁻¹]_jj) (full). More principled, but the Hessian H = ∂²L/∂w² is neither diagonal nor positive-definite in general, is approximate at best, and is intractable to compute (OBS even re-inverts H per removed weight).

A diagnostic observation about *both* dominant criteria: they depend on the **scale of the weights**. That ties them to a *trained* network — a magnitude or curvature score computed on random initial weights is uninformative because the weights have not yet organized. This is precisely why pruning has classically been a post-training step, embedded in expensive **prune–retrain cycles**: train, score, remove, retrain to recover, repeat. The retraining is what makes the loop slow and the schedule heuristic.

Two further building blocks from the field are load-bearing here:
- **Weight initialization theory** (LeCun; Glorot/Xavier variance scaling). With a fixed-variance Gaussian init, the variance of the forward signal (and of gradients) drifts layer to layer, so gradient magnitudes pick up architecture-specific scaling. Variance-scaling initializations are designed precisely so the signal variance is preserved through the layers.
- **Influence functions** (Koh & Liang, 2017): perturb an *input* example and measure the resulting change in loss to gauge that example's importance. The same perturb-and-measure idea can be redirected from inputs to *internal parameters*.

## Baselines

- **Magnitude-based pruning (Han et al., 2015; Guo et al., 2016).** Train the dense net, threshold weights by |w|, retrain. Iterating gives high sparsity at good accuracy. Gap: requires a fully trained reference and repeated retraining; scores depend on weight scale and so on learning policy and normalization choices.
- **Optimal Brain Damage / Optimal Brain Surgeon (LeCun et al., 1990; Hassibi & Stork, 1993).** Curvature-based saliency w_j² H_jj / 2 (or full-Hessian variant). Gap: requires a trained minimum (so the first-order Taylor term can be dropped), and the Hessian is intractable/ill-conditioned for large nets.
- **Early sensitivity criteria (Mozer & Smolensky, 1988; Karnin, 1990).** Identify elements whose removal least degrades performance — their saliency is essentially −∂L/∂w (or w.r.t. neuron activity), which depends on the *value of the loss before pruning* and is designed to be folded into the learning process. Gap: again requires a pretrained network and iterative optimization, and depends on weight scale.
- **Direct constrained optimization via projected gradient (Carreira-Perpiñán & Idelbayev, 2018).** Optimize the L0-constrained objective with projection-as-pruning. Gap: inferior achieved sparsity and sensitive hyperparameters compared to saliency methods.

## Evaluation settings

The natural yardsticks at the time, used purely as settings here:
- **Datasets.** MNIST (28×28 grayscale digits), CIFAR-10 (32×32 color, 10 classes), and Tiny-ImageNet (downscaled ImageNet subset). Variants of MNIST/Fashion-MNIST (including inverted versions) are useful for probing whether retained connections are task-relevant.
- **Architectures.** Fully-connected nets (LeNet-300-100), convolutional nets (LeNet-5, AlexNet-style, VGG-style), wide residual networks, and recurrent nets (RNN/LSTM/GRU) — chosen to test robustness across convolutional, residual, and recurrent structure.
- **Metric / protocol.** Classification error/accuracy of the sparse network versus the dense reference, swept over sparsity levels κ (e.g. retaining 10%, 5%, 1% of weights). Compute cost (number of prune/retrain passes) is a secondary axis. Connection saliency is computed from a single mini-batch of a reasonable size; one can also accumulate over batches or use an exponential moving average under memory limits.

## Code framework

The primitives that already exist: an autodiff framework with `Conv2d`/`Linear` layers, an SGD/momentum optimizer, a cross-entropy loss, a data pipeline yielding mini-batches, and standard variance-scaling weight initializers. What does *not* yet exist is the criterion that decides which connections to keep and when. The scaffold below leaves that as one empty slot.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def variance_scaling_init(net):
    # Glorot/Xavier or He-style init so signal variance is preserved across layers.
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

def compute_keep_mask(net, batch, keep_fraction):
    # TODO: a data-dependent criterion that scores every connection BEFORE training
    #       and returns a binary mask keeping the top-(keep_fraction) connections.
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
