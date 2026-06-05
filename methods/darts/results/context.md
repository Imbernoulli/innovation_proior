# Context

## Research question

Designing neural network architectures by hand is slow and requires expert intuition. Automating it — neural architecture search — is appealing, but the dominant approaches are brutally expensive: they treat the architecture as a point in a *discrete*, non-differentiable space and search it with reinforcement learning or evolution, training thousands of candidate networks to convergence to get a single scalar reward each. That costs thousands of GPU-days. The question: can architecture search be made *orders of magnitude* cheaper by turning the discrete search into a continuous, differentiable optimization, so that the architecture is learned by gradient descent jointly with the network weights?

## Background

The field state at the time: NAS works but is dominated by black-box optimizers over a discrete space.

- **Cell-based search spaces (Zoph et al., 2017; Real et al.; Liu et al.).** Rather than search a whole network, search a small *cell* (a directed acyclic graph of operations) and stack copies of it. This shrinks the search space and gives transferable building blocks. A convolutional network stacks the cell; a recurrent network connects it recursively. A cell has a few input nodes, several intermediate nodes, and an output formed by combining the intermediate nodes.
- **Reinforcement-learning NAS (Zoph & Le, 2016; Zoph et al., 2017; Pham et al., ENAS).** A controller RNN samples a discrete architecture; the child is trained and its validation accuracy is the reward; the controller is updated by policy gradient. Treats validation performance as a non-differentiable reward.
- **Evolutionary NAS (Real et al.; Liu et al.).** Mutate/select architectures using validation fitness. Same discrete, sample-and-evaluate structure.
- **Bilevel / gradient-based hyperparameter optimization (Maclaurin et al., 2015; Pedregosa; Franceschi et al.).** Treat hyperparameters as an upper-level variable optimized against validation loss, with weights as a lower-level variable optimized against training loss; differentiate through the inner optimization. The architecture can be viewed as a (very high-dimensional) hyperparameter.
- **One-step unrolled / truncated differentiation (Finn et al., MAML; Luketina et al.; Metz et al., unrolled GANs).** Approximate the inner argmin by a single gradient step and differentiate through it — a cheap surrogate for the exact inner solution.

A diagnostic point that motivates the whole approach: the reason NAS is expensive is precisely that the search variable is discrete, so the validation signal can only be obtained by *sampling and fully training* candidates — there is no gradient of validation performance with respect to the architecture. If the architecture choice could be made continuous, that gradient would exist, and search would become ordinary (bilevel) optimization.

## Baselines

- **NAS with RL (Zoph et al., 2017).** Controller RNN + policy gradient over discrete cell choices; state-of-the-art architectures but thousands of GPU-days. Gap: discrete search, enormous compute.
- **Evolutionary search (Real et al., 2018, AmoebaNet).** Tournament evolution over architectures; competitive accuracy, also very expensive. Gap: discrete, sample-inefficient.
- **ENAS (Pham et al., 2018).** Weight sharing across sampled child architectures cuts cost dramatically while keeping the RL controller. Gap: still a discrete controller optimized by RL; search is over sampled discrete architectures rather than a continuous, directly-differentiable representation.

## Evaluation settings

- **Datasets / tasks.** CIFAR-10 (image classification, convolutional cells), Penn Treebank (language modeling, recurrent cells), with transfer to ImageNet (mobile setting) and WikiText-2.
- **Search protocol.** Hold out half the CIFAR-10 training set as the validation split used to drive the architecture variable; search a small network (e.g. 8 cells), then build a larger network (e.g. 20 cells) from the discovered cell for final evaluation. Cells located at 1/3 and 2/3 depth are reduction cells (stride two); normal cells preserve resolution.
- **Operation set (conv).** 3×3 and 5×5 separable convolutions, 3×3 and 5×5 dilated separable convolutions, 3×3 max pooling, 3×3 average pooling, identity (skip), and a special *zero* (no connection). ReLU-Conv-BN order; separable convs applied twice.
- **Metrics.** Test error / perplexity of the network built from the searched cell, plus **search cost in GPU-days** (the headline axis NAS is judged on) and parameter count.

## Code framework

The primitives that already exist: an autodiff framework with conv/pool/identity ops and an RNN/LSTM, SGD and Adam optimizers, cross-entropy, a DAG-of-nodes cell abstraction where each intermediate node sums transformed predecessors, and a train/validation data split. What does *not* yet exist is how a *choice of operation* on each edge is represented so it can be optimized by gradient descent, and how architecture and weights are optimized jointly. Those are the empty slots.

```python
import torch, torch.nn as nn, torch.nn.functional as F

OPS = ['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5',
       'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'none']  # candidate operations

class Edge(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ops = nn.ModuleList([make_op(name, C) for name in OPS])
        # TODO: how is the CHOICE among self.ops represented so it is differentiable?
    def forward(self, x):
        # TODO: combine candidate operations into a single edge output
        pass

class Cell(nn.Module):                  # DAG of N nodes; each node sums its predecessors' edges
    def __init__(self, N, C):
        super().__init__()
        self.edges = nn.ModuleDict()    # (i, j) -> Edge
        ...
    def forward(self, s0, s1):
        ...

def search(model, train_loader, val_loader):
    # TODO: jointly optimize the architecture representation (on validation loss)
    #       and the weights (on training loss)
    pass

def derive_discrete_cell(model):
    # TODO: turn the learned continuous representation into a discrete cell
    pass
```
