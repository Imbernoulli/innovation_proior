# Context

## Research question

Designing neural network architectures by hand is slow and requires expert intuition. Neural architecture search (NAS) automates it. The dominant approaches treat the architecture as a point in a *discrete* space and search it with reinforcement learning or evolution, training many candidate networks to get a scalar validation signal for each, at a cost measured in thousands of GPU-days. The question: how to search for good cell architectures at a small fraction of that cost.

## Background

The field state at the time: NAS is driven by black-box optimizers over a discrete search space.

- **Cell-based search spaces (Zoph et al., 2017; Real et al.; Liu et al.).** Rather than search a whole network, search a small *cell* (a directed acyclic graph of operations) and stack copies of it. This shrinks the search space and gives transferable building blocks. A convolutional network stacks the cell; a recurrent network connects it recursively. A cell has a few input nodes, several intermediate nodes, and an output formed by combining the intermediate nodes.
- **Reinforcement-learning NAS (Zoph & Le, 2016; Zoph et al., 2017; Pham et al., ENAS).** A controller RNN samples a discrete architecture; the child is trained and its validation accuracy is the reward; the controller is updated by policy gradient.
- **Evolutionary NAS (Real et al.; Liu et al.).** Mutate/select architectures using validation fitness. Same discrete, sample-and-evaluate structure.
- **Bilevel / gradient-based hyperparameter optimization (Maclaurin et al., 2015; Pedregosa; Franceschi et al.).** Treat hyperparameters as an upper-level variable optimized against validation loss, with weights as a lower-level variable optimized against training loss; differentiate through the inner optimization. An architecture can be viewed as a very high-dimensional hyperparameter.
- **One-step unrolled / truncated differentiation (Finn et al., MAML; Luketina et al.; Metz et al., unrolled GANs).** Approximate the inner argmin by a single gradient step and differentiate through it — a cheap surrogate for the exact inner solution.

## Baselines

- **NAS with RL (Zoph et al., 2017).** Controller RNN + policy gradient over discrete cell choices; state-of-the-art architectures at a cost of thousands of GPU-days.
- **Evolutionary search (Real et al., 2018, AmoebaNet).** Tournament evolution over architectures; competitive accuracy at comparable cost.
- **ENAS (Pham et al., 2018).** Weight sharing across sampled child architectures cuts search cost while keeping the RL controller over sampled discrete architectures.

## Evaluation settings

- **Datasets / tasks.** CIFAR-10 (image classification, convolutional cells), Penn Treebank (language modeling, recurrent cells), with transfer to ImageNet (mobile setting) and WikiText-2.
- **Search protocol.** Hold out half the CIFAR-10 training set as the validation split used to drive the architecture variable; search a small network (e.g. 8 cells), then build a larger network (e.g. 20 cells) from the discovered cell for final evaluation. Cells located at 1/3 and 2/3 depth are reduction cells (stride two); normal cells preserve resolution.
- **Operation set (conv).** 3×3 and 5×5 separable convolutions, 3×3 and 5×5 dilated separable convolutions, 3×3 max pooling, 3×3 average pooling, identity (skip), and a special *zero* (no connection). ReLU-Conv-BN order; separable convs applied twice.
- **Metrics.** Test error / perplexity of the network built from the searched cell, plus **search cost in GPU-days** (the headline axis NAS is judged on) and parameter count.

## Code framework

The primitives that already exist: an autodiff framework with conv/pool/identity ops and recurrent activations, SGD and Adam optimizers, cross-entropy, a DAG-of-nodes cell abstraction where each intermediate node sums transformed predecessors, normal and reduction cells shared across a stacked network, and a train/validation data split. The scaffold below leaves open how the choice of operation on each edge is represented and driven by the search.

```python
import torch, torch.nn as nn, torch.nn.functional as F

OPS = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
       'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']

class Edge(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList([make_op(name, C, stride) for name in OPS])
    def forward(self, x):
        # TODO: produce this edge's output from the candidate ops
        pass

class Cell(nn.Module):                  # DAG of nodes; each node sums its predecessors' edges
    def __init__(self, steps, C, reduction=False):
        super().__init__()
        self.edges = nn.ModuleList()    # one edge for each predecessor of each intermediate node
        # TODO: instantiate edges with the correct stride pattern
    def forward(self, s0, s1):
        # TODO: sum incoming transformed predecessors at each node
        pass

class SearchNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.normal_cells = nn.ModuleList()
        self.reduction_cells = nn.ModuleList()
        # TODO: represent the per-edge operation choice for normal/reduction cells
    def arch_parameters(self):
        # TODO: return only the architecture-choice variables
        pass
    def weight_parameters(self):
        # TODO: return only ordinary network weights
        pass

def search(model, train_loader, val_loader):
    # TODO: optimize the architecture and the weights
    pass

def derive_discrete_cell(model):
    # TODO: produce the final discrete cell from the searched model
    pass
```
