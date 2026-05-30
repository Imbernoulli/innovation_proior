# Context

## Research question

Language models have grown to billions of parameters, and at that size the model plus its optimizer state no longer fits in the memory of a single accelerator. Adam alone keeps a momentum and a variance buffer per parameter, roughly tripling the memory the weights demand, and activation checkpointing only partly relieves the pressure. The problem is to train a model whose parameters and optimizer state are too large for one device by splitting them *across* devices — and to do so in a way that is simple enough to adopt: ideally just a few primitives inserted into an existing framework's transformer, with no new compiler, no domain-specific language, and no rewrite of the model, while keeping the accelerators compute-bound rather than stalled on communication. A solution must partition the heavy computation of a transformer layer across $P$ devices, keep the number and size of cross-device communications small, and compose cleanly with the other ways of scaling (splitting the batch across devices, and splitting the layer stack into a pipeline).

## Background

**Why bigger models, and why they no longer fit.** Larger pretrained transformers are dramatically more useful across NLP tasks, so the field keeps scaling parameter count. But as models approach billions of parameters they approach and exceed the memory capacity of a single accelerator — the weights, the activations, and especially the per-parameter optimizer state (Adam's two extra buffers) cannot all reside on one device at once.

**Data parallelism and its hard limit.** The standard way to use many devices is data parallelism: replicate the model on each worker, split the minibatch, and average gradients. Increasing the batch with the number of workers gives near-linear throughput scaling (with large-batch optimization caveats), and combining it with activation checkpointing stretches the trainable size further. But all of this shares one fundamental limitation: *the model must fit on a single worker*. Data parallelism does nothing for a model too big for one device.

**Model parallelism, in two flavours.** To break that limit one must distribute the model itself. There are two paradigms. *Pipeline (layer-wise) parallelism* places different layers on different devices and streams activations down the pipeline; frameworks like GPipe make this work with synchronous gradient descent, but it suffers from pipeline bubbles that waste compute, needs extra logic to schedule communication and computation, and can require optimizer changes. *Distributed tensor computation* is more general: partition an individual tensor operation across devices. Mesh-TensorFlow expresses such partitions in a dedicated language compiled to collective primitives, and FlexFlow searches for good partitionings. These are powerful but heavy — a new language, a compiler, a reworked model.

**The structure of a transformer layer.** A transformer layer is a self-attention block followed by a two-layer MLP, and both are dominated by general matrix multiplies (GEMMs). The MLP is $Z = \text{dropout}(\,B\,\cdot\,\sigma(A\,X)\,)$ where the first GEMM with weight $A$ expands the hidden width (typically by $4\times$), a nonlinearity $\sigma$ (GeLU in these models) is applied, and the second GEMM with weight $B$ projects back. Multi-head attention computes, per head, $\text{softmax}(QK^\top/\sqrt{d})V$ from query/key/value projections of the input, concatenates the heads, and applies an output projection. The heads are independent of one another until the output projection mixes them. The blocks also contain layer normalization, residual additions, and dropout, which are cheap elementwise/normalization operations rather than large GEMMs. (In these models layer normalization is applied to the *inputs* of the attention and feed-forward sub-layers, and GeLU is the nonlinearity.)

**Collective communication primitives.** Distributed training relies on collectives: an *all-reduce* sums (or otherwise reduces) a tensor across all devices and returns the result to each; an *all-gather* concatenates per-device pieces onto every device. All-reduce of a tensor of $m$ elements moves $O(m)$ data per device; communicating large tensors (e.g. logits of size batch $\times$ sequence $\times$ vocabulary) is expensive, while communicating small ones (e.g. scalar losses of size batch $\times$ sequence) is cheap. In an autodiff framework these collectives can be packaged as custom differentiable functions with hand-written forward and backward behaviour.

## Baselines

**Data parallelism (e.g. distributed SGD).** Replicate the model, split the batch, all-reduce gradients. Core idea and math: each worker computes $\nabla L$ on a shard of the batch; gradients are averaged. The gap: the full model and its optimizer state must fit on one worker, so it cannot train a model larger than a single device's memory.

**GPipe-style pipeline parallelism.** Split the layer stack across devices, micro-batch the input, and pass activations along the pipeline with synchronous gradient descent. Core idea: assign contiguous groups of layers to devices. The gaps: pipeline bubbles leave devices idle and reduce efficiency, scheduling the overlap of communication and computation needs extra machinery, and some variants alter the optimizer in ways that affect accuracy.

**Mesh-TensorFlow / general distributed tensor frameworks.** Specify how each tensor dimension is partitioned across a device mesh; a compiler inserts the right collectives. Core idea: a language for distributed tensor algebra. The gap: it requires adopting a new language and compiler and re-expressing the model, rather than making small edits to an existing implementation — heavyweight to adopt and still maturing.

**Parameter sharing to shrink the model.** Tie or reuse weights across layers to cut the memory footprint. Core idea: fewer distinct parameters. The gap: it caps the model's representational capacity, which defeats the purpose of scaling.

## Evaluation settings

The natural yardsticks are of two kinds. For *systems* efficiency: sustained FLOPs and FLOP-utilization on multi-GPU servers, and weak-scaling efficiency as the model and device count grow together (e.g. holding parameters-per-GPU roughly fixed). For *model quality*: language-modeling perplexity on WikiText-103, cloze-style next-word accuracy on LAMBADA, and reading-comprehension accuracy on RACE, for decoder (GPT-2-style) and encoder (BERT-style) transformers trained at a range of sizes. The hardware setting is multi-GPU servers (V100-class accelerators) with high-bandwidth interconnect, where the cost of cross-device collectives relative to local GEMM throughput is what determines whether the partition keeps the devices compute-bound. These benchmarks and metrics predate the method and are the yardstick it would be measured on.

## Code framework

The primitives that already exist: PyTorch linear layers and the GEMMs inside attention and MLP, `torch.distributed` collectives (`all_reduce`, `all_gather`), and `torch.autograd.Function` for defining custom forward/backward. The transformer's MLP and attention blocks need to be re-expressed so their GEMM weights are sharded across the model-parallel devices, bracketed by two small communication primitives that handle synchronization in the forward and backward passes. The slots to fill:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BoundaryInput(torch.autograd.Function):
    # TODO: the primitive placed at the INPUT boundary of a sharded region.
    @staticmethod
    def forward(ctx, x):
        pass
    @staticmethod
    def backward(ctx, grad):
        pass


class _BoundaryOutput(torch.autograd.Function):
    # TODO: the primitive placed at the OUTPUT boundary of a sharded region.
    @staticmethod
    def forward(ctx, x):
        pass
    @staticmethod
    def backward(ctx, grad):
        pass


class ColumnShardedLinear(nn.Module):
    # TODO: a linear layer whose weight is split so each device owns part of the OUTPUT features.
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        pass
    def forward(self, x):
        pass


class RowShardedLinear(nn.Module):
    # TODO: a linear layer whose weight is split so each device owns part of the INPUT features.
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        pass
    def forward(self, x):
        pass


class ParallelMLP(nn.Module):
    # TODO: the two-GEMM MLP block, sharded with minimal communication.
    def __init__(self, hidden, world_size):
        super().__init__()
        pass
    def forward(self, x):
        pass


class ParallelSelfAttention(nn.Module):
    # TODO: multi-head self-attention, sharded with minimal communication.
    def __init__(self, hidden, n_heads, world_size):
        super().__init__()
        pass
    def forward(self, x):
        pass
```
