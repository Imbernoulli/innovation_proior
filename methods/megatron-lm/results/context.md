# Context

## Research question

Language models have grown to billions of parameters, and at that size the model plus its optimizer state no longer fits in the memory of a single accelerator. Adam alone keeps a momentum and a variance buffer per parameter, roughly tripling the memory the weights demand, and activation checkpointing only partly relieves the pressure. How can a model whose parameters and optimizer state are too large for one device be trained by splitting them across devices, using the existing transformer architecture and training framework?

## Background

**Why bigger models, and why they no longer fit.** Larger pretrained transformers are dramatically more useful across NLP tasks, so the field keeps scaling parameter count. As models approach billions of parameters they approach and exceed the memory capacity of a single accelerator — the weights, the activations, and especially the per-parameter optimizer state (Adam's two extra buffers) cannot all reside on one device at once.

**Data parallelism and its hard limit.** The standard way to use many devices is data parallelism: replicate the model on each worker, split the minibatch, and average gradients. Increasing the batch with the number of workers gives near-linear throughput scaling (with large-batch optimization caveats), and combining it with activation checkpointing stretches the trainable size further. This requires the full model and its optimizer state to fit on one worker.

**Model parallelism, in two flavours.** To distribute the model itself there are two paradigms. *Pipeline (layer-wise) parallelism* places different layers on different devices and streams activations down the pipeline; frameworks like GPipe make this work with synchronous gradient descent. *Distributed tensor computation* is more general: partition an individual tensor operation across devices. Mesh-TensorFlow expresses such partitions in a dedicated language compiled to collective primitives, and FlexFlow searches for good partitionings.

**The structure of a transformer layer.** A transformer layer is a self-attention block followed by a two-layer MLP, and both are dominated by general matrix multiplies (GEMMs). The MLP is $Z = \text{dropout}(\,B\,\cdot\,\sigma(A\,X)\,)$ where the first GEMM with weight $A$ expands the hidden width (typically by $4\times$), a nonlinearity $\sigma$ (GeLU in these models) is applied, and the second GEMM with weight $B$ projects back. Multi-head attention computes, per head, $\text{softmax}(QK^\top/\sqrt{d})V$ from query/key/value projections of the input, concatenates the heads, and applies an output projection. The heads are independent of one another until the output projection mixes them. The blocks also contain layer normalization, residual additions, and dropout, which are cheap elementwise/normalization operations rather than large GEMMs. (In these models layer normalization is applied to the *inputs* of the attention and feed-forward sub-layers, and GeLU is the nonlinearity.)

**Collective communication primitives.** Distributed training relies on collectives: an *all-reduce* sums (or otherwise reduces) a tensor across all devices and returns the result to each; an *all-gather* concatenates per-device pieces onto every device. All-reduce of a tensor of $m$ elements moves $O(m)$ data per device; communicating large tensors (e.g. logits of size batch $\times$ sequence $\times$ vocabulary) is expensive, while communicating small ones (e.g. scalar losses of size batch $\times$ sequence) is cheap. In an autodiff framework these collectives can be packaged as custom differentiable functions with hand-written forward and backward behaviour.

## Baselines

**Data parallelism (e.g. distributed SGD).** Replicate the model, split the batch, all-reduce gradients. Each worker computes $\nabla L$ on a shard of the batch; gradients are averaged.

**GPipe-style pipeline parallelism.** Split the layer stack across devices, micro-batch the input, and pass activations along the pipeline with synchronous gradient descent. Contiguous groups of layers are assigned to devices.

**Mesh-TensorFlow / general distributed tensor frameworks.** Specify how each tensor dimension is partitioned across a device mesh; a compiler inserts the right collectives. Provides a language for distributed tensor algebra.

**Parameter sharing to shrink the model.** Tie or reuse weights across layers to cut the memory footprint, reducing the number of distinct parameters.

## Evaluation settings

The natural yardsticks are of two kinds. For *systems* efficiency: sustained FLOPs and FLOP-utilization on multi-GPU servers, and weak-scaling efficiency as the model and device count grow together (e.g. holding parameters-per-GPU roughly fixed). For *model quality*: language-modeling perplexity on WikiText-103, cloze-style next-word accuracy on LAMBADA, and reading-comprehension accuracy on RACE, for decoder (GPT-2-style) and encoder (BERT-style) transformers trained at a range of sizes. The hardware setting is multi-GPU servers (V100-class accelerators) with high-bandwidth interconnect, where the cost of cross-device collectives relative to local GEMM throughput determines whether the partition keeps the devices compute-bound. These benchmarks and metrics predate the method and are the yardstick it would be measured on.

## Code framework

The primitives that already exist: PyTorch linear layers and the GEMMs inside attention and MLP, `torch.distributed` collectives (`all_reduce`, `all_gather`), and `torch.autograd.Function` for defining custom forward/backward. The task is to re-express the transformer's MLP and attention blocks so their GEMM weights are sharded across the model-parallel devices, keeping the cross-device communication small. The slot to fill:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelTransformerLayer(nn.Module):
    # TODO: a transformer layer (self-attention + two-GEMM MLP) whose heavy GEMMs
    # are sharded across `world_size` model-parallel devices.
    def __init__(self, hidden, n_heads, world_size):
        super().__init__()
        pass
    def forward(self, x):
        pass
```
