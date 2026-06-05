# Context

## Research question

Across vision and language, bigger neural networks keep delivering better accuracy — top-1 on ImageNet rises with model capacity, and in machine translation deeper/wider models beat shallow ones. So the natural move is to keep scaling. But a model that is large enough to be interesting no longer fits in a single accelerator's memory, and that single fact forces the model to be split across multiple accelerators.

The precise problem: **how do we train a very large neural network — one that does not fit on a single accelerator — by partitioning it across devices, in a way that (a) works for essentially any architecture, (b) keeps almost all accelerators busy almost all the time, and (c) does not change the optimization the user would have run on one device?** Existing model-parallel schemes were architecture- and task-specific, hard to design, and forced practitioners to trade away flexibility, capacity, or efficiency. We want a *general* tool: define the model as a sequence of layers, hand it more accelerators, and have it scale — without bespoke distributed-operator engineering per model and without altering the gradient updates.

## Background

**The capacity–accuracy trend.** Empirically, model quality scales with capacity in both domains, so the demand is to train models far larger than one device can hold. Memory and inter-device bandwidth on GPUs/TPUs are the binding constraints.

**Where training memory goes.** During backpropagation, computing a layer's gradient needs both the gradient flowing back from the layer above *and* that layer's cached forward activations. So a naive forward over an L-layer network with a mini-batch of N caches activations for every layer — memory grows like O(N × L). For large L this is what overflows the device, often before the parameters themselves do.

**Re-materialization (activation checkpointing).** Instead of caching every layer's activations, cache only a sparse set (e.g. partition boundaries) during the forward; in the backward, recompute the dropped activations on demand from the nearest cached point. This trades extra compute (one extra forward over the recomputed region) for a large drop in activation memory. It is the standard memory-for-compute lever.

**Synchronous mini-batch SGD.** The reference semantics we want to preserve: accumulate gradients over a whole mini-batch, then apply one update. Any parallelization scheme that changes this (e.g. asynchronous or weight-stale updates) changes what the user is optimizing and what convergence they get. A scheme that keeps the update *identical regardless of how many devices are used* is far easier to trust and to reason about.

**Sequential dependency is the obstacle to naive splitting.** A feed-forward network is a chain: layer i+1 cannot start until layer i finishes; the backward of layer i cannot start until the backward of layer i+1 delivers its incoming gradient. So if we simply place consecutive layer-groups (cells) on consecutive devices and push one mini-batch through, only one device is ever active at a time — every other device sits idle waiting for the data dependency. The split "works" for memory but wastes (K−1)/K of the compute on K devices.

## Baselines

- **Single-accelerator training.** The reference for correctness and for the optimization trajectory. Hard cap: the model plus its activations must fit in one device's memory; activation memory O(N × L) is usually the first wall.

- **Naive model parallelism (sequential layer placement).** Partition the L layers into K consecutive cells, one per accelerator; communicate the activation tensor at each cell boundary. Memory is divided across devices, and boundary communication is light (only the activations crossing partition lines move). But because of the chain dependency, at any moment exactly one accelerator computes while the others idle — utilization collapses to ~1/K. This is the gap a good scheme must close: keep the partitioning, kill the idleness.

- **Tensor / horizontal model parallelism (e.g. Mesh-TensorFlow-style).** Split individual operations (matmuls, embeddings) across devices. Can scale specific architectures well but requires the model to be expressed in a way that exposes the parallel structure and requires per-operator distributed implementations — architecture-specific and effortful, exactly the inflexibility we want to avoid.

- **Asynchronous / weight-stale pipelining.** Keeps devices busy by letting different micro-batches use slightly different weight versions. Improves utilization but breaks synchronous-SGD semantics — the update now depends on the pipeline depth, complicating convergence reasoning.

## Evaluation settings

- **Architectures.** A convolutional image model (AmoebaNet, scaled by width/filters and number of cells) and a Transformer sequence-to-sequence model (scaled by depth L, with model dimension and feed-forward hidden dimension and number of attention heads as width knobs).
- **Tasks / data.** ImageNet 2012 image classification (with transfer to CIFAR-10/100, Stanford Cars, Oxford Pets, Food-101, FGVC Aircraft, Birdsnap); large-scale multilingual machine translation over 102 languages ↔ English (tens of billions of training examples, spanning low- to high-resource languages), with per-language BLEU against bilingual baselines.
- **Hardware.** Cloud TPUv2 (8 GB/core) and TPUv3 (16 GB/core); also multi-GPU hosts (NVIDIA P100, with and without NVLink) to stress communication when only slow PCIe device-to-host transfers are available.
- **Metrics / protocol.** Maximum trainable model size per device count; normalized training throughput as a function of number of partitions K and number of micro-batches M; communication-overhead sensitivity (interconnect vs. not); fixed input sizes (e.g. 224×224 images; sequence length 1024, vocabulary 32k, batch 32 for Transformers); each parameter costs 12 bytes under the optimizer used (RMSProp).

## Code framework

The pre-existing pieces: a framework that represents a network as an ordered list of layers, each with a forward function and parameters; automatic differentiation that can build a backward function from any forward; standard collective/point-to-point communication between accelerators; and a synchronous SGD training loop. The scaffold is a generic layer-sequence harness with empty slots for *how* to place the layers across devices and *how* to schedule the forward/backward so devices are not idle.

```python
from typing import Callable, List

class Layer:
    def __init__(self, forward: Callable, params, cost_fn: Callable = None):
        self.f = forward          # f_i
        self.w = params           # w_i
        self.cost = cost_fn       # optional c_i: estimated compute cost

def autodiff_backward(forward_fn):
    """Build the backward function B for a (possibly composite) forward F."""
    ...

# --- the slot the contribution fills ---
class ModelParallelEngine:
    def __init__(self, layers: List[Layer], num_devices: int):
        self.layers = layers
        self.K = num_devices
        # TODO: group the L layers into K cells and place cell k on device k.
        #       How to choose the grouping so the devices are balanced?
        self.cells = self._partition_into_cells(layers, num_devices)

    def _partition_into_cells(self, layers, K):
        # TODO: split the layer sequence into K consecutive cells.
        ...

    def forward(self, mini_batch):
        # TODO: a chain of cells across devices has a sequential dependency --
        #       running one mini-batch leaves K-1 devices idle. Schedule the
        #       computation so devices are NOT idle, without changing the result.
        ...

    def backward(self, grad_out):
        # TODO: backward has the reverse sequential dependency; and caching every
        #       layer's activations costs O(N*L) memory. Reduce that cost.
        ...

    def train_step(self, mini_batch, optimizer):
        # synchronous mini-batch SGD: one update per mini-batch, and the update
        # MUST be identical regardless of self.K.
        out = self.forward(mini_batch)
        loss = out
        grads = self.backward(loss)
        optimizer.apply(grads)   # accumulate over the mini-batch, then step once
```
