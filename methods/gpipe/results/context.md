## Research question

Across vision and language, the pressure is toward larger models: image classifiers had grown sharply alongside accuracy improvements, and deeper/larger language models had displaced simpler shallow sentence representations. The natural move is to keep scaling. But once a model no longer fits in a single accelerator's memory, scaling becomes a systems problem: the model must be split across accelerators.

The setting: **how do we train a very large neural network — one that does not fit on a single accelerator — by partitioning it across devices?** We want a general way to take a model defined as a sequence of layers, hand it more accelerators, and run training across them.

## Background

**The capacity trend.** Empirically, model quality had been improving as capacity grew in both domains, so the demand is to train models far larger than one device can hold. Memory and inter-device bandwidth on GPUs/TPUs are the binding constraints.

**Where training memory goes.** During backpropagation, computing a layer's gradient needs both the gradient flowing back from the layer above *and* that layer's cached forward activations. So a naive forward over an L-layer network with a mini-batch of N caches activations for every layer — memory grows like O(N × L).

**Re-materialization (activation checkpointing).** Instead of caching every layer's activations, cache only a sparse set (e.g. partition boundaries) during the forward; in the backward, recompute the dropped activations on demand from the nearest cached point. This trades extra compute (one extra forward over the recomputed region) for reduced activation memory. It is the standard memory-for-compute lever.

**Synchronous mini-batch optimization.** The reference semantics: accumulate gradients over a whole mini-batch, then apply one update. Schemes that change this (e.g. asynchronous or weight-stale updates) change what the user is optimizing and what convergence they get.

**Sequential dependency in a feed-forward chain.** A feed-forward network is a chain: layer i+1 cannot start until layer i finishes; the backward of layer i cannot start until the backward of layer i+1 delivers its incoming gradient. So if consecutive layer groups are placed on consecutive devices and one mini-batch is pushed through, one device is active at a time while the others wait on the data dependency.

## Baselines

- **Single-accelerator training.** The reference for correctness and for the optimization trajectory. The model plus its activations fit in one device's memory; activation memory is O(N × L).

- **Naive model parallelism (sequential layer placement).** Partition the L layers into K consecutive layer groups, one per accelerator; communicate the activation tensor at each group boundary. Memory is divided across devices, and boundary communication is light (only the activations crossing partition lines move). Because of the chain dependency, at any moment one accelerator computes while the others wait.

- **Tensor / horizontal model parallelism (e.g. Mesh-TensorFlow-style).** Split individual operations (matmuls, embeddings) across devices. Requires the model to be expressed in a way that exposes the parallel structure, with per-operator distributed implementations.

- **Asynchronous / weight-stale pipelining.** Keeps devices busy by letting different micro-batches use slightly different weight versions. The update depends on the pipeline depth.

## Evaluation settings

- **Architectures.** A convolutional image model (AmoebaNet, scaled by width/filters and number of cells) and Transformer sequence-to-sequence models (scaled mainly by depth L, with feed-forward hidden dimension and attention-head count used as width knobs in translation).
- **Tasks / data.** ImageNet 2012 image classification (with transfer to CIFAR-10/100, Stanford Cars, Oxford Pets, Food-101, FGVC Aircraft, Birdsnap); large-scale multilingual machine translation over 102 languages plus English (25 billion training examples, spanning low- to high-resource languages), with per-language BLEU against bilingual baselines.
- **Hardware.** Cloud TPUv2 accelerators with 8 GB and Cloud TPUv3 cores with 16 GB; also multi-GPU hosts (NVIDIA P100 without NVLink) to stress communication when only slow PCIe device-to-host transfers are available.
- **Metrics / protocol.** Maximum trainable model size per device count; normalized training throughput as a function of number of partitions K and number of micro-batches M; communication-overhead sensitivity (interconnect vs. not); fixed input sizes (e.g. 224×224 images; sequence length 1024, vocabulary 32k, batch 32 for Transformers); each parameter costs 12 bytes under the optimizer used (RMSProp).

## Code framework

The pre-existing pieces: a framework that represents a network as an ordered list of layers, each with a forward function and parameters; automatic differentiation that can build a backward function from any forward; communication primitives between neighboring accelerators; and a synchronous mini-batch training loop. The scaffold is a generic layer-sequence harness with empty slots for the parts that turn this sequence into a multi-device training procedure.

```python
from typing import Callable, List

class Layer:
    def __init__(self, forward: Callable, params, cost_fn: Callable = None):
        self.f = forward          # f_i
        self.w = params           # w_i
        self.cost = cost_fn       # optional c_i: estimated compute cost

def autodiff_backward(forward_fn):
    """Build a backward function for a forward computation."""
    ...

class LayerSequenceTrainer:
    def __init__(self, layers: List[Layer], devices: List[object], schedule_config=None):
        self.layers = layers
        self.devices = devices
        self.schedule_config = schedule_config
        self.layer_groups = self.build_layer_groups()

    def build_layer_groups(self):
        # TODO
        ...

    def run_forward(self, mini_batch):
        # TODO
        ...

    def run_backward(self, loss):
        # TODO
        ...

    def train_step(self, mini_batch, optimizer):
        # one optimizer update after accumulating the full mini-batch gradient
        out = self.run_forward(mini_batch)
        loss = compute_loss(out)
        grads = self.run_backward(loss)
        optimizer.apply(grads)   # apply once after the mini-batch gradient is accumulated
```
