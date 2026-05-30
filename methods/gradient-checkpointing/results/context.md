# Context: the ground a memory-efficient training method stands on

## Research question

Training a deep neural network by backpropagation requires the backward pass to access intermediate results from the forward pass: most gradient operators depend on the forward activations (feature maps) of the layer they differentiate. The standard implementation therefore *stores every layer's activations* during the forward pass and keeps them alive until the backward pass consumes them. For an `n`-layer network — or a recurrent network unrolled `n` steps — this activation memory grows **linearly in `n`**, `O(n)`. In common convolutional and recurrent architectures the parameters are small relative to these feature maps, so the activations dominate memory.

This is a hard ceiling. Device (GPU) memory caps how deep, how wide, and how long-unrolled a model can be trained; state-of-the-art models already hit the limit. Reducing activation memory would let us train deeper nets, unroll RNNs over longer sequences, use larger batches (improving device utilization and the stability of batchwise operators like batch normalization), and shift from model-parallel to data-parallel training. The precise question: **can the activation memory for training an `n`-layer net be reduced below `O(n)` — ideally far below — while computing the exact same gradients (no approximation) and paying only a small amount of extra computation?**

## Background

Training is naturally expressed on a **computation graph**: nodes are operations, edges are data dependencies. Given the forward graph, the backward (gradient) pathway is built by traversing in reverse topological order and emitting a backward operator for each forward op; the whole gradient computation then becomes a single forward pass over the combined graph. This explicit-graph view (pioneered by Theano, and used by TensorFlow, CNTK, MXNet) is the substrate everything here is built on; the analogy to compiler optimization is direct — allocating memory for graph nodes is like register allocation in a compiler (Aho et al.).

Two memory optimizations are already standard on this graph. **In-place operation**: write an op's output into the memory of one of its inputs, when that input is not needed elsewhere afterward. **Memory sharing**: recycle the memory of an intermediate once it is no longer needed by any pending consumer, handing it to a later node. Deciding which nodes can share requires a *liveness* analysis: two values can share memory only if their lifetimes do not overlap. The exact version is to build a conflict graph (a node per value, an edge between values with overlapping lifespans) and graph-color it, at `O(n²)` cost; a cheaper `O(n)` heuristic traverses the graph in topological order with a per-node counter of how many pending consumers remain, performing in-place when a value's last consumer is running and recycling a value's memory when its counter reaches zero.

Crucially, these optimizations help *prediction* a lot — inference memory can drop from `O(n)` to nearly `O(1)`, because each intermediate dies as soon as the next layer consumes it — but they help *training* only by a constant factor, because the forward intermediates must stay alive until the backward pass reaches them, so training memory remains `O(n)`. The relevant idea from outside deep learning is from the automatic-differentiation literature (Griewank), where deliberately *not storing* some intermediate results and recomputing them in the reverse pass is known as checkpointing; it had been applied in limited special cases but not as a general method for arbitrary deep networks. Other ways to fit big models — swapping tensors to CPU memory, or model-parallel training across devices — are orthogonal (they cost communication bandwidth) and can be combined with a recompute-based method.

## Baselines

**Naive allocation (no optimization).** Allocate fresh memory for every node's output and gradient. Training activation memory `O(n)`, with the largest constant. The reference point.

**In-place + sharing (liveness analysis).** Apply in-place operations and memory sharing via the `O(n)` liveness heuristic. Reduces memory by a constant factor (measured 2–3× on deep ResNets) and shrinks inference to near `O(1)`. Gap: training memory is still `O(n)` — linear in depth — because gradient operators hold the forward feature maps alive; a 2–3× constant does not change the asymptotic ceiling, so very deep nets still run out of memory.

**CPU↔GPU memory swapping / model parallelism.** Move activations to CPU or split the model across devices to fit larger models. Gap: these spend communication bandwidth (PCIe, inter-device) and do not reduce the fundamental amount of activation memory; they are orthogonal to, and combinable with, a recomputation approach.

The recurring gap: every available technique either leaves training memory at `O(n)` (in-place/sharing) or pays it down with communication cost (swapping/parallelism); none trades *computation* for memory to break the linear-in-depth scaling for general networks.

## Evaluation settings

The yardstick is the memory cost of storing intermediate feature maps during training, reported two ways: the *exact* feature-map memory from a static allocation plan (possible when all intermediates are allocated before execution, as in MXNet), and the *runtime total* memory measured with `nvidia-smi`, both as a function of network size (number of layers, or RNN unrolling length), on a single GPU (Titan X). The natural workloads are a deep residual network for ImageNet image classification (counting a conv-bn-relu as one layer, batch size 32, input `(3,224,224)`, depth swept by deepening each residual stage) and an LSTM unrolled over long sequences (four layers, 1024 hidden units, unrolled e.g. 64 steps, batch 64, 50-dim inputs, 5000-class softmax output — a speech-recognition-style setting). Training speed (runtime per fixed number of batches) is the cost side of the trade. All of these are pre-existing benchmarks and instruments.

## Code framework

A deep sequential network and an ordinary training loop, in PyTorch. The model is a list of layers applied in order; the loss is computed on the final output; backward and an optimizer step follow. What is *not* yet decided is how the backward pass obtains the per-layer activations it needs — by default autograd stores all of them. That mechanism is the empty slot.

```python
import torch, torch.nn as nn

class DeepSequential(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)   # n layers applied in sequence

    def forward(self, x):
        # TODO: how does the backward pass get each layer's intermediate
        # activations?  The default is to store them all (O(n) memory).
        for layer in self.layers:
            x = layer(x)
        return x

def train_step(model, batch, target, opt, loss_fn):
    out = model(batch)
    loss = loss_fn(out, target)
    loss.backward()          # needs the forward activations to compute gradients
    opt.step(); opt.zero_grad()
    return loss
```
