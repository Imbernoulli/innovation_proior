## Research question

Large convolutional networks are accurate but hard to deploy on embedded and mobile systems because their parameters dominate storage and memory bandwidth. AlexNet Caffe models are over 200 MB, and VGG-16 Caffe models are over 500 MB when weights are stored as 32-bit floats. Under 45 nm CMOS, a 32-bit floating-point add costs about 0.9 pJ, a 32-bit SRAM access about 5 pJ, and a 32-bit DRAM access about 640 pJ. A one-billion-connection network running at 20 frames per second would spend $(20\text{Hz})(10^9)(640\text{pJ})=12.8\text{W}$ on DRAM weight fetches alone.

The working question is how to reduce the storage of a trained network by an order of magnitude or more, enough that weights can sit in on-chip cache rather than off-chip DRAM, while preserving the original accuracy. The constraint is especially sharp for latency-bound batch-size-1 inference, where dense matrix-vector layers cannot reuse weights across a batch.

## Background

- **Networks are massively over-parameterized.** Deep models contain substantial parameter redundancy; Denil et al. showed that many parameters can be predicted from a small subset. In VGG-16, fully connected layers account for more than 90% of the model size, while convolutional layers are smaller and more precision-sensitive.
- **Pruning has a long history.** Biased weight decay, Optimal Brain Damage, and Optimal Brain Surgeon all remove connections to reduce network complexity; the Hessian-based methods use second-order saliency to decide which connections to delete. The more recent large-CNN pruning recipe is to train normally, delete small-magnitude weights, and retrain the surviving sparse connections.
- **Quantization and weight sharing reduce bits per connection.** Fixed-point activations, ternary weights with low-bit activations, L2-error-minimizing quantization, hashing into shared buckets, and vector quantization all attack the number of stored bits. Hashed sharing fixes the buckets a weight maps to before training. Vector quantization has mainly been applied to fully connected layers.
- **Low-rank and structural changes attack particular layers.** SVD-style low-rank factorization replaces a layer's weights with a lower-rank approximation. Replacing fully connected layers with global average pooling shrinks the model and is used in transfer workflows that reuse ImageNet features and retrain only the fully connected head.
- **Sparse storage carries metadata.** A pruned matrix stored in compressed sparse row or compressed sparse column format needs values, one index per value, and row or column pointers: $2a+n+1$ stored numbers for $a$ nonzeros and $n$ rows or columns before bit packing. A compression-rate calculation counts those indices and pointers, not just the surviving weight values. If relative indexes are packed into a bounded bit width, a rare long jump is split with zero-valued filler entries; each filler entry consumes both an index delta and a value symbol, and the chosen index width is itself a counted storage parameter.

## Baselines

- **Magnitude pruning alone:** delete weights below a threshold and retrain the survivors. This removes roughly an order of magnitude of parameters from AlexNet/VGG-style models; survivors are stored as full-precision floats with sparse indices.
- **Scalar or vector quantization alone:** reduce the bits per weight on a dense network using fixed-point or codebook representations.
- **Low-rank factorization:** replace a weight matrix or convolutional kernel tensor with a lower-rank approximation.

Each baseline is reported in isolation. A reported compression rate charges for every stored byte, including sparse-index, filler, pointer, and any auxiliary-table overhead.

## Evaluation settings

- **MNIST:** LeNet-300-100 with about 266K weights and LeNet-5 with about 431K weights.
- **ImageNet ILSVRC-2012:** 1.2M training images and 50K validation images; AlexNet with about 61M parameters and VGG-16 with about 138M parameters, both from the Caffe model zoo.
- **Framework:** Caffe. Ordinary training and fine-tuning are available; sparse updates can be enforced by masking pruned-weight blob updates.
- **Metrics:** top-1/top-5 error, parameter storage, and compression rate, where the rate charges for every stored byte including any sparse-index, pointer, and auxiliary-table overhead, plus batch-size-1 layerwise latency/energy on CPU, desktop GPU, and mobile GPU. Dense fully connected layers are benchmarked as GEMV; pruned sparse layers are benchmarked as CSR sparse matrix-vector products.

## Code framework

A trained network and ordinary training loop are available; the storage-reduction procedure is the open slot.

```python
import numpy as np

class Layer:
    def __init__(self, W):
        self.W = W.astype(np.float32)

def train_to_convergence(net, data):
    ...
    return net

def finetune(net, data, steps):
    ...
    return net

def compress(net, data):
    net = train_to_convergence(net, data)
    # TODO: reduce the stored size of the trained network while preserving
    # accuracy, honestly counting any sparse-index, pointer, and table overhead.
    raise NotImplementedError
```
