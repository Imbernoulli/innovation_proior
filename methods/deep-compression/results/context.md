# Context

The setting is deploying large convolutional networks (AlexNet, VGG-16) for inference on resource-constrained devices around 2015. These networks are accurate but enormous: the AlexNet Caffe model is over 200 MB, VGG-16 over 500 MB. The weights are stored as 32-bit floats and there are tens to hundreds of millions of them. This is a problem for mobile deployment (app-store size limits, download bandwidth) and, more fundamentally, for energy.

## Research question

How can the *storage* required by a trained network be reduced by an order of magnitude or more — small enough to fit in on-chip SRAM rather than off-chip DRAM — **without any loss of accuracy**? The energy argument makes the target concrete: under 45 nm CMOS, a 32-bit float add costs ~0.9 pJ, a 32-bit SRAM access ~5 pJ, but a 32-bit DRAM access ~640 pJ — three orders of magnitude more than the arithmetic. Energy is dominated by memory access, and large models don't fit on chip, forcing the expensive DRAM fetches. A billion-connection net at 20 fps would burn $20\,\text{Hz}\times 10^9 \times 640\,\text{pJ} = 12.8\,\text{W}$ on DRAM access alone — beyond a mobile power budget. So the real goal is: shrink the model enough to live in cache, preserving accuracy exactly. Where this matters most is latency-bound, batch-size-1 inference (real-time, e.g. on-device detection), where there is no batching to amortize weight fetches.

## Background

- **Networks are massively over-parameterized.** There is large redundancy in deep models (Denil et al. 2013 showed most parameters are predictable from a few), so in principle most weights can be removed or coarsened without hurting accuracy. The fully-connected layers dominate model size (>90% in VGG-16) and are the most redundant; convolutional layers are smaller but more sensitive to precision.
- **Pruning has a long history.** Biased weight decay (Hanson & Pratt 1989); Optimal Brain Damage (LeCun et al. 1990) and Optimal Brain Surgeon (Hassibi & Stork 1993) delete connections using second-order (Hessian) saliency, and argue this is more accurate than magnitude-based deletion. Recently, Han et al. (2015) showed that simple iterative *magnitude* pruning — train, delete the smallest-magnitude weights, retrain the survivors — removes an order of magnitude of parameters from AlexNet/VGG with no accuracy loss. Larger-magnitude weights matter more than smaller ones.
- **Quantization / weight sharing.** Reducing the bits per weight: fixed-point 8-bit (Vanhoucke et al. 2011), ternary weights with 3-bit activations (Hwang & Sung 2014), L2-error-minimizing quantization (Anwar et al. 2015). HashedNets (Chen et al. 2015) force weights into shared buckets via a hash function — but the sharing is fixed *before* the network sees data, so it cannot adapt to the trained weight distribution. Vector quantization of conv-nets (Gong et al. 2014) gives ~16–24× with ~1% accuracy loss but studies only FC layers.
- **Low-rank / structural.** SVD / low-rank factorization of weight matrices (Denton et al. 2014) keeps accuracy within ~1% but compresses only modestly (a few ×). Replacing FC layers with global average pooling (NiN, GoogLeNet) shrinks the net but hurts transfer learning.
- **Sparse storage facts.** A pruned matrix stored in compressed sparse row/column (CSR/CSC) format needs $2a+n+1$ numbers for $a$ nonzeros and $n$ rows/columns — the index overhead is real and must be counted in any honest compression rate. A Huffman code (Huffman 1952) is an optimal prefix code: frequent symbols get shorter codewords, so biased symbol distributions compress further losslessly.

## Baselines

- **Magnitude pruning alone** (Han et al. 2015): delete sub-threshold-magnitude weights, retrain. Gives ~9–13× fewer parameters with no accuracy loss, but each surviving weight is still a 32-bit float and the sparse indices add overhead. Accuracy collapses if pushed much below ~8% remaining weights.
- **Scalar/vector quantization alone:** reduce bits per weight, but on the *full* dense network; accuracy also collapses below ~8% of original size, and HashedNets-style pre-data hashing can't match the trained distribution.
- **Low-rank (SVD):** cheap but poor compression (~2–5×) and noticeable accuracy loss.

The gap: each technique alone tops out around 8% of original size before accuracy drops, and none combines a learned sparse structure with a learned low-bit codebook while accounting for index/codebook overhead.

## Evaluation settings

- **MNIST:** LeNet-300-100 (FC, ~266K params, ~1.6% error) and LeNet-5 (conv, ~431K params, ~0.8% error).
- **ImageNet ILSVRC-2012:** 1.2M train / 50K val images. AlexNet (61M params; top-1 ~57.2%, top-5 ~80.3%) and VGG-16 (138M params), both from the Caffe model zoo.
- **Framework:** Caffe. Pruning realized by masking blob updates; quantization by a per-layer codebook plus indices.
- **Metrics:** top-1 / top-5 error, parameter count, and **compression rate including index and codebook overhead**. Hardware benchmarks (CPU Core i7-5930K, GPU Titan X, mobile Tegra K1) report layerwise latency and energy at **batch size 1**, using dense GEMV vs sparse CSR matrix-vector kernels.

## Code framework

Pre-method scaffold: a trained network and the slots for the compression stages.

```python
import numpy as np

class Layer:
    def __init__(self, W):
        self.W = W                  # dense float32 weight matrix, already trained

def train_to_convergence(net, data):
    ...                             # ordinary training exists already
    return net

def finetune(net, data, steps):
    ...                             # ordinary fine-tuning loop exists already
    return net

def prune_stage(net, data):
    # TODO: learn which connections matter, remove the rest, refit survivors;
    #       store the result as a sparse structure (with its index overhead).
    raise NotImplementedError

def quantize_stage(net, data):
    # TODO: make many connections share a small set of values, store indices
    #       into a per-layer codebook, and refit the shared values.
    raise NotImplementedError

def entropy_code(net):
    # TODO: losslessly pack the shared values and indices given their
    #       (biased) frequency distribution.
    raise NotImplementedError

def compress(net, data):
    net = train_to_convergence(net, data)
    net = prune_stage(net, data)    # TODO
    net = quantize_stage(net, data) # TODO
    net = entropy_code(net)         # TODO
    return net
```
