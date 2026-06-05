# Deep Compression

## Problem

Large CNNs (AlexNet 240 MB, VGG-16 >500 MB) are too big to deploy on mobile devices. Energy is dominated by memory access — a 32-bit DRAM fetch (~640 pJ) costs ~3 orders of magnitude more than a float add (~0.9 pJ) — and big models don't fit in on-chip SRAM. Shrink storage by an order of magnitude or more, with **no loss of accuracy**, so the model fits in cache.

## Key idea

A three-stage pipeline whose stages compress along independent axes and stack multiplicatively:

1. **Pruning** — fewer connections.
2. **Trained quantization / weight sharing** — fewer bits per connection.
3. **Huffman coding** — lossless entropy coding of the resulting biased symbol streams.

Pruning and quantization do not interfere (pruning leaves fewer weights per centroid, making quantization if anything *easier*), so the rates multiply: ~9–13× from pruning, ~27–31× through quantization, ~35–49× after Huffman.

## Method

**Stage 1 — Pruning.** Train to convergence; remove all connections with $|w|$ below a threshold; retrain the surviving sparse weights (masked update keeps pruned weights at 0). Store the sparse matrix in CSR/CSC ($2a+n+1$ numbers), using **relative index differences** (5 bits FC, 8 bits conv) and a padded filler zero whenever a gap exceeds the index range.

**Stage 2 — Trained quantization / weight sharing.** Per layer, cluster the weights into $k$ shared values by 1-D k-means, minimizing within-cluster sum of squares $\arg\min_C \sum_{i=1}^k\sum_{w\in c_i}(w-c_i)^2$, **after** training (so the codebook matches the learned distribution). Store one $\log_2 k$-bit index per connection plus the $k$-entry codebook; compression rate
$$r = \frac{nb}{n\log_2(k) + kb}.$$
Use **linear centroid initialization** (uniform over $[\min,\max]$) so the rare but important large-magnitude weights keep a nearby centroid. Fine-tune the shared values; the gradient of a centroid is the sum of the per-connection gradients sharing it:
$$\frac{\partial\mathcal{L}}{\partial C_k} = \sum_{i,j}\frac{\partial\mathcal{L}}{\partial W_{ij}}\,\mathbb{1}(I_{ij}=k).$$
Conv layers get more bits (~8), FC layers fewer (~5).

**Stage 3 — Huffman coding.** The quantized weights (piled around the distribution's peaks) and the relative index differences (rarely large) are biased; a Huffman code (optimal prefix code) packs them losslessly for ~20–30% extra savings. Offline, no training.

## Code

```python
import numpy as np
from sklearn.cluster import KMeans

# Stage 1: magnitude pruning
def prune_stage(net, data, thresholds):
    net = train_to_convergence(net, data)
    masks = {}
    for name, layer in net.layers.items():
        masks[name] = np.abs(layer.W) > thresholds[name]
        layer.W *= masks[name]
    for _ in range(N_RETRAIN):                      # retrain survivors, pruned held at 0
        backprop_step(net, data)
        for name, layer in net.layers.items():
            layer.W *= masks[name]
    return net, masks

# Stage 2: trained quantization / weight sharing
def quantize_layer(W, mask, k):
    w = W[mask].reshape(-1, 1)
    init = np.linspace(w.min(), w.max(), k).reshape(-1, 1)  # linear init
    km = KMeans(n_clusters=k, init=init, n_init=1).fit(w)
    idx = np.zeros_like(W, dtype=np.int32); idx[mask] = km.labels_
    return km.cluster_centers_.reshape(-1), idx             # codebook, indices

def finetune_centroids(net, data, C, I, M, lr, steps):
    for _ in range(steps):
        g = backprop_grads(net, data)               # dL/dW per connection
        for name, layer in net.layers.items():
            gC = np.zeros_like(C[name])
            np.add.at(gC, I[name][M[name]], g[name][M[name]])   # sum grads per centroid
            C[name] -= lr * gC
            layer.W = C[name][I[name]] * M[name]     # write shared values back
    return C, I

# Stage 3: Huffman (lossless, offline)
def entropy_code(C, I):
    return {n: (huffman_encode(quantize_symbols(C[n], I[n])),
                huffman_encode(relative_index_diffs(I[n]))) for n in C}

def compress(net, data, thresholds, bits):
    net, M = prune_stage(net, data, thresholds)
    C, I = {}, {}
    for name, layer in net.layers.items():
        k = 2 ** bits[name]                          # 8 conv / 5 FC
        C[name], I[name] = quantize_layer(layer.W, M[name], k)
    C, I = finetune_centroids(net, data, C, I, M, lr=LR, steps=N)
    return entropy_code(C, I)
```
