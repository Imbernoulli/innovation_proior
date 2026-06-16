Deep Compression reduces CNN model storage with a three-stage pipeline: magnitude pruning plus retraining, trained quantization with weight sharing, and lossless Huffman coding. The stages attack different storage terms: pruning lowers the number of live connections, weight sharing lowers the bits per live connection, and Huffman coding removes remaining entropy waste in the centroid-ID and sparse-index streams.

Stage 1 trains the dense network, removes connections whose absolute weight is below a layer threshold, and retrains the surviving sparse weights while masked updates keep pruned weights at zero. The sparse matrix is stored in CSR or CSC form, so the storage is not just the live values: for $a$ nonzeros and $n$ rows or columns, the uncompressed sparse representation accounts for values, indices, and pointers as $2a+n+1$ stored numbers. Bit-packed storage uses relative index differences, with a counted per-layer index width and filler zero entries when a gap exceeds the available range.

Stage 2 clusters each layer's surviving weights into $k$ shared values by one-dimensional k-means after the network has already been trained. For dense weight sharing alone, $n$ original $b$-bit weights become $n\log_2(k)$ bits of centroid IDs plus $kb$ bits of codebook:
$$
r_{\text{share}}=\frac{nb}{n\log_2(k)+kb}.
$$
For the pruned sparse layer before Huffman coding, the denominator must include the sparse pattern too:
$$
r_{\text{sparse+share}}=
\frac{mb}{(a+f)(q+d)+kb+p(n+1)},
$$
where $m$ is the original dense weight count, $a$ is the live nonzero count, $f$ is any zero-valued filler count needed for relative-index overflow, $q$ is the stored value-symbol bit width after making room for any filler-zero representation, $d$ is the relative-index bit width, and $p(n+1)$ is the row/column-pointer storage. With no overflow filler and a pure $k$-centroid stream, $q=\log_2(k)$; if overflow padding is represented as an extra zero symbol or a reserved fixed-width value code, that representation is counted through $q$ and through the $(a+f)$ stored entries, while the codebook remains $kb$. The 4-by-4 example with $16$ 32-bit weights and $k=4$ shared values gives $16\cdot32/(16\cdot2+4\cdot32)=3.2$ before sparse-index costs.

The k-means centroids are initialized linearly over the layer's $[\min,\max]$ weight range, rather than by random samples or density-spaced quantiles, so rare large-magnitude weights still get nearby centroids. During fine-tuning, assignments stay fixed and each centroid receives the summed gradient of every connection assigned to it:
$$
\frac{\partial\mathcal L}{\partial C_k}
=\sum_{i,j}\frac{\partial\mathcal L}{\partial W_{ij}}\mathbf 1(I_{ij}=k).
$$
Convolutional layers use more centroid bits than fully connected layers because they are more precision-sensitive; the ImageNet setting uses 8-bit shared weights for convolutional layers and 5-bit shared weights for fully connected layers.

Stage 3 applies Huffman coding after all pruning, clustering, and centroid fine-tuning are finished. The value-symbol stream contains centroid IDs for the stored live values plus any filler-zero symbols needed by relative-index overflow; the sparse-position stream contains the relative deltas. Huffman coding is lossless and is applied to those biased streams, with filler entries already included in the encoded symbol counts, while the codebook entries and sparse row/column pointers remain explicit overhead. The final compression rate is the original dense float storage divided by the Huffman-coded value stream, the Huffman-coded index stream, the codebooks, and the pointers.

```python
import numpy as np
from sklearn.cluster import KMeans

ZERO_SYMBOL = -1

def prune_stage(net, data, thresholds, retrain_steps):
    net = train_to_convergence(net, data)
    masks = {}
    for name, layer in net.layers.items():
        mask = np.abs(layer.W) > thresholds[name]
        layer.W = np.where(mask, layer.W, 0.0)
        masks[name] = mask
    for _ in range(retrain_steps):
        backprop_step(net, data)
        for name, layer in net.layers.items():
            layer.W *= masks[name]
    return net, masks

def quantize_layer(W, mask, k):
    live = W[mask].reshape(-1, 1)
    init = np.linspace(live.min(), live.max(), k).reshape(-1, 1)
    km = KMeans(n_clusters=k, init=init, n_init=1).fit(live)
    codebook = km.cluster_centers_.reshape(-1)
    indices = np.zeros(W.shape, dtype=np.int32)
    indices[mask] = km.labels_
    return codebook, indices

def write_shared_weights(net, codebooks, indices, masks):
    for name, layer in net.layers.items():
        layer.W = codebooks[name][indices[name]] * masks[name]

def finetune_centroids(net, data, codebooks, indices, masks, lr, steps):
    write_shared_weights(net, codebooks, indices, masks)
    for _ in range(steps):
        grads = backprop_grads(net, data)
        for name, layer in net.layers.items():
            C, I, M = codebooks[name], indices[name], masks[name]
            centroid_grads = np.zeros_like(C)
            np.add.at(centroid_grads, I[M], grads[name][M])
            C -= lr * centroid_grads
        write_shared_weights(net, codebooks, indices, masks)
    return codebooks, indices

def csr_symbol_streams(mask, centroid_ids, index_bits):
    # Use the bounded relative-index convention: a d-bit field stores jumps up
    # to 2**d, and filler zero symbols split longer jumps.
    max_stored_delta = 1 << index_bits
    weight_symbols, index_deltas, row_ptr = [], [], [0]
    filler_count = 0
    for row in range(mask.shape[0]):
        previous_col = 0
        for col in np.flatnonzero(mask[row]):
            gap = int(col) - previous_col
            while gap > max_stored_delta:
                index_deltas.append(max_stored_delta)
                weight_symbols.append(ZERO_SYMBOL)
                gap -= max_stored_delta
                filler_count += 1
            index_deltas.append(gap)
            weight_symbols.append(int(centroid_ids[row, col]))
            previous_col = int(col)
        row_ptr.append(len(index_deltas))
    return (
        np.array(weight_symbols),
        np.array(index_deltas),
        np.array(row_ptr),
        filler_count,
    )

def entropy_code(codebooks, indices, masks, index_bits):
    packed = {}
    for name in codebooks:
        weight_syms, index_deltas, row_ptr, filler_count = csr_symbol_streams(
            masks[name], indices[name], index_bits[name]
        )
        packed[name] = {
            "codebook": codebooks[name],
            "row_ptr": row_ptr,
            "filler_count": filler_count,
            "weights": huffman_encode(weight_syms),
            "indices": huffman_encode(index_deltas),
        }
    return packed

def compress(net, data, thresholds, weight_bits, index_bits, retrain_steps, tune_steps):
    net, masks = prune_stage(net, data, thresholds, retrain_steps)
    codebooks, indices = {}, {}
    for name, layer in net.layers.items():
        k = 2 ** weight_bits[name]
        codebooks[name], indices[name] = quantize_layer(layer.W, masks[name], k)
    codebooks, indices = finetune_centroids(
        net, data, codebooks, indices, masks, lr=LR, steps=tune_steps
    )
    return entropy_code(codebooks, indices, masks, index_bits)
```
