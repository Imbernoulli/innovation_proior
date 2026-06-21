The central bottleneck for running large convolutional networks on phones and other embedded devices is not the cost of floating-point arithmetic but the cost of moving weights from off-chip DRAM. A 32-bit DRAM access consumes roughly three orders of magnitude more energy than an on-chip float add, so a billion-connection network can easily spend over ten watts just fetching weights at modest frame rates. The only way to fix this without redesigning the hardware is to make the model small enough to live in SRAM. Pruning helps by removing redundant connections, and quantization helps by spending fewer bits per remaining connection, but neither alone attacks both the value count and the bits per value, and neither honestly accounts for sparse-index overhead. We need a method that prunes, shares weights, and then losslessly entropy-codes the resulting sparse-index and centroid streams, while keeping the original accuracy.

The method is Deep Compression. It is a three-stage pipeline applied to an already-trained network. First, magnitude pruning removes every connection whose absolute weight is below a layer-specific threshold, and the surviving weights are retrained with a fixed mask so the pruned connections stay at zero. Second, trained quantization with weight sharing clusters each layer's surviving weights into a small codebook of shared values using one-dimensional k-means. Third, Huffman coding compresses the biased streams of centroid identifiers and relative sparse-index deltas. Because the storage accounting must be honest, the final size includes the entropy-coded value stream, the entropy-coded index stream, the codebook entries, and the sparse row or column pointers.

Pruning reduces the number of live values, but a sparse matrix still needs to know where those values live. I store the sparse pattern using compressed sparse row form with relative index differences. Each nonzero contributes a value symbol and an index delta, and each row contributes a pointer. The index field has a fixed per-layer bit width; if a gap between consecutive live weights exceeds the representable range, I insert a zero-valued filler entry, advance by the maximum delta, and continue. That filler costs both an index delta and a value symbol, and it is counted in the final storage.

Weight sharing turns the surviving full-precision floats into small centroid identifiers. For each layer, I collect only the live weights and run k-means with k chosen so that the identifiers fit in a desired number of bits. Convolutional layers are usually given 8-bit shared weights because they are precision-sensitive, while fully connected layers, which dominate storage, can use 5-bit shared weights. The centroids are initialized linearly across the layer's weight range rather than by random sampling, so rare large-magnitude weights still get nearby representatives. After clustering, the assignments are fixed and the centroid values are fine-tuned: every connection assigned to a centroid contributes its gradient to that centroid, so the shared value is updated by the summed gradient of all its members.

Once pruning, clustering, and centroid fine-tuning are complete, the value-symbol stream and the relative-index stream are heavily biased: small centroid IDs and small index gaps occur much more often than large ones. Huffman coding is applied to both streams as the final lossless packing step. The codebooks and row pointers are stored explicitly, not Huffman-coded. The final compression rate is the original dense float storage divided by the total of all counted terms.

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


def compress(net, data, thresholds, weight_bits, index_bits, retrain_steps, tune_steps, lr):
    net, masks = prune_stage(net, data, thresholds, retrain_steps)
    codebooks, indices = {}, {}
    for name, layer in net.layers.items():
        k = 2 ** weight_bits[name]
        codebooks[name], indices[name] = quantize_layer(layer.W, masks[name], k)
    codebooks, indices = finetune_centroids(
        net, data, codebooks, indices, masks, lr=lr, steps=tune_steps
    )
    return entropy_code(codebooks, indices, masks, index_bits)
```
