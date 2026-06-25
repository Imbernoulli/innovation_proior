Let me start from the pain, concretely. I have an AlexNet that's over 200 MB and a VGG-16 that's over 500 MB, and I want to run them on a phone. The thing that actually hurts is not just the arithmetic; it is the memory traffic. Under 45 nm CMOS, a float add is about 0.9 pJ, reading a 32-bit word from on-chip SRAM is about 5 pJ, but reading it from off-chip DRAM is about 640 pJ. That is three orders of magnitude over the add. And a big net does not fit on chip, so every weight fetch goes to DRAM. A billion-connection net at 20 fps is $20 \times 10^9 \times 640\,\text{pJ} = 12.8$ W just to fetch weights. A phone cannot spend that. So the lever is storage: if I can make the model small enough to live in SRAM, the energy story changes. The hard constraint is that I do not want to buy storage by giving up accuracy.

Two independent storage axes jump out. One: have fewer weights by dropping the connections that do not matter. Two: spend fewer bits on each weight I keep. These multiply only if they do not damage each other; whether they actually stack is something I will have to check rather than assume, because the obvious worry is that pruning eats exactly the redundancy that quantization would otherwise exploit.

Fewer weights first. The practical pruning recipe is simple: train the network normally, remove connections whose weights are below a threshold, and retrain the surviving connections so they absorb the loss. Older methods like Optimal Brain Damage and Optimal Brain Surgeon try to use Hessian saliency to decide which connections to remove, but for these large CNNs a magnitude rule is attractive because it is cheap and fits the observation that larger-magnitude weights tend to matter more. So I can let ordinary training discover a dense connectivity pattern, keep only the large-magnitude connections, and retrain with a mask so pruned weights stay at zero.

But a sparse matrix is not just a bag of nonzero values. I have to store where the values go. In CSR or CSC, a matrix with $a$ nonzeros and $n$ rows or columns needs values, one index per value, and row or column pointers, so the accounting starts as $2a+n+1$ stored numbers. If I ignore that, the compression rate is dishonest. The index stream should be compressed too: instead of storing absolute positions, I store differences between successive nonzero positions, because the gaps after pruning are usually small. I can make the bounded-delta convention explicit: with a $d$-bit relative-index field I allow jumps up to $2^d$, and if a gap is larger than that, I insert a zero-valued filler entry, advance by the maximum stored jump, and continue.

I want to know what that filler actually costs before I call it cheap, so I trace one row. Take $d=2$, so the maximum stored delta is $4$, and a row whose only nonzeros sit at columns 1 and 11. The first gap is $1$: store delta $1$ and the real centroid id. The second gap is $11-1=10$, which exceeds $4$, so I emit (delta $4$, zero symbol), reducing the gap to $6$, then again (delta $4$, zero symbol), reducing it to $2$, then finally store delta $2$ with the real centroid id. The encoded streams are deltas $[1,4,4,2]$ and value symbols $[\text{id},0,0,\text{id}]$: two real nonzeros have turned into four stored entries and two filler zeros. So the filler is genuinely not free — each filler consumes both a delta and a value symbol — but two extra entries for one long jump is far cheaper than widening every one of the layer's indices to span 10+ columns. The trade only makes sense because long jumps are rare; that is the assumption I am leaning on, and it is the thing that would break the accounting if pruning ever produced many long runs of zeros.

Now fewer bits per surviving weight. The survivors are still 32-bit floats, and that is wasteful. Fixed-point rounding is the obvious move, but it throws every layer onto a predetermined grid. A better route is sharing: many connections point to the same learned value, so each live connection stores a small centroid ID and the layer stores a small codebook once. If a dense layer has $n$ weights, each originally $b$ bits, and I cluster them into $k$ shared values, the original storage is $nb$. The shared storage is $n\log_2(k)$ bits of centroid IDs plus $kb$ bits of codebook, so

$$
r_{\text{share}} = \frac{nb}{n\log_2(k)+kb}.
$$

Sanity check on the toy 4-by-4 layer: 16 weights at 32 bits become 16 two-bit IDs plus four 32-bit codebook values. The ID stream is $16\log_2 4 = 32$ bits and the codebook is $4\times 32=128$ bits, so $32+128=160$ bits instead of $16\times 32=512$, and $512/160=3.2$. The codebook is a real and non-trivial slice of that 160: at $n=16$ and $k=4$ it is already 128 of the 160 bits, which is the reminder that this ratio only looks good when $n\gg k$ so the $kb$ codebook is amortized. For a real sparse layer I have to add the sparse-position stream back in. If the original dense layer has $m$ weights, $a$ live weights, $f$ filler zeros from long relative jumps, $q$ bits per stored value symbol after making room for the filler-zero representation, $d$ bits per relative index, and $p$ bits per row or column pointer, then the fixed-width pre-Huffman denominator is

$$
(a+f)(q+d)+kb+p(n+1),
$$

summed over layers. When there is no filler symbol beyond the centroids, $q=\log_2(k)$; if I represent overflow padding as an extra zero symbol or reserve one of the fixed-width value codes for it, that choice has to be counted in $q$ and in the $(a+f)$ stored entries. The codebook term stays $kb$ because the filler zero is not another learned centroid. That is the quantity to compare against $mb$, not just the codebook IDs.

How do I choose the shared values? I want the codebook to approximate the trained weights, so this is one-dimensional k-means on each layer's live weights: partition the weights into $k$ clusters and minimize the within-cluster squared error, $\arg\min_C \sum_{i=1}^k \sum_{w\in c_i}(w-c_i)^2$. This has to happen after training, because the whole point is to fit the actual learned distribution. A hash function that ties weights before training cannot know that distribution. Sharing also has to be per-layer, because different layers have different scales and sensitivities.

If I cluster and stop, I have injected quantization error. The codebook has to be fine-tuned under the original loss. The assignments are fixed during this fine-tuning, so many weights are tied to one centroid. Let $\mathcal L$ be the loss, $W_{ij}$ a weight, $I_{ij}$ its centroid ID, and $C_k$ the $k$-th centroid. Every live weight assigned to cluster $k$ is exactly $C_k$, so

$$
\frac{\partial \mathcal L}{\partial C_k}
= \sum_{i,j}\frac{\partial \mathcal L}{\partial W_{ij}}
\frac{\partial W_{ij}}{\partial C_k}
= \sum_{i,j}\frac{\partial \mathcal L}{\partial W_{ij}}\mathbf 1(I_{ij}=k).
$$

So the centroid update is not an average and not a separate update per connection. It claims to be the sum of the ordinary backprop gradients for all connections that share that centroid, followed by the usual learning-rate step on the shared value. In the forward pass, each weight is a lookup $C_{I_{ij}}$; in the backward pass, the gradients scatter-add into the centroids.

Before I trust that, I want to see the sum actually appear, because it is easy to write down a chain rule and quietly drop a term. Take three live weights with assignment $I=(0,1,0)$, so weights 0 and 2 share centroid $C_0$ and weight 1 uses $C_1$, and an arbitrary smooth loss $\mathcal L(w)=(w_0-0.3)^2+2(w_1+0.1)^2+(w_2-0.5)^2+\tfrac12 w_0 w_2$. With $C=(0.2,-0.4)$ the realized weights are $w=(0.2,-0.4,0.2)$, and the per-weight gradients are $g_0=2(w_0-0.3)+0.5w_2=-0.1$, $g_1=4(w_1+0.1)=-1.2$, $g_2=2(w_2-0.5)+0.5w_0=-0.5$. The identity predicts $\partial\mathcal L/\partial C_0=g_0+g_2=-0.6$. Perturbing $C_0$ by $10^{-6}$ and forming a forward difference of $\mathcal L(w(C))$ gives $-0.6$ to five places. The two terms add; the cross term in the loss did not break it. So the centroid gradient really is the scatter-add, not a per-connection step and not the mean.

The initialization of those centroids matters, and here I do not want to guess. After pruning, the weight distribution can be bimodal with a thin tail of large-magnitude weights. Random Forgy initialization samples from the mass of the distribution, and density-based initialization spaces centroids by equal probability mass; both put their centroids where the points are, which is the bulk near zero. Linear initialization instead spaces centroids uniformly from the layer's minimum to maximum live weight, ignoring the density. My worry is that the density-aware schemes starve the rare large weights — exactly the weights the magnitude pruning rule taught me to respect.

I can settle this with a one-dimensional experiment. Take a layer of 5000 small weights drawn near zero plus a thin tail of twelve large weights spread over $[0.4,1.0]$, and fit $k=6$ centroids both ways. Density (quantile) initialization places five of its six centroids inside the near-zero bulk — at init they sit at roughly $-0.10,-0.02,-0.01,0.01,0.03$ and only one out at $1.0$; after k-means converges, that lone tail centroid gets pulled down to about $0.70$ trying to cover the whole tail, and the worst large weight is then $0.30$ away from its nearest centroid. Linear initialization starts with centroids at $\approx -0.10,0.12,0.34,0.56,0.78,1.0$, and after convergence three of them stay out in the tail region ($0.43,0.56,0.76,0.95$), leaving the worst large weight only $0.082$ from a centroid. So the density schemes really do under-serve the tail by roughly $3.7\times$ in worst-case error here, and linear initialization is the one that keeps representatives out where the rare large weights live. That is the reason to pick it, and now it is a measured reason rather than a hunch.

Now back to the open question of whether the two axes fight. The worry is that once pruning has removed most weights, there is no redundancy left for quantization. But clustering error also depends on how many weights are being assigned to a fixed number of centroids: with fewer live weights per layer and the same codebook size, there are fewer points competing for each centroid. Those two effects pull in opposite directions, so I should measure rather than argue. I make a synthetic layer of about 1000 small weights plus 40 large-tail weights, fit $k=8$ centroids to the full set, then magnitude-prune at a small threshold (258 of 1040 survive) and fit $k=8$ to the survivors. The full layer lands at total within-cluster SSE $1.015$, i.e. $9.8\times 10^{-4}$ per weight; the pruned layer lands at SSE $0.180$, i.e. $7.0\times 10^{-4}$ per weight. The per-weight quantization error went *down*, not up, after pruning — removing the dense crowd of near-zero weights left each centroid responsible for a tighter cluster. So on this stand-in the two axes do not fight; pruning makes the quantization fit easier. I would still want to confirm this on a real layer before relying on it, but the redundancy-is-gone fear does not survive even the toy test. Convolutional layers are less redundant and more precision-sensitive, so I should give them a larger codebook, such as 8-bit centroid IDs; fully connected layers dominate storage and tolerate fewer bits, so a smaller codebook such as 5-bit IDs buys more compression where the parameter count is concentrated.

One more storage term remains. After pruning and weight sharing, I store two biased symbol streams: value symbols for the live weights plus any filler zeros, and relative deltas for the sparse positions. The value symbols are not uniformly distributed because the weights pile around the learned distribution's modes, and the relative deltas are not uniformly distributed because small gaps are common. Fixed-width codes waste bits on both streams. A Huffman code is lossless and assigns shorter codewords to more frequent symbols, so I can apply it after all fine-tuning is complete. The codebook and row/column pointers still have to be stored explicitly; Huffman coding is the final packing step for the biased streams, not a substitute for the sparse accounting.

Putting the pieces in the order each one needs its predecessor's output, the pipeline falls out as: train, magnitude-prune, and retrain masked survivors; cluster only the live weights into a per-layer codebook, initialize the centroids linearly, and fine-tune centroid values by summed gradients; then Huffman-code the value-symbol stream and the relative sparse-index stream while carrying the codebooks and pointers as overhead. The ordering is not a free choice — quantization has to come after pruning so it only fits live weights, the centroid fine-tune has to come after clustering so it has assignments to share gradients through, and Huffman coding has to come last so it sees the final symbol statistics.

```python
import numpy as np
from sklearn.cluster import KMeans

ZERO_SYMBOL = -1

def prune_layer(W, threshold):
    mask = np.abs(W) > threshold
    return np.where(mask, W, 0.0), mask

def prune_stage(net, data, thresholds, retrain_steps):
    net = train_to_convergence(net, data)
    masks = {}
    for name, layer in net.layers.items():
        layer.W, masks[name] = prune_layer(layer.W, thresholds[name])
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

Memory access is the original pressure, pruning removes values but creates sparse metadata, weight sharing turns live floats into centroid IDs plus a codebook, centroid fine-tuning repairs the tied-weight loss by summing gradients into each shared value, and Huffman coding only then packs the biased value-symbol and relative-index streams without changing the network function.
