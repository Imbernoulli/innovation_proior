Let me start from the pain, concretely. I have an AlexNet that's 240 MB and a VGG-16 that's over 500 MB, and I want to run them on a phone. The thing that actually hurts isn't the arithmetic — it's the memory. Under 45 nm CMOS a float add is about 0.9 pJ, reading a 32-bit word from on-chip SRAM is about 5 pJ, but reading it from off-chip DRAM is about 640 pJ. That's three orders of magnitude over the add. And a big net doesn't fit on chip, so every weight fetch goes to DRAM. A billion-connection net at 20 fps is $20 \times 10^9 \times 640\,\text{pJ} = 12.8$ W just to fetch weights — a phone can't do that. So the lever is storage: if I can make the model small enough to live in SRAM, the energy story flips. And the constraint I'm not willing to give up is accuracy — I want order-of-magnitude shrinkage with essentially zero accuracy loss.

So what reduces storage? Two independent axes jump out. One: have *fewer* weights — drop the ones that don't matter. Two: spend *fewer bits* on each weight I keep. Let me think about each, and then whether they can be combined.

Fewer weights first. Han et al. just showed that plain magnitude pruning works embarrassingly well: train the net, throw away every connection whose weight is below some threshold, then retrain the survivors so they take up the slack. Iterate. You can lose ~9–13× of the parameters of AlexNet/VGG with no accuracy loss. There's older, fancier machinery — Optimal Brain Damage and Optimal Brain Surgeon use the Hessian to pick which connections to delete, arguing second-order saliency beats magnitude — but magnitude pruning is dead simple and empirically suffices once you retrain. So step one: learn the connectivity by normal training; prune small-weight connections; retrain to learn the final weights of the sparse survivors. Good.

But here's a thing I have to be honest about. Once the matrix is sparse, I can't just store the nonzeros — I have to store *where* they are. If I keep the matrix in compressed sparse row/column form, that's $2a + n + 1$ numbers for $a$ nonzeros and $n$ rows: the value, plus an index per value, plus the row pointers. So the index overhead is real, and if I'm sloppy it eats my savings. Two tricks. First, store the *difference* between successive indices rather than absolute positions, since after pruning the gaps between nonzeros are usually small — and small differences need few bits. I'll budget something like 5 bits for the index difference in FC layers and 8 bits in conv layers. Second, what if a gap exceeds what those bits can hold? Say I have 3 bits for the difference (max 8) and the actual gap is bigger. Then I pad: I insert a filler *zero* at the position the 3-bit jump can reach, and continue the jump from there. So overflow is handled by zero-padding rather than by widening every index. Fine — pruning plus careful sparse indexing is stage one.

Now fewer bits per weight. The survivors are still 32-bit floats. Do I need 32 bits? Almost certainly not — these weights are redundant and noisy. The naive thing is to round each weight to a low-bit fixed-point grid. But I can do better by *sharing*: force groups of connections to use the exact same value, store one small index per connection pointing into a tiny table of shared values (a codebook), and store the codebook once. If I cluster $n$ weights in a layer into $k$ shared values, then I need $\log_2(k)$ bits per index plus $k$ full-precision codebook entries. Let me write the compression rate. Originally $n$ weights at $b$ bits each is $nb$. After sharing: $n\log_2(k)$ bits of indices plus $kb$ bits of codebook. So

  r = nb / ( n·log₂(k) + k·b ).

Sanity check on a toy: a 4×4 layer, 16 weights at 32 bits, clustered to $k=4$ shared values. Indices: $16 \times \log_2 4 = 16 \times 2 = 32$ bits. Codebook: $4 \times 32 = 128$ bits. Total $160$. Original $16 \times 32 = 512$. Ratio $512/160 = 3.2$. Yes — matches the back-of-envelope $16\cdot32/(4\cdot32 + 2\cdot16)$. With $n \gg k$ the $kb$ codebook term is negligible and the rate is basically $b / \log_2 k$, e.g. 32/5 ≈ 6.4× from 5-bit indices.

How do I choose the shared values? I want the shared values to approximate the trained weights as closely as possible, so this is a clustering problem: partition the $n$ weights into $k$ clusters minimizing the within-cluster sum of squares, $\arg\min_C \sum_{i=1}^k \sum_{w\in c_i} (w - c_i)^2$. That's exactly 1-D k-means, with the centroids as the shared values. Crucially I do this *after* the network is fully trained, so the codebook is fit to the real weight distribution — unlike HashedNets, which fixes the sharing by a hash function before seeing any data and so can't match what the network learned. Sharing is per-layer; I don't share across layers.

Wait — if I just round/cluster and stop, I've injected quantization error and accuracy will drop. I need to *recover* it. So after clustering, fine-tune the codebook. But the weights are now tied — many connections share one centroid — so I can't update them independently. I have to update the *shared value*. Let me get the gradient right. Let $\mathcal{L}$ be the loss, $W_{ij}$ the weight at row $i$ column $j$, $I_{ij}$ its centroid index, and $C_k$ the $k$-th centroid. Every weight assigned to cluster $k$ *is* $C_k$. So

  ∂𝓛/∂C_k = Σ_{i,j} (∂𝓛/∂W_{ij}) · (∂W_{ij}/∂C_k) = Σ_{i,j} (∂𝓛/∂W_{ij}) · 𝟙(I_{ij} = k).

In words: the gradient on a shared value is the *sum* of the ordinary backprop gradients of all the connections that share it. So fine-tuning is: backprop as usual to get per-connection gradients, then group them by centroid index and sum within each group, multiply by the learning rate, and subtract from the centroid. The connections follow because they're just lookups into the updated table. There's one extra level of indirection on both the forward pass (look up $C_{I_{ij}}$) and the backward pass (scatter-add gradients by index), and that's it. This refits the codebook to the loss and recovers the accuracy lost to clustering.

Now, how do I *initialize* the k centroids? This matters more than I'd expect. Look at the weight distribution of a pruned conv layer — it's bimodal, two humps of typical-magnitude weights with a thin tail of large-magnitude ones. Three candidate initializations: Forgy (pick $k$ actual weights at random as centroids), density-based (space the centroids by equal probability mass — equivalently invert the CDF at equally spaced quantiles), and linear (space the centroids uniformly across the $[\min, \max]$ range of weights, ignoring the distribution). Let me reason about which is best, because this isn't obvious. Pruning already taught me that *large* weights matter more than small ones — they carry the important connections — but after pruning the large weights are *few* (the thin tail). Forgy and density-based both put centroids where the *mass* is, i.e. near the two humps; because the large weights are rare, almost no centroid lands out on the tail. So the few important large weights get snapped to a much smaller centroid — they're badly represented. Linear initialization doesn't care about the distribution at all, so it spreads centroids evenly all the way out to $\max$, giving the large weights a centroid near them. So linear initialization should best preserve the large, important weights. That argues for linear init — and it's a nice consequence of the "large weights matter" principle.

Now the question that decides whether this whole thing is worth it: do pruning and quantization *interfere*? It would be plausible that after I've already removed most weights, quantizing the survivors to a few bits would push accuracy off a cliff, because there's no redundancy left to absorb the rounding. Let me think about it the other way. Unpruned AlexNet has ~60M weights to cluster into $k$ centroids; pruned AlexNet has only ~6.7M. With the *same* number of centroids, fewer weights per centroid means a tighter fit — less quantization error per weight, not more. So pruning should make quantization *easier*, or at least not harder. Empirically the accuracy-vs-bits curve for "quantize the pruned net" sits right on top of the curve for "quantize the unpruned net" — they drop at the same bit-width. So the two compress *independently*: pruning removes count, quantization removes bits, and they don't fight. That's the key insight that makes the rate multiply rather than collide. Each alone falls off a cliff below ~8% of original size; combined, I can get down to ~3% with no accuracy loss. And the layers behave differently: conv layers need more precision (accuracy drops below ~4 bits) while FC layers tolerate fewer (robust down to ~2 bits) — so I'll spend more bits on conv (say 8) and fewer on FC (say 5).

One more free win. After pruning+quantization, what am I actually storing? Two streams of symbols: the centroid indices (the weights) and the index differences (the sparsity pattern). Both are *biased*: the quantized weights pile up around the two humps of the distribution, and the index differences are almost always small (rarely above ~20). A fixed-width code wastes bits on a biased source. A Huffman code is the optimal prefix code for a known symbol-frequency distribution — frequent symbols get shorter codewords. So just Huffman-code both streams. This is lossless, costs nothing at accuracy, and squeezes out roughly another 20–30%. It's an offline post-processing step after all fine-tuning is done — no training involved.

So the pipeline lands as three stages that stack: **prune** (learn connectivity, delete small weights, retrain the sparse survivors, store as CSR with relative-index differences and zero-padding for overflow) → **trained quantization / weight sharing** (per-layer 1-D k-means with linear centroid init, store indices + codebook, fine-tune the centroids by gradient-summing within each cluster) → **Huffman coding** (lossless entropy coding of the biased weight and index streams). Pruning gives ~9–13×; adding quantization pushes it to ~27–31× (the rate including index and codebook overhead); Huffman gets to ~35–49×.

Now the code, stage by stage.

```python
import numpy as np
from sklearn.cluster import KMeans

# ---------- Stage 1: magnitude pruning ----------
def prune_layer(W, sparsity_threshold):
    mask = np.abs(W) > sparsity_threshold        # keep large-magnitude connections
    return W * mask, mask

def prune_stage(net, data, thresholds):
    net = train_to_convergence(net, data)        # learn the connectivity first
    masks = {}
    for name, layer in net.layers.items():
        layer.W, masks[name] = prune_layer(layer.W, thresholds[name])
    # retrain survivors; pruned weights are held at 0 by re-applying the mask
    for step in range(N_RETRAIN):
        backprop_step(net, data)
        for name, layer in net.layers.items():
            layer.W *= masks[name]               # masked update: pruned stay 0
    return net, masks

def to_csr_relative(W):
    # store nonzeros + relative index differences; 5 bits (FC) / 8 bits (conv);
    # pad a filler zero when a gap exceeds the index range (overflow handling).
    ...

# ---------- Stage 2: trained quantization / weight sharing ----------
def quantize_layer(W, mask, k):
    w = W[mask].reshape(-1, 1)
    lo, hi = w.min(), w.max()
    init = np.linspace(lo, hi, k).reshape(-1, 1) # LINEAR init: keeps the rare large weights
    km = KMeans(n_clusters=k, init=init, n_init=1).fit(w)
    centroids = km.cluster_centers_.reshape(-1)  # the shared values (codebook)
    idx = np.zeros_like(W, dtype=np.int32)
    idx[mask] = km.labels_                       # per-connection index into codebook
    return centroids, idx

def finetune_centroids(net, data, codebooks, indices, masks, lr, steps):
    for step in range(steps):
        grads = backprop_grads(net, data)        # dL/dW per connection
        for name, layer in net.layers.items():
            C, I, m = codebooks[name], indices[name], masks[name]
            # dL/dC_k = sum over connections sharing centroid k of dL/dW_ij
            gC = np.zeros_like(C)
            np.add.at(gC, I[m], grads[name][m])  # group-by-index sum
            C -= lr * gC                          # update shared values
            layer.W = C[I] * m                     # write shared values back (lookup)
    return codebooks, indices

def quantize_stage(net, data, masks, bits):
    codebooks, indices = {}, {}
    for name, layer in net.layers.items():
        k = 2 ** bits[name]                        # 8 bits conv, 5 bits FC
        codebooks[name], indices[name] = quantize_layer(layer.W, masks[name], k)
    return finetune_centroids(net, data, codebooks, indices, masks, lr=LR, steps=N)

# ---------- Stage 3: Huffman coding (lossless, offline) ----------
def huffman_encode(symbols):
    # optimal prefix code from the (biased) symbol frequencies
    ...

def entropy_code(codebooks, indices):
    streams = {}
    for name in codebooks:
        streams[name] = (huffman_encode(quantize_to_symbols(codebooks[name], indices[name])),
                         huffman_encode(relative_index_diffs(indices[name])))
    return streams

def compress(net, data, thresholds, bits):
    net, masks = prune_stage(net, data, thresholds)
    codebooks, indices = quantize_stage(net, data, masks, bits)
    return entropy_code(codebooks, indices)
```

The causal chain: memory access, not arithmetic, dominates energy, and a big net forces DRAM fetches; shrinking it to fit SRAM is the goal. Removing redundant connections by magnitude pruning cuts the weight count ~10× but leaves 32-bit survivors and sparse-index overhead, so I store relative index differences with zero-padding for overflow. Spending fewer bits per survivor via per-layer k-means weight sharing — initialized linearly so the rare large weights keep a centroid, fine-tuned by summing each cluster's per-connection gradients — cuts bits to ~5 (FC) / ~8 (conv) with no accuracy loss, because pruning leaves fewer weights per centroid and the two stages don't interfere. Finally Huffman coding the biased weight and index streams is a free lossless squeeze. Together they take AlexNet/VGG down to a few percent of their size with the accuracy intact.
