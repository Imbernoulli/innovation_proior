We want to train a deep self-attention sequence model on inputs that are tens of thousands of tokens long — whole documents, music with tens of thousands of events, images flattened into thousands of pixels — and we want it to fit and train on a single accelerator at length $L = 64\text{K}$ without giving up the quality of the standard architecture. The frustrating part is that a first arithmetic check says this should be easy: a very large reported Transformer layer holds about $0.5\text{B}$ parameters, roughly $2\text{GB}$ in 32-bit floats, and a batch of $8$ sequences of length $64\text{K}$ at width $1024$ is again about $0.5\text{B}$ floats, another $2\text{GB}$. One layer's parameters plus one layer's activations should sit comfortably on one device. Yet these models cannot even be fine-tuned on a single machine, so the naive estimate is missing the real multipliers, and the question becomes precisely where the memory goes.

It goes to three separate failures of that same hope, each multiplicative. Backpropagation needs every layer's forward activations to form that layer's gradient, so the single-layer activation cost is multiplied by the depth $n_l$. The position-wise feed-forward sublayer projects up to an intermediate width $d_{ff}$ that is typically several times $d_{model}$ — a common setting is $d_{ff}=4096$ against $d_{model}=1024$ — so its intermediate is a $[b,L,d_{ff}]$ tensor, often the single largest activation in the model. And dense scaled dot-product attention forms $QK^\top$ of shape $[b,L,L]$; at $L=64\text{K}$ even one head at batch one is a $64\text{K}\times 64\text{K}$ matrix, about $16\text{GB}$ in fp32. The existing options each address at most one of these and leave the rest intact. Memory-efficient attention computes $\mathrm{softmax}(q_i K^\top/\sqrt{d_k})V$ one query row at a time so it never stores $QK^\top$, dropping memory to $O(L)$, but the time stays $O(L^2)$ because every query is still compared to every key. Fixed-pattern sparse attention cuts the $O(L^2)$ cost with a hand-designed strided or local sparsity pattern, but that pattern is content-independent and fixed in advance, so it cannot route a query to an arbitrary far-away position whose relevance is decided by the data. Reversible residual networks remove stored activations for convolutional image classifiers but had not been brought to attention-plus-feed-forward stacks, and they say nothing about the feed-forward width. No single tool removes the $L^2$ attention term, the $n_l$ activation multiplier, and the $d_{ff}$ peak together.

I propose Reformer, which removes all three with three matched ideas: locality-sensitive-hashing (LSH) attention for the $L^2$ term, reversible layers for the depth multiplier, and feed-forward (and loss) chunking for the $d_{ff}$ peak. The starting observation for attention is that for a query $q_i$ the output is $\mathrm{softmax}(q_i\cdot k_j)V$, and a softmax is dominated by its largest logits, so almost all the weight falls on a few keys with the largest dot products and the rest contribute essentially nothing. If I could cheaply find only the likely-nearest keys per query, attention would become content-dependent sparse — but I cannot find them by first computing all dot products, since that is exactly the quadratic work I am trying to avoid. Locality-sensitive hashing is the right tool: a hash $h(x)$ for which nearby vectors collide and distant ones usually do not. I want angular similarity, because after normalizing the key a dot product is the query norm times a cosine, so I draw a random $R\in\mathbb{R}^{d_k\times b/2}$, form $xR$, concatenate its negation, and read off the bucket as
$$h(x)=\arg\max([xR;\,-xR]),$$
giving $b$ signed-axis buckets from $b/2$ random directions through one batched matmul and an argmax; near-in-angle vectors land in the same bucket with high probability.

There is a mismatch that has to be fixed before hashing can live inside attention. In ordinary multi-head attention $Q$ and $K$ are different learned projections, so a query and the key it should attend to can have a large dot product without sharing any hash geometry. The clean fix is to share the query-key projection: produce one $QK$ tensor and a separate $V$ tensor, use the query unnormalized as the query vector, and use its unit-normalized version $k_j=q_j/\lVert q_j\rVert$ as the key. The key direction then matches the angular hash exactly, while $\lVert q_i\rVert$ survives as a per-query softmax temperature. The attention target for a position is $o_i=\sum_{j\in\mathcal{P}_i}\exp(q_i\cdot k_j-z(i,\mathcal{P}_i))\,v_j$ with $z(i,\mathcal{P}_i)=\log\sum_{j\in\mathcal{P}_i}\exp(q_i\cdot k_j)$, and for hashed attention the allowed set is $\mathcal{P}_i=\{j:h(q_i)=h(k_j)\}$, intersected with the causal mask when the model is autoregressive.

The next problem is mechanical rather than mathematical: hash buckets are ragged, and an accelerator wants fixed-size matrix multiplies, not a variable key count per query. I sort positions by the pair $(h(q_i),i)$ so that items in the same bucket become contiguous and sequence order breaks ties, then cut the sorted sequence into chunks of length $m$ and let each chunk attend to itself and to one previous chunk. The one-previous-chunk rule matters because a bucket can straddle a chunk boundary; as long as every bucket is smaller than $m$, this chunk-plus-one-back window covers all true same-bucket partners. Since the average bucket holds $l/n_{buckets}$ items, I set the chunk to twice that, $m=2l/n_{buckets}$, equivalently $n_{buckets}=2l/m$. Sorting contributes a $\log l$ factor; the chunked dot products scale with $l$ times the fixed local window and the number of hash rounds, so the dense $l^2$ score matrix is gone.

A single random rotation can miss a genuine neighbor, so I run $n_{rounds}$ independent hashes and take the union, $\mathcal{P}_i=\bigcup_r \mathcal{P}_i^{(r)}$ with $\mathcal{P}_i^{(r)}=\{j:h^{(r)}(q_i)=h^{(r)}(q_j)\}$. The exact union has a double-counting hazard: a key appearing in several rounds would be summed several times. Writing $N_{i,j}=|\{r:j\in\mathcal{P}_i^{(r)}\}|$ for its multiplicity, the exact correction subtracts $\log N_{i,j}$ inside the round mask so each of the $N_{i,j}$ appearances carries weight $1/N_{i,j}$, and the round outputs are recombined by rebasing their local log-partitions onto the union denominator,
$$o_i=\sum_r \exp\!\big(z(i,\mathcal{P}_i^{(r)})-z(i,\mathcal{P}_i)\big)\,o_i^{(r)}.$$
The sign here is what makes it correct: each $o_i^{(r)}$ already has its local round denominator divided out, and the factor $\exp(z^{(r)}-z)$ moves that local softmax back onto the shared union denominator. Two masks do separate jobs. Causality cannot read sorted order, since sorting has scrambled positions, so I carry each token's original index through the same permutation and compare original indices within a chunk. The self-mask is subtler: because $q_i\cdot(q_i/\lVert q_i\rVert)=\lVert q_i\rVert$, a position's logit against itself is typically the largest in its bucket, and letting it attend freely to itself drives the layer toward identity. I therefore subtract a large but finite penalty, $10^5$, for $i=j$; it must be finite rather than $+\infty$ because the first causal token may have no other legal key, and then the finite self logit is the only finite option keeping the softmax defined. The future/padding mask, by contrast, uses a much larger $10^9$ penalty.

That handles the attention term. The depth multiplier on stored activations is removed by reversible layers. An ordinary residual layer $y=x+F(x)$ cannot be inverted without solving for $x$, but a reversible block can: split the state into two full-width halves and compute
$$Y_1=X_1+\mathrm{Attention}(X_2),\qquad Y_2=X_2+\mathrm{FeedForward}(Y_1),$$
which inverts exactly by subtraction,
$$X_2=Y_2-\mathrm{FeedForward}(Y_1),\qquad X_1=Y_1-\mathrm{Attention}(X_2).$$
Here $F$ is the attention sublayer and $G$ the feed-forward sublayer, with normalization moved inside those residual functions. On the backward pass each layer's inputs are reconstructed from its outputs and the local gradient is computed immediately, so no per-layer activations are cached and the $n_l$ multiplier disappears. The one requirement is that any randomness inside $F$ or $G$ — dropout, for instance — see the same draws during the recompute as during the forward pass, so those layers must be made deterministic across the reconstruction. Finally, the feed-forward peak: the FFN is independent across sequence positions, so I split the sequence axis into chunks, apply the same network one chunk at a time, and concatenate, $Y_2=[X_2^{(1)}+\mathrm{FF}(Y_1^{(1)});\ldots;X_2^{(c)}+\mathrm{FF}(Y_1^{(c)})]$. This is functionally identical to the dense FFN but turns a full $[b,l,d_{ff}]$ intermediate into one chunk's intermediate, and the same chunking applies to the reverse computation, the backward pass, and the large-vocabulary log-probability/loss. Together the irreducible activation scale drops to roughly $bld_{model}$: LSH attention removes $l^2$, reversible layers remove the depth multiplier, feed-forward chunking removes the $d_{ff}$ peak, and what remains is still a Transformer in its modeling ingredients, just one whose memory is no longer dictated by dense scores and stored activations.

```python
TOKEN_SELF_ATTN_PENALTY = 1e5
CAUSAL_OR_PADDING_PENALTY = 1e9

def length_normalized(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)

def hash_vecs(vecs, n_buckets, n_hashes, rng):
    assert n_buckets % 2 == 0
    rotations = random.normal(rng, (vecs.shape[-1], n_hashes, n_buckets // 2))
    rotated = np.einsum("td,dhb->htb", vecs, rotations)
    return np.argmax(np.concatenate([rotated, -rotated], axis=-1), axis=-1)

def look_adjacent(x, n_before=1, n_after=0):
    parts = []
    for offset in range(-n_before, n_after + 1):
        if offset == 0:
            parts.append(x)
        else:
            parts.append(np.concatenate([x[offset:], x[:offset]], axis=0))
    return np.concatenate(parts, axis=1)

def attend_shared_qk(sorted_q, sorted_v, sorted_pos, chunk_len, causal=True):
    q = sorted_q.reshape((-1, chunk_len, sorted_q.shape[-1]))
    v = sorted_v.reshape((-1, chunk_len, sorted_v.shape[-1]))
    q_pos = sorted_pos.reshape((-1, chunk_len))

    k = length_normalized(q) / np.sqrt(q.shape[-1])
    k = look_adjacent(k, n_before=1)
    v = look_adjacent(v, n_before=1)
    kv_pos = look_adjacent(q_pos, n_before=1)

    dots = np.matmul(q, np.swapaxes(k, -1, -2))
    if causal:
        dots -= CAUSAL_OR_PADDING_PENALTY * (q_pos[:, :, None] < kv_pos[:, None, :])
    dots -= TOKEN_SELF_ATTN_PENALTY * (q_pos[:, :, None] == kv_pos[:, None, :])

    log_z = logsumexp(dots, axis=-1, keepdims=True)
    probs = np.exp(dots - log_z)
    return np.matmul(probs, v).reshape((-1, v.shape[-1])), log_z.reshape((-1,))

def lsh_self_attention(qk, v, chunk_len, n_hashes, rng):
    length = qk.shape[0]
    n_buckets = 2 * max(1, length // chunk_len)
    buckets = hash_vecs(qk, n_buckets, n_hashes, rng)
    buckets = (buckets + np.arange(n_hashes)[:, None] * n_buckets).reshape(-1)

    ticker = np.arange(n_hashes * length)
    sort_key = length * buckets + (ticker % length)
    order = np.argsort(sort_key)
    undo = np.argsort(order)
    sorted_pos = ticker[order] % length

    sorted_qk = qk[sorted_pos]
    sorted_v = v[sorted_pos]
    out, logits = attend_shared_qk(sorted_qk, sorted_v, sorted_pos, chunk_len)

    out = out[undo].reshape((n_hashes, length, -1))
    logits = logits[undo].reshape((n_hashes, length, 1))
    round_weights = np.exp(logits - logsumexp(logits, axis=0, keepdims=True))
    return np.sum(out * round_weights, axis=0)

def reversible_block(x1, x2, attention, feed_forward):
    y1 = x1 + attention(x2)
    y2 = x2 + feed_forward(y1)
    return y1, y2

def reverse_reversible_block(y1, y2, attention, feed_forward):
    x2 = y2 - feed_forward(y1)
    x1 = y1 - attention(x2)
    return x1, x2
```
