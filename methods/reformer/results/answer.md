# Reformer: The Efficient Transformer

## Problem

A long-sequence Transformer runs out of accelerator memory through three separate
costs: dense attention materializes $L\times L$ scores; backpropagation stores
activations for all $n_l$ layers; and the position-wise FFN materializes a wide
$d_{ff}$ intermediate. The method removes those costs with LSH attention,
reversible layers, and chunked feed-forward/loss computation.

## Method

**LSH attention.** Use a shared query-key projection plus a separate value
projection. The query remains unnormalized; the key is $k_j=q_j/\lVert q_j\rVert$,
so key direction matches angular hashing and query norm acts as softmax
temperature. With $R\in\mathbb{R}^{d_k\times b/2}$,
$$h(x)=\arg\max([xR;\,-xR])$$
is the angular hash. Sort by $(h(x),\text{position})$, chunk the sorted sequence
into length-$m$ chunks, and let each chunk attend to itself plus one previous
chunk. The chunking relation is $m=2l/n_{buckets}$, so code that treats `chunk_len`
as $m$ should default to `n_buckets = 2 * length // chunk_len`.

For multiple hashes,
$$\mathcal{P}_i=\bigcup_r \mathcal{P}_i^{(r)},\qquad
\mathcal{P}_i^{(r)}=\{j:h^{(r)}(q_i)=h^{(r)}(q_j)\}.$$
The exact union correction uses
$$N_{i,j}=|\{r:j\in\mathcal{P}_i^{(r)}\}|,\qquad
m^{(r)}_{i,j}=\begin{cases}
\infty & j\notin\mathcal{P}_i^{(r)}\\
10^5 & i=j\\
\log N_{i,j} & \text{otherwise}
\end{cases}$$
and combines local round outputs by
$$o_i=\sum_r \exp(z(i,\mathcal{P}_i^{(r)})-z(i,\mathcal{P}_i))\,o_i^{(r)}.$$
The $10^5$ case is a positive mask penalty subtracted from the logit; it is
finite so a token with no other legal target can still attend to itself.

**Reversible layers.** Split each layer state into two full-width halves:
$$Y_1=X_1+\mathrm{Attention}(X_2),\qquad
Y_2=X_2+\mathrm{FeedForward}(Y_1).$$
Invert by subtraction:
$$X_2=Y_2-\mathrm{FeedForward}(Y_1),\qquad
X_1=Y_1-\mathrm{Attention}(X_2).$$
Inputs are reconstructed during backpropagation, so activations need not be
stored once per layer. Layer normalization moves inside the residual functions.

**Chunked FFN and loss.** The FFN is independent across sequence positions, so it
can be evaluated one sequence chunk at a time and concatenated. This is
functionally identical to the dense FFN but reduces peak memory from a full
$[b,l,d_{ff}]$ intermediate to one chunk's intermediate. The output
log-probability/loss can be chunked similarly for large vocabularies.

## Complexity

With $n_c=l/32$ LSH chunks and $c=(4l/n_c)^2=128^2$, the model-level
memory table is:

| Model | Memory |
|---|---|
| Transformer | $\max(bld_{ff}, bn_hl^2)n_l$ |
| Reversible Transformer | $\max(bld_{ff}, bn_hl^2)$ |
| Chunked Reversible Transformer | $\max(bld_{model}, bn_hl^2)$ |
| LSH Transformer | $\max(bld_{ff}, bn_hln_rc)n_l$ |
| Reformer | $\max(bld_{model}, bn_hln_rc)$ |

The corresponding Reformer time term is
$(bld_{ff}+bn_hn_rlc)n_l$: reversible layers save memory, not compute, while LSH
replaces dense attention with local chunk attention plus hashing/sorting.

## Canonical Implementation Notes

Google Trax serves as the reference implementation. In Trax commit
`31022d6cd7dd525ed11a04d84cd3936228499173`:

- `hash_vecs` uses random rotations, concatenates `[rotated, -rotated]`, and
  `argmax`es signed axes; for large bucket counts it factorizes the bucket space.
- `PureLSHSelfAttention.hash_vectors` defaults to
  `n_buckets = 2 * max(1, length // chunk_len)`, then offsets bucket IDs by hash
  round before sorting.
- `attend(..., k=None)` implements shared-QK by setting `k=q`, applying RMS
  length normalization, then dividing by `sqrt(d)`, which is equivalent to an
  L2-normalized key.
- `mask_self_attention` subtracts `1e9` for future/padding masks and `1e5` for
  self masks.
- The Trax round combiner weights per-round outputs by softmaxed per-round
  log-partitions. It does not explicitly compute the
  `log N_{i,j}` duplicate correction in the current code path, so a code
  artifact should not claim exact duplicate-set union unless it also computes
  those counts.
- `ReformerLM` starts the reversible stack with `Dup()`, applies
  `ReversibleSerial` blocks built from `ReversibleHalfResidual` and
  `ReversibleSwap`, then `Concatenate()`s the two halves before the final norm
  and output layer.

## Code Skeleton

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
