# Graph Attention Networks (GAT)

## Problem

Build a neural layer that operates on arbitrarily structured graphs and recovers the strengths of convolution: shared, learnable parameters applied across all nodes, handling neighborhoods of varying size and no canonical ordering. Free spectral filters tie the learned filter to one graph's Laplacian eigenbasis, while Chebyshev/GCN-style approximations make the operation local and cheap but still use graph-structural propagation coefficients. GCN pins each unweighted edge contribution to the fixed, untrainable weight `1/√(d̃_i d̃_j)`, so it cannot assign different importances within a neighborhood. GraphSAGE achieves inductiveness by learning feature-based aggregators, but samples a fixed-size neighborhood, weights neighbors uniformly (mean/GCN aggregators), and uses an LSTM aggregator that forces an artificial ordering on the unordered neighbor set.

## Key idea

Let each node attend over its neighbors with a shared self-attention mechanism. Because the mechanism is a function of node *features* (not of a fixed `N×N` graph operator), the same parameters apply to any graph — making the layer inductive — while learned attention coefficients assign different importances to different neighbors, over the entire neighborhood, with no imposed ordering and no eigendecomposition.

## The graph attentional layer

Given node features `h = {h_1, …, h_N}`, `h_i ∈ ℝ^F`, the layer produces `h_i' ∈ ℝ^{F'}`:

1. **Shared linear transform.** Apply `W ∈ ℝ^{F'×F}` to every node: `W h_i`.
2. **Attention scores (masked self-attention).** With a shared weight vector `a⃗ ∈ ℝ^{2F'}` and LeakyReLU (negative slope 0.2), score only over neighbors `j ∈ 𝒩_i` (first-order neighbors, including `i`):

   `e_ij = LeakyReLU( a⃗ᵀ [W h_i ‖ W h_j] )`.
3. **Normalize over the neighborhood:**

   `α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_{k∈𝒩_i} exp(e_ik)`.
4. **Aggregate:**

   `h_i' = σ( Σ_{j∈𝒩_i} α_ij W h_j )`.
5. **Multi-head attention.** Run `K` independent heads. Concatenate them in hidden layers,

   `h_i' = ‖_{k=1}^{K} σ( Σ_{j∈𝒩_i} α_ij^k W^k h_j )`   (→ `K F'` features),

   and **average** them at the prediction layer (concatenation is nonsensical when the output is class scores), producing logits before the task softmax/sigmoid,

   `z_i = (1/K) Σ_{k=1}^{K} Σ_{j∈𝒩_i} α_ij^k W^k h_j`.

**Efficient scoring.** Since `a⃗ᵀ[W h_i ‖ W h_j] = a⃗_1ᵀ(W h_i) + a⃗_2ᵀ(W h_j)`, compute one scalar per node for each half and form the logit matrix by a broadcast sum, rather than materializing a `2F'` concatenation per edge.

**Properties.** Per-head method cost `O(|V| F F' + |E| F')` with a sparse edge-index implementation (on par with GCN; no eigendecomposition or inversion); fully parallel across edges (scores) and nodes (outputs); applies to directed graphs (omit `α_ij` for absent edges); inductive — the identical parameters run on completely unseen graphs. The dense TensorFlow 1 code below materializes an `N×N` attention tensor for simplicity, so that implementation has dense attention-tensor cost before masking.

## Configurations

- **Transductive (Cora, Citeseer):** 2 layers; layer 1 = 8 heads × 8 features (concat) + ELU; layer 2 = 1 head → `C` classes → softmax. L2 `λ=5e-4`, dropout `p=0.6` on inputs and on attention coefficients (dropping coefficients = sampling the neighborhood each step). Pubmed: 8 averaged output heads and `λ=1e-3`.
- **Inductive (PPI):** 3 layers; layers 1–2 = 4 heads × 256 features (concat) + ELU, with a skip connection across the intermediate layer; output = 6 heads × 121 features, averaged → logistic sigmoid (multi-label). No dropout/L2; batch of 2 graphs.
- Glorot init, Adam (lr 0.01 for Pubmed, 0.005 otherwise), early stopping on validation loss/accuracy (transductive) or micro-F1 (inductive), patience 100.
- **Const-GAT ablation:** set `a(x, y) = 1` (uniform neighbor averaging, same architecture) to isolate the value of learned attention.

## Working TensorFlow 1 code

```python
import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d


def attn_head(seq, out_sz, bias_mat, activation,
              in_drop=0.0, coef_drop=0.0, residual=False):
    """One attention head.

    seq: [batch, N, F]; bias_mat is broadcastable to [batch, N, N],
    with 0 on edges/self-loops and -1e9 off edges.
    """
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # shared linear transform W h_i (1x1 conv = per-node linear map)
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # additive attention a = [a1 || a2]: a^T[Wh_i || Wh_j] = a1^T Wh_i + a2^T Wh_j
        f_1 = tf.layers.conv1d(seq_fts, 1, 1, use_bias=False)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1, use_bias=False)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])          # raw_score[i,j] = f_1[i] + f_2[j]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits, alpha=0.2) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)    # attention dropout
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)                     # h_i' = sum_j alpha_ij W h_j
        ret = tf.contrib.layers.bias_add(vals)

        if residual:
            in_dim = seq.get_shape().as_list()[-1]
            out_dim = ret.get_shape().as_list()[-1]
            if in_dim != out_dim:
                ret = ret + conv1d(seq, out_dim, 1)
            else:
                ret = ret + seq

        return activation(ret)


class GAT:
    @staticmethod
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        # first hidden layer: heads concatenated
        attns = [attn_head(inputs, bias_mat=bias_mat, out_sz=hid_units[0],
                           activation=activation, in_drop=ffd_drop,
                           coef_drop=attn_drop, residual=False)
                 for _ in range(n_heads[0])]
        h_1 = tf.concat(attns, axis=-1)

        # further hidden layers: concatenate heads (residual optional)
        for i in range(1, len(hid_units)):
            attns = [attn_head(h_1, bias_mat=bias_mat, out_sz=hid_units[i],
                               activation=activation, in_drop=ffd_drop,
                               coef_drop=attn_drop, residual=residual)
                     for _ in range(n_heads[i])]
            h_1 = tf.concat(attns, axis=-1)

        # output layer: heads produce class scores, AVERAGED
        out = [attn_head(h_1, bias_mat=bias_mat, out_sz=nb_classes,
                         activation=lambda x: x, in_drop=ffd_drop,
                         coef_drop=attn_drop, residual=False)
               for _ in range(n_heads[-1])]
        logits = tf.add_n(out) / float(n_heads[-1])
        return logits


def masked_softmax_cross_entropy(logits, labels, mask):
    labels = tf.cast(labels, tf.float32)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    return tf.reduce_mean(loss * mask)


def masked_sigmoid_cross_entropy(logits, labels, mask):
    labels = tf.cast(labels, tf.float32)
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels),
        axis=1)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    return tf.reduce_mean(loss * mask)


def training(loss, lr, l2_coef):
    vars = tf.trainable_variables()
    skip = {'bias', 'biases', 'gamma', 'b', 'g', 'beta'}
    decay_vars = [v for v in vars if v.op.name.rsplit('/', 1)[-1] not in skip]
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in decay_vars]) * l2_coef if decay_vars else 0.0
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss + lossL2)


# --- transductive Cora: build the neighbor bias mask and train ---
def adj_to_bias(adj, sizes=None, nhood=1):
    if adj.ndim == 2:
        adj = adj[np.newaxis, :, :]
    if sizes is None:
        sizes = [adj.shape[1]] * adj.shape[0]
    nb_graphs = adj.shape[0]
    nb_nodes = adj.shape[1]
    mt = np.empty(adj.shape, dtype=np.float32)
    eye = np.eye(nb_nodes, dtype=np.float32)
    for g in range(nb_graphs):
        mt[g] = eye
        adj_with_self = adj[g].astype(np.float32) + eye
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], adj_with_self)
        mt[g] = (mt[g] > 0.0).astype(np.float32)
        if sizes is not None:
            n = int(sizes[g])
            mt[g, :n, n:] = 0.0
            mt[g, n:, :n] = 0.0
            mt[g, n:, n:] = np.eye(nb_nodes - n, dtype=np.float32)
    return -1e9 * (1.0 - mt)        # 0 on neighbors, -1e9 elsewhere


hid_units = [8]
n_heads = [8, 1]
lr, l2_coef = 0.005, 0.0005         # dropout 0.6 on inputs and attention coefficients at train time
```
