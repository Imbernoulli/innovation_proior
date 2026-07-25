I need a neural layer that operates directly on graph-structured data and recovers what makes convolution so effective on grids: shared, learnable parameters applied at every position, with no dependence on a fixed neighborhood size or ordering. On a grid every pixel has four canonical neighbors, but on a graph a node may have two neighbors or two thousand, and there is no "up" or "left" to align against. Spectral graph convolutions define the operation through the graph Fourier transform, but that welds the learned filter to one graph's Laplacian eigenbasis: an O(N^3) eigendecomposition is required, and the model is simply not defined on a graph with a different structure. Chebyshev and GCN-style approximations drive the cost down to O(|E|), yet GCN pins every unweighted edge contribution to the fixed, untrainable coefficient 1/√(d̃_i d̃_j), so it cannot learn that one neighbor matters more than another. GraphSAGE escapes by learning feature-based aggregators, which makes the layer inductive, but it samples a fixed-size neighborhood, weights neighbors uniformly in its mean/GCN aggregators, and its best aggregator is an LSTM that must be fed neighbors in random permutations because the neighbor set has no natural order.

The missing primitive is one that takes a variable-sized, unordered set of neighbors and returns a learned weight for each. That is exactly attention. I propose the Graph Attention Network, or GAT. The core idea is masked self-attention over the local neighborhood: each node attends over its neighbors (including itself via a self-loop), scores each neighbor with a shared additive function of node features, normalizes the scores into a softmax distribution over the neighborhood, and aggregates the neighbors' transformed features with those learned coefficients. Because the scoring and transformation are shared functions of node features, the same parameters apply to any graph, making the layer inductive. Because the coefficients are learned from content rather than fixed by degrees, different neighbors can receive different importances. Because the aggregation is a weighted sum over a set, no neighborhood ordering is ever imposed.

Concretely, given node features h_i, a shared linear transform W is applied to every node. An attention vector a is split into source and destination halves so the score for edge i←j is the broadcast sum a_s^T W h_i + a_d^T W h_j, passed through LeakyReLU with negative slope 0.2. Softmax normalizes these scores over each node's neighborhood, and the updated feature is h_i' = σ(Σ_{j∈N(i)} α_ij W h_j). Multiple independent attention heads are run in parallel: their outputs are concatenated in hidden layers to enrich the representation, and averaged at the output layer so that class logits remain a proper score vector. Dropout on the attention coefficients provides a stochastic neighborhood sample at each training step, which is an effective regularizer in the small-label regimes typical of citation networks. The per-head cost is O(|V|FF' + |E|F') with a sparse edge-list implementation, on par with GCN, with no eigendecomposition or inversion. The working code below takes the simpler dense route instead: it builds the shared linear map and the two attention halves as width-1 convolutions over the node axis, forms the pairwise score matrix by a broadcast sum f_1(i) + f_2(j), and masks it down to each node's neighborhood with an additive bias that is 0 on an edge and -1e9 off it before the softmax — trading an O(N^2) score tensor for simplicity of implementation.

```python
import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d


def graph_layer(seq, out_sz, neigh, activation,
                in_drop=0.0, op_drop=0.0, residual=False):
    """One local graph-attention operator.

    seq: [batch, N, F]; neigh is broadcastable to [batch, N, N],
    with 0 on edges/self-loops and -1e9 off edges.
    """
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # shared linear transform W h_i (1x1 conv = per-node linear map)
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # additive scoring split a = [a1 || a2], implemented as two one-channel scorers
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])          # raw_score[i,j] = f_1[i] + f_2[j]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + neigh)

        if op_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - op_drop)      # coefficient dropout
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)                     # h_i' = sum_j alpha_ij W h_j
        ret = tf.contrib.layers.bias_add(vals)

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)
            else:
                ret = ret + seq

        return activation(ret)


class NodeModel:
    @staticmethod
    def inference(inputs, nb_classes, nb_nodes, training, op_drop, ffd_drop,
                  neigh, hid_units, layer_repeats,
                  activation=tf.nn.elu, residual=False):
        # first hidden layer: heads concatenated
        attns = [graph_layer(inputs, neigh=neigh, out_sz=hid_units[0],
                             activation=activation, in_drop=ffd_drop,
                             op_drop=op_drop, residual=False)
                 for _ in range(layer_repeats[0])]
        h_1 = tf.concat(attns, axis=-1)

        # further hidden layers: concatenate heads (residual optional)
        for i in range(1, len(hid_units)):
            attns = [graph_layer(h_1, neigh=neigh, out_sz=hid_units[i],
                                 activation=activation, in_drop=ffd_drop,
                                 op_drop=op_drop, residual=residual)
                     for _ in range(layer_repeats[i])]
            h_1 = tf.concat(attns, axis=-1)

        # output layer: heads produce class scores, AVERAGED
        out = [graph_layer(h_1, neigh=neigh, out_sz=nb_classes,
                           activation=lambda x: x, in_drop=ffd_drop,
                           op_drop=op_drop, residual=False)
               for _ in range(layer_repeats[-1])]
        logits = tf.add_n(out) / layer_repeats[-1]
        return logits


def masked_softmax_cross_entropy(logits, labels, mask):
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


def masked_accuracy(logits, labels, mask):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def micro_f1(logits, labels, mask):
    predicted = tf.round(tf.nn.sigmoid(logits))
    predicted = tf.cast(predicted, dtype=tf.int32)
    labels = tf.cast(labels, dtype=tf.int32)
    mask = tf.cast(mask, dtype=tf.int32)
    mask = tf.expand_dims(mask, -1)

    tp = tf.count_nonzero(predicted * labels * mask)
    tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
    fp = tf.count_nonzero(predicted * (labels - 1) * mask)
    fn = tf.count_nonzero((predicted - 1) * labels * mask)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall)
    return tf.cast(fmeasure, tf.float32)


def training(loss, lr, l2_coef):
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                       in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    return opt.minimize(loss + lossL2)


# --- transductive Cora: build the neighbor bias mask and train ---
def neighborhood_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)        # 0 on neighbors, -1e9 elsewhere


hid_units = [8]
layer_repeats = [8, 1]
lr, l2_coef = 0.005, 0.0005         # dropout 0.6 on inputs and attention coefficients at train time
```
