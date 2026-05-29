# Graph Convolutional Networks (GCN), distilled

A Graph Convolutional Network is a neural network f(X, A) for semi-supervised node classification on a single fixed graph. Instead of using the graph as a smoothness penalty in the loss, it conditions the model directly on the adjacency matrix, propagating node features along edges with one cheap, localized operation per layer. Receptive field and expressiveness come from stacking layers, not from a high-order filter.

## Problem

Transductive node classification: one graph with adjacency A ∈ R^{N×N} (binary or weighted, undirected), node features X ∈ R^{N×C}, and labels for only a small subset Y_L of nodes. All nodes, features, and edges are visible at training time; only the labels of the test nodes are held out. The goal is to label the unlabeled nodes using both X and the graph structure, learning from very few labels, with cost linear in the number of edges |E|.

## Key idea and why each step

A convolution can't be defined on a graph by sliding a filter — *translation* is undefined when nodes have different numbers of neighbors. The translation-free definition is the convolution theorem: convolution = pointwise multiplication in a Fourier domain. The graph Fourier basis is the orthonormal eigenbasis of the **symmetric normalized Laplacian** L = I_N − D^{−1/2}AD^{−1/2} = UΛUᵀ (chosen because it is symmetric PSD, hence has an orthonormal eigenbasis and a bounded spectrum λ ∈ [0, 2]).

- **Spectral convolution (Bruna et al., 2014):** g_θ ⋆ x = U·diag(θ)·Uᵀ x. *Rejected:* O(N) free parameters, not localized, needs the O(N³) eigendecomposition plus O(N²) multiplies by U, and θ does not transfer across graphs.
- **Chebyshev / ChebNet (Defferrard et al., 2016; Hammond et al., 2011):** write the filter as a function of the spectrum and expand it in Chebyshev polynomials, g_{θ'} ⋆ x ≈ Σ_{k=0}^{K} θ'_k T_k(L̃) x, with L̃ = (2/λ_max)L − I_N rescaled into [−1, 1] (where the Chebyshev recurrence is bounded and stable). Using (UΛUᵀ)^k = UΛ^kUᵀ the eigenvectors telescope away, so the filter is applied via the recurrence T_k(L̃)x = 2L̃ T_{k−1}(L̃)x − T_{k−2}(L̃)x directly on the sparse L̃ — K-localized (K-hop), O(K|E|), no eigendecomposition, K parameters, transferable.
- **Truncate to K=1.** In ChebNet, K controls *both* filter expressiveness and receptive-field radius. Untangle them: let depth supply the hops (stacking k one-hop layers reaches k hops) and the point-wise nonlinearities supply the expressiveness; then each layer needs only a first-order filter. Fewer parameters per layer also fights overfitting the large 1-hop neighborhoods of high-degree hubs.
- **Approximate λ_max ≈ 2.** Since λ ∈ [0, 2], drop the per-graph eigenvalue computation; trainable weights absorb the scale. Then L̃ = L − I_N = −D^{−1/2}AD^{−1/2}, giving g ⋆ x ≈ θ'_0 x − θ'_1 D^{−1/2}AD^{−1/2} x.
- **Symmetric, not random-walk, normalization.** D^{−1}A is non-symmetric (breaks the spectral framework) and is literally neighbor-averaging; symmetric D^{−1/2}AD^{−1/2} keeps the orthonormal eigenbasis and weights edge (i,j) by 1/√(d_i d_j), which is more than an average and down-weights hub neighbors. Normalizing at all is required because raw A rescales feature magnitudes by degree and destabilizes propagation.
- **Tie θ = θ'_0 = −θ'_1.** One parameter, fewer matmuls, more regularization: g ⋆ x ≈ θ (I_N + D^{−1/2}AD^{−1/2}) x.
- **Renormalization trick.** I_N + D^{−1/2}AD^{−1/2} has eigenvalues in [0, 2] (the self-loop was added *after* normalizing, so it's over-weighted), and stacking an operator of spectral radius ≈ 2 explodes/vanishes signals with depth. Fold the self-loop in *before* normalizing: Â = D̃^{−1/2}ÃD̃^{−1/2} with Ã = A + I_N, D̃_ii = Σ_j Ã_ij. This is the normalized adjacency of the self-looped graph, so its spectral radius is 1 and it stacks safely.

## Final model

Multi-channel propagation (Θ ∈ R^{C×F}):

  Z = Â X Θ,  Â = D̃^{−1/2} Ã D̃^{−1/2},  cost O(|E| F C).

Layer-wise rule (σ = ReLU, H^{(0)} = X) — this is the K=1 Chebyshev filter, stacked:

  H^{(l+1)} = σ( Â H^{(l)} W^{(l)} ).

Two-layer classifier (Â precomputed once):

  Z = f(X, A) = softmax( Â · ReLU( Â X W^{(0)} ) · W^{(1)} ),

with W^{(0)} ∈ R^{C×H}, W^{(1)} ∈ R^{H×F}, softmax row-wise. Loss is cross-entropy over labeled nodes only, while outputs are computed for all N nodes so features and gradients still flow across edges to the unlabeled nodes:

  L = − Σ_{l∈Y_L} Σ_{f=1}^{F} Y_{lf} ln Z_{lf}.

Training: full-batch gradient descent over the whole graph with Adam (lr 0.01), dropout 0.5, L2 on the first layer (5·10⁻⁴), 16 hidden units, Glorot init, row-normalized features, early stopping on validation loss (window 10), max 200 epochs. Sparse Â keeps memory O(|E|).

**Node-wise reading (the Weisfeiler–Lehman connection).** With c_ij = √(d̃_i d̃_j), the rule h_i^{(l+1)} = σ(Σ_{j∈N_i∪{i}} (1/c_ij) h_j^{(l)} W^{(l)}) is exactly Â-propagation. This is a differentiable, parameterized, normalized generalization of one step of 1-dimensional Weisfeiler–Lehman relabeling: replace the hash of a node's current color and neighboring-color multiset with σ(·W), and replace the unweighted aggregate with 1/c_ij. Because the aggregation itself encodes graph structure, even an *untrained* random-weight network of these layers (e.g. featureless X = I_N on a small community graph) yields structurally organized node embeddings; training only sharpens them.

## Code

Faithful to the canonical TensorFlow implementation (tkipf/gcn); the PyTorch layer (pygcn) is included for clarity.

Preprocessing — build Â once (the renormalization trick), all sparse:

```python
import numpy as np
import scipy.sparse as sp

def sparse_to_tuple(sparse_mx):
    """Convert a scipy sparse matrix, or a list of them, to TensorFlow sparse tuples."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        return coords, mx.data, mx.shape
    return [to_tuple(mx) for mx in sparse_mx] if isinstance(sparse_mx, list) else to_tuple(sparse_mx)

def normalize_adj(adj):
    """Symmetric normalization  D^{-1/2} A D^{-1/2} (not random-walk D^{-1}A:
    stays symmetric/in the spectral frame, and is more than a neighbor-average)."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Ahat = D~^{-1/2} (A + I) D~^{-1/2}  -- self-loops added BEFORE normalizing,
    so the spectral radius is 1 (raw I + D^{-1/2}AD^{-1/2} sits in [0,2] and
    destabilizes deep stacks)."""
    return sparse_to_tuple(normalize_adj(adj + sp.eye(adj.shape[0])))

def chebyshev_polynomials(adj, K):
    """Richer ChebNet support that the K=1 model collapses: L~ in [-1,1] then
    the stable recurrence T_k = 2 L~ T_{k-1} - T_{k-2}."""
    from scipy.sparse.linalg import eigsh
    L = sp.eye(adj.shape[0]) - normalize_adj(adj)
    lambda_max = eigsh(L, 1, which='LM')[0][0]          # one top eigenvalue only
    L_tilde = (2. / lambda_max) * L - sp.eye(adj.shape[0])
    Tk = [sp.eye(adj.shape[0]), L_tilde]                 # T_0 = I, T_1 = L~
    for _ in range(2, K + 1):
        Tk.append(2 * L_tilde.dot(Tk[-1]) - Tk[-2])
    return sparse_to_tuple(Tk)
```

Layer (TensorFlow) — Â (X W) as a sparse-dense matmul, generalized to a list of `support` operators (length 1 for the renormalization-trick model; T_0(L̃)…T_K(L̃) for the Chebyshev variant, summed):

```python
import numpy as np
import tensorflow as tf

def glorot(shape, name=None):
    """Glorot & Bengio uniform initialization, matching tkipf/gcn."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range,
                                dtype=tf.float32)
    return tf.Variable(initial, name=name)

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for TensorFlow SparseTensor inputs."""
    random_tensor = keep_prob + tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    dropped = tf.sparse_retain(x, dropout_mask)
    return dropped * (1. / keep_prob)

def dot(x, y, sparse=False):
    """tf.matmul that dispatches to the sparse kernel."""
    return tf.sparse_tensor_dense_matmul(x, y) if sparse else tf.matmul(x, y)

class GraphConvolution:
    def __init__(self, input_dim, output_dim, support, act=tf.nn.relu,
                 dropout=0., sparse_inputs=False, num_features_nonzero=None):
        self.support = support          # list of sparse operators; [Ahat] for GCN
        self.act, self.dropout = act, dropout
        self.sparse_inputs = sparse_inputs
        self.num_features_nonzero = num_features_nonzero
        self.weights = [glorot([input_dim, output_dim]) for _ in support]

    def __call__(self, x):
        x = (sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero) if self.sparse_inputs
             else tf.nn.dropout(x, 1 - self.dropout))
        out = []
        for s, W in zip(self.support, self.weights):
            xw = dot(x, W, sparse=self.sparse_inputs)   # X W
            out.append(dot(s, xw, sparse=True))         # Ahat (X W)
        return self.act(tf.add_n(out))
```

Two-layer model:

```python
class GCN:
    def __init__(self, placeholders, input_dim, num_classes,
                 hidden=16):
        support = placeholders['support']               # [Ahat]
        dropout = placeholders['dropout']
        self.gc1 = GraphConvolution(input_dim, hidden, support,
                                    act=tf.nn.relu, dropout=dropout,
                                    sparse_inputs=True,
                                    num_features_nonzero=placeholders['num_features_nonzero'])
        self.gc2 = GraphConvolution(hidden, num_classes, support,
                                    act=lambda z: z, dropout=dropout)
        h = self.gc1(placeholders['features'])          # ReLU( Ahat X W0 )
        self.outputs = self.gc2(h)                      # Ahat h W1  -> logits
```

Masked loss + full-batch training:

```python
def masked_softmax_cross_entropy(logits, labels, mask):
    """Softmax cross-entropy averaged over labeled nodes only."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    return tf.reduce_mean(loss * mask)

model = GCN(placeholders, input_dim=num_features, num_classes=num_classes,
            hidden=16)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)   # full-batch, lr 0.01
loss = masked_softmax_cross_entropy(model.outputs,
                                    placeholders['labels'],
                                    placeholders['labels_mask'])
loss += 5e-4 * tf.add_n([tf.nn.l2_loss(W) for W in model.gc1.weights])
train_op = optimizer.minimize(loss)

feed_dict = {
    placeholders['features']: features,                  # sparse row-normalized X
    placeholders['labels']: y_train,
    placeholders['labels_mask']: train_mask,
    placeholders['num_features_nonzero']: features[1].shape,
    placeholders['dropout']: 0.5,
}
feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
_, train_loss = sess.run([train_op, loss], feed_dict=feed_dict)
# Repeat the full-batch step for up to 200 epochs and early-stop on validation loss.
```

PyTorch layer (mirrors pygcn) — the same Â(XW) in two lines:

```python
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """One GCN layer: Ahat (X W)."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):              # adj = sparse Ahat
        support = torch.mm(x, self.weight)  # X W
        output = torch.spmm(adj, support)   # Ahat (X W)
        return output + self.bias if self.bias is not None else output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))                           # Ahat X W0 -> ReLU
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)                                   # Ahat H W1
        return F.log_softmax(x, dim=1)
```
