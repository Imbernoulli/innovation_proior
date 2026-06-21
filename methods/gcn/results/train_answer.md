The problem is transductive node classification on a single large graph: we have an adjacency matrix A, node features X, and labels for only a handful of nodes, often twenty per class or fewer. The unlabeled nodes, their features, and their edges are all visible during training; the only held-out information is the labels themselves. The goal is to predict those missing labels by using both the per-node features and the graph structure, and to do it cheaply enough that the cost scales with the number of edges rather than with N squared.

Most existing ideas either keep the graph out of the predictor or pay too much to put it in. Laplacian regularization and label propagation add a smoothness penalty to the loss, which encodes the assumption that every edge means the two endpoints should share a label. That assumption is often wrong, and more importantly the graph never enters the model itself, so feature information cannot flow from labeled nodes to unlabeled ones through the architecture. Spectral graph CNNs do put the graph inside the model, but they need the full eigenvectors of the Laplacian, which is cubic in the number of nodes and stores dense N-by-N matrices. ChebNet removes the eigendecomposition by expanding the filter in Chebyshev polynomials, but its order K simultaneously controls how expressive the filter is and how many hops it reaches, so a high-degree hub can drag in a huge neighborhood and overfit. Attention over the edges is flexible, yet with only a handful of labeled nodes it learns unstable edge weights that vary widely across random seeds. What is needed is a single, graph-conditioned neural network that propagates features along edges in one cheap localized operation per layer, with wide-degree robustness built in through normalization rather than learned attention.

The method I propose is a Graph Convolutional Network, or GCN. It is a neural network f(X, A) whose layers directly condition on the adjacency matrix. Each layer first projects the node features with a learned weight matrix, then mixes each node's transformed features with those of its immediate neighbors using a fixed, symmetrically normalized adjacency operator. The receptive field grows by stacking layers rather than by using a high-order polynomial filter, and a point-wise nonlinearity between layers supplies the expressiveness that a first-order filter alone would lack.

The operator is derived from spectral graph convolution. A convolution on a graph cannot be defined by sliding a filter because there is no canonical "shift by one" when nodes have different degrees. The clean alternative is the convolution theorem: transform the node signal into the orthonormal eigenbasis of the symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}, multiply by a spectral filter, and transform back. Free-form spectral filters are expensive, so the filter is approximated by a Chebyshev polynomial of order one in L. Approximating the largest eigenvalue by 2 and tying the two Chebyshev coefficients gives the operator I + D^{-1/2} A D^{-1/2}, which mixes a node with its symmetrically normalized neighbors. This version still has spectral radius near 2 because the self-loop was added after normalization, which would make deep stacks explode or vanish. The fix is the renormalization trick: add self-loops first, then normalize using the new degrees. Define A_tilde = A + I and D_tilde_ii = sum_j A_tilde_ij, then A_hat = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}. This is the normalized adjacency of the self-looped graph, so its eigenvalues lie in [-1, 1] and it stacks safely to arbitrary depth. The layer rule is H^{(l+1)} = ReLU(A_hat H^{(l)} W^{(l)}), and a two-layer classifier gives Z = softmax(A_hat ReLU(A_hat X W^{(0)}) W^{(1)}).

Training is full-batch over the whole graph, but the cross-entropy loss is averaged only over the labeled nodes using a mask. Because logits are computed for every node, features and gradients still flow across edges to the unlabeled nodes, which is exactly how the model leverages the graph. The features are row-normalized, dropout is 0.5, the first-layer weights get L2 regularization, and optimization uses Adam with learning rate 0.01 and early stopping on validation loss. Read node-wise, the propagation rule h_i^{(l+1)} = sigma(sum_{j in N_i union {i}} (1 / sqrt(d_tilde_i d_tilde_j)) h_j^{(l)} W^{(l)}) is a differentiable, normalized generalization of one step of the Weisfeiler-Lehman graph isomorphism test. The method is therefore scalable, degree-robust, and fully end-to-end, with the graph inside the model rather than tacked onto the loss.

```python
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(adj):
    """Build A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}."""
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat.dot(adj).dot(d_mat).tocoo()

def sparse_to_torch(adj):
    """Convert a scipy COO matrix to a torch sparse tensor."""
    indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    values = torch.from_numpy(adj.data.astype(np.float32))
    shape = torch.Size(adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def preprocess_features(features):
    """Row-normalize feature matrix."""
    rowsum = np.array(features.sum(1)).flatten()
    r_inv = np.power(rowsum, -1.0)
    r_inv[np.isinf(r_inv)] = 0.0
    return sp.diags(r_inv).dot(features)

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = torch.mm(x, self.weight)
        x = torch.sparse.mm(adj, x)
        if self.bias is not None:
            x = x + self.bias
        return x

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNLayer(in_channels, hidden_channels)
        self.conv2 = GCNLayer(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return x

# Example usage with precomputed graph support
adj = sp.load_npz('adj.npz')              # N x N sparse adjacency
features = sp.load_npz('features.npz')    # N x C sparse features
labels = np.load('labels.npy')            # N (integer class indices)
train_mask = np.load('train_mask.npy')    # boolean length N

adj_hat = sparse_to_torch(normalize_adj(adj))
x = torch.from_numpy(preprocess_features(features).todense().astype(np.float32))
y = torch.from_numpy(labels.astype(np.int64))
train_mask = torch.from_numpy(train_mask)

model = GCN(in_channels=x.shape[1], hidden_channels=16, num_classes=int(y.max()) + 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    logits = model(x, adj_hat)
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    l2_reg = 5e-4 * torch.sum(model.conv1.weight ** 2)
    (loss + l2_reg).backward()
    optimizer.step()
```
