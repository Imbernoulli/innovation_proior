# Research question

We are given a single, fixed graph G = (V, E) with N nodes, and we want to assign a class label to every node — but labels exist for only a tiny subset Y_L of the nodes. In the standard citation-network benchmarks the label rate is between roughly 0.3% and 5% of nodes; in the extreme knowledge-graph case it is a single labeled node per class. Each node carries a feature vector — for documents in a citation network this is a sparse bag-of-words — collected into a matrix X ∈ R^{N×C}, and the connectivity is given by an adjacency matrix A ∈ R^{N×N} (binary or weighted, undirected). This is the *transductive* regime: the unlabeled test nodes, together with their features and their edges, are all present at training time; nothing is held out except the labels themselves. There is no second graph to generalize to — only labels to fill in on the one graph in front of us.

The precise problem is to predict the labels of the unlabeled nodes using **both** sources of information at once: the per-node features X *and* the graph structure A. A citation link between two documents, or a typed relation in a knowledge graph, is itself evidence about the label — sometimes evidence that the bag-of-words features do not contain (two papers can cite each other across a subfield boundary while their word distributions look nothing alike). A solution lets label and feature information flow along edges from the few labeled nodes to the many unlabeled ones.

The graphs of interest are large (tens of thousands of nodes, hundreds of thousands of edges, thousands of feature dimensions) and have *wide* degree distributions (citation networks, social networks, knowledge graphs — a few extremely high-degree hubs, a long tail of low-degree nodes). The yardstick is classification accuracy on the unlabeled nodes, with wall-clock training time as a secondary concern.

# Background

The field at this moment splits into several largely separate lines of thinking about "learning on graphs," and the load-bearing technical machinery comes from spectral graph theory.

**The graph as a smoothness prior.** A common way to use a graph in semi-supervised learning is to add a penalty to the loss that forces the predictor f to vary slowly across edges:

  L = L_0 + λ · L_reg,  with  L_reg = (1/2)Σ_{i,j} A_{ij} ‖f(X_i) − f(X_j)‖² = f(X)ᵀ Δ f(X),

where L_0 is the supervised loss on labeled nodes and Δ = D − A is the **unnormalized graph Laplacian**, D being the diagonal degree matrix with D_ii = Σ_j A_{ij}. For a symmetric ordered double sum, the quadratic form satisfies f(X)ᵀΔf(X) = (1/2)Σ_{i,j} A_{ij} ‖f(X_i) − f(X_j)‖², so it is small precisely when adjacent nodes get similar outputs; minimizing it diffuses labels along the edges. The modeling commitment is "an edge means the two endpoints share a label," and f itself is a function of X alone; the graph enters the loss, not the predictor.

**Spectral graph theory and the graph Fourier transform.** The machinery that makes "convolution on a graph" well-defined comes from the **symmetric normalized Laplacian**

  L = I_N − D^{−1/2} A D^{−1/2} = U Λ Uᵀ,

a real symmetric positive-semidefinite matrix. The reason this object, rather than the raw adjacency, sits at the center: classical convolution is defined through *translation* (slide a filter across the signal), and translation is meaningless on an irregular graph — there is no canonical "shift by one" when nodes have different numbers of neighbors. The escape is the convolution theorem: in ordinary signal processing, convolution in the spatial domain equals pointwise multiplication in the Fourier domain. So one can *define* graph convolution as "transform to a frequency domain, multiply by a filter, transform back," sidestepping translation entirely — provided one has a Fourier basis. Because L is real symmetric it diagonalizes with an orthonormal eigenbasis U; those eigenvectors are the graph's Fourier modes (low eigenvalue ↔ eigenvector that varies slowly over the graph; high eigenvalue ↔ rapidly oscillating eigenvector), the eigenvalues play the role of frequencies, the **graph Fourier transform** of a node signal x ∈ R^N is Uᵀx, and the inverse is U(·). The quadratic form xᵀLx measures the signal's roughness. The eigenvalues of the *normalized* Laplacian lie in the bounded interval [0, 2]. Forming U requires an O(N³) eigendecomposition, and applying U or Uᵀ to a vector is an O(N²) dense multiply.

**Approximating functions of the Laplacian without its eigenvectors.** A practical fact from the spectral-graph-wavelets literature (Hammond, Vandergheynst & Gribonval, 2011) is that a function of the Laplacian g(L) can be *applied* without ever forming U. Any smooth spectral function can be expanded in **Chebyshev polynomials** T_k of a rescaled Laplacian, and because (UΛUᵀ)^k = UΛ^kUᵀ, a polynomial in Λ evaluated through U is identical to the same polynomial in L applied directly. Since L is sparse, applying a k-th-order polynomial in L to a vector costs k sparse matrix-vector products — O(k|E|), with no eigendecomposition. Chebyshev polynomials are bounded and orthogonal on [−1, 1] and their three-term recurrence T_k(y) = 2y·T_{k−1}(y) − T_{k−2}(y), with T_0 = 1, T_1 = y, makes the expansion numerically stable for smooth functions — which is why the Laplacian is first rescaled into [−1, 1].

**Depth and stacked layers.** A strand of prevailing wisdom from the broader neural-network field is that stacking many simple layers, each followed by a point-wise nonlinearity, recovers expressive functions a single wide layer cannot — and that very deep models, made trainable with residual connections (He et al., 2015), improve modeling capacity across many domains. A general fact about repeatedly-applied linear operators also sits in the background: a fixed operator applied many times is, spectrally, close to a power iteration, so its spectral radius governs whether signals grow or decay under composition.

# Baselines

The natural points of comparison split into Laplacian-regularization methods, graph-embedding methods, and the recent neural-networks-on-graphs line.

**Label propagation — Zhu, Ghahramani & Lafferty (2003).** Places a Gaussian random field on the graph whose energy is the Laplacian quadratic form; the minimizer is a *harmonic function* in which each unlabeled node's value is the degree-weighted average of its neighbors' values, with labeled nodes clamped, solvable in closed form. Node features X are used only to build edge weights, if at all.

**Manifold regularization — Belkin, Niyogi & Sindhwani (2006).** A general RKHS framework that adds f(X)ᵀΔf(X) to a supervised loss so the learned function is smooth on the data manifold the graph approximates. The graph enters through the loss.

**Deep semi-supervised embedding — Weston et al. (2012).** Attaches a graph-based embedding penalty as an auxiliary loss at hidden layers of a neural net, pulling neighbors' hidden representations together.

**Skip-gram graph embeddings — DeepWalk (Perozzi, Al-Rfou & Skiena, 2014); LINE (Tang et al., 2015); node2vec (Grover & Leskovec, 2016).** Borrow the skip-gram objective from word2vec (Mikolov et al., 2013): sample node sequences by random walks over the graph (LINE and node2vec use more elaborate walk / breadth-first schemes), train embeddings to predict a node's sampled neighborhood, then feed the frozen embeddings to a separate classifier — a multi-step pipeline of walk generation, unsupervised embedding, and supervised classification, with each stage optimized separately. **Planetoid (Yang, Cohen & Salakhutdinov, 2016)** injects label information into the embedding objective.

**Iterative classification (ICA) — Lu & Getoor (2003); Sen et al. (2008).** A relational classifier: train a local logistic-regression model on node features, bootstrap labels for unlabeled nodes, then iterate a relational classifier that uses local features plus an aggregate (count or proportion) of neighbors' *current label estimates*, run to convergence.

**Spectral CNNs on graphs — Bruna et al. (2014).** The first attempt to define a CNN in the graph spectral domain: a filter is a diagonal matrix of free parameters g_θ = diag(θ) in the Fourier basis, and the convolution is g_θ ⋆ x = U·diag(θ)·Uᵀx. The filter has O(N) free parameters (one per eigenvalue), evaluating it requires the full O(N³) eigendecomposition plus O(N²) multiplications by U and Uᵀ, and θ is defined relative to one graph's eigenvectors.

**Fast localized spectral filtering — Defferrard, Bresson & Vandergheynst (2016).** Replaces the free-form spectral filter with a truncated Chebyshev expansion of g_θ(Λ): with a Laplacian rescaled to [−1, 1] as L̃ = (2/λ_max)L − I_N,

  g_{θ'} ⋆ x ≈ Σ_{k=0}^{K} θ'_k T_k(L̃) x ,

evaluated through the Chebyshev recurrence on L̃ directly. Because this is a K-th-order polynomial in L, it depends only on each node's K-hop neighborhood (it is K-localized), costs O(K|E|), and needs no eigendecomposition; the free parameters number K+1 coefficients, and the filter is a *function of the spectrum*. Each filter carries θ'_0 through θ'_K, and K controls both filter expressiveness and receptive-field radius.

**Other neural-networks-on-graphs.** Graph neural networks as recurrent fixed-point models (Gori et al., 2005; Scarselli et al., 2009) repeatedly apply a *contraction map* to convergence; Li et al. (2016) modernize this with gated recurrent updates. Duvenaud et al. (2015) introduce a convolution-like propagation for molecular fingerprints using **degree-specific weight matrices** — a separate learned matrix for each distinct node degree. Atwood & Towsley (2016, diffusion-convolutional networks) report O(N²) complexity. Niepert et al. (2016) linearize local neighborhoods into sequences for an ordinary 1D CNN, choosing a node ordering in preprocessing.

# Evaluation settings

The natural benchmarks for transductive node classification are three citation networks and one knowledge-graph dataset, with a standardized split following Yang, Cohen & Salakhutdinov (2016).

- **Citeseer, Cora, Pubmed (Sen et al., 2008).** Nodes are documents, edges are citation links (treated as undirected, giving a binary symmetric A). Each node has a sparse bag-of-words feature vector and one class label. Cora: 2,708 nodes / 5,429 edges / 7 classes / 1,433 features. Citeseer: 3,327 / 4,732 / 6 / 3,703. Pubmed: 19,717 / 44,338 / 3 / 500.
- **NELL (Carlson et al., 2010; preprocessed as in Yang et al., 2016).** A bipartite graph extracted from a knowledge base, with relation nodes and entity nodes. Directed typed (entity, relation, entity) triples are turned into an undirected graph by assigning a separate relation node per triple side; entities carry sparse features, relation nodes get one-hot identifiers. 65,755 nodes, 266,144 edges, 210 classes — a label rate of a single labeled example per class.

**Protocol.** Use only 20 labeled nodes per class for training on the citation networks (a single labeled node per class on NELL), but all feature vectors. Evaluate prediction accuracy on a fixed test set of 1,000 labeled nodes, using the same splits as Yang et al. (2016), plus a validation set of 500 labeled nodes for early stopping and for choosing hyperparameters (dropout rate, L2 regularization strength, hidden-layer width); validation labels are never used in training. The metric is classification accuracy on the test nodes; secondary reporting is wall-clock training time per epoch, including on simulated random graphs (N nodes, 2N random edges, identity feature matrix) to probe how cost scales with graph size. Optimization is full-batch gradient descent over the whole graph with Adam (learning rate 0.01), weight initialization following Glorot & Bengio (2010), row-normalized input features, and dropout (Srivastava et al., 2014) as the source of training stochasticity. A featureless variant (X = I_N) on a small community graph — Zachary's karate club (Zachary, 1977), 34 nodes, four communities by modularity clustering — is the natural sanity check for whether structure alone organizes representations.

# Code framework

The implementation substrate is a deep-learning framework with automatic differentiation and GPU support (**TensorFlow**, Abadi et al., 2015) together with **SciPy sparse matrices** for the graph, because the data is one large graph and we must never materialize a dense N×N matrix. The available primitives are: loading the graph and features as sparse/dense arrays; converting SciPy sparse matrices to sparse tensors; stacking layers, each a linear map of features followed by a point-wise nonlinearity; the standard training kit (Adam, dropout, Glorot initialization, cross-entropy); and the semi-supervised bookkeeping trick of computing outputs for **all** nodes while supervising only the labeled ones through a mask on the loss. The graph-specific slots are the sparse graph support and the layer that applies it.

**(1) Sparse data loading and generic preprocessing.** The graph arrives as a sparse adjacency matrix, the features as a sparse node-by-feature matrix, the labels as one-hot matrices defined through masks, and three boolean masks splitting nodes into train / validation / test. Feature row-normalization and sparse tuple conversion are ordinary input plumbing.

```python
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

def load_graph_data():
    """Return (A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask):
    A          sparse N x N adjacency (binary/weighted, undirected),
    X          N x C node features (sparse bag-of-words or dense),
    y_*        N x F one-hot labels, zero outside each split,
    *_mask     boolean length-N selectors."""
    A = sp.load_npz(...)            # raw adjacency, untouched
    X = sp.load_npz(...)            # node features
    y_train, y_val, y_test = ...
    train_mask, val_mask, test_mask = ...   # node splits
    return A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """Convert a scipy sparse matrix, or a list of them, to TensorFlow sparse tuples."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        return coords, mx.data, mx.shape
    return [to_tuple(mx) for mx in sparse_mx] if isinstance(sparse_mx, list) else to_tuple(sparse_mx)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    return sparse_to_tuple(sp.diags(r_inv).dot(features))
```

**(2) The sparse graph support.** The raw adjacency is transformed once into the sparse operator used by every layer:

```python
def normalize_adj(adj):
    """Choose and apply the degree normalization for the sparse graph operator."""
    # TODO: graph normalization
    pass

def preprocess_adj(adj):
    """Build the sparse graph support from the raw adjacency."""
    # TODO: graph support construction
    pass
```

**(3) The layer and model stack.** The layer owns one trainable matrix per sparse support, applies dropout, projects features, propagates through the sparse support, sums the terms, and applies a point-wise activation. The model stacks two such layers, with a linear output layer producing logits for every node:

```python
def glorot(shape, name=None):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range,
                                dtype=tf.float32)
    return tf.Variable(initial, name=name)

def sparse_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob + tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    dropped = tf.sparse_retain(x, dropout_mask)
    return dropped * (1. / keep_prob)

def dot(x, y, sparse=False):
    return tf.sparse_tensor_dense_matmul(x, y) if sparse else tf.matmul(x, y)

class GraphLayer:
    def __init__(self, input_dim, output_dim, support, act=tf.nn.relu,
                 dropout=0., sparse_inputs=False, num_features_nonzero=None):
        # TODO: allocate the weights once the graph support is fixed
        pass

    def __call__(self, x):
        # TODO: dropout, feature projection, sparse graph propagation, activation
        pass

class GraphModel:
    def __init__(self, placeholders, input_dim, hidden, num_classes):
        # TODO: stack two GraphLayer instances and expose per-node logits
        pass
```

**(4) Masked loss + full-batch training loop.** Because there is one graph, training is full-batch: compute logits for every node in one pass, but average the cross-entropy over the labeled nodes only, via a mask. This bookkeeping is independent of the graph support chosen above:

```python
def masked_softmax_cross_entropy(logits, labels, mask):
    """Softmax cross-entropy averaged over the labeled nodes only."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)        # renormalize so the average is over masked nodes
    return tf.reduce_mean(loss * mask)

A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_graph_data()
features = preprocess_features(X)
support = [preprocess_adj(A)]
num_supports = len(support)

placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
}

model = GraphModel(placeholders, input_dim=features[2][1],
                   hidden=16, num_classes=y_train.shape[1])
loss = masked_softmax_cross_entropy(model.outputs, placeholders['labels'],
                                    placeholders['labels_mask'])
loss += 5e-4 * tf.add_n([tf.nn.l2_loss(W) for W in model.layers[0].weights])
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
```

This masking is the mechanism that lets a model whose outputs are computed for *all* N nodes train against labels on only the labeled subset, while node information can flow through `GraphLayer` to the unlabeled ones. The model, the masked loss, the optimizer, and dropout stay generic.
