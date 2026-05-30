# ChebNet

## Problem

Generalize CNNs from regular grids (images, audio) to arbitrary graphs (social networks, brain connectomes, word-embedding/k-NN graphs), keeping the three properties that make CNNs work: **localized** filters (small, controllable support), **weight-shared / stationary** filters (O(K) parameters independent of graph size n), and **compositional** stacking with pooling. The blocker is that convolution and pooling have no canonical definition on an irregular graph — there is no spatial translation operator to slide and share a filter.

## Key idea

Define convolution **spectrally** through the graph Laplacian's eigenbasis, then make the spectral filter a **Chebyshev polynomial of the Laplacian** so it is exactly localized and can be applied by sparse matrix–vector products without ever computing the eigendecomposition.

1. **Graph Fourier transform.** For the normalized Laplacian L = I − D^{−1/2}WD^{−1/2} (real, symmetric, PSD), diagonalize L = UΛU^T. The eigenvectors U are the graph Fourier modes, the eigenvalues Λ the frequencies. Transform: x̂ = U^Tx, inverse x = Ux̂.

2. **Spectral filtering.** A filter g_θ acts as y = g_θ(L)x = U g_θ(Λ) U^T x. The fully free choice g_θ(Λ) = diag(θ) (θ∈R^n) is **not localized**, costs an O(n³) eigendecomposition plus two dense O(n²) multiplications by U per pass, and has O(n) parameters.

3. **Polynomial ⇒ localization.** Restrict g_θ(L) = Σ_{k=0}^{K−1} θ_k L^k. Since (L^k)_{ij} = 0 whenever the shortest-path distance d_G(i,j) > k (no length-k walk exists), a degree-(K−1) polynomial of L is exactly **(K−1)-hop localized** with only **K parameters**.

4. **Chebyshev recurrence ⇒ eigendecomposition-free, O(K|E|).** Apply the polynomial directly through L via the stable Chebyshev basis. Rescale eigenvalues to [−1,1] with Λ̃ = 2Λ/λ_max − I, L̃ = 2L/λ_max − I (for the normalized Laplacian λ_max ≤ 2, so take λ_max = 2 ⇒ L̃ = L − I). With T_k(y) = 2yT_{k−1}(y) − T_{k−2}(y), T_0=1, T_1=y:

   ```
   g_θ(L) x = Σ_{k=0}^{K−1} θ_k T_k(L̃) x ,
   x̄_0 = x ,  x̄_1 = L̃ x ,  x̄_k = 2 L̃ x̄_{k−1} − x̄_{k−2} ,
   y = [x̄_0, …, x̄_{K−1}] θ .
   ```

   Each step is one sparse mat-vec by L̃ (O(|E|)), so filtering is **O(K|E|)** — no Fourier basis, no eigendecomposition, no n×n storage.

5. **Filter bank / layer.** y_{s,j} = Σ_{i=1}^{F_in} g_{θ_{i,j}}(L) x_{s,i}, with F_in×F_out trainable coefficient vectors θ_{i,j}∈R^K shared across all vertices (stationarity). Cost O(K|E| F_in F_out S). The Chebyshev basis is built once per input and reused; backprop is the same cost as the forward pass.

6. **Coarsening + fast pooling.** Coarsen the graph with the greedy multilevel **Graclus** matching (match unmarked vertex i with the unmarked neighbor j maximizing the local normalized cut W_{ij}(1/d_i + 1/d_j); sum merged weights), roughly halving |V| per level. Arrange the hierarchy as a **balanced binary tree**: add inert *fake* nodes (neutral value, disconnected, so filtering ignores them) so every node has exactly two children; order the coarsest level and propagate (node k → children 2k, 2k+1). The finest level is then ordered so merged vertices are adjacent, making graph pooling a plain **1D max-pool** of size p (power of 2), GPU-friendly.

## Final architecture

Stack blocks of `graph Chebyshev conv → bias + ReLU → graph max-pool`, then flatten, fully connected layers, softmax. Loss = cross-entropy + ℓ2 on FC weights; optimized with SGD+momentum or Adam.

## Code

Grounded in the TensorFlow `mdeff/cnn_graph` implementation (`lib/graph.py` and `lib/models.py`).

```python
import numpy as np
import scipy.sparse
import tensorflow as tf

# ---- graph.py ------------------------------------------------------------

def laplacian(W, normalized=True):
    """Graph Laplacian. Normalized: L = I - D^{-1/2} W D^{-1/2} (symmetric, PSD)."""
    d = W.sum(axis=0)
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        return D - W
    d = 1 / np.sqrt(d + np.spacing(np.array(0, W.dtype)))
    D = scipy.sparse.diags(d.A.squeeze(), 0)
    I = scipy.sparse.identity(d.size, dtype=W.dtype)
    return I - D * W * D

def rescale_L(L, lmax=2):
    """L_tilde = (2/lmax) L - I, mapping the spectrum into [-1, 1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L = L * (2 / lmax)
    L = L - I
    return L

def chebyshev(L, X, K):
    """Stack T_k(L_tilde) X, k = 0..K-1, by the three-term recurrence. O(K|E|N)."""
    M, N = X.shape
    Xt = np.empty((K, M, N), L.dtype)
    Xt[0, ...] = X                                    # T_0 X = X
    if K > 1:
        Xt[1, ...] = L.dot(X)                         # T_1 X = L_tilde X
    for k in range(2, K):
        Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
    return Xt

# ---- models.py : the Chebyshev graph convolutional layer -----------------

class GraphConvNet:

    def chebyshev5(self, x, L, Fout, K):
        """Localized spectral filter, Fin -> Fout. x: N x M x Fin."""
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale the Laplacian and store it as a sparse tensor.
        L = scipy.sparse.csr_matrix(L)
        L = rescale_L(L, lmax=2)                       # L_tilde = L - I
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Chebyshev basis via the recurrence, directly on the sparse Laplacian.
        x0 = tf.reshape(tf.transpose(x, perm=[1, 2, 0]), [M, Fin * N])   # M x Fin*N
        x = tf.expand_dims(x0, 0)                                        # 1 x M x Fin*N
        def concat(x, x_):
            return tf.concat([x, tf.expand_dims(x_, 0)], axis=0)
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)                   # x_bar_1
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0          # x_bar_k
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])
        x = tf.transpose(x, perm=[3, 1, 2, 0])                          # N x M x Fin x K
        x = tf.reshape(x, [N * M, Fin * K])
        # Learned Chebyshev coefficients (theta), shared across vertices.
        W = self._weight_variable([Fin * K, Fout], regularization=False)
        x = tf.matmul(x, W)                                             # N*M x Fout
        return tf.reshape(x, [N, M, Fout])

    def b1relu(self, x):
        """One bias per feature map, then ReLU."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def mpool1(self, x, p):
        """1D max-pool of size p on the binary-tree-reordered graph signal."""
        if p > 1:
            x = tf.expand_dims(x, 3)                                    # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1, p, 1, 1],
                               strides=[1, p, 1, 1], padding='SAME')
            return tf.squeeze(x, [3])                                   # N x M/p x F
        return x

    def fc(self, x, Mout, relu=True):
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _inference(self, x, dropout):
        x = tf.expand_dims(x, 2)                                        # N x M x F=1
        for i in range(len(self.p)):                                   # conv blocks
            with tf.variable_scope('conv{}'.format(i + 1)):
                x = self.chebyshev5(x, self.L[i], self.F[i], self.K[i])
                x = self.b1relu(x)
                x = self.mpool1(x, self.p[i])
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M * F)])                        # flatten
        for M in self.M[:-1]:                                          # FC hidden
            x = self.fc(x, M)
            x = tf.nn.dropout(x, dropout)
        return self.fc(x, self.M[-1], relu=False)                      # logits -> softmax
```
