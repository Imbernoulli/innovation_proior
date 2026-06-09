# Context

## Research question

Convolutional neural networks dominate on data living on regular grids — images on a 2D lattice, audio on a 1D lattice. Their power comes from three properties of the grid that the architecture exploits directly: convolutional filters are **local** (a small support, much smaller than the input), **translation-invariant / weight-shared** (the same filter slides everywhere, so the parameter count O(S) depends only on the support size S, not on the input size n), and **compositional** (stacking convolution with pooling builds multi-scale hierarchical features). A single convolutional layer costs O(n) and learns O(S) parameters, and pooling trades spatial resolution for feature resolution cheaply.

A huge amount of important data does *not* live on a grid: users on a social network, genes on a regulatory network, log records on a telecommunication network, words on an embedding, or any dataset turned into a k-nearest-neighbor similarity graph. These are naturally described by graphs — universal encodings of pairwise relationships — and we would like the same local-stationary-compositional feature learning on them.

The obstacle is that convolution and pooling are *defined* only on regular grids. On a general graph there is no canonical way to "shift a filter by one node": neighborhoods have different sizes, and there is no consistent ordering of a node's neighbors, so a spatial filter cannot share weights in a well-defined way. The central problem is therefore to define convolutional filters on a graph that are simultaneously:

- **localized** — supported within a small, controllable number of hops from a center vertex,
- **cheap to learn** — O(K) parameters per filter, independent of the graph size n, exactly as on a grid,
- **cheap to evaluate** — ideally linear in the number of edges, with no dense n×n operations,

together with a pooling/coarsening mechanism that lets such layers be stacked into a deep, multi-scale network.

## Background

**Spectral graph theory and the graph Laplacian.** A weighted undirected graph is G=(V,E,W) with |V|=n vertices and a symmetric weighted adjacency matrix W∈R^{n×n}. A signal on the graph is a vector x∈R^n (one value per vertex). The central operator is the graph Laplacian: combinatorial L = D − W with the degree matrix D=diag(Σ_j W_{ij}), or normalized L = I_n − D^{-1/2} W D^{-1/2} (Chung 1997). L is real, symmetric, and positive semidefinite, so it has a complete set of orthonormal eigenvectors {u_l}_{l=0}^{n-1} with ordered nonnegative eigenvalues 0 ≤ λ_0 ≤ … ≤ λ_{n-1}. Writing U=[u_0,…,u_{n-1}] and Λ=diag(λ_0,…,λ_{n-1}), the Laplacian diagonalizes as L = UΛU^T.

The eigenvectors play the role of Fourier modes and the eigenvalues the role of frequencies. The reason is the Dirichlet energy: for the normalized Laplacian, x^T L x = ½ Σ_{ij} W_{ij}(x_i/√d_i − x_j/√d_j)^2 measures how much a signal varies across edges. Eigenvectors with small λ vary slowly over the graph (low frequency); eigenvectors with large λ oscillate (high frequency). On the special case of a ring graph the Laplacian is exactly the discrete second-difference operator and its eigenvectors are the ordinary sinusoids, recovering classical Fourier analysis — so the Laplacian eigenbasis is the natural generalization of Fourier to a graph.

**Graph signal processing (GSP).** A field (surveyed by Shuman et al. 2013) bridging signal processing and spectral graph theory, aiming to lift grid operations — convolution, translation, filtering, downsampling — to irregular graph domains. Several grid notions have no direct analogue and require new definitions while keeping the original intuition.

**The graph Fourier transform.** Using the eigenbasis, the graph Fourier transform of x is x̂ = U^T x and the inverse is x = U x̂ (well-defined because U^T U = I). This is the device through which filtering can be defined on a graph at all.

**Graph wavelets via spectral graph theory (Hammond, Vandergheynst & Gribonval 2011).** This line constructs wavelet operators on graphs by defining a spectral kernel g(λ) and applying it as U g(Λ) U^T, the general device for turning a frequency-domain kernel into a vertex-domain operator.

**Chebyshev polynomials.** T_k(y) = 2y T_{k-1}(y) − T_{k-2}(y), with T_0(y)=1 and T_1(y)=y, computed by this stable three-term recurrence; they are an orthogonal basis of L^2([-1,1], dy/√(1−y²)). T_k is a polynomial of degree exactly k. Because their domain is [−1,1], any spectrum must be affinely rescaled into [−1,1] before they are used.

**Spectrum bound for the normalized Laplacian.** The eigenvalues of the normalized Laplacian satisfy 0 ≤ λ_l ≤ 2 (Chung 1997), so λ_max ≤ 2; this upper bound is known a priori, without computing the spectrum.

**Classical CNN building blocks.** Convolution layers extract local stationary features with weight sharing (LeCun et al. 1998); pooling/subsampling on a grid halves resolution and gives multi-scale composition (LeCun, Bengio & Hinton 2015). The convolution theorem (Mallat) characterizes convolutions as exactly the linear operators that diagonalize in the Fourier basis.

**Graph clustering for coarsening.** Multi-scale pooling on a graph needs meaningful neighborhoods, i.e. a clustering that groups similar vertices and can be applied repeatedly to produce coarser graphs. Exact graph partitioning is NP-hard (Bui & Jones 1992), so fast multilevel heuristics are used. Graclus (Dhillon, Guan & Kulis 2007), built on Metis (Karypis & Kumar 1998), greedily matches each unmarked vertex i with the unmarked neighbor j that maximizes the local normalized cut W_{ij}(1/d_i + 1/d_j), merges them, and roughly halves the vertex count per level — giving a controllable hierarchy of coarser graphs.

## Baselines

**First spectral graph CNN (Bruna, Zaremba, Szlam & LeCun 2013, "Spectral Networks").** The pioneering spectral construction. A filter is parametrized in the Fourier domain by a smooth basis:

```
g_θ(Λ) = B θ ,
```

where B∈R^{n×K} is a cubic B-spline basis and θ∈R^K are control points; filtering is y = U g_θ(Λ) U^T x. Smoothness of the filter in the Fourier domain induces *some* spatial decay, giving approximate localization. Two limitations remain open. (i) **No precise control of support.** Spatial localization is obtained only indirectly via Fourier-domain smoothness, so the kernel's local support cannot be set exactly. (ii) **Cost.** Evaluating the filter requires the full Fourier basis U: an eigendecomposition of L at O(n^3) (and storage of the n×n basis), and — the dominant recurring cost — two dense multiplications by U (forward and inverse transform) at O(n^2) per forward/backward pass. This does not scale, a well-known bottleneck of this approach.

**Learned-graph spectral CNN (Henaff, Bruna & LeCun 2015).** Extends the above to also learn the graph structure from data and applies it to image, text and bioinformatics tasks; it inherits the same O(n^2) Fourier-basis bottleneck and raises three concerns for the field: making complexity linear in n, the importance of input-graph quality, and whether the locality/stationarity assumptions actually hold on real data.

**Non-parametric spectral filter.** The most direct spectral filter is fully free in the Fourier domain:

```
g_θ(Λ) = diag(θ) ,    θ ∈ R^n .
```

Each frequency gets its own learnable gain. It is maximally flexible but (i) **not localized** in the vertex domain (a generic spectral multiplier has global spatial support), (ii) has **O(n) parameters** that grow with the graph, and (iii) again needs the full eigendecomposition and two O(n^2) multiplications by U. It serves as the "no structure imposed" reference point.

**Graph Neural Network (Scarselli et al. 2009; simplified as Gated GNN, Li et al. 2015).** Embeds each node via a transition function iterated to a fixed point / over steps and reads out node states. Setting the transition function to a simple diffusion s = Wx and the output function to θ(s − Dx) + x gives x − θLx for the combinatorial Laplacian L = D − W; the sign can be absorbed into the learned scalar, so one step is a Laplacian diffusion plus a node-local map, and a K-step version stacks K such diffusions. This is a spatial/message-passing route to localized graph operators.

**Spatial constructions.** Local-receptive-field methods (Gregor & LeCun 2010; Coates & Ng 2011) group features by similarity to cut connections, achieving locality but **no weight sharing / stationarity**. Geodesic CNNs (Masci et al. 2015) define convolution on 3D meshes via geodesic polar coordinates, but only for smooth low-dimensional manifolds, not general graphs. Inducing weight sharing in any spatial construction is hard because it requires selecting and ordering neighborhoods, and a problem-specific ordering is generally missing.

## Evaluation settings

**MNIST as a sanity check.** 70,000 handwritten digits on a 28×28 grid (LeCun et al. 1998). To test a graph model on a case where the right answer is known, the 2D grid is turned into a graph: an 8-nearest-neighbor graph of the 784 pixel coordinates, with edge weights W_{ij} = exp(−‖z_i − z_j‖_2^2 / σ^2) where z_i is the 2D coordinate of pixel i, giving |E|=3198 weighted edges. A graph model should approximately recover a classical CNN here. Architectures are written as GCk (graph conv with k feature maps), Pk (pooling/stride k), FCk (fully connected, k units); each conv/FC layer is followed by ReLU, the last layer is softmax. Loss is cross-entropy with ℓ2 regularization on FC weights; optimization is SGD with momentum, learning-rate decay, dropout; minibatch size 100; 20 epochs.

**20NEWS for unstructured data.** 18,846 documents (11,314 train / 7,532 test) in 20 classes (Joachims 1996). The 10,000 most frequent words are kept; each document is a normalized bag-of-words vector. A feature graph over words is built as a 16-nearest-neighbor graph with the same Gaussian weighting, where z_i is the word2vec embedding (Mikolov et al. 2013) of word i, giving n=10,000 vertices and |E|=132,834 edges. Trained with Adam; minibatch 100; 20 epochs.

**Metrics and protocol.** Classification accuracy is the primary metric. Compute cost is measured as wall-clock time to process a minibatch (S=100) on CPU and GPU, and as runtime scaling with the number of vertices n. Comparison points include the non-parametric and spline spectral filters above, plain fully connected nets, and a classical CNN on the grid.

## Code framework

Existing building blocks include sparse linear algebra for the adjacency/Laplacian, an eigendecomposition routine, graph clustering for coarser vertex sets, and a standard deep-learning training loop (minibatch SGD/Adam, cross-entropy loss, ReLU, dropout, softmax). The graph-specific filtering and pooling operations remain as stubs.

```python
import numpy as np
import scipy.sparse

# Spectral graph primitives.

def laplacian(W, normalized=True):
    """Graph Laplacian of a sparse weighted adjacency matrix W.
    Combinatorial L = D - W ; normalized L = I - D^{-1/2} W D^{-1/2}."""
    d = W.sum(axis=0)
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        return D - W
    d = 1 / np.sqrt(d + np.spacing(np.array(0, W.dtype)))
    D = scipy.sparse.diags(d.A.squeeze(), 0)
    I = scipy.sparse.identity(d.size, dtype=W.dtype)
    return I - D * W * D

def fourier(L):
    """Eigendecomposition L = U diag(lamb) U^T. O(n^3); kept for reference filters."""
    lamb, U = np.linalg.eigh(L.toarray())
    return lamb, U

# Multilevel graph clustering.

def coarsen(A, levels):
    """Greedy multilevel matching (Graclus/Metis): roughly halve |V| per level.
    Returns the hierarchy of coarser graphs and the parent map between levels."""
    pass  # TODO: multilevel matching

# Graph signal operators.

def graph_filter(L, x, support_size, output_features):
    """Localized, cheap-to-evaluate convolutional filter of a graph signal x.
    The support size should control the hop radius; the implementation should
    avoid dense n-by-n Fourier-basis operations."""
    pass  # TODO

# Standard neural-network harness.

class GraphConvNet:
    """Stack of (graph_filter -> bias+ReLU -> graph pool) blocks, then FC,
    then softmax. Trained with minibatch SGD/Adam + cross-entropy + l2 on FC."""

    def conv_layer(self, x, L, output_features, support_size):
        x = graph_filter(L, x, support_size, output_features)  # TODO
        return relu(x + self.bias)

    def pool(self, x, pooling_map):
        pass  # TODO: graph pooling from the coarsening hierarchy

    def fc(self, x, Mout, relu=True):
        x = x @ self.W + self.b
        return relu_(x) if relu else x

    def inference(self, x):
        for L, output_features, support_size, pooling_map in self.layers:
            x = self.pool(self.conv_layer(x, L, output_features, support_size), pooling_map)
        x = flatten(x)
        for Mout in self.fc_sizes[:-1]:
            x = dropout(self.fc(x, Mout))
        return self.fc(x, self.fc_sizes[-1], relu=False)  # logits -> softmax
```
