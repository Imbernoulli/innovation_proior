OK, let me think this through from scratch. What I actually want is a CNN, but on a graph instead of a grid. Why do I even want that? Because the thing CNNs do — slide a small filter everywhere, share its weights, stack with pooling — is exactly the trick that makes them cheap and powerful on images and audio: a convolutional layer costs O(n), learns only O(S) parameters where S is the tiny support of the filter (independent of how big the input is), and the locality + weight-sharing is what lets it pick out the same local pattern wherever it appears. Tons of data I care about isn't on a grid though — people on a social network, genes on a regulatory network, words sitting on an embedding, or just any dataset I've turned into a k-nearest-neighbor similarity graph. I'd love the same local, stationary, compositional feature extraction there. So the question is really narrow: how do I define a convolutional filter on a graph that is localized, has O(K) parameters, and is cheap to apply?

The honest obstacle is that "convolution" on a grid leans on a structure a graph doesn't have. On an image I can translate a filter by one pixel — there's a canonical shift. On a graph, what does "shift the filter by one vertex" even mean? Vertices have different numbers of neighbors, and there's no consistent way to say which neighbor is "to the left." So if I try to build the filter spatially — just learn a weight per neighbor and slide it — I immediately hit the wall that I can't match up neighborhoods across vertices to share weights. The locality I'd get for free (a finite kernel only touches nearby vertices), but the weight-sharing, the thing that makes it a *convolution* and not a pile of independent local maps, has no well-defined meaning. Local-receptive-field models can group similar coordinates and cut connections, but they do not share a filter across the domain. Mesh methods can use geodesic polar coordinates, but that relies on a smooth low-dimensional surface with a local coordinate system. A general graph has neither. The spatial route keeps forcing an ordering or coordinate choice that the graph itself does not provide.

Message-passing graph neural networks are closer, because they already move information along edges. If I strip a node-state model down to a linear diffusion, the transition can be s = Wx. A pointwise output such as θ(s − Dx) + x is x − θLx when L = D − W; the sign is not important because θ is learned, but the structure is important: one step gives me a node-local combination of x and Lx. If I stack several such diffusion steps, I can build a polynomial in L. So this route is telling me something useful — Laplacian polynomials are the right algebra for local graph operators — but it is still a recurrent diffusion view, not yet a clean convolutional filter bank with a controllable support, a small coefficient vector, and the same filter shared everywhere.

So let me try the other door. There's a classical fact — the convolution theorem — that convolutions are precisely the linear operators that become diagonal in the Fourier basis. On a grid, convolving is the same as: Fourier-transform, multiply pointwise by the filter's spectrum, transform back. That's appealing because "Fourier transform, multiply, transform back" doesn't need any notion of spatial shift — it only needs a Fourier basis. Do I have a Fourier basis on a graph? I think I do, and it comes from the Laplacian.

Let me set this up properly. Take an undirected weighted graph with adjacency W and degree matrix D = diag(Σ_j W_{ij}). The Laplacian is L = D − W, or in normalized form L = I − D^{-1/2} W D^{-1/2}. I'll lean on the normalized one. The key structural fact: L is real, symmetric, positive semidefinite. Symmetric ⇒ a complete orthonormal eigenbasis; PSD ⇒ nonnegative eigenvalues. So L = U Λ U^T with U = [u_0,…,u_{n-1}] orthonormal (U^T U = I) and Λ = diag(λ_0,…,λ_{n-1}), 0 ≤ λ_0 ≤ … ≤ λ_{n-1}.

Why should I believe these eigenvectors are "Fourier modes" and the λ's are "frequencies"? Look at the quadratic form. For the normalized Laplacian, x^T L x = ½ Σ_{ij} W_{ij} (x_i/√d_i − x_j/√d_j)^2 — it's literally a sum over edges of squared differences, i.e. how much the signal disagrees across connected vertices. So x^T L x is a smoothness penalty. An eigenvector u_l satisfies u_l^T L u_l = λ_l. Small λ_l means u_l barely changes across edges — it's a slowly varying, low-frequency mode. Large λ_l means u_l flips sign across edges — high frequency. And if I specialize the graph to a ring, L becomes the discrete second-difference operator, whose eigenvectors are exactly the sines and cosines and whose eigenvalues grow like the squared classical frequency. So this isn't an analogy I'm forcing — on the grid it *is* ordinary Fourier. Good. The graph Fourier transform of a signal x is then x̂ = U^T x, and the inverse is x = U x̂ (consistent because U^T U = I).

Now I can define convolution the only way that makes sense here: by fiat, in the Fourier domain. Filtering x by a spectral filter g_θ means multiply each Fourier coefficient x̂_l by g_θ(λ_l), then transform back:

    y = U g_θ(Λ) U^T x.

And here's a clean way to see this as "applying g_θ to the operator L itself": since L = U Λ U^T, powers telescope using U^T U = I,

    L^k = (U Λ U^T)(U Λ U^T)⋯(U Λ U^T) = U Λ^k U^T,

so any function g defined by a power series satisfies g(L) = U g(Λ) U^T. Therefore y = g_θ(L) x. So a spectral filter is just "evaluate the function g_θ on the Laplacian." That's a very nice handle.

The most flexible choice is to let every frequency have its own free gain:

    g_θ(Λ) = diag(θ),    θ ∈ R^n.

Maximally expressive — n free parameters, one per frequency. But now let me actually try to *use* this and see what hurts. Three things hurt, and they all hurt a lot.

First, cost of *applying* it. To compute y = U g_θ(Λ) U^T x I need U. Getting U means eigendecomposing L, which is O(n^3), and storing it is an n×n matrix. Worse, even if I had U sitting around, every forward pass multiplies x by U^T and then by U — two dense matrix–vector products at O(n^2) each, and again on the backward pass. For a graph with, say, 10^4 word-vertices that's 10^8 per pass per signal, and there's no fast Fourier transform on a general graph to rescue me the way the FFT rescues grids. So this is genuinely expensive and doesn't scale.

Second, parameters. θ ∈ R^n means the number of learnable weights grows with the size of the graph. That's the opposite of what made CNNs cheap — there the filter had O(S) weights regardless of image size. A filter that needs as many parameters as there are vertices isn't a convolution in spirit; it's a full per-frequency reweighting.

Third — and this is the subtle one — is it even localized? A convolution on a grid is local: the output at a pixel depends only on a small patch around it. What does my spectral filter's output at vertex i depend on? Let me probe it with a delta. Put a spike at vertex i: δ_i. Then (g_θ(L) δ_i)_j = (g_θ(L))_{ij}, the (i,j) entry of the operator. For a generic diag(θ) the matrix U diag(θ) U^T is dense — its (i,j) entry is Σ_l θ_l (u_l)_i (u_l)_j, which is nonzero even when i and j are far apart on the graph. So a generic spectral multiplier spreads all over the graph; it's *not* spatially localized at all. That kills the whole point — I wanted local filters.

So free-per-frequency is wrong on all three counts: dense/expensive, O(n) parameters, and not localized. I need to constrain g_θ. The constraint has to simultaneously (a) make the operator localized, (b) cut parameters to O(K), and ideally (c) make it cheap to apply. Let me chase localization first, because that's the property I understand least and care about most.

What family of g should I pick? Let me think about what "depends only on nearby vertices" means in terms of L. Look again at (L^k)_{ij}. Expanding the product, (L^k)_{ij} = Σ over all sequences i = v_0, v_1, …, v_k = j of the product L_{v_0 v_1} L_{v_1 v_2} ⋯ L_{v_{k-1} v_k}. Each factor L_{ab} is nonzero only if a=b or a,b are adjacent. So a nonzero term requires a walk of length k from i to j along edges, with possible self-steps from the diagonal. If the shortest path distance d_G(i,j) is greater than k, no such length-k walk can reach j from i, every term vanishes, and (L^k)_{ij} = 0. The k-th power of the Laplacian only connects vertices within k hops. That's exactly localization, sitting right there in the algebra of L.

So if I make g_θ a *polynomial* in L,

    g_θ(Λ) = Σ_{k=0}^{K-1} θ_k Λ^k    ⟺    g_θ(L) = Σ_{k=0}^{K-1} θ_k L^k,

then the operator's (i,j) entry is Σ_k θ_k (L^k)_{ij}, and by the walk argument every term with k < d_G(i,j) — wait, let me be careful with the direction — a term θ_k (L^k)_{ij} is zero whenever d_G(i,j) > k. So the only surviving terms at a far-apart pair (i,j) are those with k ≥ d_G(i,j); if d_G(i,j) exceeds the top degree K−1, *all* terms vanish and the entry is zero. The highest power present is L^{K-1}, so the filter reaches at most K−1 hops: a degree-(K−1) polynomial of L is exactly (K−1)-hop localized. The support is set by the degree, exactly and provably — no fiddling with smoothness in the Fourier domain hoping for spatial decay. And the parameter count is K, the number of coefficients, which I get to choose independent of n. That's two of my three problems solved in one move: localized with controllable support, O(K) parameters.

This is much better than the smooth-spectrum approach I'd otherwise reach for, where I'd parametrize g_θ(Λ) = Bθ with B some smooth basis (say cubic B-splines) and θ a few control points, betting that smoothness in frequency gives decay in space. That gives *approximate* localization with no precise control of the support, and it still needs U. The polynomial gives me the support exactly as a knob (the degree) and is defined purely through L. So polynomial-of-L is the right shape for the filter.

But I've only fixed two of the three problems. Cost is still bad if I'm not careful. If I literally compute y = U g_θ(Λ) U^T x by forming the diagonal g_θ(Λ) and sandwiching it in U, I still need U — still the O(n^3) eigendecomposition and the two O(n^2) dense multiplications. The polynomial parametrization by itself doesn't save me there. But it *opens a door*: because g_θ(L) is a polynomial in L, I never have to diagonalize at all. I can apply it directly through L:

    y = Σ_{k=0}^{K-1} θ_k L^k x.

And L is sparse — it has on the order of |E| nonzeros. So computing L^k x by repeated multiplication, L^k x = L (L^{k-1} x), is k sparse matrix–vector products, each O(|E|). Building the whole set {x, Lx, L^2 x, …, L^{K-1} x} costs O(K|E|), and then y is a θ-weighted sum of them. For sparse real-world graphs |E| ≪ n^2, so O(K|E|) ≪ O(n^2). No U, no eigendecomposition, no storing an n×n basis — I only ever need L, a sparse matrix. That's the third problem solved. So the plan is: never form the Fourier basis; apply the polynomial of L by sparse mat-vecs.

Hold on, let me not be sloppy about *which* polynomial basis I push through that recurrence. The naive thing is the monomial basis: compute x, Lx, L^2 x by squaring up, θ-combine. Algebraically fine. But numerically the monomials L^k are a bad basis — the powers of an operator whose spectrum is spread out blow up or collapse unevenly across the spectrum, and the monomials are far from orthogonal, so the conditioning is poor and learning the θ's is ill-posed (small changes in high-order coefficients swing the filter wildly). I want a basis that (i) I can still generate by a cheap recurrence in L, so I keep the O(K|E|) cost, and (ii) is well-conditioned/orthogonal so the coefficients are tame. This is a classic situation in approximating a function on an interval, and the classic answer is Chebyshev polynomials.

Recall the Chebyshev polynomials: T_0(y)=1, T_1(y)=y, and the three-term recurrence T_k(y) = 2y T_{k-1}(y) − T_{k-2}(y). They're orthogonal on [−1,1] with respect to the weight dy/√(1−y²), they're bounded and equi-oscillating on that interval (so high-order coefficients don't explode), and crucially that recurrence is *exactly* the kind of cheap, stable, two-term-memory recursion I can run on L. The one catch is the domain: T_k lives on [−1,1], but the Laplacian's eigenvalues λ live in [0, λ_max]. So I have to map [0, λ_max] → [−1,1] affinely. The map is λ ↦ 2λ/λ_max − 1: it sends 0 ↦ −1 and λ_max ↦ +1, linear in between, so the rescaled eigenvalues all land in [−1,1]. In matrix form,

    Λ̃ = 2Λ/λ_max − I_n,    and correspondingly    L̃ = 2L/λ_max − I_n.

Now parametrize the filter as a truncated Chebyshev expansion in the rescaled variable:

    g_θ(Λ) = Σ_{k=0}^{K-1} θ_k T_k(Λ̃),    θ ∈ R^K,

so the filtering operation is

    y = g_θ(L) x = Σ_{k=0}^{K-1} θ_k T_k(L̃) x.

Is this still localized with the same radius? T_k(L̃) is a degree-k polynomial in L̃, and L̃ = 2L/λ_max − I is an affine function of L, so T_k(L̃) is a degree-k polynomial in L. A degree-(K−1) polynomial in L is (K−1)-hop localized by the exact same walk argument as before — the affine rescaling and the choice of orthogonal basis don't change which entries are forced to zero. So I keep exact radius-(K−1) localization with K coefficients, and now the basis is numerically stable.

And does the recurrence carry over to the cheap application? Let me actually do the algebra rather than assume it. Define the vector x̄_k = T_k(L̃) x. Apply the Chebyshev recurrence at the operator level: T_k(L̃) = 2 L̃ T_{k-1}(L̃) − T_{k-2}(L̃). Right-multiply by x:

    x̄_k = T_k(L̃) x = 2 L̃ T_{k-1}(L̃) x − T_{k-2}(L̃) x = 2 L̃ x̄_{k-1} − x̄_{k-2},

with the base cases x̄_0 = T_0(L̃) x = x and x̄_1 = T_1(L̃) x = L̃ x. So I build the sequence

    x̄_0 = x,    x̄_1 = L̃ x,    x̄_k = 2 L̃ x̄_{k-1} − x̄_{k-2},

each step a single sparse multiply by L̃ plus a scale and a subtract — two-vector memory, O(|E|) per step, O(K|E|) total. Then the filter output is just a linear combination,

    y = Σ_{k=0}^{K-1} θ_k x̄_k = [x̄_0, x̄_1, …, x̄_{K-1}] θ.

That's the whole filter: stack the K Chebyshev-basis vectors and multiply by the coefficient vector. Eigendecomposition-free, radius-(K−1) localized with K coefficients, O(K|E|). Every wall I hit with the free-per-frequency filter is gone.

Now, what's λ_max in practice? Computing the true largest eigenvalue would mean a power iteration on L every time — a cost I'd rather avoid. But for the *normalized* Laplacian there's a free fact: its spectrum is bounded by 2, i.e. 0 ≤ λ_l ≤ 2 always. So I can just set λ_max = 2 without computing anything. Then L̃ = 2L/2 − I = L − I_n. If the true maximum is a bit below 2, the rescaled eigenvalues land in something like [−1, 1−ε) instead of exactly [−1,1] — but that's a harmless slack: the Chebyshev basis is still well-conditioned there, and the learned coefficients θ simply absorb the mild rescaling. So I take λ_max = 2 and L̃ = L − I_n. Clean.

Let me make this a real convolutional *layer*, not just a single filter, because a CNN layer maps a stack of F_in input feature maps to F_out output feature maps, with weight sharing across the whole domain. So the j-th output map on sample s is a sum over input maps, each filtered by its own coefficient vector:

    y_{s,j} = Σ_{i=1}^{F_in} g_{θ_{i,j}}(L) x_{s,i} ∈ R^n,

where the F_in × F_out vectors θ_{i,j} ∈ R^K are the layer's trainable parameters — and notice these θ's don't depend on the vertex, so the same filter is applied everywhere on the graph; that's the weight-sharing/stationarity I wanted, transplanted to the graph. Over a minibatch of S samples, the sparse recurrence costs O(K|E|F_inS), and the coefficient mixing is the dense product from N·M rows and F_inK basis channels to F_out outputs, O(SnKF_inF_out); for fixed K and channel counts on sparse k-nearest-neighbor graphs, the whole layer scales linearly in n and parallelizes as sparse/dense tensor operations. And for backprop I need the gradients through this. Since y_{s,j} = [x̄_{s,i,0},…,x̄_{s,i,K-1}] θ_{i,j} summed over i, the coefficient gradient is

    ∂E/∂θ_{i,j} = Σ_s [x̄_{s,i,0}, …, x̄_{s,i,K-1}]^T (∂E/∂y_{s,j}),

i.e. project the upstream gradient onto the same Chebyshev basis I already built. And the gradient to the input is another application of the (symmetric) filter,

    ∂E/∂x_{s,i} = Σ_j g_{θ_{i,j}}(L) (∂E/∂y_{s,j}),

which is again K sparse mat-vecs — the backward pass costs the same as the forward pass, and the basis [x̄_{s,i,0},…,x̄_{s,i,K-1}] is computed once per input and reused everywhere. So the whole layer is cheap both ways.

That's the convolution sorted. But a CNN isn't just convolution — it's convolution *and pooling*, stacked, to get multi-scale composition. On a grid, pooling is trivial: group 2×2 blocks, take the max, halve the resolution. On a graph there's no grid to block up. I need to (i) decide which vertices to group — a clustering — and (ii) actually compute the pooling fast.

For the clustering: I want repeated coarsening, each level producing a graph roughly half the size of the previous one (so a pooling of size 2 corresponds to one coarsening level, size 4 to two levels, and so on — a clean power-of-two control like grid pooling). The trouble is that good graph clustering is NP-hard, so I can't do it exactly; I need a fast heuristic. A greedy multilevel matching does the job: at each level, walk over unmarked vertices; for an unmarked vertex i, match it with the unmarked neighbor j that maximizes the local normalized cut W_{ij}(1/d_i + 1/d_j) — i.e. the neighbor it's most tightly and "cheaply" connected to — mark both, and set the merged super-vertex's edge weights to the sum of the two. Repeat until everything's matched. Each level roughly halves the vertex count (with a few leftover singletons that don't get a partner), and I get a hierarchy of coarser graphs — the analogue of an image pyramid.

Now the second part — pooling *fast*. After matching, the vertices of each coarsened graph come out in no particular order. If I pool naively I'd need a lookup table mapping each coarse vertex to its two children, scattered all over memory — slow, and terrible for a GPU where I want contiguous, local memory access. There's a trick to make graph pooling look exactly like 1D pooling. Arrange the coarsening hierarchy as a *balanced binary tree*. After matching, a coarse vertex has either two children (it was a matched pair) or one child (it was a singleton). Force every node to have exactly two children by inserting *fake* nodes: a singleton gets one real child plus one fake child, and a fake node gets two fake children all the way down. Fake nodes are disconnected from the graph and carry a neutral value (0, since I'll use ReLU with max-pooling), so filtering — which only mixes connected vertices — never touches them; they're inert padding. Then order the coarsest level arbitrarily and propagate the order down: node k has children 2k and 2k+1. After this propagation the finest level is ordered so that the vertices that will be merged together upward sit *adjacent* in the array. So pooling the rearranged signal is literally a regular 1D max-pool of size p (a power of two) — contiguous, no lookup table, GPU-friendly. The cost of the fake nodes is a slightly larger signal, but in practice the matching leaves few singletons, so it's cheap.

So the full picture of one block: take the input feature maps, apply the Chebyshev graph convolution (build the K-term basis by the L̃ recurrence, combine by θ), add a per-filter bias and a ReLU, then graph-pool by reordering into the binary tree and running a 1D max-pool. Stack a few of these, flatten, finish with fully connected layers and a softmax, train with cross-entropy and the usual SGD/Adam. Localized, stationary, compositional — the three CNN properties — now on an arbitrary graph, at O(K|E|) per layer with O(K) parameters per filter and no eigendecomposition anywhere.

Let me write the core operator the way it actually runs, in terms of sparse mat-vecs on the rescaled Laplacian, so I'm sure the recurrence and the reshaping are exactly the algebra above.

```python
import numpy as np
import scipy.sparse
import tensorflow as tf

def laplacian(W, normalized=True):
    # Normalized graph Laplacian L = I - D^{-1/2} W D^{-1/2}; symmetric, PSD.
    d = W.sum(axis=0)
    d = 1 / np.sqrt(d + np.spacing(np.array(0, W.dtype)))
    D = scipy.sparse.diags(d.A.squeeze(), 0)
    I = scipy.sparse.identity(d.size, dtype=W.dtype)
    return I - D * W * D

def rescale_L(L, lmax=2):
    # Map eigenvalues [0, lmax] -> [-1, 1]:  L_tilde = (2/lmax) L - I.
    # For the normalized Laplacian lmax <= 2, so we can just use lmax = 2.
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L = L * (2 / lmax)
    L = L - I
    return L

def chebyshev_basis(L, X, K):
    # Build [x_bar_0, ..., x_bar_{K-1}] with the three-term recurrence:
    #   x_bar_0 = X ;  x_bar_1 = L_tilde X ;  x_bar_k = 2 L_tilde x_bar_{k-1} - x_bar_{k-2}.
    # K-1 sparse mat-vecs by L -> O(K |E|), no eigendecomposition.
    M, N = X.shape
    Xt = np.empty((K, M, N), L.dtype)
    Xt[0, ...] = X                                   # T_0(L) X = X
    if K > 1:
        Xt[1, ...] = L.dot(X)                        # T_1(L) X = L_tilde X
    for k in range(2, K):
        Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
    return Xt

def chebyshev_conv(x, L, Fout, K, weight_variable):
    # One graph convolutional layer mapping Fin -> Fout feature maps.
    # x: N x M x Fin  (N signals, M vertices, Fin input maps)
    N, M, Fin = x.get_shape()
    N, M, Fin = int(N), int(M), int(Fin)

    # Rescale Laplacian once and keep it as a sparse tensor.
    L = scipy.sparse.csr_matrix(L)
    L = rescale_L(L, lmax=2)                          # L_tilde = L - I
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    L = tf.sparse_reorder(L)

    # Chebyshev recurrence directly on the (sparse) Laplacian.
    x0 = tf.reshape(tf.transpose(x, perm=[1, 2, 0]), [M, Fin * N])   # M x Fin*N
    stack = tf.expand_dims(x0, 0)                                    # 1 x M x Fin*N
    def append(stack, x_):
        return tf.concat([stack, tf.expand_dims(x_, 0)], axis=0)
    if K > 1:
        x1 = tf.sparse_tensor_dense_matmul(L, x0)                   # L_tilde x_bar_0
        stack = append(stack, x1)
    for k in range(2, K):
        x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0          # 2 L_tilde x_bar_{k-1} - x_bar_{k-2}
        stack = append(stack, x2)
        x0, x1 = x1, x2
    # stack: K x M x Fin*N  ->  N*M x Fin*K
    stack = tf.reshape(stack, [K, M, Fin, N])
    stack = tf.transpose(stack, perm=[3, 1, 2, 0])                  # N x M x Fin x K
    stack = tf.reshape(stack, [N * M, Fin * K])

    # Apply the learned Chebyshev coefficients: Fin*K -> Fout (shared over vertices).
    W = weight_variable([Fin * K, Fout])                            # the theta's
    y = tf.matmul(stack, W)                                          # N*M x Fout
    return tf.reshape(y, [N, M, Fout])                              # N x M x Fout
```

To recap the causal chain: I wanted CNN-style local, weight-shared, compositional filters on a graph; the spatial route has no well-defined translation to share weights, so I defined convolution spectrally through the Laplacian's eigenbasis (the graph Fourier transform), giving filtering as y = U g_θ(Λ) U^T x = g_θ(L) x; the free-per-frequency filter g_θ = diag(θ) is dense, costs an O(n^3) eigendecomposition plus O(n^2) per pass, has O(n) parameters, and isn't localized; restricting g_θ to a polynomial of L makes it exactly K-hop localized (because (L^k)_{ij}=0 beyond k hops) with O(K) parameters; expressing that polynomial in the Chebyshev basis and running its three-term recurrence x̄_k = 2L̃x̄_{k-1} − x̄_{k-2} directly on the sparse rescaled Laplacian L̃ = L − I makes filtering eigendecomposition-free at O(K|E|); a greedy multilevel matching coarsens the graph by ~½ per level, and a balanced-binary-tree reordering with fake nodes turns graph pooling into ordinary 1D pooling; stacking conv–ReLU–pool blocks then fully connected layers gives a genuine CNN on graphs.
