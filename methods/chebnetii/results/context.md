# Context: learnable polynomial spectral graph filters (circa 2021-2022)

## Research question

A graph neural network for node classification has to push each node's features through the
graph structure before reading them out, and the operator that does the pushing — a *graph
filter* — decides what the network can learn. On a graph where connected nodes share labels
(a homophilic graph: citation networks, where two linked papers usually share a topic) a
smoothing, low-pass operator is exactly right: averaging a node with its neighbors sharpens
the class signal. On a graph where connected nodes tend to *differ* (a heterophilic graph:
webpage link graphs where a hub links to pages of many categories) smoothing is actively
harmful — the discriminative signal lives in how a node *contrasts* with its neighborhood, a
high-frequency component that averaging destroys. The same fixed operator cannot serve both
regimes.

So the goal is a single graph filter that can *learn* whatever frequency response a given
graph needs — low-pass, high-pass, band-pass, or any mixture — from labels alone, and do it
without three failure modes that the prior art keeps hitting. First, it must be cheap:
linear, not quadratic, in the polynomial order `K`, because `K` controls how many hops of
structure the filter can see and we want to push it to 10 or more. Second, it must not
overfit: a freely-parameterized high-order filter has enough capacity to memorize the
training labels through a wild, oscillatory frequency response that generalizes terribly.
Third, it should accept *side constraints* on the learned response — most importantly,
non-negativity of the controlled response values — cleanly, because some designs need a
positivity constraint and the prior art either guarantees it through a special basis or cannot
express it as a simple parameter constraint. A method that learns arbitrary filters, stays
linear in `K`, resists overfitting, and admits sample-value constraints by construction is what
is missing.

## Background

**Spectral filtering on a graph.** Take an undirected graph `G=(V,E)` with `n=|V|` nodes,
adjacency `A`, degree matrix `D`. A graph signal is a vector `x in R^n`, one value per node
(for a feature matrix `X`, each column is a signal). The central operator is the symmetric
normalized Laplacian `L = I - D^{-1/2} A D^{-1/2}`, which is real, symmetric, positive
semidefinite, so it diagonalizes as `L = U Λ U^T` with orthonormal eigenvectors `U` and
ordered eigenvalues `0 ≤ λ_1 ≤ ... ≤ λ_n`. The eigenvectors play the role of Fourier modes
and the eigenvalues the role of frequencies: small-`λ` eigenvectors vary slowly across edges,
large-`λ` eigenvectors oscillate. A spectral filter applies a scalar function `h` to each
frequency,

```
y = U diag[h(λ_1), ..., h(λ_n)] U^T x = U h(Λ) U^T x.
```

Computing `U` is an `O(n^3)` eigendecomposition, infeasible at scale, so in practice `h` is
restricted to a polynomial of `L`, `h(λ) = sum_{k=0}^K w_k λ^k`, giving
`y ≈ sum_k w_k L^k x`, which needs only `K` sparse matrix-vector products and never forms
`U`. A crucial structural fact: the eigenvalues of the normalized Laplacian satisfy
`0 ≤ λ ≤ 2` (Chung 1997), so `λ_max ≤ 2` is known a priori without touching the spectrum.

**Why the filter's shape is the whole game for node classification.** Picture a ring graph
with a one-hot signal. An impulse low-pass response (`h(λ)=1` at `λ=0`, else `0`) projects
onto the constant eigenvector and spreads the signal evenly — perfect separation when all
nodes share a label (homophily). An impulse high-pass response (`h=1` at `λ=2`, else `0`)
projects onto the most-oscillatory eigenvector, producing alternating signs — perfect
separation when adjacent nodes alternate labels (heterophily). A band-pass response handles a
mixture. The lesson is that *which* frequency response is learned, not merely the depth of
propagation, determines whether classification succeeds, and homophilic and heterophilic
graphs demand opposite responses.

**Chebyshev polynomials and approximation theory.** The Chebyshev polynomials are defined by
the three-term recurrence `T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)` with `T_0(x)=1`, `T_1(x)=x`,
and have the closed form `T_k(x) = cos(k · arccos x)` on `[-1,1]`; higher `k` means
higher-frequency oscillation. They are an orthogonal basis of `L^2([-1,1], dx/√(1-x²))`, and
their domain is `[-1,1]`, so any spectrum must be affinely rescaled into `[-1,1]` before they
apply. The reason they are the natural basis for approximation is that the truncated Chebyshev
expansion of a function is a *near-minimax* polynomial — its worst-case (uniform-norm) error
is within a small factor of the best possible degree-`K` polynomial. That small factor is the
**Lebesgue constant** `ρ`: a near-best approximation satisfies
`||f - P_K|| ≤ (1+ρ)||f - P_best||` in the uniform norm `||g|| = max_{x∈[-1,1]}|g(x)|`.

**Two facts about *how* one builds a polynomial that matches a function** — both pre-method
numerical-analysis results. (a) *Coefficients of an analytic filter must decay.* If `f` is
analytic in `(-1,1)` (locally a convergent power series) and at worst weakly singular at the
boundaries, then its Chebyshev coefficients `w_k` decrease asymptotically like `1/k^q` for
some positive `q` (Zhang et al. 2021). Since large-`k` Chebyshev polynomials are exactly the
high-frequency modes, this says a *well-behaved* (analytic) filter must put vanishing weight
on high frequencies; a response with non-decaying high-`k` coefficients is a non-analytic,
hard-to-approximate, wildly oscillating function. Empirically the Chebyshev coefficients of a
smooth analytic filter such as `exp(λ_hat)` (the response used by diffusion-based GDC) are
visibly convergent. (b) *Interpolation and the Runge phenomenon.* Instead of an expansion one
can build the degree-`K` polynomial that *matches* the filter's values at `K+1` chosen points
(polynomial interpolation; the coefficients solve a Vandermonde system, or equivalently the
Lagrange form `P_K = sum_k h(λ_k) L_k`). The interpolation error is
`R_K(λ_hat) = h^{(K+1)}(ζ)/(K+1)! · π_{K+1}(λ_hat)`, where the nodal polynomial
`π_{K+1}(λ_hat) = prod_{k}(λ_hat - λ_hat_k)` collects the chosen points. With *equispaced*
points the nodal polynomial swings violently near the interval ends, so for Runge-type targets
the high-degree interpolation error can grow without bound — the Runge phenomenon. The known
cure is to place the points where the nodal polynomial stays small.

**Decoupling transformation from propagation.** A separate line of GNN design (APPNP,
Klicpera et al. 2019) observed that the feature transformation (an MLP) and the graph
propagation need not be interleaved layer by layer the way the original convolutional GNNs do.
Running an MLP `f_θ(X)` first and then applying a fixed propagation operator
`sum_k α(1-α)^k P̃^k` (a Personalized-PageRank polynomial, `P̃` the renormalized adjacency)
keeps the parameter count tied to the MLP and the feature dimension, independent of how many
propagation hops `K` you take, and improves scalability. This decoupled `MLP → polynomial
propagation → readout` template is the harness the learnable-filter methods below all reuse.

## Baselines

**ChebNet (Defferrard, Bresson & Vandergheynst 2016).** The original learnable spectral GNN.
It approximates the filtering operation with a Chebyshev polynomial of the *scaled* Laplacian,
`y ≈ sum_{k=0}^K w_k T_k(L_hat) x` with `L_hat = 2L/λ_max - I` mapping the spectrum into
`[-1,1]`, computed by the stable three-term recurrence. Its layer is
`Y = sum_{k=0}^K T_k(L_hat) X W_k`: the Chebyshev coefficients are *implicitly* absorbed into a
full trainable weight matrix `W_k` per order. In theory, as `K` grows it can represent
arbitrary spectral filters. **Limitation:** on the standard citation benchmarks it is beaten
by GCN — its own first-order simplification — and the gap *widens* as `K` goes from 2 to 10
(Cora: 80.5 at `K=2`, 74.9 at `K=10`, versus GCN 81.3). Carrying a separate `W_k` per order
also makes the parameter count grow with `K` (230k at `K=10` on Cora versus ~92k for the
decoupled designs). The coefficients are fit freely by gradient descent with nothing tying
them to any well-behaved frequency response.

**GCN (Kipf & Welling 2017).** ChebNet collapsed to first order: set `K=1`, `w_0 = -w_1 = w`,
`λ_max = 2`, giving `y = w(I + D^{-1/2} A D^{-1/2}) x`, then a renormalization trick replaces
`I + D^{-1/2} A D^{-1/2}` with `D̃^{-1/2} Ã D̃^{-1/2}` (self-loops added). The result is a
fixed low-pass filter, applied as `H^{(l+1)} = σ(P̃ H^{(l)} W^{(l)})`. **Limitation:** the
response is fixed and low-pass; it cannot represent the high-pass or band-pass responses that
heterophilic graphs need, and it is not learnable as a filter.

**GPR-GNN (Chien, Peng, Li & Milenkovic 2021).** Decouples like APPNP — MLP first — then
propagates with a *Monomial* basis whose coefficients are learned:
`Y = sum_{k=0}^K γ_k P̃^k f_θ(X)`, equivalently the spectral filter
`h(λ̃) = sum_k γ_k λ̃^k`. Learning the `γ_k` (Generalized PageRank weights) lets the filter
become low- or high-pass and adapt to the homophily level; the learned weights even alternate
in sign on heterophilic graphs. The coefficients are commonly initialized to the PPR pattern
`γ_k = α(1-α)^k`. **Limitation:** the monomial basis `{λ^k}` is numerically ill-conditioned
— the powers become nearly collinear for large `K` (the Vandermonde conditioning problem), so
high-order monomial fits are unstable and the basis is a *poor* approximator despite the
flexibility; on the citation graphs a plain Chebyshev-basis filter in the same harness
underperforms it, which is the puzzle, since Chebyshev is the near-minimax basis.

**BernNet (He, Wei, Huang & Xu 2021).** Same decoupled harness, but the filter is built in the
*Bernstein* basis: `Y = sum_{k=0}^K θ_k · (1/2^K) C(K,k) (2I - L)^{K-k} L^k f_θ(X)`, i.e.
`h(λ) = sum_k θ_k · (1/2^K) C(K,k) (2-λ)^{K-k} λ^k`. The Bernstein coefficient `θ_k`
controls a "bump" located at frequency `≈ 2k/K`, which makes the design interpretable, and
since every Bernstein basis function is non-negative on the spectrum, constraining all
`θ_k ≥ 0` yields a guaranteed non-negative filter response — a property BernNet argues some
valid filters require. **Limitation:** each forward pass costs `O(K^2 m d)` — quadratic in
`K` — because building the `(2I-L)^{K-k} L^k` terms re-walks the graph; this is acknowledged
as the price with no linear-time fix offered. Bernstein approximation also converges more
slowly than the near-minimax option, so it needs a larger `K` for the same fidelity.

**ChebBase (the Chebyshev basis dropped into the decoupled harness).** As a diagnostic, the
Chebyshev basis is plugged into the exact GPR-GNN/BernNet template:
`Y = sum_{k=0}^K w_k T_k(L_hat) f_θ(X)`, with `w_k` learned freely and initialized to
`w_0 = 1`, `w_k = 0` otherwise (start from no propagation). **Limitation:** this is the
counter-intuitive observation that sets up the whole problem — ChebBase is the *worst* of the
three bases on the citation graphs (Cora 79.3 versus GPR-GNN 84.0, BernNet 83.2), even though
approximation theory says the Chebyshev basis should be the *best* approximator. The freely
learned `w_k` carry no constraint tying them to a well-behaved response.

## Evaluation settings

The natural yardsticks, all pre-existing:

- **Datasets** spanning homophily levels. Homophilic citation graphs: Cora (2,708 nodes, 7
  classes), Citeseer (3,327 nodes, 6 classes), Pubmed (19,717 nodes, 5 classes). Heterophilic
  WebKB webpage graphs: Texas, Cornell (183 nodes, 5 classes each), plus Wikipedia/actor
  graphs Chameleon, Squirrel, Actor. The homophily level `H(G)` ranges from ~0.81 (Cora) down
  to ~0.11 (Texas), so a method must be tested across the full homophily-to-heterophily span.
- **Task and protocol.** Semi-supervised node classification with the standard fixed
  split (20 training nodes per class, 500 validation, 1,000 test) for the
  ChebNet-vs-GCN-vs-bases diagnostics, and full-/semi-supervised splits with random 60/20/20
  train/val/test partitions and early stopping for the broader study; results are averaged
  over 10 runs per dataset.
- **Architecture and optimizer for the decoupled methods.** A 2-layer MLP with 64 hidden
  units, propagation order `K=10`, dropout 0.5 on the linear layers and a separate dropout
  on the propagation step, trained with Adam; the linear layers and the propagation
  parameters get their own learning rate and weight decay. Coefficient initializations follow
  each method's convention (PPR for GPR-GNN, all-ones for BernNet).
- **Metric.** Mean test classification accuracy over the runs, higher is better.
- **Scale.** Beyond the small graphs, the natural large-scale yardsticks are ogbn-arxiv and
  the billion-edge ogbn-papers100M, where any quadratic-in-`K` cost becomes prohibitive.

## Code framework

The filter plugs into the decoupled `MLP → propagation → log-softmax` harness that already
exists for the learnable-filter baselines. The MLP, the dropout, the symmetric-normalized
Laplacian builder, the self-loop utility, the single-step message passing, and the Chebyshev
polynomial evaluator are all standard primitives. What is *not* settled is the propagation
layer's parameterization — how the trainable filter coefficients are defined and combined —
which is exactly the slot to fill. The scaffold leaves that one body empty.

```python
import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian


def cheby(i, x):
    """Evaluate the i-th Chebyshev polynomial T_i(x) by the three-term recurrence."""
    if i == 0:
        return 1
    if i == 1:
        return x
    T0, T1 = 1, x
    for _ in range(2, i + 1):
        T0, T1 = T1, 2 * x * T1 - T0
    return T1


class CustomProp(MessagePassing):
    """Polynomial graph propagation layer. Owns the trainable filter parameters and
    applies a degree-K polynomial of the (shifted) normalized Laplacian to the input.

    Available primitives:
      - get_laplacian(edge_index, edge_weight, normalization='sym')  -> L = I - D^{-1/2} A D^{-1/2}
      - add_self_loops(edge_index, norm, fill_value=-1.0)            -> shift the operator
      - self.propagate(edge_index, x=x, norm=norm)                   -> one sparse mat-vec
      - cheby(i, x)                                                  -> T_i(x) at a scalar node
    """

    def __init__(self, K, alpha=0.1, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K = K
        # TODO: the trainable filter parameters we will define for this layer.
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: initialize the filter parameters.
        pass

    def forward(self, x, edge_index, edge_weight=None):
        # TODO: define how the K+1 trainable parameters become the polynomial
        #       coefficients, build the (shifted) normalized Laplacian, and combine
        #       the propagated signals into the filtered output.
        pass

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(torch.nn.Module):
    """Decoupled harness: MLP transform first, then graph propagation, then readout."""

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super().__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K, alpha)
        self.dropout = dropout
        self.dprate = dprate

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)
```

The MLP, dropout, Laplacian, self-loops, message passing, and `cheby` are given; the empty
`CustomProp` body — the parameterization of the trainable filter and how it assembles the
polynomial — is the slot the method fills.
