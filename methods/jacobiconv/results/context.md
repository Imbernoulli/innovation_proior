# Context: orthogonal polynomial bases for spectral graph filters (circa 2022)

## Research question

A spectral graph neural network classifies nodes by pushing features through the graph with a
*graph filter* — a polynomial `h(L)` of the normalized Laplacian whose frequency response `h(λ)`
decides what the network can see. On a homophilic graph (linked nodes share labels) the right
response is a smooth low-pass; on a heterophilic graph (linked nodes differ) the discriminative
content is high-frequency and the response must be high- or band-pass. A single learnable filter
should reach any of these from labels alone. The prior learnable filters do reach them — GPR-GNN
learns monomial coefficients, BernNet learns Bernstein coefficients, ChebNetII interpolates at
Chebyshev nodes — and they differ only in *which polynomial basis* parameterizes the response.

So the question this work asks is sharper than "can the filter take any shape": **given that several
complete polynomial bases all have the same expressive power, why do they reach such different
accuracies, and is there a basis that is optimal?** And a second, more provocative question: the
established models all wrap the polynomial propagation in a nonlinear MLP — **is the nonlinearity
even necessary**, or is a *linear* spectral GNN already powerful enough for node classification?

## Background

**Spectral filtering and the polynomial restriction.** With the symmetric normalized Laplacian
`L = I - D^{-1/2} A D^{-1/2} = U Λ U^T`, a spectral filter applies a scalar `h` per frequency,
`y = U h(Λ) U^T x`. The eigendecomposition is `O(n^3)`, so `h` is restricted to a degree-`K`
polynomial of `L`, evaluated by `K` sparse mat-vecs. The normalized adjacency `Â = I - L =
D^{-1/2} A D^{-1/2}` has spectrum in `[-1,1]`, which is exactly the domain on which the classical
orthogonal polynomials live, so a filter is naturally written as `Z = sum_k α_k g_k(Â) X` for a
basis `{g_k}`.

**The decoupled harness.** Following APPNP (Klicpera et al. 2019), the learnable-filter methods run
an MLP `f_θ(X)` first and then the polynomial propagation, `Z = sum_k α_k g_k(Â) f_θ(X)`, keeping
the parameter count independent of the order `K`. The three learnable baselines all use this harness
and differ only in `g_k`.

**Baselines, by basis.**
- **GPR-GNN (Chien et al. 2021)** — *Monomial* basis `g_k(λ) = λ^k`. Maximally simple, learns signed
  hop-weights, but the monomial powers are ill-conditioned (the Vandermonde collinearity), and one
  can prove the monomial basis *cannot* be orthogonal under any weight function.
- **ChebNet / ChebNetII (Defferrard et al. 2016; He et al. 2022)** — *Chebyshev* basis. Near-minimax
  and `O(K)`; ChebNetII fixes the free-coefficient overfitting by interpolating at Chebyshev nodes.
  But Chebyshev is orthogonal w.r.t. only *one* fixed weight `(1-λ²)^{-1/2}`.
- **BernNet (He et al. 2021)** — *Bernstein* basis. Non-negative and interpretable, but `O(K²)` and a
  non-orthogonal basis whose Hessian need not be diagonal.

**Two pre-method facts the analysis turns on.**
(a) *Universality of linear spectral GNNs (Wang & Zhang's own analysis).* A linear GNN
`Z = g(L̂) X W` can produce *any* one-dimensional prediction provided the Laplacian has no repeated
eigenvalues and the features contain all frequency components — and on the ten standard benchmarks
fewer than 1% of eigenvalues are multiple and *no* frequency component is missing, so the
universality conditions hold in practice. ReLU's only role is to *mix* frequency components across
the spectrum, which the no-missing-frequency condition makes unnecessary. This is the license to
*drop the nonlinearity* and study a purely linear filter.
(b) *Convergence speed depends on the basis through the Hessian.* For the linear filter trained under
a squared loss `R = ½‖Z − Y‖²`, the Hessian w.r.t. the coefficients `α` has entries
`H_{k₁k₂} = ∫₀² g_{k₁}(λ) g_{k₂}(λ) f(λ) dλ`, where `f(λ)` is the *spectral density of the graph
signal*. Gradient descent's convergence rate is governed by the condition number `κ(H)`, which is
minimized — `κ = 1` — exactly when `H = I`, i.e. when `{g_k}` is *orthonormal under the inner product
weighted by the signal density `f`*. So the optimal basis is not fixed; it is the orthogonal family
whose weight function matches `f(λ)`.

**The Jacobi family.** The Jacobi polynomials `P_k^{a,b}` are orthogonal w.r.t. the weight
`(1-x)^a (1+x)^b` on `[-1,1]`, with two free parameters `a, b` that reshape the weight — putting
more mass near `x=1` (low frequency of `Â`, i.e. low `λ`) or near `x=-1` (high frequency). Chebyshev
is the single special case `a=b=-1/2`; Legendre is `a=b=0`. So Jacobi is the one basis that can
*adapt its weight function* to the graph's spectral density, exactly the `κ(H)→1` knob fact (b)
demands, while the monomial basis (provably non-orthogonal) and Chebyshev (one fixed weight) cannot.

## Evaluation settings

- **Datasets** spanning homophily: homophilic citation graphs Cora, Citeseer, Pubmed; heterophilic
  WebKB (Texas, Cornell, Wisconsin), Wikipedia/actor graphs (Chameleon, Squirrel, Actor); plus a
  synthetic filter-learning benchmark (fit known low/high/band/comb responses on a fixed image graph).
- **Protocol.** Random 60/20/20 train/val/test splits, 10 runs, early stopping; mean test accuracy,
  higher is better. The synthetic benchmark reports filter-fitting squared error.
- **Architecture.** Decoupled MLP (or single linear layer) + degree-`K` polynomial propagation,
  `K=10`, hidden 64, dropout, Adam, with separate learning rate / weight decay for the filter
  coefficients and the encoder.

## Code framework

The filter fills the propagation slot of the decoupled `transform → polynomial propagation →
log-softmax` harness; the empty body is the Jacobi-basis recurrence and how the `K+1` learnable
coefficients combine the propagated signals.

```python
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian, add_self_loops


class CustomProp(MessagePassing):
    """Polynomial graph propagation. Owns the K+1 learnable filter coefficients and
    applies a degree-K polynomial of the (shifted) normalized Laplacian to the input.

    Available primitives:
      - get_laplacian(edge_index, edge_weight, normalization='sym') -> L = I - D^{-1/2} A D^{-1/2}
      - add_self_loops(edge_index, norm, fill_value=-1.0)           -> shift the operator
      - self.propagate(edge_index, x=x, norm=norm)                  -> one sparse mat-vec
    """

    def __init__(self, K, alpha=0.1, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K = K
        # TODO: the trainable filter coefficients and any basis hyperparameters.
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: initialize the coefficients.
        pass

    def forward(self, x, edge_index, edge_weight=None):
        # TODO: build the (shifted) normalized Laplacian, run the basis recurrence,
        #       and combine the propagated signals with the learned coefficients.
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

The MLP, dropout, Laplacian, self-loops, and message passing are given; the empty `CustomProp`
body — the basis recurrence and how the trainable coefficients assemble the filter — is the slot the
method fills.
