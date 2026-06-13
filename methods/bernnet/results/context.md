# Context: polynomial spectral filters on graphs (circa 2020-2021)

## Research question

A graph neural network propagates a node signal `x` across the graph by applying a function of
the graph's structure. Almost every popular model can be read as a *polynomial graph filter*: it
forms `sum_{k=0}^K w_k L^k x`, where `L = I - D^{-1/2} A D^{-1/2}` is the symmetric normalized
Laplacian (or, equivalently, `sum_k c_k P^k x` with `P = D^{-1/2} A D^{-1/2} = I - L` the
normalized adjacency). In the spectral domain this is `U diag[g(lambda_1), ..., g(lambda_n)] U^T x`
where `L = U Lambda U^T` and `g(lambda) = sum_k w_k lambda^k` is the *spectral response* the model
applies to each eigen-component. The eigenvalues of `L` lie in `[0, 2]`; small `lambda` is the
smooth / low-frequency end, large `lambda` the oscillatory / high-frequency end.

The precise goal is a single polynomial filter that simultaneously: (1) can approximate an
*arbitrary* response shape over `[0, 2]` — not only low-pass, but high-pass, band-pass,
band-rejection, comb — to any precision as the order `K` grows; (2) is *valid*, meaning its
response stays in the legal range a propagation/energy filter is allowed to take, rather than
producing negative responses that have no meaning as a filter; (3) is *interpretable*, so that
after training one can read off what shape was learned and, before training, dial in a desired
shape by hand; and (4) is end-to-end learnable from `(x, z)` pairs or from node labels, dropping
into the same "transform features with an MLP, then propagate" pipeline already used by the
strongest baselines. Each existing filter below achieves a subset; none achieves all four.
Closing that gap is the problem.

## Background

Spectral graph filtering rests on the eigendecomposition `L = U Lambda U^T`. A filter applies a
scalar function `g` to the spectrum, `g(L) = U diag[g(lambda_i)] U^T`. Computing `U` is `O(n^3)`
and not localized, so the field uses *polynomial* filters `g(L) = sum_k w_k L^k`, which need only
sparse matrix-vector products (`O(K|E|)` work, `K`-hop localized) and never form `U`. The whole
design question becomes: how to choose / parameterize the coefficients `w_k` so that the induced
`g(lambda) = sum_k w_k lambda^k` has the response shape you want.

Two facts from the period frame the difficulty. First, the **homophily/heterophily split** in the
data. On homophilic graphs (citation networks: connected nodes share labels) the useful signal is
smooth, so a low-pass response — keep small `lambda`, suppress large `lambda` — is what helps; the
classical models were built for exactly this. On heterophilic graphs (web-page graphs, certain
Wikipedia graphs: connected nodes often differ) the discriminative signal lives at *higher*
frequencies, so a filter locked to low-pass actively throws away the information, and a plain MLP
that ignores the graph can beat a low-pass GNN. A filter that works across both regimes cannot be
hard-wired to one frequency band.

Second, the **graph-optimization view of propagation** (Zhou, Bousquet, Lal, Weston & Schölkopf,
NeurIPS 2004; unified by Zhu, Wang, Shi, Ji & Cui, WWW 2021). Many propagations are the minimizer
of an energy
```
f(z) = (1 - alpha) z^T gamma(L) z + alpha || z - x ||_2^2,
```
a fit-to-input term plus a smoothness penalty `gamma(L)` acting on the spectrum. For
`gamma(L) = L` the optimum is `z* = alpha (I - (1-alpha)P)^{-1} x = sum_k alpha(1-alpha)^k P^k x`,
whose truncation is exactly Personalized-PageRank propagation; a heat-kernel `gamma` gives the
diffusion / heat-kernel propagations. This says propagation = solving a regularized smoothing
problem, and the *response* `g` is then a function of the chosen `gamma`. Two diagnostic
observations about this design space are load-bearing. (a) For the energy to be a well-posed
(convex, bounded-below) smoothing problem, `gamma(L)` must be positive semidefinite, i.e.
`gamma(lambda) >= 0` on `[0, 2]`; otherwise `f(z)` has no minimum (it runs to `-infinity` along an
eigenvector with negative `gamma`) and its stationary point is a saddle. (b) Under
`gamma(lambda) >= 0`, the induced response `h(lambda) = alpha / (alpha + (1-alpha) gamma(lambda))`
satisfies `0 < h(lambda) <= 1` for `lambda in [0, 2]` — it never leaves `(0, 1]`. So a filter that
is meant to approximate a propagation of this form should itself land in `[0, 1]` across the whole
spectrum; a polynomial that dips negative or overshoots is approximating something that is not a
valid smoothing at all. This is the diagnostic that flags some learned filters as "ill-posed."

The Laplacian convention also matters: GCN's renormalization trick `tilde P = (I + D)^{-1/2}
(I + A)(I + D)^{-1/2}` shrinks the spectrum and softens the negative-response problem, but the
top eigenvalue of `tilde L = I - tilde P` still exceeds 1, so the corresponding response can still
go negative. Shrinking the spectrum is a patch, not a guarantee.

## Baselines

These are the prior polynomial filters a new design would be measured against and react to.

**GCN (Kipf & Welling, ICLR 2017).** Propagate one hop with the renormalized adjacency, `z = P x =
(I - L) x`, a fixed first-order filter. Its response is `g(lambda) = 1 - lambda`, a fixed linear
*low-pass*: it weights small `lambda` near 1 and suppresses large `lambda`. Stacking layers raises
the order but ties the coefficients to a single shape, and the response `1 - lambda` becomes
*negative* once `lambda > 1` — half the spectrum gets a sign-flipped weight, which is the
ill-posed regime above; deep stacks also oversmooth (everything collapses toward the dominant
low-frequency component). **Gap:** one fixed low-pass shape, and not even a valid one on the upper
half of the spectrum.

**APPNP (Klicpera, Bojchevski & Günnemann, ICLR 2019).** Decouple the feature transform from
propagation: run an MLP to get predictions, then propagate them with truncated Personalized
PageRank, `z = sum_{k=0}^K alpha(1-alpha)^k P^k x`, with teleport probability `alpha`. This is the
`gamma(L) = L` optimum above, so its response is the monotone decreasing
`h(lambda) = alpha / (alpha + (1-alpha) lambda)`-shaped curve — a *fixed low-pass* with one tunable
knob `alpha`. **Gap:** the shape is pinned to low-pass; there is no setting of `alpha` that turns
it into a high-pass or band filter, so it cannot serve heterophilic data.

**ChebNet (Defferrard, Bresson & Vandergheynst, NeurIPS 2016).** Learn the filter in the Chebyshev
basis: `g(L) = sum_{k=0}^{K-1} theta_k T_k(tilde L)`, where `tilde L = 2L/lambda_max - I` rescales
the spectrum into `[-1, 1]` and the Chebyshev polynomials obey the stable recurrence
`T_k(y) = 2 y T_{k-1}(y) - T_{k-2}(y)`, `T_0 = 1`, `T_1 = y`. The `theta_k` are free trainable
coefficients, computed by repeated sparse products (`O(K|E|)`), and the orthogonality of the
Chebyshev basis controls approximation error. In principle the free `theta_k` can fit many shapes.
**Gap:** the coefficients are unconstrained, so the learned `g(lambda)` can take negative values
(ill-posed), and a Chebyshev coefficient `theta_k` is an abstract projection onto `T_k` — it does
not tell you what response the filter applies at any particular frequency, so what was learned is
opaque.

**GPR-GNN (Chien, Peng, Li & Milenkovic, ICLR 2021).** Learn the filter directly in the
*monomial* basis: `h(P) = sum_{k=0}^K gamma_k P^k`, with the `gamma_k` ("Generalized PageRank"
weights) trained by gradient descent. The key theorem is about the non-negative monomial regime:
when the weights are normalized (`sum_k gamma_k = 1`), non-negative, and nontrivial, the resulting
filter is low-pass. To become high-pass and so handle heterophily, this basis has to allow some
negative weights. **Gap:** precisely because the `gamma_k` are unconstrained (and must be allowed
negative to be expressive), the learned filter has no validity guarantee — it can produce negative
spectral responses — and the monomial basis is both numerically delicate (powers of `P` on a wide
spectrum) and uninterpretable: the work can only show that a small subset of learned weight
sequences corresponds to recognizable low/high-pass filters, so in general one cannot read the
learned shape off the `gamma_k`.

The common thread: the *fixed* filters (GCN, APPNP) have clear smoothing interpretations but are
stuck on one low-pass shape;
the *learned* filters (ChebNet, GPR-GNN) gain expressiveness only by letting their coefficients
roam unconstrained, which buys flexibility at the cost of validity and interpretability. Nobody has
a parameterization where expressiveness, validity, and interpretability hold at once.

## Evaluation settings

The natural yardsticks already in use at the time:

- **Filter-fitting on image grids.** Treat a `100×100` image as a 2D regular 4-neighborhood grid
  graph (a `10,000`-node graph, pixel intensities as the signal). Apply a *known* target response
  to the spectrum (low-pass `exp(-10 lambda^2)`, high-pass `1 - exp(-10 lambda^2)`, band-pass
  `exp(-10(lambda-1)^2)`, band-rejection `1 - exp(-10(lambda-1)^2)`, comb `|sin(pi lambda)|`) to
  produce the supervised target `z`, then ask each model to recover the filter from `(x, z)`. All
  models use two convolution units plus a linear layer, ~2k trainable parameters, no
  regularization/dropout, Adam at lr 0.01, up to 2000 epochs with early stopping. Metric: sum of
  squared error and `R^2` between output and target (this isolates pure filter-approximation
  capacity).
- **Node classification on real graphs.** Homophilic: Cora, CiteSeer, PubMed citation networks
  (Sen et al. 2008; Yang et al. 2016) and the Amazon co-purchase graphs Computers, Photo (McAuley
  et al. 2015). Heterophilic: Wikipedia graphs Chameleon, Squirrel (Rozemberczki et al. 2021), the
  Actor co-occurrence graph, and WebKB web-page graphs Texas, Cornell (Pei et al. 2020). Protocol
  (following GPR-GNN): random 60%/20%/20% train/val/test, 10 random splits shared across models,
  full-supervised, early stopping. The pipeline is a 2-layer MLP (64 hidden units) feeding the
  propagation. Metric: mean test accuracy over the 10 splits, higher is better. Learning rate
  searched over `{0.001, 0.002, 0.01, 0.05}`, weight decay over `{0.0, 0.0005}`, with separate
  rates/dropout allowed for the linear and propagation parts.
- **Implementations** for baselines come from the standard graph-learning library (PyTorch
  Geometric, Fey & Lenssen 2019); the propagation step is built from its message-passing
  primitive.

## Code framework

The new filter plugs into the same MLP-then-propagate pipeline the baselines use. The feature
transform (a 2-layer MLP), the dropout schedule, the `log_softmax` head, and the message-passing
substrate already exist; what is *not* settled is the propagation layer's parameterization or how it
will combine sparse products of the graph operator. The scaffold therefore keeps one generic
propagation slot open.

```python
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian


class PropLayer(MessagePassing):
    """Graph propagation layer. The parameterization and sparse-product recipe
    are the open design slot."""

    def __init__(self, K, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K = K                          # polynomial order (number of hops)
        # TODO: trainable parameters for the graph filter
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: initialize the filter parameters
        pass

    def forward(self, x, edge_index, edge_weight=None):
        # symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
        edge_index_L, norm_L = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim))
        # available primitive: self.propagate(edge_index_L, x=x, norm=norm_L)
        # computes one sparse step with the current operator weights.
        # TODO: design the filter and combine the needed sparse products.
        out = x
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j       # one sparse mat-vec: (operator @ x)


class FilterModel(torch.nn.Module):
    """Existing pipeline: transform node features with an MLP, then propagate."""

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 dropout=0.5, dprate=0.5):
        super().__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = PropLayer(K)
        self.dropout = dropout
        self.dprate = dprate                # dropout rate on the propagation input

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

The MLP, dropout, and head are fixed; `PropLayer.forward` is the one empty slot where the
filter's parameterization and its construction from sparse Laplacian products will live.
