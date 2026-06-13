# Context: graph-signal propagation for node classification (circa 2019-2020)

## Research question

Given a graph `G = (V, E)` with `n` nodes, a node-feature matrix `X ∈ R^{n×f}`, and labels on a
subset of nodes drawn from `C` classes, the task is semi-supervised node classification: predict the
labels of the remaining nodes. The data carries two intertwined signals — node features and graph
topology — and a good method must exploit both. The central difficulty is that *the right way to use
the topology depends on the graph*, and existing propagation rules silently assume one regime.

Concretely, two pain points must be resolved at once. First, **homophily versus heterophily**. The
homophily principle says connected nodes tend to share a label (citation graphs, co-purchase graphs);
under it, averaging a node's neighborhood denoises the label signal. But many graphs are
*heterophilic* / disassortative — connected nodes tend to *differ* (dating graphs link opposite sexes,
some protein contact graphs link distinct amino-acid classes, the WebKB webpage graphs link pages of
different roles). On those, neighborhood averaging actively destroys the discriminative signal, which
lives in how a node *contrasts* with its neighbors rather than how it agrees with them. A method that
is hard-wired to average is nonuniversal: it is tuned for one end of the homophily axis and fails at
the other. Second, **depth / over-smoothing**. Useful label information can sit several hops away, so
one would like to propagate over a large neighborhood (many steps). Yet stacking propagation steps is
known to wash features out: after enough steps every node looks alike. So in practice models stay
shallow (2-4 steps) and cannot reach the informative long-range structure.

A solution would have to: handle homophilic and heterophilic graphs alike, ideally tailoring how it
uses the topology to the graph in front of it rather than to a hand-set assumption; remain usable at
large propagation depth without collapsing features into the degree profile; and ideally stay
interpretable, so one can read off from the trained model what kind of graph one is dealing with.

## Background

A graph neural network propagates and transforms node features over the graph topology, usually via
message passing: each layer aggregates a node's neighbors and applies a learned transform. Write
`Ã = A + I` for the adjacency with self-loops, `D̃` for its diagonal degree matrix, and
`Â = D̃^{-1/2} Ã D̃^{-1/2}` for the symmetric normalized adjacency with self-loops. `Â` is the
workhorse aggregation operator; one application replaces each node's vector by a degree-weighted
average of itself and its neighbors.

**Spectral picture.** `Â` is symmetric, so it diagonalizes as `Â = U Λ U^T` with real eigenvalues and
orthonormal eigenvectors. The symmetric Laplacian `L̃_sym = I − Â` is positive semidefinite: for any
`u`, with `f = D̃^{-1/2} u`,
```
u^T L̃_sym u = (1/2) Σ_{i,j} Ã_{ij} (f_i − f_j)^2  ≥ 0,
```
so its eigenvalues are `≥ 0`, hence the eigenvalues of `Â` are `≤ 1`. For a connected graph the
all-ones-in-`f`-coordinates vector gives the bound's minimum: `0` is a simple eigenvalue of `L̃_sym`
with unit eigenvector `π`, `π_i = √D̃_ii / √(Σ_v D̃_vv)`, so `Â` has top eigenvalue `λ_1 = 1` with
eigenvector `π`. The same quadratic form gives `u^T L̃_sym u ≤ 2`, with equality only for a bipartite
graph; self-loops make `G` non-bipartite, so the smallest eigenvalue of `Â` satisfies `λ_n > −1`. Thus
`1 = λ_1 > λ_2 ≥ ... ≥ λ_n > −1`. In graph signal processing the eigenvectors with eigenvalue near
`λ_1 = 1` are the *low-frequency* components (smooth over the graph) and those with small or negative
eigenvalue are the *high-frequency* components (oscillating across edges).

**Over-smoothing, as a fact about `Â^k`.** Because `1 = λ_1 > |λ_i|` for all `i ≥ 2`, raising `Â` to a
power suppresses every eigencomponent except the top one geometrically:
```
lim_{k→∞} Â^k = π π^T  (rank one),   so   Â^k H^(0) = π β^T + o_k(1),   β^T = π^T H^(0).
```
Every node's representation converges to the same direction `β`, scaled only by its degree term `π_i`;
the features become non-discriminative. This is over-smoothing, established for the linear case (Li,
Han & Wu 2018) and for nonlinear rectifier networks (Oono & Suzuki 2020). It is why deep stacks of
propagation degrade.

**Graph filters and polynomials of `Â`.** Any function `g` applied to the spectrum defines a graph
filter `g(Â) = U g(Λ) U^T`, reshaping the eigencomponent at `λ` by `g(λ)`. The polynomials of `Â`,
`Σ_{k=0}^K γ_k Â^k = U (Σ_k γ_k Λ^k) U^T`, are exactly the filters with frequency response
`g_{γ,K}(λ) = Σ_k γ_k λ^k`. Polynomial filters can approximate any graph filter (Shuman et al. 2013),
and larger `K` gives a finer approximation. One known property of this particular representation: the
powers `{Â, Â^2, …, Â^K}` form a *non-orthogonal* (power / monomial) basis, so the map from
coefficients to response can be poorly conditioned as `K` grows — a numerical cost the power basis pays
relative to better-conditioned polynomial bases such as Chebyshev.

**Generalized PageRank (GPR).** In unsupervised seed-expansion clustering, given a seed node `s` with
one-hot vector `H^(0)_v = δ_{vs}`, the GPR score `Σ_{k=0}^∞ γ_k Â^k H^(0)` accumulates contributions
from walks of every length, weighted by scalars `γ_k` (the GPR weights). Personalized PageRank and
heat-kernel PageRank are particular weight choices. Li, Chien & Milenkovic (2019) introduced *Inverse
PR* and showed that, with the right `γ_k`, long walks help clustering more than previously assumed — on
homophilic structure; the right weights for heterophilic structure were not known.

A standard tool for `homophily` level is `H(G) = (1/|V|) Σ_v (# neighbors of v with v's label) / (deg
v)`: `H(G) → 1` is strongly homophilic, `H(G) → 0` strongly heterophilic. (It is a coarse measure: two
graphs can both have `H(G) = 0` yet carry very different amounts of label information in their
topology.)

## Baselines

**GCN (Kipf & Welling 2017).** The standard graph-convolution stack:
```
H^(k) = ReLU(Â H^(k-1) W^(k)),   H^(0) = X,   Ẑ = softmax(Â H^(K-1) W^(K)).
```
Each layer couples one propagation step `Â·` with a learned transform `W^(k)` and a nonlinearity.
**Gap:** a single application of `Â` is a low-pass average, so the model is biased toward homophily;
and because transform and propagation are fused, reaching `k` hops requires `k` stacked layers, which
drives `Â^k H → π β^T` and over-smooths. Practical depth is 2-4 layers; long-range structure is out of
reach, and high-frequency (heterophilic) signal is suppressed by the averaging.

**SGC (Wu et al. 2019).** Drop every nonlinearity from a `K`-layer GCN; the stacked `Â`'s collapse into
one operator and the model becomes a linear classifier on pre-propagated features:
```
Ẑ = softmax(Â^K X Θ).
```
Fast and simple. In terms of the polynomial response above this is the single-term filter `γ_k =
δ_{kK}` — all weight on the `K`-th hop. **Gap:** the propagation is fixed and is pure `Â^K` (low-pass,
and at large `K` exactly the over-smoothing operator), with no ability to weight hops or to keep
high-frequency content; nothing about it adapts to the graph.

**APPNP / PPNP (Klicpera, Bojchevski & Günnemann 2018).** Decouple prediction from propagation: first
compute per-node predictions `H = f_θ(X)` with an MLP, then propagate them with personalized PageRank.
Exact PPNP applies the closed-form operator
```
Ẑ = softmax( α (I − (1−α) Â)^{-1} H ),
```
and APPNP approximates it by power iteration `Z^(k+1) = (1−α) Â Z^(k) + α H`, `Z^(0) = H`, for `K`
steps (default `α = 0.1`, `K = 10`). Expanding the inverse as a geometric series,
`α (I − (1−α)Â)^{-1} = Σ_k α(1−α)^k Â^k`, so APPNP is the polynomial response with the *fixed* weights
`γ_k = α(1−α)^k` (and the truncation tail `γ_K = (1−α)^K`). Because `f_θ` is applied once and the limit
operator `Π_ppr = α(I − (1−α)Â)^{-1}` still depends on `H`, APPNP provably does not over-smooth even as
`K → ∞`. **Gap:** the weights are fixed, all positive, and geometrically decreasing — a low-pass
response by construction; on heterophilic graphs, where the discriminative signal is high-frequency, it
suppresses exactly what is needed. And its escape from over-smoothing is set by `α` alone, independent
of the node labels, so it cannot tailor its propagation to the actual label structure of the graph in
front of it.

**GCN-Cheby (Defferrard et al. 2016).** Each layer is itself a polynomial graph filter in the
Chebyshev basis, `Σ_k θ_k T_k(L̃)`, propagating multiple hops per layer (typically order 2). This is
more expressive per layer than plain GCN. **Gap:** still shallow in practice; the filter is per-layer
rather than a single interpretable global response, and the coefficients are entangled with the
per-layer transforms.

**JK-Net (Xu et al. 2018).** Run several GCN layers and combine their outputs at the end
(concatenation, max, or an LSTM), so the final representation can draw on multiple hop-depths rather
than only the last. **Gap:** the hops being combined are still full GCN layers (transform and
propagation fused), so the combination is not a clean, readable set of per-hop filter coefficients, and
the model is not deep in practice.

## Evaluation settings

Semi-supervised node classification in the transductive setting on standard benchmarks, spanning the
homophily axis:

- **Homophilic:** the citation graphs Cora (2,708 nodes, 7 classes), CiteSeer (3,327 nodes, 6 classes),
  PubMed; the Amazon co-purchase graphs Computers and Photo.
- **Heterophilic:** WebKB webpage graphs Texas and Cornell (183 nodes, 5 classes each), the Actor
  co-occurrence graph, and the Wikipedia graphs Chameleon and Squirrel. These come with a low `H(G)`
  (e.g. Texas `H(G) ≈ 0.06`, Cornell `≈ 0.30`) versus high for the citation graphs (Cora `≈ 0.83`).
- **Synthetic spectrum:** the contextual stochastic block model (cSBM, Deshpande et al. 2018), which
  smoothly interpolates the relative informativeness of features versus topology via a parameter `φ`,
  with `φ < 0` heterophilic, `φ > 0` homophilic, so a method can be probed across the whole
  homophily/heterophily range; cSBM has a known information-theoretic recovery threshold.

Protocol: random train/validation/test splits, repeated many times with different initializations; a
sparse split (`2.5% / 2.5% / 95%`) closer to the original GCN setting and a dense split
(`60% / 20% / 20%`) used for the heterophilic study; early stopping on validation loss; Adam optimizer.
The natural feature transform is a 2-layer MLP with 64 hidden units; propagation depth on the order of
`K = 10`. Metric: mean test accuracy (higher is better). The evaluation harness, the splits, and the
metric are fixed; only the propagation module varies.

## Code framework

The classifier is fixed: transform each node's features with a small MLP, then run a graph-propagation
module, then a `log_softmax` head — the same MLP-then-propagate pipeline the decoupled baselines use.
The MLP (`lin1 → ReLU → dropout → lin2`), the dropout schedule, the `log_softmax`, the
`gcn_norm`/Laplacian utilities, and the single-step message-passing primitive `self.propagate` already
exist. The open question is what graph-propagation rule should occupy the slot after the MLP.

```python
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GraphPropagation(MessagePassing):
    """Graph propagation slot."""

    def __init__(self, K, alpha=0.1, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K = K
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        # TODO: the propagation rule we will design.
        pass

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j        # one sparse mat-vec: (Â @ x)


class NodeClassifier(nn.Module):
    """Existing pipeline: transform features with an MLP, then propagate."""

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super().__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = GraphPropagation(K, alpha)
        self.dropout = dropout
        self.dprate = dprate                 # dropout rate on the propagation input

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))             # f_θ: per-node feature transform
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)
```

The MLP, dropout, and head are in place; the propagation rule is the single slot to fill.
