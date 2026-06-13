# GPR-GNN, distilled

GPR-GNN (Generalized PageRank Graph Neural Network) decouples representation from propagation —
transform node features with an MLP, then propagate with a learnable Generalized PageRank, and makes
the per-hop propagation weights `γ_k` **free, signed parameters** trained end-to-end. The propagation
`Z = Σ_{k=0}^K γ_k Â^k H^(0)` is a degree-`K` polynomial graph filter in the monomial basis; learning
the `γ_k` is learning the filter's frequency response. Because the `γ_k` may be negative, the response
can be low-pass (good for homophilic graphs) or high-pass (good for heterophilic graphs), chosen from
the labels rather than hard-wired. Because over-smoothing hops get pushed back toward zero by the loss
gradient, the model can attempt deep propagation without letting collapsed hops dominate.

## Problem it solves

Semi-supervised node classification on graphs that may be homophilic (neighbors share labels) or
heterophilic (neighbors differ), without per-dataset tuning of the propagation rule and without the
shallow-depth limit imposed by over-smoothing. Fixed low-pass aggregators (GCN, SGC, APPNP) are biased
toward homophily and lose the high-frequency signal heterophily needs; stacking propagation to reach
far hops over-smooths features into the degree profile.

## Key idea

With `Â = D̃^{-1/2} Ã D̃^{-1/2}` the symmetric GCN-normalized adjacency with self-loops:

1. **Decouple:** `H^(0) = f_θ(X)` (a 2-layer MLP, per node, no graph), then propagate.
2. **Learnable Generalized PageRank propagation:** `Z = Σ_{k=0}^K γ_k H^(k)`, `H^(k) = Â H^(k-1)`,
   with `γ_k ∈ R` trained jointly with `θ`. Final prediction `softmax(Z)`.
3. **Signed coefficients span the homophily axis.** `Σ_k γ_k Â^k = U g_{γ,K}(Λ) U^T` with response
   `g_{γ,K}(λ) = Σ_k γ_k λ^k`. The spectrum of `Â` is `1 = λ_1 > λ_2 ≥ … ≥ λ_n > −1` (top eigenvalue
   `1`, eigenvector `π_i = √D̃_ii/√(Σ_v D̃_vv)`; self-loops make `|λ_n| < 1`).

## Why signed weights matter (low-pass vs high-pass)

- **Nonnegative weights are provably low-pass.** If `γ_k ≥ 0`, `Σ_k γ_k = 1`, and some `γ_{k'} > 0`
  with `k' ≥ 1`, then `g(λ_1) = 1` and for `i ≥ 2`, since `|λ_i| < 1`,
  `|g(λ_i)| ≤ Σ_k γ_k |λ_i|^k < Σ_k γ_k = 1` (strict because `|λ_i|^k < 1` for `k ≥ 1` and some
  `γ_{k'>0} > 0`). So `|g(λ_i)/g(λ_1)| < 1` — the low-frequency component dominates. APPNP
  (`γ_k = α(1−α)^k`) and SGC (`γ_k = δ_{kK}`) are nonnegative, hence intrinsically low-pass, hence
  inadequate on heterophily. For finite APPNP the last truncated coefficient is `γ_K = (1−α)^K`,
  still nonnegative.
- **Signed weights give high-pass.** For `γ_k = (−α)^k`, `α ∈ (0,1)`, `K → ∞`, `g(λ) = 1/(1+αλ)`, and
  `|g(λ_i)/g(λ_1)| = |(1+α)/(1+αλ_i)| > 1` for `i ≥ 2` (since `λ_i < 1`), peaking as `λ → −1`. The
  high-frequency components now dominate.

A learnable signed `γ` interpolates between these, so one model handles both regimes; the learned `γ_k`
are also directly interpretable (alternating-sign = high-pass/heterophilic; positive = low-pass).

## Why deep propagation is safe (label-guided over-smoothing escape)

For hops `k` in the over-smoothing regime, `H^(k) = Â^k H^(0) = π β^T + o_k(1)`, `β^T = π^T H^(0)`.
With temperatured softmax `P̂ = softmax_η(Z)` and cross-entropy `L = Σ_{i∈T} −log⟨P̂_{i:}, Y_{i:}⟩`,
chain rule gives `∂L/∂γ_k = Σ_{i∈T} η ⟨P̂_{i:} − Y_{i:}, H^(k)_{i:}⟩ = Σ_{i∈T} η π_i ⟨P̂_{i:} − Y_{i:},
β⟩ + o_k(1)`. In the over-smoothed state the prediction is node-independent; as `η → ∞`,
`softmax_η → 𝟙[·]`, so for the dominant `γ_k > 0`,
```
∂L/∂γ_k = Σ_{i∈T} η π_i ( max_j β_j − β_{ℓ(i)} ) + o_k(1) + o_η(1) ≥ 0,
```
and for the dominant `γ_k < 0` the argmax flips to argmin:
```
∂L/∂γ_k = Σ_{i∈T} η π_i ( min_j β_j − β_{ℓ(i)} ) + o_k(1) + o_η(1) ≤ 0,
```
with `π_i > 0` (self-loops). Apart from degenerate tied collapsed scores, a training set containing
every class makes the inequalities strict, so gradient descent with a decreasing learning rate drives
`|γ_k| → 0` for over-smoothing hops until they no longer dominate `Z`. The escape is *label-guided*
unlike APPNP's label-blind, `α`-fixed escape.

## Defaults and why

- **`K = 10`** propagation hops — enough depth to approximate the optimal filter / reach the relevant
  neighborhood; over-smoothing hops are muted by the sign of their loss gradients.
- **Uniform init `γ_k = 1/(K+1)`** — equal weight on every hop. This is a low-pass initial filter
  because the weights are nonnegative, but the coefficients are free parameters and can move to
  alternating signs when the labels call for high-pass behavior. (Other inits — PPR `α(1−α)^k`,
  SGC `δ_{kK}`, normalized random — act as implicit priors when labels are scarce.)
- **No weight decay on `γ`** (`temp`) — the `(K+1)`-vector *is* the filter; decaying it would shrink
  the learned response. Use a separate optimizer group for the propagation parameters with zero weight
  decay, while ordinary weight decay applies to the MLP weights.
- **GCN-normalized self-looped `Â`** — symmetric (real spectrum), `λ_1 = 1`, `|λ_n| < 1` (self-loops
  kill the bipartite `λ = −1`), which makes the low/high-pass dichotomy and the over-smoothing limit
  well-defined.
- **Monomial basis `Â^k`** — directly interpretable (each `γ_k` is the weight on `k`-hop propagation)
  and equals Generalized PageRank; cost is a non-orthogonal basis that can be ill-conditioned at large
  `K`, accepted for the interpretability instead of switching to a better-conditioned polynomial basis.

## Final algorithm

```
H^(0) = f_θ(X)                          # MLP, per node, no graph
for k = 1..K:  H^(k) = Â H^(k-1)        # sparse propagation step
Z = Σ_{k=0}^K γ_k H^(k)                 # learnable polynomial filter (signed γ_k)
P̂ = softmax(Z)                          # prediction
# train γ_0..γ_K jointly with θ by cross-entropy; γ init 1/(K+1), no wd on γ
```

## Relation to prior methods

- **APPNP** = this model with `γ_k` frozen at `α(1−α)^k` (PPR; nonnegative, decreasing → low-pass).
- **SGC** = this model with `γ_k = δ_{kK}` and the MLP reduced to linear (single deep hop, low-pass).
- **GCN** = repeated `Â`-then-transform layers (fused, low-pass, over-smooths with depth).
- **JK-Net** = combine multiple GCN-layer outputs at the end, but the hops are full layers, not clean
  signed filter coefficients.
The new ingredient over all of these is making the per-hop weights free, **signed**, and learned, which
gives one filter family spanning low-pass and high-pass and a label-guided escape from over-smoothing.

## Working code

Filling the propagation slot of the MLP-then-propagate pipeline; `temp` holds the `(K+1)` GPR weights
`γ_0..γ_K`, and propagation is `K` sparse `Â·x` message-passing steps with the running accumulation
`Σ_k γ_k Â^k x`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GPR_prop(MessagePassing):
    """Generalized PageRank propagation: learnable monomial-basis polynomial
    filter  Z = sum_{k=0}^{K} gamma_k * Â^k x,  with free (possibly negative)
    gamma_k so the response spans low-pass (homophily) to high-pass (heterophily)."""

    def __init__(self, K, alpha=0.1, Gamma=None, **kwargs):
        super(GPR_prop, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.alpha = alpha                            # kept for compatibility with common training args
        self.Gamma = Gamma
        if Gamma is None:
            temp = torch.ones(K + 1, dtype=torch.float) / (K + 1)
        else:
            temp = torch.as_tensor(Gamma, dtype=torch.float)
        self.temp = Parameter(temp)                   # GPR weights gamma_0..gamma_K

    def reset_parameters(self):
        # uniform 1/(K+1): equal-hop low-pass start; weights remain unconstrained
        if self.Gamma is None:
            nn.init.constant_(self.temp, 1.0 / (self.K + 1))
        else:
            gamma = torch.as_tensor(
                self.Gamma, dtype=self.temp.dtype, device=self.temp.device)
            self.temp.data.copy_(gamma)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(                  # Â = D̃^{-1/2} Ã D̃^{-1/2}
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        hidden = x * self.temp[0]                      # gamma_0 * Â^0 x
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)   # x <- Â x
            hidden = hidden + self.temp[k + 1] * x           # + gamma_{k+1} Â^{k+1} x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j                  # sparse mat-vec (Â @ x)


class GPRGNN(torch.nn.Module):
    """GPR-GNN: MLP feature transform f_theta, then learnable GPR propagation."""

    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = GPR_prop(args.K, args.alpha, getattr(args, "Gamma", None))
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))                       # H^(0) = f_theta(X)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


def build_optimizer(model, args):
    return torch.optim.Adam([
        {"params": model.lin1.parameters(), "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": model.lin2.parameters(), "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": model.prop1.parameters(), "weight_decay": 0.0, "lr": args.lr},
    ], lr=args.lr)
```
