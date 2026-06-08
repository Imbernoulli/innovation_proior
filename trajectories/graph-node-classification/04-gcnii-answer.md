**Problem (from step 3).** All three baselines share one wall: they are two-layer models, capped at a
two-hop receptive field. gcn won (Cora 0.8207, CiteSeer 0.7177, PubMed 0.7863) but cannot fetch evidence
more than two hops away — plausibly why CiteSeer (sparsest) stays worst. Stacking more layers should
widen the field but instead over-smooths: $K$ renormalized layers realize $\tilde{\mathbf P}^{K}$, which
contracts every signal to the degree-only stationary vector
$\boldsymbol\pi\propto\tilde{\mathbf D}^{1/2}\mathbf 1$, and a ResNet residual to the *previous* layer
only slows the lazy walk to the same fixed point.

**Key idea.** Two changes make depth a resource:
$$\mathbf H^{(\ell+1)}=\sigma\Big(\big((1-\alpha)\tilde{\mathbf P}\mathbf H^{(\ell)}+\alpha\mathbf H^{(0)}\big)\big((1-\beta_\ell)\mathbf I+\beta_\ell\mathbf W^{(\ell)}\big)\Big).$$
- **Initial residual** to $\mathbf H^{(0)}$ (a learned map of the features), not the previous layer: the
  deep limit becomes the input-carrying PageRank diffusion $\alpha(\mathbf I-(1-\alpha)\tilde{\mathbf
  P})^{-1}\mathbf H^{(0)}$ instead of the degree vector. $\alpha=0.1$.
- **Identity mapping** $(1-\beta_\ell)\mathbf I+\beta_\ell\mathbf W^{(\ell)}$,
  $\beta_\ell=\log(\frac{\lambda}{\ell}+1)$ decaying with depth: holds the max singular value $s\approx1$
  so the $s^{K}$ subspace collapse is defused; strong regularization on $\mathbf W$ is safe because the
  linear-residual optimum has small norm and a unique global minimum. $\lambda=0.5$.

**Why it works.** Initial residual stops the *propagation* over-smoothing; identity mapping stops the
*weights* collapsing rank / overfitting the ~20 labels per class. Together a $K$-layer model expresses an
order-$K$ polynomial filter with *arbitrary* coefficients (vanilla GCN's are fixed), so it can learn not
to over-smooth and genuinely use a deep receptive field.

**Scaffold edit / hyperparameters.** One GCNII conv in `CustomMessagePassingLayer`; `CustomGNN` =
input-FC → $L$ convs (all $H\to H$, carrying $\mathbf H^{(0)}$) → output-FC, ignoring the
passed `num_layers=2`. The fixed hidden width ($H=64$) and the $1.05\times$-largest-baseline parameter
budget cap depth at $L=16$ (64 layers would exceed Cora's ~194k budget) — still 8× deeper than every
baseline and past the over-smoothing wall. $\alpha=0.1$, $\lambda=0.5$; scaffold-fixed dropout 0.5, lr
0.01, single weight decay (no split $L_2$), relying on the identity mapping for weight regularization.

**The bar to beat.** gcn: Cora 0.8207, CiteSeer 0.7177, PubMed 0.7863. Falsifiable: the biggest gain
should be on CiteSeer (where the two-hop wall most plausibly capped the baselines — clearly past 0.7177);
Cora up past ~0.82; PubMed past 0.7863 by the smallest margin. Any dataset that *drops* with depth, or a
CiteSeer that fails to beat 0.7177, would falsify the construction. No leaderboard row exists here; the
endpoint is the model that goes 16 hops deep without over-smoothing.

```python
# EDITABLE region of custom_nodecls.py — finale: deep GCN via initial residual + identity mapping
import math


class CustomMessagePassingLayer(MessagePassing):
    """One deep conv: ((1-alpha) P̃ H + alpha H0) ((1-theta) I + theta W), theta = log(lambda/l + 1).
    P̃ = D̃^{-1/2} Ã D̃^{-1/2} (renormalization trick)."""

    def __init__(self, channels: int):
        super().__init__(aggr="add")
        self.weight = nn.Parameter(torch.empty(channels, channels))   # H x H, depth-independent size
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor, edge_index: Adj, h0: Tensor,
                alpha: float, lam: float, layer: int) -> Tensor:
        theta = math.log(lam / layer + 1.0)                 # identity-mapping strength, decays with depth
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        hi = self.propagate(edge_index, x=x, norm=norm)     # P̃ H^{(l)}
        support = (1 - alpha) * hi + alpha * h0             # initial residual: anchor to H^{(0)}
        # (1-theta) support + theta (support W) == support ((1-theta) I + theta W)
        return (1 - theta) * support + theta * torch.mm(support, self.weight)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class CustomGNN(nn.Module):
    """Deep GCN: input FC -> L identity-mapped, initial-residual convs -> output FC.
    Sets its own depth (L) to go past the over-smoothing wall within the parameter budget;
    the passed num_layers is ignored."""

    NUM_PROP_LAYERS = 16          # 8x deeper than the baselines; fits the 1.05x-largest-baseline budget
    ALPHA = 0.1                   # initial-residual fraction
    LAMBDA = 0.5                  # identity-mapping schedule scale

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            CustomMessagePassingLayer(hidden_channels) for _ in range(self.NUM_PROP_LAYERS))
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        h0 = F.relu(self.fc_in(x))                          # H^{(0)} = transformed features
        h = h0
        for i, conv in enumerate(self.convs):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(conv(h, edge_index, h0, self.ALPHA, self.LAMBDA, i + 1))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc_out(h)                               # class logits (loop applies cross-entropy)
```
