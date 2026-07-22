Graph neural networks for semi-supervised node classification face a pair of coupled difficulties. First, the standard propagation rule is a weighted average over a node's neighborhood, which is exactly the right thing when neighbors share labels (homophily) but actively destroys the signal when neighbors tend to differ (heterophily). Second, even when a larger neighborhood is informative, stacking many propagation steps drives representations toward the degree profile and washes out node-specific features, a phenomenon known as over-smoothing. Existing fixes only address part of the problem. APPNP decouples feature transformation from propagation and uses personalized PageRank weights, but those weights are fixed and nonnegative, so the resulting filter remains low-pass and therefore biased toward homophily. SGC collapses a deep GCN into a single power of the normalized adjacency, which is even more aggressively low-pass and over-smoothed. Chebyshev filters offer richer, better-conditioned frequency responses, but at the cost of the direct per-hop interpretability that comes from working in the plain monomial basis. What is needed is a propagation rule that can adapt its frequency response to the graph at hand while remaining computationally simple, directly interpretable, and deep without collapsing.

The method is GPR-GNN, the Generalized PageRank Graph Neural Network. The core idea is to decouple the prediction from the propagation: a small MLP first transforms each node's raw features into per-class scores independently of the graph, and a separate propagation module then spreads those scores over the graph using a learnable polynomial filter. Concretely, with Â = D̃^{-1/2} Ã D̃^{-1/2} the symmetric normalized adjacency with self-loops, propagation is Z = Σ_{k=0}^{K} γ_k Â^k H^{(0)}, where H^{(0)} is the MLP output and each γ_k is a free, real, learnable scalar. This is a degree-K polynomial graph filter with frequency response g(λ) = Σ_k γ_k λ^k applied to the eigenvalues of Â. Because the coefficients are signed, the filter can be low-pass when the γ_k are nonnegative (reinforcing neighbor agreement, as homophily needs) or high-pass when the signs alternate (amplifying contrast between neighbors, as heterophily needs). The same model therefore spans both regimes, with the exact shape learned end-to-end from the labels rather than imposed by the designer.

A nonnegative weighted filter is provably low-pass: if every γ_k is nonnegative and at least one higher-hop weight is positive, then |g(λ_i)| < g(λ_1) = 1 for every i ≥ 2, because |λ_i| < 1. That is why APPNP and SGC, whose coefficients are nonnegative, cannot represent the high-frequency responses that heterophilic graphs require. Allowing negative coefficients breaks this restriction. For example, γ_k = (-α)^k gives the high-pass response g(λ) = 1/(1 + αλ), whose gain exceeds one for all non-top frequencies. The cost of this freedom is the well-known ill-conditioning of the monomial basis at large K, since the powers Â, Â^2, ..., Â^K are not orthogonal and become nearly collinear once the deep hops approach the over-smoothing limit; a Chebyshev or other orthogonal basis would condition better. But at the moderate depths used in practice (K = 10) the interpretability of reading each γ_k directly as the weight on exactly k-hop propagation outweighs that conditioning cost — it is a deliberate trade for transparency, not an oversight.

GPR-GNN also escapes over-smoothing in a label-driven way. When a hop k is deep enough that Â^k H^{(0)} has collapsed to the rank-one degree profile πβ^T, the cross-entropy gradient with respect to γ_k points in the same direction as γ_k itself, so gradient descent pushes the weight of an uninformative deep hop toward zero. This is a different guarantee from APPNP's fixed geometric decay: the model can afford a large K because the optimizer, guided by the labels, mutes the hops that would otherwise dominate and collapse the representation. The coefficients are initialized uniformly at γ_k = 1/(K+1), a simple dataset-agnostic starting point that the optimizer can move toward either low-pass or high-pass behavior. Weight decay is applied to the MLP weights but not to the propagation coefficients, since shrinking γ would impose an unwanted all-zero filter prior. The implementation below fills the propagation slot of the standard MLP-then-propagate pipeline with K sparse message-passing steps and a running accumulation of the weighted hop contributions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GPR_prop(MessagePassing):
    """Generalized PageRank propagation.

    Z = sum_{k=0}^{K} gamma_k * A_hat^k @ x,
    where A_hat is the GCN-normalized adjacency with self-loops and
    gamma_k are free, signed, learnable hop weights.
    """

    def __init__(self, K, alpha=0.1, Gamma=None, **kwargs):
        super(GPR_prop, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.alpha = alpha  # kept for compatibility with training args
        self.Gamma = Gamma
        if Gamma is None:
            temp = torch.ones(K + 1, dtype=torch.float) / (K + 1)
        else:
            temp = torch.as_tensor(Gamma, dtype=torch.float)
        self.temp = Parameter(temp)  # GPR weights gamma_0 .. gamma_K

    def reset_parameters(self):
        if self.Gamma is None:
            nn.init.constant_(self.temp, 1.0 / (self.K + 1))
        else:
            gamma = torch.as_tensor(
                self.Gamma, dtype=self.temp.dtype, device=self.temp.device
            )
            self.temp.data.copy_(gamma)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )
        hidden = x * self.temp[0]  # gamma_0 * A_hat^0 @ x
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)  # x <- A_hat @ x
            hidden = hidden + self.temp[k + 1] * x          # gamma_{k+1} * A_hat^{k+1} @ x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GPRGNN(nn.Module):
    """GPR-GNN: MLP feature transform, then learnable GPR propagation."""

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.0):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop1 = GPR_prop(K, alpha)
        self.dropout = dropout
        self.dprate = dprate

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))  # H^(0) = f_theta(X)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


def build_optimizer(model, lr=0.05, weight_decay=5e-4):
    """Use weight decay on the MLP but not on the propagation coefficients."""
    return torch.optim.Adam([
        {"params": model.lin1.parameters(), "lr": lr, "weight_decay": weight_decay},
        {"params": model.lin2.parameters(), "lr": lr, "weight_decay": weight_decay},
        {"params": model.prop1.parameters(), "lr": lr, "weight_decay": 0.0},
    ])
```
