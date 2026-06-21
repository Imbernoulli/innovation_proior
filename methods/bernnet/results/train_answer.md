A graph filter is just a polynomial response applied to the spectrum of the normalized Laplacian. If L has eigenvalues in [0, 2], the model learns a scalar function g(λ) on that interval and applies it to every eigen-component. That makes the design problem very concrete: we need one parameterization of g that can be low-pass on homophilic data, high-pass or band-pass on heterophilic data, and remain inside the valid range of a smoothing propagator. Existing ideas do not manage all of this at once. GCN and APPNP are fixed low-pass filters, so they cannot represent the high-frequency signal that heterophilic graphs need. ChebNet and GPR-GNN learn free coefficients, which gives flexibility but removes any guarantee that the learned response stays non-negative, and their coefficients are not readable as a frequency response. The missing piece is a basis in which a simple non-negativity constraint on the coefficients is the same thing as a non-negative response, while still spanning arbitrary shapes.

The fix is to parameterize the filter in the Bernstein basis. On the spectrum mapped to [0, 1] by t = λ/2, the degree-K Bernstein basis is b_k^K(t) = C(K,k)(1-t)^{K-k} t^k. Each basis function is non-negative, the family sums to one everywhere by the binomial theorem, and each bump peaks at t = k/K, which corresponds to graph frequency λ = 2k/K. Therefore, if the learned coefficients θ_k are non-negative, the response h(λ) = Σ_k θ_k b_k^K(λ/2) is non-negative everywhere on [0, 2] for free. Because Bernstein approximation converges uniformly to any continuous response as K grows, constraining the coefficients does not sacrifice expressiveness the way non-negative monomial coefficients would. More importantly, θ_k is the response value near frequency 2k/K, so the entire learned filter is readable as a uniform sample of the frequency response.

The method is BernNet. It lifts the scalar Bernstein filter to the operator level using the symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}. With the affine map t = λ/2, the propagation becomes

  z = Σ_{k=0}^K ReLU(θ_k) · C(K,k)/2^K · (2I - L)^{K-k} L^k x.

The ReLU enforces θ_k ≥ 0, which is the only hard validity constraint needed; the upper-bound side of the [0, 1] range is a scale condition handled by normalization or by simply interpreting the coefficients as relative filter strengths. The coefficients are initialized to all ones, which makes the initial filter the identity (all-pass) and therefore unbiased toward any particular frequency band. During training, gradient descent can move the mass toward low, high, or middle frequencies depending on the data. This keeps the same MLP-then-propagate pipeline used by GCN, APPNP, and GPR-GNN: a two-layer MLP transforms the node features, dropout is applied, and then BernNet propagates the transformed logits.

The trade-off is computational. ChebNet and GPR-GNN need O(K) sparse propagations, but the Bernstein construction needs to multiply both (2I - L)^{K-k} and L^k into each term, giving O(K^2) sparse propagations in total. The implementation caches the chain (2I - L)^i x for i = 0..K and reuses it, but each remaining term still needs up to K additional applications of L. That cost buys a direct, cheap, and interpretable non-negativity constraint that no monomial or Chebyshev parameterization offers. The canonical name of the method is BernNet.

```python
import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian, add_self_loops
from scipy.special import comb


class Bern_prop(MessagePassing):
    """Bernstein-basis propagation layer.

    z = sum_k ReLU(theta_k) * C(K,k) / 2^K * (2I - L)^{K-k} * L^k * x
    where L = I - D^{-1/2} A D^{-1/2} is the symmetric normalized Laplacian.
    """

    def __init__(self, K, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)  # all-pass (identity) start

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)  # enforce non-negative Bernstein coefficients

        # symmetric normalized Laplacian L
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim)
        )
        # 2I - L (negate off-diagonal L weights, self-loop value 2)
        edge_index2, norm2 = add_self_loops(
            edge_index1, -norm1, fill_value=2.0,
            num_nodes=x.size(self.node_dim)
        )

        # tmp[i] = (2I - L)^i x for i = 0..K
        tmp = [x]
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        # k = 0 term
        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        # k = i+1 terms: apply L^{i+1} to tmp[K-i-1]
        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for _ in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class BernNet(torch.nn.Module):
    """MLP feature transform followed by Bernstein polynomial propagation."""

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 dropout=0.5, dprate=0.5):
        super().__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop1 = Bern_prop(K)
        self.dropout = dropout
        self.dprate = dprate

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
```
