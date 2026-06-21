A graph filter is the operator that spreads node features across the structure before classification, and its frequency response decides whether a model succeeds. On homophilic graphs a low-pass response is right because averaging neighbors reinforces the class signal, but on heterophilic graphs connected nodes tend to differ, so the signal lives in high-frequency contrast and smoothing destroys it. The prior art therefore needs a single learnable filter that can become low-pass, high-pass, or band-pass from labels alone. The obvious candidates each fail in a different way. ChebNet learns free Chebyshev coefficients, yet on standard benchmarks it loses to GCN, its own first-order truncation, and accuracy falls as the order K grows. GPR-GNN uses a monomial basis, which is flexible but numerically ill-conditioned because powers of the spectrum become nearly collinear. BernNet uses a Bernstein basis that guarantees non-negativity and is interpretable, but its forward pass costs O(K^2 m d), quadratic in K, and Bernstein approximation converges more slowly than the near-minimax alternative. What is missing is a filter that keeps linear cost, avoids overfitting through wild high-frequency responses, and lets side constraints such as non-negativity appear as simple parameter constraints.

The method I propose is ChebNetII. It keeps the decoupled APPNP-style harness, an MLP first and then polynomial propagation, but reparameterizes the filter by its values at the Chebyshev interpolation nodes rather than by free coefficients. The idea starts from the fact that a smooth analytic filter has Chebyshev coefficients that decay like 1/k^q, so unconstrained coefficient learning can memorize training labels through a jagged, high-frequency response. A crude 1/k penalty fixes the symptom but is hard to extend to other constraints. Instead, ChebNetII treats the filter values gamma_j = h(x_j) at the Chebyshev nodes x_j = cos((j + 1/2) pi / (K + 1)) as the trainable parameters. Because these nodes are the roots of T_{K+1}, they minimize the nodal polynomial that drives the Runge phenomenon, giving a Lebesgue constant of order log K and a near-minimax interpolant. The coefficients of that interpolant are recovered by the discrete orthogonality of Chebyshev polynomials at their own nodes: c_k = (2/(K+1)) sum_j gamma_j T_k(x_j), with the constant term applied as c_0/2. Since the parameters are sampled filter values, constraining gamma_j to be non-negative is exactly the ReLU used before the coefficient transform, turning a global functional constraint into a simple box constraint.

The spectrum of the symmetric normalized Laplacian is contained in [0, 2], so ChebNetII rescales with the a-priori bound lambda_max = 2 rather than computing eigenvalues. The shifted operator is L_hat = L - I, mapping [0, 2] to [-1, 1]. The filter is applied with the stable Chebyshev three-term recurrence on the operator, which needs only K sparse matrix-vector products and never forms powers of L explicitly. Forming the coefficients costs O(K^2) scalar work and is dominated by the O(K m d) propagation cost, so the whole layer is linear in K, matching GPR-GNN and ChebNet and beating BernNet's quadratic cost. The interpolation error scales like C omega(K^{-1}) log K, faster than Bernstein's omega(K^{-1/2}), so a smaller K often suffices. The parameters are initialized to all ones, which corresponds to the constant all-pass filter h = 1 and imposes no low-pass or high-pass bias, letting the labels discover the needed frequency shape.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
from utils import cheby


class CustomProp(MessagePassing):
    """ChebNetII propagation: learn filter values at Chebyshev nodes,
    convert them to Chebyshev coefficients, and apply the filter."""

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

    def forward(self, x, edge_index, edge_weight=None):
        coe_tmp = F.relu(self.temp)  # enforce non-negative sampled filter values
        coe = coe_tmp.clone()

        # Discrete cosine transform of sampled values:
        # c_i = (2/(K+1)) sum_j gamma_j T_i(x_j)
        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        # L = I - D^{-1/2} A D^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim)
        )
        # L_hat = L - I, spectrum [0,2] -> [-1,1]
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0,
            num_nodes=x.size(self.node_dim)
        )

        # Chebyshev recurrence: T_0(x)=x, T_1(x)=x, T_k = 2 x T_{k-1} - T_{k-2}
        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)
        out = coe[0] / 2.0 * Tx_0 + coe[1] * Tx_1
        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2.0 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    """ChebNetII: MLP transform decoupled from Chebyshev-interpolation propagation."""

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K)
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
