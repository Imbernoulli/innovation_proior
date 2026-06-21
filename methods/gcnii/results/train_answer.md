The task is semi-supervised node classification on citation graphs, where only about twenty nodes per class are labeled and the rest must be inferred from bag-of-words features and citation links. The standard approach stacks graph convolutional layers so that each node's representation mixes with its neighbors. The obstacle is depth: a two-layer renormalized graph convolution works well, but adding more layers makes accuracy collapse rather than improve. This is over-smoothing. When we strip away the per-layer weights and nonlinearities, K stacked renormalized layers reduce to the fixed operator P̃^K applied to the input signal. Because P̃ has top eigenvalue one and all other eigenvalues strictly smaller in magnitude, raising it to the K-th power annihilates every spectral component except the top mode. In the limit the representation depends only on node degrees and a single global inner product with the features, so nodes with the same degree become indistinguishable. A ResNet-style residual connection to the previous layer does not fix this: it only makes the associated lazy random walk converge more slowly to the same degree-only stationary distribution.

The method I propose is GCNII, short for Graph Convolutional Networks with Initial residual and Identity mapping. The first change is an initial residual. Instead of adding a skip from the previous layer, which is already smoothed, each layer re-injects a fixed fraction α of the initial representation H^(0), a learned map of the raw node features. The update mixes the propagated signal with the original transformed input: (1 − α) P̃ H^(ℓ) + α H^(0). Because the input is reintroduced at every layer, the deep limit is no longer the degree vector; it becomes the input-carrying personalized PageRank diffusion α (I − (1 − α) P̃)^(−1) H^(0). Information from the features is preserved no matter how deep the network goes.

The second change is identity mapping. Restoring a learnable weight matrix W^(ℓ) and a nonlinearity at each layer reintroduces a different depth pathology: the product of many weight matrices can contract or expand the feature space, driving representations into a low-dimensional subspace at a rate governed by the K-th power of the largest singular value. It also overfits the tiny labeled set. GCNII therefore writes each layer's effective transform as (1 − β_ℓ) I + β_ℓ W^(ℓ). When β_ℓ is small, the layer is close to the identity and behaves like the safe linear propagation step; when β_ℓ is larger, it performs a modest learned feature transformation. The coefficient decays with depth according to β_ℓ = log(λ / ℓ + 1), so deeper layers are pushed closer to identity. This keeps the maximum singular value near one and makes strong L2 regularization on W safe, because the linear residual network literature shows that the optimal weights have small norm and the only critical point is the global minimum.

Together these two changes make depth a resource. The initial residual prevents propagation from collapsing to a degree-only fixed point, and the identity mapping prevents the weight matrices from collapsing the feature rank or overfitting the scarce labels. In the weak scalar version of the layer, the unrolled K-layer network realizes a polynomial filter of order K with learnable coefficients, whereas the vanilla GCN is stuck with fixed coefficients. The same structure also appears naturally as one unrolled step of ISTA for sparse feature recovery. The architecture is simple: one fully connected layer maps input features to a fixed hidden dimension H to form H^(0), then L graph convolution layers operate at the same width H so that the initial residual is a direct addition, and a final fully connected layer maps to class logits. The parameter count grows only linearly in depth because each convolution layer uses an H-by-H weight matrix.

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """One GCNII layer: sparse renormalized propagation, initial residual, identity mapping."""

    def __init__(self, channels):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(channels, channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, h0, lamda, alpha, layer):
        theta = math.log(lamda / layer + 1.0)
        hi = torch.spmm(adj, x)                            # P̃ H^(ℓ)
        support = (1 - alpha) * hi + alpha * h0            # initial residual
        return theta * torch.mm(support, self.weight) + (1 - theta) * support


class GCNII(nn.Module):
    """Input FC -> L identity-mapped initial-residual graph convs -> output FC."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout, alpha=0.1, lamda=0.5):
        super().__init__()
        self.convs = nn.ModuleList(
            GraphConvolution(hidden_channels) for _ in range(num_layers)
        )
        self.fcs = nn.ModuleList([
            nn.Linear(in_channels, hidden_channels),
            nn.Linear(hidden_channels, out_channels),
        ])
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))          # H^(0)
        h0 = layer_inner
        for i, conv in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                conv(layer_inner, adj, h0, self.lamda, self.alpha, i + 1)
            )
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)
```

On the standard Planetoid semi-supervised splits, the proposed hyperparameters are Cora with 64 layers, α = 0.1, λ = 0.5, and H = 64; CiteSeer with 32 layers, α = 0.1, λ = 0.6, and H = 256; and PubMed with 16 layers, α = 0.1, λ = 0.4, and H = 256. Training uses Adam with learning rate 0.01 and early stopping with patience 100. The key regularization choice is stronger weight decay on the convolution weights than on the input and output fully connected layers, which exploits the small-norm optimum property of identity-mapped transforms.
