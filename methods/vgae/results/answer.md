# VGAE: Variational Graph Auto-Encoder

VGAE is an unsupervised latent-variable model for one attributed graph. A GCN encoder maps
`(X, A_train)` to a diagonal Gaussian posterior for every node, an inner-product Bernoulli decoder
reconstructs adjacency entries, and training maximizes an ELBO. Its deterministic sibling, GAE,
removes the posterior variance, sampling, and KL term.

## Model

Encoder:

```text
q(Z | X,A) = prod_i N(z_i | mu_i, diag(sigma_i^2))
mu        = GCN_mu(X,A)
log sigma = GCN_sigma(X,A)
```

`GCN_mu` and `GCN_sigma` share the first GCN layer. The paper writes the normalized adjacency as
`D^{-1/2} A D^{-1/2}` because `A` is defined with self-loops already present. The canonical code
starts from `adj_train`, adds self-loops in preprocessing, and uses symmetric normalization.

Decoder and prior:

```text
p(A_ij = 1 | z_i, z_j) = sigmoid(z_i^T z_j)
p(A | Z) = prod_i prod_j p(A_ij | z_i, z_j)
p(Z) = prod_i N(z_i | 0,I)
```

Training objective:

```text
ELBO = E_q[log p(A | Z)] - KL(q(Z | X,A) || p(Z)).
```

Reparameterization:

```text
z_i = mu_i + sigma_i * eps_i,     eps_i ~ N(0,I)
```

Use sampled `z` while training and `mu` for evaluation.

## KL And Loss

For one coordinate:

```text
-KL(N(mu, sigma^2) || N(0,1))
  = 0.5 * (1 + log sigma^2 - mu^2 - sigma^2)
  = 0.5 * (1 + 2*logstd - mu^2 - exp(2*logstd)).
```

Positive KL, averaged over nodes:

```python
kl_loss = -0.5 * mean_nodes(sum_dim(
    1 + 2 * logstd - mu**2 - exp(2 * logstd)
))
```

The official TensorFlow implementation adds this node-averaged KL with an extra `1 / num_nodes`
factor by subtracting the negative-KL term `(0.5 / N) * mean_nodes(...)` from the reconstruction
cost. PyG's `VGAE.kl_loss()` returns the same positive node-averaged KL, so the matching sampled
training loss is:

```text
loss = reconstruction_loss + kl_loss / num_nodes
```

Dense canonical reconstruction uses all adjacency entries with positive reweighting:

```text
pos_weight = (N^2 - sum(adj_train)) / sum(adj_train)
norm       = N^2 / ((N^2 - sum(adj_train)) * 2)
labels     = adj_train + I
recon      = norm * mean(weighted_BCE_with_logits(flatten(Z Z^T), flatten(labels), pos_weight))
```

Sparse/scaffold training may instead score positive edges plus sampled non-edges, which is the
paper's allowed zero-subsampling alternative.

## Reference-Faithful Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LinkPredictor(nn.Module):
    """Variational graph auto-encoder for link prediction."""

    def __init__(self, in_channels, hidden_channels=32, latent_channels=16, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.conv_hidden = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_channels)
        self.conv_logstd = GCNConv(hidden_channels, latent_channels)
        self.mu = None
        self.logstd = None

    def encode(self, x, edge_index):
        h = self.conv_hidden(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        self.mu = self.conv_mu(h, edge_index)
        self.logstd = self.conv_logstd(h, edge_index)
        if self.training:
            return self.mu + torch.randn_like(self.logstd) * torch.exp(self.logstd)
        return self.mu

    def decode(self, z_src, z_dst):
        return (z_src * z_dst).sum(dim=-1)

    def kl_loss(self):
        return -0.5 * torch.mean(torch.sum(
            1 + 2 * self.logstd - self.mu.pow(2) - torch.exp(2 * self.logstd),
            dim=-1,
        ))

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z[edge_label_index[0]], z[edge_label_index[1]])
```

Sampled-edge training loop:

```python
def train_step(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_label_index)
    recon = F.binary_cross_entropy_with_logits(logits, data.edge_label)
    loss = recon + model.kl_loss() / data.num_nodes
    loss.backward()
    optimizer.step()
    return float(loss)
```

GAE special case: replace the two posterior heads with one embedding head `Z = GCN(X,A)`, remove
sampling and `kl_loss`, and keep the same dot-product decoder `sigmoid(Z Z^T)`. Featureless case:
set `X = I`.
