# VGAE (Variational Graph Auto-Encoder), distilled

VGAE is an unsupervised latent-variable model for a single attributed graph. A graph
convolutional encoder maps each node to a Gaussian posterior over a latent vector; a
parameter-free inner-product decoder reconstructs the adjacency from those vectors; and the
model is trained by maximizing the evidence lower bound (reconstruction minus a KL pull toward a
standard-normal prior). It fuses graph structure and node features in one end-to-end model and,
having learned to reconstruct the observed edges, scores unobserved node pairs for link prediction.
Dropping the stochastic part recovers the deterministic Graph Auto-Encoder (GAE).

## Problem it solves

Link prediction on one undirected graph with node features and a partially observed adjacency:
learn a per-node representation and a pair-scoring rule, unsupervised (only the edges supervise),
that uses *both* topology and features, so that held-out edges rank above held-out non-edges.
Prior unsupervised methods (random-walk skip-gram embeddings such as DeepWalk/node2vec, spectral
embeddings) use structure only; the feature-aware GCN encoder existed but was tied to a supervised
node-classification loss.

## Key idea

- **Encoder (GCN, two heads).** A two-layer graph convolution computes, per node, the parameters
  of a diagonal-Gaussian posterior q(z_i | X, A) = N(z_i | μ_i, diag(σ_i²)). A shared first GCN
  layer feeds two second-layer GCN heads: μ = GCN_μ(X, A) and log σ = GCN_σ(X, A) (output log σ so
  σ = exp(log σ) > 0 with no constraint to enforce). Each layer mixes a node with a renormalized
  average of its neighbors, so z_i depends on structure and features together.
- **Reparameterization.** Sample z_i = μ_i + σ_i ⊙ ε, ε ~ N(0, I) at train time (low-variance
  pathwise gradient into μ, σ); use z_i = μ_i at test time.
- **Decoder (inner product).** p(A_ij = 1 | z_i, z_j) = σ(z_iᵀ z_j), i.e. Â = σ(Z Zᵀ). No
  parameters; symmetric (undirected); permutation-equivariant; geometry = proximity in latent
  space → high edge probability.
- **Objective (ELBO).** L = E_{q(Z|X,A)}[log p(A|Z)] − D_KL(q(Z|X,A) ‖ p(Z)) with prior
  p(Z) = ∏_i N(z_i | 0, I). The reconstruction is a (re-weighted / negative-sampled) binary
  cross-entropy on edges vs. non-edges; the KL is closed-form and weighted 1/N to match the
  per-edge reconstruction scale.

## The Gaussian KL (closed form)

For a diagonal-Gaussian posterior N(μ, diag(σ²)) against the standard-normal prior N(0, I) in J
latent dimensions, per node:

```
-D_KL( q || p ) = 1/2 * sum_j ( 1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 )
               = 1/2 * sum_j ( 1 + 2*log_sigma_j - mu_j^2 - exp(2*log_sigma_j) ).
```

Maximized (KL = 0) at μ = 0, σ² = 1, i.e. when the posterior equals the prior. Derivation:
∫q log p = −J/2 log(2π) − ½Σ(μ_j² + σ_j²) and ∫q log q = −J/2 log(2π) − ½Σ(1 + log σ_j²);
subtracting cancels the constants and leaves the expression above.

## Design choices and why

- **GCN encoder, not an MLP:** an MLP sees a node's features only; the GCN aggregates the
  neighborhood, fusing topology and features. Random-walk / spectral baselines fuse neither with
  features.
- **Two heads sharing a trunk:** mean and spread are both functions of the same neighborhood
  evidence; sharing the first layer avoids duplicating the network.
- **Output log σ:** keeps σ positive for any real network output; numerically stable. (Clamp log σ
  at a ceiling, e.g. 10, so exp(log σ) cannot explode early in training.)
- **Inner-product decoder:** cheapest scoring rule with the right "near = linked" bias; no params;
  symmetric for undirected graphs; equivariant. An MLP-on-[z_i; z_j] decoder adds parameters and
  breaks the clean geometry.
- **Positive re-weighting / negative sampling:** A is sparse, so unbalanced BCE collapses to
  "predict no edge"; re-weight the rare positives (or pair each edge with a sampled non-edge).
- **KL weighted by 1/N:** reconstruction is a per-edge mean, the KL a per-node sum; the 1/N puts
  them on a comparable per-unit footing so the trade-off does not drift with graph size.
- **Centered Gaussian prior:** a standard, analytic-KL default; in mild tension with the
  inner-product decoder, which rewards large-norm embeddings while the prior pulls toward the
  origin. Gentle (1/N) weighting keeps reconstruction able to spread the embeddings out.
- **Full-batch training:** small graphs fit in memory; GCN is O(|E|) per layer; Adam, Glorot init.
- **GAE special case:** drop the log-σ head, sampling, and KL → deterministic Z = GCN(X, A),
  Â = σ(Z Zᵀ). Featureless: set X = I (one-hot node identity), structure-only.

## Final model (code)

Encoder/decoder filling the `LinkPredictor` interface, with the variational heads, reparameterization,
inner-product decoder, and closed-form KL; grounded in the canonical implementation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

MAX_LOGSTD = 10  # clamp on log sigma; exp(log sigma) stays bounded


class LinkPredictor(nn.Module):
    """Variational Graph Auto-Encoder.

    GCN encoder with mu / log-sigma heads (shared trunk) -> per-node Gaussian posterior;
    reparameterized sampling at train time, mu at eval; inner-product decoder sigma(z_i^T z_j);
    ELBO = reconstruction (BCE, applied by the loop) - KL(q || p), p = N(0, I).
    """

    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.shared_convs = nn.ModuleList()
        if num_layers > 1:
            self.shared_convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.shared_convs.append(GCNConv(hidden_channels, hidden_channels))
            last_in = hidden_channels
        else:
            last_in = in_channels
        self.conv_mu = GCNConv(last_in, hidden_channels)       # mu  = GCN_mu(X, A)
        self.conv_logstd = GCNConv(last_in, hidden_channels)   # log sigma = GCN_sigma(X, A)
        self._mu = None
        self._logstd = None

    def encode(self, x, edge_index):
        for conv in self.shared_convs:                          # shared GCN trunk
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        self._mu = self.conv_mu(x, edge_index)
        self._logstd = self.conv_logstd(x, edge_index).clamp(max=MAX_LOGSTD)
        if self.training:
            std = torch.exp(self._logstd)                       # z = mu + sigma * eps
            return self._mu + torch.randn_like(std) * std
        return self._mu                                         # eval: posterior mean

    def decode(self, z_src, z_dst):
        return (z_src * z_dst).sum(dim=-1)                      # logit = z_i^T z_j

    def kl_loss(self):
        mu, logstd = self._mu, self._logstd                    # +D_KL(q || p)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=-1)
        )

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z[edge_label_index[0]], z[edge_label_index[1]])
```

Training objective (reconstruction + per-node-averaged KL):

```python
def train_step(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_label_index)
    recon = F.binary_cross_entropy_with_logits(logits, data.edge_label)   # -E_q[log p(A|Z)]
    loss = recon + (1.0 / data.num_nodes) * model.kl_loss()               # + (1/N) D_KL(q || p)
    loss.backward()
    optimizer.step()
    return float(loss)
```

Typical config: 2 GCN layers, 32-dim hidden, 16-dim latent, Glorot init, Adam lr ≈ 0.01,
full-batch, a couple hundred iterations. GAE = the same model with the log-σ head, sampling, and
KL removed (deterministic Z = GCN(X, A), Â = σ(Z Zᵀ)).
