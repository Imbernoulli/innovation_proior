# gcn_dot: Graph Auto-Encoder with a Dot-Product Decoder

The graph auto-encoder is an unsupervised model for graph-structured data that encodes nodes
with a graph convolutional network (GCN) and decodes node pairs with an inner product. It
turns the supervised GCN node encoder into a relational, unsupervised learner by training it
to *reconstruct the adjacency matrix*. The non-probabilistic GAE computes `Z = GCN(X, A)` and
reconstructs `Â = σ(Z Z^T)`; its variational sibling VGAE makes the encoder probabilistic and
adds a KL regularizer toward a Gaussian prior. The "GCN encoder + dot-product decoder" used
as a link-prediction baseline is exactly the non-variational GAE.

## Problem it solves

Unsupervised representation learning and link prediction on an undirected graph given by
adjacency `A` (with optional node features `X`): learn a per-node embedding `z_i` and a way to
score node pairs so that held-out true edges rank above sampled non-edges. Unlike random-walk
embeddings (DeepWalk), spectral clustering, or plain matrix factorization — all structure-only,
transductive, and trained as multi-stage pipelines — it ingests node features and trains end
to end under a single objective.

## Key idea

Cast link prediction as auto-encoding the graph's connectivity.

- **Encoder (GCN):** `Z = GCN(X, A)`, a two-layer graph convolutional network mapping features
  and structure to node embeddings. Each layer is
  `H^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)})` with `Ã = A + I` (self-loops) and
  symmetric normalization; concretely `Z = Â_norm · ReLU(Â_norm X W_0) · W_1` with the second
  layer linear. The encoder carries all the parameters.
- **Decoder (inner product):** the logit for pair `(i,j)` is `z_i^T z_j`; the reconstructed
  adjacency is `Â = σ(Z Z^T)`, i.e. `p(A_{ij}=1 | z_i, z_j) = σ(z_i^T z_j)`. This is the
  matrix-factorization decoder: reconstructing `A` by a low-rank `Z Z^T` *is* scoring pairs by
  latent similarity. It has zero parameters.
- **Objective:** model each `A_{ij}` as Bernoulli, giving binary cross-entropy on the logits.
  Because real graphs are sparse (`|E| ≪ N²`), the loss is dominated by non-edges, so the rare
  positive terms are up-weighted (or negatives are subsampled) to keep the edge signal alive.

## GAE (non-probabilistic) — the dot-product baseline

```
Z = GCN(X, A)
Â = σ(Z Z^T)
```

Loss — balanced reconstruction cross-entropy. Dense form (small graphs), positives up-weighted:

```
L = norm · mean_{i,j}  weighted_BCE_with_logits( logit = z_i^T z_j, target = A_ij, pos_weight )
pos_weight = (N² − ΣA) / ΣA          # negatives-to-positives ratio
norm       = N² / ((N² − ΣA) · 2)    # keeps the loss on a comparable scale
```

Here `ΣA` is the number of positive entries in the training adjacency used for rebalancing. In
the dense TensorFlow path, the label matrix is then formed as `adj_train + I` before flattening.

Sparse / scalable form (subsample non-edges per step):

```
L = − mean_{(i,j) ∈ E} log σ(z_i^T z_j)  −  mean_{(i,j) ∈ neg} log(1 − σ(z_i^T z_j))
```

Featureless variant: replace `X` with the identity `I_N` (each node a one-hot indicator).

## VGAE (variational sibling)

Inference (probabilistic encoder, two GCN heads sharing the first layer):

```
q(Z | X, A) = Π_i N(z_i | μ_i, diag(σ_i²)),   μ = GCN_μ(X, A),  log σ = GCN_σ(X, A)
```

Generative model (same inner-product decoder) and standard-normal prior:

```
p(A | Z) = Π_{i,j} σ(z_i^T z_j)^{A_ij} (1 − σ(z_i^T z_j))^{1−A_ij},     p(Z) = Π_i N(z_i | 0, I)
```

Trained by maximizing the evidence lower bound (or minimizing its negative):

```
ELBO = E_{q(Z|X,A)}[ log p(A | Z) ]  −  KL[ q(Z | X, A) || p(Z) ]
```

with the reparameterization trick `z_i = μ_i + σ_i ⊙ ε_i`, `ε_i ∼ N(0, I)` (use the mean at
eval). The Gaussian KL is analytic; per coordinate
`KL(N(μ,σ²) || N(0,1)) = −½(1 + log σ² − μ² − σ²)`, so

```
total_KL = −½ · Σ_nodes Σ_dim ( 1 + 2 log σ − μ² − exp(2 log σ) )
kl_loss  = −½ · mean_nodes Σ_dim ( 1 + 2 log σ − μ² − exp(2 log σ) )
```

The dense TensorFlow loss subtracts `(0.5/N) · mean_nodes Σ_dim(1 + 2 log σ − μ² − exp(2 log σ))`
from the weighted all-pairs reconstruction cost; equivalently it adds the positive
node-averaged KL divided by `N`. PyG's `VGAE.kl_loss()` returns the positive node-averaged KL, so
training loops add the same `1/num_nodes` scale when matching that objective.

The KL regularizes the latent space into a smoother geometry. Caveat: the inner-product decoder
profits from large-norm embeddings (to make `σ(z_i^T z_j)` confident), which is in tension with
a zero-centered `N(0, I)` prior.

## Limitations of the inner-product decoder

Scores pairs purely by symmetric bilinear latent similarity, so it is natural for
homophily/proximity but not a universal edge model. Link rules that require richer pair
interactions would need a more flexible generative model, but the dot product is the canonical
zero-parameter choice that defines the model.

## Defaults

32-dim first hidden layer, 16-dim latent; Glorot init; full-batch gradient descent with Adam
at learning rate 0.01; ~200 iterations. Evaluated by AUC / average precision on held-out edges
vs. sampled non-edges.

## Working code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

EPS = 1e-15


def normalized_adjacency(A):
    """D̃^{-1/2} Ã D̃^{-1/2} with Ã = A + I  (renormalization)."""
    A = A + torch.eye(A.size(0), device=A.device)
    d_inv_sqrt = A.sum(1).pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H, A_norm):
        return A_norm @ self.W(H)


def inner_product(z, pair_index, sigmoid=False):
    """Decoder logit z_i^T z_j  (Â = σ(Z Z^T))."""
    s = (z[pair_index[0]] * z[pair_index[1]]).sum(dim=-1)
    return torch.sigmoid(s) if sigmoid else s


class GAE(nn.Module):
    """GCN encoder + inner-product decoder; reconstructs the adjacency."""

    def __init__(self, in_dim, hidden_dim=32, emb_dim=16):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, emb_dim)

    def encode(self, X, A_norm):
        H = F.relu(self.gc1(X, A_norm))
        return self.gc2(H, A_norm)                 # Z = GCN(X, A), linear 2nd layer

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is None:                 # subsample non-edges
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        pos = inner_product(z, pos_edge_index, sigmoid=True)
        neg = inner_product(z, neg_edge_index, sigmoid=True)
        return -(torch.log(pos + EPS).mean() + torch.log(1 - neg + EPS).mean())

    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index):
        from sklearn.metrics import average_precision_score, roc_auc_score
        pos = inner_product(z, pos_edge_index, sigmoid=True)
        neg = inner_product(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos, neg]).cpu().numpy()
        y = torch.cat([torch.ones(pos.size(0)), torch.zeros(neg.size(0))]).numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(GAE):
    """Variational sibling: GCN posterior + reparameterization + KL regularizer."""

    def __init__(self, in_dim, hidden_dim=32, emb_dim=16):
        super().__init__(in_dim, hidden_dim, emb_dim)
        self.gc1 = GraphConv(in_dim, hidden_dim)            # shared first layer W_0
        self.gc_mu = GraphConv(hidden_dim, emb_dim)         # mean head
        self.gc_logstd = GraphConv(hidden_dim, emb_dim)     # log-std head

    def encode(self, X, A_norm):
        H = F.relu(self.gc1(X, A_norm))
        self.mu = self.gc_mu(H, A_norm)
        self.logstd = self.gc_logstd(H, A_norm).clamp(max=10.0)
        if self.training:                                   # z = μ + σ ⊙ ε
            return self.mu + torch.randn_like(self.logstd) * torch.exp(self.logstd)
        return self.mu

    def kl_loss(self):
        return -0.5 * torch.mean(torch.sum(
            1 + 2 * self.logstd - self.mu**2 - torch.exp(2 * self.logstd), dim=1))


def train(model, X, A_norm, pos_edge_index, num_nodes,
          n_iters=200, lr=0.01, variational=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   # full-batch GD
    for _ in range(n_iters):
        model.train()
        optimizer.zero_grad()
        z = model.encode(X, A_norm)
        loss = model.recon_loss(z, pos_edge_index)
        if variational:
            loss = loss + (1.0 / num_nodes) * model.kl_loss()  # negative ELBO scaling
        loss.backward()
        optimizer.step()
```
