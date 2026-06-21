I am trying to predict missing links in an undirected graph. I am given an adjacency matrix A over N nodes and, usually, a node-feature matrix X; some true edges are held out, and I must score every candidate pair so that the held-out edges rank above genuinely unconnected pairs. There are no labels, so the model must learn entirely from observed connectivity and features.

The existing ideas each solve only part of the problem. Hand-designed topology heuristics such as common neighbors, Adamic-Adar, and Katz give a fixed structural lens, ignore node features, and fail whenever the true link mechanism differs from the chosen score. Matrix factorization and DeepWalk learn embeddings instead, and they produce a clean similarity decoder the inner product falls naturally out of reconstructing A by a low-rank Z Z^T. But their encoder is just a lookup table indexed by node ID; it is transductive, structure-only, and has no way to read X. Meanwhile, graph convolutional networks are excellent encoders that fuse node features with local structure, but they had been trained only on supervised node-label objectives and never scored node pairs for an unsupervised relational task. What is missing is a single model that uses a GCN-like encoder to feed a matrix-factorization-like decoder and is trained end to end against a link-reconstruction loss.

The method is the Graph Auto-Encoder, or GAE. It consists of a two-layer graph convolutional network encoder and an inner-product decoder. The encoder maps features and structure to per-node embeddings, and the decoder reconstructs the adjacency from pairwise latent similarity. Concretely, the first graph convolution applies ReLU to the propagated features, and the second convolution outputs the latent vectors linearly, giving Z = A_norm ReLU(A_norm X W_0) W_1, where A_norm is the symmetrically normalized adjacency with self-loops, D_tilde^{-1/2} A_tilde D_tilde^{-1/2}. The logit for a candidate pair (i, j) is the dot product z_i^T z_j, and the reconstructed edge probability is sigma(z_i^T z_j). In matrix form the reconstruction is A_hat = sigma(Z Z^T). The decoder has no parameters; all capacity is in the encoder, and the inner product acts as a hard inductive bias that links should correspond to aligned latent vectors.

Training treats each A_ij as a Bernoulli observation, so the natural loss is binary cross-entropy over the logits z_i^T z_j. Real graphs are extremely sparse, so non-edges dominate the full N^2 loss and push the model toward the trivial solution of predicting no edges. I rebalance the loss by subsampling a set of negative edges comparable in size to the positive edges at each step, or equivalently by up-weighting positives in a dense formulation. When node features are unavailable, I set X to the identity matrix I_N, making each node a one-hot indicator and yielding a purely structure-only baseline with the same architecture.

The same idea can be extended to a variational Graph Auto-Encoder, VGAE, where the GCN outputs per-node Gaussian parameters mu and log sigma, embeddings are sampled via the reparameterization trick, and an analytic KL term regularizes the posterior toward a standard normal prior. That sibling is useful when a smoother, more regularized latent space is desired, but the core dot-product baseline and the method proposed here is the deterministic GAE.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


def normalized_adjacency(A):
    """D̃^{-1/2} Ã D̃^{-1/2} with Ã = A + I (renormalization trick)."""
    A = A + torch.eye(A.size(0), device=A.device)
    d_inv_sqrt = A.sum(1).pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, H, A_norm):
        return A_norm @ self.W(H)


def inner_product(z, pair_index, sigmoid=False):
    """Decoder logit z_i^T z_j (Â = σ(Z Z^T))."""
    s = (z[pair_index[0]] * z[pair_index[1]]).sum(dim=-1)
    return torch.sigmoid(s) if sigmoid else s


class GAE(nn.Module):
    """Graph Auto-Encoder: GCN encoder + inner-product decoder."""

    def __init__(self, in_dim, hidden_dim=32, emb_dim=16):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, emb_dim)

    def encode(self, X, A_norm):
        H = F.relu(self.gc1(X, A_norm))
        return self.gc2(H, A_norm)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        eps = 1e-15
        pos = inner_product(z, pos_edge_index, sigmoid=True)
        neg = inner_product(z, neg_edge_index, sigmoid=True)
        return -(torch.log(pos + eps).mean() + torch.log(1 - neg + eps).mean())

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
        self.gc1 = GraphConv(in_dim, hidden_dim)
        self.gc_mu = GraphConv(hidden_dim, emb_dim)
        self.gc_logstd = GraphConv(hidden_dim, emb_dim)

    def encode(self, X, A_norm):
        H = F.relu(self.gc1(X, A_norm))
        self.mu = self.gc_mu(H, A_norm)
        self.logstd = self.gc_logstd(H, A_norm).clamp(max=10.0)
        if self.training:
            return self.mu + torch.randn_like(self.logstd) * torch.exp(self.logstd)
        return self.mu

    def kl_loss(self):
        return -0.5 * torch.mean(torch.sum(
            1 + 2 * self.logstd - self.mu ** 2 - torch.exp(2 * self.logstd), dim=1))


def train(model, X, A_norm, pos_edge_index, num_nodes,
          n_iters=200, lr=0.01, variational=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_iters):
        model.train()
        optimizer.zero_grad()
        z = model.encode(X, A_norm)
        loss = model.recon_loss(z, pos_edge_index)
        if variational:
            loss = loss + (1.0 / num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
```
