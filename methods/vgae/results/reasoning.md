I start with the actual pressure in the task. I have one sparse undirected graph, not independent labeled examples. Some real edges are hidden, and I need to rank those hidden edges above sampled non-edges. I also have node features, so if I learn embeddings only from node IDs or random walks I am throwing away a source of evidence that should matter on citation graphs. The model I need has to turn `(X, A_train)` into one vector per node and then turn a pair of node vectors into a link score, using only the observed graph as supervision.

The older unsupervised choices do part of this but not enough. Spectral embeddings give me a topology summary from graph eigenvectors, and random-walk methods like DeepWalk give me node embeddings by treating walks as sentences and running SkipGram. Both can work for link prediction after I score pairs by a similarity in the learned space, but both are structure-only. They do not let a bag-of-words vector on a document change the embedding except through a separate downstream model. On the other side, the graph convolutional encoder does combine features with the graph: after adding self-loops and symmetric normalization, one layer applies `D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H W`, so every node receives a normalized mixture of its own features and its neighbors' features. That is the right encoder shape, but its original objective is supervised node classification. I have no node labels. I need to give the encoder an unsupervised graph-reconstruction job.

The simplest decoder is the one I should try first: put linked nodes near each other in latent space and score a candidate pair by an inner product. If node `i` has embedding `z_i` and node `j` has embedding `z_j`, then `z_i^T z_j` should serve as a logit for an undirected edge, with `p(A_ij = 1 | z_i, z_j) = sigmoid(z_i^T z_j)` and reconstructed adjacency `sigmoid(Z Z^T)` in matrix form. Before I lean on this I want to confirm it actually has the properties I am hoping for. Take three nodes with two-dimensional latents `z_0 = (1,0)`, `z_1 = (0.5,0.5)`, `z_2 = (0,1)`. The logit for `(0,1)` is `1*0.5 + 0*0.5 = 0.5`, and the logit for `(1,0)` is `0.5*1 + 0.5*0 = 0.5` as well, so the score is symmetric in the two endpoints, which is what an undirected edge needs. The pair `(0,2)` is orthogonal, logit `0`, so `sigmoid(0) = 0.5`: maximally uncertain, the right default for two unrelated nodes. And if I relabel nodes — swap indices `0` and `2` — the physical pair that was `(0,1)` is now accessed as `(2,1)` but its latents are unchanged, so its probability is still `sigmoid(0.5) = 0.6225`. The scoring rule carries no parameters of its own and does not depend on node ordering. A pair MLP could be more flexible, but it would add parameters and I would have to symmetrize it by hand. The dot product gives me symmetry and relabeling-invariance for free, so it is the clean first choice.

With a GCN encoder and that decoder, I already have the deterministic graph auto-encoder:

```text
Z = GCN(X, A)
A_hat = sigmoid(Z Z^T).
```

Training it as a Bernoulli reconstruction loss exposes the sparsity problem again. If I average binary cross-entropy over all `N^2` pairs, zeros dominate. The dense solution is to up-weight positive entries and rescale the mean loss; the equivalent sparse-harness solution is to sample non-edges and train on positives plus sampled negatives. These are the same case split in spirit: either reweight the rare ones or avoid showing all easy zeros at once.

Now I ask what the deterministic model is missing. It learns one point per node, and the only pressure on the space is reconstruction. A latent-variable version gives me a principled regularizer: draw node latents from an approximate posterior, reconstruct the adjacency from them, and pull the posterior toward a simple prior. The standard variational objective already has exactly that shape,

```text
ELBO = E_q[log p(A | Z)] - KL(q(Z | X,A) || p(Z)).
```

I have to adapt the VAE to a graph. A usual VAE encoder maps one independent datapoint to one posterior. Here, a node's posterior should depend on its neighborhood and features, so the recognition model has to be a GCN over the observed graph. I keep a factorized posterior over nodes for tractability, but each factor is produced by message passing:

```text
q(Z | X,A) = prod_i q(z_i | X,A)
q(z_i | X,A) = N(z_i | mu_i, diag(sigma_i^2)).
```

The encoder now needs two outputs per node. I use the same first GCN layer for both because the mean and uncertainty should be read from the same neighborhood evidence. Then two linear GCN heads produce `mu` and `log sigma`. It matters that this is `log sigma`, not `log sigma^2`, to match the official TensorFlow implementation. The scale is recovered as `sigma = exp(logstd)`, which is always positive.

To sample while keeping gradients, I use the reparameterization trick:

```text
z_i = mu_i + sigma_i * eps_i,     eps_i ~ N(0,I).
```

The randomness is now in `eps`, not in a parameter-dependent sampling operation, so the reconstruction term can be differentiated through `mu` and `logstd`. At evaluation time I should not inject sampling noise into link scores; the official code evaluates with `z_mean`, so I use `mu`.

The KL term is where a sign or factor error would silently change the method, so I derive it carefully. For one latent coordinate with `q = N(mu, sigma^2)` and `p = N(0,1)`,

```text
E_q[log p(z)] = -0.5 log(2*pi) - 0.5 (mu^2 + sigma^2)
E_q[log q(z)] = -0.5 log(2*pi) - 0.5 (1 + log sigma^2).
```

Subtracting gives the contribution that is added to the ELBO:

```text
-KL(q || p) = 0.5 * (1 + log sigma^2 - mu^2 - sigma^2).
```

I do not fully trust a hand derivation of a quantity this easy to flip a sign in, so I check it two ways. First against the textbook closed form `KL(N(m,s^2) || N(0,1)) = 0.5 (s^2 + m^2 - 1 - log s^2)`: negating that gives `0.5 (1 + log s^2 - m^2 - s^2)`, term for term what I have. At `(m, s) = (0.5, 2)` both expressions evaluate to `-0.9319`, and at `(1.3, 0.4)` to `-1.3413`. Second, as a guard against me having mis-copied the textbook form too, I estimate `E_q[log q - log p]` directly by sampling: drawing `z ~ N(m, s^2)` and averaging `log q(z) - log p(z)` gives about `0.934` at `(0.5, 2)` and `1.341` at `(1.3, 0.4)`, matching the closed-form KL (`0.9319`, `1.3413`) to sampling noise. The formula is right.

Summing over latent dimensions and nodes gives the full negative KL. Since my encoder emits `logstd = log sigma`, the implementation form is

```text
-KL = 0.5 * sum_j(1 + 2 * logstd_j - mu_j^2 - exp(2 * logstd_j)).
```

The natural place for this bracket to be largest is where the posterior matches the prior. Setting `mu = 0` and `sigma = 1` gives `0.5 (1 + 0 - 0 - 1) = 0`, and the verified KL at that point is also `0`, so the regularizer is exactly zero when the posterior is standard normal and strictly positive otherwise — the pull-toward-prior behavior I wanted. When I minimize a loss rather than maximize the ELBO, I add the positive KL:

```text
KL = -0.5 * sum_j(1 + 2 * logstd_j - mu_j^2 - exp(2 * logstd_j)).
```

I also have to get the scale faithful to the reference code. The official TensorFlow optimizer computes a dense weighted reconstruction mean, then subtracts

```text
(0.5 / N) * mean_i sum_j(1 + 2logstd_ij - mu_ij^2 - exp(logstd_ij)^2).
```

That expression is negative KL. Subtracting it therefore adds positive KL. The `mean_i` has already averaged over nodes; the extra `1/N` is still present in the reference implementation, so a PyG-style helper that returns positive node-averaged KL must be added as `kl_loss() / num_nodes`. I should not describe `kl_loss()` itself as an unaveraged total KL, and I should not omit the extra `1/N` if I am matching the canonical implementation and the PyG example.

There is one more notation trap: if I define the adjacency with diagonal entries already set to one, the GCN equation writes `D^{-1/2} A D^{-1/2}`. The GCN ancestor and the official code instead start from an adjacency without self-loops and add `I` during preprocessing. Those are the same convention if I say it explicitly. In a PyG scaffold, `GCNConv` with its default normalization adds self-loops before normalizing, matching the code convention.

Putting these pieces together gives the variational graph auto-encoder: shared GCN trunk, mean and log-standard-deviation heads, reparameterized node latents during training, posterior mean at evaluation, and the same inner-product Bernoulli decoder. The deterministic graph auto-encoder is the special case where I remove the log-standard-deviation head, the sampling step, and the KL. The featureless case is also simple: set `X = I`, so structure is the only input signal.

Here is the scaffold-shaped implementation I arrive at. It follows the canonical two-layer shape and keeps hidden width separate from latent width. I leave out the PyG `MAX_LOGSTD` clamp because it is a modern library guard, not part of the original TensorFlow reference.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LinkPredictor(nn.Module):
    """GCN posterior encoder plus inner-product Bernoulli decoder."""

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
            eps = torch.randn_like(self.logstd)
            return self.mu + eps * torch.exp(self.logstd)
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

For the sampled-edge scaffold, the surrounding loss is the negative-sampling alternative to the dense positive reweighting:

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

If I instead reproduce the original dense TensorFlow path, I flatten all entries of `Z Z^T`, use labels `adj_train + I`, compute `pos_weight = (N^2 - sum(adj_train)) / sum(adj_train)` and `norm = N^2 / ((N^2 - sum(adj_train)) * 2)`, and minimize `norm * weighted_BCE - negative_KL_term`. I want to confirm these two constants do the rebalancing I expect rather than just copy them. Take `N = 4` with `P = sum(adj_train) = 6` positive entries, so `N^2 = 16` and `neg = N^2 - P = 10`. Then `pos_weight = 10/6 = 1.667`, greater than one, so each rare positive entry is up-weighted relative to a zero — correct direction for a graph that is mostly zeros. And `norm = 16/(10*2) = 0.8`; the effective weight the negatives carry after the `norm` rescale is `norm * 2 * neg / N^2 = 0.8 * 2 * 10 / 16 = 1.0` exactly, so the averaged weighted loss is put back on the scale of a balanced per-entry loss instead of being dominated by the count of zeros. That is the same imbalance fix as negative sampling, applied densely. The sparse code above is the same model and the zero-subsampling case, adapted to the harness.
