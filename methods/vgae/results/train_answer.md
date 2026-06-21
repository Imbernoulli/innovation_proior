I frame the problem as unsupervised link prediction on a single attributed graph. The input is a feature matrix and an incomplete adjacency, and the only supervision comes from the observed edges themselves. The method I propose is the Variational Graph Auto-Encoder, abbreviated VGAE. It is the latent-variable counterpart of the deterministic graph auto-encoder: instead of learning one fixed embedding per node, it learns a diagonal Gaussian posterior over node embeddings, reconstructs the graph through an inner-product decoder, and regularizes the posterior toward a standard normal prior. The result is an end-to-end generative model of the adjacency that uses both node features and graph structure.

The first design choice is the encoder. A node’s latent distribution should depend on its own features and on its neighborhood, so a graph convolutional network is the natural recognition model. I use a shared two-layer GCN trunk that transforms raw node features into hidden representations by repeated propagation and nonlinearity. From that shared trunk I split into two separate GCN heads: one head outputs the mean vector mu_i for each node, and the other outputs the log standard deviation logstd_i. Sharing the first layer is important because the mean and the uncertainty should be read from the same neighborhood evidence. Using log standard deviation rather than log variance keeps the scale recovery numerically stable, matching the canonical reference implementation.

The second design choice is the decoder. Because the graph is undirected and we want a symmetric score for each candidate pair, I use the simplest possible decoder: the inner product between two latent vectors followed by a sigmoid. For nodes i and j, the reconstructed edge probability is sigmoid(z_i^T z_j). This decoder has no extra parameters, respects permutation equivariance, and makes the geometric bias of link prediction explicit: pairs with high latent similarity are predicted to be connected. In matrix form the full reconstruction is sigmoid(Z Z^T).

The third design choice is how to train without labels. VGAE optimizes the evidence lower bound, or ELBO, which consists of a reconstruction term and a KL divergence term. The reconstruction term rewards the model for putting probability mass on observed edges, while the KL term pulls each approximate posterior toward the standard normal prior. Because the exact posterior over all node latents is intractable, the model factorizes across nodes: q(Z | X, A) is the product over i of q(z_i | X, A). Each factor is a diagonal Gaussian whose parameters come from the GCN encoder.

To make the expectation in the ELBO differentiable, I use the reparameterization trick. During training I sample eps from a standard normal and compute z_i = mu_i + sigma_i * eps_i, where sigma_i is exp(logstd_i). The randomness is independent of the parameters, so gradients flow back through mu and logstd. At evaluation time I discard the sampling noise and use the posterior mean mu_i. This is the standard protocol for variational auto-encoders and gives deterministic ranking scores at test time.

For the KL term I start from the closed form for a one-dimensional Gaussian against a standard normal prior. The contribution that is added to the ELBO is one half times one plus log sigma squared minus mu squared minus sigma squared. Since the encoder outputs logstd, the implementation form is one half times one plus two times logstd minus mu squared minus exp(2 * logstd). When I minimize a loss rather than maximize the ELBO, I add the positive KL, which is the negative of that expression. Averaging over nodes gives a compact per-node KL loss.

In the dense original formulation the reconstruction loss is taken over all N squared pairs with positive reweighting to account for sparsity. The number of possible edges is usually enormous compared with the number of observed edges, so training on every zero would drown the signal from the positive edges. In a sparse harness I instead sample a fixed number of non-edges and train on the observed positive edges together with those sampled negatives. This subsampling is mathematically equivalent to a stochastic approximation of the dense objective and is much more practical.

I also keep the deterministic graph auto-encoder in mind as a special case. If I remove the log-standard-deviation head, the sampling step, and the KL term, VGAE reduces to GAE. In the featureless setting I simply replace the feature matrix with the identity matrix, so each node is represented only by its structural position and all signal comes from the graph topology. These variants are useful sanity checks, but the canonical model is VGAE.

Hyperparameter choices follow the reference and the trajectory experience. The hidden width is set generously, the latent width is smaller than the hidden width, and dropout is applied to the hidden layer but not to the final mu and logstd heads. A small KL coefficient is used when the surrounding training loop expects only a scalar score, so the KL gradient still reaches the encoder without overwhelming the reconstruction gradient. Training proceeds with Adam and early stopping on a validation set of held-out edges, exactly as in the standard link-prediction benchmark.

The code below puts everything together in a self-contained PyTorch Geometric module. The LinkPredictor class implements the GCN encoder, the reparameterized sampling, the inner-product decoder, and the per-node KL loss. A short script at the bottom builds a synthetic graph, trains the model for a few iterations, and prints the training loss so the whole file is runnable.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LinkPredictor(nn.Module):
    """Variational Graph Auto-Encoder (VGAE) for unsupervised link prediction."""

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


def train_step(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_label_index)
    recon = F.binary_cross_entropy_with_logits(logits, data.edge_label)
    loss = recon + model.kl_loss() / data.num_nodes
    loss.backward()
    optimizer.step()
    return float(loss)


if __name__ == "__main__":
    # Tiny synthetic graph: 30 nodes, 5-dimensional features, 60 undirected edges.
    num_nodes = 30
    in_channels = 5
    x = torch.randn(num_nodes, in_channels)

    torch.manual_seed(0)
    src = torch.randint(0, num_nodes, (60,))
    dst = torch.randint(0, num_nodes, (60,))
    edge_index = torch.unique(torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0), dim=1)

    # Build a small training batch with positive edges and sampled negatives.
    pos_edges = edge_index[:, :20]
    neg_src = torch.randint(0, num_nodes, (20,))
    neg_dst = torch.randint(0, num_nodes, (20,))
    neg_edges = torch.stack([neg_src, neg_dst], dim=0)
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=1)
    edge_label = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))])

    class SimpleData:
        pass

    data = SimpleData()
    data.x = x
    data.edge_index = edge_index
    data.edge_label_index = edge_label_index
    data.edge_label = edge_label
    data.num_nodes = num_nodes

    model = LinkPredictor(in_channels, hidden_channels=16, latent_channels=8, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        loss = train_step(model, data, optimizer)
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | loss {loss:.4f}")
```
