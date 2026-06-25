**Problem.** DiGress tamed GRAN's catastrophic seed (overall mean 0.301 → 0.265, seed-42 ego 0.819 →
0.082) but left a clustering/degree residual on the sparse datasets — ego clustering 0.446 (seed-456 at
1.022), enzymes clustering 0.527 — because its dense `Bernoulli(0.5)` corruption limit forces the
reverse process to re-sparsify and can stall on some seeds. Both GRAN's collapse and DiGress's residual
are *multi-step generation* pathologies driven by a dense intermediate state.

**Key idea.** Remove the multi-step trajectory entirely: a one-shot VAE that draws `z ~ N(0,I)` once and
decodes the whole adjacency in a single deterministic pass — there is no refinement sweep and no reverse
chain to stall. Make the encoder *graph-aware* (a 2-layer GCN, `X' = D^{-1/2}(A+I)D^{-1/2}XW`, with
identity node features and a permutation-invariant mean-pool to the latent), keep a node-existence head
so a fixed-`N` decoder can drop empty slots, and train on entrywise reconstruction BCE plus a tiny KL.

**Why (and the gap vs. the full version).** This fill is a *matching-free* GraphVAE. The signature
contribution of the latent-variable graph generator — the **graph-matching-aligned reconstruction loss**
(a pairwise-similarity assignment solved by max-pooling power iteration, discretized with the Hungarian
algorithm, then `A' = X A Xᵀ`) that makes the objective permutation-*invariant* — is **entirely absent**.
The harness trains on plain entrywise BCE against the adjacency in the order the loop padded it, which is
the order-sensitive loss matching was invented to fix. That `O(k⁴)` matching cannot run within the
budget (`max_nodes` up to 125, 500 epochs, 1.05× param cap), so it is dropped; the encoder is a plain
degree-normalized GCN, not the edge-conditioned conv with a gated readout.

**Why it still wins here.** A one-shot decode has no stochastic multi-step trajectory, so it cannot
suffer GRAN's stall or DiGress's re-sparsification collapse — the seed-to-seed variance comes only from
the latent draw. The KL is down-weighted to `0.001` to avoid posterior collapse.

**Hyperparameters.** `hidden_dim=256`, `latent_dim=64`, GCN encoder (2 layers) + MLP decoder (3 layers),
node-existence MLP, node-loss weight `0.5`, KL weight `0.001`, Adam `lr=1e-3`.

**What to watch.** Expect the lowest overall mean mmd_avg of the three rungs and the tamest worst seed —
decisive on `ego_small`, with no seed blowing up the way DiGress's seed-456 did. The shared ceiling
should remain `enzymes` (largest graphs, most orderings to waste capacity on, biggest decoder), where the
dropped graph-matching would be needed to break past ~0.4.

```python
# EDITABLE region of custom_graphgen.py (lines 446-590) — step 3: GraphVAE (GCN-encoder VAE, no matching)
class GCNLayer(nn.Module):
    """Simple GCN layer: X' = D^{-1/2} A_hat D^{-1/2} X W."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # Add self-loops
        I = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        A_hat = adj + I
        # Degree normalization
        D = A_hat.sum(dim=-1, keepdim=True).clamp(min=1)
        D_inv_sqrt = 1.0 / torch.sqrt(D)
        A_norm = A_hat * D_inv_sqrt * D_inv_sqrt.transpose(-1, -2)
        out = torch.bmm(A_norm, x)
        return self.linear(out)


class GraphGenerator(nn.Module):
    """GraphVAE: Variational Autoencoder for graph generation.

    Uses GCN encoder to produce graph-level latent representation,
    and MLP decoder to produce adjacency matrix probabilities.
    """

    def __init__(self, max_nodes, hidden_dim=256, latent_dim=64, lr=1e-3, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim
        adj_size = max_nodes * max_nodes

        # GCN encoder
        self.gcn1 = GCNLayer(max_nodes, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, adj_size),
        )

        # Node existence predictor
        self.node_pred = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_nodes),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def encode(self, adj):
        B, N, _ = adj.shape
        # Use identity as node features (one-hot position)
        x = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        h = F.relu(self.gcn1(x, adj))
        h = F.relu(self.gcn2(h, adj))
        # Graph-level readout (mean pooling)
        h_graph = h.mean(dim=1)  # [B, hidden]
        return self.fc_mu(h_graph), self.fc_logvar(h_graph)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B = z.shape[0]
        logits = self.decoder(z).view(B, self.max_nodes, self.max_nodes)
        # Symmetrize
        logits = (logits + logits.transpose(1, 2)) / 2
        # Zero diagonal
        mask = 1 - torch.eye(self.max_nodes, device=z.device).unsqueeze(0)
        return logits * mask

    def train_step(self, adj, node_counts):
        self.train()
        self.optimizer.zero_grad()

        mu, logvar = self.encode(adj)
        z = self.reparameterize(mu, logvar)
        adj_logits = self.decode(z)
        node_logits = self.node_pred(z)  # [B, max_nodes]

        # Reconstruction loss
        recon_loss = F.binary_cross_entropy_with_logits(adj_logits, adj, reduction="mean")

        # Node existence loss
        node_target = (adj.sum(dim=-1) > 0).float()  # [B, max_nodes]
        node_loss = F.binary_cross_entropy_with_logits(node_logits, node_target, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + 0.5 * node_loss + 0.001 * kl_loss
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "recon": recon_loss.item(), "kl": kl_loss.item()}

    def sample(self, n_samples, device):
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            adj_logits = self.decode(z)
            adj = (torch.sigmoid(adj_logits) > 0.5).float()
            node_logits = self.node_pred(z)
            node_probs = torch.sigmoid(node_logits)
            node_mask = (node_probs > 0.5).float()
            # Mask adjacency by existing nodes
            adj = adj * node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
            node_counts = node_mask.sum(dim=-1).long()
            node_counts = torch.clamp(node_counts, min=2)
        return adj, node_counts
```
