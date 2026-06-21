## Research question

Learn an unconditional distribution over small graphs and sample new graphs whose degree distribution, clustering, and small-substructure counts match a held-out reference set. The design problem is the **generative model** (`GraphGenerator`): how a graph is represented, how the likelihood is trained, and how a new graph is sampled. Everything around it — datasets, padded adjacency representation, train loop, and MMD scoring — is fixed.

Two structural facts make this hard. First, **graphs are unordered**: the same graph on `n` nodes corresponds to up to `n!` adjacency matrices, so any entrywise reconstruction loss penalizes a correct graph produced under the wrong permutation, and the exact likelihood — a sum over `n!` permutations — is intractable. Second, **graphs are sparse and discrete**: only `O(n)` edges out of `O(n²)` slots, binary entries, and the object being scored is topology, not pixel values. A good model must respect permutation and keep the object an honest graph.

## Prior art / Background / Baselines

Representative alternatives attack the same two facts from different angles. Each has a clear observed limitation.

- **String surrogates (SMILES + RNN; Grammar VAE).** Linearize the graph to a string and run a sequence model. This borrows mature text-generation machinery and sidesteps ordering and discreteness, but the linearization hides the atoms-and-bonds object, and a character sampler emits many invalid strings; grammar decoding fixes syntax but not semantics, and the model is still not operating on graphs. Gap: the structure being modeled is hidden behind a fragile string encoding.
- **One-shot latent-variable model (GraphVAE).** Encode a graph to a vector `z` and decode a full probabilistic adjacency of fixed size at once, with edges independent Bernoulli given `z`. It is parallel and order-free in spirit, but reconstruction requires aligning the decoded graph to the target — a graph-matching problem costing `O(k⁴)` with `O(k²)` parameters — so it caps at a few dozen nodes; edge independence given `z` also hurts sample quality, because whether `(i,j)` is an edge depends on the rest of the structure. Gap: poor scaling and uncorrelated edges.
- **GNN-in-the-loop autoregression (DeepGMG).** Build the graph through local decisions (add a node, add an edge, choose an endpoint), running full message passing before each decision so every choice sees the current topology. It is expressive, but a fresh propagation per atomic decision costs on the order of `m·n²·diam(G)`, which is prohibitive past a few tens of nodes. Gap: topology-aware but far too slow.
- **RNN autoregression (GraphRNN).** Under a BFS ordering, emit the graph as a sequence of adjacency vectors using a graph-level RNN for state and an edge-level RNN for each row. It scales as `O(M·n)` with BFS bounding row width, but it makes `O(n²)` sequential decisions, conditions only through the recurrent state rather than directly on the already-generated topology, and dependencies between graph-adjacent nodes must travel through many recurrent steps. Gap: scalable but not directly topology-aware, with a long-horizon bottleneck.
- **Score-based / diffusion model for graphs (GDSS).** Embed the adjacency in continuous space and learn to reverse a Gaussian corruption process. Gaussian noise, however, turns the sparse 0/1 adjacency into a dense matrix where degree, clustering, and cycle counts are no longer defined, so the denoiser must recover structure from a non-graph object. Gap: continuizing a discrete object removes the statistics being evaluated.

## Fixed substrate / Code framework

The data, representation, and evaluation are frozen. Three datasets span the size range: `community_small` (100 two-community graphs, 12–20 nodes), `ego_small` (200 Citeseer ego graphs, 4–18 nodes), and `enzymes` (587 BRENDA protein-structure graphs, 10–125 nodes). Each graph is delivered as a binary, symmetric, zero-diagonal adjacency padded to `max_nodes`, with a `node_counts` tensor giving the true size; an 80/20 train/reference split is taken inside the loop. The loop runs 500 epochs, batch size 32, Adam, calls `train_step` per batch, then calls `sample(n_gen, device)`, thresholds output at 0.5, symmetrizes, drops self-loops, and scores. A parameter budget of 1.05× the largest baseline is enforced. Scoring is **MMD between graph statistics** of generated and reference graphs: degree and clustering MMD use a Gaussian-EMD kernel over normalized histograms; orbit MMD uses a plain Gaussian kernel (σ=30) on raw per-graph 4-orbit count vectors. **All four metrics are lower-is-better.**

## Editable interface

Only one region is editable: the `GraphGenerator` class in `custom_graphgen.py` (lines 446–590). Every method fills the same three-method contract:

- `__init__(self, max_nodes, **kwargs)` — build the model and create its optimizer.
- `train_step(self, adj, node_counts) -> dict` — one optimizer step on a batch of padded adjacency matrices `[B, max_nodes, max_nodes]` with sizes `[B]`; return a dict with at least `'loss'`.
- `sample(self, n_samples, device) -> (adj, node_counts)` — draw graphs: `adj` is `[n_samples, max_nodes, max_nodes]` (binary, symmetric, no self-loops) and `node_counts` is `[n_samples]` (each ≥ 2).

Available inside the region: `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, `numpy`, `math`. Helper classes/functions may be defined alongside `GraphGenerator`. The starting point is the scaffold default below — a **plain-MLP VAE**: flatten the adjacency, encode to a Gaussian latent, decode the whole adjacency back through an MLP, train on per-entry reconstruction BCE plus a small KL, and sample by drawing `z ~ N(0,I)` and thresholding decoded probabilities. It is the simplest order-blind one-shot fill; the ladder replaces exactly this class.

```python
# EDITABLE region of custom_graphgen.py (lines 446-590) — default fill: plain-MLP VAE
class GraphGenerator(nn.Module):
    """Default baseline: flatten-adjacency VAE with an MLP encoder/decoder."""

    def __init__(self, max_nodes, hidden_dim=256, latent_dim=64, lr=1e-3, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        adj_size = max_nodes * max_nodes

        self.encoder = nn.Sequential(
            nn.Linear(adj_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, adj_size),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def encode(self, adj):
        B = adj.shape[0]
        h = self.encoder(adj.view(B, -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        B = z.shape[0]
        logits = self.decoder(z).view(B, self.max_nodes, self.max_nodes)
        logits = (logits + logits.transpose(1, 2)) / 2                 # symmetric
        mask = 1 - torch.eye(self.max_nodes, device=z.device).unsqueeze(0)
        return logits * mask                                           # no self-loops

    def train_step(self, adj, node_counts):
        self.train()
        self.optimizer.zero_grad()
        mu, logvar = self.encode(adj)
        z = self.reparameterize(mu, logvar)
        adj_logits = self.decode(z)
        recon_loss = F.binary_cross_entropy_with_logits(adj_logits, adj, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.001 * kl_loss
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "recon_loss": recon_loss.item(), "kl_loss": kl_loss.item()}

    def sample(self, n_samples, device):
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            adj = (torch.sigmoid(self.decode(z)) > 0.5).float()
            node_mask = (adj.sum(dim=-1) > 0).float()
            node_counts = torch.clamp(node_mask.sum(dim=-1).long(), min=2)
        return adj, node_counts
```

## Evaluation settings

Three datasets — `community_small`, `ego_small`, `enzymes` — each over three seeds {42, 123, 456}. Four metrics, **all lower is better**: `mmd_degree`, `mmd_clustering`, `mmd_orbit`, and their average `mmd_avg`. The fixed schedule (500 epochs, batch 32, single GPU, Adam) is identical for every method; the reported figure of merit is per-dataset `mmd_avg` and the overall mean across the three datasets.
