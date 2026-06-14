## Research question

Unconditional graph generation: learn a distribution over a set of small graphs and sample new
graphs whose *structure* — degree distribution, clustering, the counts of small substructures — is
statistically indistinguishable from a held-out reference set. The single thing being designed is the
**generative model** (`GraphGenerator`): how a graph is represented, how the likelihood is trained,
and how a fresh graph is drawn. Everything around it — the datasets, the padded adjacency
representation, the train loop, and the MMD scoring — is fixed.

Two structural facts make this hard, and every method on the ladder is a different answer to them.
First, **graphs are unordered**: the same graph on `n` nodes is one of up to `n!` adjacency matrices,
so any loss that compares adjacency matrices entrywise punishes a *correct* graph produced in the
wrong order, and the honest likelihood — a sum over `n!` permutations — is intractable. Second,
**graphs are sparse and discrete**: `O(n)` edges out of `O(n²)` slots, binary entries, and the thing
being scored is topology, not pixel values. A good model has to be permutation-aware and keep the
object an honest graph.

## Prior art before the first rung (the generation-paradigm lineage)

The first rung reacts to the autoregressive line of graph generators — the family that builds a graph
out of local, topology-aware decisions — while the one-shot and string lineages mark what it is
trying to avoid.

- **String surrogates (SMILES + RNN; Grammar VAE, Kusner et al. 2017).** Linearize the graph to a
  string and run a sequence model. Borrows all of text generation's machinery and dodges both
  graph-specific walls because a string is already ordered, but the linearization is lossy (the model
  never sees the atoms-and-bonds object) and a character sampler emits a flood of *invalid* strings;
  grammar decoding fixes syntax but not semantics, and it is still not a graph. Gap: hides the
  structure being modeled.
- **One-shot latent-variable (GraphVAE, Simonovsky & Komodakis 2018; arXiv:1802.03480).** Encode a
  graph to `z`, decode a whole probabilistic adjacency at fixed size `k`, edges independent Bernoulli
  given `z`. Parallel and order-free in spirit, but to score the reconstruction it must *align* the
  decoded graph to the target — a graph-matching problem costing `O(k⁴)` with `O(k²)` parameters, so
  it caps at a few dozen nodes — and edge independence given `z` is exactly what wrecks sample
  quality, since whether `(i,j)` is an edge depends on the rest of the structure. Gap: doesn't scale,
  edges uncorrelated.
- **GNN-in-the-loop autoregression (DeepGMG, Li et al. 2018).** Build the graph by local decisions
  (add node? add edge? to whom?), running full message passing before *each* decision, so every
  decision sees the real topology. Expressive, but a fresh propagation per atomic decision costs on
  the order of `m·n²·diam(G)` — hopeless past a few tens of nodes. Gap: topology-aware but far too
  slow.
- **RNN autoregression (GraphRNN, You et al. 2018; arXiv:1802.08773).** Drop the GNN: under a BFS
  ordering, emit the graph as a sequence of adjacency vectors with a graph-level RNN holding state and
  an edge-level RNN emitting each row entry-by-entry. Scales (`O(M·n)` with BFS bounding the row
  width), but spends `O(N²)` sequential decisions, conditions only through the recurrent state rather
  than directly on topology, and the long generation sequence stretches the dependency between
  graph-adjacent nodes across many recurrent steps. Gap: scalable but not directly topology-aware, and
  a long-horizon bottleneck.
- **Score-based / diffusion (GDSS, Jo et al. 2022; arXiv:2202.02514).** Borrow the strongest image
  paradigm — a forward corruption process and a learned reverse — by embedding the adjacency in a
  continuous space and adding Gaussian noise. But Gaussian noise on a graph turns the sparse 0/1
  adjacency into a dense fog where degree, clustering, and cycle counts are no longer even defined, so
  the denoiser is reconstructing structure from something that has no structure. Gap: continuizing a
  discrete object destroys the very statistics being modeled.

## The fixed substrate

The data, representation, and evaluation are frozen and must not be touched. Three datasets span the
size range — `community_small` (100 two-community graphs, 12–20 nodes), `ego_small` (200 Citeseer ego
graphs, 4–18 nodes), and `enzymes` (587 BRENDA protein-structure graphs, 10–125 nodes). Each graph is
delivered as a binary, symmetric, zero-diagonal adjacency padded to `max_nodes`, with a `node_counts`
tensor giving the true size; an 80/20 train/reference split is taken inside the loop. The loop runs
the same schedule for every method — 500 epochs, batch size 32, Adam — calls `train_step` per batch,
then at the end calls `sample(n_gen, device)`, thresholds the output at 0.5, symmetrizes, drops
self-loops, and scores. A parameter budget of 1.05× the largest baseline is enforced. Scoring is
**MMD between graph statistics** of generated vs. reference graphs: degree and clustering MMD use a
Gaussian-EMD kernel over normalized histograms, orbit MMD uses a plain Gaussian kernel (σ=30) on raw
per-graph 4-orbit count vectors. **All four metrics are lower-is-better.**

## The editable interface

Exactly one region is editable — the `GraphGenerator` class in `custom_graphgen.py` (lines 446–590).
Every method on the ladder is a fill of the same three-method contract:

- `__init__(self, max_nodes, **kwargs)` — build the model and create its optimizer.
- `train_step(self, adj, node_counts) -> dict` — one optimizer step on a batch of padded adjacency
  matrices `[B, max_nodes, max_nodes]` with sizes `[B]`; return a dict with at least `'loss'`.
- `sample(self, n_samples, device) -> (adj, node_counts)` — draw graphs: `adj` is
  `[n_samples, max_nodes, max_nodes]` (binary, symmetric, no self-loops) and `node_counts` is
  `[n_samples]` (each ≥ 2).

Available inside the region: `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, `numpy`,
`math`. Helper classes/functions may be defined alongside `GraphGenerator`. The starting point is the
scaffold default below — a **plain-MLP VAE**: flatten the adjacency, encode to a Gaussian latent,
decode the whole adjacency back through an MLP, train on per-entry reconstruction BCE plus a tiny KL,
and sample by drawing `z ~ N(0,I)` and thresholding the decoded probabilities. It is the simplest
order-blind one-shot fill, and the ladder replaces exactly this class.

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

Three datasets — `community_small`, `ego_small`, `enzymes` — each over three seeds {42, 123, 456}.
Four metrics, **all lower is better**: `mmd_degree`, `mmd_clustering`, `mmd_orbit`, and their average
`mmd_avg`. The fixed schedule (500 epochs, batch 32, single GPU, Adam) is identical for every method;
the reported figure of merit is per-dataset `mmd_avg` and the overall mean across the three datasets.
