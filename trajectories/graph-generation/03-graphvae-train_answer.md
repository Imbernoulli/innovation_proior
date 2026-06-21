DiGress did what I predicted at the variance end and exposed exactly the weakness I flagged. The scheduled 50-step reverse from the true noise prior killed GRAN's catastrophic seed — seed-42 ego dropped from $0.819$ to $0.082$, and the overall mean fell from $0.301$ to $0.265$. But its $\mathrm{mmd\_clustering}$ stayed high almost everywhere: $0.446$ mean on `ego_small` (seed-456 a brutal $1.022$) and $0.527$ mean on `enzymes`, with $\mathrm{mmd\_degree}$ on enzymes at $0.682$. That residual is exactly what I warned about: the uniform $\mathrm{Bernoulli}(0.5)$ corruption limit is a *dense* random graph, so the reverse process spends its early steps merely re-sparsifying, and on a seed where that re-sparsification stalls the clustering and degree statistics blow up. The common thread under both GRAN's collapse and DiGress's residual is the same: both are sampling-process pathologies driven by a *dense intermediate state* that the generator must fight its way back from through a process that does not natively respect sparsity. Each rung made the generation dynamics *more* elaborate — five refinement sweeps, then fifty scheduled steps — to tame that, which invites the contrarian question: what if the elaborate multi-step process is itself the source of the variance, and a single-shot generator that draws one latent and decodes the whole graph in one pass would simply have no trajectory to collapse?

I propose **GraphVAE**, a one-shot GCN-encoder variational autoencoder with no graph matching. The latent-variable family is the natural home for single-shot generation: draw $z\sim\mathcal N(0,I)$ once and decode the entire adjacency in a single deterministic forward pass, with no iterative sweep, no scheduled reverse chain, nothing to stall. I set it up as a proper VAE — prior $p(z)=\mathcal N(0,I)$, a recognition model $q(z\mid G)$ emitting $(\mu, \log\sigma^2)$, the reparameterization $z = \mu + \sigma\cdot\epsilon$ with $\epsilon\sim\mathcal N(0,I)$ so gradients flow through the sampling, and the bound
$$L = \mathbb E_q[-\log p(G\mid z)] + \mathrm{KL}[q(z\mid G)\,\|\,p(z)],\qquad \mathrm{KL} = -\tfrac12\sum_j\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)$$
with the Gaussian KL in closed form. The only pieces I design per data type are the encoder and the reconstruction term $\log p(G\mid z)$.

Two things matter for *this* task. First, the encoder should be *graph-aware*, not the flat MLP over the flattened adjacency the default used, because the whole lesson of GRAN and DiGress is that edge-relevant representations come from message passing, not a flat readout. So I encode with a GCN: identity node features $x$, two layers of $X' = D^{-1/2}(A+I)D^{-1/2}XW$ with self-loops and degree normalization, then a permutation-invariant mean-pool over nodes into a graph-level vector, then linear heads to $(\mu, \log\sigma^2)$. The degree-normalized aggregation is the right inductive bias here precisely because, unlike GRAN's attention over an *uncertain* adjacency, the encoder always sees the *true* clean adjacency, so a fixed normalized aggregation is well-behaved and cheap. Second, the decoder: an MLP from $z$ to the full $\mathrm{max\_nodes}\times\mathrm{max\_nodes}$ logit matrix, symmetrized $(L+L^\top)/2$ and zero-diagonalled — the symmetric, self-loop-free output the loop requires, produced in one shot.

The reconstruction loss is where the latent family classically breaks, and where I make the central design decision of this rung. A graph has no canonical node order, so the same graph is one of up to $n!$ adjacency matrices; comparing the decoded adjacency to the target *entrywise* punishes a correct graph with two nodes swapped as if it were completely wrong. The principled cure is *graph matching*: find a one-to-one assignment between decoded slots and ground-truth nodes by maximizing a pairwise structural similarity, then score the reconstruction in the aligned frame, making the loss permutation-aware. That matching is the signature move of the latent-variable approach to graphs (Simonovsky & Komodakis 2018) — it is what makes the reconstruction term well-defined at all — but it costs $O(k^4)$, since the similarity is indexed by pairs of pairs, and needs a power-iteration solver plus a Hungarian discretization inside every forward pass. Within this harness, with $\mathrm{max\_nodes}$ up to $125$ on enzymes, a $1.05\times$ parameter budget, and a 500-epoch schedule shared across three datasets, that matching is simply not affordable. So I make the deliberate harness-forced simplification: drop the matching entirely and train on the **entrywise reconstruction BCE** against the target in the order the loop padded it.

I owe an argument for why the order-sensitive loss is still the right rung here rather than a regression. The argument is that the datasets are small and the harness fixes a *consistent* node ordering: each graph is delivered in one padded order, the encoder produces a latent from that ordered adjacency, and the decoder is asked to reproduce *that same* ordered adjacency — so within a single graph the loss is self-consistent even without matching, and the model is effectively learning to autoencode the dataset's particular orderings rather than the abstract graph. On these small, fairly regular datasets that is enough to learn a reasonable latent of *structure type*, because the structure — two-community, ego, enzyme motif — is strongly correlated with the ordered adjacency the harness presents. What I give up is the permutation *invariance* of the objective, so the model wastes some capacity on orderings; what I gain is a `train_step` that is a single GCN encode plus a single MLP decode plus a closed-form KL, with no $O(k^4)$ matching and no inner solver, the only version that fits the budget. This is the central same-versus-different note for the rung: the load-bearing contribution of the latent-variable graph generator — the matching-aligned reconstruction — is *exactly the part this fill omits*, and what remains is a strictly simpler GCN-encoder VAE chosen because the matched version cannot run here.

Two pieces complete the fill. A **node-existence predictor**: an MLP from $z$ to $\mathrm{max\_nodes}$ logits, trained with BCE against the true node mask $\mathrm{adj}.\mathrm{sum}(-1)>0$, weighted at $0.5$ against the edge reconstruction — this lets a fixed-$N$ decoder represent a graph on $n<N$ nodes, and it is cleaner than DiGress's mechanism because here the latent directly drives both the adjacency and the node mask in one shot, so they are consistent by construction rather than read off a refined adjacency. And the **KL weight**, kept tiny at $0.001$: with a strong reconstruction target and a small latent, a full-weight KL would dominate early and collapse the posterior toward the prior before the decoder learns any structure — the classic posterior-collapse failure — so a down-weighted KL lets the latent carry real structural information while still regularizing toward $\mathcal N(0,I)$ enough that sampling from the prior produces sensible graphs. The total training loss is $\mathrm{recon\_loss} + 0.5\cdot\mathrm{node\_loss} + 0.001\cdot\mathrm{KL}$.

Sampling is the part that makes me believe this rung is *stable* where the others were not. I draw $z\sim\mathcal N(0,I)$ once, decode the full adjacency logits in a single pass, threshold at $0.5$, and mask by the node-existence prediction; node counts are the masked count, clamped to $\ge 2$. There is no iterative refinement to stall (GRAN's failure mode) and no multi-step reverse chain whose early steps must re-sparsify a dense sample (DiGress's failure mode). The whole generation is one deterministic decode of a single Gaussian draw, so the seed-to-seed variance comes *only* from the latent draw and the learned decoder, not from a stochastic multi-step trajectory that can diverge. That is the structural reason I expect the tightest seed spread of the three: the elaborate generation dynamics I added at steps 1 and 2 to fight collapse were themselves a source of the collapse, and removing them removes the failure mode. I expect the lowest overall mean of the three rungs and the tamest worst seed — no blow-up like DiGress's ego seed-456 — with the biggest gains exactly on the statistics DiGress struggled with, `ego_small` clustering and degree. Where the dropped matching should bite is `enzymes`: the largest, most variable graphs (up to 125 nodes) give the entrywise order-sensitive loss the most orderings to waste capacity on and the largest decoder output, so I expect enzymes to stay the shared ceiling near $\sim 0.4$ — naming exactly the permutation-matching machinery a further rung would have to afford.

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
