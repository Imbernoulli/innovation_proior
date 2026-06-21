GRAN's overall mmd_avg of $0.301$ was one catastrophic seed averaged with two good ones: seed-42 came in at $0.219$ on `community_small`, $0.819$ on `ego_small`, and $0.814$ on `enzymes`, each wrecked the same way by a blown-up $\mathrm{mmd\_degree}$ (1.22 on ego, 1.44 on enzymes) and $\mathrm{mmd\_orbit}$ (1.12, 0.97), while seeds 123 and 456 were perfectly respectable with orbit MMD essentially zero. So GRAN is not a bad edge predictor; it is a *seed-fragile sampler*. The one limitation that forces the next move is precisely there: an unanchored five-step resampling from an arbitrary, far-too-dense $\mathrm{Bernoulli}(0.3)$ init has no schedule and no defined corruption process, so it has no reason to converge — whether a seed lands on a clean sparse graph or stalls on a dense blob is a coin flip. I want a generator whose *sampling process itself* is principled: a controlled trajectory from noise to a clean graph, not a five-shot gamble.

I propose **DiGress**, a discrete edge-flip diffusion model. Diffusion is the strongest paradigm I know for "iteratively turn noise into a sample": a fixed forward process slowly corrupts a clean datapoint into noise through a Markov chain, and a learned network inverts one step at a time; to sample, start from the noise prior and denoise repeatedly. This is the right answer to GRAN's failure structurally — it replaces the unanchored sweep with a *defined* corruption process whose exact reverse the network is trained to follow, step by calibrated step. And I keep the subtlety that makes diffusion train well: rather than predict the previous noisy state — a high-variance target that depends on whatever noise I happened to draw — I train the network to predict the *clean* graph and reconstruct the reverse step from that, the same $x_0$-prediction DDPM uses to strip out label noise.

The obvious port, the one GDSS (Jo et al. 2022) took, is to embed the adjacency in continuous space and add Gaussian noise — and it breaks for a reason that is the whole point. Run that forward process on one of these graphs: at $t=0$ the adjacency is sparse 0/1, almost all zeros; add a little Gaussian noise, then more, and by the middle of the chain every entry is some real number around the noise scale and the matrix is *dense*. Now ask the degree of node $i$, or its triangle count — undefined, because there are no edges, just a fog of continuous values. The sparsity that *was* the data is gone, destroyed by the very noise I added, and these are exactly the degree, clustering, and orbit statistics I am scored on. That is the same wall as GRAN's over-dense init in different clothes: a process that passes through dense, structureless intermediate states has no good way to land on a sparse structured graph. The problem is not diffusion; it is continuizing a discrete object.

So I keep the graph discrete the whole way down: add noise that *edits the graph* — flips an edge in or out — but leaves it, at every step, an honest discrete graph. Then the intermediate states stay sparse, the structural notions stay defined, and denoising becomes a clean question: given a corrupted graph, say which edges should really be there. For a binary adjacency this is the simplest discrete corruption: at step $t$, *flip* each edge entry with some probability, symmetrically. I tie the flip probability to a cosine noise schedule for the cumulative survival,
$$\bar\alpha_t = \cos^2\!\Big(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\Big),$$
normalized so $\bar\alpha_0 = 1$ with a small $s\approx 0.008$ to corrupt gently near both ends, and set the per-step flip probability $\mathrm{flip\_prob} = 0.5\,(1 - \bar\alpha_t)$. At $t\to 0$ nothing flips (the graph is clean); as $t$ grows toward $T$ the flip probability rises toward $0.5$, the point at which each edge entry is an independent fair coin — so the corruption limit is a $\mathrm{Bernoulli}(0.5)$ random graph, independent of the clean input, which is exactly the property I need for a sampling prior. To apply the corruption I draw a flip mask, symmetrize it (keep the upper triangle, mirror it) so the noisy adjacency stays symmetric, and XOR it into the clean adjacency, $\mathrm{adj\_noisy} = |\mathrm{adj} - \mathrm{flip\_mask}|$. I sample a random timestep per graph and jump straight to the corrupted graph, so the whole chain trains in parallel with no unrolling.

The denoiser takes the noisy adjacency and the timestep and predicts the *clean* edge logits. Because each edge's target is a binary label — edge or no-edge — the entire generative problem dissolves into a pile of independent per-edge classifications: "is this edge really present?" No graph matching (GraphVAE's $O(k^4)$ curse), no decoding a continuous adjacency, no alignment — just binary cross-entropy of the predicted edge logits against the clean adjacency. That is the payoff of staying discrete: where GRAN had to gamble on a refinement sweep, diffusion with an $x_0$ target turns generation into supervised classification, because the forward process already told me, for each edge, exactly what the clean answer was. I keep the node-existence head from GRAN — a per-node MLP trained with BCE against the true node mask $\mathrm{adj}.\mathrm{sum}(-1)>0$, weighted at $0.5$ — so the fixed-$N$ adjacency can still drop empty slots; the total loss is $\mathrm{edge\_loss} + 0.5\cdot\mathrm{node\_loss}$ with a grad-norm clip at $1.0$.

I build the denoiser as a graph transformer, because attention is a natural fit for edge prediction: every pair of nodes already has an attention score, exactly the object I want to turn into an edge decision. Each layer does multi-head self-attention over node features, but the *edges modulate the attention* rather than sitting passive — I take the current noisy adjacency, embed each entry into a per-head bias through a linear map, and add that bias into the attention scores, so an edge between two nodes raises or lowers how much they attend. Node features come from a linear embedding of an identity input plus a time embedding (a small SiLU MLP of the normalized timestep, broadcast over nodes), so the network knows *how noisy* the graph it is looking at is and can denoise aggressively early and gently late. After the stack, the edge predictor reads $[n_i, n_j, \mathrm{adj\_noisy}_{ij}]$ per pair into a logit, symmetrizes $(L+L^\top)/2$, and zeroes the diagonal — the same symmetric, self-loop-free output the loop requires — with residual connections and LayerNorm throughout. This is the attentive edge predictor I already trusted from GRAN, now placed inside a *defined* denoising process instead of an unanchored refinement.

Sampling is where the diffusion structure earns its keep over GRAN's gamble. I start from the corruption limit — a $\mathrm{Bernoulli}(0.5)$ random symmetric adjacency, the prior the forward process converges to — and walk $t$ from $T-1$ down to $0$. At each step I run the denoiser on the current adjacency and timestep to get $\mathrm{edge\_probs} = \sigma(\mathrm{edge\_logits})$; for every step but the last I *resample* $\mathrm{adj} = \mathrm{Bernoulli}(\mathrm{edge\_probs})$, symmetrizing; at the final step I threshold at $0.5$ for a clean discrete output. Then I mask the adjacency by the node-existence prediction and read node counts off it, clamped to $\ge 2$. Contrast this with GRAN: GRAN started from an arbitrary $\mathrm{Bernoulli}(0.3)$ init matching no corruption limit and took a fixed five sweeps with no schedule; here I start from the *exact* prior the forward process defines and take $50$ scheduled denoising steps, each a calibrated partial cleanup. The trajectory is anchored at both ends — known noise prior, known clean target — and the schedule controls how much structure each step commits. That is the mechanism GRAN's collapse-prone seed-42 was missing.

I want to be honest that this is a deliberately reduced form of discrete graph diffusion, forced by the harness contract. The fuller version diffuses *node types* as well as edges with their own transition matrix; here only the binary edges diffuse and node existence is a separate auxiliary BCE head. It uses *marginal* transition matrices $Q = \alpha I + \beta\,\mathbf{1}m^\top$ whose limit is the data's empirical edge frequency — keeping the noisy graphs as sparse as the data all the way down; here the corruption is *uniform* edge-flipping whose limit is the dense $\mathrm{Bernoulli}(0.5)$ random graph, so the reverse process must spend its early steps merely re-sparsifying, exactly the inefficiency the marginal prior was designed to remove. It samples by marginalizing an *analytic Bayes posterior* $q(z_{t-1}\mid z_t, x)$ over the network's clean-graph belief; here I skip the analytic posterior and resample directly from the predicted edge probabilities, a coarser reverse step. And it feeds the denoiser *structural and spectral features* — cycle counts, Laplacian eigenvalues — to beat the 1-Weisfeiler-Leman expressivity ceiling, the one big advantage discreteness unlocks, whereas this denoiser is a plain edge-bias graph transformer with none of those extras. So I keep the load-bearing idea — discrete edge-flipping diffusion with an $x_0$-predicting graph-transformer denoiser and a scheduled reverse process — and drop the marginal prior, the analytic posterior, the node diffusion, and the structural features. I expect the scheduled 50-step reverse to stop GRAN's seed-42 collapse, bringing the worst-seed ego/enzymes numbers down from $\sim 0.8$ toward $0.2$–$0.4$ and the overall mean below $0.301$; and I expect to pay for the uniform prior and missing features in exactly $\mathrm{mmd\_clustering}$ and $\mathrm{mmd\_orbit}$ on the sparser graphs, where re-sparsification and the absent cycle features cannot be hidden — a tamer worst seed but a clustering/orbit residual a better-matched prior could still beat.

```python
# EDITABLE region of custom_graphgen.py (lines 446-590) — step 2: DiGress (discrete edge-flip diffusion)
class GraphTransformerLayer(nn.Module):
    """Graph transformer layer with edge-aware attention."""

    def __init__(self, dim, n_heads=4, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or 4 * dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.edge_bias = nn.Linear(1, n_heads)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x, adj):
        B, N, C = x.shape
        # Multi-head attention with edge bias
        q = self.q(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Edge bias
        edge_b = self.edge_bias(adj.unsqueeze(-1))  # [B, N, N, n_heads]
        attn = attn + edge_b.permute(0, 3, 1, 2)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.norm1(x + self.proj(out))
        x = self.norm2(x + self.ff(x))
        return x


class DiscreteDenoiser(nn.Module):
    """Denoiser network for discrete adjacency diffusion."""

    def __init__(self, max_nodes, hidden_dim=128, n_layers=4, n_heads=4):
        super().__init__()
        self.node_embed = nn.Linear(max_nodes, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, n_heads)
            for _ in range(n_layers)
        ])
        self.edge_pred = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, adj_noisy, t):
        B, N, _ = adj_noisy.shape
        device = adj_noisy.device

        # Node features
        x = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        x = self.node_embed(x)

        # Add time conditioning
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, hidden]
        x = x + t_emb

        # Graph transformer layers
        for layer in self.layers:
            x = layer(x, adj_noisy)

        # Edge prediction
        ni = x.unsqueeze(2).expand(-1, -1, N, -1)
        nj = x.unsqueeze(1).expand(-1, N, -1, -1)
        edge_input = torch.cat([ni, nj, adj_noisy.unsqueeze(-1)], dim=-1)
        edge_logits = self.edge_pred(edge_input).squeeze(-1)
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        mask = 1 - torch.eye(N, device=device).unsqueeze(0)
        edge_logits = edge_logits * mask

        # Node prediction
        node_logits = self.node_pred(x).squeeze(-1)

        return edge_logits, node_logits


class GraphGenerator(nn.Module):
    """DiGress: Discrete denoising diffusion for graphs.

    Uses a discrete corruption process (edge flipping) and a graph
    transformer denoiser to predict the clean graph.
    """

    def __init__(self, max_nodes, hidden_dim=128, n_layers=4, n_heads=4,
                 n_diffusion_steps=50, lr=2e-4, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.n_steps = n_diffusion_steps

        # Beta schedule: cosine schedule for discrete diffusion
        steps = torch.arange(n_diffusion_steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos((steps / n_diffusion_steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clamp(betas, max=0.999)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alpha_bar", alpha_bar[1:].float())

        self.denoiser = DiscreteDenoiser(max_nodes, hidden_dim, n_layers, n_heads)
        self.optimizer = optim.Adam(self.denoiser.parameters(), lr=lr)

    def _corrupt(self, adj, t_idx):
        """Discrete corruption: flip edges with probability depending on t."""
        B = adj.shape[0]
        device = adj.device

        # Flip probability = 0.5 * (1 - alpha_bar_t)
        alpha_bar_t = self.alpha_bar[t_idx].view(B, 1, 1)
        flip_prob = 0.5 * (1 - alpha_bar_t)

        # Sample flip mask
        flip_mask = (torch.rand_like(adj) < flip_prob).float()
        # Make symmetric
        flip_mask = torch.triu(flip_mask, diagonal=1)
        flip_mask = flip_mask + flip_mask.transpose(1, 2)

        # Apply flips: XOR with flip mask
        adj_noisy = torch.abs(adj - flip_mask)
        return adj_noisy

    def train_step(self, adj, node_counts):
        self.train()
        self.optimizer.zero_grad()
        B = adj.shape[0]
        device = adj.device

        # Sample random timestep
        t_idx = torch.randint(0, self.n_steps, (B,), device=device)

        # Corrupt adjacency
        adj_noisy = self._corrupt(adj, t_idx)

        # Predict clean adjacency
        t_float = t_idx.float() / self.n_steps
        edge_logits, node_logits = self.denoiser(adj_noisy, t_float)

        # Cross-entropy loss to predict original clean graph
        edge_loss = F.binary_cross_entropy_with_logits(edge_logits, adj, reduction="mean")

        # Node existence loss
        node_target = (adj.sum(dim=-1) > 0).float()
        node_loss = F.binary_cross_entropy_with_logits(node_logits, node_target, reduction="mean")

        loss = edge_loss + 0.5 * node_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "edge_loss": edge_loss.item()}

    def sample(self, n_samples, device):
        """Generate graphs via iterative discrete denoising."""
        self.eval()
        N = self.max_nodes
        mask = 1 - torch.eye(N, device=device).unsqueeze(0)

        with torch.no_grad():
            # Start from random binary adjacency (Bernoulli(0.5))
            adj = (torch.rand(n_samples, N, N, device=device) > 0.5).float()
            adj = torch.triu(adj, diagonal=1)
            adj = adj + adj.transpose(1, 2)

            for step in range(self.n_steps - 1, -1, -1):
                t_float = torch.ones(n_samples, device=device) * (step / self.n_steps)
                edge_logits, node_logits = self.denoiser(adj, t_float)
                edge_probs = torch.sigmoid(edge_logits)

                if step > 0:
                    # Sample with some noise
                    adj = (torch.rand_like(edge_probs) < edge_probs).float()
                else:
                    # Final step: use threshold
                    adj = (edge_probs > 0.5).float()

                # Ensure symmetry and no self-loops
                adj = torch.triu(adj, diagonal=1)
                adj = adj + adj.transpose(1, 2)

            # Node counts from predictor
            node_probs = torch.sigmoid(node_logits)
            node_mask_pred = (node_probs > 0.5).float()
            adj = adj * node_mask_pred.unsqueeze(-1) * node_mask_pred.unsqueeze(-2)
            node_counts = node_mask_pred.sum(dim=-1).long()
            node_counts = torch.clamp(node_counts, min=2)

        return adj, node_counts
```
