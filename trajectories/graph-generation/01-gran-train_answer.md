The starting point is the harness's default fill, a plain-MLP VAE that flattens each adjacency, encodes it to a single Gaussian latent, and decodes the whole matrix back through an MLP trained on per-entry reconstruction. That floor is order-blind in the worst way: it compares the decoded adjacency to the target *entrywise*, so a correct graph emitted under a different node permutation is punished hard, and the latent is forced to memorize orderings instead of structure. The one specific limitation I need to fix is that every edge is read off a flat bottleneck vector with no sense of the topology around it. I want a generator where each edge decision $(i,j)$ is a function of *node representations that have absorbed the surrounding structure* — topology-aware edge prediction — without dragging in the imposed generation order and the $O(N^2)$ sequential unroll that the autoregressive lineage (DeepGMG, Li et al. 2018; GraphRNN, You et al. 2018) would force into `train_step`.

I propose **GRAN**, a one-shot attentive refinement model. The heretical move for a method that descends from the autoregressive family is to keep only the part that mattered — the *topology-aware edge predictor* built from message passing — and to drop the sequence entirely. DeepGMG was good because every edge saw the real graph through a GNN pass; it was hopeless here because it ran that pass once per atomic decision at a cost on the order of $m\cdot n^2\cdot\mathrm{diam}(G)$ and serialized the decisions. So I keep the message passing and predict the *entire* adjacency in one shot from refined node states, then refine that prediction a fixed handful of times. No imposed ordering, no recurrent rollout, and the output is exactly the full symmetric adjacency the `sample` contract asks for.

The refinement core is where the topology-awareness lives. The state is a node-feature tensor $[B,N,D]$ and an edge-feature tensor $[B,N,N,1]$ initialized from the current adjacency. One refinement block does three things in sequence. First, **multi-head node attention**: each node's representation is updated by a learned weighting of all other nodes, $\mathrm{attn} = \mathrm{softmax}(QK^\top/\sqrt{d})$, $\mathrm{out} = \mathrm{attn}\cdot V$, with a residual and LayerNorm. This is the message-passing step. I use attention rather than a fixed degree-normalized GCN aggregation precisely because the edges I condition on are themselves uncertain — during sampling they begin as guesses — so I want the node update to *learn* which neighbors to trust rather than average them uniformly. Second, an **edge update**: for every pair $(i,j)$ I form $[n_i, n_j, e_{ij}]$, push it through an MLP, and add the result back to the edge feature, so each edge representation accumulates evidence from both endpoints. Third, an **edge-to-node aggregation**: I sum each node's incident edge features (masked to real nodes when a mask is available), concatenate with the node state, run an MLP, and apply a residual and norm — so the node state also hears from its edges, not only from attention. Stacking three of these blocks and treating the stack as the refinement gives me, in one forward pass, node states that have integrated several rounds of structure.

From the refined node states I emit two predictions. The **edge logits**: for each pair I again form $[n_i, n_j, e_{ij}]$, an MLP to a scalar, then symmetrize $(L + L^\top)/2$ and zero the diagonal — the two constraints the loop requires of `sample`'s output, baked into the predictor rather than hoped for. The **node-existence logits**: an MLP per node to a scalar, which lets the fixed-$N$ adjacency declare a slot empty so a graph on $n < N$ nodes can come out. Training is then almost embarrassingly direct compared to the autoregressive likelihoods: there is no ordering to marginalize and no sequence to unroll, so I teacher-force the *true* adjacency through the refinement and ask the edge logits to reconstruct it. The edge loss is per-entry binary cross-entropy of the edge logits against the target adjacency; the node loss is BCE of the node-existence logits against the true node mask $\mathrm{adj}.\mathrm{sum}(-1) > 0$; the total is $\mathrm{edge\_loss} + 0.5\cdot\mathrm{node\_loss}$, with the node term down-weighted because there are $O(N)$ node terms against $O(N^2)$ edge terms and I do not want node existence to dominate. A global grad-norm clip at $1.0$ keeps the attention stack from taking a destructive early step. The entire `train_step` is one forward pass on the real adjacency, one reconstruction loss, one Adam step — no sequential inner loop, which is exactly the cost I bought back from the autoregressive route.

Sampling needs the most care, because a one-shot refinement model has no ground-truth prefix to lean on; it must hallucinate a graph from nothing. If I started the refinement from an all-zeros adjacency the model would collapse: the edge features are all zero, attention has nothing structural to weight, the edge-to-node aggregation sums to zero, every node looks identical, the node-existence predictor sees uniform empty states and predicts every node *absent*, and the output is the empty graph. So I cannot start from zeros. Instead I start from a **random sparse adjacency** — each upper-triangular entry drawn $\mathrm{Bernoulli}(p_{\mathrm{init}})$ with $p_{\mathrm{init}}=0.3$, then symmetrized — so there is real edge signal for the attention and the edge MLPs to chew on. Then I refine: at each of `n_refine_steps` $=5$, run the forward pass, take $\mathrm{edge\_probs} = \sigma(\mathrm{edge\_logits})$, and *resample* the adjacency as $\mathrm{Bernoulli}(\mathrm{edge\_probs})$, symmetrizing each time. This fixed-point-style sweep pulls the random initial graph toward something the edge predictor is confident about. After the last sweep I read node counts off the realized connectivity, $(\mathrm{adj}.\mathrm{sum}(-1) > 0).\mathrm{sum}(-1)$ clamped to $\ge 2$, rather than off the node-existence head, because that head was trained on *real* adjacency rows and is unreliable when fed the random-then-refined adjacency at inference — so I trust the structure the refinement actually produced over the auxiliary head's guess.

I want to be honest about the boundaries of this construction. It is *not* a block-wise autoregressive generator with a per-step GNN, mixture-of-Bernoulli edge outputs, and a family of canonical orderings — that machinery is the principled way to get *correlated* edges (a shared mixture latent couples the edges in a block) and a tractable permutation-aware likelihood (a logsumexp over orderings). This fill keeps none of it: edges are predicted as **independent** Bernoullis given the node states, there is **no ordering and no autoregression**, the training signal is plain reconstruction BCE, and "refinement" means resampling the full adjacency five times at sampling. I keep only the *attention-based message-passing edge predictor*, re-expressed as a one-shot refiner that fits the contract. I should expect to pay for exactly what I dropped: independent edges cannot represent "both these edges fire or neither," which is precisely the correlation that makes a two-community graph or a clustered ego-graph look right, and the entrywise reconstruction still carries the order-sensitivity the default VAE had. I expect this to do genuinely well on `community_small`, where dense local block structure is easy for an attention edge predictor, but to be a high-variance, seed-fragile generator on the larger, sparser `ego_small` and `enzymes`, where the random-init-plus-five-sweeps sampler can stall on an over-dense blob and spike every MMD at once — a model that *can* produce good graphs but does not do so reliably.

```python
# EDITABLE region of custom_graphgen.py (lines 446-590) — step 1: GRAN (one-shot attentive refinement)
class AttentionBlock(nn.Module):
    """Multi-head attention for graph nodes."""

    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.norm(x + self.proj(out))


class GRANBlock(nn.Module):
    """GRAN message passing block with attention and edge prediction."""

    def __init__(self, node_dim, edge_dim=1, n_heads=4):
        super().__init__()
        self.attn = AttentionBlock(node_dim, n_heads)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, edge_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_feat, edge_feat, mask=None):
        B, N, D = node_feat.shape
        # Attention-based node update
        node_feat = self.attn(node_feat, mask)

        # Edge update
        ni = node_feat.unsqueeze(2).expand(-1, -1, N, -1)
        nj = node_feat.unsqueeze(1).expand(-1, N, -1, -1)
        edge_input = torch.cat([ni, nj, edge_feat], dim=-1)
        edge_feat = edge_feat + self.edge_mlp(edge_input)

        # Aggregate edge info to nodes
        if mask is not None:
            edge_agg = (edge_feat * mask.unsqueeze(-1).unsqueeze(-1).float()).sum(dim=2)
        else:
            edge_agg = edge_feat.mean(dim=2)
        node_input = torch.cat([node_feat, edge_agg], dim=-1)
        node_feat = self.norm(node_feat + self.node_mlp(node_input))

        return node_feat, edge_feat


class GraphGenerator(nn.Module):
    """GRAN: Graph Recurrent Attention Network.

    Iteratively refines node and edge representations using attention-based
    message passing, then predicts edge probabilities for graph generation.
    """

    def __init__(self, max_nodes, hidden_dim=128, n_layers=3, n_heads=4,
                 n_refine_steps=5, lr=1e-3, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.n_refine_steps = n_refine_steps

        # Node embedding
        self.node_embed = nn.Linear(max_nodes, hidden_dim)

        # GRAN blocks (shared across refinement steps)
        self.blocks = nn.ModuleList([
            GRANBlock(hidden_dim, edge_dim=1, n_heads=n_heads)
            for _ in range(n_layers)
        ])

        # Final edge prediction
        self.edge_pred = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Node existence prediction
        self.node_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _forward(self, adj, node_mask=None):
        B, N, _ = adj.shape
        device = adj.device

        # Initial node features (identity-based)
        x = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        node_feat = F.relu(self.node_embed(x))  # [B, N, hidden]

        # Initial edge features from adjacency
        edge_feat = adj.unsqueeze(-1)  # [B, N, N, 1]

        # Iterative refinement
        for block in self.blocks:
            node_feat, edge_feat = block(node_feat, edge_feat, node_mask)

        # Predict edges
        ni = node_feat.unsqueeze(2).expand(-1, -1, N, -1)
        nj = node_feat.unsqueeze(1).expand(-1, N, -1, -1)
        edge_input = torch.cat([ni, nj, edge_feat], dim=-1)
        edge_logits = self.edge_pred(edge_input).squeeze(-1)  # [B, N, N]

        # Symmetrize and remove self-loops
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        diag_mask = 1 - torch.eye(N, device=device).unsqueeze(0)
        edge_logits = edge_logits * diag_mask

        # Node existence
        node_logits = self.node_pred(node_feat).squeeze(-1)  # [B, N]

        return edge_logits, node_logits

    def train_step(self, adj, node_counts):
        self.train()
        self.optimizer.zero_grad()

        edge_logits, node_logits = self._forward(adj)

        # Edge loss
        edge_loss = F.binary_cross_entropy_with_logits(edge_logits, adj, reduction="mean")

        # Node existence loss
        node_target = (adj.sum(dim=-1) > 0).float()
        node_loss = F.binary_cross_entropy_with_logits(node_logits, node_target, reduction="mean")

        loss = edge_loss + 0.5 * node_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "edge_loss": edge_loss.item()}

    def sample(self, n_samples, device):
        self.eval()
        with torch.no_grad():
            # Start from random sparse adjacency (not zeros) to provide edge signal.
            # Starting from zeros leads to empty output: node aggregation is zero ->
            # node predictor predicts all nodes absent -> empty graph.
            p_init = 0.3
            adj = (torch.rand(n_samples, self.max_nodes, self.max_nodes, device=device) < p_init).float()
            adj = torch.triu(adj, diagonal=1)
            adj = adj + adj.transpose(1, 2)

            for step in range(self.n_refine_steps):
                edge_logits, node_logits = self._forward(adj)
                edge_probs = torch.sigmoid(edge_logits)
                adj = (torch.rand_like(edge_probs) < edge_probs).float()
                adj = torch.triu(adj, diagonal=1)
                adj = adj + adj.transpose(1, 2)

            # Derive node counts from connectivity (node predictor is unreliable
            # when initialized from random adjacency at inference time).
            node_counts = (adj.sum(dim=-1) > 0).long().sum(dim=-1)
            node_counts = torch.clamp(node_counts, min=2)

        return adj, node_counts
```
