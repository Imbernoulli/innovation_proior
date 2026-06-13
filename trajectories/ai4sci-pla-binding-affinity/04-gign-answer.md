**Problem.** EHIGN's heavy machinery (dual heads, attention bias correction, consistency loss) won 2016
big but was a wash on 2013 and slightly *worse* than EGNN on the 2019 temporal holdout — capacity spent
on training-era chemistry that did not transfer. The question is whether a leaner core generalizes
better.

**Key idea.** Replace the two bespoke EHIGN convolutions with one uniform GIN-style primitive applied to
all three edge sets: `mlp((1 + ε) · x + Σ msg)` with `msg = x_src + edge_proj(edge_attr)`, run on the
ligand and pocket covalent graphs and (mean-aggregated) on the ligand→pocket contacts, three residual
layers. Read out through *two channels* averaged: (1) an interface channel — score each contact from
`[lig_h[src], poc_h[dst], raw_edge_attr]`, summed over contacts — kept but stripped of EHIGN's bias
correction and dual-head consistency; (2) a graph channel — mean-pool ligand and pocket features,
concat, regress. Prediction is `(inter_pred + graph_pred) / 2`.

**Why leaner.** A uniform primitive has fewer ways to overfit a single interaction regime. Averaging a
sharp-but-brittle interface sum (most sensitive to which contacts fall in the 5 Å cutoff, which shifts on
held-out chemistry) with a smooth-but-coarse pooled graph estimate is variance reduction on the
held-out split: the graph channel anchors the prediction when the interface channel drifts. No bias
correction (the graph channel and the readout `Linear`s absorb the size offset); no `compute_loss` hook
(single `forward` output → the harness's default plain MSE is exactly right).

**Expectations vs EHIGN.** Improve 2013 (graph channel sharpens the small core set: RMSE below 1.4117
toward ~1.32, Rp above 0.8066 toward ~0.83); give a little back on 2016 (single-direction interface sum
vs EHIGN's heavy apparatus: ~1.30 / a touch below 0.8218, still ahead of EGNN/SchNet); 2019 roughly a
wash. The decisive claim is overall: averaged across both metrics and three benchmarks, the lean
two-channel model edges ahead of EHIGN.

**Hyperparameters.** `H=256`, 3 GIN layers (learnable `ε` per layer), mean aggregation on inter edges,
two-channel readout each ending in `Linear(H, 1)`, dropout 0.1 in the graph head. Trained by the fixed
harness (Adam `lr=1e-4`, `wd=1e-6`, batch 128, ≤800 epochs, patience 50) with plain MSE.

```python
# EDITABLE SECTION START — GIGN: Geometric Interaction Graph Network

class GINLayer(nn.Module):
    """GIN convolution with edge features."""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        msg = x[src] + self.edge_proj(edge_attr)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        return self.mlp((1 + self.eps) * x + agg)


class InterGINLayer(nn.Module):
    """GIN convolution for inter-molecular edges."""
    def __init__(self, src_dim, dst_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, src_dim)
        self.mlp = nn.Sequential(
            nn.Linear(src_dim + dst_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_src, x_dst, edge_index, edge_attr, num_dst):
        src, dst = edge_index
        msg = x_src[src] + self.edge_proj(edge_attr)
        agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
        count = torch.zeros(num_dst, 1, device=msg.device)
        agg.index_add_(0, dst, msg)
        count.index_add_(0, dst, torch.ones(src.size(0), 1, device=msg.device))
        agg = agg / count.clamp(min=1)
        return self.mlp(torch.cat([x_dst, agg], dim=-1))


class AffinityModel(nn.Module):
    """GIGN: Geometric Interaction Graph Network.

    Uses GIN-style message passing for both intra- and inter-molecular graphs.
    Readout via interaction-weighted sum over inter-molecular edges.
    """
    def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
        super().__init__()
        H = 256
        num_layers = 3

        self.lig_embed = nn.Linear(lig_dim, H)
        self.poc_embed = nn.Linear(poc_dim, H)

        self.lig_convs = nn.ModuleList([GINLayer(H, intra_edge_dim, H) for _ in range(num_layers)])
        self.poc_convs = nn.ModuleList([GINLayer(H, intra_edge_dim, H) for _ in range(num_layers)])
        self.inter_convs = nn.ModuleList([InterGINLayer(H, H, inter_edge_dim, H) for _ in range(num_layers)])

        # Interaction readout
        self.edge_readout = nn.Sequential(
            nn.Linear(H * 2 + inter_edge_dim, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )

        # Graph-level readout
        self.graph_readout = nn.Sequential(
            nn.Linear(H * 2, H),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(H, 1),
        )

    def forward(self, batch: PLABatch) -> torch.Tensor:
        B = batch.labels.size(0)
        lig_h = self.lig_embed(batch.lig_x)
        poc_h = self.poc_embed(batch.poc_x)

        for i in range(len(self.lig_convs)):
            lig_h = self.lig_convs[i](lig_h, batch.lig_edge_index, batch.lig_edge_attr) + lig_h
            poc_h = self.poc_convs[i](poc_h, batch.poc_edge_index, batch.poc_edge_attr) + poc_h
            if batch.l2p_edge_index.size(1) > 0:
                poc_h = self.inter_convs[i](lig_h, poc_h, batch.l2p_edge_index, batch.l2p_edge_attr, poc_h.size(0))

        # Interaction-level scoring
        if batch.l2p_edge_index.size(1) > 0:
            l2p_src, l2p_dst = batch.l2p_edge_index
            inter_feat = torch.cat([lig_h[l2p_src], poc_h[l2p_dst], batch.l2p_edge_attr], dim=-1)
            inter_scores = self.edge_readout(inter_feat)
            inter_pred = torch.zeros(B, 1, device=inter_scores.device)
            inter_pred.index_add_(0, batch.inter_batch, inter_scores)
        else:
            inter_pred = torch.zeros(B, 1, device=lig_h.device)

        # Graph-level prediction
        lig_pool = scatter_mean(lig_h, batch.lig_batch, B)
        poc_pool = scatter_mean(poc_h, batch.poc_batch, B)
        graph_pred = self.graph_readout(torch.cat([lig_pool, poc_pool], dim=-1))

        pred = (inter_pred + graph_pred) / 2
        return pred.squeeze(-1)

# EDITABLE SECTION END
```
