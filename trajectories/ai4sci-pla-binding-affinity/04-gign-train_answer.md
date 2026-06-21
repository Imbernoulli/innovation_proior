EHIGN landed where I said it would, and reading its row carefully shows the heavy machinery buys its gain in a narrow place — which is the opening for a leaner method. On 2016 it jumped to RMSE $1.2426$ / Rp $0.8218$, by far the best 2016 on the ladder, recovering the resolution EGNN lost ($1.4213$ / $0.7646$) and clearing past SchNet's $0.7767$. But the open question I flagged for 2019 resolved *against* the heavy machinery: 2019 came in at $1.4604$ / $0.6213$ — the Rp barely past EGNN's $0.6175$, and the RMSE actually *worse* than EGNN's $1.4422$. And 2013 was essentially flat versus EGNN ($1.4117$ / $0.8066$ against $1.4114$ / $0.8093$), even losing a hair of Rp. So the dual-head, attention-bias-corrected, consistency-trained apparatus delivers a big win on 2016, a wash on 2013, and a slight regression on the temporally distant 2019 — the signature of capacity spent on training-era chemistry that does not transfer. The question is now the inverse of "add more": is all that machinery *necessary*, or can a leaner geometric core hold the wins where they matter without paying the 2019 cost?

The lean answer is not "do less of EHIGN" — it is a differently shaped model. Binding is set by two kinds of interaction, covalent bonds inside each molecule and non-covalent contacts across the interface, and the point of a heterogeneous layer is to let a single atom fuse both neighbourhoods into one updated representation in a single step. EHIGN does fuse per step, but with two *different* convolution mechanisms and a readout that bets everything on the interface sum. I propose **GIGN**, a geometric interaction graph network built from a *single uniform* message-passing primitive applied to all edge sets, combined with a *two-channel readout* that uses both the interface and the whole-complex pooled representation. The intuition is that a uniform primitive has fewer ways to overfit a particular interaction regime, and a readout that does not bet everything on the interface sum has a fallback when the interface scoring transfers poorly.

The primitive is the GIN-style update, the most expressive simple message-passing form and a clean way to inject the edge. For an intra-molecular edge I project the edge feature to the node width and add it to the source, then sum over neighbours and update,

$$h_i \leftarrow \text{mlp}\Big((1 + \varepsilon)\, h_i + \sum_{j \to i}\big(h_j + \text{edge\_proj}(e_{ji})\big)\Big),$$

with a learnable scalar $\varepsilon$ per layer. The $(1+\varepsilon)\,h_i$ term keeps the centre node distinguishable from its aggregated neighbourhood — the property that makes GIN as discriminative as the Weisfeiler–Lehman test — and the MLP is $\text{Linear} \to \text{BatchNorm} \to \text{ReLU} \to \text{Linear}$. The edge projection carries the geometry: the same $17$-dim covalent features and $11$-dim contact features EHIGN used, but added into the message through a single learned projection rather than gating it. I run this same GIN layer on the ligand and pocket covalent graphs with separate weights, and a sibling inter-molecular layer on the ligand$\to$pocket contacts. The inter layer differs only in that source and destination are different node types: it projects the contact edge, adds it to the source, *mean*-aggregates over the variable contact degree (mean, not sum, for the same degree-variance reason as before), and updates the destination by $\text{mlp}([h_{\text{dst}}, \text{agg}])$ on the concatenation. Three layers, hidden width $256$, each with a residual $x \leftarrow \text{layer}(x) + x$. Crucially I apply the intra layers to both molecules and the inter layer *into the pocket* every step, so a pocket atom accumulates both its own covalent context and the ligand contacts pressing on it, fused in one step — the heterogeneous fusion, but built from one uniform primitive.

The readout is where this rung most deliberately departs from EHIGN, which bet the whole prediction on a sum of per-contact scores minus an attention bias, in two directions, with a consistency loss tying them. 2019 suggests that bet is also a *brittle* one — the interface sum is exactly the quantity most sensitive to which contacts happen to fall inside the $5$ Å cutoff, and on held-out chemistry that cutoff population shifts. So I hedge with two channels and average them. The first is the interface channel, kept in spirit but stripped of the bias correction and dual-head consistency: for each ligand$\to$pocket contact I score it from the concatenation of the final ligand-atom feature, the final pocket-atom feature, and the *raw* contact edge feature through $\text{Linear}(2H + \text{inter\_edge\_dim}, H) \to \text{ReLU} \to \text{Linear}(H, 1)$, and sum the scores over a complex's contacts via `inter_batch`. This is still "affinity as a sum over interface contacts," but a single direction and no learned offset to subtract — leaner, with fewer parameters to overfit the training-era contact population. The second is a graph channel EHIGN does not have at all: mean-pool the final ligand and pocket features over their atoms, concatenate, and regress through $\text{Linear}(2H, H) \to \text{ReLU} \to \text{Dropout}(0.1) \to \text{Linear}(H, 1)$. The graph channel is the safety net — even if the interface sum transfers poorly to a held-out complex, a pooled whole-complex representation still carries size-and-composition signal that correlates with affinity. The prediction is the *average* of the two channels, $(\text{inter\_pred} + \text{graph\_pred}) / 2$. Averaging a sharp-but-brittle interface estimate with a smooth-but-coarse graph estimate is exactly the variance-reduction move I want on the held-out split: when the interface channel is reliable the average tracks it, and when it drifts the graph channel anchors it.

Two choices are the explicit lean-vs-heavy decisions. First, no bias correction. EHIGN subtracted an attention-normalized offset to kill the size-dependence of the raw contact sum; I instead let the graph channel — itself a pooled, size-aware quantity — and the interface channel's own `Linear` absorb that offset, keeping each channel to two `Linear` layers. The bet is that the attention-bias apparatus was capacity that helped 2016 (familiar contacts) but did not transfer to 2019, so dropping it costs little on the core sets and stops paying the held-out penalty. Second, no `compute_loss` hook and no consistency term: this fill produces a single `forward` output — the average of the two channels — so the harness's default plain $\text{F.mse\_loss}(\text{pred}, \text{labels})$ is exactly right, and there is no second head to make consistent. The whole two-head, three-term EHIGN objective collapses to one prediction and one MSE. Invariance survives untouched — every input (the $35$-dim atom one-hots, the $17$-dim covalent features, the $11$-dim contact features, all built from angles, triangle areas, and distances) is rigid-motion invariant, the GIN message adds a projection of invariant edge features to invariant node features, the aggregations are sums and means, both readouts combine invariant features, and coordinates never appear.

This is the real test of "leaner generalizes better." Dropping the bias-correction and consistency machinery and adding a graph channel should *hold* the core-set quality while *helping* the splits where EHIGN's machinery did not transfer. I expect 2013 to *improve* over EHIGN — the graph channel's whole-complex signal should sharpen the small CASF-2013 set, dropping RMSE below $1.4117$ toward $\sim 1.32$ and lifting Rp above $0.8066$ toward $\sim 0.83$. On 2016 I expect to give a little back: EHIGN's heavy interface apparatus genuinely won the familiar core set ($1.2426$ / $0.8218$), and a leaner single-direction sum will likely land around $1.30$ / a touch below $0.8218$, still well ahead of EGNN and SchNet. The crux is 2019: if the lean hypothesis is right, the average-of-two-channels readout should be no worse than EHIGN there, and the graph channel may earn its keep. The decisive claim is the overall one — averaged across both metrics and the three benchmarks, the lean two-channel GIN model should edge ahead of EHIGN, winning 2013 by enough to more than offset a small 2016 give-back, with 2019 roughly a wash. If instead 2013 fails to improve and 2016 drops sharply, the lean hypothesis is wrong and EHIGN's machinery was load-bearing after all. I expect the former: this is the rung where doing less, with a second readout channel as a hedge, generalizes better than doing more.

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
