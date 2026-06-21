SEAL landed almost exactly where I predicted, and the precise shape of its result tells me the finale. The full-layer normalization and the distance feature stabilized Cora's AUC (seeds $92.2/92.3/93.0$, mean $92.5$ — tighter and higher than gcn_dot's $91.3$) and pushed `ogbl-collab` Hits@50 to $57.88$, past gcn_dot's $53.74$. But the small-graph *ranking* metrics did not improve — they regressed: Cora MRR fell $31.18\!\to\!27.49$, Hits@20 $70.27\!\to\!61.16$; CiteSeer MRR $40.84\!\to\!35.22$. SEAL bought AUC and large-graph Hits by trading away top-of-list precision on the small graphs, and the reason is the one I flagged: SEAL's "structural" signal is the absolute difference $|z_{\text{src}}-z_{\text{dst}}|$, a function of the *learned embeddings*, not of the graph. Asked to discriminate the hardest pairs at the top of the MRR list, it had no measurement of the quantity that actually decides links on citation and collaboration graphs — how many neighbors $u$ and $v$ literally share.

This is a genuine expressiveness ceiling, not a tuning issue, and naming it precisely matters. A message-passing GNN produces one embedding per node, so two structurally-identical (automorphic) destinations get the *same* embedding and therefore the *same* score against a fixed source — even when one is a true neighbor and the other is not. Worse, a plain GNN cannot count triangles, and a triangle through $u, v, w$ is exactly a common neighbor $w$. So the common-neighbor / Adamic–Adar / resource-allocation family — the signal that has dominated link prediction on these graphs for two decades — is precisely what every rung so far has been *structurally unable to represent*: VGAE, gcn_dot, and SEAL all score a pair through the geometry of two independently-encoded points, and none ever counts a shared neighbor. So I am not going to add another *learned* pairwise feature. I am going to compute the structural signal *exactly*, against the live adjacency, and hand it to the decoder.

I propose BUDDY: score a pair by a *learned function of explicit neighborhood-overlap features* fused with the GNN embeddings. For each candidate pair $(u,v)$ I compute the three canonical overlap heuristics directly from the adjacency,
$$\mathrm{CN}(u,v)=|N(u)\cap N(v)|,\quad \mathrm{AA}(u,v)=\!\!\sum_{w\in N(u)\cap N(v)}\!\!\frac{1}{\log\deg(w)},\quad \mathrm{RA}(u,v)=\!\!\sum_{w\in N(u)\cap N(v)}\!\!\frac{1}{\deg(w)}.$$
CN is the raw triangle count — the $A[1,1]$ distance-label count, the backbone signal a GNN cannot produce. AA and RA are degree-discounted variants (Adamic–Adar 2003; resource allocation): a shared neighbor that is a high-degree hub cited by everyone is weak evidence of a link, so AA down-weights it by $1/\log\deg$ and RA by $1/\deg$. Handing all three to the decoder lets the MLP learn *which* discounting the data prefers rather than committing to one formula — the learned-heuristic move. All three are pair-relative: they depend on the joint neighborhood geometry of $u$ and $v$, not either node alone, so they separate exactly the automorphic-node links the embedding-only decoders could not.

Now the realization within this contract, and I have to be honest about what the interface lets me build versus the fullest version of the idea. The fullest version computes, for each pair, the entire distance-label count table $A[d_u,d_v]$ over a $k$-hop window — $A[1,1]$ is CN, $A[1,2]/A[2,1]/A[2,2]$ the multi-hop overlaps — and, because computing those intersection cardinalities exactly over huge neighborhoods is expensive, estimates them with set sketches: MinHash for the Jaccard (intersection shape) and HyperLogLog for the cardinality (union size), propagated node-wise by elementwise min/max so per-edge cost is independent of graph size. That sketch machinery is what makes the idea scale to millions of nodes. But this scaffold hands me `decode(edge_label_index, z, edge_index)` — the *original* node indices and the *live* adjacency — on graphs of at most ${\sim}236$k nodes, with the parameter budget checked at startup. At this scale I do not need the sketches: I can compute CN/AA/RA *exactly* with a sparse-matrix routine. Build the CSR adjacency from `edge_index`, slice the rows for the batch's sources and destinations, take the elementwise product of those sparse rows to get the per-pair common-neighbor indicator, then sum it (CN), sum it weighted by $1/\log\deg$ (AA), and by $1/\deg$ (RA). The whole thing stays sparse — no dense $N\times N$ materialization — so it is memory-feasible on `ogbl-collab`. The deliberate substitution is therefore *exact* CN/AA/RA via scipy sparse in place of *sketched* multi-distance counts: I keep the load-bearing insight (explicit, pair-relative overlap counts as direct decoder inputs) and drop the sketching and the multi-hop count table the harness neither needs at this scale nor exposes a clean hook for. The harness already passes the correct adjacency at each phase — train-only during validation, train+val at test, exactly as the OGB protocol prescribes — so the counts are computed against the right graph automatically.

The fusion is the other design point. The fullest version combines the structural counts with the Hadamard product of the propagated node features, edge-pooling style. The cleaner thing here, given I already have a good GNN encoder, is to project the three overlap counts up to the hidden width with a linear layer (so they enter the MLP on the same scale as the embeddings) and *concatenate* them with the two raw node embeddings,
$$h=\big[\,z_{\text{src}}\;\big\|\;z_{\text{dst}}\;\big\|\;\mathrm{proj}(\mathrm{CN},\mathrm{AA},\mathrm{RA})\,\big],$$
then an MLP $3H\to H\to H\to 1$. Concatenation rather than Hadamard is deliberate and harness-matched: the structural counts are a *different kind* of quantity than the embeddings — integer-ish overlap magnitudes, not learned coordinates — so giving the MLP all of $z_{\text{src}}$, $z_{\text{dst}}$, and a learned embedding of the counts lets it model the interaction between "who the nodes are" and "how much they overlap" without forcing the counts through a product with the embeddings. The overlap computation runs under `no_grad` — these are fixed structural measurements, not learned features, so no gradient need flow through the sparse adjacency ops — while the projection and MLP are trained end-to-end. The encoder stays the GCN stack with BatchNorm on all layers (as in SEAL), since the embeddings now sit beside projected counts and benefit from the same scale control. The encoder also caches the encode-time `edge_index` and node count so `decode` has a sensible default adjacency when the loop does not pass one explicitly.

The bar is the strongest baseline, SEAL — Cora ($92.5/27.5/61.2$), CiteSeer ($92.9/35.2/72.1$), `ogbl-collab` Hits@50 $57.88$ — and the falsifiable claims follow from the diagnosis. First and sharpest: the small-graph ranking metrics SEAL regressed should recover and *exceed*, because the explicit overlap counts are exactly the top-of-list signal SEAL lacked — I expect Cora MRR and Hits@20 back above gcn_dot's $31.2/70.3$, clear of SEAL's $27.5/61.2$, and CiteSeer likewise. Second: `ogbl-collab` Hits@50 should pass SEAL's $57.88$ into the low-to-mid sixties, because CN/AA/RA are *the* dominant signal on a dense collaboration graph and SEAL only approximated them — the cleanest single-number verdict. Third, the consistency check: AUC should hold in the low-to-mid nineties, and if it holds *while* MRR/Hits@20/Hits@50 all rise, the trajectory's thesis is confirmed — the ceiling on every embedding-only rung was the inability to count shared neighbors, and handing the decoder the exact overlap removes it. The way this could fail: if the exact counts dominate the MLP so strongly it ignores the node features, the model could collapse toward a pure heuristic and lose the feature-driven gains on the feature-rich citation graphs — I would watch for AUC *dropping* on Cora/CiteSeer as the warning that the fusion needs the structural features down-weighted relative to the embeddings.

```python
class StructuralFeatureComputer:
    """Precomputes structural pairwise features (approximating BUDDY sketches)."""

    @staticmethod
    @torch.no_grad()
    def compute_cn_features(edge_index, num_nodes, edge_label_index):
        """Compute CN/AA/RA features using scipy sparse (memory-efficient)."""
        import scipy.sparse as sp
        device = edge_label_index.device

        row = edge_index[0].cpu().numpy()
        col = edge_index[1].cpu().numpy()
        adj = sp.csr_matrix((np.ones(len(row)), (row, col)),
                            shape=(num_nodes, num_nodes))

        src = edge_label_index[0].cpu().numpy()
        dst = edge_label_index[1].cpu().numpy()

        # Sparse row extraction + element-wise multiply stays sparse
        src_rows = adj[src]   # [batch, N] sparse
        dst_rows = adj[dst]   # [batch, N] sparse
        common = src_rows.multiply(dst_rows)  # sparse intersection

        deg = np.asarray(adj.sum(axis=1)).flatten().clip(min=1)
        cn = np.asarray(common.sum(axis=1)).flatten()
        aa = np.asarray(common.multiply(1.0 / np.log(deg).clip(min=1.0))
                        .sum(axis=1)).flatten()
        ra = np.asarray(common.multiply(1.0 / deg).sum(axis=1)).flatten()

        return torch.tensor(np.stack([cn, aa, ra], axis=1),
                            dtype=torch.float32, device=device)


class LinkPredictor(nn.Module):
    """BUDDY-inspired link predictor.

    Combines GCN node embeddings with precomputed structural features
    (common neighbors, Adamic-Adar, resource allocation) via an MLP decoder.
    This approximates BUDDY's subgraph sketching approach.

    The new decode interface takes `edge_label_index` (original node
    indices) and the full embedding table `z` directly, so we no longer
    need to recover indices via hashing/argmax.  The training graph
    `edge_index` is also passed through, enabling exact CN/AA/RA
    computation against whichever adjacency is in use (train-only during
    validation, train+val during final test, as OGB prescribes).
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # GCN encoder
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])

        # Structural feature dimension: CN, AA, RA = 3
        struct_dim = 3
        self.struct_proj = nn.Linear(struct_dim, hidden_channels)

        # MLP decoder: node features + structural features
        dec_in = hidden_channels * 2 + hidden_channels  # src, dst, struct
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        # Cached context set at encode-time so decode() has sensible
        # defaults when the caller does not pass edge_index explicitly.
        self._edge_index = None
        self._num_nodes = None

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self._edge_index = edge_index
        self._num_nodes = x.size(0)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
               edge_index: Optional[torch.Tensor] = None,
               num_nodes: Optional[int] = None) -> torch.Tensor:
        # Resolve the adjacency to use for structural features.
        ei = edge_index if edge_index is not None else self._edge_index
        N = num_nodes if num_nodes is not None else (
            self._num_nodes if self._num_nodes is not None else z.size(0))

        with torch.no_grad():
            struct_feats = StructuralFeatureComputer.compute_cn_features(
                ei, N, edge_label_index)
        struct_h = self.struct_proj(struct_feats.float())

        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        h = torch.cat([z_src, z_dst, struct_h], dim=-1)
        return self.decoder(h).squeeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(edge_label_index, z,
                           edge_index=edge_index, num_nodes=x.size(0))
```
