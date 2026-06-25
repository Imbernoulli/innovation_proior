**Problem (from step 2).** GCN+MLP fixed VGAE's ranking gap (Cora MRR 20.0→31.2, Hits@20 49.3→70.3;
ogbl-collab Hits@50 31.77→53.74) but Cora MRR has a violent seed spread ({39.5, 16.1, 38.0}). The
decoder only ever knows a pair through learned quantities — the Hadamard product and the two raw
embeddings — so when the GCN geometry is slightly off (a bad seed) it has no robust structural signal
to fall back on.

**Key idea (SEAL-inspired pairwise structure).** Keep SEAL's insight — *the decoder should see
structural/positional information about the pair, not just embedding alignment* — but realize it inside
the full-graph encode/decode interface. Add the absolute-difference (distance) feature alongside the
Hadamard (similarity) feature, so the decoder input is the four blocks
$h=[\,z_{\text{src}}\;\|\;z_{\text{dst}}\;\|\;z_{\text{src}}\odot z_{\text{dst}}\;\|\;
|z_{\text{src}}-z_{\text{dst}}|\,]$, fed to a $4H\to H\to H\to 1$ MLP. Normalize the embedding with
BatchNorm on *every* encoder layer (a `StructuralEncoder`), so the similarity (quadratic) and distance
(linear) blocks are on comparable scales.

**Why / what the harness omits.** This is *not* the full SEAL construction: there is no $k$-hop enclosing
subgraph extraction, no Double-Radius Node Labeling, no subgraph-level GNN with pooling — the
encode/decode contract gives one global embedding table and the raw `edge_index`, not a per-pair
subgraph machine (and per-edge subgraphs on 235k nodes would blow the time/parameter budget). The
structural information is approximated by the extra distance feature and normalized embeddings. The
difference block gives a structural signal robust to imperfect embedding alignment, which is what should
damp the Cora seed variance; full-layer BN stabilizes training; both are licensed because the decoder
is an MLP, not a magnitude-sensitive inner product.

**Hyperparameters.** `hidden_channels=256`, `num_layers=2`, `dropout=0.0`, decoder $4H\to H\to H\to 1$
with ReLU+dropout, BatchNorm on all encoder layers, Adam `lr=0.01`, BCE, 200 epochs, patience 20.

```python
class StructuralEncoder(nn.Module):
    """GCN encoder augmented with structural node features."""
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class LinkPredictor(nn.Module):
    """SEAL-inspired link predictor.

    Uses GCN encoder + pairwise MLP decoder with structural features
    (product, difference, L2 distance) that approximate SEAL's subgraph
    information without the expensive subgraph extraction.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.encoder = StructuralEncoder(in_channels, hidden_channels,
                                          num_layers, dropout)
        # SEAL-style pairwise features: concat, hadamard, L1, L2
        # Input: z_src || z_dst || z_src*z_dst || |z_src-z_dst|
        dec_in = hidden_channels * 4
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
               edge_index: Optional[torch.Tensor] = None,
               num_nodes: Optional[int] = None) -> torch.Tensor:
        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        h = torch.cat([
            z_src, z_dst,
            z_src * z_dst,
            torch.abs(z_src - z_dst),
        ], dim=-1)
        return self.decoder(h).squeeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(edge_label_index, z,
                           edge_index=edge_index, num_nodes=x.size(0))
```
