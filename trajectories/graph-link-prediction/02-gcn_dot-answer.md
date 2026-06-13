**Problem (from step 1).** VGAE separated positives on average (AUC ~87) but was soft on the ranking
metrics (Cora MRR 20.0 / Hits@20 49.3) and weak on large-pool ranking (ogbl-collab Hits@50 31.77),
with collapsing seeds — the bare inner-product decoder gives the model only one fixed way to compare
two nodes, and the sampling noise plus KL pull destabilize the small graphs.

**Key idea (GCN + MLP decoder).** Drop the variational machinery: the embedding is a deterministic
$z=\mathrm{GCN}(X,A)$ trained end-to-end by the loop's BCE. Replace the dot product with a *learned*
decoder. For each pair form the $3H$ representation
$h=[\,z_{\text{src}}\;\|\;z_{\text{dst}}\;\|\;z_{\text{src}}\odot z_{\text{dst}}\,]$ and run an MLP
$3H\to H\to H\to 1$ to one logit per edge.

**Why.** The Hadamard block $z_{\text{src}}\odot z_{\text{dst}}$ with a learned linear layer strictly
generalizes the dot product (it can learn which interaction dimensions matter), and the concatenated
$[z_{\text{src}}\,\|\,z_{\text{dst}}]$ lets the score also depend on each node's identity/degree-role —
information the inner product discards. The MLP sharpens the *top* of the candidate list, exactly where
VGAE's MRR failed. This is the canonical OGB GCN+MLP link predictor (reaches mid-fifties Hits@50 on
collaboration graphs where a bare dot product sits in the low thirties). Encoder unchanged from the
scaffold default (intermediate BatchNorm only); all new capacity is in the decoder.

**Hyperparameters.** `hidden_channels=256`, `num_layers=2`, `dropout=0.0`, MLP $3H\to H\to H\to 1$ with
ReLU+dropout, Adam `lr=0.01`, BCE, 200 epochs, early-stop patience 20.

```python
class LinkPredictor(nn.Module):
    """GCN encoder + MLP decoder with pairwise features.

    MLP decoder on [z_src, z_dst, z_src*z_dst] matches OGB's standard GCN
    link-prediction baseline; strictly stronger than pure dot product.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)
        ])

        # MLP decoder on concatenated pair features
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
               edge_index: Optional[torch.Tensor] = None,
               num_nodes: Optional[int] = None) -> torch.Tensor:
        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        h = torch.cat([z_src, z_dst, z_src * z_dst], dim=-1)
        return self.decoder(h).squeeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(edge_label_index, z,
                           edge_index=edge_index, num_nodes=x.size(0))
```
