VGAE gave me a clean read on its two deliberate weaknesses, and both cost me. It separated positives on average — AUC $86.8$ on Cora, $86.9$ on CiteSeer — but the ranking metrics bled (Cora MRR $20.0$, Hits@20 $49.3$; CiteSeer MRR $27.1$, Hits@20 $53.5$), seeds 123 and 456 collapsed into the mid-eighties on both graphs, and the clearest verdict came from the large graph: `ogbl-collab` Hits@50 of just $31.77$. An AUC of $87$ alongside an MRR of $20$ means positives are on average above negatives but the *top* of the candidate list is a mess — and on a quarter-million-node graph, where tens of thousands of non-edges must be pushed below each true edge, a single dot product of two independently-encoded points is simply too blunt. The diagnosis is sharp: the variational noise plus KL pull cost more in ranking precision and seed stability than they bought, and the bare inner product gives the model exactly one way to compare two nodes — the cosine-like alignment of their embeddings, weighting every dimension's product identically and summing.

I propose the GCN encoder with an MLP pair decoder — the canonical OGB GCN link predictor. The fix for both VGAE weaknesses is one move: drop the variational machinery and let the embedding be a deterministic point, $z=\mathrm{GCN}(X,A)$, trained end-to-end by the loop's BCE; and replace the bare dot product with a *learned* decoder that can weight and mix the pairwise interaction. Removing the sampling and the prior/inner-product fight alone is what should kill the seed collapses.

The decoder is where the gain has to come from, so let me derive it. The inner product $\sum_d z_{id}z_{jd}$ is a single fixed reduction — a uniform-weight sum — of the elementwise product $z_i\odot z_j$. The obvious generalization is to *not* sum that Hadamard product with fixed weights but feed it to an MLP and let the network learn which interaction dimensions matter and how to combine them. An MLP on $z_i\odot z_j$ already strictly generalizes the dot product — a single linear layer with all-ones weights and no bias recovers it exactly — so it can only help if trained well. But I can hand the decoder more than the Hadamard product. A pair carries two distinct kinds of information: *who each node is* (its own embedding) and *how they interact* (the elementwise product). The inner product throws away the first entirely. Yet "who each node is" matters: a high-degree hub and a leaf interact differently than two leaves, and the raw embeddings carry that. So I give the decoder the concatenation of all three,
$$h=\big[\,z_{\text{src}}\;\big\|\;z_{\text{dst}}\;\big\|\;z_{\text{src}}\odot z_{\text{dst}}\,\big],$$
a $3H$-dimensional pair representation, and put a small MLP on top: $3H\to H\to H\to 1$, with ReLU and dropout between layers, producing one logit per candidate edge. The first two blocks let the decoder condition the score on each node's identity and degree-like role; the third is the learned-weight generalization of the dot product. The MLP can now sharpen the *top* of the ranking, which is precisely where VGAE's MRR was failing.

There is a symmetry subtlety worth resolving, because the graph is undirected yet I am concatenating an ordered pair $[z_{\text{src}}\,\|\,z_{\text{dst}}]$, which is not symmetric under $i\leftrightarrow j$. The Hadamard block is symmetric; the first two are not. In practice the harness samples each undirected positive once with a fixed orientation, the graph is stored undirected so most pairs that matter appear in both message-passing directions, and the BCE target is orientation-independent, so the MLP learns to be approximately symmetric from the training distribution. I could symmetrize explicitly by averaging the two orderings, but that doubles decode cost on the large graph for a marginal gain, and the dominant signal — the Hadamard interaction — is already symmetric. So I leave the concatenation ordered and let the MLP absorb the asymmetry; this is the standard, cheaper choice.

I also revisit the encoder's normalization rule. For VGAE I argued *against* BatchNorm on the final embedding layer because the bare inner product needs the embedding magnitude. With an MLP decoder that argument weakens — the MLP's first linear layer can rescale whatever magnitude arrives — but the scaffold convention of BatchNorm on intermediate layers only, none on the last, remains the safe default: it stabilizes the depth of message passing without forcing a fixed scale on the embedding the MLP consumes. So I keep the encoder exactly as the scaffold default (intermediate BN, ReLU, dropout, no final BN) and put all the new capacity into the decoder. This keeps the parameter count modest — the MLP is $3H\cdot H+H\cdot H+H$ parameters, well inside budget — and makes the *only* change from the scaffold the swap of the one-line dot product for the three-block MLP, the cleanest possible test of whether a learned decoder fixes VGAE's ranking problem.

I expect the seed spread to tighten (no more mid-eighties AUC collapses), MRR and Hits to lift substantially — Cora MRR into the thirties, Hits@20 into the high sixties / low seventies, similar on CiteSeer — and the sharpest claim, `ogbl-collab` Hits@50 jumping from $31.77$ into the low-to-mid fifties, since this is the canonical baseline that reaches the mid-fifties on this exact graph. AUC may move little or even dip slightly, because AUC was never the failing metric; if it holds in the high eighties / low nineties while the ranking metrics jump, that confirms the decoder, not the encoder, was the bottleneck — and points the next rung at enriching what the decoder *sees* about a pair, with explicit structural features rather than more parametric capacity.

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
