**Problem (from the DiffPool rung).** The stripped learned-clustering pool drifted to a noisy *mean*
(PROTEINS flat at 70.98, no seed spread; MUTAG six-point variance) — discarding the counts a sum keeps
and ignoring the per-layer node embeddings entirely. The readout needs (i) an *injective* reduction and
(ii) the multi-scale signal the layer stack already holds.

**Key idea.** Among permutation-invariant reductions, only the **sum** is injective on the multiset of
node features (mean keeps only the distribution; max only the support), and an injective sum readout
makes the GIN backbone as expressive as the Weisfeiler–Lehman test. Recover scale not by learning a
hierarchy but by reading the one the backbone built: **sum-pool every layer and concatenate** (jumping
knowledge). Summing learned height-k subtree embeddings is the continuous analogue of the WL subtree
kernel.

**Why it works.** Each per-layer sum is injective on its own multiset; concatenation keeps all K depths
side by side so the classifier weights them itself, instead of trusting the single deepest (often
over-smoothed) layer. No learned clustering, no fragile assignment softmax.

**Scaffold edit / hyperparameters.** Loop over `layer_outputs`; for each layer do `global_add_pool`,
then a per-layer `BatchNorm1d(hidden_dim)`, then concatenate. The per-layer graph-level BN is the one
addition beyond textbook JK-sum: it rescales each depth's pooled vector to a common scale so the very
different per-layer/per-graph-size magnitudes don't stall optimization (some folds otherwise fail to
converge) — affine and invertible, so it preserves the sum's injectivity. `output_dim =
hidden_dim × num_layers` (= 320), no projection bottleneck; `x` is ignored since it equals
`layer_outputs[-1]`.

**What to watch.** Should clear DiffPool most clearly on PROTEINS (the diffuse-mean victim) and NCI1
(where keeping counts + reading every depth matters most); MUTAG variance should tighten as the readout
is now deterministic. If PROTEINS does not move, the "DiffPool became a mean" diagnosis was wrong.

```python
class GraphReadout(nn.Module):
    """GIN JK-Sum Readout (Xu et al., 2019).

    Concatenates sum-pooled embeddings from all GIN layers
    (Jumping Knowledge). Each layer's graph embedding is batch-normalized
    before concatenation to stabilize training -- this prevents the
    different-scale representations across layers from causing
    optimization issues (some folds failing to converge).

    The output dimension is hidden_dim * num_layers, matching the
    original GIN paper's readout.
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Full concatenated dimension -- no projection bottleneck
        self.output_dim = hidden_dim * num_layers
        # Per-layer batch normalization on graph-level embeddings
        self.graph_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, batch, layer_outputs):
        # Sum-pool each layer's node embeddings independently
        graph_embs = []
        for i, h in enumerate(layer_outputs):
            g = global_add_pool(h, batch)
            g = self.graph_bns[i](g)
            graph_embs.append(g)
        # Concatenate all layers (Jumping Knowledge)
        return torch.cat(graph_embs, dim=-1)
```
