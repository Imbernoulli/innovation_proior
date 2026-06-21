The scaffold ships a symmetric-normalized convolution, $\tilde{\mathbf P}=\tilde{\mathbf D}^{-1/2}\tilde{\mathbf A}\tilde{\mathbf D}^{-1/2}$ with $\tilde{\mathbf A}=\mathbf A+\mathbf I$, and that default bakes in two commitments I want to read separately before I commit to anything clever. It fixes every neighbor's weight to $1/\sqrt{\tilde d_i\tilde d_j}$ — nothing learned, nothing feature-driven — and, more subtly, it folds a node's own features into the *same* normalized pot as its neighbors through one shared linear map, so the self is just another self-looped neighbor in the average. For the floor of this ladder I do not yet attack the weighting; I attack the *coupling*. I want the most neutral aggregator on the table so I can measure what bare two-hop neighbor aggregation is worth before any cleverness about which neighbor matters.

I propose mean-aggregation message passing with separate self and neighbor projections — the inductive, full-batch form of GraphSAGE. The update is
$$\mathbf h_i'=\mathbf W_{\text{self}}\,\mathbf h_i+\mathbf W_{\text{neigh}}\,\mathrm{mean}_{j\in N(i)}\mathbf h_j,$$
followed by $\ell_2$ normalization of each node, $\mathbf h_i'\leftarrow\mathbf h_i'/\|\mathbf h_i'\|_2$. Every piece earns its place. Separating $\mathbf W_{\text{self}}$ from $\mathbf W_{\text{neigh}}$ is the load-bearing choice: it gives the node a dedicated, undiluted channel for itself that the aggregated neighborhood cannot wash out — a skip connection built into the layer's own arithmetic, two linear maps the layer can balance against each other rather than one shared map and one pot. The obvious alternative is exactly the scaffold default, which folds the self back in through a single $\mathbf W$; as that convolution stacks, a node's identity is averaged into a growing crowd two hops out under one transform, with no way for the layer to say "weight my own representation differently from the summary of my neighborhood." Keeping the previous-layer vector separate fixes precisely that.

The aggregator is the plain **mean**, $\mathrm{mean}_{j\in N(i)}\mathbf h_j$, the unweighted average over $N(i)$ with no degree normalization at all, and I choose it deliberately because it makes the *fewest* assumptions about which neighbor matters — every neighbor counts equally. A node's neighbors are a set: there is no first neighbor, no canonical order, so the combiner must be permutation-invariant, and an order-sensitive operator is simply wrong on a set. The mean is the cheapest permutation-invariant pooling, and it is the right neutral baseline because any later scheme — learned attention, principled degree damping — can be measured against it, and a gain will be attributable to the *weighting* rather than to the aggregation shape. The mean also has a clarifying relationship to the default I replace: fold the self back into the pot and run a single shared map over the combined mean, and I recover, up to a normalization constant, the inductive face of the symmetric convolution. The GCN rule *is* the mean aggregator with self folded in, so the default is a special case of this layer, and the separated-mean form is its right generalization — which tells me exactly which knob (fold versus separate) the floor is turning.

The deeper reason this "aggregate a set of features through shared maps" shape is the right primitive is that the learned object is not a table of per-node embeddings but a small set of *functions* — $\mathbf W_{\text{self}}$, $\mathbf W_{\text{neigh}}$ — applied to features and neighborhoods. A node's representation is the output of those functions on (its features, its neighbors' features), not a parameter indexed by node id, which is what makes message passing an architecture rather than a fitted lookup and is why every rung on this ladder is a fill of the same contract. The wider family this floor descends from also offers a trainable max-pool over the neighbors and a fixed-size neighbor *sampler* to bound per-batch cost on huge graphs; neither is what this task wants. The loop trains full-batch over the entire graph transductively — there are no minibatches over nodes and no hub-degree blowup to bound — so the sampler has nothing to do and I aggregate the whole neighborhood every step, and since the floor's job is the most neutral aggregator I take the mean over the max-pool.

The $\ell_2$ normalization handles a scale problem that mean aggregation plus two linear maps creates: nothing in the update pins the output magnitude, so a magnitude that grew or shrank in the first layer propagates into the second. Projecting every node onto the unit sphere after each layer keeps the second layer's inputs at a stable, comparable scale regardless of what the first layer did, which matters when the only supervision is about twenty labels per class and I cannot afford the first layer to silently rescale itself into a bad regime. It also has a clean semantic consequence — on the unit sphere, comparisons between node vectors are about direction, not magnitude, the right notion for an aggregator whose magnitudes are an artifact of how many neighbors got summed. I give $\mathbf W_{\text{self}}$ a bias and leave $\mathbf W_{\text{neigh}}$ bias-free, since one bias on the combined output suffices and a second would be redundant.

Depth is set by the task: the loop passes `num_layers=2`, and two layers is exactly right here — each layer reaches one hop, so two give every node a two-hop receptive field, the freshly transformed self at layer one and a summary of the two-hop ego network at layer two. So the floor is two separated-mean layers, ReLU and dropout 0.5 between them, $\ell_2$-normalizing after each, mapping in\_channels → 64 → out\_channels, at a cost of one full-batch sparse mean plus two dense linear maps per layer — cheaper than anything else on the ladder, with no attention vectors, no per-edge softmax, no eigendecomposition. I expect this to hold on the dense, homophilous graphs (Cora, PubMed) and to be weakest, and highest-variance, on the sparsest least-homophilous one (CiteSeer): the mean does no degree damping, so a high-degree off-topic neighbor counts exactly as much as a low-degree on-topic one, and a poorly-connected node's representation is then at the mercy of which particular neighbors it happens to have. That uniform neighbor weighting is the cap this floor is built to expose, and the fix it points to is not a different aggregation shape but a *learned* per-neighbor weight.

```python
# EDITABLE region of custom_nodecls.py — step 1: mean-aggregation message passing
class CustomMessagePassingLayer(MessagePassing):
    """Mean-aggregation layer with separate self/neighbor projections + L2 normalization."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.zeros_(self.lin_self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        neigh_agg = self.propagate(edge_index, x=x)        # mean over neighbors, no self-loop
        out = self.lin_self(x) + self.lin_neigh(neigh_agg)  # separate self / neighbor channels
        out = F.normalize(out, p=2, dim=-1)                 # project each node onto the unit sphere
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class CustomGNN(nn.Module):
    """Mean-aggregation GNN with L2 normalization."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(CustomMessagePassingLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(CustomMessagePassingLayer(hidden_channels, hidden_channels))
        self.convs.append(CustomMessagePassingLayer(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
```
