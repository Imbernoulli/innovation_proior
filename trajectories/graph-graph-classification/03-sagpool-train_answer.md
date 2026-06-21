The injective-sum rung confirmed its diagnosis cleanly and showed me both what it fixed and what it left on the table. PROTEINS moved exactly where I bet, from DiffPool's flat $70.98$ to $74.54$ — the diffuse mean-like pool really was the problem, and keeping counts plus reading every depth bought three and a half points. NCI1 rose to $79.52$ *robustly* across seeds $\{78.13, 80.49, 79.93\}$, the largest set where the injective-sum-feeds-WL-power argument bites hardest. MUTAG came in at $84.02$ with its seed spread tightened from DiffPool's six-point swing. So GIN JK-Sum is genuinely solid and robust — but "solid and robust" is also its ceiling, and I can see the ceiling's shape in the numbers. The readout sums *every* node with *equal weight* at every layer. On MUTAG and PROTEINS the label is not a property of all atoms equally; it is carried by a handful of substructures — a particular ring, a mutagenic functional group — and a uniform sum *dilutes* that signal across the whole graph, drowning the few decisive nodes in the many irrelevant ones. The $84.02$ on MUTAG, where the discriminative substructure is small and local, is the tell: the readout has no notion that some nodes matter more than others.

So the next move is to make the readout *selective* — let the model decide which nodes carry the graph label and pool preferentially over them. A soft attention that re-weights every node is the safe incremental step, but it shares gin-sum's weakness in kind: every node still contributes, just unevenly, so a graph dominated by uninformative nodes can still have its signal diluted. The more ambitious correction *drops* the unimportant nodes outright and keeps a coarsened graph of only the survivors. By removing nodes it builds a true hierarchy — graph $\to$ coarser graph $\to$ coarser still — which is the structured-pooling idea DiffPool was reaching for, but realized through *selection* rather than the soft *assignment* that left DiffPool diffuse and stuck.

I propose **SAGPool**, a self-attention hierarchical top-k pooling. The mechanism has two pieces: a way to *score* each node's importance, and a way to *select and coarsen* using those scores. The score must depend on both a node's features and its position, because importance is structural — a node is decisive partly because of what it is connected to. The natural object that turns (features, adjacency) into a per-node scalar is a single graph-convolution layer that outputs *one* channel,

$$Z = \sigma\big(\mathrm{GNN}(X, A)\big),$$

one self-attention score per node, computed from the node and its neighborhood. Using a graph convolution rather than a plain MLP is the load-bearing choice: it lets the score consider both features and topology, so a node's importance is informed by who it is wired to — exactly the structural awareness DiffPool's adjacency-blind assignment MLP lacked. The score is squashed by a bounded nonlinearity so it acts as a gate. Then selection: with retain ratio $k$, keep the $\lceil k\,N\rceil$ highest-scoring nodes and discard the rest — a hard top-k. This differs sharply from both predecessors. DiffPool kept *all* nodes softly redistributed into $K$ fixed clusters; top-k keeps a *size-proportional* subset of the *original* nodes (no new cluster-nodes — the survivors are real nodes), so the coarsened graph scales with the input and there is no $N_{\max}$ padding or fixed cluster count to mis-size.

The selection is differentiable in a specific, clever way. I do not just index the survivors; I *gate* their features by their own scores, so the retained features become $X_{\text{idx}} \odot Z_{\text{idx}}$, each survivor scaled by its attention value. This is what lets gradient flow back into the scoring convolution — the loss depends on the scores through the gated features, so "this node mattered" gets a gradient even though the top-k index itself is discrete. Without the gating the selection would be a hard, gradient-free argmax and the score network would never train. After selecting, I filter edges to those with both endpoints retained, so the coarsened graph is a genuine induced subgraph on the survivors, ready to be pooled or coarsened again. That gives one coarsening level; I stack it into a hierarchy and read out at each level so the graph vector sees every scale — the same multi-scale instinct gin-sum satisfied with jumping knowledge across *layers*, realized here across *coarsening levels*. The readout runs at three scales: level 0 is the original graph, level 1 is after one top-k coarsening, level 2 after a second. At each level I summarize the (coarsened) node set with a concatenation of sum-pool and mean-pool — sum to keep the count information the injective argument says matters, mean as a scale-stable companion that does not blow up as the survivor count shrinks across levels. Each level contributes $[\Sigma, \mu]$ of width $2\cdot\text{hidden\_dim}$; three levels give $6\cdot\text{hidden\_dim}$, and a single linear projection compresses that back to $\text{hidden\_dim}$ so the fixed classifier head consumes a `hidden_dim`-wide vector.

Concretely, the editable slot is only `GraphReadout`, downstream of the fixed GIN backbone, and it does receive `edge_index` and `batch` — exactly what a top-k pool needs (the score convolution needs the adjacency; the per-graph pooling needs `batch`). So, unlike DiffPool, this method's core machinery *does* fit the surface. I instantiate two `SAGPooling(hidden_dim, ratio=0.5)` modules (halving the node count each level); in `forward` I read out level 0 on the incoming `x`, apply the first pool to get $(x_1, \text{edge\_index}_1, \text{batch}_1)$, read level 1, apply the second pool, read level 2, concatenate the three $[\Sigma, \mu]$ blocks, and project. The score convolution lives *inside* each pool and operates on the *final-layer* GIN embeddings `x` — which is fine, since the GIN embeddings already encode $K$-hop structure, so scoring them with one more conv asks "given everything message passing computed, which of these nodes is decisive."

Now I reason hard about what this should and should not do, because the gin-sum numbers themselves predict a risk. Where the label is a small, local substructure — MUTAG's mutagenic group, a discriminative PROTEINS motif — selection should *help*: dropping the irrelevant nodes concentrates the readout on the decisive ones and undoes the dilution that capped gin-sum at $84$ on MUTAG. So I expect MUTAG and PROTEINS to rise above $84.02$ and $74.54$, possibly by a lot on MUTAG. But the same hard selection should *hurt* where the decision is *distributed*. NCI1 is the worst case: 4110 compounds whose activity often depends on the whole molecular context, and a top-k at $\text{ratio}=0.5$ throws away half the nodes at every level — by level 2 only a quarter survive — and a hard, possibly mis-calibrated score (trained only through the gated features, by a single conv, with no auxiliary to guarantee it selects the *right* nodes) can discard exactly the atoms that mattered. gin-sum won NCI1 robustly precisely *because* it threw nothing away. So my honest prediction is a *split*, and it is falsifiable: SAGPool should beat gin-sum on MUTAG and PROTEINS but *regress, probably sharply and with high seed variance, on NCI1*, where the discard is irreversible. The aggregate may still edge above gin-sum on the strength of a big MUTAG jump — but it would be a *less robust* win, bought by trading NCI1 away. If that NCI1 regression is as severe as I fear, it is the failure that defines whatever comes next: a readout that gains selectivity on local motifs but loses robustness on graphs whose decision is spread across the node set, because it commits to a hard, irreversible discard.

```python
class GraphReadout(nn.Module):
    """SAGPool Hierarchical Readout (Lee et al., 2019).

    Uses self-attention scores to hierarchically select top-k nodes,
    then applies sum+mean global readout on the coarsened graph.
    Two-level hierarchy: original -> coarsened.
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        from torch_geometric.nn.pool import SAGPooling
        self.pool1 = SAGPooling(hidden_dim, ratio=0.5)
        self.pool2 = SAGPooling(hidden_dim, ratio=0.5)
        # 3 levels (original + 2 coarsened), each with sum+mean
        self.output_dim = hidden_dim * 2 * 3
        self.proj = nn.Linear(self.output_dim, hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, layer_outputs):
        # Level 0: readout on original graph
        r0 = torch.cat([global_add_pool(x, batch),
                         global_mean_pool(x, batch)], dim=-1)

        # Level 1: first coarsening
        x1, edge_index1, _, batch1, perm1, score1 = self.pool1(
            x, edge_index, batch=batch)
        r1 = torch.cat([global_add_pool(x1, batch1),
                         global_mean_pool(x1, batch1)], dim=-1)

        # Level 2: second coarsening
        x2, edge_index2, _, batch2, perm2, score2 = self.pool2(
            x1, edge_index1, batch=batch1)
        r2 = torch.cat([global_add_pool(x2, batch2),
                         global_mean_pool(x2, batch2)], dim=-1)

        return self.proj(torch.cat([r0, r1, r2], dim=-1))
```
