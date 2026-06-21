The task hands me a frozen pipeline and asks me to design exactly one thing: the map from a molecule's atom/bond graph to a single vector that a small head turns into per-task logits, scored on a *scaffold* split where train and test share no Murcko scaffold. Before reaching for anything clever I want the simplest learned graph encoder whose representational power I can actually reason about from first principles — an honest floor — and then watch where it breaks, because where it breaks tells the next rung what to fix. The graph net family is small once stripped to its skeleton: each atom starts at $h_v^{(0)} = x_v$, and for $k$ rounds it aggregates a bag of neighbors and combines that with itself, $a_v^{(k)} = \mathrm{AGGREGATE}(\{h_u^{(k-1)} : u\in N(v)\})$ and $h_v^{(k)} = \mathrm{COMBINE}(h_v^{(k-1)}, a_v^{(k)})$, then a permutation-invariant readout collapses the atoms into one graph vector. GCN is this with a mean aggregator, GraphSAGE-pool with an element-wise max; the only things that vary are the squash over neighbors and the transform around it. The real question is therefore: when are two molecules *forced* to the same vector, because that collapse is the ceiling on what any such encoder can distinguish — and on a scaffold split, distinguishing genuinely novel structure is the entire game.

I propose the starter **GIN** — a Graph Isomorphism Network with an edge-aware sum message and a deliberately lossy mean readout — and the way to see why it is built this way is to measure the whole family against the Weisfeiler-Lehman isomorphism test. Trace what $h_v^{(k)}$ captures: round one an atom sees its bonded neighbors, round two each neighbor has already absorbed *its* neighbors, so after $k$ rounds $h_v^{(k)}$ summarizes the rooted subtree of height $k$ hanging off $v$. That is exactly WL color-refinement — relabel a node by hashing (its own label, the multiset of neighbor labels), iterate — and WL's hash is *injective*, so it never throws away a distinction once found. A graph net runs everything through continuous, lossy functions (a mean, a max, a linear layer) that can squash two different inputs together. The bound that follows is two-sided and worth stating exactly: if WL cannot separate two graphs they share the identical multiset of (label, neighbor-multiset) pairs at every round, and by induction equal WL labels force equal encoder features, so a permutation-invariant readout returns the same value — **no message-passing GNN beats WL**. And the encoder *reaches* WL precisely when neighbor aggregation, combine, and readout are all injective on multisets, because then the encoder's features are a faithful recoding of WL's labels. Injectivity on multisets is the whole game.

Which aggregator is injective? Over bounded multisets, **sum** can be made injective — give each element a code and sum, and the total is a positional encoding of the multiplicity profile, so distinct multisets give distinct sums. **Mean** keeps only the *distribution*: scale every multiplicity by $k$ and the mean is unchanged, so $\{green,red\}$ and $\{green,green,red,red\}$ collapse. **Max** keeps only the *support*: a second red is invisible. The discriminative ranking is strict, $\text{sum} \sqsupset \text{mean} \sqsupset \text{max}$. The transform around the sum must be a genuine MLP, not a single linear+ReLU — with all-nonnegative inputs a bias-free linear+ReLU degenerates into "sum then one linear map," which cannot separate $\{1,1,1,1,1\}$ from $\{2,3\}$. And if I fold the center atom into the neighbor bag I lose which element was the root, so I tag it: $(1+\varepsilon)\,h_v + \sum_u h_u$, injective for irrational $\varepsilon$ and a learnable scalar in practice. The maximally expressive fill is therefore: sum the neighbors, add $(1+\varepsilon)$ times the center, push through an MLP, and read out by summing.

The theory points hard at sum pooling and a sum readout, but I am building the scaffold's *starter* fill, and it departs from the theoretical maximum in three named ways that I keep on purpose — the point of rung one is a deliberately weak, honest floor. First, the per-layer message folds in the **bond features**: $\text{msg} = h_v[\text{src}] + \mathrm{edge\_proj}(e_{vw})$, summed into the destination. The clean WL-maximal GIN ignores edges, but single/double/aromatic, conjugated, and in-ring bonds are chemically load-bearing, so the message is edge-aware — an additive bond bias before the sum. Second, each `GINConv` keeps the injective recipe faithfully: a sum over neighbors, the $(1+\varepsilon)$ center tag, an MLP with a BatchNorm inside, so each layer is $\mathrm{MLP}\big((1+\varepsilon)\,x + \sum_{N(v)}(x[u]+\text{edge})\big)$. Third — and this is the load-bearing weakness — the *graph readout is **mean** pooling*, $\sum_v h_v / |V|$, off the **last** layer only, no sum and no jumping-knowledge concatenation across depths. By the ranking above, mean readout throws away exactly the multiplicity information sum keeps. The four layers are wrapped in a residual stack, $x \leftarrow x + \mathrm{dropout}(\mathrm{relu}(\mathrm{norm}(\mathrm{conv}(x))))$, so gradients flow and the encoder does not over-smooth, with hidden width 256, four layers, dropout 0.1, and a two-layer FFN head to the `num_tasks` logits.

This floor is weak by construction, and I want it to fail informatively. The encoder is *local*: four rounds reach four bonds out, smaller than a drug-molecule's diameter, so the pooled vector is an average of local views and misses anything genuinely global. It is *data-hungry*, learning its whole representation from a few hundred to a few thousand labels, and it carries *no external prior* — no pretrained weights, none of the physicochemistry the fixed-descriptor camp would hand it for free. On a *random* split such an encoder can still score by memorizing scaffolds; on a *scaffold* split memorization buys nothing, so every one of these weaknesses bites directly, and the mean readout makes it worse by discarding the size/count signal that often correlates with the property. I therefore expect this rung to be the best of the three on Tox21 (twelve assays, the most molecules, multi-task averaging stabilizes a weak local encoder), decent on BACE (a single structured enzyme-inhibition target where local substructure carries the label), and near *chance* on BBBP — a single binary target on a severe scaffold shift whose answer is global whole-molecule physicochemistry (lipophilicity, polar surface area, size) that a four-hop mean-pooled GNN with no descriptors essentially cannot see. A BBBP that collapses toward $0.5$ while BACE and Tox21 stay well above it is the diagnosis already written: swap the mean readout for a sum that keeps counts, and hand the model the cheap global descriptor prior it lacked — that is the next rung.

```python
class GINConv(nn.Module):
    """Graph Isomorphism Network convolution layer."""

    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.edge_proj = nn.Linear(edge_dim, in_dim)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr, batch_idx):
        """
        x: [total_atoms, in_dim]
        edge_index: [2, total_edges]
        edge_attr: [total_edges, edge_dim]
        batch_idx: [total_atoms]
        """
        src, dst = edge_index
        edge_msg = self.edge_proj(edge_attr)
        msg = x[src] + edge_msg

        # Aggregate messages to destination nodes
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        out = self.mlp((1 + self.eps) * x + agg)
        return out


class MoleculeModel(nn.Module):
    """Starter model: Graph Isomorphism Network (GIN) with mean pooling.

    Simple but effective baseline for molecular property prediction.
    Uses message passing on the molecular graph with learned edge features.
    """

    def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int, task_type: str):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        hidden_dim = 256
        num_layers = 4

        self.atom_embed = nn.Linear(atom_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINConv(hidden_dim, hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, batch):
        """
        Args:
            batch: MolBatch with sparse graph data.
        Returns:
            predictions: [B, num_tasks]
        """
        x = self.atom_embed(batch.x)

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, batch.edge_index, batch.edge_attr, batch.batch_idx)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x = x + self.dropout(x_new)  # residual

        # Mean pooling per graph
        num_graphs = batch.batch_idx.max().item() + 1
        graph_embed = torch.zeros(num_graphs, x.size(-1), device=x.device)
        counts = torch.zeros(num_graphs, 1, device=x.device)
        graph_embed.index_add_(0, batch.batch_idx, x)
        counts.index_add_(0, batch.batch_idx, torch.ones(x.size(0), 1, device=x.device))
        graph_embed = graph_embed / counts.clamp(min=1)

        return self.readout(graph_embed)
```
