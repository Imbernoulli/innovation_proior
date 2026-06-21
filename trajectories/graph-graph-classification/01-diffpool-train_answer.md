The readout is the entire job here: the GIN backbone hands me one embedding per node, and graph classification needs one vector per graph, so the last mile is collapsing a variable-size bag of node vectors into a single fixed-size graph vector. The scaffold ships the flattest thing that works at all, `global_add_pool(x, batch)` — a single global sum over the final layer. What nags at me about that floor is that it is *flat*: message passing moves information along edges out to $K$ hops and stops, so each node ends up holding a summary of its $K$-hop neighborhood, and then the sum dumps all $N$ of those summaries into one pot. At no point does the model represent anything at a scale *between* a node's neighborhood and the whole graph — yet a molecule is atoms $\to$ bonds $\to$ functional groups $\to$ the whole molecule, and the label I care about (mutagenic? enzyme?) often lives at that intermediate, functional-group scale a flat sum has already discarded. So before I accept the trivial sum as my bottom rung, I want a more *structured* default: a graph analogue of the conv-then-pool hierarchy that makes deep CNNs powerful, where each pooling step coarsens resolution so later stages see bigger structure.

The obstacle is that CNN pooling leans on a grid — a $2\times2$ patch means the same thing everywhere — and a graph has no grid: node 5 has three neighbors arranged one way, node 6 has eleven arranged another, and there is no canonical local window. Sorting nodes into a sequence and pooling consecutive runs needs a structure-preserving node ordering, which is essentially graph isomorphism, and would pool together nodes that are far apart in the graph. Once I drop the patch idea, the honest abstraction of graph pooling is: take $N$ nodes with embeddings and an adjacency, and output a coarser graph of $K < N$ super-nodes, where each super-node is a *cluster* of old nodes. The whole problem reduces to how to cluster and how to aggregate. A deterministic off-the-shelf clustering (spectral, a coarsening heuristic) is computed before any gradient flows by a subroutine blind to the classification loss, and is a per-graph black box with no shared, transferable pooling strategy. I want the clustering itself to be *learned*, end-to-end, by shared parameters.

I propose **DiffPool** — a learned soft-cluster pooling — in the single-level, stripped form the edit surface permits. The moment I demand differentiability the obstacle is that a clustering is a *hard* assignment (node $i$ goes to cluster $j$, a discrete choice with no gradient). The escape is to *soften* it: instead of one cluster per node, give each node a distribution over clusters. Define a soft assignment matrix $S$ whose row $i$ is node $i$ and whose entries are the probabilities that node $i$ belongs to each of the $K$ clusters, made a genuine distribution by a softmax. Then the embedding of cluster $j$ is the soft collection of the nodes assigned to it, node $i$ contributing in proportion to $S_{ij}$ — a weighted sum, which stacked over clusters is exactly the matrix product

$$X' = S^{\top} Z,$$

where $Z$ are the node features and row $j$ of $X'$ is cluster $j$'s pooled embedding. This is the coarse-graph readout, and it falls right out of treating $S$ as a soft node$\to$cluster indicator. It also respects the one invariant I must keep. Relabel the nodes by a permutation $P$: features become $PZ$ and, for an equivariant assignment network, $S$ becomes $PS$, so $X' = (PS)^{\top}(PZ) = S^{\top}P^{\top}P\,Z = S^{\top}Z$ because $P^{\top}P = I$. The coarsened embedding is invariant to node numbering, exactly what a graph-level answer needs to be well-defined.

The assignment should depend on each node's features and position, which a small network over the node embeddings computes: a per-node output of width $K$, softmaxed. Here is the one deliberate place this version departs from the canonical method, and I make it by reading the constraints I actually face. The full DiffPool generates $S$ with its own *GNN over the current adjacency* (so a node's cluster is informed by its neighborhood), generates the embeddings with a *second, separate* GNN, and — crucially — trains the pooling with two auxiliary losses: a link-prediction term $\lVert A - SS^{\top}\rVert_F$ that forces connected nodes into the same cluster, and an entropy term that pushes each assignment row toward one-hot so clusters are crisp. Those auxiliaries exist because the task gradient alone, routed back through a non-convex factorization-like clustering, is too weak to find good clusters. But my editable surface is *only* the post-backbone `GraphReadout`: it receives node embeddings from a frozen GIN and returns one graph vector. I cannot insert a clustering GNN into the backbone, I cannot feed a coarsened adjacency back into more convolution, and there is no hook to add auxiliary terms to a training step that backpropagates only the classification cross-entropy. So the version that fits is the *stripped* one: a single coarsening level, the assignment produced by a plain MLP over node features (not a GNN over the adjacency), and **no link or entropy auxiliary**. I keep DiffPool's load-bearing idea — a learned soft assignment pooled as $S^{\top}Z$ — and drop the machinery the harness will not let me wire in.

Concretely, the readout converts the batch's stacked node embeddings to dense padded form with `to_dense_batch` $\to [B, N_{\max}, D]$ plus a validity mask, computes raw assignment scores with a two-layer MLP $\text{Linear}(D,D)\to\text{ReLU}\to\text{Linear}(D,K)$ $\to [B, N_{\max}, K]$, masks padded nodes to $-\infty$ before the softmax (so phantom padding nodes get zero assignment), softmaxes *over the nodes* (`dim=1`) so each cluster is a normalized mixture of real nodes, re-zeroes the padded rows, and pools $X' = S^{\top}X_{\text{dense}} \to [B, K, D]$. The final collapse to one graph vector is a plain mean over the $K$ clusters, giving $[B, D]$. I fix $K = 25$ clusters and pad to $N_{\max} = 150$ nodes, comfortably above the per-graph node counts in these datasets, and set `output_dim = hidden_dim` so the fixed classifier head consumes it unchanged.

I want to be honest about what this should do, because that is the whole point of starting here. The single thing I kept — a learned soft assignment pooled as $S^{\top}X$ — is also the single thing the auxiliaries were there to *protect*, and I have removed the protection. Without the link loss nothing tells connected nodes to share a cluster; the MLP assigns from node *features alone*, blind to the adjacency, so "clusters" need not correspond to any graph structure. Without the entropy loss nothing stops the softmax from staying diffuse — each node spread $\approx 1/K$ across every cluster — in which case every cluster-node is a faint average of the whole graph and $S^{\top}X$ degenerates toward "the same global average copied $K$ times," whose mean over clusters is just a scaled global *mean* pool. So my prediction is that this readout behaves, at best, like a learned, noisier global mean — and a mean discards the counts a sum keeps, which makes it strictly weaker on the multiset of node features. That is exactly why it is the bottom rung: it carries the most ambitious idea on the ladder, a *learned* hierarchy, but in the one form the harness permits, shorn of the auxiliaries that make the idea pay. On tiny MUTAG (188 graphs) the assignment has almost no data and I expect high seed-to-seed variance; on PROTEINS and NCI1 it gets more gradient but, with no structural prior, should land near or just below a plain global pool. Whichever way it splits, the diagnosis is already pointed at the next rung: the weakness is that it discarded *counts* and made no use of the per-layer `layer_outputs` the scaffold hands me — so the clean next move is the most expressive *flat* reduction there is, an injective sum, read from every layer.

```python
class GraphReadout(nn.Module):
    """DiffPool Readout (Ying et al., 2018).

    Uses a learned soft assignment matrix to cluster nodes into
    a fixed number of super-nodes, then reads out from the
    coarsened graph. Two-level hierarchy.
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Assignment network: maps nodes to clusters
        self.max_nodes = 150  # Max nodes per graph (padded)
        self.num_clusters = 25
        self.assign_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_clusters),
        )
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, layer_outputs):
        # Convert to dense batch format
        x_dense, mask = to_dense_batch(x, batch)  # [B, N_max, D]
        adj = to_dense_adj(edge_index, batch)  # [B, N_max, N_max]

        # Compute soft assignment
        s = self.assign_nn(x_dense)  # [B, N_max, K]
        s = s.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        s = torch.softmax(s, dim=1)
        s = s * mask.unsqueeze(-1).float()

        # Pool: X_coarse = S^T @ X, A_coarse = S^T @ A @ S
        x_coarse = torch.bmm(s.transpose(1, 2), x_dense)  # [B, K, D]

        # Global mean pool over clusters
        return x_coarse.mean(dim=1)  # [B, D]
```
