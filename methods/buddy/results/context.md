# Research question

We are given a single partially observed undirected graph G = (V, E) — node features X ∈ R^{N×F} when
they exist, and a set of observed edges — and we want to score arbitrary node pairs (u, v) so that the
true-but-held-out edges rank above non-edges. The task is *link prediction*: friend recommendation in
social networks, paper recommendation in citation networks, knowledge-graph completion, biological
interaction prediction. The benchmark regime here is broad: small citation graphs (Cora,
CiteSeer, Pubmed — thousands of nodes, rich sparse features) and large open-benchmark graphs
(ogbl-collab, ogbl-ppa, ogbl-citation2 — hundreds of thousands to millions of nodes). The yardstick is
ranking quality (Hits@K, MRR) on held-out edges, with wall-clock training/inference time a secondary
constraint. The question is how to score node pairs using node features and graph structure together.

# Background

The field at this moment splits into three lines.

**Neighborhood-overlap heuristics.** The oldest approach scores a pair by
a fixed function of how much their neighborhoods overlap: common neighbors CN(u,v) = |N(u) ∩ N(v)|,
Adamic–Adar AA(u,v) = Σ_{w ∈ N(u)∩N(v)} 1/log deg(w) (Adamic & Adar, 2003), resource allocation
RA(u,v) = Σ 1/deg(w) (Zhou et al., 2009). These are strong on dense citation and
collaboration graphs — two papers that share many citations tend to cite each other — and they are the
backbone of classical link prediction. They use a hand-fixed formula, no node features, no learning.

**Message-passing GNNs (GCN, GraphSAGE; Kipf & Welling 2016; Hamilton et al. 2017).** A GNN encodes
each node into an embedding by aggregating its neighbors' features through L rounds of message passing,
H^{(l+1)} = σ(Â H^{(l)} W^{(l)}), and a link is scored from the two node embeddings — typically a dot
product or an MLP on [z_u, z_v, z_u ⊙ z_v]. These learn from features and structure end-to-end and
scale in O(|E|). A GNN produces a single representation per node: two nodes that are *automorphic* —
structurally identical to the GNN — receive *identical* embeddings, and standard message-passing GNNs
have known limits on counting substructures such as triangles, where a triangle through u, v, w is a
common neighbor w.

**Subgraph GNNs (SEAL; Zhang & Chen 2018; labeling-trick theory, Zhang et al. 2021; Distance Encoding,
Li et al. 2020).** These build *link* representations rather than node representations.
For each candidate (u, v), extract the k-hop enclosing subgraph, *label* every node in it by its
position relative to the pair — Double-Radius Node Labeling (DRNL) encodes each node w by the pair of
distances (d(w,u), d(w,v)); Distance Encoding (DE) generalizes this to a tuple z_w = (d(w,u),
d(w,v)) — and run a graph-level GNN with pooling over the labeled subgraph to classify the link. The
labeling trick lets a GNN over a properly labeled enclosing subgraph learn neighborhood-overlap
heuristics: the labels break the automorphism (u and v are uniquely marked) and the (d_u, d_v) tuples
encode common-neighbor structure — a node labeled (1,1) is a common neighbor, a triangle. SEAL and its
descendants are the state of the art on these benchmarks. A separate enclosing subgraph is extracted
and a GNN run per candidate edge, at both training and inference; for regular graphs this is O(deg^k),
and for power-law complex networks it grows with |E| per edge. NBFNet, a related path that shares
Bellman-Ford-style distance computation across destinations, runs faster.

The question this work addresses is whether the structural information a subgraph GNN extracts — the
(d_u, d_v) distance-label counts that subsume the overlap heuristics — can be combined with node
features through a scalable learned predictor.

## Code framework

```python
class BuddyAllocator:
    """Buddy-system memory allocator scaffold."""

    def __init__(self, total_size: int, min_block: int = 1):
        self.total_size = total_size
        self.min_block = min_block
        self.free = {total_size: [0]}
        self.allocated = {}

    def allocate(self, size: int) -> int:
        """Return start address of an allocated block, or raise."""
        raise NotImplementedError

    def free(self, addr: int) -> None:
        """Free the block starting at addr and merge buddies."""
        raise NotImplementedError

    def split(self, size: int, addr: int) -> None:
        """Split a free block into two buddy halves."""
        raise NotImplementedError

    def merge(self, addr: int, size: int) -> None:
        """Merge addr with its buddy if both are free."""
        raise NotImplementedError
```
