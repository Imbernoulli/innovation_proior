## Research question

Predict the binding affinity (`-logKd/Ki`, higher means tighter binding) of a protein-ligand complex
from its 3D structure. The complex is handed to me as a *heterogeneous graph*: a ligand graph, a
pocket graph, and two sets of inter-molecular edges across the interface (ligand→pocket and
pocket→ligand). Atom nodes carry chemical one-hots; intra-molecular edges carry bond chemistry plus
geometric statistics; inter-molecular edges carry geometry only, built between ligand-pocket atom
pairs within 5 Å. The single thing being designed is the **`AffinityModel` architecture** — how those
two molecules and their interface are encoded into one scalar. Everything else (graph construction,
featurization, splits, optimizer, loss harness, evaluation) is fixed.

## Prior art before the first rung

The first rung the ladder reacts to is the geometric-GNN line for molecular property prediction — the
methods that taught the field how to consume 3D structure while respecting that a binding affinity is
a property of the *arrangement*, not the coordinate frame. Each is an ancestor with a gap this task's
heterogeneous interface exposes.

- **Message passing on molecular graphs (Gilmer et al. 2017).** The substrate everything here is a
  fill of: node embeddings `h_i`, edge attributes `a_ij`, message `m_ij = φ_e(h_i,h_j,a_ij)`, sum
  aggregate, update `h_i' = φ_h(h_i, m_i)`, pooled readout. Permutation-equivariant by the symmetric
  sum. Gap: blind to where the atoms are — geometry never enters — and homogeneous, one message
  function for every edge, with no notion that this object is two molecules joined by an interface.
- **SchNet (Schütt et al. 2017; arXiv:1706.08566).** Makes the message depend only on the invariant
  distance through a continuous filter: expand `d_ij` in a Gaussian RBF bank, turn it into a per-edge
  filter, gate the neighbour's features elementwise, sum. Invariant by construction, learned, smooth.
  Gap: the only geometry it sees is the *scalar distance*; the angle and triangle-area statistics the
  data precomputes are thrown away — and it is one homogeneous filter for every edge, so covalent
  bonds and non-covalent contacts get identical treatment.
- **EGNN (Satorras, Hoogeboom, Welling 2021; arXiv:2102.09844).** E(n)-equivariant message passing:
  feed the message the squared relative distance (invariant), and update coordinates by a scalar-weighted
  sum of relative-difference vectors (equivariant). Gap here is twofold — the affinity target is an
  invariant *scalar*, so the equivariant coordinate channel buys nothing on this task; and like SchNet
  it is homogeneous, one convolution for all edges, with no interface decomposition.
- **The interaction-graph line (IGN → GIGN → EHIGN, Yang/Zhong et al. 2023–2024).** The lineage that
  takes the covalent/non-covalent split seriously: a heterogeneous graph with separate intra- and
  inter-molecular relations, processed by distinct convolutions, with the affinity read out from the
  interface. The strongest rungs of *this* ladder are fills of that idea; the gap each rung leaves is
  what the next one reacts to.

## The fixed substrate

A single regression harness is frozen and must not be touched. It loads pre-converted `.pt` graph
tensors into a `PLABatch` (below), trains with Adam (`lr=1e-4`, `weight_decay=1e-6`), batch size 128,
up to 800 epochs with early stopping (patience 50) on a validation RMSE, clips grad-norm to 1.0, and
selects the best-validation checkpoint for test. The loss is plain `F.mse_loss(pred, labels)` on the
model's `forward` output — *unless* the model exposes a `compute_loss(batch, labels)` method, in which
case the harness calls that instead (the hook a multi-head model uses for its own objective). The
harness hands the editable region one helper, `scatter_mean(src, index, dim_size)` (average `src`
rows by `index`), and exposes the standard `torch`, `torch.nn as nn`, `torch.nn.functional as F`.

The batch the model consumes:

```python
@dataclass
class PLABatch:
    # Ligand graph
    lig_x: Tensor              # [total_lig_atoms, 35] atom one-hots
    lig_edge_index: Tensor     # [2, total_lig_edges] COO
    lig_edge_attr: Tensor      # [total_lig_edges, 17] bond + geometric features
    lig_batch: Tensor          # [total_lig_atoms] graph id (0..B-1)
    # Pocket graph
    poc_x: Tensor              # [total_poc_atoms, 35]
    poc_edge_index: Tensor     # [2, total_poc_edges]
    poc_edge_attr: Tensor      # [total_poc_edges, 17]
    poc_batch: Tensor          # [total_poc_atoms]
    # Inter-molecular edges
    l2p_edge_index: Tensor     # [2, total_l2p_edges] (src=ligand, dst=pocket)
    l2p_edge_attr: Tensor      # [total_l2p_edges, 11] geometric features
    p2l_edge_index: Tensor     # [2, total_p2l_edges] (src=pocket, dst=ligand)
    p2l_edge_attr: Tensor      # [total_p2l_edges, 11]
    # Metadata
    num_lig_atoms: List[int]; num_poc_atoms: List[int]
    inter_batch: Tensor        # [total_l2p_edges] graph id for the l2p edges
    labels: Tensor             # [B] target -logKd/Ki
```

The feature contract is fixed: atom features are 35-dim one-hots (element/degree/valence/hybridization/
aromatic/H-count). Intra-molecular edges are 17-dim — bond type (4) + conjugated (1) + in-ring (1) +
11 geometric numbers (angle max/sum/mean, triangle-area max/sum/mean, neighbour-distance max/sum/mean,
pairwise L1, L2). Inter-molecular edges are the same 11 geometric numbers. **The geometry is already
baked into `edge_attr`; raw 3D coordinates are not handed to the model.** Any model that wants a
distance reads it off the last geometric channel (`edge_attr[:, -1:]`, the L2 distance, scaled by 0.1).

## The editable interface

Exactly one region is editable: the `AffinityModel` class (and any helper layers/modules) between the
`EDITABLE SECTION START`/`END` markers in `EHIGN_PLA/custom_pla.py`. The contract is
`__init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim)` and
`forward(self, batch: PLABatch) -> Tensor` returning shape `[B]`. Every method on the ladder is a fill
of this same slot.

The starting point is the scaffold default: two **independent** GNN encoders (ligand and pocket),
each a stack of simple edge-conditioned message-passing layers with residuals, mean-pooled per graph
and concatenated into a regression head — and it **ignores the inter-molecular edges entirely**.

```python
# EDITABLE region of EHIGN_PLA/custom_pla.py — default fill (separate encoders, no interface)

class SimpleGNNLayer(nn.Module):
    """Simple message passing layer with edge features."""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        msg_input = torch.cat([x[src], edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)
        agg = torch.zeros(x.size(0), msg.size(-1), device=x.device)
        agg.index_add_(0, dst, msg)
        out = self.update_mlp(torch.cat([x, agg], dim=-1))
        return out


class AffinityModel(nn.Module):
    """Starter: separate GNN encoders for ligand/pocket + mean-pool concat readout.
    Does NOT use inter-molecular edges."""

    def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
        super().__init__()
        hidden_dim = 256
        num_layers = 3
        self.lig_embed = nn.Linear(lig_dim, hidden_dim)
        self.lig_convs = nn.ModuleList([
            SimpleGNNLayer(hidden_dim, intra_edge_dim, hidden_dim) for _ in range(num_layers)])
        self.poc_embed = nn.Linear(poc_dim, hidden_dim)
        self.poc_convs = nn.ModuleList([
            SimpleGNNLayer(hidden_dim, intra_edge_dim, hidden_dim) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim, 1))

    def forward(self, batch: PLABatch) -> torch.Tensor:
        lig_h = self.lig_embed(batch.lig_x)
        for conv in self.lig_convs:
            lig_h = conv(lig_h, batch.lig_edge_index, batch.lig_edge_attr) + lig_h
        poc_h = self.poc_embed(batch.poc_x)
        for conv in self.poc_convs:
            poc_h = conv(poc_h, batch.poc_edge_index, batch.poc_edge_attr) + poc_h
        num_graphs = batch.labels.size(0)
        lig_pool = scatter_mean(lig_h, batch.lig_batch, num_graphs)
        poc_pool = scatter_mean(poc_h, batch.poc_batch, num_graphs)
        combined = torch.cat([lig_pool, poc_pool], dim=-1)
        pred = self.readout(combined).squeeze(-1)
        return pred
```

Each method on the ladder replaces exactly this class (and its helper layers) and nothing else.

## Evaluation settings

Train on PDBbind v2020 (general + refined). Test on three benchmarks: **PDBbind 2013 core set** (107
complexes, CASF-2013), **PDBbind 2016 core set** (285 complexes, CASF-2016), and a **PDBbind 2019
holdout** (4366 complexes, temporal split). Two metrics per benchmark: **RMSE** (lower is better) and
**Rp** = Pearson correlation between predictions and labels (higher is better). One seed (42). The
overall task score normalizes each metric against the leaderboard's worst/best baseline and combines
the three benchmarks; lower RMSE and higher Rp both help.
