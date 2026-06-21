## Research question

Predict the binding affinity (`-logKd/Ki`, higher means tighter binding) of a protein-ligand complex from its 3D structure. The input is a heterogeneous graph: a ligand graph, a pocket graph, and bidirectional inter-molecular edges within 5 Å. Atom nodes carry 35-dim chemical one-hots; intra-molecular edges carry bond chemistry plus 11 geometric statistics; inter-molecular edges carry only the 11 geometric statistics. The design target is the `AffinityModel` architecture — how the two molecules and their interface are encoded into one scalar. Everything else (graph construction, featurization, splits, optimizer, loss, evaluation) is fixed.

## Prior art / Background / Baselines

- **Message passing networks (Gilmer et al. 2017).** Core idea: propagate edge-conditioned messages between atom nodes and sum-aggregate them to update node states. Gap: geometry never enters the messages, and a single homogeneous message function treats every edge identically, so covalent bonds and ligand-pocket contacts are not distinguished.

- **SchNet (Schütt et al. 2017).** Core idea: build continuous filters from Gaussian-expanded interatomic distances and apply them as elementwise gates in message passing. Gap: only the scalar distance is used; angle and triangle-area statistics are discarded, and the same filter is applied to covalent and non-covalent edges alike.

- **EGNN (Satorras et al. 2021).** Core idea: E(n)-equivariant message passing that updates node coordinates alongside features. Gap: binding affinity is an invariant scalar, so equivariant coordinate updates do not directly target the task; it also uses a single convolution for all edges and provides no interface decomposition.

- **Interaction-graph networks (IGN, GIGN, EHIGN).** Core idea: model the complex as a heterogeneous graph with separate intra- and inter-molecular convolutions and read affinity from the interface. Gap: existing variants still leave residual prediction error on the core benchmarks, especially for complexes with subtle interface geometry or similar binding modes.

## Fixed substrate / Code framework

A single regression harness is frozen. It loads pre-converted `.pt` tensors into a `PLABatch`, trains with Adam (`lr=1e-4`, `weight_decay=1e-6`), batch size 128, up to 800 epochs with early stopping (patience 50) on validation RMSE, clips grad-norm to 1.0, and selects the best-validation checkpoint for test. The loss is `F.mse_loss(pred, labels)` unless the model exposes `compute_loss(batch, labels)`. The harness exposes `torch`, `torch.nn as nn`, `torch.nn.functional as F`, and one helper, `scatter_mean(src, index, dim_size)`.

The batch:

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

Atom features are 35-dim one-hots (element/degree/valence/hybridization/aromatic/H-count). Intra-molecular edges are 17-dim: bond type (4) + conjugated (1) + in-ring (1) + 11 geometric statistics. Inter-molecular edges are the same 11 geometric statistics. Raw 3D coordinates are not passed; the L2 distance is the last geometric channel (`edge_attr[:, -1:]`, scaled by 0.1).

## Editable interface

Only the `AffinityModel` class (and helper layers) between `EDITABLE SECTION START`/`END` markers in `EHIGN_PLA/custom_pla.py` may be changed. Contract: `__init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim)` and `forward(self, batch: PLABatch) -> Tensor` returning `[B]`.

The default fill is two independent GNN encoders (ligand and pocket) with edge-conditioned message-passing layers, mean-pooled and concatenated into a regression head. It ignores inter-molecular edges.

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

Each method fills exactly this slot.

## Evaluation settings

Train on PDBbind v2020 (general + refined). Test on PDBbind 2013 core set (107 complexes, CASF-2013), PDBbind 2016 core set (285 complexes, CASF-2016), and a PDBbind 2019 holdout (4366 complexes, temporal split). Metrics per benchmark: RMSE (lower better) and Rp = Pearson correlation (higher better). Seed 42. The overall score normalizes each metric against the leaderboard worst/best and combines the three benchmarks.
