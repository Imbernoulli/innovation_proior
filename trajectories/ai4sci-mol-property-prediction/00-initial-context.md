## Research question

Predict chemical properties — blood-brain-barrier penetration, beta-secretase inhibition, toxicity across assays — from molecular structure alone, and generalize to molecules whose *scaffold* never appeared in training. The only design object is the **molecular representation model**: the map from a molecule (an atom/bond graph, optionally with 3D coordinates) to a fixed vector that a small head turns into one logit per task. Everything else — SMILES preprocessing, conformer generation, the scaffold split, the optimizer schedule, target normalization, the masked loss for missing labels, and test-time augmentation over conformers — is frozen. The metric is ROC-AUC (higher is better), averaged over valid labels per task and across tasks, on a **scaffold** split that shares no Murcko scaffold between train and test.

## Prior art / Background / Baselines

The default encoder is a message-passing graph net built on a converged substrate (RDKit atom/bond featurization, learned graph-to-vector encoder). The relevant baselines are:

- **Fixed-descriptor models (Morgan/ECFP fingerprints, RDKit 2D descriptors + RF/FFN).** Hash a molecule into a fixed bit-vector or computed physicochemical descriptors and run a generic classifier.
- **Neural molecular fingerprints (Duvenaud et al. 2015).** Learn a differentiable fingerprint by aggregating neighbor atom features through degree-specific matrices and training end-to-end.
- **Gated graph nets / Weave (Li et al. 2016; Kearnes et al. 2016).** Apply edge-typed linear messages with GRU updates, or keep explicit edge representations.
- **MPNN framework (Gilmer et al. 2017).** Cast the above as special cases of a message phase, update phase, and permutation-invariant readout.
- **3D Transformer / Uni-Mol.** Treat atoms as a fully-connected set so distant-in-bonds-but-close-in-space contacts are visible.

## Fixed substrate / Code framework

A self-contained molecular-property pipeline is frozen and must not be touched. It loads pre-split LMDB data with pre-computed multi-conformer 3D coordinates, samples/enumerates a conformer per molecule, removes polar hydrogens, normalizes coordinates, tokenizes against the vocabulary, and builds distance matrices and atom-pair edge types. The training loop uses Adam with warmup-then-decay, normalizes regression targets, applies masked binary-cross-entropy that ignores missing labels, and averages predictions over 11 conformers at val/test time.

The GNN featurization is fixed: `ATOM_DIM = 136` (one-hot atomic number 118, degree 6, formal charge 5, num-H 5, hybridization 5, aromatic 1, in-ring 1) and `EDGE_DIM = 9` (bond type 4, stereo 3, conjugated 1, in-ring 1). A pretrained checkpoint is available inside the container at the path the strongest baseline loads from.

Each batch is a `MolBatch` carrying both representations at once, so a model picks what it needs:

- **Sparse graph (for GNNs):** `x [total_atoms, 136]`, `edge_index [2, total_edges]` (COO, bonds stored as adjacent forward/reverse pairs), `edge_attr [total_edges, 9]`, `batch_idx [total_atoms]` (graph id).
- **Dense (for Transformers):** `atom_features [B, N, 136]` zero-padded, `positions [B, N, 3]`, `dist_matrix [B, N, N]`, `mask [B, N]` (1 = real atom).
- **Uni-Mol-specific:** `atom_tokens`, `edge_types`, plus dynamically-set `_unimol_dist`, `_unimol_token_mask`.
- **Targets:** `targets [B, num_tasks]`, `target_mask [B, num_tasks]` (1 = valid label).

## Editable interface

Exactly one region is editable — the `MoleculeModel` class (and any helper modules/functions placed beside it) in `custom_molprop.py`, between the `EDITABLE SECTION START`/`END` markers. Every method fills this same contract:

- `__init__(self, atom_dim, edge_dim, num_tasks, task_type)` — build the architecture.
- `forward(self, batch) -> Tensor` — return predictions of shape `[B, num_tasks]` from a `MolBatch`.

The model chooses which batch fields to read (sparse graph, dense tensors, or Uni-Mol tokens) and may read a per-dataset `pooler_dropout` set as a class attribute by the training driver. The starting point is the scaffold default: a small **GIN with mean pooling and edge features**. Each method replaces exactly this class (plus its helpers) and nothing else.

```python
# EDITABLE region of custom_molprop.py — default fill (starter GIN, mean pooling)
class GINConv(nn.Module):
    """Graph Isomorphism Network convolution layer (edge-aware)."""

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
        src, dst = edge_index
        edge_msg = self.edge_proj(edge_attr)
        msg = x[src] + edge_msg                      # message folds in bond features
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)                  # sum neighbor messages
        out = self.mlp((1 + self.eps) * x + agg)     # (1+eps) center tag, then MLP
        return out


class MoleculeModel(nn.Module):
    """Starter model: GIN with mean pooling."""

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
        x = self.atom_embed(batch.x)
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, batch.edge_index, batch.edge_attr, batch.batch_idx)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x = x + self.dropout(x_new)               # residual

        num_graphs = batch.batch_idx.max().item() + 1
        graph_embed = torch.zeros(num_graphs, x.size(-1), device=x.device)
        counts = torch.zeros(num_graphs, 1, device=x.device)
        graph_embed.index_add_(0, batch.batch_idx, x)
        counts.index_add_(0, batch.batch_idx, torch.ones(x.size(0), 1, device=x.device))
        graph_embed = graph_embed / counts.clamp(min=1)   # MEAN pooling

        return self.readout(graph_embed)
```

## Evaluation settings

Three MoleculeNet classification benchmarks, each with the official **scaffold** split, scored by ROC-AUC (higher is better, averaged over valid labels per task and across tasks):

- **BBBP** — blood-brain-barrier penetration (2,039 molecules, 1 task).
- **BACE** — beta-secretase 1 inhibition (1,513 molecules, 1 task).
- **Tox21** — toxicity across 12 assays (7,831 molecules, 12 tasks, multi-task with missing labels).

One seed (42). Test-time augmentation averages each prediction over 11 conformers. The metric columns reported are `rocauc_BBBP`, `rocauc_BACE`, `rocauc_Tox21`.
