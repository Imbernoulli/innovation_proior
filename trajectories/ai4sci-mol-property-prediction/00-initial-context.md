## Research question

Predict chemical properties — blood-brain-barrier penetration, beta-secretase inhibition, toxicity
across a panel of assays — from molecular structure alone, and predict them on molecules whose
*scaffold* never appeared in training. The single thing being designed is the **molecular
representation model**: the map from a molecule (a SMILES string turned into an atom/bond graph, with
optional 3D coordinates) to a fixed vector that a small head turns into one logit per task. Everything
else — SMILES preprocessing, conformer generation, the scaffold split, the optimizer schedule, target
normalization, the masked loss for missing labels, and test-time augmentation over conformers — is
frozen by the scaffold. The metric is ROC-AUC (higher is better), averaged over valid labels per task
and across tasks, on a **scaffold** split: train and test share no Murcko scaffold, so any score above
chance has to come from structure the model genuinely generalizes, not from a memorized series.

## Prior art before the first rung (molecular-representation lineage)

The encoder the first rung uses — a message-passing graph net — is itself the resolution of a line of
molecular-representation methods. These precede the ladder; the substrate below is what they converged
to (RDKit featurization of atoms and bonds, then a learned graph-to-vector encoder).

- **Fixed-descriptor models (Morgan/ECFP fingerprints, RDKit 2D descriptors + RF/FFN).** Hash a
  molecule into a fixed bit-vector or a pile of computed physicochemical descriptors and run a generic
  classifier on it. They carry a strong, general chemical prior and need little data, but the
  representation is *frozen* — it can never specialize to the property actually being predicted. Gap:
  cannot adapt the encoding to the task.
- **Neural molecular fingerprints (Duvenaud et al. 2015).** Replace the fixed hash with a learned,
  differentiable one: aggregate neighbor atom features through degree-specific matrices and sum into a
  fingerprint trained end-to-end. Learnable at last, but the message is a plain concatenation of atom
  and bond features and the depth is shallow. Gap: weak message function, still atom-centered.
- **Gated graph nets / Weave (Li et al. 2016; Kearnes et al. 2016).** Edge-typed linear messages with a
  GRU update (GGNN), or keeping explicit edge representations around (Weave). More expressive updates,
  but the hidden state still lives on the *atom*, so a message sent out one bond is summed straight back
  along the same bond next step — the representation re-mixes its own echoes. Gap: atom-centered message
  passing "totters".
- **The MPNN framework (Gilmer et al. 2017).** The unifying view every model above is a special case of:
  a message phase `m_v = Σ_{w∈N(v)} M(h_v,h_w,e_vw)`, an update `h_v ← U(h_v,m_v)`, then a permutation-
  invariant readout. It names the design space — choose `M`, `U`, the readout — but leaves the choice
  open, and the default atom-centered instantiation still tots and still only reaches `T` hops, smaller
  than a drug-molecule's diameter. Gap: framework, not an answer; locality and tottering unresolved.

The 3D side has its own ancestor the strongest rung reacts to: a Transformer treats atoms as a
fully-connected set so distant-in-bonds-but-close-in-space contacts are visible, but a vanilla
Transformer is position-blind, and the position here is a *continuous, rotation/translation-arbitrary*
3D coordinate that must enter as an SE(3)-invariant signal — the gap the strongest baseline closes.

## The fixed substrate

A self-contained molecular-property pipeline is frozen and must not be touched. It mirrors the Uni-Mol
data path: load pre-split LMDB data (official scaffold splits) with pre-computed multi-conformer 3D
coordinates; for each molecule, sample/enumerate a conformer, remove polar hydrogens, normalize
coordinates, tokenize against the Uni-Mol vocabulary, and build a distance matrix and atom-pair edge
types. It then runs the training loop (Adam, warmup-then-decay schedule), normalizes regression targets,
applies a masked binary-cross-entropy loss that ignores missing labels in multi-task datasets, and at
val/test time averages predictions over 11 conformers (test-time augmentation).

The featurization the GNN path consumes is fixed: `ATOM_DIM = 136` (one-hot atomic number 118, degree 6,
formal charge 5, num-H 5, hybridization 5, aromatic 1, in-ring 1) and `EDGE_DIM = 9` (bond type 4,
stereo 3, conjugated 1, in-ring 1). The loop also exposes a pretrained Uni-Mol checkpoint inside the
container at the path the strongest baseline loads from.

Each batch is a `MolBatch` carrying both representations at once, so a model picks what it needs:

- **Sparse graph (for GNNs):** `x [total_atoms, 136]`, `edge_index [2, total_edges]` (COO, bonds stored
  as adjacent forward/reverse pairs), `edge_attr [total_edges, 9]`, `batch_idx [total_atoms]` (graph id).
- **Dense (for Transformers):** `atom_features [B, N, 136]` zero-padded, `positions [B, N, 3]`,
  `dist_matrix [B, N, N]`, `mask [B, N]` (1 = real atom).
- **Uni-Mol-specific:** `atom_tokens`, `edge_types`, plus dynamically-set `_unimol_dist`,
  `_unimol_token_mask`.
- **Targets:** `targets [B, num_tasks]`, `target_mask [B, num_tasks]` (1 = valid label).

## The editable interface

Exactly one region is editable — the `MoleculeModel` class (and any helper modules/functions placed
beside it) in `custom_molprop.py`, between the `EDITABLE SECTION START`/`END` markers (the lines the
config exposes for editing). Every method on the ladder is a fill of this same contract:

- `__init__(self, atom_dim, edge_dim, num_tasks, task_type)` — build the architecture.
- `forward(self, batch) -> Tensor` — return predictions of shape `[B, num_tasks]` from a `MolBatch`.

The model chooses which batch fields to read (sparse graph, dense tensors, or Uni-Mol tokens) and may
read a per-dataset `pooler_dropout` set as a class attribute by the training driver. The starting point
is the scaffold default: a small **GIN with mean pooling and edge features**. Each later method replaces
exactly this class (plus its helpers) and nothing else.

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

Three MoleculeNet classification benchmarks, each with the official **scaffold** split, scored by
ROC-AUC (higher is better, averaged over valid labels per task and across tasks):

- **BBBP** — blood-brain-barrier penetration (2,039 molecules, 1 task).
- **BACE** — beta-secretase 1 inhibition (1,513 molecules, 1 task).
- **Tox21** — toxicity across 12 assays (7,831 molecules, 12 tasks, multi-task with missing labels).

One seed (42). Test-time augmentation averages each prediction over 11 conformers. The metric columns
reported are `rocauc_BBBP`, `rocauc_BACE`, `rocauc_Tox21`.
