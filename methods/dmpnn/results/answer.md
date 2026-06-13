# D-MPNN, distilled

The Directed Message Passing Neural Network (D-MPNN) is a graph neural network for molecular
property prediction whose hidden states and messages live on **directed bonds** rather than on
atoms. Putting the state on directed edges lets the message into edge `v->w` be built from the
*other* edges flowing into `v`, excluding the reverse edge `w->v` — the belief-propagation
exclusion — which eliminates the immediate back-and-forth ("totter") that atom-centered message
passing suffers. After message passing it returns to an atom representation, sums atoms into a
molecule vector, optionally concatenates fixed RDKit molecule-level descriptors when the data
pipeline supplies them, and predicts with a feed-forward head.

## Problem it solves

Predict one or more chemical properties of a molecule from its structure (toxicity,
blood-brain-barrier penetration, enzyme inhibition, solubility, ...), with a single learned
representation that can compete with both fixed-fingerprint models and prior atom-centered graph
networks, and that is evaluated for generalization to structurally novel molecules (using a
scaffold split, a proxy for the chronological split of real drug discovery).

## Key idea

Atom-centered message passing aggregates over *all* neighbors each step,
`m_v^{t+1} = sum_{w in N(v)} M_t(h_v^t, h_w^t, e_vw)`, so a message sent out an edge bounces
straight back the next step — a totter that re-mixes a node's own information. The fix is the
belief-propagation exclusion `N(v)\w`, but an undirected atom state cannot express "the
reverse"; so move the state onto **directed bonds** `h_vw` (with `h_vw != h_wv`). This is the
directed-edge embedding of loopy belief propagation.

## Final architecture

Initialize each directed bond from its source atom and bond features:

```
h_vw^0 = tau( W_i · cat(x_v, e_vw) )                 tau = ReLU
```

Message passing after initialization excludes the reverse message. In the implementation convention,
`depth` counts `h^0`, so the implementation below performs `depth - 1` learned updates:

```
m_vw^{t+1} = sum_{k in N(v)\w} h_kv^t                # excludes the reverse bond w->v
h_vw^{t+1} = tau( h_vw^0 + W_m · m_vw^{t+1} )        # tied W_m every step; skip connection to h_vw^0
```

The message function is the identity (`M_t(·) = h_vw^t`); the update is a single shared linear
map with a ReLU and a residual skip back to `h_vw^0`. Return to atoms, then pool to a molecule
vector:

```
m_v = sum_{w in N(v)} h_wv^T
h_v = tau( W_a · cat(x_v, m_v) )
h   = sum_{v in G} h_v                               # permutation-invariant (sum; mean optional)
```

Predict with a feed-forward net `f`, optionally concatenating fixed RDKit molecule-level
descriptors `h_f`:

```
y_hat = f( cat(h, h_f) )
```

**Efficiency.** `sum_{k in N(v)\w}` is computed as "sum over all incoming bonds at `v`, then
subtract the one reverse bond": aggregate `a_v = sum_{k in N(v)} h_kv` once per atom (linear in
bonds), then `m_vw = a_v - h_wv`. Storing bonds in `(v->w),(w->v)` pairs makes the reverse of
bond `e` simply `e XOR 1`, an O(1) lookup. So the directed scheme costs no more than a plain
atom aggregation.

## Why each choice

- **Directed-bond state:** makes the `N(v)\w` exclusion expressible; this is embedding loopy BP
  rather than mean-field (the atom MPNN).
- **Subtract the reverse (`a_v - h_wv`):** the BP exclusion at O(1) per bond.
- **Skip to `h_vw^0` every step:** with a *tied* update the bond's raw identity would wash out
  over depth; re-injecting `h_vw^0` preserves it and shortens the gradient path.
- **Tied `W_m`:** weight tying keeps parameters flat in depth; depth `T` is a pure hyperparameter.
- **Concatenate `x_v` at atom readout and `x_v,e_vw` at init:** mix atom and bond streams inside
  one matrix so atom-bond correlations are representable (which the separate-sum neural-fingerprint
  message could not), and re-inject atom identity the bond messages may have drifted from.
- **Sum readout:** permutation-invariant over the (unordered) atoms.
- **RDKit descriptors `h_f`:** the message passing is local (`T < diam(G)`) and data-hungry on
  small datasets; a fixed global descriptor branch supplies global, prior chemical knowledge and
  regularizes the low-data regime.
- **CDF normalization of `h_f`:** descriptors span wildly different scales; mapping each through
  its CDF gives every feature the same meaning (a percentile), robust to outliers and to the
  non-normal, count-based chemical features that break z-scoring. (A streaming approximation uses
  running per-feature statistics.)

## Defaults

Hidden size `h = 300`, depth `T = 3`, FFN = 2 layers, dropout `0` by default (raised per dataset
on small/overfitting-prone sets), ReLU activation, sum aggregation. Adam optimizer with a Noam
schedule (linear warmup ~2 epochs from `1e-4` to `1e-3`, then exponential decay to `1e-4`),
Xavier-normal weights / zero biases. Masked BCE for multi-task classification with missing
labels; normalized-target MSE for regression. Evaluation on a Murcko-scaffold split (zero
scaffold overlap between train and test).

## Working code

Filling the graph-batch harness slot, with the directed update done as `a_v - h_wv`, the tied
`W_m`, and the `h_vw^0` skip:

```python
import torch
import torch.nn as nn


def scatter_sum(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class DMPNNEncoder(nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden_dim=300, depth=3, dropout=0.0):
        super().__init__()
        self.hidden_dim, self.depth = hidden_dim, depth
        self.W_i = nn.Linear(atom_dim + edge_dim, hidden_dim, bias=False)
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_a = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        n_atoms, n_bonds = x.size(0), edge_index.size(1)
        if n_bonds == 0:
            h_atom = self.act(self.W_a(torch.cat(
                [x, torch.zeros(n_atoms, self.hidden_dim, device=x.device, dtype=x.dtype)], dim=-1)))
            return self.dropout(h_atom)

        if n_bonds % 2 != 0:
            raise ValueError("Directed bonds must be stored as adjacent forward/reverse pairs.")
        rev = torch.arange(n_bonds, device=x.device) ^ 1
        if not bool(((src[rev] == dst) & (dst[rev] == src)).all().item()):
            raise ValueError("edge_index must store each reverse bond at index e XOR 1.")

        h0 = self.act(self.W_i(torch.cat([x[src], edge_attr], dim=-1)))
        h = h0
        for _ in range(self.depth - 1):
            a = scatter_sum(h, dst, n_atoms)         # a_v = sum_{k in N(v)} h_kv
            m = a[src] - h[rev]                      # m_vw = a_v - h_wv
            h = self.act(h0 + self.W_m(m))
            h = self.dropout(h)

        m_v = scatter_sum(h, dst, n_atoms)           # m_v = sum_{w in N(v)} h_wv^T
        h_atom = self.act(self.W_a(torch.cat([x, m_v], dim=-1)))
        return self.dropout(h_atom)


class MoleculeModel(nn.Module):
    def __init__(self, atom_dim, edge_dim, num_tasks, task_type):
        super().__init__()
        self.num_tasks, self.task_type = num_tasks, task_type
        hidden_dim, depth = 300, 3
        dropout = 0.0
        self.encoder = DMPNNEncoder(atom_dim, edge_dim, hidden_dim, depth, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_tasks))

    def forward(self, batch):
        h_atom = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        n_mol = int(batch.batch_idx.max().item()) + 1
        h = scatter_sum(h_atom, batch.batch_idx, n_mol)              # h = sum_v h_v
        return self.head(h)                                          # f(h) -> [B, num_tasks]
```

With fixed molecule-level descriptors, the extension is local: normalize the descriptor vector
to `h_f`, concatenate `torch.cat([h, h_f], dim=-1)`, and make the first head layer accept
`hidden_dim + feature_dim` inputs.
