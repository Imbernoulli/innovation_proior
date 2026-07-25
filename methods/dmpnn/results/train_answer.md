The problem is to predict chemical properties of a molecule, such as solubility, toxicity, blood-brain-barrier penetration, or enzyme inhibition, directly from its structure. A molecule is naturally a graph: atoms are nodes with RDKit-computed features including atomic number, degree, charge, chirality, hybridization, and aromaticity, while bonds are edges with features for bond type, conjugation, ring membership, and stereochemistry. Two families of methods compete here. One computes a fixed fingerprint or descriptor vector, such as Morgan fingerprints or RDKit descriptors, and runs a standard classifier or random forest on top. These carry a strong general chemical prior, but they cannot specialize to the particular property. The other family uses graph neural networks to learn the representation end to end. They can specialize, but the dominant atom-centered message passing networks aggregate over all neighbors at every step, so a message sent from atom v to atom w is immediately sent back from w to v on the next step. These back-and-forth totters pollute the representation with echoes of information the node already had, and because the receptive field is small the learned encoding is local while many properties depend on global structure. On small datasets these learned models also overfit and fall behind fixed descriptors.

The right fix is D-MPNN, the Directed Message Passing Neural Network. Instead of storing one undirected state on each atom, it stores a separate hidden state on each directed bond v to w, distinct from the reverse w to v. This makes the message from v to w a first-class object, so the message into edge v to w can be built from all other incoming edges at v while excluding the reverse edge w to v. That exclusion is the belief-propagation exclusion: a node never tells a neighbor what that neighbor just told it. The directed-bond update is the loopy-belief-propagation embedding of the graph, and it kills the totter by construction. The update is made efficient by computing the sum of all incoming messages at each atom once, then subtracting the single reverse bond; with bonds stored as adjacent forward-reverse pairs the reverse index is just the current index XOR 1, so the exclusion costs one subtraction per bond.

Concretely, each directed bond is initialized by concatenating the source atom features and bond features and passing them through a learned matrix with a ReLU, giving the bond's step-0 state. Message passing then repeats for the remaining depth steps with a shared update matrix and a residual skip connection back to that initial state, so the raw bond identity never washes out and depth remains cheap; at every step the incoming messages are aggregated once per atom and the single reverse bond is subtracted off, which is exactly the belief-propagation exclusion described above. After the final step the directed bond states are summed into their destination atoms, atom features are re-injected through a second learned matrix, and the atoms are summed into a single permutation-invariant molecule vector, which a small feed-forward head maps to one prediction per task. Because the message-passed representation is still local and because small datasets offer too little data to learn a strong prior, this same architecture can be extended by concatenating fixed RDKit 2D molecular descriptors, including molecular weight, logP, polar surface area, and ring counts, onto the molecule vector before the head; each descriptor is mapped through its own running empirical CDF so features on wildly different scales all read as a percentile, robust to the outliers and non-normal, count-like statistics that raw chemical descriptors have. That extension only widens the head's input and leaves the message-passing core untouched.

The code below implements that message-passing core and the prediction model in PyTorch, using scatter-sum aggregation to handle batched molecular graphs. It checks that bonds are stored as adjacent forward/reverse pairs so the constant-time `e XOR 1` reverse lookup is valid, and it falls back to a plain atom read-out when a graph has no bonds at all.

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
