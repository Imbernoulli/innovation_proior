The problem is to predict chemical properties of a molecule, such as solubility, toxicity, blood-brain-barrier penetration, or enzyme inhibition, directly from its structure. A molecule is naturally a graph: atoms are nodes with RDKit-computed features including atomic number, degree, charge, chirality, hybridization, and aromaticity, while bonds are edges with features for bond type, conjugation, ring membership, and stereochemistry. Two families of methods compete here. One computes a fixed fingerprint or descriptor vector, such as Morgan fingerprints or RDKit descriptors, and runs a standard classifier or random forest on top. These carry a strong general chemical prior, but they cannot specialize to the particular property. The other family uses graph neural networks to learn the representation end to end. They can specialize, but the dominant atom-centered message passing networks aggregate over all neighbors at every step, so a message sent from atom v to atom w is immediately sent back from w to v on the next step. These back-and-forth totters pollute the representation with echoes of information the node already had, and because the receptive field is small the learned encoding is local while many properties depend on global structure. On small datasets these learned models also overfit and fall behind fixed descriptors.

The right fix is D-MPNN, the Directed Message Passing Neural Network. Instead of storing one undirected state on each atom, it stores a separate hidden state on each directed bond v to w, distinct from the reverse w to v. This makes the message from v to w a first-class object, so the message into edge v to w can be built from all other incoming edges at v while excluding the reverse edge w to v. That exclusion is the belief-propagation exclusion: a node never tells a neighbor what that neighbor just told it. The directed-bond update is the loopy-belief-propagation embedding of the graph, and it kills the totter by construction. The update is made efficient by computing the sum of all incoming messages at each atom once, then subtracting the single reverse bond; with bonds stored as adjacent forward-reverse pairs the reverse index is just the current index XOR 1, so the exclusion costs one subtraction per bond.

Concretely, each directed bond is initialized by concatenating the source atom features and bond features and passing them through a learned matrix with a ReLU. Message passing then repeats for T steps with a shared update matrix and a residual skip connection back to the initial bond state, so the raw bond identity never washes out and depth remains cheap. After T steps the directed bond states are summed into their destination atoms, atom features are re-injected, and the atoms are summed into a single permutation-invariant molecule vector. Because the message-passed representation is still local and because small datasets offer too little data to learn a strong prior, the model also concatenates normalized RDKit 2D molecular descriptors before the final feed-forward head. Those descriptors provide global physicochemical knowledge, including molecular weight, logP, polar surface area, and ring counts, that the graph network cannot reach in a few hops and could not learn from a few hundred molecules. Each descriptor is standardized with running mean and variance so features on wildly different scales contribute evenly. The head is a small two-layer network that maps the combined vector to one prediction per task.

The code below implements the encoder and model in PyTorch, using scatter-sum aggregation to handle batched molecular graphs. The descriptor branch falls back to zeros when SMILES strings are not supplied, so the same encoder works in pure graph mode as well.

```python
import math
import torch
import torch.nn as nn
from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit.Chem import rdMolDescriptors


def scatter_sum(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def _rdkit_2d_descriptors(smi):
    if not smi:
        return [0.0] * 17
    mol = MolFromSmiles(smi)
    if mol is None:
        return [0.0] * 17
    feats = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumHeteroatoms(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        Descriptors.MolMR(mol),
        Descriptors.LabuteASA(mol),
    ]
    out = []
    for v in feats:
        try:
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                v = 0.0
        except Exception:
            v = 0.0
        out.append(v)
    return out


class _RunningNormalizer(nn.Module):
    def __init__(self, dim, momentum=0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_std', torch.ones(dim))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * x.mean(dim=0))
                self.running_std.mul_(1 - self.momentum).add_(self.momentum * x.std(dim=0).clamp(min=1e-6))
        return (x - self.running_mean) / self.running_std.clamp(min=1e-6)


class DMPNNEncoder(nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden_dim=300, depth=3, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.W_i = nn.Linear(atom_dim + edge_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        n_atoms, n_bonds = x.size(0), edge_index.size(1)
        if n_bonds == 0:
            h_atom = self.act(self.W_o(torch.cat([
                x, torch.zeros(n_atoms, self.hidden_dim, device=x.device, dtype=x.dtype)
            ], dim=-1)))
            return self.dropout(h_atom)

        rev = torch.arange(n_bonds, device=x.device) ^ 1
        h0 = self.act(self.W_i(torch.cat([x[src], edge_attr], dim=-1)))
        h = h0
        for _ in range(self.depth - 1):
            a = scatter_sum(h, dst, n_atoms)
            m = a[src] - h[rev]
            h = self.act(h0 + self.W_h(m))
            h = self.dropout(h)

        m_v = scatter_sum(h, dst, n_atoms)
        h_atom = self.act(self.W_o(torch.cat([x, m_v], dim=-1)))
        return self.dropout(h_atom)


class MoleculeModel(nn.Module):
    def __init__(self, atom_dim, edge_dim, num_tasks, task_type):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        hidden_dim, depth, dropout = 300, 3, 0.0
        self.encoder = DMPNNEncoder(atom_dim, edge_dim, hidden_dim, depth, dropout)
        self.feat_norm = _RunningNormalizer(17)
        self._smi_cache = {}
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + 17, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def _batch_rdkit_features(self, batch):
        smiles = getattr(batch, '_smiles', None)
        if smiles is None:
            n_mol = int(batch.batch_idx.max().item()) + 1
            return torch.zeros(n_mol, 17, device=batch.x.device)
        feats = []
        for smi in smiles:
            if smi not in self._smi_cache:
                self._smi_cache[smi] = _rdkit_2d_descriptors(smi)
            feats.append(self._smi_cache[smi])
        return torch.tensor(feats, dtype=torch.float32, device=batch.x.device)

    def forward(self, batch):
        h_atom = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        n_mol = int(batch.batch_idx.max().item()) + 1
        h_graph = scatter_sum(h_atom, batch.batch_idx, n_mol)
        rdkit_feats = self.feat_norm(self._batch_rdkit_features(batch))
        return self.readout(torch.cat([h_graph, rdkit_feats], dim=-1))
```
