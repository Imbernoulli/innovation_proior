The starter GIN told me what is missing, and it told me in numbers: BBBP came in at **0.5099** — chance, a coin flip — while BACE held at **0.7261** and Tox21 at **0.7470**, both clearly above it. The encoder can learn *something*; it is not broken across the board. The split is sharp and informative. The two tasks where local substructure carries the label (an enzyme-inhibition pocket in BACE, twelve toxicity assays in Tox21 with enough molecules that multi-task averaging stabilizes the score) survive a weak local encoder; the single binary target decided by *global* whole-molecule physicochemistry — does this thing cross the blood-brain barrier, mostly a story of lipophilicity, polar surface area, size — collapses to a coin flip on the scaffold split, where it had nothing global to hold onto and nothing to memorize. So this is two structural problems at once, both readable straight off the GIN design: the readout *mean*-pools and throws away the count information sum keeps, and the four-hop receptive field with no external prior cannot see the global property BBBP turns on.

There is a third flaw, deeper than the readout, in the GIN-style update itself. The starter aggregates, for each atom, a sum over *all* its neighbors — including, at the next step, the atom it just sent a message to. Trace one edge between atoms 1 and 2: at step $t$ atom 2's message sums over its neighbors, which include atom 1, so $h_2^{t+1}$ contains a piece of $h_1$; next step atom 1 sums over its neighbors, which include atom 2, so it pulls back $h_2^{t+1}$ — the piece of $h_1$ it just sent. The message bounced right back along the same bond. These out-and-immediately-back walks ($v\to w\to v$, the pattern $v_i = v_{i+2}$) are *totters*: they carry no new information, they re-mix what the atom already had, and they do it on every edge every step, steadily polluting the representation with echoes of itself. That is the structural disease of *atom-centered* message passing.

I propose **D-MPNN**, a directed message-passing network with an RDKit descriptor branch. The cure for the totter I half-recognize from belief propagation: the message from $v$ to a neighbor $w$ should be built from everything $v$ has heard *except* what came from $w$ — that exclusion is exactly why BP does not double-count. The atom-centered update violates it because a single undirected atom vector is direction-blind; the state is in the wrong place. So move it onto **directed bonds**, keeping $h_{vw}$ and $h_{wv}$ distinct. Now "the message heading from $v$ out to $w$" is a first-class object, built from the *other* directed bonds flowing into $v$, skipping the reverse: $m_{vw} = \sum_{k\in N(v)\setminus w} h_{kv}$. The reverse message $h_{wv}$ drops out by construction — exactly the term that caused the totter. This is the directed-edge (loopy-BP) embedding where the atom-centered GIN was the mean-field one.

Made concrete: initialize each directed bond from the atom it leaves and the bond's own features, $h_{vw}^0 = \mathrm{ReLU}(W_i\,[x_v ; e_{vw}])$, mixing source atom and bond inside one matrix so their interaction survives (a double bond *in a ring* differs from one *not* in a ring). The message function is trivial — the message along $k\to v$ is just $h_{kv}$ — and all the learning sits in a tied update, $h_{vw}^{t+1} = \mathrm{ReLU}(h_{vw}^0 + W_m\,m_{vw})$, the *same* $W_m$ every round so depth is a free hyperparameter, with a skip back to $h_{vw}^0$ each step so the raw bond identity never washes out under the tied recurrence — the exact failure mode that bit GIN's deep features. After $T$ rounds I return to atoms by summing the directed bonds that *end* at each atom, $m_v = \sum_{w\in N(v)} h_{wv}$, re-inject atom features, $h_v = \mathrm{ReLU}(W_a\,[x_v ; m_v])$, and — the second fix to GIN — pool by **sum**, not mean. Sum is the injective multiset aggregator; the distribution-only mean is exactly what was insufficient on BBBP. The molecule vector $\sum_v h_v$ keeps the counts. The exclusion looks expensive but factors: $\sum_{N(v)\setminus w}$ is $\sum_{N(v)}$ minus the one excluded term, so compute $a_v = \sum_{k\in N(v)} h_{kv}$ once per atom, then each outgoing message is $a_v - h_{wv}$ — one subtraction per bond. To find the reverse instantly, bonds are stored in adjacent forward/reverse pairs so the reverse of bond $e$ is $e \oplus 1$, and the scaffold already lays `edge_index` out this way, so the directed scheme costs essentially the same as a plain atom aggregation.

That fixes the message passing, but even fixed this is still a *local, data-hungry, prior-free* encoder — $T\approx 3$ hops is smaller than a drug-molecule's diameter, and BBBP's answer lives in global physicochemistry over a few hundred training molecules. So I attack the $0.51$ directly with the cheap external source of exactly that knowledge: molecule-level **RDKit 2D descriptors**. Compute a fixed-length vector — molecular weight, LogP, topological polar surface area, H-bond donor/acceptor counts, rotatable bonds, aromatic/aliphatic ring counts, fraction of $sp^3$ carbons, heteroatom count, molar refractivity, Labute ASA — and **concatenate it to the graph vector before the head**, $\hat y = \mathrm{head}([h ; h_f])$. This is a hybrid: the learned message-passed $h$ supplies task-specific, locally resolved structure; the fixed $h_f$ supplies a global chemical prior that needs neither a large $T$ nor much data and reaches across the molecule where three-hop passing cannot — precisely the lipophilicity/size/polarity story BBBP turns on. The descriptors have wildly different scales (a molecular weight in the hundreds next to a fraction in $[0,1]$ next to an integer ring count), so they must be standardized or the large-range features drown the small. The clean version maps each through its CDF fit on a large background; in this edit surface I approximate that with a BatchNorm-style running mean/std normalizer on the descriptor branch, updated as batches arrive so train and test see the same scaling. One honest limitation I respect rather than hide: the harness only reliably exposes SMILES through `batch._smiles`, so when that attribute is absent the descriptor branch falls back to a zero vector — after the running normalizer it then contributes nothing, leaving the pure GNN branch. I keep the set deliberately compact (the seventeen above) rather than the full few-hundred CDF panel, because the compact set is the chemically load-bearing subset and is robust to compute in-loop. Depth $T=3$, hidden width 300, sum pooling, a two-layer FFN over $[h ; h_f]$, dropout near zero by default and lifted per dataset by the driver.

So the delta from rung one is concrete: where GIN summed over all neighbors (tottering) and mean-pooled (losing counts) with no global prior, I move the state onto directed bonds and exclude the reverse message (BP-style, no totter), add the tied-update-plus-skip so depth is free and bond identity survives, sum-pool to keep counts, and concatenate a running-normalized RDKit descriptor vector so the global physicochemistry BBBP needs is handed in directly. BBBP is the falsifiable claim — GIN sat at chance, and if the descriptor branch is really supplying the global prior, BBBP must move clearly off $0.5$; if it barely moves, the cure is not more 2D graph machinery at all but the *3D geometry* the property actually depends on, plus pretraining to escape the data-hunger — the next rung.

```python
# =====================================================================
# EDITABLE SECTION START — D-MPNN: Directed Message Passing Neural Network
# =====================================================================

from rdkit.Chem import Descriptors as _Descriptors
from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
from rdkit.Chem import MolFromSmiles as _MolFromSmiles


# --------------------- RDKit 2D molecular descriptors -----------------
# A compact subset of normalized RDKit 2D descriptors that have been
# shown to improve D-MPNN on physicochemical / biophysical tasks (Yang
# et al. 2019, "rdkit_2d_normalized" features generator).  We compute
# them once per SMILES and per-feature standardize using running stats
# accumulated over the training batches — a robust approximation of
# chemprop's pre-computed Welford normalization.

def _rdkit_2d_descriptors(smi):
    """Compute a fixed-length RDKit 2D descriptor vector for a SMILES."""
    if not smi:
        return [0.0] * 17
    mol = _MolFromSmiles(smi)
    if mol is None:
        return [0.0] * 17
    feats = [
        _Descriptors.MolWt(mol),
        _Descriptors.MolLogP(mol),
        _Descriptors.NumHDonors(mol),
        _Descriptors.NumHAcceptors(mol),
        _Descriptors.TPSA(mol),
        _Descriptors.NumRotatableBonds(mol),
        _Descriptors.NumAromaticRings(mol),
        _Descriptors.NumAliphaticRings(mol),
        _Descriptors.HeavyAtomCount(mol),
        _Descriptors.RingCount(mol),
        _Descriptors.FractionCSP3(mol),
        _Descriptors.NumHeteroatoms(mol),
        _rdMolDescriptors.CalcNumSaturatedRings(mol),
        _rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        _rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        _Descriptors.MolMR(mol),
        _Descriptors.LabuteASA(mol),
    ]
    # NaN / inf guard
    cleaned = []
    for v in feats:
        try:
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                v = 0.0
        except Exception:
            v = 0.0
        cleaned.append(v)
    return cleaned


_RDKIT_FEAT_DIM = 17


class _RunningNormalizer(nn.Module):
    """Running mean/std normalizer for RDKit features (BatchNorm-style)."""

    def __init__(self, dim, momentum=0.01):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_std', torch.ones(dim))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                mean = x.mean(dim=0)
                std = x.std(dim=0).clamp(min=1e-6)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_std.mul_(1 - self.momentum).add_(self.momentum * std)
        return (x - self.running_mean) / self.running_std.clamp(min=1e-6)


class DMPNNEncoder(nn.Module):
    """Directed Message Passing Neural Network (Yang et al., 2019).

    Bond-level messages flow along directed edges; each message passing step
    computes new edge messages from incoming atom messages minus the reverse
    edge contribution to avoid message collision.
    """

    def __init__(self, atom_dim, edge_dim, hidden_dim=300, depth=3, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Initial bond message: linear over [atom_src || bond_attr]
        self.W_i = nn.Linear(atom_dim + edge_dim, hidden_dim, bias=False)
        # Shared message-update weight (chemprop default)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Final atom-level readout combine
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch_idx):
        """
        x: [total_atoms, atom_dim]
        edge_index: [2, total_edges] (bidirectional, paired as [i,j],[j,i])
        edge_attr: [total_edges, edge_dim]
        batch_idx: [total_atoms]
        """
        src, dst = edge_index
        num_atoms = x.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            # Fallback for atom-only molecules
            atom_hidden = self.act(self.W_o(torch.cat([x, torch.zeros(num_atoms, self.hidden_dim, device=x.device)], dim=-1)))
            return self.dropout(atom_hidden)

        # Reverse edge index: edges are added in pairs (i->j, j->i),
        # so reverse of edge e is e XOR 1.
        rev_edge_idx = torch.arange(num_edges, device=x.device) ^ 1
        rev_edge_idx = rev_edge_idx.clamp(max=num_edges - 1)

        # Initial bond input: source atom features concatenated with bond features
        bond_input = torch.cat([x[src], edge_attr], dim=-1)
        h0 = self.act(self.W_i(bond_input))  # [num_edges, hidden]
        h = h0

        # Message passing for depth-1 steps (chemprop convention)
        for _ in range(self.depth - 1):
            # Aggregate incoming messages to each atom
            atom_msg = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
            atom_msg.index_add_(0, dst, h)

            # New edge message: a_v - h_{v->u}^{rev} (avoid passing back)
            new_h = atom_msg[src] - h[rev_edge_idx]
            new_h = self.W_h(new_h)
            # Residual on h0 (chemprop style)
            new_h = self.act(h0 + new_h)
            new_h = self.dropout(new_h)
            h = new_h

        # Final atom messages
        atom_msg = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
        atom_msg.index_add_(0, dst, h)

        # Combine atom features with aggregated bond messages
        atom_hidden = self.act(self.W_o(torch.cat([x, atom_msg], dim=-1)))
        atom_hidden = self.dropout(atom_hidden)
        return atom_hidden


class MoleculeModel(nn.Module):
    """D-MPNN with RDKit 2D normalized molecular descriptors.

    Configuration follows Yang et al. 2019 chemprop defaults:
      - hidden_dim = 300
      - depth = 3 message passing steps
      - sum readout per graph
      - 2-layer FFN head with hidden=300
      - RDKit 2D descriptors concatenated at the readout ("+features" mode)
    """

    def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int, task_type: str):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        hidden_dim = 300
        depth = 3
        # `pooler_dropout` may be set by the training driver to vary dropout
        # per dataset (e.g. BACE/Tox21=0.1, BBBP=0.0, regression tasks=0.1-0.2)
        dropout = float(getattr(type(self), "pooler_dropout", 0.0))

        self.encoder = DMPNNEncoder(
            atom_dim=atom_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )

        # RDKit 2D descriptor branch
        self.feat_norm = _RunningNormalizer(_RDKIT_FEAT_DIM)

        # 2-layer FFN head over [graph_embed || rdkit_features]
        readout_in = hidden_dim + _RDKIT_FEAT_DIM
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

        # Lazy SMILES->feature cache (shared across forward calls)
        self._smi_cache = {}

    def _batch_rdkit_features(self, batch):
        """Compute RDKit features for the molecules in this batch.

        Uses LMDB SMILES via the dataset wrapper.  When SMILES are not
        available (no `_smiles` attr), falls back to a zero vector — the
        running normalizer will then produce zeros, leaving the GNN
        branch unaffected.
        """
        smiles = getattr(batch, "_smiles", None)
        if smiles is None:
            num_graphs = int(batch.batch_idx.max().item()) + 1
            return torch.zeros(num_graphs, _RDKIT_FEAT_DIM,
                               device=batch.x.device)

        feats = []
        for smi in smiles:
            if smi in self._smi_cache:
                feats.append(self._smi_cache[smi])
            else:
                f = _rdkit_2d_descriptors(smi)
                self._smi_cache[smi] = f
                feats.append(f)
        return torch.tensor(feats, dtype=torch.float32, device=batch.x.device)

    def forward(self, batch):
        atom_hidden = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch_idx)

        # Sum pooling per graph (chemprop default)
        num_graphs = int(batch.batch_idx.max().item()) + 1
        graph_embed = torch.zeros(num_graphs, atom_hidden.size(-1), device=atom_hidden.device)
        graph_embed.index_add_(0, batch.batch_idx, atom_hidden)

        # RDKit feature branch (per-graph)
        rdkit_feats = self._batch_rdkit_features(batch)
        rdkit_feats = self.feat_norm(rdkit_feats)

        combined = torch.cat([graph_embed, rdkit_feats], dim=-1)
        return self.readout(combined)

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
```
