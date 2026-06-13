# Context: structure-based protein-ligand binding affinity prediction (circa 2021-2022)

## Research question

The task is to predict the binding affinity of a protein-ligand complex — the strength with
which a small-molecule drug candidate (the ligand) sticks to its target protein — directly from
the 3D structure of the bound complex. Affinity is reported as `-logKd/Ki`: larger means tighter
binding. This is a central quantity in structure-based drug discovery, because the same docking
or co-folding pipeline that proposes a binding pose needs a fast, accurate scoring function to
rank thousands of candidate poses without running an experimental assay.

Physically, what sets the affinity is the set of *interactions* in the complex. There are two
chemically distinct kinds. Within each molecule, atoms are held together by covalent bonds — the
ligand's own bonded skeleton, and the bonded backbone and side chains of the protein near the
binding site. Across the interface, the ligand and the protein touch through *noncovalent*
interactions — hydrogen bonds, van der Waals contacts, electrostatics, hydrophobic packing —
which are precisely the interactions that the bound state adds relative to the two separated
molecules, and therefore the ones that decide how favorable binding is. Both kinds are
determined by interatomic geometry: bond lengths and angles for the covalent skeleton, and the
distances between ligand atoms and nearby protein atoms for the interface.

A model that predicts affinity from a complex must therefore (1) consume the actual 3D
coordinates, not just the 2D molecular topology or a sequence, since the noncovalent interface
is a spatial object; and (2) respect a hard physical symmetry: the affinity of a complex is
completely unchanged if the whole complex is rigidly translated or rotated in space. The bound
energy does not depend on where the complex sits or which way it points. A predictor that is not
invariant to these rigid motions is predicting something physically meaningless, and in practice
must either learn the symmetry from exhaustive rotational data augmentation or accept that it
will give different answers for the same complex viewed from different angles. The precise
problem is to build a network on the 3D complex that is exactly invariant to translations and
rotations by construction and that learns from the geometry of both the covalent and the
noncovalent interactions.

## Background

By this time the dominant tool for learning on molecules is the graph neural network, and the
standard formulation is message passing (Gilmer et al., ICML 2017). A molecule is a graph with
node embeddings `h_i` and edge attributes `a_ij`; a layer computes

```
m_ij      = phi_e(h_i, h_j, a_ij)        # edge message
m_i       = sum_{j in N(i)} m_ij          # aggregate over neighbors
h_i^{l+1} = phi_h(h_i, m_i)               # node update
```

with `phi_e`, `phi_h` small MLPs. Stacking layers lets information propagate several bonds deep,
and the symmetric sum makes the whole thing permutation equivariant — relabeling the atoms
relabels the outputs identically. This is the right substrate for chemistry, but the plain form
references only the graph topology and node/edge attributes; it has no notion of where the atoms
are in space. If one naively feeds a raw 3D coordinate in as a node feature, the network becomes
sensitive to the arbitrary global pose of the complex: rotate the input and the output changes,
which violates the physical invariance the problem demands.

The decisive geometric facts about that invariance are elementary. A rigid motion sends each
position `r_i` to `Q r_i + t`, where `t` is a translation and `Q` is an orthogonal matrix
(`Q^T Q = I`) representing a rotation or reflection. Under such a motion the *difference* of two
positions loses the translation, `(Q r_i + t) - (Q r_j + t) = Q(r_i - r_j)`, and the *distance*
loses the rotation as well,

```
||(Q r_i + t) - (Q r_j + t)|| = ||Q(r_i - r_j)|| = sqrt((r_i - r_j)^T Q^T Q (r_i - r_j))
                              = ||r_i - r_j||,
```

because `Q^T Q = I`. So the pairwise interatomic distance is the natural rotation- and
translation-invariant geometric quantity, and any function built only out of distances is
automatically invariant. An absolute coordinate, or a single difference vector treated as a
feature, is *not* invariant and changes under the group.

There is also a known empirical pitfall in feeding geometry to a graph network, observed for
distance-conditioned message passing: when a small filter network maps the single scalar
distance into many feature channels, a freshly initialized network is nearly linear, so all of
those channels come out as nearly the same near-linear function of the distance — they are
highly correlated, and training stalls on a plateau early on. The fix that circulates is to
expand the scalar distance in a fixed basis (a bank of Gaussian radial functions on a grid of
centers) *before* the filter network, which decorrelates the channels so different channels
respond to different distance ranges and the plateau disappears.

Two structural facts about this particular task shape any model. First, the complex is
*heterogeneous*: it is two distinct molecules with different chemistry, joined by an interface
whose edges (noncovalent contacts, formed only because the two molecules are close in the bound
pose) are chemically different from the covalent bonds inside each molecule, living at different
distance ranges (bonds are short and stiff, around 1.5 angstroms; interface contacts are softer
and reach out to roughly 5 angstroms). Second, the interface is what binding *adds*: the
covalent skeletons exist whether or not the molecules are bound, but the noncovalent contacts
are the new physics of the bound state, so they carry the affinity signal most directly. The
binding pocket is conventionally taken as the protein residues within about 5 angstroms of the
ligand, and the interface graph as the ligand-atom / pocket-atom pairs within that same cutoff.

## Baselines

These are the prior methods a structure-based affinity predictor would be measured against and
would react to.

**Descriptor and 2D-topology scoring (e.g. RF-Score, Ballester & Mitchell 2010).** RF-Score
trains a random forest on counts of element-pair contacts within a distance cutoff; other
predictors run a 2D GNN on the molecular bond graph or a sequence model on the protein.
*Gap:* these either discard the 3D structure of the interface entirely (sequence / 2D-graph
models) or compress it into coarse hand-binned contact counts (RF-Score); the fine spatial
arrangement of the noncovalent interactions that physically sets the affinity is not represented,
so the model cannot distinguish poses that differ in interface geometry but agree on coarse
counts.

**MPNN (Gilmer et al., ICML 2017).** The message-passing substrate above, designed for quantum
chemistry on single molecules. *Gap:* it has no spatial geometry at all. It is invariant to
nothing in 3D; feeding coordinates as features makes it pose-dependent, so it either fails the
rigid-motion invariance the physics requires or must learn it from augmentation. It also has no
mechanism that distinguishes one kind of edge from another beyond a generic edge attribute.

**SchNet (Schütt et al., NeurIPS 2017).** Make every message depend only on the invariant
interatomic distance. Distances are expanded in a Gaussian radial basis
`e_k(d_ij) = exp(-gamma (d_ij - mu_k)^2)` over centers `mu_k`, and a continuous-filter
convolution multiplies each neighbor's features by a filter generated from that expansion,
`x_i^{l+1} = sum_j x_j^l (*) W(d_ij)` with `(*)` an elementwise product, followed by atom-wise
dense layers. Because only the distance enters, every layer is `E(3)`-invariant, smooth, and
cheap. *Gap:* SchNet was built for a single homogeneous molecule, where there is effectively one
kind of edge. Every neighbor is processed by the same distance filter and the same update,
regardless of whether the bond between two atoms is a stiff intramolecular bond or a soft
interfacial contact between two different molecules — the network has no channel that treats
those two physically different interaction types differently.

**EGNN (Satorras, Hoogeboom & Welling, ICML 2021).** Invariant message passing that carries the
squared distance, `m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2, a_ij)`, alongside an equivariant
coordinate update `x_i^{l+1} = x_i + C sum_{j != i} (x_i - x_j) phi_x(m_ij)`. For a purely
*invariant* target the coordinate update is unnecessary — positions can be held fixed and the
model is exactly `E(n)`-invariant, a graph network on distances. It is cheap, needs no spherical
harmonics, and works in any dimension. *Gap:* like SchNet, it applies one message function to
every edge; it has no representation of the fact that the complex is two molecules whose
intramolecular bonds and intermolecular contacts are chemically distinct. The geometry is
handled uniformly across all neighbors.

**InteractionGraphNet — IGN (Jiang et al., J. Med. Chem. 2021).** The closest prior method on
the protein-ligand interface itself. IGN represents the complex as a ligand graph, a protein
graph, and a bimolecular protein-ligand graph, and learns interactions in two independent graph
convolution modules stacked in sequence: first a module that learns the intramolecular (covalent)
interactions, then a separate module that learns the intermolecular (noncovalent) interactions,
and finally a readout that produces the graph-level representation. This explicitly recognizes
that covalent and noncovalent interactions are different. *Gap:* the two interaction types are
processed in *separate modules run one after another*, so within any single message-passing step
a node sees only one interaction type; the covalent context and the noncovalent context are never
integrated in the same step, and the sequential separation adds stages and parameters. An atom
finishes absorbing its bonded environment before it ever sees its interfacial contacts, rather
than weighing both together as it updates.

The shape of the landscape: distance-only invariant networks (SchNet, EGNN) give the right
`E(3)` symmetry cheaply and from geometry, but treat the complex as homogeneous and process every
edge identically; the one method that takes the covalent/noncovalent distinction seriously (IGN)
does so by separating it into sequential modules that never exchange the two kinds of information
within a step, and which does not feed the continuous interaction geometry into the message in
the distance-invariant way the geometric networks do. No single method both respects the
geometry-driven `E(3)` invariance and lets a node integrate its covalent and noncovalent
environment together.

## Evaluation settings

The natural yardsticks already in use for structure-based affinity prediction, and the protocols
around them:

- **PDBbind** (Wang et al.; Liu et al.) is the standard training corpus: experimentally
  determined protein-ligand complex structures with measured binding constants `-logKd/Ki`. The
  general and refined sets together provide on the order of tens of thousands of complexes. Each
  complex supplies the ligand (mol2/sdf), the protein (pdb), and the affinity label.
- The standard held-out **core sets** are the CASF benchmarks: the **PDBbind 2013 core set**
  (107 complexes) and the **PDBbind 2016 core set** (285 complexes), curated, diverse,
  high-quality complexes used as external test sets. A **temporal holdout** (complexes deposited
  after a cutoff year) tests generalization to genuinely unseen targets.
- **Inputs** per complex: atom features for ligand and pocket atoms (one-hot element, degree,
  implicit valence, hybridization, aromaticity, hydrogen count — on the order of 35 dimensions),
  the intramolecular bond graph from the chemical structure, the intermolecular contact graph
  from a 5-angstrom distance threshold between ligand and pocket atoms, and the 3D coordinates.
- **Metrics**: root-mean-square error (RMSE, lower is better) of the predicted `-logKd/Ki`, and
  the Pearson correlation coefficient `Rp` between predicted and measured affinities (higher is
  better), reported on each external test set, typically averaged over several independent
  training runs.
- **Protocol**: train on PDBbind general+refined, validate on a held-out split, test on the
  external core sets and temporal holdout; minibatch training of the whole network end to end.

## Code framework

The open design slot sits inside the standard message-passing harness for molecular graphs.
What exists beforehand is the data pipeline that turns a complex into a graph, a node-feature
embedding, a stack of graph layers, a pooling readout, a regression head, and the training loop.
A complex datum supplies per-atom node features `x`, two edge sets (the intramolecular bond
graph and the intermolecular contact graph), the 3D atom coordinates `pos`, a batch assignment,
and the affinity label. The unsettled part is the per-layer transformation that turns the node
features, the two edge sets, and the geometry into updated node features.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool


class InteractionLayer(MessagePassing):
    """One message-passing layer over a protein-ligand complex. Consumes per-node
    features x, the intramolecular (bond) edge set, the intermolecular (contact)
    edge set, and per-node 3D coordinates pos. Geometry must enter only through
    rigid-motion-invariant quantities."""

    def __init__(self, in_channels, out_channels, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO: the per-layer transformation we will design.

    def forward(self, x, edge_index_intra, edge_index_inter, pos=None, size=None):
        # x:                 (N, in_channels)  per-atom features
        # edge_index_intra:  (2, E_intra)      intramolecular (covalent) edges
        # edge_index_inter:  (2, E_inter)      intermolecular (noncovalent) edges
        # pos:               (N, 3)            atom coordinates
        raise NotImplementedError  # TODO

    def message(self, x_j, x_i, edge_signal, index):
        raise NotImplementedError  # TODO


def encode_pair_geometry(pos, edge_index, device="cpu"):
    """Turn the coordinates attached to an edge set into an invariant edge signal."""
    raise NotImplementedError  # TODO


class AffinityModel(nn.Module):
    """Embed atom features, apply a stack of interaction layers over the complex,
    pool, and regress the affinity. The per-layer interaction rule is the open slot."""

    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.lin_node = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
        self.convs = nn.ModuleList(
            [InteractionLayer(hidden_dim, hidden_dim) for _ in range(3)]
        )
        self.fc = RegressionHead(hidden_dim, hidden_dim, 1)

    def forward(self, data):
        x = self.lin_node(data.x)
        for conv in self.convs:
            x = conv(x, data.edge_index_intra, data.edge_index_inter, data.pos)
        x = global_add_pool(x, data.batch)
        return self.fc(x).view(-1)


class RegressionHead(nn.Module):
    """Map a pooled complex representation to one affinity value."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        # TODO: choose the depth, activations, normalization, and regularization.

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


# existing training loop the model plugs into
def train(model, loss_fn, data_loader, optimizer):
    for data in data_loader:                  # minibatch of complexes
        optimizer.zero_grad()
        pred = model(data)                    # forward through the interaction stack
        loss = loss_fn(pred, data.y)          # MSE regression on -logKd/Ki
        loss.backward()
        optimizer.step()
```

The data pipeline, the node embedding, the pooling, the regression head, and the outer loop are
fixed; `InteractionLayer` is the single empty slot where the per-layer interaction rule will
live.
