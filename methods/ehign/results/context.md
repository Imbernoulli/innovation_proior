# Context: structure-based protein-ligand binding affinity prediction (circa 2022-2023)

## Research question

Given the 3D structure of a protein-ligand complex, predict the binding affinity — reported as
`-logKd/Ki` (the negative log of the dissociation/inhibition constant, so larger means tighter
binding). This is the scoring step at the heart of structure-based drug design: rank candidate
molecules by how strongly they bind a target pocket. The input is a set of atoms with 3D
coordinates, partitioned into a ligand (a small drug-like molecule, tens of atoms) and a protein
*pocket* (the residues lining the binding site, cropped to a shell around the ligand). The output
is a single scalar.

The hard part is not fitting the training set; many models do that. The hard part is
*generalization* to complexes unlike those seen in training, and *interpretability* — a score a
medicinal chemist can trust because it reflects the actual physics of binding rather than a
dataset shortcut. Binding affinity is, physically, a free energy that arises from non-covalent
contacts across the protein-ligand interface: hydrogen bonds, salt bridges, hydrophobic packing,
pi-stacking, each between specific atom pairs at specific distances and geometries. A model whose
internal computation mirrors that physics should both generalize better and expose *which contacts*
drive its prediction. The question is what assumptions to bake into a graph neural network so that
the function class it can represent is restricted to ones consistent with binding physics, while
still being trainable end-to-end from structures and affinity labels.

Concretely a solution has to: (1) represent both the internal chemistry of the ligand and pocket and
the short-range contacts across the interface, because those relations play different physical
roles; (2) be invariant to rigid motions of the complex (translating or rotating the whole complex
must not change the predicted affinity, since binding strength is a physical property of the
arrangement, not of the coordinate frame), without paying for that invariance with brute-force data
augmentation; (3) use enough 3D geometry to distinguish contacts with the same distance but different
local shape; and (4) leave room for chemically meaningful attribution instead of hiding all interface
evidence inside one opaque graph-level statistic.

## Background

Affinity is set by the non-covalent interface. When a ligand binds, the change in free energy is
dominated by the network of weak contacts it forms with pocket atoms — directional (hydrogen
bonds, with a preferred donor-acceptor angle), distance-dependent (van der Waals, falling off
sharply with separation), and shape-complementary (hydrophobic burial). Each such contact is a
short-range interaction between a specific ligand atom and a specific pocket atom, typically within
a few angstroms. Covalent bonds, by contrast, are internal to each molecule and set its
*conformation* — they don't cross the interface and don't directly contribute binding energy, but
they determine the geometry that positions the contacting atoms. So a complex carries two distinct
relational structures: a covalent intramolecular graph inside the ligand and inside the pocket, and
a non-covalent intermolecular graph across the interface.

Graphs as the natural representation. A molecule maps cleanly to a graph: atoms are nodes, bonds
are edges. Atom nodes carry chemical features — element, degree, implicit valence, hybridization
(SP/SP2/SP3/SP3D/SP3D2), aromaticity, attached-hydrogen count — encoded as one-hot vectors
(35 dimensions total in the standard featurization here). Covalent edges carry bond type
(single/double/triple/aromatic), conjugation, and ring membership. A protein-ligand complex extends
this to a *heterogeneous* graph with two node types (ligand atoms, pocket atoms) and edges of more
than one relation: covalent within each molecule, and non-covalent across the interface, the latter
created between every ligand-pocket atom pair closer than a cutoff (5 angstroms is the standard
contact shell).

Invariance is a free, physics-mandated inductive bias. Binding affinity does not change if you
translate or rotate the whole complex. A model that uses raw 3D coordinates as input is *not*
invariant — it would have to learn invariance from data, wasting capacity and demanding rotational
augmentation. The clean alternative, well established by this time, is to feed the network only
quantities that are themselves invariant to rigid motion: interatomic distances, bond/contact
angles, and areas of the triangles atoms span. These are computable from coordinates but unchanged
by any rotation or translation, so any model built on them is invariant by construction. A standard
geometric featurization of an edge `i->j` collects, over the neighbors `k` of `j`, the angle
`angle(g_j - g_i, g_k - g_i)`, the triangle area `0.5||(g_j-g_i) x (g_k-g_i)||`, and the distance
`||g_i - g_k||`, summarized by their max/sum/mean, plus the direct `i-j` distance in both L1 and L2
norm — eleven invariant geometric numbers per edge (each scaled by a small constant, ~0.1, to keep
magnitudes in a stable numeric range). For a covalent edge these eleven are concatenated with the
six bond-chemistry features (17 dimensions); a non-covalent contact has no covalent bond type, so it
carries the eleven geometric numbers alone.

The prevailing message-passing recipe. By this time graph neural networks for molecules all share a
skeleton: embed each node, then repeat a *message-passing* step — each node collects transformed
features from its neighbors (the "message"), aggregates them (sum/mean), and updates its own
representation — for a few layers, then *read out* a graph-level vector by pooling node
representations and pass it through an MLP to the scalar target. The design space is in three places:
how a message is formed (and how edge geometry enters it), how messages are aggregated, and how the
final graph readout turns node embeddings into the prediction.

Where the pooled-readout recipe is weak. The standard readout — pool all node embeddings into one
graph vector, then regress — compresses interface evidence before the scalar is produced. That makes
the prediction a black box at the interface (you cannot read off which contacts mattered) and lets
the model fit affinity through whatever node-embedding statistics happen to correlate with the label
on the training distribution, rather than through the contact physics — exactly the kind of dataset
shortcut that hurts out-of-distribution generalization.

## Baselines

These are the structure-based scoring models a new method is measured against and reacts to.

**SchNet (Schutt et al., NeurIPS 2017; arXiv:1706.08566).** A continuous-filter convolutional
network for atomistic systems. The complex is a single homogeneous graph (edges by a radius cutoff);
each atom embedding is updated by interaction blocks of the form
`x'_i = sum_{j in N(i)} x_j ⊙ h(RBF(d_ij))`, where `RBF(d)` expands the scalar interatomic
distance `d_ij` onto a bank of Gaussians and `h` is an MLP producing a continuous filter, modulated
by a smooth cosine cutoff so contributions vanish at the cutoff radius. Readout is a sum over atom
embeddings followed by an MLP. **Gaps:** the only geometric input is the scalar distance — no angles,
no triangle areas, no directionality of a contact; the graph is homogeneous, so covalent and
non-covalent interactions are processed by the same filter rather than treated as the chemically
distinct relations they are; and the readout pools atoms into one vector, so contact-level evidence
is only indirect.

**EGNN (Satorras, Hoogeboom & Welling, ICML 2021; arXiv:2102.09844).** An E(n)-equivariant
message-passing network. Each message is an MLP of the two endpoint features and the squared
distance `radial = ||x_i - x_j||^2` (a rotation/translation invariant); node features and, in the
full model, coordinates are updated equivariantly. **Gaps:** as with SchNet the geometry entering a
message is a single scalar (the squared distance) — richer invariants like contact angle and
triangle area are unused; all edges share one convolution rather than separating covalent from
non-covalent; and the affinity is read off a pooled node embedding, leaving interface attribution
implicit.

**GIGN (Yang, Zhong, Lv, Dong, Chen; J. Phys. Chem. Lett. 2023, 14(8):2020-2033;
DOI 10.1021/acs.jpclett.2c03906).** The direct predecessor in this line, and the one that first
separates the two interaction types inside message passing. GIGN keeps the complex as one node set
with two edge index sets — intramolecular (covalent) and intermolecular (non-covalent) — and a
*heterogeneous interaction layer* runs two parallel message passes over the same nodes:
`radial = phi(RBF(||g_row - g_col||))` is a learned SiLU-MLP of the RBF-expanded distance, the
message is `x_j ⊙ radial`, aggregated by sum into `agg_intra` over covalent edges and `agg_inter`
over non-covalent edges, and the node update combines them through two separate MLPs,
`x'_i = mlp_cov(x_i + agg_intra_i) + mlp_ncov(x_i + agg_inter_i)`. Three such layers, then
`global_add_pool` over node embeddings and an MLP to the scalar. Because every message depends only
on the interatomic distance (an invariant), the whole network is translation- and rotation-invariant
with no augmentation, and separating the two interaction types lets the model treat the binding-
relevant non-covalent edges differently from the conformation-setting covalent ones.
**Gaps it leaves open:** (1) the only geometry a message sees is the scalar distance through the RBF;
the angle, triangle-area, and multi-neighbor distance invariants computed for each edge never enter
the convolution. (2) The readout is `global_add_pool` of node embeddings followed by an MLP, so the
scalar prediction is separated from individual non-covalent contacts by a pooled graph vector; the
model is free to fit affinity from pooled node statistics, and contact-level attribution is only
available post-hoc, by visualizing embeddings. (3) The interface evidence is compressed into one
graph-level summary, hiding how ligand-side and pocket-side evidence each contribute to the score.

## Evaluation settings

The natural yardsticks in this field are:

- **Training data:** PDBbind (the general + refined sets), a curated collection of protein-ligand
  complexes each with a measured `-logKd/Ki`. Complexes are prepared by cropping the protein to the
  residues within a fixed shell (5 angstroms) of the ligand, giving a *pocket*; ligand and pocket
  are parsed with RDKit, 3D coordinates retained.
- **Test benchmarks:** the **CASF-2013 core set** (107 complexes) and **CASF-2016 core set** (285
  complexes), the community comparative-assessment-of-scoring-functions benchmarks, plus a temporal
  **2019 holdout** (several thousand complexes deposited after the training cutoff) to probe
  generalization to genuinely newer structures.
- **Metrics:** root-mean-square error (**RMSE**) between predicted and measured `-logKd/Ki` (lower
  better), and the **Pearson correlation Rp** between prediction and truth (higher better) — the
  ranking quality that matters for virtual screening.
- **Protocol:** regression with mean-squared-error loss; train on PDBbind, report on the three test
  sets; multiple independent runs for mean/std. Graph construction, feature extraction, splits,
  optimizer (Adam, small learning rate ~1e-4 with light weight decay), and the evaluation harness
  are all fixed; only the model architecture varies.

## Code framework

The model plugs into a fixed pipeline that already turns each complex into a batched graph object
(`PLABatch`): ligand atom features `[*, 35]` with a covalent edge list and 17-dim covalent edge
features; the same for the pocket; two intermolecular contact edge lists (ligand->pocket and
pocket->ligand) each with 11-dim geometric edge features; and per-graph batch assignment vectors. The
data pipeline, optimizer, regression label, train/test splits, and metric harness are given. What is
missing is the architecture mapping the batched complex to a scalar affinity per complex.

The scaffold below exposes only generic pieces: linear projections of node and edge features into a
hidden width, an empty message-passing slot, an empty readout slot, and a default regression loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AffinityModel(nn.Module):
    """Map a batched protein-ligand complex (PLABatch) to a scalar affinity per
    complex. Project node/edge features to a hidden width, run some message
    passing over the heterogeneous graph (covalent intra-molecular edges within
    ligand and pocket; non-covalent inter-molecular edges across the interface),
    then read out one scalar per complex. The message passing and the readout are
    what we design."""

    def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
        super().__init__()
        H = 256
        # project the given features into a common hidden width
        self.lin_node_l = nn.Linear(lig_dim, H)
        self.lin_node_p = nn.Linear(poc_dim, H)
        self.lin_edge_ll = nn.Linear(intra_edge_dim, H)   # ligand covalent edges
        self.lin_edge_pp = nn.Linear(intra_edge_dim, H)   # pocket covalent edges
        self.lin_edge_lp = nn.Linear(inter_edge_dim, H)   # ligand -> pocket contacts
        self.lin_edge_pl = nn.Linear(inter_edge_dim, H)   # pocket -> ligand contacts
        # TODO: the architecture we will design

    def _message_passing(self, lig_h, poc_h, lig_e, poc_e, lp_e, pl_e, batch):
        # TODO: update ligand/pocket atom features using the graph structure.
        raise NotImplementedError

    def _readout(self, lig_h, poc_h, lp_e, pl_e, batch):
        # TODO: turn message-passed features into one scalar affinity per complex.
        raise NotImplementedError

    def forward(self, batch):
        lig_h = self.lin_node_l(batch.lig_x)
        poc_h = self.lin_node_p(batch.poc_x)
        lig_e = self.lin_edge_ll(batch.lig_edge_attr)
        poc_e = self.lin_edge_pp(batch.poc_edge_attr)
        lp_e = self.lin_edge_lp(batch.l2p_edge_attr)
        pl_e = self.lin_edge_pl(batch.p2l_edge_attr)
        lig_h, poc_h = self._message_passing(lig_h, poc_h, lig_e, poc_e, lp_e, pl_e, batch)
        return self._readout(lig_h, poc_h, lp_e, pl_e, batch)         # [B]

    def compute_loss(self, batch):
        pred = self(batch)
        return F.mse_loss(pred, batch.labels)


# fixed training loop the model plugs into
def train(model, data_loader, optimizer):
    for batch in data_loader:
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
```
