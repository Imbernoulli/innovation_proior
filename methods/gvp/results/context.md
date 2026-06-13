## Research question

A protein is a chain of amino-acid residues whose backbone atoms (N, Cα, C, O) sit at
definite positions in 3D Euclidean space. A growing class of problems asks to map such a
structure to some property: predict the quality of a candidate structural model, predict
whether two molecules bind in a given geometry, or predict an amino-acid sequence that
would fold into a given backbone (computational protein
design, CPD, the conceptual inverse of structure prediction). The precise CPD task: given
backbone coordinates `X` of shape `(L, 4, 3)` (the four backbone atoms per residue) and a
validity mask, output a per-residue distribution over the 20 standard amino acids, and be
scored by *native sequence recovery* (fraction of residues predicted correctly on
experimentally determined structures) and *perplexity* on held-out native sequences.

The domain has two faces. On the one hand it is **geometric**: the function depends on the
arrangement and orientation of residues in space — directions, angles, the 3D shape of a
binding pocket, where points that are not themselves atoms sit relative to the chain. On the
other hand it is **relational**: which residues are in contact, the connectivity pattern of
interactions, the sequential order along the chain. A successful architecture must reason
about *both*, and — because a protein's identity does not change when you rotate or reflect
the coordinate frame it is described in — its scalar predictions must be **invariant** to
rotations, reflections, and translations of the input coordinates. The pain point is that
the two leading families of architectures each capture only one face, and bolting them
together naively forces a choice that throws away part of the geometry. Closing that gap —
full relational expressivity together with direct geometric reasoning, at a cost that scales
to molecules with hundreds of residues — is the problem.

## Background

By this time, *learning from structure* has become an active application area for deep
learning across structural biology (Townshend et al. 2019; Ingraham et al. 2019; and the
review of Graves et al. 2020). The prevailing wisdom is that progress comes from
architectures whose inductive bias matches the domain — convolutions for the translation
structure of images (Cohen & Shashua 2017), attention for the relational structure of
language (Vaswani et al. 2017) — so the question for structure is: what is the right
inductive bias, and which existing primitive supplies it?

Three things are established and load-bearing here.

**Symmetry of the domain.** A protein's properties are unchanged under a rigid motion of the
whole molecule: any rotation/reflection `R` (a unitary `3×3` matrix) and translation applied
to all atom coordinates leaves the answer fixed. So a scalar target (a quality score, a
per-residue class) must be computed *invariantly*, and any internal geometric quantity that
is itself directional (an orientation, a bond direction) transforms *equivariantly* — it
should rotate with the molecule. Two strategies exist for honoring this. One is to reduce
every directional quantity to rotation-invariant scalars at the input (lengths, angles, dot
products measured in a per-node local coordinate frame), after which an ordinary network may
process them freely. The other is to carry the directional quantities themselves through the
network as objects in `R^3` and only ever combine them through operations that commute with
`R`.

**The message-passing template (Gilmer et al. 2017).** A graph neural network over a
proximity graph computes, at each propagation step, a message for each directed edge
`m_{j→i} = g(h_i, h_j, e_{j→i})`, aggregates the incoming messages at each node (sum or
mean), and updates the node embedding `h_i ← U(h_i, agg_j m_{j→i})`, optionally interleaved
with a per-node feed-forward update. `g` and `U` are learned. This is the substrate on which
relational reasoning over a structure graph is built; what varies between methods is what `g`
and `U` are, and what the node/edge features `h, e` contain.

**The diagnostic limitation of distance-only graph features (observed by Ingraham et al.
2019).** A node update `h_i ← f(h_i, {e_{j→i}})` can only depend on the local environment to
the extent the edge features encode it. Pairwise distances alone are *not locally
informative*: knowing `‖x_a − x_i‖` and `‖x_b − x_i‖` does not say whether neighbors `a` and
`b` lie on the same side of `i` or on opposite sides. So a graph representation that hopes to
reason about local 3D geometry must carry *more than length* on its edges — enough to
reconstruct neighbor positions up to rigid motion. This representation gap forces every
structure-graph method to make a choice about how to encode directional information on
edges.

**Standard geometric input features for a protein backbone.** From the backbone atoms one
can compute, per residue, the three dihedral angles `(φ, ψ, ω)` (functions of consecutive
`C_{i-1}, N_i, Cα_i, C_i, N_{i+1}` positions), embedded on the torus as `{sin, cos}` of each
— six rotation-invariant scalars. One can also form directional quantities: the unit vectors
toward the next and previous `Cα` along the chain, and the imputed `Cβ−Cα` direction obtained
from tetrahedral geometry, `√(2/3)(n×c)/‖n×c‖ − √(1/3)(n+c)/‖n+c‖` with `n = N_i−Cα_i`,
`c = C_i−Cα_i` (the larger weight `√(2/3)` on the out-of-plane normal `n×c`, the smaller
`√(1/3)` on the in-plane bisector `n+c` — the split that reproduces the exact tetrahedral
direction). On edges, the natural quantities are the `Cα_j−Cα_i` distance, the
`Cα_j−Cα_i` direction, and the relative orientation between residues `i` and `j`. Distances
are commonly lifted into a basis of Gaussian radial basis functions (16 of them, centers
evenly spaced 0–20 Å), and the backbone offset `j−i` is encoded with a sinusoidal positional
encoding in the manner of Vaswani et al. (2017). These are the raw material of a
structure-graph featurization.

## Baselines

**Sequential / hand-crafted-feature models.** Each residue is summarized by a hand-crafted
feature vector of its 3D environment — residue contacts, locally projected orientations,
physics-inspired energy terms — and the protein is treated as a sequence/collection of such
vectors fed to a 1D-CNN, RNN, or dense network (e.g. SPIN2, O'Connell et al. 2018; and the
quality-assessment methods VoroMQA, SBROD, ProQ3D). **Gap:** the 3D structure is represented
only indirectly, through whatever the hand-crafted features happen to capture; the geometry
the features discard is unrecoverable downstream.

**Voxelized 3D-CNN models.** Atoms are rasterized into an occupancy map on a 3D voxel grid
and processed by a 3D convolutional network, whose hierarchical filters are well suited to
detecting structural motifs, pockets, and shapes — directly leveraging the geometric face of
the domain (CPD methods of Anand et al. 2020, Zhang et al. 2019; quality-assessment 3DCNN
and Ornate). **Gap:** the convolution is over a rigid voxel grid, so the model is not
naturally invariant to rotation (one re-orients by data augmentation), the grid resolution
trades memory against geometric precision, and the representation discards the explicit
graph of residue-residue relations — relational reasoning has to be re-learned through dense
volumetric filters.

**Graph neural networks with invariant scalar geometry (Structured Transformer / Structured
GNN, Ingraham et al. 2019).** The state of the art for CPD. The protein is a `k`-nearest-
neighbor graph over residues (`k = 30` by `Cα` distance), and CPD is framed as autoregressive
language modeling conditioned on the structure: `p(s | x) = ∏_i p(s_i | x, s_{<i})`. An
*encoder* of stacked self-attention/message-passing layers builds sequence-independent
per-residue embeddings from structural features; a *decoder* predicts each residue `s_i`
autoregressively, attending to the full structure and to previously decoded residues `s_{<i}`
under a causal mask (teacher-forced at training, generated left-to-right at inference). To
make the graph features *invariant and locally informative*, the method attaches a local
coordinate frame to every node, `O_i = [b_i, n_i, b_i × n_i]` (with `b_i` the negative
bisector of the rays to the chain neighbors and `n_i` the unit normal to that plane), and
encodes each edge as the triple
`e_{j→i} = ( RBF(‖x_j − x_i‖),  O_i^T (x_j − x_i)/‖x_j − x_i‖,  q(O_i^T O_j) )` —
a distance lifted to the RBF basis, the *direction to the neighbor expressed in `i`'s local
frame*, and the quaternion of the relative rotation `O_i^T O_j`. Node features are the
backbone dihedrals on the torus. Because every directional quantity is projected into a local
frame before it enters the network, the inputs are rotation-invariant by construction and the
rest of the network is an ordinary (scalar) graph transformer. Ingraham et al. further note in
their ablation that removing the attention and using plain graph propagation does not hurt —
the graph message passing, not the attention, is doing the work. **Gap:** all geometry is
collapsed into invariant scalars *at the input*; after that first projection, the
intermediate representations are scalars and the network can no longer manipulate the
directional quantities as geometric objects — it cannot, for example, propagate a position in
a shared frame, point at a location that is not itself a node, or update an orientation as
the representation deepens. Geometry is also encoded *redundantly*: an orientation is
described once per neighbor (relative to each `O_i`) rather than once, absolutely, per node.

**Equivariant networks over irreducible representations of SO(3) (Tensor Field Networks,
Thomas et al. 2018; Cormorant, Anderson et al. 2019).** A principled route to true 3D
equivariance: every layer inputs and outputs *geometric tensors* — scalars, vectors, and
higher-order tensors organized as irreducible representations of the rotation group — and
convolutional filters are built as products of a learned radial function and a spherical
harmonic, with tensor (Clebsch–Gordan) products coupling features so that the whole network
is exactly rotation- and translation-equivariant at every layer. This delivers the strongest
form of the symmetry bias and avoids data augmentation entirely. **Gap:** the spherical-
harmonic / Wigner-D and tensor-product machinery is mathematically and computationally heavy;
the cost of carrying and coupling higher-order irreps confines these networks, in practice,
to small molecules — they do not scale to proteins with hundreds of residues.

So the table is set with a tension: CNNs and the irreps networks reason geometrically (the
latter even equivariantly) but are costly or non-relational; GNNs reason relationally and
cheaply but, to stay invariant, freeze all geometry into scalars at the door. Existing
choices leave either expensive geometric machinery or cheap scalar-only graph propagation.

## Evaluation settings

- **CATH 4.2 (CPD).** Single-chain structures partitioned by CATH (class/architecture/
  topology/homologous-superfamily) classification so that held-out folds are structurally
  dissimilar to training folds; the curation of Ingraham et al. (2019), with 18204 / 608 /
  1120 train / validation / test structures. Reported on the full test set and on *short*
  (≤100 residues) and *single-chain* subsets.
- **TS50 (CPD).** An older test set of 50 native structures (Li et al. 2014), used as an
  out-of-distribution generalization check and to compare against the physics-based `fixbb`
  protocol of Rosetta (Das & Baker 2008). No canonical training set; for evaluation, training
  sequences with >30% similarity to any TS50 sequence are removed.
- **Metrics.** Primary: native sequence recovery (higher is better), reported as the median
  over structures of the average percent of residues recovered across sampled sequences.
  Secondary: per-residue perplexity (the exponential of the per-residue cross-entropy, lower
  is better), drawing the language-modeling analogy.
- **Protocol.** `k = 30` nearest neighbors by `Cα` distance; batches grouped by size; Adam
  optimizer; per-residue cross-entropy / negative-log-likelihood loss; structure-based
  train/test splits so the test folds are unseen.

## Code framework

The structure encoder plugs into a fixed inverse-folding harness. The data pipeline supplies
padded backbone coordinates and a residue mask; a set of geometric helpers already exists
(radial-basis encoding of distances, backbone dihedral angles, local-orientation vectors, and
`k`-nearest-neighbor graph construction from `Cα` positions); and the training loop, padding/
masking, optimizer schedule, per-residue cross-entropy loss, and recovery/perplexity
evaluation are all provided. What is *not* settled is the encoder itself — the module that
turns backbone geometry into per-residue embeddings, and the per-graph-step primitive it is
built from. That primitive, and how it propagates information over the graph, is exactly what
is to be designed; the scaffold leaves one big empty slot for it.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_AA = 20  # 20 standard amino acids

# ---- provided geometric helpers (already exist) ----
def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    """Lift distances into a basis of Gaussian radial basis functions."""
    ...

def _dihedrals(X, eps=1e-7):
    """Backbone dihedral angles (phi, psi, omega) as {sin, cos} features. -> (B, L, 6)"""
    ...

def knn_graph(X_ca, mask, k=30):
    """k-nearest-neighbor graph from CA coordinates. -> E_idx (B, L, K), D (B, L, K)"""
    ...


# ---- the primitive to be designed ----
class GraphLayer(nn.Module):
    """One graph-propagation step over the protein backbone graph. The way a step
    transforms node (and edge) embeddings is exactly what we will design: it must
    reason about 3D geometry yet keep the network's scalar outputs unchanged when the
    input coordinates are rotated or reflected. Nothing about its internals is fixed."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        # TODO: the per-step primitive and its parameters we will design.
        pass

    def forward(self, h_V, h_E, E_idx, mask):
        # TODO: form messages over the kNN graph, aggregate, update node embeddings.
        pass


class StructureEncoder(nn.Module):
    """Turns backbone coordinates into per-residue embeddings.
    X: (B, L, 4, 3) backbone [N, CA, C, O]; mask: (B, L). Returns h_V (B, L, hidden_dim)."""

    def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30, dropout=0.1, num_rbf=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        # TODO: how to featurize the backbone, what each embedding carries, and the
        #       stack of GraphLayer steps — all to be designed.
        pass

    def forward(self, X, mask):
        X_ca = X[:, :, 1, :]
        E_idx, D_neighbors = knn_graph(X_ca, mask, self.k_neighbors)
        # TODO: build node/edge features; run the graph-propagation stack;
        #       return per-residue embeddings h_V.
        pass


class InverseFoldingModel(nn.Module):
    """Wraps the encoder with a head that outputs amino-acid log-probabilities."""

    def __init__(self, hidden_dim=128, num_encoder_layers=3, k_neighbors=30,
                 dropout=0.1, num_rbf=16):
        super().__init__()
        self.encoder = StructureEncoder(hidden_dim, num_encoder_layers,
                                        k_neighbors, dropout, num_rbf)
        # TODO: the decoder head that maps per-residue embeddings to amino-acid logits.
        pass

    def forward(self, X, mask):
        h_V = self.encoder(X, mask)
        # TODO: decode h_V to log-probabilities (B, L, NUM_AA).
        pass
```

The training loop draws a minibatch of `(X, S, mask)`, calls `model(X, mask)` for
per-residue log-probabilities, computes the masked per-residue cross-entropy against the
native sequence `S`, and steps Adam. The single empty slot is the encoder primitive and the
representation it propagates.
