# Context: learning molecular energies and forces from raw 3D atom positions (circa 2016–2017)

## Research question

Given a molecule as a set of `n` atoms with nuclear charges `Z = (Z_1, …, Z_n)` and 3D
positions `R = (r_1, …, r_n)`, learn a single model that predicts the total potential energy
`E(r_1, …, r_n)` and, from the same model, the interatomic forces

```
F_i(r_1, …, r_n) = - ∂E/∂r_i.
```

The model must generalize across both *chemical* variation (different atom-type compositions
and molecule sizes) and *conformational* variation (the same molecule in many geometries off
equilibrium), so that it can be used not only for property prediction on relaxed structures
but for force-driven tasks like geometry optimization and molecular dynamics.

A valid model is heavily constrained by physics, and these constraints are the hard part:

- **Invariance.** The energy is a scalar physical observable, so it must be invariant to
  rigid rotation, translation, and to the (arbitrary) indexing of atoms. The force on atom
  `i` transforms as a vector, so it must be rotationally *equivariant*.
- **Energy conservation.** The force field must be curl-free, i.e. it must be the (negative)
  gradient of a scalar potential. If it were not, one could follow a closed loop of atom
  positions along which the energy strictly increases, manufacturing energy from nothing —
  a violation of energy conservation. Any model that predicts forces as a free vector field,
  decoupled from a scalar energy, can break this.
- **Smoothness.** The potential-energy surface and its derivatives must be smooth in the atom
  positions. Geometry optimization needs a continuous gradient `∂E/∂r`. Training a model on
  *forces* (not just energies) needs the energy model to be *twice* differentiable, because
  the force is already one derivative of the energy and a gradient-based loss on the force
  takes one more. Any discontinuity in `E(R)` — a binning edge, a hard neighbor cutoff, a
  one-hot bond-type flag that flips as a bond stretches — turns into a spike or a gap in the
  force and makes force training ill-posed.

The pain point is that the representations that scale and the representations that are smooth
and physically valid were, at the time, two different camps; closing that gap is the problem.

## Background

By 2016–2017 the field has two broad approaches to potential-energy surfaces, and a fast-
moving deep-learning thread that has not yet met the physics constraints above.

**Convolutions assume a grid; atoms do not live on one.** In deep learning, convolutional
layers (LeCun et al. 1989; Krizhevsky et al. 2012; for audio, van den Oord et al. 2016)
operate on discretized signals — image pixels, video frames, audio samples — where a filter
is a small tensor indexed by integer grid offsets `Δ`, and the layer computes
`x_i' = Σ_Δ x_{i+Δ} · W[Δ]`. Atoms sit at arbitrary continuous positions in `R^3`. The
obvious workaround — voxelize space and resample atoms onto a grid — forces a choice of
interpolation scheme, needs many grid points for adequate resolution, and, fatally, makes the
output change discontinuously as an atom crosses a voxel boundary. The same difficulty arises
for other unevenly-spaced data (astronomical time series, climate records, finance), and has
prompted extensions of convolution beyond Euclidean grids — to graphs (Bruna et al. 2014;
Henaff et al. 2015) and to 3D shapes / manifolds (Masci et al. 2015). A relevant building
block is the **dynamic filter network** (Jia, De Brabandere, Tuytelaars & Van Gool, NIPS
2016): instead of fixed filter weights, a small network *generates* the convolution filter
conditioned on the input — but the generated weights are still laid out on a grid.

**A smooth, infinitely-differentiable nonlinearity exists.** Softplus, `softplus(x) =
ln(1 + e^x)`, is a `C^∞` approximation to ReLU; its shape resembles the ELU (Clevert et al.
2015) but, unlike ReLU/ELU, it has derivatives of every order — which matters whenever a
model's *derivative* (here, the force) must itself be differentiable.

**Diagnostic facts about freshly-initialized filter networks.** A neural network at
initialization is close to linear, so if several output channels of a filter-generating
network are fed the same scalar input, they produce nearly the same (nearly linear) function
of it — the channels are highly correlated. In practice this is observed to create a plateau
at the very start of training that is hard to escape, because near-identical filters carry
almost no independent information. Decorrelating the inputs to such a network — for example by
passing a scalar through a bank of localized basis functions before the dense layers, so
different channels respond to different ranges of the scalar — is a standard way to break that
symmetry and speed up early training.

**Behler-style locality and smooth cutoffs.** A long line of physically-motivated potentials
writes the total energy as a sum of atomic contributions over local environments,
`E = Σ_i E_i`, with each `E_i` depending only on neighbors within a cutoff radius `r_cut`
(Behler & Parrinello 2007; Behler 2011; Bartók et al. 2010, 2013). To keep `E` smooth as
atoms enter and leave a neighborhood during dynamics, these methods multiply each neighbor's
contribution by a **cosine cutoff** that decays smoothly to zero at the boundary,

```
f_cut(r) = 0.5 · [ 1 + cos(π r / r_cut) ]   for r < r_cut,   and 0 for r ≥ r_cut,
```

which has both value and first derivative equal to zero at `r = r_cut`, so a neighbor fades in
and out without any jump in the energy or a delta in the force. This locality also makes the
per-step cost scale linearly with system size, since each atom has a bounded number of
neighbors within `r_cut`.

## Baselines

The prior methods a new architecture would be measured against, and the specific limitation
each leaves open.

**High-dimensional NN potentials with hand-crafted descriptors (Behler & Parrinello 2007;
Behler 2011; Bartók et al. 2010, 2013).** Write `E = Σ_i E_i` and feed each atomic network a
fixed vector of **atom-centered symmetry functions** — radial and angular functions of the
neighbor distances/angles within `r_cut`, smoothed by a cosine cutoff. These are invariant,
smooth, energy-conserving, and scale linearly. **Gap:** the descriptors are *hand-engineered
and fixed*. The radial/angular symmetry functions, their parameters, and the cutoff must be
chosen by the practitioner, and a good set for one class of systems is not automatically good
for another; the representation is not learned from data.

**Deep Tensor Neural Networks — DTNN (Schütt, Arbabzadah, Chmiela, Müller & Tkatchenko,
Nature Communications 2017).** Learns the representation directly from `Z` and the distance
matrix. Each atom carries a feature vector `c_i` (initialized from a type-specific embedding),
refined over `T` passes by summing pairwise messages,

```
c_i^{t+1} = c_i^t + Σ_{j ≠ i} v_{ij},
```

where the message `v_{ij}` couples the neighbor's features with the (Gaussian-expanded)
interatomic distance through a **parameter tensor**, kept affordable by a low-rank
factorization. It is invariant by construction and reaches high accuracy. **Gap:** the
distance enters through a factorized bilinear tensor coupling rather than as a learned spatial
filter, and the interaction has no convolution interpretation; its accuracy trails the best
message-passing models, and it shares interaction parameters across passes, limiting
expressiveness.

**Message-passing / enn-s2s (Gilmer, Schoenholz, Riley, Vinyals & Dahl, ICML 2017).** Unifies
graph networks as message passing: with node states `h_v` and edge features `e_{vw}`,

```
m_v^{t+1} = Σ_{w ∈ N(v)} M_t(h_v, h_w, e_{vw}),    h_v^{t+1} = U_t(h_v, m_v^{t+1}),
```

then a readout `R({h_v})`. Its strongest variant uses an **edge network** message
`M(h_v, h_w, e_{vw}) = A(e_{vw}) · h_w`, where a small network maps the edge features to a
full `F × F` matrix `A(e_{vw})`, a GRU update `U_t`, and a set2set readout; it reaches
state-of-the-art on the QM9 property benchmark. **Gap:** its edge features include *one-hot
bond types* (single/double/triple). Those discrete inputs make the predicted energy
discontinuous as a bond changes character along a trajectory, so the model cannot produce a
smooth potential-energy surface and **cannot be trained on forces or used for molecular
dynamics**. The full `F × F` matrix per edge is also heavier than necessary.

**Gradient-domain machine learning — GDML (Chmiela, Tkatchenko, Sauceda, Poltavsky, Schütt &
Müller, Science Advances 2017).** A kernel method that fits the *force* field directly in the
gradient domain, with the kernel constructed so that the predicted force is the gradient of a
scalar — hence energy-conserving by construction; the energy is recovered by re-integration
(energies only fix the integration constant). Remarkably accurate with as few as ~1,000
training points. **Gap:** the kernel (Gram) matrix grows quadratically with both the number
of atoms and the number of training examples, so it does not scale to large datasets (e.g.
50,000 examples) and is not designed to span different atom-type compositions — one model per
molecule.

A pair of architectural primitives also sit in the background as reusable pieces. **Residual
learning (He et al. 2016)** writes a block as `x + F(x)` so very deep stacks stay trainable.
**Depthwise-separable convolution (Chollet 2016, Xception)** splits a convolution into a
cheap per-channel (feature-wise) spatial part and a separate cross-channel mixing, cutting
cost relative to a full dense filter.

## Evaluation settings

The natural yardsticks already in use for atomistic ML at the time:

- **QM9** (Ramakrishnan et al. 2014; Blum & Reymond 2009): ~130k small organic molecules with
  up to nine heavy atoms from {C, O, N, F}, all in *equilibrium* (forces zero by definition).
  The standard property-prediction benchmark; total energy `U_0` is a common target. Metric:
  mean absolute error (kcal/mol). Training-set size is varied across prior work (e.g. 50k,
  100k, 110k).
- **MD17** (Chmiela et al. 2017): eight molecular-dynamics trajectories of single small
  organic molecules (benzene, toluene, malonaldehyde, salicylic acid, aspirin, ethanol,
  uracil, naphthalene), each spanning many *off-equilibrium* conformations with both energies
  and forces. A separate model per trajectory; the regime where accurate forces matter.
  Metrics: MAE for energy (kcal/mol) and forces (kcal/mol/Å), with training sizes such as
  1,000 and 50,000.
- **Isomer / chemical-plus-conformational sets**: short trajectories of many chemically
  distinct isomers with a *shared* composition, used to test a single joint model across both
  chemical and conformational change, with held-out conformations and held-out molecules.
- Protocol: split into train / a small validation set (e.g. 1,000) for early stopping / test;
  train with the Adam optimizer (Kingma & Ba 2015), minibatches of ~32 molecules, an initial
  learning rate of `1e-3` with exponential decay, and an exponential moving average over
  weights for the model used at test time.

For *protein structure* representation, the corresponding yardsticks are per-residue /
per-protein function and structure prediction (enzyme-class, gene-ontology, fold
classification) from alpha-carbon coordinates, with accuracy / F-max metrics. The encoder
surface consumes precomputed node feature vectors, a spatial neighbor graph, coordinates, and
batch indices rather than atom-type identifiers alone.

## Code framework

The encoder plugs into a generic geometric-graph harness that already exists: a routine to
build a spatial neighbor graph from coordinates, an embedding of the per-node input features,
a stack of message-passing blocks (whose internals are exactly what is to be designed), and a
pooling step to read out per-graph vectors. The pieces that already exist are the graph
construction, the elementwise primitives, scatter-based pooling, and the residual/atom-wise
dense-layer abstractions. The empty slots are how spatial geometry is turned into the
quantity that drives a message, and the message/interaction layer itself.

```python
import torch.nn as nn
from torch_scatter import scatter


def edge_distances(pos, edge_index):
    """Given a spatial graph over 3D coordinates, compute the scalar that
    the geometry will be summarized by (the interatomic distance)."""
    u, v = edge_index
    dist = (pos[u] - pos[v]).norm(dim=-1)          # rotation/translation-invariant scalar
    return edge_index, dist


class DistanceFeature(nn.Module):
    """Turn a scalar interatomic distance into the input a geometric message
    layer consumes. What this map should be is not yet decided."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        # TODO: how raw geometry enters the network.
        pass

    def forward(self, dist):
        # TODO
        pass


class InteractionLayer(nn.Module):
    """One geometry-aware message-passing block: update each node from its
    neighbors using the geometric features. The form of this update — how a
    neighbor's features and the pairwise geometry combine — is the contribution
    to be designed."""
    def __init__(self, hidden_channels, *args, **kwargs):
        super().__init__()
        # TODO: the interaction/message layer we will design.
        pass

    def forward(self, h, edge_index, dist_feat):
        # TODO: compute and return the per-node update (residual added by caller)
        pass


class GeometricEncoder(nn.Module):
    """Generic encoder: embed node features -> stack of interaction blocks ->
    per-node output -> pooled per-graph output."""
    def __init__(self, hidden_channels, out_dim, num_layers, readout="add"):
        super().__init__()
        self.readout = readout
        self.embedding = nn.LazyLinear(hidden_channels)
        self.distance_feature = DistanceFeature()       # TODO slot
        self.interactions = nn.ModuleList(
            [InteractionLayer(hidden_channels) for _ in range(num_layers)]
        )
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = None                                  # TODO: the nonlinearity
        self.lin2 = nn.LazyLinear(out_dim)

    def forward(self, batch):
        edge_index, dist = edge_distances(batch.pos, batch.edge_index)
        dist_feat = self.distance_feature(dist)
        h = self.embedding(batch.x)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, dist_feat)   # residual stack
        h = self.lin1(h)
        h = self.act(h)
        node_emb = self.lin2(h)
        graph_emb = scatter(node_emb, batch.batch, dim=0, reduce=self.readout)
        return {"node_embedding": node_emb, "graph_embedding": graph_emb}
```

The harness receives the graph and pools; the update rule inside `InteractionLayer`, the map
from distance to a message-driving feature inside `DistanceFeature`, and the choice of
nonlinearity are the empty slots the method will fill.
