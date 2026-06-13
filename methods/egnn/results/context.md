## Research question

A large family of machine-learning problems lives not on grids or on abstract graphs but on
*sets of points embedded in Euclidean space*: a point cloud of a scanned object, the atoms of a
molecule with their 3D coordinates, the particles of an N-body simulation, each carrying a
position and possibly a velocity. Every such problem comes with the same built-in symmetry. The
laws governing it do not care where the origin is, which way the axes point, or whether we
mirror the scene: a molecule's energy is the same if we slide it across the room or rotate it;
the future positions of charged particles rotate and translate in lockstep with their initial
conditions; the labelling of the points is arbitrary. Concretely, the relevant group is the
Euclidean group `E(n)` — translations `g ∈ R^n`, plus the orthogonal group `O(n)` of rotations
and reflections (the matrices `Q` with `Q^T Q = I`) — together with arbitrary permutations of
the point set.

The goal is an architecture whose predictions respect these symmetries *exactly, by
construction*, rather than approximately, by hoping the network learns them from augmented data.
Two flavours of "respect" are needed depending on the target. Some targets are **invariant**: a
scalar like energy or a class label must be completely unchanged when the input is rotated,
translated, reflected, or its points relabelled. Other targets are **equivariant**: a predicted
displacement, an updated set of coordinates, or a velocity must rotate and translate *the same
way the input did* — rotate the input by `Q` and the output vectors come out rotated by the same
`Q`. A solution has to deliver both, on the same graph, cheaply, and ideally without being
pinned to three dimensions: some uses (embedding a graph into a continuous latent geometry) want
the same machinery in `n > 3` dimensions, so any construction that hard-codes 3D is a dead end
there. The precise problem is to build a message-passing network on a point set that is
permutation equivariant (as graph networks already are) *and* exactly `E(n)`-equivariant on the
coordinates and `E(n)`-invariant on the scalar features, with no expensive per-datapoint
geometric precomputation and no restriction to `n = 3`.

## Background

The dominant inductive-bias success stories of the period all come from baking a symmetry into
the architecture. Convolutional networks are translation equivariant: shift the input image and
the feature map shifts identically, which is why CNNs need no translation augmentation. Graph
neural networks are permutation equivariant: relabel the nodes and the outputs are relabelled
identically. The lesson is general — if a function `phi: X -> Y` must commute with a group of
transformations, enforcing `phi(T_g x) = S_g(phi(x))` for every group element `g` (with `T_g`
acting on the input and `S_g` the corresponding action on the output) removes a huge amount of
wasted model capacity and the need to learn the symmetry from data. The missing symmetry for
point sets in space is the continuous one: rotation and reflection, i.e. the orthogonal group,
extended by translation to `E(n)`.

The load-bearing facts about that symmetry are elementary but decisive. An orthogonal matrix
preserves inner products and hence norms: `(Qa)^T(Qb) = a^T Q^T Q b = a^T b`. A translation
shared by two points cancels in their difference: `(a + g) - (b + g) = a - b`. Consequently the
pairwise relative differences `x_i - x_j` are unchanged by translation, and the pairwise
distances `||x_i - x_j||` are unchanged by translation *and* by rotation/reflection. These are
the only "free" invariants of the geometry, and they are the raw material every method below
reaches for. Quantities that are *not* invariant — an absolute coordinate, a single difference
vector treated as a feature — change under the group and therefore must be handled with care: a
network that consumes a raw coordinate as a scalar input is not rotation invariant, full stop.

Geometric vectors also come in graded "types". A scalar that is unaffected by rotation (energy,
temperature, a distance) is called type-0; a vector that rotates with the input (velocity,
momentum, a displacement) is type-1. Higher types exist (the way a `2`-tensor transforms, and
beyond), and one line of work leans heavily on them; another line tries to get by with type-0
scalars and type-1 vectors only, which is all most physical inputs and outputs actually are.

The standard substrate is message passing on a graph `G = (V, E)`. With node embeddings
`h_i ∈ R^nf` and edge attributes `a_ij`, a graph-convolutional layer in the Gilmer et al. (2017)
form computes

```
m_ij    = phi_e(h_i^l, h_j^l, a_ij)        # edge message
m_i     = sum_{j ∈ N(i)} m_ij               # aggregate
h_i^{l+1} = phi_h(h_i^l, m_i)               # node update
```

with `phi_e`, `phi_h` small MLPs. This is permutation equivariant — permute the nodes and the
`h_i^{l+1}` permute the same way, because the aggregation is a symmetric sum over neighbours.
What it is emphatically *not* is translation or rotation aware: nothing in it references the
points' positions in space, and if you smuggle the raw coordinate in as a feature, a rotation of
the input scrambles the output. That is the hole the field is trying to fill.

A diagnostic phenomenon that recurs and motivates a chunk of the design space: on
featureless graphs (graphs given only by their adjacency, with no distinguishing node
attributes), a plain GNN assigns *identical* embeddings to nodes with identical neighbourhood
topology. A cycle graph is the sharp example — every node sees the same local structure, so every
embedding is the same, and the original edges cannot be recovered from the embeddings. Adding
i.i.d. noise to the input node features is a known ad-hoc fix (Liu et al. 2019): it breaks the
symmetry so distinct embeddings emerge, at the cost of forcing the network to generalize over a
fresh noise distribution.

## Baselines

These are the prior methods a new geometric architecture would be measured against and would
react to.

**Message-passing GNN / MPNN (Gilmer et al., ICML 2017).** The neutral substrate above:
permutation-equivariant message passing with MLP edge and node functions, originally for quantum
chemistry. *Gap:* it has no notion of Euclidean geometry at all. It is invariant to nothing
spatial; feeding coordinates as node features makes it sensitive to the arbitrary global pose of
the input, so it either fails to respect rotation/translation or must learn the symmetry from
exhaustive augmentation. It also has no channel that *produces* a geometric vector output — it
emits only relabel-equivariant scalars.

**SchNet (Schütt et al., NeurIPS 2017).** Make the message depend only on the rotation- and
translation-invariant interatomic distance. Distances are expanded in a Gaussian radial basis,
`e_k(r_ij) = exp(-gamma (r_ij - mu_k)^2)` over a grid of centres `mu_k`, and a continuous-filter
convolution multiplies neighbour embeddings by a filter generated from that expansion:
`m_ij = phi_cf(||x_i - x_j||) ⊙ phi_s(h_j)`, then `h_i^{l+1} = phi_h(h_i, sum_j m_ij)`. Because
only the invariant distance enters, every layer's output is `E(n)`-**invariant**, which is
exactly right for predicting a scalar like energy and is cheap and dimension-agnostic. *Gap:*
the network is invariant *everywhere* — it can only ever emit invariant scalars. It has no way to
output an equivariant geometric quantity (an updated position, a velocity, a displacement field),
because it never represents or transforms a type-1 vector; the spatial information collapses to
scalars at the first layer and the vector structure is gone.

**Tensor Field Networks (Thomas et al., 2018) and the SE(3)-Transformer (Fuchs et al., NeurIPS
2020).** Achieve true rotation equivariance by working with higher-type (steerable)
representations. The filters are built from spherical harmonics and Clebsch–Gordan coefficients,
giving learnable kernels `W^{lk}: R^3 -> R^{(2l+1)×(2k+1)}` that map between a type-`k` input
irrep and a type-`l` output irrep while commuting with `SO(3)`; the SE(3)-Transformer adds
attention on top. These are expressive and genuinely equivariant (rotation, and translation by
using relative positions). *Gaps:* the spherical harmonics must be evaluated for every datapoint
(every distinct relative geometry), which is computationally heavy; the whole construction is
tied to `SO(3)` in three dimensions and has no known extension to arbitrary `n`; and the
Clebsch–Gordan / irrep bookkeeping is intricate to implement and tune.

**Radial Field / equivariant flow update (Köhler et al., 2019/2020).** An `E(n)`-equivariant
update that acts directly and only on coordinates: `m_ij = phi_rf(||x_i - x_j||) (x_i - x_j)`,
then `x_i^{l+1} = x_i + sum_{j≠i} m_ij`, with `phi_rf` an MLP of the scalar distance. This is
cheap, equivariant for any `n`, and needs no spherical harmonics. *Gap:* it operates *only* on
the coordinates `x`; it carries no learnable node-feature channel `h` and never propagates such
features between nodes. Its strong geometric bias helps in the small-data regime but caps its
flexibility — it cannot absorb the rich, learned per-node information that a full GNN feature
channel provides, so it underfits once data is plentiful.

**Angle/directional and `SO(3)`-irrep message passing (Cormorant, Anderson et al. 2019; DimeNet,
Klicpera et al. 2020).** Enrich invariant message passing with bond angles and directional
information (DimeNet) or carry `SO(3)`-equivariant intermediate activations (Cormorant). More
expressive than distance-only invariant nets. *Gap:* the angular/Bessel-and-spherical-harmonic
basis (DimeNet) and the irrep machinery (Cormorant) reintroduce the same cost and 3D-specificity
that the steerable methods carry.

The shape of the landscape: invariant methods (SchNet) are cheap and dimension-free but can only
emit scalars; equivariant methods that can emit vectors (TFN, SE(3)-Transformer) pay with
spherical-harmonic cost and 3D-only scope; the one cheap equivariant method (Radial Field) throws
away the learnable feature channel that makes GNNs flexible. No single method is simultaneously
cheap, dimension-agnostic, equipped with a full feature channel, and able to emit both invariant
scalars and equivariant vectors.

## Evaluation settings

The natural yardsticks already in use for geometric point-set learning, and the protocols around
them:

- **Charged-particle N-body system** (Kipf et al. 2018, extended to 3D as in Fuchs et al. 2020).
  Five particles in 3D, each with a charge in `{-1, +1}`, attract/repel by Coulomb-like rules.
  Given initial positions `p^(0) ∈ R^{5×3}`, initial velocities `v^(0) ∈ R^{5×3}`, and charges,
  predict the positions after 1000 timesteps; metric is mean squared error of the estimated
  positions. The task is exactly equivariant (rotating/translating the initial state rotates and
  translates the whole trajectory). Standard split 3000 train / 2000 val / 2000 test, with a
  data-efficiency sweep from 100 to 50000 training samples; trained with Adam, batch 100.
- **Graph autoencoding** on Community-Small (You et al. 2018) and Erdős–Rényi random graphs
  (`gnp_random_graph`, edge probability `p = 0.25`, `7 ≤ M ≤ 16` nodes). Encode a featureless
  graph into per-node latent coordinates `z ∈ R^{M×n}`, decode the adjacency with a
  distance-based edge probability `Â_ij = sigmoid(-(w ||z_i - z_j||^2 + b))`; train with binary
  cross-entropy `sum_ij BCE(Â_ij, A_ij)`; report BCE, percentage of wrong edges, and F1. The
  latent dimension `n` can exceed 3, so this stresses dimension-agnostic geometry. Adam,
  learning rate `1e-4`, 5000 train / 500 val / 500 test, plus a 100-graph overfitting probe
  sweeping sparsity `p` from 0.1 to 0.9.
- **QM9 molecular property prediction** (Ramakrishnan et al. 2014). Up to 29 atoms per molecule,
  each with a 3D position and a 5-dim atom-type one-hot (H, C, N, O, F). Regress 12 chemical
  properties (`alpha`, `Δε`, `ε_HOMO`, `ε_LUMO`, `mu`, `C_v`, `G`, `H`, `R^2`, `U`, `U_0`,
  `ZPVE`), each invariant to rotation/translation/reflection of the atom positions; metric is
  mean absolute error. Partitions from Anderson et al. 2019 (100K train / 18K val / 13K test).
  Adam, batch 96, weight decay `1e-16`, cosine learning-rate decay from `5e-4` (or `1e-3` for
  HOMO/LUMO/gap), properties normalized to zero mean and unit mean-absolute-deviation.

## Code framework

The open design slot sits inside the same message-passing harness the baseline GNN already uses.
What exists beforehand is a node-embedding lookup, a stack of layers, a pooling readout, and an
edge-construction utility. A point-set datum supplies node scalar features and `n`-dimensional
coordinates; the unsettled part is the per-layer transformation inside the geometric layer.

```python
import torch
import torch.nn as nn


class GeometricLayer(nn.Module):
    """One message-passing layer over a point set. Consumes per-node scalar
    features h, per-node coordinates x (in R^n), and the edge set."""

    def __init__(self, hidden_nf, act_fn=nn.SiLU()):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.act_fn = act_fn
        self.update_rule = None  # TODO: geometry-respecting message-passing rule.

    def forward(self, h, x, edge_index, edge_attr=None):
        # h: (N, hidden_nf) per-node scalar features
        # x: (N, n)         per-node coordinates
        # edge_index: (2, E) sender/receiver node indices
        if self.update_rule is None:
            raise NotImplementedError
        return self.update_rule(h, x, edge_index, edge_attr)


class GeometricNet(nn.Module):
    """Generic message-passing network over a point set: embed node features,
    apply a stack of geometric layers, read out. The per-layer geometric rule
    is the open slot."""

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_layers=4,
                 act_fn=nn.SiLU()):
        super().__init__()
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)
        self.layers = nn.ModuleList(
            [GeometricLayer(hidden_nf, act_fn) for _ in range(n_layers)]
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        h = self.embedding_in(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        h = self.embedding_out(h)
        return h, x


def build_edges(n_nodes):
    # Fully connected graph (j != i) when no adjacency is provided.
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return [rows, cols]


# existing training loop the network plugs into
def train(model, loss_fn, data_loader, optimizer):
    for h, x, edge_index, target in data_loader:   # point-set minibatch
        optimizer.zero_grad()
        h_out, x_out = model(h, x, edge_index)      # forward through the stack
        loss = loss_fn(h_out, x_out, target)        # task loss (invariant or equivariant target)
        loss.backward()
        optimizer.step()
```

The outer loop and the readout are fixed; `GeometricLayer.forward` is the single empty slot
where the geometric rule will live.
