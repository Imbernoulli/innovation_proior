# Context: updating pairwise relations between residues so they stay geometrically consistent

## Research question

A protein of length N can be described by a single representation `s_i ∈ ℝ^{c_m}` per
residue, but the quantity that actually predicts 3D structure is *relational*: how residue
`i` sits relative to residue `j` — their distance, their relative orientation. The natural
home for that is a **pair representation** `z_ij ∈ ℝ^{c_z}` (here `c_z = 128`): a learned
embedding on every ordered pair `(i, j)`, i.e. an *edge feature* on the complete graph whose
nodes are residues. The network reads `z` out into a distogram (a predicted distribution over
binned inter-residue distances) and feeds it to the module that places atoms in space.

The problem: a single pairwise relation is cheap to predict in isolation, but a *collection*
of pairwise relations is only meaningful if it is **mutually realizable in 3D**. A list of
proposed distances `{d_ij}` corresponds to actual points in space only when every triple
`(i, j, k)` is geometrically consistent — at minimum the triangle inequality
`d_ij ≤ d_ik + d_kj` holds, with higher-order distance-matrix constraints beyond that. So the
edge `(i, j)` is *not free*: it is constrained by the two other edges of every triangle it sits
in, `(i, k)` and `(k, j)`, for every third residue `k`. The goal is a learnable update of the
pair representation that lets each edge feel those constraints, so the representation is pushed
toward a globally consistent geometry rather than N² independently-guessed numbers — and that
remains cheap enough to run inside a deep stack on sequences of hundreds to thousands of
residues.

## Background

**The pair representation and how it is born.** Before any update, `z_ij` is assembled from two
sources. (1) A relative-position encoding: with residue indices `f^{idx}_i`, form
the signed sequence separation `d_ij = f^{idx}_i − f^{idx}_j`, clip it to `[−32, 32]`,
one-hot it into 65 bins, and linearly project to `c_z` (this is the `relpos` operator). (2) An
outer sum of the single embeddings, `z_ij ← Linear(s_i) + Linear(s_j)`. During the trunk it is
further updated from the multiple-sequence alignment (MSA) by an **outer-product mean**: project
each MSA entry `m_{si}` to two small vectors `a_{si}, b_{si} ∈ ℝ^{c}` (`c = 32`), then
`o_ij = flatten(mean_s(a_{si} ⊗ b_{sj}))`, `z_ij = Linear(o_ij)`. So coevolutionary signal from
the alignment is the raw evidence for which residue pairs are in contact, and `z` is the running
estimate of all pairwise relations.

**Geometric realizability is a hard constraint, not a soft preference.** Metric geometry says an
arbitrary symmetric matrix of "distances" is almost never embeddable in ℝ³. The triangle
inequality on every triple is the first necessary condition; higher-order
Euclidean distance-matrix constraints further restrict which finite metric spaces can live in
three dimensions. A representation that predicts each `z_ij` without reference to its
neighboring edges will, generically, propose distance sets that no real structure can satisfy.

**Coevolution and contact prediction.** Correlated mutations between two alignment columns have
long been read as evidence of spatial contact. This is why an MSA is the input and why the pair
representation is the place where structure must be reasoned about.

**Gating as learned selection.** The surrounding architecture already uses sigmoid gates as a
differentiable, per-channel "should this contribute?" filter — e.g. the MSA gated self-attention
computes `g = sigmoid(Linear(·))` and multiplies it onto the attention output.

## Baselines

**Independent / axis-local edge updates (the thing to beat).** The cheapest ways to update a
2D grid of edge features mix information along *one* residue axis at a time. *Axial attention*
(Ho et al. 2019, "Axial Attention in Multidimensional Transformers"; cf. the sparse-attention
factorizations of Child et al. 2019) factorizes attention over an `H × W` grid into attention
along rows then along columns, costing `O(N)` interactions per axis instead of the full `O(N²)`
over all grid cells. The core idea: for a fixed `i`,
let `z_{ik}` attend over all `k`; or for a fixed `j`, let `z_{kj}` attend over all `k`. Each such
operation couples an edge only to other edges *sharing one endpoint along a single axis*: for a
target `(i, j)` it can look at `(i, k)` (same row) or at `(k, j)` (same column), in separate
operations along separate axes.

**Per-pair MLP / convolution on the distance map.** Treating the pair map like an image and
running 2D convolutions or per-pixel MLPs (the dominant pre-attention approach to contact maps)
has the same flaw at the level of geometry: a local receptive field on the `(i, j)` plane mixes
*nearby pairs in index space*, which has no relation to geometric adjacency — spatial locality in
`(i, j)` is unrelated to which residues actually constrain a given distance.

**Generic graph message passing.** A graph network updates a node or edge by aggregating over
its neighbors. Standard message-passing frameworks are generic: they fix neither *what function*
combines feature vectors nor the aggregation pattern, and an unconstrained aggregation over the
complete edge set can be much costlier than `O(N³)`.

## Evaluation settings

The natural yardstick is **single-domain protein structure prediction** on the CASP-style
benchmark protocol: predict 3D coordinates from sequence plus a multiple-sequence alignment (and
optionally templates), on recently-released structures held out from training (e.g. a CASP14
test set and a held-out set of recent PDB chains). Inputs are the query sequence, an MSA, and
template structures; the representations are the per-residue single `s_i`, the pair `z_ij`, and
the MSA `m_{si}`. Standard intermediate readouts are the **distogram** (binned inter-residue
distance distribution) and per-residue confidence; the standard structural metrics are backbone
accuracy measures such as lDDT-Cα, GDT, TM-score, and RMSD. The operation under design sits
inside one block of a deep trunk that is run on sequences of a few hundred to a few thousand
residues, so per-block memory and the `O(N³)` cost are first-class constraints.

## Code framework

The available primitives are a layer-normalized linear layer, rowwise and columnwise dropout,
a standard gated multi-head attention module, a pair-transition MLP, and the trunk's block loop.
The pair representation enters each block already formed; what is missing is the update operator
that lets the pair map move toward a consistent geometry.

```python
import torch
import torch.nn as nn

class Linear(nn.Linear):
    """Affine layer with the trunk's init conventions (e.g. 'gating', 'final')."""
    ...

class LayerNorm(nn.LayerNorm):
    """Channel-wise layer norm over the last dim."""
    ...

class DropoutRowwise(nn.Module):
    def forward(self, x):
        ...

class DropoutColumnwise(nn.Module):
    def forward(self, x):
        ...

class Attention(nn.Module):
    """Standard gated multi-head attention: q,k,v projections, scaled dot-product
    softmax over keys, optional additive bias terms, sigmoid output gate, output
    projection. Used elsewhere in the trunk."""
    def forward(self, q_x, kv_x, biases):  # biases: list of additive logit tensors
        ...

class PairTransition(nn.Module):
    """2-layer MLP on z with a 4x channel expansion."""
    def __init__(self, c_z, n=4):
        super().__init__()
        self.ln = LayerNorm(c_z)
        self.lin1 = Linear(c_z, n * c_z)
        self.lin2 = Linear(n * c_z, c_z)
    def forward(self, z, mask=None):
        out = self.lin2(torch.relu(self.lin1(self.ln(z))))
        return out if mask is None else out * mask.unsqueeze(-1)

class PairStack(nn.Module):
    """Residual block over pair features."""
    def __init__(self, c_z, pair_dropout=0.25):
        super().__init__()
        self.pair_transition = PairTransition(c_z, n=4)
        self.dropout_row = DropoutRowwise(pair_dropout)
        self.dropout_col = DropoutColumnwise(pair_dropout)
        # TODO: build the pair-update operator(s).

    def forward(self, z, pair_mask):
        # z: [*, N, N, c_z]
        # TODO: insert the pair-update residual update(s).
        z = z + self.pair_transition(z, mask=pair_mask)
        return z
```
