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
`d_ij ≤ d_ik + d_kj` holds, and more fully the Cayley–Menger determinant conditions. So the
edge `(i, j)` is *not free*: it is constrained by the two other edges of every triangle it sits
in, `(i, k)` and `(k, j)`, for every third residue `k`. The goal is a learnable update of the
pair representation that lets each edge feel those constraints, so the representation is pushed
toward a globally consistent geometry rather than N² independently-guessed numbers — and that
remains cheap enough to run inside a deep stack on sequences of hundreds to thousands of
residues.

## Background

**The pair representation and how it is born.** Before any update, `z_ij` is assembled from two
pre-method sources. (1) A relative-position encoding: with residue indices `f^{idx}_i`, form
the signed sequence separation `d_ij = f^{idx}_i − f^{idx}_j`, clip it to `[−32, 32]`,
one-hot it into 65 bins, and linearly project to `c_z` (this is the `relpos` operator). (2) An
outer sum of the single embeddings, `z_ij ← Linear(s_i) + Linear(s_j)`. During the trunk it is
further updated from the multiple-sequence alignment (MSA) by an **outer-product mean**: project
each MSA entry `m_{si}` to two small vectors `a_{si}, b_{si} ∈ ℝ^{c}` (`c = 32`), then
`o_ij = flatten(mean_s(a_{si} ⊗ b_{sj}))`, `z_ij = Linear(o_ij)`. So coevolutionary signal from
the alignment is the raw evidence for which residue pairs are in contact, and `z` is the running
estimate of all pairwise relations.

**Geometric realizability is a hard constraint, not a soft preference.** It is a known fact of
metric geometry that an arbitrary symmetric matrix of "distances" is almost never embeddable in
ℝ³. The triangle inequality on every triple is the first necessary condition; the full set of
conditions (non-negativity of Cayley–Menger determinants up to order 4) pins down embeddability
in three dimensions. A representation that predicts each `z_ij` without reference to its
neighboring edges will, generically, propose distance sets that no real structure can satisfy.
This is the load-bearing pre-method fact: the constraint that should regularize edge `(i, j)`
physically lives in the *other two edges* of the triangles through `(i, j)`.

**Coevolution and contact prediction.** Correlated mutations between two alignment columns have
long been read as evidence of spatial contact; the row-attention map of an alignment transformer
empirically lights up on contacting pairs. This is why an MSA is the input and why the pair
representation is the place where structure must be reasoned about.

**Gating as learned selection.** The surrounding architecture already uses sigmoid gates as a
differentiable, per-channel "should this contribute?" filter — e.g. the MSA gated self-attention
computes `g = sigmoid(Linear(·))` and multiplies it onto the attention output. Gates are the
standard tool here for letting the network softly choose which signals to keep.

## Baselines

**Independent / axis-local edge updates (the thing to beat).** The cheapest ways to update a
2D grid of edge features mix information along *one* residue axis at a time. *Axial attention*
(Ho et al. 2019, "Axial Attention in Multidimensional Transformers"; cf. the sparse-attention
factorizations of Child et al. 2019) factorizes attention over an `H × W` grid into attention
along rows then along columns, costing `O(N)` interactions per axis instead of the full `O(N²)`
over all grid cells. Adapted to alignments, tied row attention plus column attention over the
(sequences × positions) matrix gives a strong contact predictor (Rao et al. 2021, MSA
Transformer), and its row-attention maps approximate contacts. The core idea: for a fixed `i`,
let `z_{ik}` attend over all `k`; or for a fixed `j`, let `z_{kj}` attend over all `k`. The gap:
each such operation couples an edge to other edges *sharing one endpoint along a single axis*. It
never combines the **two incident edges of a triangle at once** — for a target `(i, j)` it can
look at `(i, k)` (same row) or at `(k, j)` (same column), but not at the pair `(i, k)` and
`(k, j)` together. So it cannot express, let alone enforce, the triple constraint
`d_ij ≤ d_ik + d_kj`. That precise blind spot is what an update must close.

**Per-pair MLP / convolution on the distance map.** Treating the pair map like an image and
running 2D convolutions or per-pixel MLPs (the dominant pre-attention approach to contact maps)
has the same flaw at the level of geometry: a local receptive field on the `(i, j)` plane mixes
*nearby pairs in index space*, not the *third-node* neighbors `(i, k), (k, j)` that actually
constrain the triangle. Spatial locality in `(i, j)` is unrelated to geometric adjacency.

**Generic graph message passing.** A graph network updates a node or edge by aggregating over
its neighbors. The natural aggregation for an *edge* `(i, j)` is over a third node `k`, combining
the two edges incident to that triangle. Standard message-passing frameworks support this in
principle, but leave open *what function* combines two edge vectors and aggregates over the apex
`k`, and how to do it at `O(N³)` rather than something worse — the design the update must pin
down.

## Evaluation settings

The natural yardstick is **single-domain protein structure prediction** on the CASP-style
benchmark protocol: predict 3D coordinates from sequence plus a multiple-sequence alignment (and
optionally templates), on recently-released structures held out from training (e.g. a CASP14
test set and a held-out set of recent PDB chains). Inputs are the query sequence, an MSA, and
template structures; the representations are the per-residue single `s_i`, the pair `z_ij`, and
the MSA `m_{si}`. Standard intermediate readouts are the **distogram** (binned inter-residue
distance distribution) and per-residue confidence; the standard structural metrics are backbone
accuracy measures such as lDDT-Cα, GDT, TM-score, and RMSD. The operation under design sits
inside one block of a deep trunk (here 48 blocks) that is run on sequences of a few hundred to a
few thousand residues, so per-block memory and the `O(N³)` cost are first-class constraints.

## Code framework

The pre-existing primitives are a layer-normalized linear layer, a standard gated multi-head
attention module, and the trunk's block loop. The pair representation enters each block already
formed; what is missing is the operator that mixes edges *through a third node* so the pair map
becomes geometrically consistent. Lay out the empty slots:

```python
import torch
import torch.nn as nn

# --- Pre-existing primitives ---

class Linear(nn.Linear):
    """Affine layer with the trunk's init conventions (e.g. 'gating', 'final')."""
    # already implemented

class LayerNorm(nn.LayerNorm):
    """Channel-wise layer norm over the last dim."""
    # already implemented

class Attention(nn.Module):
    """Standard gated multi-head attention: q,k,v projections, scaled dot-product
    softmax over keys, optional additive bias terms, sigmoid output gate, output
    projection. Used elsewhere in the trunk."""
    def forward(self, q_x, kv_x, biases):  # biases: list of additive logit tensors
        ...

# --- The slot the method will fill: an edge update that couples the two other
#     edges of every triangle through residue i,j so the pair map is consistent ---

class PairEdgeUpdate(nn.Module):
    """Update z_ij using the other two edges (i,k),(k,j) of every triangle on (i,j).
    Two endpoint conventions (which node is shared) will fall out of the geometry."""
    def __init__(self, c_z, c_hidden):
        super().__init__()
        # TODO: projections, gates, and normalization for combining two incident edges
        pass

    def forward(self, z, mask=None):
        # z: [*, N, N, c_z]
        # TODO: combine the two triangle edges over the shared apex k and aggregate
        # TODO: gate and project the aggregated update back to c_z
        pass


class PairTransition(nn.Module):
    """2-layer MLP on z with a 4x channel expansion (already a known block)."""
    def __init__(self, c_z, n=4):
        super().__init__()
        self.ln = LayerNorm(c_z)
        self.lin1 = Linear(c_z, n * c_z)
        self.lin2 = Linear(n * c_z, c_z)
    def forward(self, z):
        return self.lin2(torch.relu(self.lin1(self.ln(z))))


def pair_block(z, pair_update_a, pair_update_b, edge_attn_a, edge_attn_b, transition):
    # Residual updates of the pair representation within one trunk block.
    z = z + pair_update_a(z)   # TODO: triangle edge update, sharing convention A
    z = z + pair_update_b(z)   # TODO: triangle edge update, sharing convention B
    z = z + edge_attn_a(z)     # TODO: content-routed triangle edge update, convention A
    z = z + edge_attn_b(z)     # TODO: content-routed triangle edge update, convention B
    z = z + transition(z)
    return z
```
