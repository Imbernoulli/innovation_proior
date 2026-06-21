## Research question

A protein backbone is a chain of residues, each carrying its four backbone atoms `N, Cα, C, O` at
definite positions in 3D Euclidean space. Inverse folding (computational protein design, the inverse of
structure prediction) asks: given only the backbone, output a per-residue distribution over the 20
amino acids that would fold into that shape, scored by native sequence recovery (fraction of residues
predicted correctly) and perplexity. The map is degenerate — many sequences fold to nearly the same
backbone — so the target is `p(s | X)`, and because a protein's identity is unchanged under a proper
rigid motion of the whole molecule, the prediction must be invariant to rotation and translation.

The component that decides accuracy is the **structure encoder** — the module that turns backbone
geometry into per-residue embeddings. The leading encoders attach a local coordinate frame to every
residue (built from `N, Cα, C`) and reason about the *relative* pose of neighboring residues' frames.
The question is how an encoder can use those frames more expressively to improve inverse folding.

## Background

Three things are established and load-bearing here.

**Frames from three backbone atoms (AlphaFold2).** Each residue is assigned a rigid transform
`T_i = (R_i, t_i)` with translation `t_i = Cα_i` and rotation `R_i = [e1, e2, e3] ∈ SO(3)` built by a
Gram–Schmidt process (`rigidFrom3Points`) from `N, Cα, C`: `e1 = norm(C−Cα)`,
`ẽ2 = (N−Cα) − (e1·(N−Cα)) e1`, `e2 = norm(ẽ2)`, `e3 = e1 × e2`. A point `x` in world coordinates maps
into residue `i`'s local frame as `T_i^{-1}∘x = R_i^T (x − t_i)`, and a point `q` given in `i`'s local
frame maps to world as `T_i∘q = R_i q + t_i`. Composing, the rigid transform that carries residue `j`'s
local frame into residue `i`'s is `T_{i←j} = T_i^{-1} ∘ T_j`. Anything computed from atoms expressed in
a *consistent* local frame is invariant to a global rigid motion, because the global rotation cancels in
`R_i^T R_g^T R_g (·) = R_i^T(·)`.

**Invariant Point Attention (IPA, AlphaFold2 / Jumper et al. 2021).** The standard frame-aware
attention. Each residue emits query/key/value *points* in its local frame; the geometric attention term
between residues `i` and `j` is built from the **squared distance** between `i`'s query points and `j`'s
key points after both are expressed in a common frame — i.e. the geometry is reduced to a scalar
(a sum of squared point distances) before it enters the network.

**Frame-anchored virtual atoms.** Rather than only using real backbone atoms, an encoder can carry a
set of *learnable* points (virtual atoms) whose coordinates are parameters expressed in each residue's
local frame; because they live in the frame, they rotate and translate with the residue, so distances
and frame-relative directions involving them stay invariant. Virtual atoms let the network learn a
transferable geometric "probe" of the local environment that real backbone atoms alone do not provide
(the side chain, which distinguishes amino acids, is not in the fixed-backbone input). PiFold (Gao et
al. 2023) uses fixed/learnable virtual atoms and reads their pairwise *distances*.

## Baselines

**IPA-based encoders (AlphaFold2; structure-prediction and design models built on IPA).** Frame-aware
attention whose geometric term is a scalar squared-distance pooling of query/key points.

**Distance-only graph encoders with virtual atoms (PiFold, Gao et al. 2023).** A `k`-NN graph with rich
*invariant scalar* features — multi-atom-pair RBF distances (real and virtual atoms), dihedrals,
frame-projected direction dot products — fed to attention-based message passing with edge updates and a
global context gate, decoded one-shot by a linear head. The virtual atoms contribute through their
pairwise *distances*.

**Geometric vector / equivariant encoders (GVP, Jing et al. 2021; tensor-field networks).** Carry vector
features through the graph and combine them only through rotation-commuting operations (channel-linear
maps, norms, norm-scalings) — GVP — or through spherical-harmonic tensor products — TFN.

## Evaluation settings

- **CATH 4.2 / 4.3 (inverse folding).** Structure-split single-chain design benchmarks; native sequence
  recovery (higher is better) and perplexity (lower is better) on held-out folds.
- **TS50.** 50 native/designed structures used as an out-of-distribution generalization check.
- **PDB-scale training.** The method also reports a larger model trained on the full PDB.
- **Metrics.** Primary: median per-structure native sequence recovery; secondary: per-residue perplexity.
- **Protocol.** `k`-nearest-neighbor graph by `Cα` distance; per-residue cross-entropy; one-shot
  (non-autoregressive) decoding; structure-based splits so test folds are unseen.

## Code framework

The structure encoder plugs into a fixed inverse-folding harness that supplies padded backbone
coordinates `X (B, L, 4, 3)` and a residue `mask (B, L)`, builds the `k`-NN graph, computes the
per-residue frames, runs the encoder to per-residue embeddings, and decodes them one-shot to amino-acid
log-probabilities under a masked cross-entropy loss. What is to be designed is the *layer primitive*:
how it computes geometric features between frame-anchored virtual atoms, and how those features update
the node, edge, and virtual-atom states.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_AA = 20

# ---- provided ----
def _rbf(D, ...):  # Gaussian radial basis encoding of distances
    ...
def rigid_frames(X):  # per-residue (R, t) via AlphaFold rigidFrom3Points from N, CA, C
    ...
def knn_graph(X_ca, mask, k=30):  # k-NN graph from CA coordinates
    ...


# ---- the primitive to be designed ----
class VFNLayer(nn.Module):
    """One vector-field-network layer. Computes geometric feature VECTORS between
    frame-anchored virtual atoms (not just their scalar distances), then updates the
    node / edge / virtual-atom states. Internals to be designed."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, h_V, h_E, Q, R, t, E_idx, mask):
        # TODO: express neighbor virtual atoms in the center frame; compute learnable
        #       vector combinations; stabilize to invariant scalars; attend; update.
        pass
```

The single empty slot is the vector-field operator and the layer it lives in.
