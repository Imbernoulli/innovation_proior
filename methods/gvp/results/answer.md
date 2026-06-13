# Geometric Vector Perceptron (GVP / GVP-GNN), distilled

The Geometric Vector Perceptron is a drop-in replacement for a dense layer that operates on
a *tuple* of scalar and vector features `(s, V)`, `s ∈ ℝ^n`, `V ∈ ℝ^{ν×3}`. It produces
`(s', V')` such that the scalar output is **invariant** and the vector output is
**equivariant** under any rotation/reflection of the input coordinates, using only three
operations on vectors: bias-free channel-linear maps, the `L2` norm, and scaling by a
function of the norm. A graph neural network built from GVPs (the GVP-GNN) reasons about 3D
macromolecular structure both relationally (it is a message-passing GNN) and geometrically
(directional features stay as live vectors at every layer), at a cost that scales to whole
proteins — unlike voxel CNNs (geometric but non-relational, not invariant), invariant-scalar
GNNs (relational but geometry frozen into scalars at input), or SO(3)-irrep equivariant nets
(equivariant but too costly for large molecules).

## Problem it solves

Learning from protein backbone structure. The flagship task here is computational protein
design / inverse folding: given backbone coordinates `X` of shape `(L, 4, 3)` ([N, Cα, C, O])
and a residue mask, predict the amino-acid sequence (per-residue distribution over 20
classes). Scored by native sequence recovery (higher better) and perplexity (lower better).

## The GVP (Algorithm)

Input `(s, V)`, `s ∈ ℝ^n`, `V ∈ ℝ^{ν×3}`; output `(s', V')`, `s' ∈ ℝ^m`, `V' ∈ ℝ^{μ×3}`.
Let `h = max(ν, μ)`.

```
V_h   = W_h V                         in R^{h x 3}    # W_h: nu -> h, no bias
V_mu  = W_mu V_h                      in R^{mu x 3}   # W_mu: h -> mu, no bias
s_h   = ||V_h||  (row-wise)           in R^h          # norms feed the scalar path (invariant)
v_mu  = ||V_mu|| (row-wise)           in R^mu
s_m   = W_m concat(s_h, s) + b        in R^m          # scalar linear (biased)
s'    = sigma(s_m)                                    # scalar nonlinearity (e.g. ReLU)
V'    = sigma+(v_mu) (.) V_mu  (row-wise scaling)     # vector nonlinearity = scale by norm
return (s', V')
```

Learned weights: `W_h, W_mu` (bias-free, on vectors), `W_m, b` (biased, on scalars).

**Why each piece.** Vectors are touched only by `W_h, W_mu` (mix *channels*, not the 3 spatial
coordinates), by the `L2` norm, and by per-channel scaling — exactly the operations that
commute with right-multiplication by a unitary `U`. A vector bias is forbidden (no nonzero
constant vector is rotation-invariant); a coordinate-wise nonlinearity is forbidden (the
coordinates are axis-dependent). The norm `s_h` is the *only* bridge from vectors into the
scalar output, so `s'` is a function of both `s` and `V` yet stays invariant. The separate
`W_mu` decouples the number of norms extracted (`h`, feeds scalars) from the number of output
vectors (`μ`). `σ+` scales each output vector by a nonlinear function of its own norm — the
only equivariant vector nonlinearity.

## Properties

**Equivariance / invariance.** Write a rotation/reflection as right-multiplication of `V` by
unitary `U ∈ ℝ^{3×3}`. Then `‖W_h(VU)‖ = ‖(W_h V)U‖ = ‖W_h V‖`, so `s'` is unchanged
(invariant). For vectors, `V' = D W_μ W_h V` with `D = diag(σ+(‖V_μ‖))` an invariant diagonal
scaling, and `D W_μ W_h (VU) = (D W_μ W_h V)U`, so `V'` is equivariant.

**Universal approximation of invariant functions.** A GVP `G_s` with vector-only input,
scalar-only output, and sigmoidal `σ`, followed by a dense layer, can `ε`-approximate any
continuous rotation/reflection-invariant `F: Ω^ν → ℝ` (on the bounded set whose first three
vectors are linearly independent). Sketch: any invariant `F` factors as `F = F̃ ∘ ω` where the
orientation function `ω` extracts `3ν−3` invariant coordinates built from norms and inner
products of the `v_i`; inner products follow from norms by the cosine law
`v_i·v_j = (‖v_i‖² + ‖v_j‖² − ‖v_i − v_j‖²)/2`; choosing `W_h` so its rows are the `v_i` and
the Gram-Schmidt differences `v_i−v_1` for `i≥2`, `v_i−v_2` for `i≥3`, and `v_i−v_3` for
`i≥4` makes `s_h = ‖W_h V‖` supply the required `4ν−6` norms; the dense layer + Cybenko
(1989) universal approximation finishes.

## Supporting modules (equivariant)

- **Vector LayerNorm:** `V ← V / sqrt((1/ν)‖V‖²_F)` — scale rows to unit RMS norm (one
  invariant scale, no params). Scalars use ordinary LayerNorm (trainable `γ, β`).
- **Vector dropout:** drop entire vector *channels* at random (dropping coordinates would
  break equivariance). Scalars use ordinary dropout.

## GVP-GNN (message passing)

```
m_{j->i} = g( concat( h_v^{j}, h_e^{j->i}, h_v^{i} ) )        # g = stack of 3 GVPs
h_v^{i} <- LayerNorm( h_v^{i} + (1/k') Dropout( sum_j m_{j->i} ) )   # mean aggregation
h_v^{i} <- LayerNorm( h_v^{i} + Dropout( g(h_v^{i}) ) )       # g = 2 GVPs, hidden (4*s, 2*v)
```

Both scalar and vector channels are updated at every step, so vector features are *refined*
through depth (not frozen at input).

## Protein representation

`k = 30` nearest neighbors by `Cα` distance.

- **Node `(6, 3)`:** scalars = `{sin, cos}` of dihedrals `(φ, ψ, ω)`; vectors = forward unit
  vector `Cα_{i+1}−Cα_i`, reverse `Cα_{i−1}−Cα_i`, and imputed `Cβ−Cα`
  `√(2/3)(n×c)/‖n×c‖ − √(1/3)(n+c)/‖n+c‖` with `n = N−Cα, c = C−Cα` (equivalently
  `−√(1/3)·norm(c+n) − √(2/3)·norm(c×n)`, the canonical data-pipeline form).
- **Edge `(32, 1)`:** scalars = 16 Gaussian RBFs of `‖Cα_j−Cα_i‖` (centers 0–20 Å) + 16-dim
  sinusoidal encoding of `j−i`; vector = unit vector `Cα_j−Cα_i`.
- Hidden dims: node `(100, 16)`, edge `(32, 1)`; 3 encoder + 3 decoder layers; Adam.

## Design model (autoregressive)

`p(s | x) = ∏_i p(s_i | x, s_{<i})`. Encoder: 3 GVP layers on structure only. Inject
sequence: embed amino acids, append the *neighbor's* sequence embedding to each edge's scalars
but zero it whenever `j ≥ i` (causal mask). Decoder: 3 autoregressive GVP layers — backward
edges (`src ≥ dst`) form messages from the *encoder* embeddings, forward edges from live
decoder embeddings; sum and divide by neighbor count. Output `GVP((100,16) → (20,0))` logits,
with softmax/log-softmax applied for sampling or loss. Loss: per-residue cross-entropy.
Teacher-forced at training; sampled left-to-right at inference.

## Working code

```python
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


class GVP(nn.Module):
    """(s, V) -> (s', V'); scalar out invariant, vector out equivariant."""
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid)):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.scalar_act, self.vector_act = activations
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
        else:
            self.ws = nn.Linear(self.si, self.so)
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)                                  # V_h = W_h V
            vn = _norm_no_nan(vh, axis=-2)                   # ||V_h||
            s = self.ws(torch.cat([s, vn], -1))             # scalars see vector norms
            if self.vo:
                v = self.wv(vh)                              # V_mu = W_mu V_h
                v = torch.transpose(v, -1, -2)
                if self.vector_act:                          # V' = sigma+(||V_mu||) (.) V_mu
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        return (s, v) if self.vo else s


def tuple_sum(*args):
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    return x[0][idx], x[1][idx]

def _merge(s, v):
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)

def _split(x, nv):
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    return x[..., :-3*nv], v


class _VDropout(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.drop_rate) *
                               torch.ones(x.shape[:-1], device=self.dummy_param.device)).unsqueeze(-1)
        return mask * x / (1 - self.drop_rate)


class Dropout(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))      # RMS norm over channels
        return self.scalar_norm(s), v / vn


class GVPConv(MessagePassing):
    def __init__(self, in_dims, out_dims, edge_dims, n_layers=3, aggr="mean"):
        super().__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        GVP_ = functools.partial(GVP)
        module_list = [GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)]
        for _ in range(n_layers - 2):
            module_list.append(GVP_(out_dims, out_dims))
        module_list.append(GVP_(out_dims, out_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        x_s, x_v = x
        message = self.propagate(edge_index, s=x_s,
                                 v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                                 edge_attr=edge_attr)
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        m = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        return _merge(*self.message_func(m))


class GVPConvLayer(nn.Module):
    def __init__(self, node_dims, edge_dims, n_message=3, n_feedforward=2,
                 drop_rate=.1, autoregressive=False):
        super().__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                            aggr="add" if autoregressive else "mean")
        GVP_ = functools.partial(GVP)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])
        hid = (4 * node_dims[0], 2 * node_dims[1])
        self.ff_func = nn.Sequential(GVP_(node_dims, hid),
                                     GVP_(hid, node_dims, activations=(None, None)))

    def forward(self, x, edge_index, edge_attr, autoregressive_x=None, node_mask=None):
        if autoregressive_x is not None:
            src, dst = edge_index
            fwd = src < dst
            bwd = ~fwd
            dh = tuple_sum(
                self.conv(x, edge_index[:, fwd], tuple_index(edge_attr, fwd)),
                self.conv(autoregressive_x, edge_index[:, bwd], tuple_index(edge_attr, bwd)))
            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)
        else:
            dh = self.conv(x, edge_index, edge_attr)
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


class CPDModel(nn.Module):
    """GVP-GNN for structure-conditioned autoregressive protein design.
    node_in_dim=(6,3), node_h_dim=(100,16), edge_in_dim=(32,1), edge_h_dim=(32,1)."""
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.1):
        super().__init__()
        self.W_v = nn.Sequential(GVP(node_in_dim, node_h_dim, activations=(None, None)),
                                 LayerNorm(node_h_dim))
        self.W_e = nn.Sequential(GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
                                 LayerNorm(edge_h_dim))
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) for _ in range(num_layers))
        self.W_s = nn.Embedding(20, 20)
        edge_h_dim = (edge_h_dim[0] + 20, edge_h_dim[1])       # sequence appended to edge scalars
        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate, autoregressive=True)
            for _ in range(num_layers))
        self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None))

    def forward(self, h_V, edge_index, h_E, seq):
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        encoder_embeddings = h_V
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0               # causal: i sees only seq of j<i
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)
        return self.W_out(h_V)                                 # (n_nodes, 20) logits
```

## Load-bearing design roles

Replacing each tuple layer with an MLP on coordinates would either break invariance or force
the geometry back into input-only scalar features. Propagating only scalars removes live
directional state. Propagating only vectors discards scalar inputs such as dihedrals and
amino-acid identity, and it also removes the scalar output path used in the approximation
argument. Removing `W_mu` re-couples the number of extracted norms to the number of output
vectors. The dual scalar/vector tuple and the split `W_h`, `W_mu` design are the pieces that
keep geometry expressive while preserving the required symmetry.
