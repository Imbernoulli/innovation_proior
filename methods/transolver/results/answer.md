# Transolver, distilled

Transolver is a Transformer-based neural operator for solving PDEs on general (including large,
unstructured) geometries. It keeps the canonical pre-norm residual Transformer architecture but
replaces full attention with **Physics-Attention**: instead of attending over the `N` mesh
points (`O(N^2)`, and prone to drowning physical correlations in low-level point relations), it
soft-assigns the points into `M` learnable *slices* — groups of points under a similar physical
state, of any shape and possibly spatially non-local — encodes each slice into one
physics-aware token, runs ordinary softmax attention among the `M` tokens, and broadcasts the
result back to the points. With `M` a small constant and `M ≪ N`, the operator is linear in `N`
(`O(NMC + M^2 C)`) and needs no mesh graph.

## Problem it solves

Learn a fast surrogate operator mapping a discretized geometry `g ∈ R^{N×C_g}` (and optional
observed quantities `u ∈ R^{N×C_u}`) to a solution field, on meshes with `N` from ~1k to ~32k
points and complex, non-periodic boundaries (cars, airfoils). The model must be cheap on large
irregular meshes, capture global/long-range physical correlations, and ingest arbitrary
geometry without assuming a grid — none of FNO/geo-FNO (grid/periodic), graph kernels (local),
or point-wise (linear) attention Transformers (attention still over massive points) does all
three.

## Key idea: Physics-Attention

The physics lives in *states*, not in the discretization; spatially distant points often share
a state (a car's windshield, plate, and headlights are all the drag-relevant front region).
Quadrature the integral operator over `M` learned states rather than `N` mesh points.

For a deep feature `x ∈ R^{N×C}`:

1. **Slice** — soft-assign each point across `M` slices with a softmax over the slice axis:
   `w_i = Softmax(Project(x_i)) ∈ R^{1×M}`, `Σ_j w_{i,j} = 1`. Softmax over slices gives a
   partition of unity per point and a sharpened (low-entropy) assignment so slices specialize
   into distinct physical states instead of collapsing to a domain-wide average.
2. **Encode tokens** — each slice becomes one physics-aware token by a mass-normalized weighted
   mean: `z_j = (Σ_i w_{i,j} x_i) / (Σ_i w_{i,j}) ∈ R^{1×C}` (normalize so a large slice does
   not get an inflated token; the implementation uses a `1e-5` denominator floor).
3. **Attention among tokens** — per head, `q,k,v = Linear(z)`,
   `z' = Softmax(q k^T / sqrt(d)) v`, `q,k,v,z' ∈ R^{M×d}`. Full softmax attention is kept
   (M is small, M^2 is cheap), modeling global state-to-state correlations.
4. **Deslice** — broadcast tokens back to points with the *same* weights:
   `x'_i = Σ_j w_{i,j} z'_j`. Tying slice and deslice weights makes the sandwich a single
   change of variables (see theorem).

Write `x' = Physics-Attn(x)`, complexity `O(NMC + M^2 C)`, linear in `N`. Multi-head: split the
`C` channels into `heads` subspaces and run an independent slicing/attention/deslicing per head,
so different heads learn different physical-state decompositions.

## Overall architecture

Canonical Transformer with full attention replaced by Physics-Attention; `L` pre-norm residual
layers:

```
x_hat^l = Physics-Attn(LayerNorm(x^{l-1})) + x^{l-1}
x^l     = FeedForward(LayerNorm(x_hat^l)) + x_hat^l
```

`x^0 = Linear(Concat(g, u))` (linear embedding of geometry ++ observed quantities); output is a
linear projection of `x^L`. FeedForward is the usual ~4×-wide MLP.

## Theorem (Physics-Attention is a learnable integral operator on Ω)

Physics-Attention approximates `G(u)(g*) = ∫_Ω κ(g*, ξ) u(ξ) dξ`. Derivation:

- **Lemma (attention = Monte-Carlo integral).** With the row-normalized attention kernel
  `κ(g*,ξ) = exp((W_q u(g*))(W_k u(ξ))^T / sqrt(d)) W_v /
  ∫_Ω exp((W_q u(g*))(W_k u(η))^T / sqrt(d)) dη`, discretizing both integrals by
  Monte-Carlo over the `N` mesh points yields
  `Σ_i [exp(q(g*) k(g_i)^T/sqrt(d)) / Σ_j exp(q(g*) k(g_j)^T/sqrt(d))] W_v u(g_i)`:
  the query `g*` is fixed and the softmax denominator sums only over key/value nodes.
- **Lemma (projection).** For countable Ω, define a countable slice-coordinate domain Ω_s and map
  each input point to an unclaimed slice coordinate in its block by maximum slice weight. Taking
  Ω_s as the claimed coordinates gives a bijection Ω ≅ Ω_s; with smooth slice weights on the
  continuations, the map `g` is a diffeomorphism.
- Define the slice-domain value function `u_s(ξ_s) = (∫_Ω w_{ξ,ξ_s} u(ξ) dξ)/(∫_Ω w_{ξ,ξ_s} dξ)`
  (the continuous token encoding). Change variables `ξ = g^{-1}(ξ_s)`, so
  `d g^{-1}(ξ_s) = |det ∇g^{-1}(ξ_s)| dξ_s`. The slice coordinates are implemented as
  permutation-invariant token slots with counting measure, so the measure-preserving
  simplification is `|det ∇g^{-1}| = 1`; otherwise this determinant would be carried in the
  measure or absorbed into the induced kernel. Writing the mesh→slice kernel as a `w`-weighted
  combination of the slice→slice kernel and using the slice-softmax partition of unity over the
  slice axis, `∫_{Ω_s} w_{g,ξ_s'} dξ_s' = 1`:
  `G(u)(g) = ∫_{Ω_s} w_{g,ξ_s'} [∫_{Ω_s} κ_ss(ξ_s', ξ_s) u_s(ξ_s) dξ_s] dξ_s'`,
  whose three factors are exactly **deslice** ∘ **attention among tokens** ∘ **token encoding**.
  Monte-Carlo over the `M` slices gives
  `G(u)(g_i) ≈ Σ_j w_{i,j} Σ_t [exp(q_j k_t^T/sqrt(d))/Σ_p exp(q_j k_p^T/sqrt(d))] v_t`:
  `j` is the output/query slice used by deslicing, `t` is the key/value slice, and `p` is the
  softmax normalization index for each fixed `j`.

So Physics-Attention is the same integral operator, evaluated by Monte-Carlo over the slice
domain rather than the mesh domain.

## Defaults and why

- **Slices `M`** (the one new hyperparameter): `M = 1` collapses Physics-Attention to global
  pooling with no state-to-state correlations; very large `M` over-fragments the physics domain
  into noisy tokens and drifts back toward point-level distraction. Useful range ~32–256, traded
  against width: `M = 64` for `C = 128`, `M = 32` for `C = 256` (keeps params/runtime comparable).
- **Heads = 8, layers `L = 8`, FFN width 4×**: carried over from the canonical Transformer (not
  where the contribution is) to keep comparisons fair.
- **Temperature `τ`** (learnable, per head, init 0.5; on structured paths clamped to [0.1, 5]):
  divides the slice logits before softmax; `τ < 1` sharpens the assignment.
- **`Project`**: a point-wise Linear for unstructured meshes (geometry-general); a local
  convolution with kernel size 3 for structured grids, so a point's assignment can see neighbors.
- **Init**: orthogonal init for the slice projection (decorrelate slices at start, faster
  specialization); truncated-normal (std 0.02) for other Linears; LayerNorm weight 1, bias 0.

## Working code

Physics-Attention for irregular meshes, the block, and the model, on the unstructured path
(point-wise `Project`, the design-task setting). `MLP(..., n_layers=0)` is
`Linear(input→hidden), act, Linear(hidden→output)`.

```python
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_


ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'ELU': nn.ELU,
    'silu': nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        act = ACTIVATION[act]
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class Physics_Attention_Irregular_Mesh(nn.Module):
    """Soft-slice N points into M physical-state tokens, attend among tokens, broadcast back."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)       # assignment stream
        self.in_project_fx = nn.Linear(dim, inner_dim)      # content stream
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape

        # (1) Slice + token encoding
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()                # B H N dim_head
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()                # B H N dim_head
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature)  # B H N M  (softmax over M)
        slice_norm = slice_weights.sum(2)                     # B H M
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None]
                                     .repeat(1, 1, 1, self.dim_head))   # mass-normalized mean

        # (2) Attention among M tokens
        q, k, v = self.to_q(slice_token), self.to_k(slice_token), self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.softmax(dots))
        out_slice_token = torch.matmul(attn, v)               # B H M dim_head

        # (3) Deslice with the same weights
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Transolver_block(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout=0., act='gelu', mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__name__ = 'Transolver'
        self.args = args
        self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                              n_layers=0, res=False, act=args.act)
        self.blocks = nn.ModuleList([
            Transolver_block(num_heads=args.n_heads, hidden_dim=args.n_hidden, dropout=args.dropout,
                             act=args.act, mlp_ratio=args.mlp_ratio, out_dim=args.out_dim,
                             slice_num=args.slice_num, last_layer=(i == args.n_layers - 1))
            for i in range(args.n_layers)])
        self.placeholder = nn.Parameter((1 / args.n_hidden) * torch.rand(args.n_hidden))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx, T=None, geo=None):
        # x: (1, N, space_dim) coords; fx: (1, N, fun_dim) features; geo (mesh graph) unused
        if fx is not None:
            fx = torch.cat((x, fx), dim=-1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx
```

For structured meshes, only `Project` changes: `in_project_x`/`in_project_fx` become a Conv1d/2d/3d
with kernel size 3 over the reshaped grid, and the temperature is clamped to `[0.1, 5]`; the
token-attention and deslice are identical. Typical config: `L = 8`, heads `= 8`, `C ∈ {128, 256}`,
`M = 64` for `C = 128` and `M = 32` for `C = 256`.
