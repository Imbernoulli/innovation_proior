# Context: learning fast neural operators for PDEs on large, irregular meshes (circa 2023)

## Research question

Solving a partial differential equation numerically means discretizing a domain
`Omega ⊂ R^{C_g}` into a finite set of `N` mesh points `g ∈ R^{N×C_g}` and computing target
physical quantities at those points — surface pressure and surrounding velocity around a car,
inner stress in a loaded material, the future of a fluid field. Classical solvers do this
accurately but slowly: a single complex structure can take hours or days. The goal is a
*learned* surrogate operator that, given the discretized geometry `g` (and any observed input
quantities `u ∈ R^{N×C_u}`), predicts the solution field in one fast forward pass, and that
does so on the meshes practitioners actually use — `N` from about a thousand up to thirty-odd
thousand points, arranged on **unstructured** meshes with **complex, non-periodic boundaries**
(a car body, an airfoil), not just on tidy uniform grids.

The question is how to build a neural operator layer that is both efficient on large irregular
meshes and capable of capturing long-range physical correlations without assuming any particular
grid structure.

## Background

The dominant framing is **operator learning**: instead of solving one PDE instance, learn the
mapping from input functions (geometry, boundary/initial conditions, coefficients) to the
solution function. Li et al. (2020), in the graph-kernel "Neural Operator," formalized PDE
solving as an iterative architecture whose layers are integral operators; Li et al. (2021),
with the Fourier Neural Operator, refined each iteration into the composition of a *non-local
integral operator* and a *local nonlinear activation*. Since the activation is just a
pointwise feed-forward layer, the load-bearing object to learn is the **non-local integral
operator**

```
G(u)(g*) = ∫_Omega  kappa(g*, xi)  u(xi)  d xi,
```

with `kappa` a kernel on `Omega × Omega`. Different neural operators are different ways to
parameterize and evaluate this integral.

A second thread connects this integral to the **Transformer**. Vaswani et al. (2017) built the
Transformer on scaled dot-product attention, `softmax(Q K^T / sqrt(d)) V`, stacked in
pre-norm residual blocks (`x + Attn(LayerNorm(x))`, then `x + FFN(LayerNorm(x))`), with a
multi-head split over channel subspaces and a roughly 4×-wide feed-forward sublayer. Cao
(2021), in the Galerkin/Fourier Transformer line, and Kovachki et al. made the link precise:
attention is a **Monte-Carlo approximation of an integral operator with a learnable kernel**,
using the input tokens themselves as the quadrature nodes. Where FNO fixes the kernel to a
truncated Fourier basis, attention *learns* the kernel from data.

## Baselines

These are the prior approaches a new operator would be measured against and would react to.

**Canonical Transformer / attention as a learnable integral operator (Vaswani et al. 2017;
Cao 2021; Kovachki et al.).** `q,k,v = Linear(x)`, `Attn(x) = softmax(q k^T / sqrt(d)) v`,
multi-head over channel subspaces, in pre-norm residual blocks with a wide FFN. Interpreted as
operator learning, attention computes the integral `G(u)(g*)` by Monte-Carlo quadrature with
the `N` input points as nodes and a row-normalized kernel `kappa` built from
`exp((W_q u(g*))(W_k u(g_i))^T / sqrt(d))`, with the softmax denominator summing over
key/value nodes for the fixed query. It is a *learnable* kernel, more flexible than FNO's fixed
Fourier multipliers.

**Fourier Neural Operator and geometry-aware variants (Li et al. 2021; Li et al. 2022).** FNO
parameterizes the kernel in the Fourier domain — a fixed basis with learnable spectral
multipliers, truncated to low modes — and evaluates it with the FFT in `O(N log N)`. geo-FNO
learns a deformation that maps an irregular input domain onto a uniform latent grid, runs FNO
there, and maps back.

**Graph-kernel neural operator (Li et al. 2020).** Casts each operator iteration as an integral
approximated by a learnable kernel over *local* graph neighborhoods — message passing on the
mesh graph. Handles arbitrary unstructured meshes.

**Linear-attention PDE Transformers (Cao 2021; Li et al. 2023, OFformer; Hao et al. 2023,
GNOT).** To reduce the cost of attention, these replace softmax attention with linear-complexity
attention (Galerkin-type `Q(K^T V)`, Performer, Reformer, etc.), so the integral is evaluated
in `O(N)`. GNOT was the prior best on standard benchmarks.

**Patch-based Vision Transformers (Dosovitskiy et al. 2021, ViT; Liu et al. 2021, Swin).**
Patchify reduces token count and injects local context by grouping a regular square block of
pixels into one token.

## Evaluation settings

The natural yardsticks already in use, spanning point clouds, structured meshes, regular grids,
and unstructured meshes in 2D and 3D:

- **FNO / geo-FNO standard benchmarks**: Elasticity (point cloud, 972 points, predict inner
  stress), Plasticity (structured mesh, 3,131 points, time-dependent displacement), Airfoil
  (11,271 points, Mach number), Pipe (16,641 points, velocity), Navier–Stokes (64×64 regular
  grid, autoregressive velocity), Darcy (downsampled 85×85 grid, pressure). Train/test splits
  on the order of (1000, 200).
- **Practical design tasks on unstructured meshes**: **Shape-Net Car** (Umetani & Bickel 2018;
  car shapes from ShapeNet, 32,186-point unstructured mesh simulating 72 km/h driving; input is
  point position, signed-distance function, and surface normal; predict surface pressure and
  surrounding air velocity; preprocessed split 789 train / 100 test). **AirfRANS** (Bonnet et
  al. 2022; Reynolds-Averaged Navier–Stokes on NACA 4- and 5-digit airfoils across varying
  shape, Reynolds number, and angle of attack; 32,000-point mesh; 800 train / 200 test;
  velocity, pressure, viscosity recorded).
- **Metrics.** Relative L2 error between the predicted and ground-truth physics field,
  `||u - u_hat|| / ||u||`. For design tasks, the drag/lift coefficient is integrated from the
  predicted surface fields, `C = (2 / (v^2 A)) ( ∫_∂Ω p (n_hat · i_hat) dξ + ∫_∂Ω τ · i_hat dξ )`
  (`v` inlet speed, `A` reference area, `n_hat` outward normal, `i_hat` inlet direction, `τ`
  wall shear), and reported as its relative L2 error and as the Spearman rank correlation
  `ρ = cov(R(C), R(C_hat)) / (σ_{R(C)} σ_{R(C_hat)})` between predicted and true coefficients
  across the test set — a high rank correlation means the surrogate orders candidate designs
  correctly.
- **Protocol.** Stack of `L` layers (commonly 8), hidden width `C` of 128 or 256 depending on
  input dimensionality, multi-head attention with 8 heads, relative-L2 training loss (a spatial
  gradient regularizer is added for Darcy; volume and surface losses are summed for the design
  tasks), Adam/AdamW, identical training configuration across all baselines for fairness, run on
  a single A100 and repeated three times.

## Code framework

The new operator plugs into the same neural-operator training harness used for the baselines:
an embedding that lifts the raw per-point inputs into a hidden feature, a stack of identical
operator layers, and a linear read-out — trained against the relative-L2 loss. Everything about
the operator sublayer is still open. The substrate below is only the generic machinery already
available: a standard pre-norm residual layer with one neutral empty slot between normalization
and residual addition, plus the feed-forward sublayer and read-out.

```python
import torch
import torch.nn as nn


ACTIVATION = {'gelu': nn.GELU, 'relu': nn.ReLU, 'silu': nn.SiLU}  # etc.


class MLP(nn.Module):
    """Generic point-wise MLP used for embedding, the feed-forward sublayer, and read-out."""

    def __init__(self, n_in, n_hidden, n_out, n_layers=0, act='gelu', res=False):
        super().__init__()
        act = ACTIVATION[act]
        self.pre = nn.Sequential(nn.Linear(n_in, n_hidden), act())
        self.mid = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])
        self.post = nn.Linear(n_hidden, n_out)
        self.res = res

    def forward(self, x):
        x = self.pre(x)
        for layer in self.mid:
            x = layer(x) + x if self.res else layer(x)
        return self.post(x)


class OperatorLayer(nn.Module):
    """One pre-norm residual operator layer with one empty operator slot."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4, act='gelu', dropout=0.):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        # TODO: fill in the operator sublayer.
        self.op = None
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, act=act)

    def forward(self, x):
        # x: (B, N, C) features on N mesh points
        x = self.op(self.ln_1(x)) + x
        x = self.ffn(self.ln_2(x)) + x     # feed-forward sublayer
        return x


class Model(nn.Module):
    """Neural-operator backbone: embed raw inputs, stack L operator layers, read out."""

    def __init__(self, args):
        super().__init__()
        # lift per-point inputs (coordinates concatenated with observed quantities) to width C
        self.embed = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                         n_layers=0, act=args.act)
        self.layers = nn.ModuleList([
            OperatorLayer(args.n_hidden, args.n_heads, args.mlp_ratio, args.act, args.dropout)
            for _ in range(args.n_layers)])
        self.head = nn.Linear(args.n_hidden, args.out_dim)

    def forward(self, x, fx, T=None, geo=None):
        # x: (1, N, space_dim) coordinates; fx: (1, N, fun_dim) input features
        h = self.embed(torch.cat((x, fx), dim=-1))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)
```

The embedding, residual layer recipe, feed-forward sublayer, and read-out are fixed and generic.
The single empty slot is `OperatorLayer.op`; the reasoning has to fill it with an operator that
scales to large unstructured meshes without giving up non-local correlations.
