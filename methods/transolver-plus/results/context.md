## Research question

Neural PDE solvers are being pushed from academic meshes with thousands or tens of
thousands of points toward industrial geometries with hundreds of thousands to
millions of points. The useful target is a solver that learns a map from geometry,
boundary or condition features, and observed fields to the solution field while
remaining accurate on irregular meshes such as car bodies and aircraft surfaces.
The hard constraint is that the point count `N` is no longer a mild nuisance: a
single layer may have to touch 1e6 to 2.5e6 points, and the solver must not fall
back to quadratic attention over the point set.

The strongest available direction is to avoid point-to-point attention by grouping
points into a small set of learned physical-state tokens. That keeps the number of
global tokens `M` small, usually 32 or 64, and makes the expensive token
interaction nearly independent of `N`.

The question is: how can one extend the slice-attention operator's linear-in-`N`
structure to industrial-scale meshes with millions of points?

## Background

Operator learning frames PDE solving as learning a map between functions. A common
layer is a nonlocal integral operator followed by a pointwise nonlinearity,
approximating a kernel interaction over discretized points. Standard attention is
a flexible parameterization of such a global interaction, but full attention over
`N` mesh points costs `O(N^2)` and is impractical for high-resolution geometry.
Linear-attention neural operators reduce the cost, but they still operate directly
on noisy point tokens rather than on a smaller set of physically meaningful
representatives.

The prior slice-attention solver changes the quadrature nodes. For each point
feature `x_i in R^C`, it learns a weight vector `w_i in R^M` over `M` physical
states by applying a linear projection and a softmax along the state dimension.
Then it forms state tokens as weighted means:

```text
w = Softmax(Linear(x) / tau0)
s_j = sum_i w_ij x_i / sum_i w_ij
```

Full attention is applied only among the `M` state tokens, and the updated states
are broadcast back to points by the same slice weights:

```text
q, k, v = Linear(s)
s' = Softmax(q k^T / sqrt(C)) v
x'_i = sum_j w_ij s'_j
```

Its complexity is `O(N M C + M^2 C)`, which is linear in `N` when `M` is fixed.
The original analysis relates this state-token operator to an integral operator
over a learned slice domain, so the state-token view is not merely an engineering
compression trick.

## Baselines

**Slice-attention neural operator.** It learns physical slices, averages point
features into `M` tokens, attends among tokens, and deslices back to points. Its
main advantage is linear complexity in the point count and strong behavior on
irregular geometry.

**Linear-attention PDE Transformers.** Galerkin-style attention, GNOT, and related
methods reduce attention complexity by changing the attention algebra. They are
efficient and easy to batch, and they treat mesh points as the attention tokens.

**Fourier and graph neural operators.** Fourier methods are powerful on regular or
latent grids, while graph neural operators and mesh networks naturally handle
unstructured meshes. Graph models support local message passing between neighboring
points and can aggregate information across the mesh through successive layers.

**Generic parallelism for long sequences.** Tensor parallelism, pipeline/model
parallelism, RingAttention, and Ulysses-style feature partitioning reduce memory
or distribute attention across devices.

## Evaluation settings

The standard benchmark suite contains six PDE tasks: Elasticity, Plasticity,
Airfoil, Pipe, Navier-Stokes, and Darcy. Relative L2 error is the primary metric.
The usual model budget uses 8 heads, feature width 128 or 256, 4 to 8 operator
blocks depending on the dataset, and 32 or 64 learned states. Training uses
relative L2 loss with Adam or AdamW and comparable parameter budgets across
Transformer-style baselines.

The industrial tests are the important scaling target. Car and aircraft
benchmarks include DrivAerNet++ and an aircraft CFD dataset. They contain
surface-only settings and full 3D field settings; the largest cases reach roughly
2.5 million mesh points. Metrics include field relative L2 on volume and surface
fields, coefficient error for drag or lift, and coefficient fit quality such as
`R^2`. The coefficient signs follow the aerodynamic force convention:

```text
C = 1 / (0.5 rho v_inf^2 A) * int_S (-p n dot d + tau n dot d) dS
```

where `d` is the drag or lift direction, `p` is surface pressure, `n` is the
outward unit normal, and `tau` is the shear stress tensor. The pressure term has
the negative sign because pressure acts opposite the outward normal.

## Code framework

The available implementation scaffold is a neural-operator backbone: preprocess
per-point inputs into a hidden width, apply repeated pre-norm residual operator
blocks, and return per-point output features. The open slot is the attention-like
operator inside each block.

```python
import torch
import torch.nn as nn


ACTIVATION = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=0, act="gelu", res=False):
        super().__init__()
        act_layer = ACTIVATION[act]
        self.pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_layer())
        self.mid = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_layer())
             for _ in range(n_layers)]
        )
        self.post = nn.Linear(n_hidden, n_output)
        self.res = res

    def forward(self, x):
        x = self.pre(x)
        for layer in self.mid:
            y = layer(x)
            x = x + y if self.res else y
        return self.post(x)


class OperatorBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4, act="gelu", dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.operator = None  # TODO: fill with the scalable state-token operator.
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, act=act)

    def forward(self, x):
        x = self.operator(self.ln_1(x)) + x
        x = self.ffn(self.ln_2(x)) + x
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = MLP(
            args.fun_dim + args.space_dim,
            args.n_hidden * 2,
            args.n_hidden,
            act=args.act,
        )
        self.layers = nn.ModuleList(
            [OperatorBlock(args.n_hidden, args.n_heads, args.mlp_ratio,
                           args.act, args.dropout)
             for _ in range(args.n_layers)]
        )
        self.head = nn.Linear(args.n_hidden, args.out_dim)

    def forward(self, x, fx):
        h = self.embed(torch.cat((x, fx), dim=-1))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)
```

The embedding, residual layout, feed-forward block, and readout are fixed. The
empty slot must implement the slice-attention operator within this backbone.
