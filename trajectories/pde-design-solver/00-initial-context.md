## Research question

Industrial aerodynamic design: given a 3D unstructured point cloud of a body (a car, an
airfoil, an aircraft) with its boundary conditions baked in, predict the full steady flow field
at every mesh point in one forward pass — surface pressure and the surrounding velocity
components — so a designer can rank candidate shapes by drag without running a CFD solver. The
single thing being designed is the **neural operator itself**: the `Model` class in
`models/Custom.py`. Everything around it — the dataset loaders, the 200-epoch OneCycleLR
schedule, the loss, the metric computation, and a parameter-budget check — is frozen. Each mesh
has a *variable* number of points (~5000–10000), batch size is always 1 (one mesh per forward
pass), and the model must work across three benchmarks with different output widths (4 channels
for Car/AirfRANS, 6 for AirCraft).

## Prior art before the first rung (operator-on-geometry lineage)

The ladder lives inside the operator-learning frame: instead of solving one PDE instance, learn
the map from (geometry, boundary conditions) to the solution field, with each layer a non-local
integral operator followed by a pointwise nonlinearity. The rungs below are the families that
frame react to.

- **Fourier Neural Operator (Li et al. 2021) and geo-FNO (Li et al. 2022).** Parameterize the
  integral kernel in the Fourier domain — fixed basis, learnable spectral multipliers, truncated
  to low modes, evaluated by FFT in O(N log N); geo-FNO learns a deformation onto a latent
  uniform grid. **Gap:** the FFT *is* the periodic-uniform-grid assumption; on a car surface or
  an airfoil — irregular, non-periodic boundaries — the deformation degenerates, so fixed-basis
  spectral operators are not on this task's edit surface at all.
- **Graph-kernel neural operators / message passing on the mesh graph (Li et al. 2020).**
  Approximate each operator iteration by a learnable kernel over *local* graph neighborhoods.
  Handles arbitrary unstructured meshes — this is the family the graph baselines here belong to.
  **Gap:** the kernel is local, so carrying information from the nose of a car to its wake needs
  many message-passing steps, and global correlation is exactly what local kernels are worst at.
- **Attention as a learnable integral operator (Vaswani et al. 2017; Cao 2021; Kovachki et
  al.).** Softmax attention is a Monte-Carlo discretization of the integral operator with a
  *learned* kernel and the mesh points as quadrature nodes — the most expressive, most
  geometry-agnostic parameterization. **Gap:** with the N mesh points as nodes the cost is
  O(N²), infeasible at tens of thousands of points; even made linear it dilutes the physics
  across a sea of meaningless point-to-point relations.

## The fixed substrate

The training pipeline is frozen and must not be touched: the dataset loaders for Car
(ShapeNet-Car), AirfRANS (2D RANS airfoils), and AirCraft (a custom 3D probe), the 200-epoch
OneCycleLR schedule, the combined volume+surface relative-L2 loss, the drag-coefficient
integration and metric computation, and `budget_check.py`, which rejects any model whose
parameter count exceeds **1.05× the largest paper-faithful baseline** (Transolver at
`n_hidden=256, slice_num=32`). The loop hands the model `args` at construction with the relevant
knobs — `n_hidden`, `n_layers`, `n_heads`, `space_dim` (2 for AirfRANS, 3 otherwise),
`fun_dim=7`, `out_dim`, `act`, `mlp_ratio`, `dropout`, `geotype='unstructured'`, `radius` (for
graph construction), `slice_num` — and the read-only reference modules `layers.Basic.MLP`,
`layers.Embedding.unified_pos_embedding`, and `layers.Physics_Attention.*`. The shell scripts
default to `--n_hidden 128 --slice_num 32`.

## The editable interface

Exactly one region is editable: lines 1–64 of `models/Custom.py` (the imports through the end of
the `Model` class) plus line 74, the `CONFIG_OVERRIDES` dict. The contract is fixed —

- `Model(args)` constructs the network from `args`;
- `forward(self, x, fx, T=None, geo=None) -> output`, where `x` is `(1, N, space_dim)` spatial
  coordinates, `fx` is `(1, N, 7)` features (boundary conditions + geometry), `T` is always
  `None`, `geo` is the **edge_index** tensor for graph connectivity (graph models squeeze the
  batch dim and use it for message passing; non-graph models ignore it), and `output` is
  `(1, N, out_dim)`;
- `CONFIG_OVERRIDES` may set only `n_hidden` (int) and `slice_num` (int); the shell scripts read
  these from the file at runtime and pass them as `--n_hidden` / `--slice_num`. Different model
  families need different widths to be competitive and to stay under the parameter budget.

The starting point is the scaffold **default**: a per-point encoder MLP, *no operator in between*
(the body is a TODO), and a per-point decoder MLP. With nothing between encode and decode, the
default has no cross-point interaction at all — every point is predicted from its own features.
Each rung replaces the body of `Model` (and sets `CONFIG_OVERRIDES`) and nothing else.

```python
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from layers.Basic import MLP
from layers.Embedding import unified_pos_embedding


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Custom'
        self.args = args

        # Input encoding: spatial coords (3D) + features (7D) -> hidden_dim
        self.encoder = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                           n_layers=0, res=False, act=args.act)

        # TODO: Define your custom model architecture here.
        # This model operates on UNSTRUCTURED 3D point clouds (car meshes).
        # Each mesh has variable number of points (~5000-10000). Batch size is always 1.
        # args.geotype = 'unstructured'.
        # Reference models: PointNet (global pooling), GraphSAGE (message passing),
        # Graph_UNet (multi-scale graph), Transolver (physics attention).

        # Output projection: hidden_dim -> out_dim (velocity xyz + pressure)
        self.decoder = MLP(args.n_hidden, args.n_hidden * 2, args.out_dim,
                           n_layers=0, res=False, act=args.act)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx, T=None, geo=None):
        # x: (1, N, 3) spatial coords, fx: (1, N, 7) features
        # geo: edge_index tensor if graph connectivity is needed (can be None)
        z = torch.cat((x, fx), dim=-1)  # (1, N, 10)
        z = self.encoder(z)             # (1, N, n_hidden)

        # TODO: Implement your custom forward pass here.
        # Input z has shape (1, N, n_hidden) where N varies per mesh.
        # Output should have shape (1, N, out_dim).

        out = self.decoder(z)           # (1, N, out_dim)
        return out


# CONFIG_OVERRIDES: per-method hyperparameter overrides.
# Allowed keys: n_hidden (int), slice_num (int). Default follows the shell scripts
# (n_hidden=128, slice_num=32). Reference paper widths: PointNet=16, Graph_UNet=16,
# GraphSAGE=128, Transolver=256.
CONFIG_OVERRIDES = {}
```

## Evaluation settings

Three steady-aerodynamic benchmarks, each trained 200 epochs (OneCycleLR), seed 42:
**Car** (ShapeNet-Car), **AirfRANS** (2D RANS airfoils), and **AirCraft** (a custom 3D aircraft
probe with no published baseline; its numbers are task-internal references). Metrics, all
reported: **rho_d** — Spearman rank correlation of the drag coefficient (higher is better);
**c_d** — relative error of the drag coefficient (lower is better); and **relative L2 error**
of the pressure and velocity fields, reported per benchmark (lower is better). The
parameter-budget check runs before training; a model over 1.05× Transolver(256) is rejected.
