## Research question

Industrial aerodynamic design: given a 3D unstructured point cloud of a body with boundary
conditions baked in, predict the steady flow field at every mesh point in one forward pass —
surface pressure and surrounding velocity — so a designer can rank candidate shapes by drag
without running a CFD solver. The only editable piece is the neural operator itself, the
`Model` class in `models/Custom.py`. Everything else is frozen: the dataset loaders, the
200-epoch OneCycleLR schedule, the combined volume+surface relative-L2 loss, the drag metrics,
and the parameter-budget check. Each mesh has a variable number of points (~5000–10000), batch
size is 1, and the model must work across three benchmarks with different output widths (4
channels for Car/AirfRANS, 6 for AirCraft).

## Prior art / Background / Baselines

The task is operator learning: learn the map from (geometry, boundary conditions) to the
solution field, with each layer a non-local integral operator followed by a pointwise
nonlinearity. The relevant families are:

- **Fourier Neural Operator (FNO) and geo-FNO (Li et al. 2021; 2022).** They parameterize the
  integral kernel in the Fourier domain with fixed basis and learnable spectral multipliers,
  evaluated by FFT.
- **Graph-kernel neural operators / message passing on the mesh graph (Li et al. 2020).** They
  approximate operator iterations by learnable kernels over local graph neighborhoods and handle
  arbitrary unstructured meshes.
- **Attention-based neural operators (Vaswani et al. 2017).** They treat the mesh points as
  quadrature nodes for a learned integral kernel, giving an expressive, geometry-agnostic
  parameterization.

## Fixed substrate / Code framework

The training pipeline is frozen and must not be touched: dataset loaders for Car
(ShapeNet-Car), AirfRANS (2D RANS airfoils), and AirCraft (custom 3D aircraft probe), the
200-epoch OneCycleLR schedule, the combined volume+surface relative-L2 loss, the drag
integration and metrics, and `budget_check.py`, which rejects any model whose parameter count
exceeds **1.05× the largest paper-faithful baseline** (Transolver at `n_hidden=256,
slice_num=32`). At construction the model receives `args` with `n_hidden`, `n_layers`,
`n_heads`, `space_dim` (2 for AirfRANS, 3 otherwise), `fun_dim=7`, `out_dim`, `act`,
`mlp_ratio`, `dropout`, `geotype='unstructured'`, `radius`, and `slice_num`, plus the read-only
reference modules `layers.Basic.MLP`, `layers.Embedding.unified_pos_embedding`, and
`layers.Physics_Attention.*`. The shell scripts default to `--n_hidden 128 --slice_num 32`.

## Editable interface

Only one region is editable: lines 1–64 of `models/Custom.py` (the imports through the end of
the `Model` class) plus line 74, the `CONFIG_OVERRIDES` dict. The contract is fixed:

- `Model(args)` builds the network from `args`;
- `forward(self, x, fx, T=None, geo=None) -> output`, where `x` is `(1, N, space_dim)`
  spatial coordinates, `fx` is `(1, N, 7)` features, `T` is always `None`, `geo` is the
  **edge_index** tensor for graph connectivity (graph models use it; non-graph models ignore
  it), and `output` is `(1, N, out_dim)`;
- `CONFIG_OVERRIDES` may set only `n_hidden` (int) and `slice_num` (int); the shell scripts
  read these at runtime and pass them as `--n_hidden` / `--slice_num`.

The default scaffold is a per-point encoder MLP, no operator body, and a per-point decoder MLP,
so there is no cross-point interaction at all. Each entry replaces the body of `Model` and sets
`CONFIG_OVERRIDES`, nothing else.

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
        # Graph_UNet (multi-scale graph).

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
# GraphSAGE=128.
CONFIG_OVERRIDES = {}
```

## Evaluation settings

Three steady-aerodynamic benchmarks, each trained 200 epochs (OneCycleLR), seed 42: **Car**
(ShapeNet-Car), **AirfRANS** (2D RANS airfoils), and **AirCraft** (custom 3D aircraft probe
with no published baseline; its numbers are task-internal references). Metrics, all reported:
**rho_d** — Spearman rank correlation of the drag coefficient (higher is better); **c_d** —
relative error of the drag coefficient (lower is better); and **relative L2 error** of the
pressure and velocity fields, reported per benchmark (lower is better). The parameter-budget
check runs before training; a model over 1.05× Transolver(256) is rejected.
