**Problem.** The graph U-Net's multi-scale hierarchy raised the drag *ordering* (rho_d 0.981) but its
random pooling and piecewise-constant unpooling *destroyed* local field fidelity — every field error
got worse than PointNet (Car 0.113/0.050, AirfRANS 0.064/0.099). The rung must reclaim field accuracy
by keeping learned neighbor aggregation but doing it **losslessly, at full resolution** — no pooling,
no interpolation.

**Key idea (flat GraphSAGE message passing).** Stack graph-convolution layers at the original mesh
resolution: each layer lets a point aggregate from its mesh neighbors and combines that with its own
carried-forward feature, `h_v = σ(W·[h_v ; AGG(h_u : u∈N(v))])`. The separate self-channel
(concat-then-transform) gives each point an undiluted channel for its own signal, resisting the
over-smoothing that would average sharp pressure ridges away. Depth gives reach; nothing is ever
down/up-sampled, so local structure survives intact.

**Match to the harness (not the paper).** This is the design-task SAGE variant, **not** the full
sample-and-aggregate algorithm: (1) `torch_geometric.nn.SAGEConv` uses **mean** aggregation, not the
paper's per-neighbor-MLP-then-max pool; (2) **no neighbor sampling** — it aggregates the *full*
neighbor set every layer (affordable at ~5000–10000 points, batch 1; the paper's 200k-node scaling
motivation is absent), which is more faithful to the local field than subsampling; (3) the encoder
concatenates **coordinates + features** into the node feature (the U-Net fed only `fun_dim`); (4)
BatchNorm with `track_running_stats=False` (batch is one mesh) + ReLU per layer.

**Why.** Lossless full-resolution message passing spends no capacity on lossy coarsening, and the
self-channel keeps local fidelity — the metric that the lossy hierarchy gave up. The trade is shorter
reach (k layers ≈ k hops), so the global drag *ordering* may dip slightly vs the U-Net, but field
errors and drag *magnitude* should improve.

**Hyperparameters.** `CONFIG_OVERRIDES = {'n_hidden': 128}` — the paper-faithful GraphSAGE width, 8×
the prior graph models, the capacity that lets a flat stack represent the field. `n_layers`, `act`,
`out_dim` from `args`. No new loss terms.

```python
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from layers.Basic import MLP


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Custom'

        self.nb_hidden_layers = args.n_layers
        self.size_hidden_layers = args.n_hidden
        self.bn_bool = True
        self.activation = nn.ReLU()

        self.encoder = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden, n_layers=0, res=False,
                           act=args.act)
        self.decoder = MLP(args.n_hidden, args.n_hidden * 2, args.out_dim, n_layers=0, res=False, act=args.act)

        self.in_layer = nng.SAGEConv(
            in_channels=args.n_hidden,
            out_channels=self.size_hidden_layers
        )

        self.hidden_layers = nn.ModuleList()
        for n in range(self.nb_hidden_layers - 1):
            self.hidden_layers.append(nng.SAGEConv(
                in_channels=self.size_hidden_layers,
                out_channels=self.size_hidden_layers
            ))

        self.out_layer = nng.SAGEConv(
            in_channels=self.size_hidden_layers,
            out_channels=self.size_hidden_layers
        )

        if self.bn_bool:
            self.bn = nn.ModuleList()
            for n in range(self.nb_hidden_layers):
                self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False))

    def forward(self, x, fx, T=None, geo=None):
        if geo is None:
            raise ValueError('Please provide edge index for Graph Neural Networks')
        z, edge_index = torch.cat((x, fx), dim=-1).squeeze(0), geo
        z = self.encoder(z)
        z = self.in_layer(z, edge_index)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)

        for n in range(self.nb_hidden_layers - 1):
            z = self.hidden_layers[n](z, edge_index)
            if self.bn_bool:
                z = self.bn[n + 1](z)
            z = self.activation(z)
        z = self.out_layer(z, edge_index)
        z = self.decoder(z)
        return z.unsqueeze(0)


CONFIG_OVERRIDES = {'n_hidden': 128}
```
