**Problem.** The scaffold default predicts every point's flow field from its own features alone —
zero cross-point coupling — but a PDE field is non-local (the wake depends on the nose). The first
rung must add the *cheapest* mechanism that lets a point's prediction depend on the rest of the mesh,
so that what it leaves on the table defines the next rung.

**Key idea (global max-pool set encoder).** Lift each point with a shared per-point MLP, take one
element-wise **max-pool over all N points** to get a single global descriptor (permutation-invariant
by construction; a provably sufficient summary for continuous set functions), then broadcast that
descriptor back to every point, concatenate it with the point's own local feature, and decode to the
output channels. Global coupling with no mesh graph — the leanest non-trivial floor.

**Why.** Max-pool is the cheapest symmetric reduction over an unordered set and its output is pinned
by a few "critical" points, so the encoder is the canonical permutation-invariant set summary. It
adds real cross-point signal over the empty default while staying edge-free, so locality is left as a
deliberate budget for the next rung.

**Match to the harness (not the paper).** This task's `pointnet` is a stripped design-task variant,
**not** full PointNet: no input/feature T-nets and no orthogonality regularizer (the meshes arrive in
a consistent physical frame and the loss is frozen), one *global* pool over the single mesh with the
batch dimension squeezed (`global_max_pool` with a zero batch vector), and the design-task channel
widths (`encoder→n_hidden`, `in_block→2·n_hidden`, `max_block→32·n_hidden`, `out_block` on the
`(2+32)·n_hidden` concat → `4·n_hidden`). It is registered as a graph model, so it raises if `geo` is
`None`, even though it never message-passes over the edges.

**Hyperparameters.** `CONFIG_OVERRIDES = {'n_hidden': 16}` (paper-faithful PointNet width; keeps it a
clean baseline and well under budget). `act` from `args`; `out_dim` per benchmark (4 Car/AirfRANS,
6 AirCraft). No new loss terms.

```python
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from layers.Embedding import unified_pos_embedding
from layers.Basic import MLP


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Custom'

        self.in_block = MLP(args.n_hidden, args.n_hidden * 2, args.n_hidden * 2, n_layers=0, res=False,
                            act=args.act)
        self.max_block = MLP(args.n_hidden * 2, args.n_hidden * 8, args.n_hidden * 32, n_layers=0, res=False,
                             act=args.act)

        self.out_block = MLP(args.n_hidden * (2 + 32), args.n_hidden * 16, args.n_hidden * 4, n_layers=0, res=False,
                             act=args.act)

        self.encoder = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden, n_layers=0, res=False,
                           act=args.act)
        self.decoder = MLP(args.n_hidden, args.n_hidden * 2, args.out_dim, n_layers=0, res=False, act=args.act)

        self.fcfinal = nn.Linear(args.n_hidden * 4, args.n_hidden)

    def forward(self, x, fx, T=None, geo=None):
        if geo is None:
            raise ValueError('Please provide edge index for Graph Neural Networks')
        z, batch = torch.cat((x, fx), dim=-1).float().squeeze(0), torch.zeros([x.shape[1]]).cuda().long()

        z = self.encoder(z)
        z = self.in_block(z)

        global_coef = self.max_block(z)
        global_coef = nng.global_max_pool(global_coef, batch=batch)
        nb_points = torch.zeros(global_coef.shape[0], device=z.device)

        for i in range(batch.max() + 1):
            nb_points[i] = (batch == i).sum()
        nb_points = nb_points.long()
        global_coef = torch.repeat_interleave(global_coef, nb_points, dim=0)

        z = torch.cat([z, global_coef], dim=1)
        z = self.out_block(z)
        z = self.fcfinal(z)
        z = self.decoder(z)

        return z.unsqueeze(0)


CONFIG_OVERRIDES = {'n_hidden': 16}
```
