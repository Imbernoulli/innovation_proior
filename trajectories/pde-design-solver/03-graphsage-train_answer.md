The Graph U-Net confirmed the trade I was most worried about, and cleanly. The spatial hierarchy bought a better drag *ordering* — rho_d rose to 0.981, up from PointNet's 0.962, the coarse levels with growing radius indeed a better global summary than one max-pool — but it *cost* field accuracy on every front: Car pressure relative-L2 climbed to 0.113 (was 0.106), Car velocity to 0.050 (was 0.042), and AirfRANS got markedly worse, pressure 0.064 (was 0.039) and velocity a striking 0.099 (was 0.039), while c_d stayed roughly flat at 0.023. So the lossy-coarsening diagnosis was right: **random** node sampling over five scales throws away nodes blindly, and **nearest-neighbor unpooling** is piecewise-constant — every fine point in a coarse cell gets the same upsampled feature — which blurs exactly the sharp field gradients relative-L2 punishes. The AirfRANS collapse is the tell: in 2D the radius graph and interpolation are even blunter and the velocity error nearly tripled. The hierarchy reaches far, but it destroys local fidelity on the way down and does not fully rebuild it on the way up.

The lesson is precise: the *coarsening* is the weak link, not the message passing. I want to keep learned neighbor aggregation — the thing the U-Net got right at each level — but do it at **full resolution**, with **no random pooling and no interpolation loss**, so every point keeps its identity through the whole network and the field is never down-then-up-sampled. That is flat message passing: stack graph-convolution layers at the original mesh resolution, each layer letting a point aggregate from its neighbors, with depth giving reach ($k$ layers $\approx k$ hops) and nothing lossy anywhere. The trade is shorter reach, so very long-range coupling is weaker than the U-Net's coarse levels — but the U-Net just showed that its long-range reach was *not worth* the field accuracy it cost. On these design tasks the field errors and the drag-magnitude error are the fidelity signals I most need to drive down, and they live in *local* structure that a lossless flat stack should reconstruct far better. So this rung trades the U-Net's aggressive multi-scale reach for lossless full-resolution message passing, betting field accuracy is where the points are.

The choice of layer comes from asking what makes a graph convolution work as a per-node feature generator on a variable-size, hub-heavy mesh. I propose **GraphSAGE**. Its defining move, the one that separates it from a plain GCN, is to keep a node's own representation on a *separate channel* rather than folding it into the same averaging pot as its neighbors — which over $K$ layers would dilute the node's identity into a growing crowd of neighbors-of-neighbors, exactly the over-smoothing I want to avoid. The update is concat-then-transform,
$$h_v = \sigma\!\big(W \cdot [\,h_v \,;\, \mathrm{AGG}(h_u : u \in N(v))\,]\big),$$
a skip connection across depth that gives each node a clean, undiluted channel for its own signal at every layer. On a PDE mesh this matters: a point on a sharp pressure ridge must keep its own distinctive feature even as it aggregates from smoother neighbors, or the ridge gets averaged away — precisely the kind of gradient the U-Net's coarsening destroyed.

The code matches *this task's* `graphsage` variant, the design-task SAGE, not the full sample-and-aggregate algorithm of the inductive-embedding setting, and two differences are load-bearing. First, the aggregator: the most expressive choice is a per-neighbor-MLP-then-elementwise-max pool (a soft existential over the neighbor set), with mean as the cheap special case; this variant uses `torch_geometric.nn.SAGEConv`, whose default aggregation is the **mean** — a degree-mean neighbor summary, not the pool aggregator's existential semantics, so I do not claim the max-pool's ability to isolate a single distinctive neighbor. Second, and more important: SAGE's signature move is **fixed-size uniform neighbor sampling** (draw $S_k$ neighbors per node per layer) so per-batch cost is bounded regardless of hub degree; this harness does **no sampling** — `SAGEConv(z, edge_index)` aggregates over the *full* neighbor set every layer. That is affordable precisely because the meshes are modest (~5000–10000 points, batch one) and the radius graph the loop builds has bounded degree; the scaling problem that *motivated* sampling does not bite here. So the "inductive minibatching over a 200k-node graph" story is simply absent — and full-neighborhood aggregation is actually *more* faithful to the local field than a subsampled one would be, a virtue for field accuracy, not a compromise.

The plumbing: the encoder lifts the concatenated coordinates and features (`fun_dim + space_dim → n_hidden`) — note this *does* concatenate the coordinates, unlike the Graph U-Net which fed only `fun_dim`; here the point's position is part of its node feature, which is right for a flat model with no other geometric channel. Then a stack of `SAGEConv` layers all at width `n_hidden`: an `in_layer` from `n_hidden → n_hidden`, `n_layers − 1` hidden SAGE layers, then an `out_layer`, each followed by **BatchNorm** (`track_running_stats=False`, since batch is one mesh and running stats over single graphs would be meaningless) and a ReLU. The decoder maps `n_hidden → out_dim`. The whole thing runs on the squeezed $(N, C)$ tensor, re-adds the batch dimension at the end, and raises if `geo` is `None` like the other graph baselines.

The width is the canonical GraphSAGE setting and notably *larger* than the graph models before it: `CONFIG_OVERRIDES = {'n_hidden': 128}` — eight times PointNet's and Graph_UNet's 16. This is deliberate and part of why I expect SAGE to win on fields: a flat stack at full resolution with eight times the width has far more capacity to represent the local field, and spends none of it on lossy pooling.

The falsifiable bet: the whole point of going flat-and-lossless is field accuracy, so I expect the **field errors to drop below both prior rungs** — the U-Net's 0.113/0.050 on Car and especially its blown-out AirfRANS 0.064/0.099 should come down substantially, with AirfRANS recovering the most since that is where the geometric coarsening was bluntest. I also expect the **drag-magnitude error (c_d) to improve** below both 0.0247 and 0.0231, since a sharper lossless pressure field integrates to a more accurate coefficient. The one place I am *less* sure SAGE wins is the drag *rank correlation*: flat message passing reaches only `n_layers` hops, so its summary of front-to-back pressure asymmetry is weaker than the U-Net's explicit coarse levels — I would not be surprised if **rho_d comes in slightly below the U-Net's 0.981** even as everything else improves. If that is the pattern — SAGE better on c_d and all field errors, U-Net marginally better on rho_d alone — the verdict is that lossless local message passing dominates the lossy hierarchy on fidelity, and the remaining gap is the *global* correlation that even a deep flat SAGE cannot cheaply carry across the whole body. That gap is exactly what motivates leaving message passing behind for a whole-domain attention operator at the next rung.

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
