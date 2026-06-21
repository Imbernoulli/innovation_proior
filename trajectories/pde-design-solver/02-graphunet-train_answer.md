PointNet's numbers came in exactly as the single global bottleneck should produce them, with a clean split. On Car the drag *ordering* was strong — rho_d 0.962, the model can rank designs by their gross pressure asymmetry from one max-pool — but the drag *magnitude* was loose (c_d 0.0247) and the field errors were the worst I expect from anything with real spatial structure: Car pressure relative-L2 0.106, velocity 0.042; AirfRANS pressure 0.039, velocity 0.039. That is the global-bottleneck signature: a point on the stagnation nose and a point in the wake receive the *identical* global descriptor, so the model recovers the coarse front/back contrast (enough to rank drag) but cannot resolve the locally varying field the relative-L2 metrics reward. The diagnosis is sharp: the floor lacks *spatial locality*. Each point needs context from its actual mesh neighbors, not from one shape-wide vector — and the harness hands me the tool, the `geo` edge_index that PointNet ignored.

The cheapest use of those edges is message passing: each point pulls a feature from its neighbors and combines with its own. But flat message passing alone fixes the wrong half of the problem. A graph convolution at a single resolution mixes each node with its one-hop neighborhood, and $r$ layers reach $r$ hops; that repairs PointNet's *local* blindness, but to carry information from the nose of a car to its wake through one-hop steps I would need as many layers as the graph diameter, and repeated neighbor averaging pushes node features toward one another (over-smoothing), blurring exactly the distinctions the field reconstruction needs. Flat message passing therefore trades PointNet's "all-global, no-local" for "all-local, no-global." I want *both* — local message passing *and* a cheap path for information to travel across the whole body.

On images, the architecture that delivers both is the U-Net: shrink the grid while convolving so the deep representation sees large regions, then expand back to full resolution, copying high-resolution features across the bottleneck so the decoder can localize. Convolution has a graph analogue (message passing); the shrinking and expanding do not — and that missing piece is what gives long-range reach without diameter-many flat layers, because at a coarsened resolution a single hop spans a large region of the original mesh. So I propose a **Graph U-Net**: a multi-scale encoder-decoder over the mesh graph. At each scale it runs a graph convolution (local context), then coarsens the graph; after the bottleneck it uncoarsens back to full resolution, concatenating the saved encoder features across the bottleneck. The coarse levels are a learned *spatial* analogue of PointNet's global pool — they keep the long-range reach — while the per-scale convolutions supply the local context PointNet lacked.

The hard part is graph pooling: a graph has no canonical node ordering and no fixed rectangular windows, so I cannot "take every 2×2 block." The coarsening must be graph-structured, adaptive, and cheap — and here the code must match *this task's* `graphunet` variant, which is the design-task geometric one, not the gPool/gUnpool graph U-Net of the literature. Three differences carry the design. First, the pooling is **random node sampling by a fixed ratio**, not learned top-k-by-projection. The paper scores every node by a normalized scalar projection onto a learned direction, keeps the top $k$, and gates the survivors by their sigmoid scores so the projection gets a gradient; this variant instead samples `pool_ratio=0.5` of the nodes *uniformly at random* (`id_sampled = random.sample(range(n), k)`) and keeps those rows — no learned selection vector, no gate. The coarsening is stochastic and unlearned; only the convolutions are learned, a real capacity reduction versus the paper.

Second, and the geometric heart of the variant: after sampling, the coarse graph is **rebuilt by a radius graph in physical space**, not by restricting a power of the adjacency. The paper repairs connectivity lost to dropped nodes by squaring the self-looped adjacency so two-hop neighbors through a removed node stay connected; this variant instead calls `nng.radius_graph(pos_x, r=list_r[n], ...)` on the *positions* of the surviving points, with the radius growing across scales (`list_r = [0.05, 0.2, 0.5, 1, 10]`). Locality at each level is therefore *geometric* — who is within radius $r$ of whom in 3D — and the growing radius is what supplies long-range reach: at the coarsest level $r=10$ connects essentially everything, the spatial analogue of PointNet's global pool, but reached through a learned multi-scale stack rather than a single max. This is sensible precisely because the mesh carries real coordinates (`pos_x` is `x[:, :2]` at the top level, the sampled positions thereafter); a radius graph is the natural neighborhood on a metric point cloud.

Third, the unpooling is **nearest-neighbor interpolation in space**, not scatter-by-saved-index. The paper's gUnpool scatters the coarse rows back to their recorded indices and leaves dropped rows zero until a skip fills them; this variant, on the way up, assigns each fine point the feature of its *nearest* coarse point (`cluster = nng.nearest(pos_x_up, pos_x_down); x_up = x[cluster]`) — a geometric interpolation that fills *every* fine point, not just survivors — then concatenates that interpolated feature with the saved encoder feature at that level and runs a SAGE convolution. The skip is **concatenation** (`torch.cat([z, z_list[n-1]], dim=1)`), and the down path *doubles* the hidden width at each scale (`out_channels = 2 * size_hidden_layers`), so the up convolutions consume $3\cdot$`size_hidden_layers_init` channels — the interpolated coarse feature plus the saved finer feature. The convolutions are `SAGEConv` with `BatchNorm` and ReLU. Note the encoder lifts only `fun_dim` — *not* the coordinates (`x = fx.squeeze(0)`, encoder `MLP(args.fun_dim, ...)`); the coordinates enter only through the radius graph and the spatial interpolation, not the node features.

I expect this to beat PointNet on the global summary and to expose its own weak link. The multi-scale stack gives every point genuine local context (SAGE at each resolution) *and* long-range reach (the coarse levels with growing radius), so the **drag rank correlation should climb past PointNet's 0.962, plausibly above 0.98** — the spatial hierarchy is a better global summary than one max. But two things in *this* variant worry me. The pooling is *random*, so the coarsening throws away nodes blindly — at `n_hidden=16` and 0.5 ratios over five scales the coarsest graph is tiny and which points survive is luck, injecting variance that may *hurt* the fine-detail field reconstruction, especially velocity. And nearest-neighbor unpooling is piecewise-constant — every fine point in a coarse cell gets the *same* upsampled feature — which is geometrically crude and blurs sharp field gradients. So my falsifiable expectation is that the Graph U-Net **improves rho_d over PointNet but does not uniformly beat its field errors**; I would not be surprised if Car pressure relative-L2 *rises* above 0.106 and AirfRANS — a 2D case where the radius graph and interpolation are even blunter — looks worse on the fields. If that is the pattern, the lesson for the next rung is written: the *coarsening* is the weak link, so a method that does *flat*, learned, full-resolution neighbor aggregation — no random pooling, no interpolation loss — should reclaim the field accuracy the lossy hierarchy gave up.

The width is the paper-faithful Graph_UNet setting, `CONFIG_OVERRIDES = {'n_hidden': 16}`, with `scale=5`, `pool_ratio=0.5` per scale, `max_neighbors=64`, `layer='SAGE'`, BatchNorm on. Like the other graph baselines, it raises if `geo` is `None`.

```python
import torch
import torch.nn as nn
import torch_geometric.nn as nng
import random
from layers.Basic import MLP


def DownSample(id, x, edge_index, pos_x, pool, pool_ratio, r, max_neighbors):
    y = x.clone()
    n = int(x.size(0))

    if pool is not None:
        y, _, _, _, id_sampled, _ = pool(y, edge_index)
    else:
        k = int((pool_ratio * torch.tensor(n, dtype=torch.float)).ceil())
        id_sampled = random.sample(range(n), k)
        id_sampled = torch.tensor(id_sampled, dtype=torch.long)
        y = y[id_sampled]

    pos_x = pos_x[id_sampled]
    id.append(id_sampled)

    edge_index_sampled = nng.radius_graph(x=pos_x.detach(), r=r, loop=True, max_num_neighbors=max_neighbors)

    return y, edge_index_sampled


def UpSample(x, pos_x_up, pos_x_down):
    cluster = nng.nearest(pos_x_up, pos_x_down)
    x_up = x[cluster]

    return x_up


class Model(nn.Module):
    def __init__(self, args, pool='random', scale=5, list_r=[0.05, 0.2, 0.5, 1, 10],
                 pool_ratio=[0.5, 0.5, 0.5, 0.5, 0.5], max_neighbors=64, layer='SAGE', head=2):
        super(Model, self).__init__()
        self.__name__ = 'Custom'

        self.L = scale
        self.layer = layer
        self.pool_type = pool
        self.pool_ratio = pool_ratio
        self.list_r = list_r
        self.size_hidden_layers = args.n_hidden
        self.size_hidden_layers_init = args.n_hidden
        self.max_neighbors = max_neighbors
        self.dim_enc = args.n_hidden
        self.bn_bool = True
        self.res = False
        self.head = head
        self.activation = nn.ReLU()

        self.encoder = MLP(args.fun_dim, args.n_hidden * 2, args.n_hidden, n_layers=0, res=False,
                           act=args.act)
        self.decoder = MLP(args.n_hidden, args.n_hidden * 2, args.out_dim, n_layers=0, res=False, act=args.act)

        self.down_layers = nn.ModuleList()

        if self.pool_type != 'random':
            self.pool = nn.ModuleList()
        else:
            self.pool = None

        if self.layer == 'SAGE':
            self.down_layers.append(nng.SAGEConv(
                in_channels=self.dim_enc,
                out_channels=self.size_hidden_layers
            ))
            bn_in = self.size_hidden_layers

        elif self.layer == 'GAT':
            self.down_layers.append(nng.GATConv(
                in_channels=self.dim_enc,
                out_channels=self.size_hidden_layers,
                heads=self.head,
                add_self_loops=False,
                concat=True
            ))
            bn_in = self.head * self.size_hidden_layers

        if self.bn_bool == True:
            self.bn = nn.ModuleList()
            self.bn.append(nng.BatchNorm(
                in_channels=bn_in,
                track_running_stats=False
            ))
        else:
            self.bn = None

        for n in range(1, self.L):
            if self.pool_type != 'random':
                self.pool.append(nng.TopKPooling(
                    in_channels=self.size_hidden_layers,
                    ratio=self.pool_ratio[n - 1],
                    nonlinearity=torch.sigmoid
                ))

            if self.layer == 'SAGE':
                self.down_layers.append(nng.SAGEConv(
                    in_channels=self.size_hidden_layers,
                    out_channels=2 * self.size_hidden_layers,
                ))
                self.size_hidden_layers = 2 * self.size_hidden_layers
                bn_in = self.size_hidden_layers

            elif self.layer == 'GAT':
                self.down_layers.append(nng.GATConv(
                    in_channels=self.head * self.size_hidden_layers,
                    out_channels=self.size_hidden_layers,
                    heads=2,
                    add_self_loops=False,
                    concat=True
                ))

            if self.bn_bool == True:
                self.bn.append(nng.BatchNorm(
                    in_channels=bn_in,
                    track_running_stats=False
                ))

        self.up_layers = nn.ModuleList()

        if self.layer == 'SAGE':
            self.up_layers.append(nng.SAGEConv(
                in_channels=3 * self.size_hidden_layers_init,
                out_channels=self.dim_enc
            ))
            self.size_hidden_layers_init = 2 * self.size_hidden_layers_init

        elif self.layer == 'GAT':
            self.up_layers.append(nng.GATConv(
                in_channels=2 * self.head * self.size_hidden_layers,
                out_channels=self.dim_enc,
                heads=2,
                add_self_loops=False,
                concat=False
            ))

        if self.bn_bool == True:
            self.bn.append(nng.BatchNorm(
                in_channels=self.dim_enc,
                track_running_stats=False
            ))

        for n in range(1, self.L - 1):
            if self.layer == 'SAGE':
                self.up_layers.append(nng.SAGEConv(
                    in_channels=3 * self.size_hidden_layers_init,
                    out_channels=self.size_hidden_layers_init,
                ))
                bn_in = self.size_hidden_layers_init
                self.size_hidden_layers_init = 2 * self.size_hidden_layers_init

            elif self.layer == 'GAT':
                self.up_layers.append(nng.GATConv(
                    in_channels=2 * self.head * self.size_hidden_layers,
                    out_channels=self.size_hidden_layers,
                    heads=2,
                    add_self_loops=False,
                    concat=True
                ))

            if self.bn_bool == True:
                self.bn.append(nng.BatchNorm(
                    in_channels=bn_in,
                    track_running_stats=False
                ))

    def forward(self, x, fx, T=None, geo=None):
        if geo is None:
            raise ValueError('Please provide edge index for Graph Neural Networks')
        x, edge_index = fx.squeeze(0), geo
        id = []
        edge_index_list = [edge_index.clone()]
        pos_x_list = []
        z = self.encoder(x)
        if self.res:
            z_res = z.clone()

        z = self.down_layers[0](z, edge_index)

        if self.bn_bool == True:
            z = self.bn[0](z)

        z = self.activation(z)
        z_list = [z.clone()]
        for n in range(self.L - 1):
            pos_x = x[:, :2] if n == 0 else pos_x[id[n - 1]]
            pos_x_list.append(pos_x.clone())

            if self.pool_type != 'random':
                z, edge_index = DownSample(id, z, edge_index, pos_x, self.pool[n], self.pool_ratio[n], self.list_r[n],
                                           self.max_neighbors)
            else:
                z, edge_index = DownSample(id, z, edge_index, pos_x, None, self.pool_ratio[n], self.list_r[n],
                                           self.max_neighbors)
            edge_index_list.append(edge_index.clone())

            z = self.down_layers[n + 1](z, edge_index)

            if self.bn_bool == True:
                z = self.bn[n + 1](z)

            z = self.activation(z)
            z_list.append(z.clone())
        pos_x_list.append(pos_x[id[-1]].clone())

        for n in range(self.L - 1, 0, -1):
            z = UpSample(z, pos_x_list[n - 1], pos_x_list[n])
            z = torch.cat([z, z_list[n - 1]], dim=1)
            z = self.up_layers[n - 1](z, edge_index_list[n - 1])

            if self.bn_bool == True:
                z = self.bn[self.L + n - 1](z)

            z = self.activation(z) if n != 1 else z

        del (z_list, pos_x_list, edge_index_list)

        if self.res:
            z = z + z_res

        z = self.decoder(z)

        return z.unsqueeze(0)


CONFIG_OVERRIDES = {'n_hidden': 16}
```
