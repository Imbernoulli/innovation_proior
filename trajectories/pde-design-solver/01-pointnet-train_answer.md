The scaffold default predicts every point's flow field from its own ten inputs — three coordinates and seven features — straight to four output channels, with no point ever seeing any other point. But a PDE field is the opposite of pointwise: the pressure at the rear of a car is set by the geometry at its nose, the velocity in the wake by the whole upstream body. A model with zero cross-point coupling cannot represent that at all; it can only learn a local regression from "these are my boundary conditions" to "this is my field value," and on steady aerodynamics that is a ceiling. So the first rung answers one question — what is the *cheapest* mechanism that lets a point's prediction depend on the rest of the mesh? — and it should answer it with the least machinery possible, so that whatever it leaves on the table tells me what the next rung must add.

The data type forces the shape of the answer. The input is an unstructured 3D point cloud — a *set* of ~5000–10000 points with no canonical ordering, varying per mesh, one mesh per forward pass. I want to avoid anything that needs the mesh graph on this floor: building and message-passing over the `geo` edges is a whole machinery (neighborhoods, normalization, edge bookkeeping) I would rather introduce deliberately later. So the question becomes: can I get global coupling out of the point set *without* edges, with one symmetric reduction? I propose **PointNet** — a shared per-point MLP followed by a single global max-pool. The cheapest symmetric reduction over an unordered set is a pool (sum, mean, or max), and of those, max has the cleanest reading. I lift every point to a feature with a shared per-point MLP, then take the element-wise maximum across all $N$ points, producing one global descriptor whose $j$-th coordinate is "the strongest response of learned feature $j$ anywhere on the mesh." That descriptor is permutation-invariant by construction — max does not care about order — and it summarizes the entire shape in one vector. I then hand that vector back to every point: concatenate the global descriptor onto each point's own local feature and run another shared MLP, so every point predicts its field value from both its local boundary conditions *and* a global summary of the whole body. That is precisely the minimal cross-point coupling: one global max-pool, broadcast back, no edges.

What makes this the right floor rather than a hack is that the global descriptor is provably a sufficient statistic of the set for a wide class of set functions — a shared lift followed by a symmetric max can approximate any continuous set function given enough pooling width, with the output pinned down by a handful of "critical" points whose features win the maxima. So it is the canonical permutation-invariant set encoder, and it gives a principled floor: *what can you predict about every point's field if the only thing you know about the rest of the body is one max-pooled summary?*

Landing this in the task's edit surface requires care, because the `pointnet` baseline here is a stripped, design-task-specific variant, not the full PointNet of the classification/segmentation literature, and the code must match the variant. Three differences are load-bearing. First, there are **no T-nets** — the signature input and feature alignment networks (the 3×3 and 64×64 learned transforms, the latter kept orthogonal by a regularizer on the loss) are dropped. That is defensible here: the design meshes already arrive in a consistent physical frame (a car is oriented the same way, the inlet direction fixed), so there is no pose ambiguity to canonicalize, and the orthogonality regularizer would have to be threaded through the *frozen* loss, which the edit surface forbids. Second, the pooling is **global over the entire mesh in one shot**, the result repeated back to every point and concatenated — the segmentation-style local-plus-global concatenation, but with a single global descriptor, because there is one mesh per forward pass (batch is one, so `global_max_pool` over a single graph is just a column-wise max and `repeat_interleave` broadcasts it back across all $N$ points). Third, the channels are the design-task ones and wide on purpose: the encoder lifts the ten inputs to `n_hidden`, an `in_block` to $2\cdot$`n_hidden`, a `max_block` to $32\cdot$`n_hidden` before the global max, and an `out_block` consumes the concatenation of the local $2\cdot$`n_hidden` feature with the $32\cdot$`n_hidden` global descriptor down to $4\cdot$`n_hidden`, after which a final linear and the decoder produce the four channels. The shared per-point MLPs are the `MLP(... n_layers=0 ...)` blocks run on the squeezed $(N, C)$ tensor — the batch dimension squeezed because the loop guarantees one mesh, with a zero `batch` index vector feeding the pool.

The width is forced by the budget: `CONFIG_OVERRIDES = {'n_hidden': 16}`, the canonical PointNet width, which keeps it a clean baseline and makes the $32\cdot$`n_hidden` global descriptor a modest 512-wide. One more literal-scaffold detail: the model **raises if `geo` is `None`**, because the harness registers PointNet as a graph model and builds an `edge_index` for every forward pass even though this model never message-passes over the edges — it uses only the squeezed point set and the zero batch vector for the global pool. I keep that guard.

Reasoning about what this floor should and should not do gives the diagnosis the next rung inherits. A single global max-pool is a very coarse channel: two points in completely different flow regimes — one on the high-pressure stagnation nose, one deep in the low-pressure wake — receive the *identical* global descriptor, so the only thing distinguishing their predictions is their own local feature. The model can express "this is a nose-type point in a body whose summary is $X$" but no *relation* between nose and wake beyond what survives the one-vector bottleneck. Since drag is a surface integral of pressure projected on the inlet direction, the drag *coefficient* depends mostly on the coarse front-to-back pressure contrast, which a global descriptor can plausibly capture — so I expect the **drag rank correlation (rho_d) to be reasonable**. But the *field* errors, especially velocity, depend on local spatial structure that one global vector cannot reconstruct: velocity varies sharply and non-locally near the surface, and a point that knows only "my features + one global max" has no idea *where it sits relative to its neighbors*. So I expect the **relative-L2 field errors to be the worst of any rung that adds real spatial structure**, and the **drag-magnitude error (c_d) to be loose** even if the ordering is decent. That gap — locality — is exactly what the next rung must close by letting each point pool from its actual mesh neighbors through the `geo` edges PointNet ignored.

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
