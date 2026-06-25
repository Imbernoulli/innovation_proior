**Problem.** The three graph rungs resolved into one statement: lossless local message passing
(GraphSAGE) wins on field fidelity (Car 0.089/0.033, AirfRANS 0.046/0.037, c_d 0.0193) but gives back a
little global *ordering* (rho_d 0.978, just under the U-Net's 0.981), because any message-passing stack
reaches only `n_layers` hops. The shared residual is **global physical correlation without a diameter
limit and without coarsening loss**.

**Key idea (Transolver / Physics-Attention).** Step out of message passing. Each layer is a non-local
integral operator, and attention is its most expressive parameterization — but over the N mesh points
it is O(N²) and dilutes the physics. So change the quadrature nodes from points to **learned physical
states**: soft-assign every point across M slices via a per-point softmax over the slice axis
(partition of unity; the exponential sharpens so slices specialize), encode each slice into a
mass-normalized weighted-mean token, run **full softmax attention among the M tokens** (M=32, so M² is
trivial; this is where front-region and wake states interact directly), and broadcast back through the
**same** slice weights. Cost O(NMC + M²C) — linear in N — with the attention over M meaningful nodes.

**Why.** Tying slice and deslice weights makes the sandwich a single change of variables that
reproduces the integral operator term-for-term, so this is the same operator the graph methods
approximated locally, now evaluated globally over states. Grouping by *state* (not location) is what
lets scattered same-regime points share a token — exactly the non-local correlation graphs missed.

**Match to the harness (canonical here).** This baseline stays close to the canonical formulation. It uses the
shipped `layers.Physics_Attention.Physics_Attention_Irregular_Mesh` (two-stream point projection — one
stream decides the slice, one supplies content; a learnable per-head temperature init ~0.5; an
orthogonally-initialized slice projection; mass-normalized tokens; `dim_head^{-1/2}` token scale). The
edit is the `Transolver_block` (canonical pre-norm residual, last block carrying the read-out) and
`Model`. `geo` is **ignored** — Physics-Attention needs no mesh graph, so this is the one rung that does
not raise on `geo=None`. `geotype='unstructured'` selects the irregular-mesh class (plain linear
projections), the geometry-general default.

**Hyperparameters.** `CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}` — the canonical
width (2× GraphSAGE, the model the task's 1.05× budget is defined against) and the new slice count;
M=32 sits between M=1 (collapses to global pooling) and M→N (fragments into noise). `n_heads=8`,
`n_layers=8`, `mlp_ratio`, `dropout`, `act`, `out_dim` from `args`.

```python
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from layers.Basic import MLP
from layers.Embedding import timestep_embedding, unified_pos_embedding
from layers.Physics_Attention import Physics_Attention_Irregular_Mesh
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_1D
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_2D
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_3D

PHYSICS_ATTENTION = {
    'unstructured': Physics_Attention_Irregular_Mesh,
    'structured_1D': Physics_Attention_Structured_Mesh_1D,
    'structured_2D': Physics_Attention_Structured_Mesh_2D,
    'structured_3D': Physics_Attention_Structured_Mesh_3D
}


class Transolver_block(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            geotype='unstructured',
            shapelist=None
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = PHYSICS_ATTENTION[geotype](hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                               dropout=dropout, slice_num=slice_num, shapelist=shapelist)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Custom'
        self.args = args
        if args.unified_pos and args.geotype != 'unstructured':
            self.pos = unified_pos_embedding(args.shapelist, args.ref)
            self.preprocess = MLP(args.fun_dim + args.ref ** len(args.shapelist), args.n_hidden * 2,
                                  args.n_hidden, n_layers=0, res=False, act=args.act)
        else:
            self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                                  n_layers=0, res=False, act=args.act)
        if args.time_input:
            self.time_fc = nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden), nn.SiLU(),
                                         nn.Linear(args.n_hidden, args.n_hidden))

        self.blocks = nn.ModuleList([Transolver_block(num_heads=args.n_heads, hidden_dim=args.n_hidden,
                                                      dropout=args.dropout,
                                                      act=args.act,
                                                      mlp_ratio=args.mlp_ratio,
                                                      out_dim=args.out_dim,
                                                      slice_num=args.slice_num,
                                                      last_layer=(_ == args.n_layers - 1),
                                                      geotype=args.geotype,
                                                      shapelist=args.shapelist)
                                     for _ in range(args.n_layers)])
        self.placeholder = nn.Parameter((1 / (args.n_hidden)) * torch.rand(args.n_hidden, dtype=torch.float))
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

    def structured_geo(self, x, fx, T=None):
        if self.args.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
        return fx

    def unstructured_geo(self, x, fx, T=None):
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
        return fx

    def forward(self, x, fx, T=None, geo=None):
        if self.args.geotype == 'unstructured':
            return self.unstructured_geo(x, fx, T)
        else:
            return self.structured_geo(x, fx, T)


CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}
```
