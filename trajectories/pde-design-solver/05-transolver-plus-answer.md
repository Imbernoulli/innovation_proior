**Problem.** Transolver swept the published benchmarks (rho_d 0.9896, c_d 0.0136, Car 0.0809/0.0218,
AirfRANS 0.0335/0.0156), confirming that grouping points by learned physical state closes the
global-correlation gap. The one open lever was the slicing itself: Transolver's **single per-head
temperature** cannot vary commitment by location, and its plain softmax over M=32 logits leaves slices
overlapping, so tokens are *blended* rather than sharply distinguishable physical states.

**Key idea (Transolver++ / eidetic states).** Make the slice assignment eidetic — sharply
distinguishable — with two coupled changes. (1) **Adaptive per-point temperature**: predict a
temperature for each point from its own feature (a small MLP `dim_head→slice_num→1`) plus a learnable
bias, clamped to a small positive floor, so clean-regime points commit hard (low temperature) and
ambiguous boundary points hedge (high temperature) — Transolver is the constant-temperature special
case. (2) **Gumbel-softmax** slice weights: add Gumbel noise before the softmax so each point's
assignment is pushed toward a near-categorical (one-hot-like) choice while staying differentiable, and
the noise regularizes against the lazy near-uniform assignment. Clean commitments give clean,
distinguishable tokens; attention among M clean states is a sharper operator at the same O(M²) cost.

**Single-stream collapse (budget-critical).** Once the assignment is eidetic, the slice-decision
feature and the token content can share **one** projection `x_mid` instead of Transolver's two-stream
(`x_mid` + `fx_mid`). This removes a `Linear(dim, inner_dim)` per layer — the 30–70% footprint
reduction — which more than pays for the small temperature MLP, so the finale runs at the *same*
`n_hidden=256, slice_num=32` as Transolver while landing **under** Transolver's parameter count, inside
the task's 1.05×-Transolver-256 budget.

**Match to the reference (faithful, with two strips).** Re-expressed from the canonical
`Transolver_plus/models/Transolver_plus.py` `Physics_Attention_1D_Eidetic`, defined **inline in
Custom.py** because the shipped `layers.Physics_Attention` is read-only. Two pieces of the reference are
inapplicable here and removed: the `torch.distributed.nn.all_reduce` over slice statistics (multi-GPU
million-scale only; here one mesh, one device, batch 1) and the gradient-checkpointing wrapper (a
memory optimization the frozen loop does not need). The block and `Model` are the canonical pre-norm
Transolver skeleton unchanged; the slice projection keeps the orthogonal init.

**Reference.** Luo, Wu, Zhou, Xing, Di, Wang, Long, "Transolver++: An Accurate Neural Solver for PDEs
on Million-Scale Geometries", arXiv:2502.02414 (2025); official code `thuml/Transolver_plus`.

**Bar to clear (no feedback; this is the endpoint).** Match-or-beat Transolver's rho_d 0.9896 and c_d
0.0136 and reduce the field errors below Car 0.0809/0.0218 and AirfRANS 0.0335/0.0156 (velocity most,
where sharp separation/wake boundaries are the ambiguous regions the adaptive temperature targets),
while using *fewer* parameters than Transolver. AirCraft is treated as task-internal.

**Hyperparameters.** `CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}` (equal to Transolver for a
fair comparison; admissible because single-stream drops parameters). `n_heads=8`, `n_layers=8`,
`mlp_ratio`, `dropout`, `act`, `out_dim` from `args`. Adaptive-temperature bias init 0.5, clamp floor
0.01, Gumbel `tau=1` noise.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from einops import rearrange
from layers.Basic import MLP


def gumbel_softmax(logits, tau=1.0):
    # continuous relaxation of a hard slice assignment; differentiable, near-categorical
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = (logits + gumbel_noise) / tau
    return F.softmax(y, dim=-1)


class Physics_Attention_Irregular_Mesh_Eidetic(nn.Module):
    """Eidetic Physics-Attention: single-stream slicing with a per-point adaptive temperature
    and a Gumbel-softmax assignment, so slices are sharply distinguishable physical states.
    Cost O(N*M*C + M^2*C). all_reduce / checkpointing from the multi-GPU reference are dropped."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # learnable per-head bias added to the predicted per-point temperature
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        # per-point temperature predictor: dim_head -> slice_num -> 1 (positive via GELU)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )

        # SINGLE stream: x_mid both decides the slice AND supplies the token content
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)  # decorrelate slice directions at init

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape

        # (1) single-stream per-head point features
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()                       # B H N dim_head

        # (2) adaptive per-point temperature, floored positive
        temperature = self.proj_temperature(x_mid) + self.bias      # B H N 1
        temperature = torch.clamp(temperature, min=0.01)

        # (3) eidetic slice assignment: Gumbel-softmax over the M slice axis
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)  # B H N M
        slice_norm = slice_weights.sum(2)                           # B H M  (slice mass)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights)        # weighted sum
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None]
                                     .repeat(1, 1, 1, self.dim_head))               # -> weighted mean

        # (4) attention among the M physics-aware tokens
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(q, k, v)   # B H M dim_head

        # (5) deslice back to points through the SAME slice weights
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Transolver_block(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout=0., act='gelu', mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh_Eidetic(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num)
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
        self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                              n_layers=0, res=False, act=args.act)
        self.blocks = nn.ModuleList([Transolver_block(num_heads=args.n_heads, hidden_dim=args.n_hidden,
                                                      dropout=args.dropout,
                                                      act=args.act,
                                                      mlp_ratio=args.mlp_ratio,
                                                      out_dim=args.out_dim,
                                                      slice_num=args.slice_num,
                                                      last_layer=(_ == args.n_layers - 1))
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

    def forward(self, x, fx, T=None, geo=None):
        # geo (mesh graph) ignored: Physics-Attention needs no graph
        if fx is not None:
            fx = torch.cat((x, fx), dim=-1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx


CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}
```
