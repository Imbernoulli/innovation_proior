# Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries

## Core method

Start from Physics-Attention: assign `N` point features to `M` learned physical
states, average point features into state tokens, attend among the `M` states,
and deslice back to the point mesh.

```text
w = Softmax(Linear(x) / tau0)
s_j = sum_i w_ij x_i / sum_i w_ij
s' = Attention(s)
x'_i = sum_j w_ij s'_j
```

Transolver++ keeps this skeleton and changes the state-learning and scaling
machinery:

1. **Local adaptive temperature.**
   Replace the single temperature with a pointwise temperature
   `tau_i = tau0 + Linear(x_i)`. In the reference code this is a small MLP
   `dim_head -> slice_num -> 1` with GELU activations, plus a learnable per-head
   bias initialized to `0.5`, clamped with `min=0.01`.

2. **Gumbel-softmax slice reparameterization.**
   Use a differentiable categorical sample for slice weights:

   ```text
   g_i = -log(-log(u_i)),  u_i ~ Uniform(0, 1)
   w_i = softmax((Linear(x_i) + g_i) / tau_i).
   ```

   This is equivalent to the `Linear(x_i) - log(-log epsilon_i)` form.
   The sign-sensitive point is that `-log(-log u)` is the Gumbel sample, not
   `log(-log u)`.

3. **Single-stream state content.**
   Remove the old duplicate content projection `f`. One projection `x_mid`
   both computes the slice logits/temperatures and supplies the state content.
   This is the speedup marked in the single-stream algorithm: a single
   `in_project_x` with no `in_project_fx`.

4. **Distributed state means.**
   If points are sharded across GPUs, compute local state masses and local
   weighted numerators, SUM all-reduce both, and divide only after reduction:

   ```text
   norm_j = AllReduce_sum_k sum_i w_ij^(k)
   num_j  = AllReduce_sum_k sum_i w_ij^(k) x_i^(k)
   s_j = num_j / (norm_j + 1e-5)
   ```

   Communication is `O(#gpu * M * (C + 1))`, independent of `N`. For one GPU,
   the reductions are identities.

5. **Training memory.**
   Wrap the attention and feed-forward sublayers in activation checkpointing
   during training; call them directly at evaluation.

## Correctness checks

- **Temperature cases.** Constant predicted temperature recovers the prior
  constant-temperature slice operator. Low `tau_i` gives near one-hot state
  assignment. High `tau_i` gives a softer mixture for ambiguous regions.
- **Slice-count cases.** `M = 1` collapses to global pooling and loses physical
  correlations. Larger `M` increases cost and can fragment states; I
  use `{32, 64}` in standard settings and `32` for million-scale industrial
  settings with width 256.
- **Gumbel sign.** The implementation's `logits + (-log(-log(u)))` is the same
  as `logits - log(-log(u))`; reversing that sign would sample from the wrong
  perturbation.
- **Attention scale.** The code uses `F.scaled_dot_product_attention(q, k, v)`,
  so the `1/sqrt(dim_head)` scaling is handled by PyTorch. Do not apply another
  manual scale.
- **Normalization.** The state token denominator is `slice_norm + 1e-5`. The
  stabilizer is added after the SUM reduction of masses.
- **Aerodynamic coefficient signs.** The reported drag/lift formulas use
  `-p n dot d + tau n dot d` divided by `0.5 rho v_inf^2 A`, where the pressure
  term is negative because pressure acts opposite the outward normal.

## Reference-faithful implementation

In the PyTorch 2.7.1 API, `torch.distributed.nn.all_reduce` returns the
reduced tensor rather than mutating its argument in place, so a call like
`dist_nn.all_reduce(slice_norm, ...)` must have its return value assigned back
for the reduction to take effect. The code below binds those return values to
implement the distributed-state algorithm exactly.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn as dist_nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint


ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


def gumbel_softmax(logits, tau=1, hard=False):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = (logits + gumbel_noise) / tau
    y = F.softmax(y, dim=-1)
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y


class Physics_Attention_1D_Eidetic(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU(),
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        bsz, num_points, _ = x.shape
        x_mid = self.in_project_x(x).reshape(
            bsz, num_points, self.heads, self.dim_head
        ).permute(0, 2, 1, 3).contiguous()

        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)

        slice_norm = slice_weights.sum(2)
        slice_norm = dist_nn.all_reduce(slice_norm, op=dist_nn.ReduceOp.SUM)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        slice_token = dist_nn.all_reduce(slice_token, op=dist_nn.ReduceOp.SUM)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(
            q_slice_token, k_slice_token, v_slice_token
        )

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError
        act_layer = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_layer())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_layer())
             for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            y = self.linears[i](x)
            x = x + y if self.res else y
        return self.linear_post(x)


class Transolver_plus_block(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_1D_Eidetic(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        if self.training:
            fx = checkpoint(self.Attn, self.ln_1(fx), use_reentrant=True) + fx
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=True) + fx
        else:
            fx = self.Attn(self.ln_1(fx)) + fx
            fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx
```

## Result summary

The measured gains are 13% average relative improvement across six standard PDE
benchmarks and more than 20% improvement on million-scale industrial tasks.
Single-GPU input capacity reaches up to 1.2 million points and full
DrivAerNet++ cases with roughly 2.5 million points on up to 4 A100 GPUs.
