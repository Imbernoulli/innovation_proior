We want a neural PDE solver that stays accurate as the mesh grows from the academic regime of thousands of points to industrial geometries — car bodies, aircraft surfaces, full 3D flow fields — where a single layer may have to touch one to two and a half million points. The map we are learning sends geometry, boundary or condition features, and observed fields to a solution field on an irregular mesh, and the hard constraint is that the point count $N$ can no longer be treated as a mild nuisance. Full attention over the mesh costs $O(N^2 C)$ and is simply impossible at this scale. The strongest existing direction avoids point-to-point attention by grouping points into a small set of learned physical-state tokens: for each point feature $x_i \in \mathbb{R}^C$ a weight vector $w_i \in \mathbb{R}^M$ over $M$ states is learned with a linear projection and a softmax along the state axis, the points are averaged into $M$ state tokens $s_j = \sum_i w_{ij} x_i / \sum_i w_{ij}$, attention is run only among the $M$ tokens, and the updated states are broadcast back to the points by the same weights, $x'_i = \sum_j w_{ij} s'_j$. This slice operator costs $O(N M C + M^2 C)$, linear in $N$ for fixed $M$, and it behaves well on irregular geometry because the state-token view corresponds to an integral operator over a learned slice domain rather than a mere engineering trick. So the skeleton is right and I do not want a new backbone — I want the precise reasons this skeleton breaks at million-point scale, and the minimal repairs.

Three failures show up exactly when the mesh becomes huge. First, the slice weights $w = \mathrm{Softmax}(\mathrm{Linear}(x)/\tau_0)$ have a single global sharpness scale. If the weights go flat, every state token collapses toward the same global average of all points, and the later $M$-token attention degenerates into attention among copies of one mean — the physical-state story is gone and the operator behaves like average pooling. A single temperature cannot simultaneously let a point deep in a slowly varying region commit hard to one state and let a point on a sharp transition honestly hedge across several. Second, the published slice operator projects each point into one stream that chooses the slice and a second stream that supplies the content being averaged; that doubles pointwise projection memory at precisely the scale where memory is scarcest. Third, when the points are sharded across GPUs, the numerator $\sum_i w_{ij} x_i$ and denominator $\sum_i w_{ij}$ are global sums, so a GPU's local shard statistics are not enough, and generic long-sequence parallelism communicates the feature stream or the point sequence, whose cost grows with $N$.

I propose Transolver++, which keeps the points $\to$ weights $\to$ state means $\to$ state attention $\to$ deslice skeleton and changes only the state-learning and scaling machinery. The first change is a local adaptive temperature. Instead of one global $\tau_0$, point $i$ gets its own temperature $\tau_i = \tau_0 + \mathrm{Linear}(x_i)$, implemented as a small per-head predictor $\texttt{dim\_head} \to \texttt{slice\_num} \to 1$ with GELU, a learnable per-head bias initialized at $0.5$, and a lower clamp at $0.01$. The clamp is load-bearing because the temperature is a denominator: if it reaches zero the logits and their gradients blow up. The limiting cases are exactly what I want — a constant predicted temperature recovers the old constant-temperature operator, a small $\tau_i$ sharpens the state distribution where the physics is locally clean, and a large $\tau_i$ softens it only where the local physics is genuinely mixed. The second change recognizes that even a locally tempered softmax is still a soft vote, and that most mesh points should choose one state rather than weakly vote for many. A hard $\arg\max$ blocks gradients, so I reparameterize the slice assignment with Gumbel-softmax: draw $u \sim \mathrm{Uniform}(0,1)$, form the Gumbel sample $g = -\log(-\log u)$, and compute

$$ w_i = \mathrm{softmax}\!\left(\frac{\mathrm{Linear}(x_i) + g_i}{\tau_i}\right). $$

The sign is the delicate point: the sampled Gumbel variable is $-\log(-\log u)$, so adding $g_i$ is the same as $\mathrm{Linear}(x_i) - \log(-\log u_i)$, and writing $+\log(-\log u)$ would sample the wrong perturbation. With low $\tau_i$ this relaxation approaches a one-hot categorical draw, giving crisp state choices in clean regions; with higher $\tau_i$ it stays soft, giving multi-state hedging around fast-changing boundaries.

The third change removes the duplicate content stream. With the assignment now sharper and more local, one projection can do both jobs: project $x$ once to $x_{\text{mid}}$, use it to compute the slice logits and the temperature, and also use it as the value averaged into states. The old content projection $f$ is dropped entirely, which is a real reduction of per-point projection memory, not a cosmetic cleanup. The state token is still a weighted mean, $s_j = \sum_i w_{ij} x_i / (\sum_i w_{ij} + 10^{-5})$, where the $10^{-5}$ only protects against division by zero for an empty or nearly empty state and does not perturb the estimator when a state carries nontrivial mass. The number of states is a genuine compromise: $M = 1$ collapses to global pooling and discards correlations, while too large an $M$ raises cost and fragments states into small noisy groups, so the standard settings use $32$ or $64$ states, and $32$ at width $256$ for the million-scale industrial cases.

The fourth change makes the distributed state computation exact and cheap. With the points sharded so GPU $k$ holds $N_k$ local points and local weights $w^{(k)}$, each GPU forms its local mass $\sum_i w^{(k)}_{ij}$ and local numerator $\sum_i w^{(k)}_{ij} x^{(k)}_i$. The one thing a GPU must not do is normalize its own shard first, because that would give each shard equal influence regardless of how much state mass it actually holds. So I SUM all-reduce the numerator and the denominator separately and divide only afterward,

$$ s_j = \frac{\sum_k \sum_i w^{(k)}_{ij} x^{(k)}_i}{\sum_k \sum_i w^{(k)}_{ij} + 10^{-5}}, $$

with both reductions being SUM reductions and the stabilizer added after the mass reduction. Every GPU then holds the same $M$ global states, runs the identical small state attention, and deslices only to its own local points with its own local weights. The communication is $O(\#\mathrm{gpu} \cdot M \cdot (C+1))$ — $M$ masses plus $M\!\cdot\!C$ numerator entries — entirely independent of $N$, and on one GPU the all-reduces are identities. Two implementation details keep this faithful: the state attention is ordinary scaled dot-product attention over tensors shaped $(B, H, M, \texttt{dim\_head})$, and since I call `F.scaled_dot_product_attention(q, k, v)` PyTorch already applies the $1/\sqrt{\texttt{dim\_head}}$ scaling, so I keep that factor only as a harmless stored attribute and never multiply by it again; and because the distributed functional all-reduce returns the reduced tensor, I must bind its return value, `slice_norm = dist_nn.all_reduce(slice_norm, ...)`, rather than call it for effect, or the formula above would not actually hold in code. The fifth and last change is purely about memory at scale: during training the attention and feed-forward sublayers are wrapped in activation checkpointing so their forward activations are recomputed in the backward pass instead of stored, which represents the same function and trades compute for memory exactly where that trade pays off; at evaluation the sublayers are called directly. Together these keep the slice operator's linear-in-$N$ structure while making the learned states locally adaptive and near-categorical, halving the per-point projection memory, and reducing distributed communication to a few small state statistics.

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
