We want a fast learned surrogate for PDEs on the meshes practitioners actually use — a car body, an airfoil, tens of thousands of irregularly placed points with complex, non-periodic boundaries. The job is to take the discretized geometry $g \in \mathbb{R}^{N\times C_g}$, together with whatever quantities $u \in \mathbb{R}^{N\times C_u}$ are observed on it, and predict the solution field at every point in a single forward pass. Two pressures pull against each other. The field is governed by intricate, often long-range physical interactions — the wake behind a car depends on its nose, the pressure on one surface couples to flow far away — so we need a model that captures *correlations* across the whole domain, not just local smoothing. And the domain has $N$ in the tens of thousands, so anything quadratic in $N$ is dead before it starts. Cheap on 32k irregular points, globally correlated, and able to swallow an arbitrary unstructured geometry: each existing tool does some of that and none does all of it. The right frame is operator learning — each layer is a non-local integral operator $G(u)(g^*) = \int_\Omega \kappa(g^*,\xi)\,u(\xi)\,d\xi$ followed by a pointwise nonlinearity — and every neural operator is a way to parameterize and evaluate that integral. The Fourier Neural Operator picks $\kappa$ in the Fourier domain with a fixed basis, learnable spectral multipliers, and the FFT for speed; but the FFT *is* the periodic-uniform-grid assumption, so on a car shape the deformation trick that maps the domain onto a latent grid degenerates badly even after a full hyperparameter sweep. Graph-kernel operators handle irregular meshes with a learnable kernel over local neighborhoods, but the kernel is *local*, so carrying information from the nose of the car to the wake demands many message-passing steps, and global correlation is exactly what local kernels are worst at. Attention gives a learnable kernel with global reach — it is precisely the Monte-Carlo discretization of $G$ with the mesh points as quadrature nodes — but at $32\text{k}$ points its $O(N^2)$ cost is hopeless (even gradient-checkpointed full attention tops out near 7k tokens on a 40GB card), and the linear-attention patches that fix the complexity leave the attention computed *over the mesh points themselves*, drowning the informative physical correlations in a sea of low-level point-to-point relations and making the optimization jittery.

I propose **Transolver**, a Transformer-based neural operator whose one new ingredient is **Physics-Attention**. The key realization is that mesh points are an artifact of discretization: the physics does not live at the points, it lives in *states*. Spatially distant points are often in the *same* physical state — a car's windshield, license plate, and headlights are all in the drag-relevant front region — and spatially adjacent points can be in different states. So the move is not "make the quadrature cheaper" but "use better nodes": quadrature the integral operator over $M$ learned physical states rather than $N$ mesh points, where the groups are free to be any shape and to span the domain non-locally — something square patches and hand-carved FEM subdomains can never express. For a deep feature $x \in \mathbb{R}^{N\times C}$, the operator runs in three movements. First, **slice**: soft-assign every point across $M$ slices with a softmax taken *over the slice axis*,
$$w_i = \mathrm{Softmax}\!\big(\mathrm{Project}(x_i)\big) \in \mathbb{R}^{1\times M}, \qquad \sum_{j=1}^{M} w_{i,j} = 1,$$
so $w_{i,j}$ is the degree to which point $i$ belongs to slice $j$ and the per-point weights form a partition of unity. The softmax-over-slices choice is load-bearing: it both gives each point a clean distribution over slices (the opposite axis would split a single point arbitrarily with no "belongs here" reading) and, through the exponential, *sharpens* the assignment so it is low-entropy and peaky — without it the model could sit at a lazy near-uniform assignment in which every slice encodes the same domain-wide average and all slices collapse into one. $\mathrm{Project}$ is a pointwise linear map $C \to M$, the one component that adapts to any geometry without assuming a grid. I add a learnable per-head temperature $\tau$ that divides the slice logits before the softmax, initialized on the sharp side at $0.5$ so a head can choose how committed its slicing is; this temperature belongs to the slice assignment only, while the token attention keeps the usual scale. Second, **encode each slice into one physics-aware token** as a *mass-normalized* weighted mean of its members,
$$z_j = \frac{\sum_{i=1}^{N} w_{i,j}\, x_i}{\sum_{i=1}^{N} w_{i,j}} \in \mathbb{R}^{1\times C},$$
where dividing by the slice mass $\sum_i w_{i,j}$ (floored with a tiny $\varepsilon$ in code to avoid dividing by an empty slice) makes $z_j$ a genuine weighted average rather than a weighted sum, so a slice that happens to own many points does not get an artificially inflated token. Third, run **ordinary softmax attention among the $M$ tokens**,
$$q,k,v = \mathrm{Linear}(z), \qquad z' = \mathrm{Softmax}\!\big(q k^{\top}/\sqrt{d}\big)\,v, \qquad q,k,v,z' \in \mathbb{R}^{M\times d},$$
and here I deliberately keep full softmax — the most expressive learnable-kernel operator — rather than crippling it with a linear approximation, because $M$ is small (32–64) so $M^2$ is trivial and this is where the global physical correlations are modeled, every state attending directly to every other; the $1/\sqrt{d}$ is the usual scaling so the dot products do not saturate the softmax as the per-head width grows. Finally, **deslice**: broadcast the transited tokens back to the points with the *same* weights,
$$x'_i = \sum_{j=1}^{M} w_{i,j}\, z'_j, \qquad 1 \le i \le N.$$
Reusing the same $w$ for the broadcast is not an economy but a requirement: it makes slice-then-deslice a single change of variables — move from the point domain into the slice domain, do the work, and come back through the *same* map — rather than two unrelated projections glued together. Calling the sandwich $\mathrm{Physics\text{-}Attn}(x) = x'$, the cost is slicing and encoding $O(NMC)$, token attention $O(M^2 C)$, and deslicing $O(NMC)$ again, so $O(NMC + M^2 C)$ — linear in $N$ — with the global-correlation work done over the $M$ meaningful nodes, not the $N$ noisy ones. I run this multi-head, splitting the $C$ channels into $\text{heads}$ subspaces with an independent slicing/attention/deslicing in each and concatenating, because the domain admits more than one meaningful decomposition at once (pressure regimes, velocity regimes, geometric regimes) and each head learns its own slice weights and so its own physical-state grouping.

What convinces me this is principled rather than an ad-hoc trick is that Physics-Attention is the *same* integral operator $G$, only evaluated through slice coordinates. Canonical attention is already the Monte-Carlo discretization of $G$ with the mesh points as nodes, the query fixed and the softmax denominator summing over key/value nodes. To move into the slice domain I build a diffeomorphic projection $g$: for countable $\Omega$, march through the points and send point $i$ to the unclaimed slice coordinate in its own block of size $K$ with the largest slice weight, which is injective and, restricting $\Omega_s$ to the claimed coordinates, bijective, so $\Omega \cong \Omega_s$; with smooth slice weights this is a diffeomorphism. Defining the slice-domain value function $u_s(\xi_s) = \big(\int_\Omega w_{\xi,\xi_s}\,u(\xi)\,d\xi\big)\big/\big(\int_\Omega w_{\xi,\xi_s}\,d\xi\big)$ — exactly the continuous token encoding — and pushing $G$ through the change of variables $\xi = g^{-1}(\xi_s)$, the slice coordinates are implemented as permutation-invariant token slots with counting measure, so the measure-preserving simplification $|\det \nabla g^{-1}| = 1$ applies (otherwise the determinant would have to be carried in the measure or absorbed into the induced kernel). Writing the mesh-to-slice kernel as a $w$-weighted combination of the slice-to-slice kernel and using the slice-softmax partition of unity $\int_{\Omega_s} w_{g,\xi_s'}\,d\xi_s' = 1$ gives
$$G(u)(g) = \int_{\Omega_s} w_{g,\xi_s'}\,\Big[\int_{\Omega_s} \kappa_{ss}(\xi_s',\xi_s)\, u_s(\xi_s)\, d\xi_s\Big]\, d\xi_s',$$
whose three factors are exactly deslice $\circ$ attention-among-tokens $\circ$ token-encoding. Discretizing the slice-domain integrals by Monte-Carlo over the $M$ learned slice nodes yields
$$G(u)(g_i) \approx \sum_{j=1}^{M} w_{i,j} \sum_{t=1}^{M} \frac{\exp\!\big(q_j k_t^{\top}/\sqrt{d}\big)}{\sum_{p=1}^{M}\exp\!\big(q_j k_p^{\top}/\sqrt{d}\big)}\, v_t,$$
where the outer $j$-sum is the deslice step, the inner $t$-sum is the token-attention row for query slice $j$, and the $p$-sum normalizes only over key/value slices for that fixed $j$. The tying of slice and deslice weights is *forced* by this single map $g$ used both ways. Two refinements make the implementation honest. The feature best for *deciding* which state a point is in need not be the feature I want to *carry* into the token, so I project $x$ twice with separate linear maps — an assignment stream feeding the slice weights and a content stream supplying the values that get averaged — both at width $\dim_\text{head}$. And to keep the $M$ slice directions from starting correlated and differentiating slowly, I initialize the slice-projection linear with an orthogonal weight matrix, while everything else gets truncated-normal init (std $0.02$) and LayerNorms start at unit weight, zero bias. The number of slices $M$ is the one genuinely new hyperparameter: $M=1$ collapses Physics-Attention to a single global pooling-and-broadcast with no state-to-state correlation, and $M$ near $N$ over-fragments the domain into noisy slivers and drifts back toward attention-over-points, so the useful regime is a few dozen, traded against width — $M=64$ for $C=128$, $M=32$ for $C=256$ to keep parameters and runtime comparable. Heads stay at $8$, layers at $L=8$, the feed-forward at the usual $4\times$ width, none of which is where the contribution lives. Dropped into the canonical pre-norm Transformer in place of full attention,
$$\hat{x}^{\,l} = \mathrm{Physics\text{-}Attn}\big(\mathrm{LayerNorm}(x^{l-1})\big) + x^{l-1}, \qquad x^{l} = \mathrm{FeedForward}\big(\mathrm{LayerNorm}(\hat{x}^{\,l})\big) + \hat{x}^{\,l},$$
with $x^0 = \mathrm{Linear}(\mathrm{Concat}(g,u))$ and a linear read-out of $x^L$, the whole model is the canonical Transformer with its one expensive, geometry-blind sublayer swapped for the cheap, geometry-general one — linear in $N$ and needing no mesh graph at all. For a genuinely unstructured mesh $\mathrm{Project}$ is the pointwise linear above; when the mesh is structured the two point projections become local convolutions of kernel size $3$ so a point's assignment can see its neighbors, with the temperature clamped to $[0.1, 5]$, and the token-attention and deslice unchanged.

```python
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_


ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'ELU': nn.ELU,
    'silu': nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        act = ACTIVATION[act]
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class Physics_Attention_Irregular_Mesh(nn.Module):
    """Soft-slice N points into M physical-state tokens, attend among tokens, broadcast back."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)       # assignment stream
        self.in_project_fx = nn.Linear(dim, inner_dim)      # content stream
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape

        # (1) Slice + token encoding
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()                # B H N dim_head
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()                # B H N dim_head
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature)  # B H N M  (softmax over M)
        slice_norm = slice_weights.sum(2)                     # B H M
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None]
                                     .repeat(1, 1, 1, self.dim_head))   # mass-normalized mean

        # (2) Attention among M tokens
        q, k, v = self.to_q(slice_token), self.to_k(slice_token), self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.softmax(dots))
        out_slice_token = torch.matmul(attn, v)               # B H M dim_head

        # (3) Deslice with the same weights
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Transolver_block(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout=0., act='gelu', mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__name__ = 'Transolver'
        self.args = args
        self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                              n_layers=0, res=False, act=args.act)
        self.blocks = nn.ModuleList([
            Transolver_block(num_heads=args.n_heads, hidden_dim=args.n_hidden, dropout=args.dropout,
                             act=args.act, mlp_ratio=args.mlp_ratio, out_dim=args.out_dim,
                             slice_num=args.slice_num, last_layer=(i == args.n_layers - 1))
            for i in range(args.n_layers)])
        self.placeholder = nn.Parameter((1 / args.n_hidden) * torch.rand(args.n_hidden))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx, T=None, geo=None):
        # x: (1, N, space_dim) coords; fx: (1, N, fun_dim) features; geo (mesh graph) unused
        if fx is not None:
            fx = torch.cat((x, fx), dim=-1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx
```
