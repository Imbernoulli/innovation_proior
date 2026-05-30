# Performer (FAVOR+)

## Problem

Regular softmax self-attention forms an `L × L` score matrix, costing `O(L²d)` time and `O(L² + Ld)` memory in the sequence length `L`. This is prohibitive for long sequences. The goal is to approximate the **same** full, dense softmax attention in **linear** time and memory, with provable accuracy, and without assuming sparsity or low rank.

## Key idea

Softmax attention is `Att(Q,K,V) = D⁻¹ A V`, `A = exp(QKᵀ/√d)`, `D = diag(A 1_L)`. The `L²` cost is only there because `A` is built before multiplying by `V`. If the score is a dot product of feature maps, `A(i,j) = φ(q_i)ᵀφ(k_j)`, associativity removes the `L²` object:

    Att̂(Q,K,V) = D̂⁻¹ (Q'((K')ᵀ V)),   D̂ = diag(Q'((K')ᵀ 1_L)),

where `Q', K' ∈ R^{L×m}` stack the feature rows `φ(q_i)ᵀ, φ(k_j)ᵀ`. Computing `(K')ᵀV` (an `m×d` matrix) then `Q'·(…)` costs `O(Lmd)` time and `O(Lm + Ld + md)` memory — linear in `L`.

The feature map comes from **FAVOR+** (Fast Attention Via positive Orthogonal Random features):

- **Positive random features (R+).** The softmax kernel `SM(x,y) = exp(xᵀy)` is, by completing the square in a Gaussian integral,
  `SM(x,y) = E_{ω∼N(0,I)}[ exp(ωᵀx − ‖x‖²/2) · exp(ωᵀy − ‖y‖²/2) ]`,
  giving the unbiased, strictly **non-negative** feature map
  `φ⁺(u) = (exp(−‖u‖²/2)/√m)(exp(ω_1ᵀu), …, exp(ω_mᵀu))`.
  Non-negativity keeps `D̂` a positive normalizer. Its MSE is `(1/m)exp(‖x+y‖²)·SM(x,y)²·(1−exp(−‖x+y‖²))`, which **→ 0** as `SM → 0` — the opposite of the trigonometric (sin/cos) feature map, whose MSE carries a factor `SM⁻²` and **diverges** as `SM → 0`, producing negative/blown-up normalizers and unstable training. A symmetrized `cosh` variant (`f = exp(±u)`) lowers the variance further.

- **Orthogonal random features (O).** Sampling `ω_1,…,ω_m` (with `m ≤ d`) exactly orthogonal — Gram-Schmidt on a Gaussian block, which leaves each marginal `N(0,I)` so the estimator stays unbiased — reduces MSE by an explicit, strictly positive gap that holds for **every** dimension `d`:
  `MSE(orthogonal) ≤ MSE(iid) − (2(m−1)/(m(d+2)))·(SM(x,y) − exp(−(‖x‖²+‖y‖²)/2))²`.
  Positivity of the feature coefficients is what makes this gap non-negative.

- **Causal case via prefix sums.** Masking to `j ≤ i` turns the global sum `Σ_j φ(k_j)(v_j,1)ᵀ` into a running prefix sum `G_i = Σ_{j≤i} φ(k_j)(v_j,1)ᵀ`; the output row is `φ(q_i)ᵀG_i`. This is a cumulative sum: `O(L)` work, `O(log L)` parallel depth, no `L×L` matrix.

- **Generalized attention.** Any non-negative feature `φ(x) = f(ωᵀx) + ε` works; `f = ReLU` is a strong, numerically clean default.

The number of random features needed for a uniform `ε`-approximation of `A` is `m = Θ(d log d)`, **independent of `L`**.

## Algorithm

Inputs `Q, K, V ∈ R^{L×d}`, feature count `m`, orthogonal projection `W ∈ R^{m×d}`:

1. `Q' = φ(Q)`, `K' = φ(K)` rowwise, with `φ` either the positive softmax map (`exp`, with `d^{-1/4}` scaling and a max-subtraction for stability) or the generalized map (`ReLU(Wu)+ε`).
2. Append a ones column: `C = [V | 1_L]`.
3. **Bidirectional:** `Buf = Q'((K')ᵀC) ∈ R^{L×(d+1)}`. **Causal:** `Buf_i = G_i^{PS} Q'_i` with `G_i^{PS} = Σ_{j≤i} K'_j C_jᵀ` (prefix sum).
4. Split `Buf = [AV̂ | D̂]`; return `diag(D̂)⁻¹ AV̂`.

Periodically redraw `W` to avoid an unlucky projection.

## Code

```python
import math
from functools import partial
import torch
import torch.nn as nn
from einops import repeat

# Orthogonal Gaussian projection (the "O" of FAVOR+): orthogonal rows with
# chi-distributed lengths, so each row is marginally N(0, I_d) -> unbiased.
def orthogonal_block(d, device=None):
    g = torch.randn((d, d), device=device)
    q, _ = torch.linalg.qr(g)
    return q.t()

def gaussian_orthogonal_matrix(m, d, scaling=0, device=None):
    blocks = [orthogonal_block(d, device) for _ in range(m // d)]
    rem = m - (m // d) * d
    if rem > 0:
        blocks.append(orthogonal_block(d, device)[:rem])
    W = torch.cat(blocks)
    if scaling == 0:
        mult = torch.randn((m, d), device=device).norm(dim=1)   # ||N(0,I_d)||
    else:
        mult = math.sqrt(d) * torch.ones((m,), device=device)   # SMREG sphere
    return torch.diag(mult) @ W

# Positive softmax features (the "R+"): unbiased, non-negative, low variance
# as the kernel -> 0. d^{-1/4} folds softmax's 1/sqrt(d); max-subtraction
# (cancels in D^{-1}AV) keeps exp() in range.
def softmax_kernel(data, projection, is_query, eps=1e-4):
    b, h, *_ = data.shape
    norm = data.shape[-1] ** -0.25
    ratio = projection.shape[0] ** -0.5
    W = repeat(projection, 'm d -> b h m d', b=b, h=h).type_as(data)
    data_dash = torch.einsum('...id,...md->...im', norm * data, W)
    diag = (data ** 2).sum(dim=-1, keepdim=True) * (norm ** 2) / 2.0
    if is_query:
        data_dash = ratio * (torch.exp(data_dash - diag
                     - data_dash.amax(dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (torch.exp(data_dash - diag
                     - data_dash.amax(dim=(-1, -2), keepdim=True).detach()) + eps)
    return data_dash.type_as(data)

# Generalized features: phi(u) = f(W u) + eps, default f = ReLU.
def generalized_kernel(data, projection, kernel_fn=nn.ReLU(), eps=1e-3):
    b, h, *_ = data.shape
    norm = data.shape[-1] ** -0.25
    W = repeat(projection, 'm d -> b h m d', b=b, h=h).type_as(data)
    data_dash = torch.einsum('...id,...md->...im', norm * data, W)
    return (kernel_fn(data_dash) + eps).type_as(data)

# Reassociation: (K'^T V) first, then Q' times it. Never builds L x L.
def linear_attention(q, k, v):
    k_sum = k.sum(dim=-2)
    D_inv = 1.0 / torch.einsum('...id,...d->...i', q, k_sum)
    context = torch.einsum('...id,...ie->...de', k, v)        # K'^T V
    return torch.einsum('...de,...id,...i->...ie', context, q, D_inv)

# Causal: prefix-sum so token i sees only j <= i.
def causal_linear_attention(q, k, v, chunk=128, eps=1e-6):
    last_k, last_ctx, outs = 0, 0, []
    for q_, k_, v_ in zip(*(t.chunk(chunk, dim=-2) for t in (q, k, v))):
        k_cumsum = last_k + k_.cumsum(dim=-2)
        D_inv = 1.0 / torch.einsum('...id,...id->...i', q_, k_cumsum + eps)
        ctx = torch.einsum('...id,...ie->...ide', k_, v_)
        ctx_cumsum = last_ctx + ctx.cumsum(dim=-3)
        outs.append(torch.einsum('...ide,...id,...i->...ie', ctx_cumsum, q_, D_inv))
        last_k, last_ctx = k_cumsum[..., -1:, :], ctx_cumsum[..., -1:, :, :]
    return torch.cat(outs, dim=-2)

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, causal=False,
                 generalized=False, kernel_fn=nn.ReLU(), ortho_scaling=0):
        super().__init__()
        nb_features = nb_features or int(dim_heads * math.log(dim_heads))  # m = Theta(d log d)
        self.create_proj = partial(gaussian_orthogonal_matrix,
                                   m=nb_features, d=dim_heads, scaling=ortho_scaling)
        self.register_buffer('projection', self.create_proj())
        self.causal, self.generalized, self.kernel_fn = causal, generalized, kernel_fn

    @torch.no_grad()
    def redraw(self, device):
        self.projection.copy_(self.create_proj(device=device))

    def forward(self, q, k, v):
        if self.generalized:
            feat = partial(generalized_kernel, projection=self.projection,
                           kernel_fn=self.kernel_fn)
            q, k = feat(q), feat(k)
        else:
            feat = partial(softmax_kernel, projection=self.projection)
            q, k = feat(q, is_query=True), feat(k, is_query=False)
        attn = causal_linear_attention if self.causal else linear_attention
        return attn(q, k, v)
```

`FastAttention` is a drop-in replacement for the scaled-dot-product attention inside a standard multi-head block: project to `q, k, v`, call `FastAttention.forward`, recombine heads. The rest of the Transformer is unchanged, so a pretrained softmax model can be converted with light finetuning.
