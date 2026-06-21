# Monarch: hardware-efficient, expressive, projectable structured matrices

## Problem

Replace the dense weight matrices of neural networks (attention projections, FFN matrices)
with structured matrices that are simultaneously (1) **fast in wall-clock time on a GPU**, not
just low in FLOPs; (2) **expressive** enough to match dense quality and to represent the fast
transforms (Fourier, DCT/DST, convolution, Hadamard); and (3) **projectable** — given a dense
pretrained matrix, find the closest member of the class efficiently and optimally. Prior
classes fail at least one: scattered sparsity is slow on GPUs; one block-diagonal matrix or a
low-rank matrix is too weak; butterfly matrices are expressive but run as `log n` tiny
irregular factors and have no tractable dense-to-structured projection.

## Key idea

A **Monarch matrix** is the product of two block-diagonal matrices up to a fixed permutation:

```
  M = P L Pᵀ R         (n = m²)
```

- `L`, `R` are block-diagonal, each with `m` dense blocks of size `m × m`.
- `P` is the stride/transpose permutation: reshape a length-`n` vector to `m × m`, transpose,
  flatten; `P = Pᵀ`, fixed, no parameters.

This is the four-step FFT (Bailey 1990) lifted into a definition: the whole `log n`-factor
butterfly/FFT chain reorganized as **two batched dense matmuls separated by a transpose**.

- **Parameters:** `2n√n = O(n^{1.5})` (each factor: `√n` blocks of `√n × √n`).
- **Speed:** `O(n√n)` FLOPs, but executed as two batched matrix multiplies over dense blocks
  → ~2× faster than dense in wall-clock (regularity beats raw FLOP count; this is more FLOPs
  than butterfly's `O(n log n)` but far faster because there are no tiny irregular passes).
- **Expressiveness:** the class `𝓜` strictly contains all butterfly matrices `𝓑` (condensing
  the `log n` butterfly factors into two enlarges the class — there exist `𝓜` matrices that are
  not butterflies; any `𝓜` matrix needs `≥ n^{3/2}` parameters to describe, which exceeds
  butterfly's `2n log₂ n` for `n > 256`). Hence
  products inherit the kaleidoscope expressiveness: `𝓜𝓜*` represents convolution, Hadamard,
  Toeplitz, AFDF; `(𝓜𝓜*)²` represents Fourier, DST/DCT, `(HD)³`, Fastfood, ACDC.
- **General form:** make the block size `b` a free knob (`b | n`); parameters `n²/b + nb`. In
  practice use 2–4 blocks (25–50% of dense). Rectangular `out × in` is supported with
  rectangular blocks.

## The projection theorem (the surprise)

Although `argmin_{M ∈ 𝓜} ‖A − M‖²_F` is nonconvex (`M` is a product), it has a closed-form
optimum, computable in `O(n^{5/2})` — the Monarch analogue of Eckart–Young.

**Why:** reading the multiply `z_{ℓj} = Σ_{k,i} L_{jℓk} R_{kji} x_{ki}` off as a 4D tensor,

```
  M_{ℓjki} = L_{jℓk} · R_{kji}.
```

For fixed `(j,k)`, the `(ℓ,i)` slice factors as (vector in `ℓ`) ⊗ (vector in `i`) — it is
**rank 1**. So `‖A − M‖²_F = Σ_{j,k} ‖A_{:,j,k,:} − u_{jk} v_{jkᵀ}‖²_F` splits into `m²`
independent rank-1 approximation problems, each solved by the top SVD component (Eckart–Young).
`m²` SVDs of `m × m` matrices = `O(m⁵) = O(n^{5/2})`. If `A ∈ 𝓜`, this recovers its factors
exactly.

**Algorithm — projection onto `𝓜`:**

```
Input: A ∈ ℝ^{n×n}, n = m².
1. Reshape A to 4D tensor Ã_{ℓjki} = A[(ℓ-1)m + j, (k-1)m + i].
2. For each (j,k): take the m×m slice Ã_{:,j,k,:}, compute its best rank-1
   approximation u_{jk} v_{jkᵀ} via SVD.
3. R̃_{kji} = (v_{jk})_i ;  L̃_{jℓk} = (u_{jk})_ℓ .
4. Return L, R as block-diagonal matrices (block b of L,R is L̃_{b,:,:}, R̃_{b,:,:}).
```

## Factorizing products of Monarch matrices (`𝓜𝓜*`)

To store an exact transform (FFT, DCT ∈ `𝓜𝓜*`) as cheap factors. Write
`M̂ = P M Pᵀ`; its `(i,j)` block is `M̂_{ij} = Aᵢ D_{ij} Cⱼ` (`Aᵢ, Cⱼ` dense, `D_{ij}`
diagonal). Define `F(i,j) = M̂_{i1}⁻¹ M̂_{ij} M̂_{1j}⁻¹ M̂_{11}`; the `A`'s and `C`'s telescope,
leaving `F(i,j) = C₁⁻¹(diagonal)C₁`, so `C₁` **simultaneously diagonalizes** all `F(i,j)`.
Compute any simultaneous diagonalizer `Ĉ₁`, then `Âᵢ = M̂_{i1}Ĉ₁⁻¹`, `Ĉⱼ = Â₁⁻¹M̂_{1j}`,
`D̂_{ij} = Âᵢ⁻¹M̂_{ij}Ĉⱼ⁻¹` (provably diagonal). Runtime `O(n³/b)` (= `O(n^{5/2})` at `b=√n`).
Assumptions: `M` invertible and `R`'s blocks have no zero entries.

## Usage modes

- **End-to-end sparse training:** swap dense `nn.Linear` (attention projections + FFN) for
  `MonarchLinear(nblocks=4)`, init blocks like a dense layer, train with Adam through the BMMs.
- **Sparse-to-dense ("reverse sparsification"):** train Monarch for ~90% of iterations
  (fast), then densify by multiplying out `L`, `R` (+ permutation) and finish the last 10%
  dense — same iterations, most of them cheaper, ending in a full dense model.
- **Dense-to-sparse fine-tuning:** project a pretrained dense checkpoint onto `𝓜` (closed
  form), then fine-tune — transfers pretrained knowledge without retraining from scratch.
- **FFT-initialized layers (PDE/MRI):** project the (I)FFT onto `𝓜` to initialize at the right
  transform, then learn — fewer parameters than a CNN, less overfit in data-limited regimes.

## Code

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange


class MonarchMultiply(torch.autograd.Function):
    """Apply a Monarch matrix M = P L Pᵀ R to x as two batched dense matmuls.
    x: (batch, n);  w1 = R: (k, q, p), k*p = n;  w2 = L: (l, s, r), l*r = k*q.
    Square Monarch: k=q=p=l=s=r=sqrt(n). Manual backward for fast bmm."""
    @staticmethod
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = int(np.prod(batch_shape))
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)       # multiply by R
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)  # permutation P
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)             # multiply by L
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = int(np.prod(batch_shape))
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        dx = dw1 = dw2 = None
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous().transpose(0, 1)
        if ctx.needs_input_grad[2]:
            dw2 = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj())
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.bmm(dout1, w1_bfly.conj()).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1 = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1, dw2

monarch_multiply = MonarchMultiply.apply


def low_rank_project(M, rank):
    """Best rank-`rank` Frobenius approximation (Eckart-Young), batched over leading dims."""
    U, S, Vt = torch.linalg.svd(M)
    S_sqrt = S[..., :rank].sqrt()
    U = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
    return U, Vt


def monarch_project(M, sizes=None):
    """Closed-form projection of a dense square M onto the Monarch class.
    Reshape so each Monarch block is one batched matrix, take its rank-1 SVD."""
    m, n = M.shape
    assert m == n, 'square only'
    if sizes is None:
        f = [(i, n // i) for i in range(1, int(math.isqrt(n)) + 1) if n % i == 0][-1]
        sizes = (f[1], f[0])                        # block sizes closest to sqrt(n)
    assert n == sizes[0] * sizes[1]
    M_batched = rearrange(M, '(p k) (r s) -> k r p s', k=sizes[1], r=sizes[0])
    U, Vt = low_rank_project(M_batched, rank=1)     # each block -> best rank-1
    w1_bfly = rearrange(Vt, 'k r 1 s -> r k s')     # R
    w2_bfly = rearrange(U, 'k r s 1 -> k s r')      # L
    return w1_bfly, w2_bfly


class MonarchLinear(nn.Module):
    """Drop-in replacement for nn.Linear whose weight is a Monarch matrix."""
    def __init__(self, in_features, out_features, nblocks=4, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        in_blksz = math.ceil(in_features / nblocks)
        out_blksz = math.ceil(out_features / nblocks)
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        if self.in_features_extended < self.out_features_extended:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, in_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
        else:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for blkdiag in [self.blkdiag1, self.blkdiag2]:
            fan_in = blkdiag.shape[-1]
            std = init.calculate_gain('leaky_relu', math.sqrt(5)) / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)

    def forward(self, x):
        if x.shape[-1] < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - x.shape[-1]))
        out = monarch_multiply(x, self.blkdiag1, self.blkdiag2)
        out = out[..., :self.out_features]
        return out + self.bias.to(out.dtype) if self.bias is not None else out


# FFT-initialized Monarch transform: project the DFT onto the Monarch class and learn.
class MonarchFFT(nn.Module):
    def __init__(self, n, direction='fft', norm='ortho'):
        super().__init__()
        f = [(i, n // i) for i in range(1, int(math.isqrt(n)) + 1) if n % i == 0][-1]
        sizes = (f[1], f[0])
        eye = torch.eye(n, dtype=torch.complex128)
        transform = torch.fft.fft if direction == 'fft' else torch.fft.ifft
        dft = transform(eye, norm=norm).t()
        self.register_buffer('perm', rearrange(torch.arange(n), '(i j) -> (j i)', j=sizes[0]))
        w1, w2 = monarch_project(dft[:, self.perm], sizes=sizes)  # exact: DFT lives in MM*
        # store as real so Adam/AdamW behave well on complex params
        self.w1 = nn.Parameter(torch.view_as_real(w1.cfloat()))
        self.w2 = nn.Parameter(torch.view_as_real(w2.cfloat()))

    def forward(self, x):
        w1, w2 = torch.view_as_complex(self.w1), torch.view_as_complex(self.w2)
        return monarch_multiply(rearrange(x[..., self.perm], '... n -> (...) n'),
                                w1, w2).reshape_as(x)
```
