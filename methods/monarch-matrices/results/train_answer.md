Almost all the compute and memory in a large Transformer sits in a small handful of dense $n \times n$ matrices — the query/key/value/output projections in attention and the two fat matrices of every feed-forward block — each costing $O(n^2)$ parameters and $O(n^2)$ work per token. The obvious way to make training cheaper is to replace each dense $W$ with a *structured* matrix that has $o(n^2)$ parameters and a fast multiply. The reason nobody has simply done this and walked away with a free speedup is that every candidate class fails at least one of three demands that must hold simultaneously: it must be **hardware-efficient** (fast in wall-clock time on a GPU, not merely cheap in FLOPs), **expressive** (rich enough to match dense quality and to contain the useful fast transforms — Fourier, DCT/DST, convolution, Hadamard), and **projectable** (given a *pretrained dense* matrix $A$, the closest member of the class can be found efficiently and optimally, so a checkpoint can be converted instead of retrained from scratch).

The failures are instructive, because each one points at the constraint I have to design around. Entrywise sparsity cuts FLOPs by $10\times$ but consistently runs *slower* on a GPU, and it has to: a GPU is a block machine whose fast units multiply dense tiles, so scattered nonzeros turn every multiply into an irregular gather that defeats memory coalescing and leaves the dense-tile hardware idle. FLOPs are a bookkeeping fiction here; the real currency is whether the work lands on dense tiles. So the nonzeros of whatever I build must be packed into dense blocks — non-negotiable. The naive way to honor that, a single block-diagonal matrix, runs as a clean batched matmul but is *block-separable*: coordinate $i$ only ever talks to coordinates in its own block, so it cannot represent a DFT, a convolution that straddles block boundaries, or even a permutation that swaps two blocks. Low-rank $UV^\top$ is hardware-fine and even has a closed-form projection (Eckart–Young: the best rank-$r$ Frobenius approximation is the top-$r$ SVD), but a DFT and a permutation are full rank, so low-rank caps quality below the transforms I want. Hand-picked transforms like an FFT layer are fast and exact but fixed, not learnable, and a general orthogonal-polynomial transform has no fast GPU kernel. The expressive classes (butterflies, kaleidoscope) are hardware-hostile and lack a projection; the hardware-efficient classes (block-sparse, low-rank) are not expressive enough. No class is all three.

I propose **Monarch matrices**, and the route to them runs through the butterflies, since those are the maximally expressive structured class: products of butterfly factors represent *any* structured matrix — anything whose multiply is a low-depth arithmetic circuit — with near-optimal $O(n \log n)$ parameters (the kaleidoscope result of Dao et al. 2020). A butterfly matrix is a product $B = B_n B_{n/2} \cdots B_2$ of $\log_2 n$ block-diagonal factors whose blocks are $2 \times 2$ arrangements of diagonal matrices — exactly the sparsity pattern the Cooley–Tukey FFT recursion produces. The trouble is precisely those $\log n$ *sequential* factors: each is a multiply by tiny irregular $2 \times 2$-of-diagonals blocks, so the work dribbles out in $\log n$ small kernel launches over awkwardly-strided data instead of landing on big dense tiles. The expressiveness is real and the FLOPs are great, but it runs slowly. The factors are the disease, so the cure is to stop applying them one at a time. Matrix multiplication is associative: nothing forces $\log n$ steps. If I condense the first half of the factors into one matrix and the second half into another, I replace $\log n$ small multiplies with two big ones — provided the two condensed matrices keep a block structure I can still run as batched matmul.

That they do is exactly what Bailey's four-step FFT shows. Bailey computes a length-$n$ DFT, $n = m^2$, by reshaping the input into an $m \times m$ matrix and doing: FFT every column, multiply by twiddles, transpose, FFT every row. "FFT every column" is $m$ independent size-$m$ transforms in parallel — a multiply by a **block-diagonal** matrix with $m$ blocks of size $m \times m$; "FFT every row," after the transpose, is *another* block-diagonal multiply. The whole $\log n$-factor chain reorganizes into **two block-diagonal multiplies separated by a permutation**. Made exact in butterfly terms for $n$ a power of $4$: split $B = B_n \cdots B_2$ in half, let $R$ be the product of the last $(\log_2 n)/2$ factors and $L'$ the product of the first half. The small-block factors compose into a genuinely block-diagonal $R = \mathrm{diag}(R_1, \dots, R_m)$, $m$ dense $m \times m$ blocks. The top half $L'$ comes out as an $m \times m$ array of blocks where *each block is itself diagonal* — not block-diagonal, but one permutation away. Let $P$ be the stride/transpose permutation (reshape a length-$n$ vector to $m \times m$, transpose, flatten); it is its own inverse, $P = P^\top$. Conjugating, $L := P L' P^\top$ swaps "which block" with "where in the block," turning the array-of-diagonal-blocks into a genuine block-diagonal $L = \mathrm{diag}(L_1, \dots, L_m)$. So $L' = P^\top L P$ and $B = L'R = P^\top L P R = P L P^\top R$. I lift this orientation straight into a definition, not requiring it to come from a butterfly at all: a matrix $M$ of size $n \times n$, $n = m^2$, is a **Monarch matrix** if

$$M = P\,L\,P^\top\,R,$$

where $L$ and $R$ are each block-diagonal with $m$ dense blocks of size $m \times m$ and $P$ is the fixed transpose permutation. Two block-diagonal matrices, one parameter-free permutation.

This buys all three demands. **Expressiveness:** every butterfly fits this form (I just condensed the factors), and condensing into a free product can only enlarge the set, so $\mathcal{B} \subset \mathcal{M}$ strictly — even pinning $R = I$ leaves $L$ a free block-diagonal matrix with $\sqrt{n}$ blocks of $\sqrt{n} \times \sqrt{n}$, i.e. $n^{3/2}$ free entries, which exceeds a butterfly's $2n\log_2 n$ once $n > 256$, so beyond that size no butterfly can reach every Monarch matrix. Because $\mathcal{M} \supset \mathcal{B}$, products of Monarch matrices contain products of butterflies: $\mathcal{M}\mathcal{M}^* \supset \mathcal{B}\mathcal{B}^*$ and $(\mathcal{M}\mathcal{M}^*)^2 \supset (\mathcal{B}\mathcal{B}^*)^2$, so $\mathcal{M}\mathcal{M}^*$ represents convolution, Hadamard, Toeplitz, AFDF, and $(\mathcal{M}\mathcal{M}^*)^2$ represents Fourier, DST/DCT, $(HD)^3$, Fastfood, ACDC — the entire kaleidoscope story inherited for free. The only cost is an $O(d\sqrt{s})$ representation overhead versus kaleidoscope, the price of trading a polylog parameter count for hardware regularity. **Efficiency:** parameters $2n\sqrt{n} = O(n^{1.5})$, sub-quadratic; FLOPs $O(n\sqrt{n})$ — which is *more* than the butterfly's $O(n\log n)$, and that is fine, because those FLOPs are organized as two batched matmuls over dense $m \times m$ blocks, the one operation a GPU is fastest at, whereas the butterfly's fewer FLOPs were $\log n$ tiny irregular passes. Two big BMMs beat $\log n$ small kernels in wall-clock by a wide margin; the asymptotic FLOP regression is the price of regularity, and regularity is exactly what I am buying. Concretely the multiply never materializes $P$ or $M$: viewing $x$ as an $m \times m$ tensor $x_{ki}$, multiply by $R$ block-wise ($y_{kj} = \sum_i R_{kji} x_{ki}$, a batched matmul over $k$), then apply $L$ along the other axis ($z_{\ell j} = \sum_k L_{j\ell k} y_{kj}$, a batched matmul over $j$), giving $z_{\ell j} = \sum_{k,i} L_{j\ell k} R_{kji} x_{ki}$ — Bailey's two passes, the permutation just deciding which axis the matmul runs along.

The third demand, projection, is where the real surprise lives. Finding the Monarch $M$ minimizing $\|A - M\|_F^2$ for a dense pretrained $A$ looks hopelessly nonconvex, since $M = PLP^\top R$ is a *product* of two unknowns — this is exactly where prior butterfly-projection work fell back to first-order heuristics with no optimality guarantee. But write $M$ in coordinates. Reading the multiply off as a 4D tensor of size $m \times m \times m \times m$ indexed $(\ell, j, k, i)$,

$$M_{\ell j k i} = L_{j\ell k}\,\cdot\,R_{kji}.$$

For *fixed* $(j,k)$, viewed as a function of the remaining indices, the first factor depends only on $\ell$ and the second only on $i$ — an outer product, hence a **rank-1** matrix. The tangled product, after this one reshape, is simply "every block, in this reshaping, has rank 1." Since the Frobenius norm ignores shape, reshape $A$ the same way and the cross terms separate completely:

$$\|A - M\|_F^2 = \sum_{j,k} \big\|\,A_{:,j,k,:} - u_{jk} v_{jk}^\top\,\big\|_F^2,$$

so the nonconvex global problem breaks into $m^2$ **independent** rank-1 approximation subproblems. Each is exactly the Eckart–Young problem, solved in closed form by the top singular component $\sigma_1 u_1 v_1^\top$ of an SVD. Take that rank-1 piece per slice, set $L_{j\ell k} = (u_{jk})_\ell$ and $R_{kji} = (v_{jk})_i$, and reshape back into block-diagonal $L$ and $R$. This is an *analytical optimum* for a nonconvex objective — the Monarch analogue of Eckart–Young — at a cost of $m^2$ SVDs of $m \times m$ matrices, $O(n^{5/2})$ total. If $A$ was already Monarch, each slice is exactly rank 1 and the routine recovers its factors perfectly, so the same procedure both projects and factorizes.

One step further handles the transforms I most want to store cheaply — the FFT and DCT live in $\mathcal{M}\mathcal{M}^*$, a *product* of two Monarch matrices, where the per-block-rank-1 trick does not apply. Conjugate $\hat M = P M P^\top$; its $(i,j)$ block is $\hat M_{ij} = A_i D_{ij} C_j$ with $A_i, C_j$ dense and $D_{ij}$ diagonal. Define $F(i,j) = \hat M_{i1}^{-1} \hat M_{ij} \hat M_{1j}^{-1} \hat M_{11}$; substituting, every $A_i$ and $C_j$ telescopes away and $F(i,j) = C_1^{-1}(\text{diagonal})C_1$, so the single matrix $C_1$ **simultaneously diagonalizes** all the $F(i,j)$. Compute any simultaneous diagonalizer $\hat C_1$ (diagonalize one $F(i,j)$; refine within degenerate eigenspaces using the others), then $\hat A_i = \hat M_{i1}\hat C_1^{-1}$, $\hat C_j = \hat A_1^{-1}\hat M_{1j}$, and $\hat D_{ij} = \hat A_i^{-1}\hat M_{ij}\hat C_j^{-1}$, the last provably diagonal. Runtime $O(n^3/b)$, $= O(n^{5/2})$ at $b = \sqrt{n}$, assuming $M$ invertible and $R$'s blocks free of zero entries.

For real weights, which are rarely square and need a parameter knob, make the **block size** $b$ a free choice ($b \mid n$): parameters become $n^2/b + nb$, with the right knob in practice being the *number* of blocks — 2 to 4 blocks lands at 25–50% of dense, enough capacity to hope for dense-level quality while the two BMMs comfortably beat a single dense GEMM. Everything proven goes through with $b$ in place of $\sqrt n$, and rectangular $\text{out} \times \text{in}$ matrices are handled with rectangular blocks. The whole family then serves three usage modes from one construction: **end-to-end sparse training** (swap dense `nn.Linear` for `MonarchLinear(nblocks=4)`, init each block like a dense layer, train with Adam through the BMMs); **sparse-to-dense "reverse sparsification"** (train Monarch for ~90% of iterations, then densify by multiplying out $L$ and $R$ with the permutation and finish the last 10% dense, ending in a full dense model that pure sparse training would have struggled to fit); **dense-to-sparse fine-tuning** (project a pretrained checkpoint onto $\mathcal{M}$ in closed form, then fine-tune, transferring pretrained knowledge with one SVD per block); and **FFT-initialized layers for PDE/MRI** (project the exact (I)FFT onto $\mathcal{M}$ to start at the right transform, then learn, with far fewer parameters than a CNN). The multiply is two batched matmuls with a reshape between; the projection is reshape-into-batched-slices, a batched rank-1 SVD, and reshape into the two block-diagonal factors; the layer wraps both as a drop-in for `nn.Linear`.

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
