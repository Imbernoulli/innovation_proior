# Context: structured weight matrices for efficient neural network training

## Research question

Large neural networks are expensive to train and fine-tune; the dominant cost is the dense
linear layers — the projection matrices inside attention blocks and the two big matrices of
every feed-forward (FFN) block. Each dense `n × n` weight matrix costs `O(n²)` parameters and
`O(n²)` FLOPs per token, and these layers are where the wall-clock time goes.

The standing idea for cutting this cost is to replace each dense weight matrix `W` with a
**structured** matrix — a matrix with sub-quadratic (`o(n²)`) parameters and a sub-quadratic
matrix–vector multiply. Examples: entrywise-sparse matrices, low-rank matrices, and the
classical fast transforms (Fourier, sine/cosine, Chebyshev, Hadamard). The question is
whether there is a structured class that makes this trade favorable in practice, and what
properties such a class would need to have.

## Background

**FLOPs are not wall-clock time on a GPU.** A GPU is a block-oriented machine: it is fast when
it multiplies dense tiles (the tensor cores / batched-matrix-multiply, BMM, units operate on
dense blocks) and slow when it chases scattered nonzeros, because irregular memory access
defeats coalescing and the dense-tile units. It is well documented that unstructured-sparse
training, despite cutting FLOPs, usually slows down training in wall-clock time (Gale et al.
2019; Hooker 2020). A structure built from dense blocks can map onto BMM and actually be fast;
a structure built from scattered entries cannot.

**Fast transforms and their recursive structure.** The discrete Fourier transform (DFT) of
size `N=2^m` factors by the Cooley–Tukey recursion (Cooley & Tukey 1965). Writing `F_N` for
the DFT matrix and `ω_N = e^{2πi/N}`, the size-`N` transform reduces to two size-`N/2`
transforms on the even and odd indices:

```
F_N = [ I_{N/2}   Ω_{N/2} ] [ F_{N/2}   0     ] P_N
      [ I_{N/2}  −Ω_{N/2} ] [   0     F_{N/2} ]
```

where `Ω_{N/2} = diag(1, ω_N^{-1}, …, ω_N^{-(N/2−1)})` and `P_N` is the permutation that
sorts even indices before odd. Unrolling the recursion (Dao et al. 2019) gives

```
F_N = B_N · B_{N/2} · … · B_2 · (permutations),
```

a product of `log₂ N` factors.

**Butterfly matrices** (Parker 1995; Dao et al. 2019; Dao et al. 2020 "Kaleidoscope"). A
*butterfly factor* of size `k` (`k` even) is a `2×2` block of `k/2 × k/2` **diagonal**
matrices, `B_k = [[D₁, D₂],[D₃, D₄]]`. A *butterfly factor matrix* `B_k^{(n)}` of size `n` is
block-diagonal with `n/k` such factors. A *butterfly matrix* of size `n=2^s` is a product
`B^{(n)} = B_n^{(n)} B_{n/2}^{(n)} … B_2^{(n)}` of `log₂ n` butterfly factor matrices — exactly
the sparsity pattern that the Cooley–Tukey recursion above produces. Each butterfly matrix has
`O(n log n)` parameters and an `O(n log n)` matrix–vector multiply. They are *expressive*: Dao
et al. 2020 prove the **kaleidoscope hierarchy** built from butterflies,
`BB*` (products `M₁ M₂*` of two butterfly matrices) and `(BB*)^w_e` (width-`w`, expansion-`e`
products), can represent **any** `n×n` matrix whose multiply is a depth-`d`, `s`-gate linear
arithmetic circuit, with `M ∈ (BB*)^{O(d)}_{O(s/n)}` and only `O(ds log s)` parameters/runtime
— i.e. essentially all structured matrices, near-optimally.

**Bailey's four-step FFT** (Bailey 1990) is a memory-hierarchy algorithm that computes a
length-`n` DFT by viewing the input as an `m × m` matrix (`n = m²`): (1) FFT each column,
(2) multiply by twiddle factors, (3) transpose, (4) FFT each row. Each FFT pass is a *batched*
set of `m` independent size-`m` transforms applied along one axis of the reshaped input.

**Projection: the closed-form cases.** For a few structured classes, the closest member to a
given dense `A` (in Frobenius norm) is known in closed form: entrywise-sparse → keep the
largest-magnitude entries (magnitude pruning, Tewarson 1973); **low-rank → truncated SVD, by
the Eckart–Young theorem** (Eckart & Young 1936), which states the best rank-`k` Frobenius
approximation of `A` is `Σ_{i≤k} σ_i u_i v_iᵀ` from the SVD; orthogonal → the orthogonal
Procrustes solution (Schönemann 1966). For richer classes — butterfly, products of sparse
matrices — only iterative heuristics exist (first-order optimization, alternating
minimization; Le Magoarou & Gribonval 2016; Dao et al. 2019; Lin et al. 2021), with no
guarantee of optimality (Pan 2012).

## Baselines

**Dense linear layer.** `W ∈ ℝ^{n×n}`, full `O(n²)` parameters and FLOPs, executed as one
GEMM. Its multiply is maximally hardware-efficient (a single dense matmul).

**Entrywise-sparse / pruned layers** (Han et al. 2015; Frankle & Carlin 2018 lottery tickets;
RigL, Top-KAST). Keep an unstructured subset of weights. Cuts parameter count and FLOPs, and
can match quality at sparse inference.

**Block-sparse layers** (Gray et al. 2017; Child et al. 2019). Restrict the sparsity to dense
blocks so the multiply maps onto BMM and is fast.

**Low-rank layers.** `W ≈ UVᵀ`, rank `r`, `O(nr)` parameters, two GEMMs. Hardware-efficient
and has the Eckart–Young closed-form projection.

**Hand-picked fast transforms** (FFT/DCT layers; used in PDE solvers and MRI). Fixed, very
cheap, encode strong domain priors. Only specific instances have fast GPU kernels (FFT does;
a general orthogonal-polynomial transform does not).

**Butterfly layers** (Dao et al. 2019; Dao et al. 2020). A product of `log n` butterfly factor
matrices: expressive (the kaleidoscope result above), learnable, `O(n log n)` params and FLOPs.
The `log n` sequential factors are each a multiply by a matrix of tiny `2×2` diagonal blocks.

**Pixelated butterfly** (Chen et al. 2021). Makes butterflies hardware-friendly by fixing a
flat (non-recursive) block sparsity pattern.

## Evaluation settings

The natural yardsticks are the standard training benchmarks for the layers being replaced, in
each of the three usage modes. End-to-end-from-scratch: ViT on ImageNet classification, and
GPT-2 on WikiText-103 language modeling (compare quality at matched wall-clock training time).
Science/medical, where hand-crafted transforms are the incumbent: PDE operator learning
(e.g. Navier–Stokes vorticity prediction) and accelerated multi-coil MRI reconstruction
(k-space is the spatial Fourier domain; the metric is reconstruction error). Sparse-to-dense
("reverse sparsification") pretraining: GPT-2 on OpenWebText and BERT pretraining, against the
optimized NVIDIA/Megatron baselines (the BERT one being the MLPerf-1.1-record implementation),
measured by time-to-quality. Dense-to-sparse fine-tuning: project a pretrained BERT and
fine-tune on the GLUE suite, comparing accuracy, parameter count, and fine-tuning speed to the
dense model. Metrics throughout: task quality (accuracy / perplexity / reconstruction error),
parameter count, and especially **wall-clock** time on real GPUs (A100), not FLOP counts.

## Code framework

The structured layer is a drop-in replacement for `nn.Linear`, so the model code (Transformer
blocks, optimizer, training loop) is untouched; only the linear primitive changes. Everything
below already exists before the method.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange

# --- Eckart-Young: best rank-`rank` Frobenius approximation of a matrix via SVD ---
def low_rank_project(M, rank):
    """Batched: M has shape (..., p, q); returns U (..., p, rank), Vt (..., rank, q)
    with U @ Vt the best rank-`rank` approximation of each matrix in the batch."""
    U, S, Vt = torch.linalg.svd(M)
    S_sqrt = S[..., :rank].sqrt()
    U = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
    return U, Vt

# --- the dense baseline primitive the structured layer must replace ---
class DenseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# --- base class for any structured linear layer (handles padding/bias) ---
class StructuredLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
    def preprocess(self, x):                      # pad input up to extended size
        if x.shape[-1] < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - x.shape[-1]))
        return x
    def postprocess(self, out):                   # crop output back down
        return out[..., :self.out_features] if out.shape[-1] > self.out_features else out
    def forward_matmul(self, x):
        raise NotImplementedError                 # subclasses implement the structured multiply
    def forward(self, x):
        out = self.forward_matmul(x)
        return out + self.bias.to(out.dtype) if self.bias is not None else out


# === SLOT 1: the structured matrix-vector multiply we will design ===
def structured_multiply(x, *factors):
    """Apply our structured weight matrix to x, built from dense-block factors,
    using only batched dense matmuls (no scattered-nonzero ops)."""
    # TODO: the multiplication algorithm we will design
    raise NotImplementedError


# === SLOT 2: project a dense matrix onto the structured class ===
def structured_project(A):
    """Given a dense matrix A, return the factors of the closest matrix in our
    structured class (Frobenius norm)."""
    # TODO: the projection algorithm we will design
    raise NotImplementedError


# === SLOT 3: the structured linear layer (drop-in for DenseLinear) ===
class StructuredWeightLinear(StructuredLinear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias)
        # TODO: declare the dense-block parameters of our structured matrix
        # TODO: initialize them
    def forward_matmul(self, x):
        # TODO: call structured_multiply on the preprocessed x with our parameters
        raise NotImplementedError
```

The three slots are where the work goes: a multiply built only from batched dense matmuls
(Slot 1), a dense→structured projection (Slot 2), and an `nn.Linear`-compatible layer wrapping
them (Slot 3).
