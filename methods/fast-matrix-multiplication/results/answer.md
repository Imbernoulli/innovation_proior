# Strassen's algorithm and the tensor-rank viewpoint on matrix multiplication

## Problem

Multiply two n×n matrices over a field. The schoolbook algorithm evaluates c_ij = Σ_k a_ik b_kj
in Θ(n³) arithmetic operations. The question is whether o(n³) is possible — i.e. whether the
exponent ω = inf{β : n×n matrices can be multiplied in O(n^β) operations} is strictly below 3.

## Key idea

Recursively splitting each matrix into four (n/2)×(n/2) blocks reduces an n×n product to **block
products**, and the running time is T(n) = (#block products)·T(n/2) + O(n²), which solves to
Θ(n^{log₂(#block products)}). The naive eight block products give log₂ 8 = 3 (no gain). A single
algebraic identity that multiplies two 2×2 matrices with **seven** products instead of eight —
and, crucially, never multiplies two A-entries or two B-entries together, so it remains valid when
the entries are non-commuting matrix blocks — recurses to

    T(n) = 7·T(n/2) + O(n²)  ⟹  T(n) = Θ(n^{log₂ 7}),   log₂ 7 ≈ 2.807.

The deeper reframing: the minimum number of multiplications for the bilinear map ⟨n,n,n⟩ is the
**rank** R(t) of its matrix-multiplication tensor t (the minimum number of rank-one tensors
u⊗v⊗w summing to t). Matrix tensors multiply, ⟨k,m,n⟩ ⊗ ⟨k',m',n'⟩ = ⟨kk',mm',nn'⟩, and rank is
submultiplicative, so R(⟨2,2,2⟩) ≤ 7 yields R(⟨2^L,2^L,2^L⟩) ≤ 7^L and ω ≤ log₂ 7. The recursion
is a tensor power; the base rank becomes the exponent.

Tensor rank is not closed under limits: the two-output bilinear map
(a0b0, a1b0 + a0b1) has rank 3 but border rank 2. **Border rank** R̲ — approximation up to higher
order in a formal ε — can therefore be strictly smaller than rank and still bounds the exponent,
because coefficient extraction after recursion costs only n^{o(1)}. Bini's partial 2×2 product
(compute z11, z12, z21 and omit z22) has rank 6 and border rank 5; two copies give
R̲(⟨2,2,3⟩) ≤ 10 and therefore

    ω ≤ log_{(2·2·3)^{1/3}} 10 = 3·log₁₂ 10 ≈ 2.78.

Schönhage's asymptotic-sum theorem generalizes this: if
R̲(⊕_i ⟨k_i,m_i,n_i⟩) ≤ r and τ solves Σ_i (k_i m_i n_i)^τ = r, then ω ≤ 3τ. All of it is one
principle: the cost of matrix multiplication is the (border) rank of its tensor, and that rank
powers up multiplicatively.

## The 2×2 identity (seven multiplications)

For A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]] (entries may be matrix blocks):

    P1 = (A11 + A22)(B11 + B22)
    P2 = (A21 + A22) B11
    P3 = A11 (B12 − B22)
    P4 = A22 (B21 − B11)
    P5 = (A11 + A12) B22
    P6 = (A21 − A11)(B11 + B12)
    P7 = (A12 − A22)(B21 + B22)

    C11 = P1 + P4 − P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 − P2 + P3 + P6

Seven multiplications, eighteen additions/subtractions. Every P_k is (an A-side combination) ×
(a B-side combination), so no factor is ever reordered and the identity holds for non-commuting
entries — which is what licenses the recursion.

## Algorithm

Pad to even size whenever a recursive level needs it; recurse via the seven products down to
a crossover leaf size (below which the eighteen additions plus recursion overhead lose to a tuned
cubic kernel); call the cubic kernel at the leaf; unpad. Asymptotic cost Θ(n^{log₂ 7}). Numerical
stability is worse than the classical algorithm in floating point because the extra subtractions
introduce more cancellation, so practical use depends on the field, precision, and crossover.

## Code

```python
import numpy as np

def matmul_naive(A, B):
    return A @ B

def split(M):
    m = M.shape[0] // 2
    return M[:m, :m], M[:m, m:], M[m:, :m], M[m:, m:]

def block_product(A, B, recurse):
    a, b, c, d = split(A)
    e, f, g, h = split(B)

    p1 = recurse(a + d, e + h)       # (A11+A22)(B11+B22)
    p2 = recurse(c + d, e)           # (A21+A22)B11
    p3 = recurse(a, f - h)           # A11(B12-B22)
    p4 = recurse(d, g - e)           # A22(B21-B11)
    p5 = recurse(a + b, h)           # (A11+A12)B22
    p6 = recurse(c - a, e + f)       # (A21-A11)(B11+B12)
    p7 = recurse(b - d, g + h)       # (A12-A22)(B21+B22)

    C = np.empty(A.shape, dtype=np.result_type(A, B))
    m = a.shape[0]
    C[:m, :m] = p1 + p4 - p5 + p7    # C11
    C[:m, m:] = p3 + p5              # C12
    C[m:, :m] = p2 + p4              # C21
    C[m:, m:] = p1 - p2 + p3 + p6    # C22

    return C

def fast_matmul(A, B, leaf=64):
    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("fast_matmul expects two n x n matrices")

    def recurse(X, Y):
        n = X.shape[0]
        if n <= leaf:
            return matmul_naive(X, Y)

        original_n = n
        if n % 2 == 1:
            X = np.pad(X, ((0, 1), (0, 1)), mode="constant")
            Y = np.pad(Y, ((0, 1), (0, 1)), mode="constant")

        C = block_product(X, Y, recurse)
        return C[:original_n, :original_n]

    return recurse(A, B)
```
