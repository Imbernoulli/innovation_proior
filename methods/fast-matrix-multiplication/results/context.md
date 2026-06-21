## Research question

Given two n×n matrices A and B over a field, compute the product C = AB, whose entries are

    c_ij = Σ_{k=1}^n a_ik b_kj,     1 ≤ i, j ≤ n.

The textbook evaluation of this formula performs, for each of the n² output entries, n scalar
multiplications and n−1 additions: in total n³ multiplications and n²(n−1) additions, i.e.
Θ(n³) arithmetic operations. Matrix multiplication is the inner loop of a vast amount of
numerical and combinatorial computation — solving linear systems, inverting matrices,
computing determinants, least squares, transitive closure, many graph problems — so its cost
sets the cost of all of them. The question is whether two n×n matrices can be multiplied in
o(n³) arithmetic operations, and if so how small the exponent can be made.

A useful way to phrase "how hard is it" is the **exponent** ω: the infimum of all β such that
n×n matrices can be multiplied in O(n^β) operations. The naive algorithm shows ω ≤ 3. The
trivial lower bound is ω ≥ 2 (the output has n² entries, each of which must be written). The
question is where in [2, 3] the truth lies, and how to construct algorithms for the upper
bound.

## Background

**Counting multiplications versus additions.** Over a field, additions and scalar
multiplications are both unit-cost in the standard model, but they play different roles in a
*recursive* algorithm. If a divide-and-conquer scheme reduces an n×n product to r products of
(n/2)×(n/2) matrices plus O(n²) additions to glue the pieces, the running time obeys T(n) =
r·T(n/2) + O(n²). The Master theorem gives T(n) = Θ(n^{log₂ r}) whenever r > 4. The exponent is
set entirely by r, the number of recursive subproblems (multiplications); the gluing additions
sit at the cheap O(n²) level and are dominated.

**Block decomposition.** Partition each n×n matrix (n even) into four (n/2)×(n/2) blocks:

    A = [[A11, A12],[A21, A22]],   B = [[B11, B12],[B21, B22]].

The product blocks are
    C11 = A11 B11 + A12 B21,   C12 = A11 B12 + A12 B22,
    C21 = A21 B11 + A22 B21,   C22 = A21 B12 + A22 B22.
Computed directly this is 8 block multiplications and 4 block additions, giving the recurrence
T(n) = 8·T(n/2) + O(n²), i.e. T(n) = Θ(n^{log₂ 8}) = Θ(n³). One subtlety that any block-level
manipulation must respect is that the block entries are themselves matrices and therefore do
not commute.

**The Karatsuba precedent.** A related algebraic trade exists for integers and polynomials. To
multiply two degree-1 polynomials (a0 + a1X)(b0 + b1X) = a0b0 + (a0b1 + a1b0)X + a1b1 X², the
naive method uses 4 coefficient multiplications. Karatsuba (1962) observes that the cross term
a0b1 + a1b0 = (a0+a1)(b0+b1) − a0b0 − a1b1 reuses the two corner products, so three
multiplications — a0b0, a1b1, (a0+a1)(b0+b1) — suffice, at the cost of a few extra additions.
Applied recursively to n-bit numbers this gives Θ(n^{log₂ 3}) ≈ n^{1.585}, below the
schoolbook n². The coefficients here commute, and the scheme concerns the three coefficients of
a single product.

**Bilinearity of the product.** Matrix multiplication is *bilinear*: each output entry c_ij is
a sum of products (linear form in A)·(linear form in B). One is not obliged to realize such a
form by computing each monomial a_ik b_kj separately; in principle one may form combinations of
A-entries and of B-entries, multiply those, and recombine. A relevant subtlety for recursion:
if every product formed is (a linear form in one matrix's entries only) times (a linear form in
the other's), it never multiplies two A-entries or two B-entries together — so it remains valid
when the entries are replaced by non-commuting objects such as matrix blocks, which is what
makes recursion possible.

**Numerical and practical context.** The classical algorithm is backward stable. A scheme that
trades multiplications for additions introduces extra subtractions and hence extra
cancellation, so its accuracy is a separate practical question. Recursion also carries overhead
(the gluing additions and the bookkeeping of splitting/recombining blocks), so on real hardware
a recursive scheme has a matrix-size crossover, below which a tuned cubic kernel is used.

## Baselines

**Schoolbook / inner-product algorithm.** Evaluate c_ij = Σ_k a_ik b_kj directly. n³
multiplications, n²(n−1) additions, Θ(n³). It computes every term once and is backward stable.

**Naive block recursion.** Split into 2×2 blocks and recurse on the eight block products.
T(n) = 8T(n/2)+O(n²) = Θ(n³). The exponent is set by the eight block products, and 8 = 2³.

**Karatsuba-style multiplication (for integers/polynomials).** Three multiplications for a
degree-1 polynomial product instead of four, recursed to n^{log₂ 3}. This is a
scalar/polynomial algorithm whose coefficients commute, concerning the three coefficients of
one product rather than four separate output blocks. It stands as a precedent that an algebraic
identity reusing partial products can change a multiplication count and, recursed, a recurrence
exponent.

## Evaluation settings

The natural yardstick is **arithmetic operation count** (additions, subtractions, scalar
multiplications, divisions) as a function of n, and its asymptotic exponent ω. Correctness is
checked by the algebraic identity holding over an arbitrary field (equivalently as a
polynomial identity in indeterminate entries), and empirically by agreeing with the schoolbook
product on random integer/floating matrices of assorted sizes (including odd n, which require
padding). For a recursive scheme one also measures wall-clock crossover against a tuned cubic
kernel, and — since extra additions cause cancellation — backward error ‖Ĉ − AB‖ relative to
the classical algorithm. The recursive harness below focuses on square matrices over ℚ, ℝ
(floating point), or a finite field, where the exponent question is already visible.

## Code framework

The existing ingredients are a cubic leaf kernel and a generic divide-and-conquer harness. The
unresolved slot is how a 2×2 block product is computed.

```python
import numpy as np

def matmul_naive(A, B):
    """Cubic leaf kernel and ground truth."""
    return A @ B

def split(M):
    """Partition a 2m x 2m matrix into four m x m blocks."""
    m = M.shape[0] // 2
    return M[:m, :m], M[:m, m:], M[m:, :m], M[m:, m:]

def block_product(A, B, recurse):
    """
    Multiply two even-size matrices by treating them as 2x2 block matrices.
    The blocks are matrices and do NOT commute, so any identity used here
    may only multiply an A-block by a B-block (never two A-blocks or two B-blocks).
    `recurse(X, Y)` multiplies two same-size blocks.

    The naive scheme forms the eight block products
    A11 B11 + A12 B21, A11 B12 + A12 B22, A21 B11 + A22 B21, A21 B12 + A22 B22.

    TODO: compute the four output blocks here.
    """
    pass  # TODO

def recursive_matmul(A, B, leaf=64):
    """
    Recursive divide-and-conquer matrix product for square inputs. Pads to
    an even size when needed, recurses via block_product down to a crossover
    `leaf`, then calls the cubic kernel.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if leaf < 1:
        raise ValueError("leaf must be at least 1")
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("this square recursive harness expects two n x n matrices")

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
