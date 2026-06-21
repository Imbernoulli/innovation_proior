The problem is to multiply two n×n matrices faster than the schoolbook Θ(n³) bound. At first glance the cubic cost seems unavoidable: each output entry c_ij is the sum Σ_k a_ik b_kj, so there are n³ monomial products a_ik b_kj and one naturally believes each must be formed explicitly. Naive divide-and-conquer does not help either—splitting each matrix into four (n/2)×(n/2) blocks and recursing on the eight block products gives T(n)=8T(n/2)+O(n²), whose exponent log₂8=3 simply reproduces cubic time. The lesson from Karatsuba multiplication for polynomials and integers shows that the intuition is wrong: a bilinear form can sometimes be assembled from combined products whose unwanted cross-terms cancel, but the polynomial case does not directly transfer to matrices because the four output blocks are four separate bilinear forms and because the blocks do not commute.

The lever for beating cubic time is the number of recursive block multiplications, not the additions. For a block recursion with r multiplications the recurrence is T(n)=rT(n/2)+O(n²), so the exponent is log₂r. Eight gives 3; any r<8 breaks the cubic barrier. The question becomes whether two 2×2 block matrices can be multiplied with fewer than eight products while respecting non-commutativity—every intermediate product must be (a linear combination of A-blocks) times (a linear combination of B-blocks), never an A-block on the right of a B-block or a product of two blocks from the same matrix.

The new method is Strassen's algorithm. It multiplies two 2×2 block matrices using exactly seven bilinear products and recombines them with additions and subtractions to recover all four output blocks. The seven products are P1=(A11+A22)(B11+B22), P2=(A21+A22)B11, P3=A11(B12−B22), P4=A22(B21−B11), P5=(A11+A12)B22, P6=(A21−A11)(B11+B12), and P7=(A12−A22)(B21+B22). Each Pk has an A-side combination on the left and a B-side combination on the right, so the identity remains valid when the blocks themselves are matrices. The four outputs are then C11=P1+P4−P5+P7, C12=P3+P5, C21=P2+P4, and C22=P1−P2+P3+P6. Expanding these combinations formally cancels every unwanted monomial, leaving exactly the eight required block products. Because the seven products can be substituted for the eight naive block products at every level of recursion, the running time becomes T(n)=7T(n/2)+O(n²), giving Θ(n^{log₂7}) with log₂7≈2.807.

Strassen's algorithm is therefore not just a clever base case; it is a tensor-rank witness. A bilinear multiplication algorithm with r products is a decomposition of the matrix-multiplication tensor into r rank-one tensors u⊗v⊗w, and the minimum such r is the tensor rank. Matrix tensors multiply by taking tensor powers, and rank is submultiplicative, so a rank-7 decomposition of the 2×2×2 base tensor yields rank O(n^{log₂7}) for n×n matrices. This is why saving one multiplication at the base level compounds across all recursive levels and lowers the exponent. In practice one switches back to a tuned cubic kernel below a crossover size, because the extra additions and recursion overhead make Strassen's method advantageous only for sufficiently large matrices. Floating-point implementations also incur somewhat larger backward error than the classical algorithm because the additional subtractions create more cancellation, but over exact fields the algorithm is algebraically exact.

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
    a, b, c, d = split(A)            # A = [[a, b], [c, d]]
    e, f, g, h = split(B)            # B = [[e, f], [g, h]]

    # The seven Strassen products.  Each is (combination of A-blocks) x (combination of B-blocks):
    # the A combination always stays on the left, so the identity lifts to non-commuting blocks.
    p1 = recurse(a + d, e + h)       # (A11 + A22)(B11 + B22)
    p2 = recurse(c + d, e)           # (A21 + A22) B11
    p3 = recurse(a, f - h)           # A11 (B12 - B22)
    p4 = recurse(d, g - e)           # A22 (B21 - B11)
    p5 = recurse(a + b, h)           # (A11 + A12) B22
    p6 = recurse(c - a, e + f)       # (A21 - A11)(B11 + B12)
    p7 = recurse(b - d, g + h)       # (A12 - A22)(B21 + B22)

    # Recombine; cross terms cancel exactly by the Strassen identity.
    C = np.empty(A.shape, dtype=np.result_type(A, B))
    m = a.shape[0]
    C[:m, :m] = p1 + p4 - p5 + p7    # C11
    C[:m, m:] = p3 + p5              # C12
    C[m:, :m] = p2 + p4              # C21
    C[m:, m:] = p1 - p2 + p3 + p6    # C22

    return C

def recursive_matmul(A, B, leaf=64):
    """Recursive Strassen matrix product for square inputs."""
    A = np.asarray(A)
    B = np.asarray(B)
    if leaf < 1:
        raise ValueError("leaf must be at least 1")
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("recursive_matmul expects two n x n matrices")

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
