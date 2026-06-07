# Winograd minimal-filtering fast convolution for convnets

## Problem

A convnet layer correlates K filters (C channels, R×S) against N images (C channels, H×W). Direct convolution spends R·S multiplications per output (9 for a 3×3 filter), so the layer costs about R·S·N·H'·W'·C·K multiplications — the bottleneck for both training and low-latency inference. FFT convolution helps only for large filters, large tiles, and large batches; the dominant 3×3 small-batch regime needs something else. The goal: compute the *exact* convolution with far fewer multiplications, keeping the heavy stage a dense matrix multiply that is efficient even at batch size 1, with a small memory workspace.

## Key idea

Linear convolution is polynomial multiplication. Computing m outputs of an r-tap FIR filter, F(m,r), needs a minimum of µ(F(m,r)) = m+r−1 general multiplications (one per distinct input the m-window touches), versus m·r for direct. The minimum is reached by Cook–Toom: evaluate filter and data polynomials at m+r−1 points (small integers 0, ±1, ±2, … plus the point at infinity), multiply the values pointwise (the only general multiplies), and Lagrange-interpolate the product back. Filtering is the transpose of linear convolution, so by Winograd's matrix-exchange the filtering algorithm is

  Y = Aᵀ [ (G g) ⊙ (Bᵀ d) ]   (1D),

with data-independent transforms: filter transform G (r→α), data transform Bᵀ (α→α), elementwise multiply (⊙), inverse transform Aᵀ (α→m), α = m+r−1. Nesting a minimal 1D F(m,r) with F(n,s) gives a minimal 2D F(m×n, r×s) using (m+r−1)(n+s−1) multiplications:

  Y = Aᵀ [ (G g Gᵀ) ⊙ (Bᵀ d B) ] A   (2D).

For F(2×2, 3×3): 4×4 = 16 multiplies versus direct 36 → 2.25×, on a 4×4 tile. For F(4×4, 3×3): 6×6 = 36 versus 144 → 4×, on a 6×6 tile. The cost is paid in cheap additions and small constant-scalings, which grow quadratically with tile size, and the transform coefficients grow in magnitude with tile size (the F(4,3) filter transform already has 1/24 entries), degrading accuracy — so F(2×2,3×3) is the small, best-conditioned operating point, while F(4×4,3×3) is the larger speed/accuracy tradeoff when the precision budget allows it.

The layer-level payoff: sum over channels *inside* the transform domain (amortizing the inverse transform over C), so for each transform-domain coordinate (ξ,ν) ∈ {0..α−1}²,

  M^(ξ,ν) = Σ_c U^(ξ,ν)_{k,c} V^(ξ,ν)_{c,b} = U^(ξ,ν) V^(ξ,ν),

a (K×C)·(C×P) matrix multiply. The whole layer's heavy stage is α² independent dense GEMMs whose reduction dimension is channels, not batch — so it is efficient even at batch 1, and its multiply stage is exactly 1 real multiply per input, strictly beating FFT's ≥1.5 (with Hermitian symmetry + fast complex multiply).

## Final algorithm (forward layer, F(m×m, r×r))

1. Filter transform: U = G g Gᵀ for every filter/channel; scatter to α² matrices U^(ξ,ν) of shape K×C.
2. Data transform: divide each channel into α×α tiles overlapping by r−1; V = Bᵀ d B; scatter to α² matrices V^(ξ,ν) of shape C×P.
3. Multiply stage: α² batched GEMMs M^(ξ,ν) = U^(ξ,ν) V^(ξ,ν) (K×P).
4. Inverse transform: gather M back to α×α per output tile; Y_tile = Aᵀ M A (m×m).

Backprop reuses the same machinery: the input gradient is a convolution with flipped filters (same algorithm); the weight gradient F(R×S, H×W) is decomposed into a sum of small convolutions, e.g. F(3×3, 2×2) overlap-add.

## Code

Transform generator (Cook–Toom from interpolation points; builds and symbolically verifies G, Bᵀ, Aᵀ):

```python
import operator
from functools import reduce
from sympy import IndexedBase, Matrix, Poly, Rational as Q, simplify, symbols, zeros

def At(a, m, n):                      # Vandermonde: evaluate poly at points a[i]
    return Matrix(m, n, lambda i, j: a[i] ** j)

def A(a, m, n):                       # append the "infinity" row -> top coefficient
    return At(a, m - 1, n).row_insert(m - 1, Matrix(1, n, lambda i, j: 1 if j == n - 1 else 0))

def T(a, n):                          # modified Cook-Toom degree-reduction column
    return Matrix(Matrix.eye(n).col_insert(n, Matrix(n, 1, lambda i, j: -(a[i] ** n))))

def Lx(a, n):                         # Lagrange numerators prod_{k!=i}(x - a[k])
    x = symbols("x")
    return Matrix(n, 1, lambda i, j: Poly(
        reduce(operator.mul, ((x - a[k] if k != i else 1) for k in range(n)), 1).expand(), x).as_expr())

def F(a, n):                          # interpolation denominators -> the 1/4,1/6,1/24 scalings
    return Matrix(n, 1, lambda i, j: reduce(
        operator.mul, ((a[i] - a[k] if k != i else 1) for k in range(n)), 1))

def Fdiag(a, n):
    f = F(a, n); return Matrix(n, n, lambda i, j: (f[i, 0] if i == j else 0))

def FdiagPlus1(a, n):
    f = Fdiag(a, n - 1); f = f.col_insert(n - 1, zeros(n - 1, 1))
    return f.row_insert(n - 1, Matrix(1, n, lambda i, j: (1 if j == n - 1 else 0)))

def L(a, n):
    x = symbols("x"); lx, f = Lx(a, n), F(a, n)
    return Matrix(n, n, lambda i, j: lx[i, 0].coeff(x, j) / f[i]).T

def Bt(a, n): return L(a, n) * T(a, n)
def B(a, n):  return Bt(a, n - 1).row_insert(n - 1, Matrix(1, n, lambda i, j: 1 if j == n - 1 else 0))

def cook_toom_filter(points, n_out, r):
    alpha = n_out + r - 1
    f = FdiagPlus1(points, alpha)
    if f[0, 0] < 0: f[0, :] *= -1
    AT = A(points, alpha, n_out).T
    G  = (A(points, alpha, r).T * f ** (-1)).T   # fractions land in the offline filter transform
    BT = f * B(points, alpha).T
    return AT, G, BT

def filter_verify(n_out, r, AT, G, BT):
    alpha = n_out + r - 1
    d = Matrix(alpha, 1, lambda i, j: IndexedBase("d")[i])
    g = Matrix(r, 1, lambda i, j: IndexedBase("g")[i])
    return simplify(AT * (G * g).multiply_elementwise(BT * d))

# F(2,3) -> 4 multiplies; F(4,3) -> 6 multiplies (G has 1/24-scale entries)
AT, G, BT = cook_toom_filter((0, 1, -1), 2, 3)
assert BT == Matrix([[1, 0, -1, 0],
                     [0, 1,  1, 0],
                     [0, -1, 1, 0],
                     [0, -1, 0, 1]])
assert G == Matrix([[1, 0, 0],
                    [Q(1, 2), Q(1, 2), Q(1, 2)],
                    [Q(1, 2), -Q(1, 2), Q(1, 2)],
                    [0, 0, 1]])
assert AT == Matrix([[1, 1, 1, 0],
                     [0, 1, -1, 1]])
assert list(filter_verify(2, 3, AT, G, BT)) == [
    IndexedBase("d")[0]*IndexedBase("g")[0] + IndexedBase("d")[1]*IndexedBase("g")[1] + IndexedBase("d")[2]*IndexedBase("g")[2],
    IndexedBase("d")[1]*IndexedBase("g")[0] + IndexedBase("d")[2]*IndexedBase("g")[1] + IndexedBase("d")[3]*IndexedBase("g")[2],
]
AT4, G4, BT4 = cook_toom_filter((0, 1, -1, 2, -2), 4, 3)
assert G4 == Matrix([[Q(1, 4), 0, 0],
                     [-Q(1, 6), -Q(1, 6), -Q(1, 6)],
                     [-Q(1, 6),  Q(1, 6), -Q(1, 6)],
                     [Q(1, 24),  Q(1, 12), Q(1, 6)],
                     [Q(1, 24), -Q(1, 12), Q(1, 6)],
                     [0, 0, 1]])
assert BT4 == Matrix([[4, 0, -5, 0, 1, 0],
                      [0, -4, -4, 1, 1, 0],
                      [0, 4, -4, -1, 1, 0],
                      [0, -2, -1, 2, 1, 0],
                      [0, 2, -1, -2, 1, 0],
                      [0, 4, 0, -5, 0, 1]])
assert AT4 == Matrix([[1, 1, 1, 1, 1, 0],
                      [0, 1, -1, 2, -2, 0],
                      [0, 1, 1, 4, 4, 0],
                      [0, 1, -1, 8, -8, 1]])
AT32, G32, BT32 = cook_toom_filter((0, 1, -1), 3, 2)
assert BT32 == Matrix([[1, 0, -1, 0],
                       [0, 1,  1, 0],
                       [0, -1, 1, 0],
                       [0, -1, 0, 1]])
assert G32 == Matrix([[1, 0],
                      [Q(1, 2), Q(1, 2)],
                      [Q(1, 2), -Q(1, 2)],
                      [0, 1]])
assert AT32 == Matrix([[1, 1, 1, 0],
                       [0, 1, -1, 0],
                       [0, 1, 1, 1]])
```

Numeric F(2×2, 3×3) convolution layer (matches direct convolution to fp32 rounding):

```python
import numpy as np

BT = np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,-1,0,1]], dtype=np.float32)
G  = np.array([[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]], dtype=np.float32)
AT = np.array([[1,1,1,0],[0,1,-1,1]], dtype=np.float32)

def direct_conv_valid(D, W):
    D = np.asarray(D, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    N, C, H, Wd = D.shape
    K, Cw, R, S = W.shape
    assert C == Cw
    Y = np.zeros((N, K, H - R + 1, Wd - S + 1), np.float32)
    for n in range(N):
        for k in range(K):
            for c in range(C):
                for i in range(H - R + 1):
                    for j in range(Wd - S + 1):
                        Y[n, k, i, j] += np.sum(D[n, c, i:i+R, j:j+S] * W[k, c])
    return Y

def conv_layer(D, W):
    # D: (N,C,H,W)  W: (K,C,3,3)  ->  Y: (N,K,H-2,W-2)   (valid F(2x2,3x3))
    D = np.asarray(D, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    N, C, H, Wd = D.shape
    K, Cw, R, S = W.shape
    assert (C, R, S) == (Cw, 3, 3)
    m, r, alpha = 2, 3, 4
    Ho, Wo = H - r + 1, Wd - r + 1
    tH, tW = (Ho + m - 1)//m, (Wo + m - 1)//m
    P = N*tH*tW
    Dp = np.zeros((N, C, tH*m + r - 1, tW*m + r - 1), np.float32)
    Dp[:, :, :H, :Wd] = D

    U = np.zeros((alpha, alpha, K, C), np.float32)          # filter transform G g G^T
    for k in range(K):
        for c in range(C):
            U[:, :, k, c] = G @ W[k, c] @ G.T

    V = np.zeros((alpha, alpha, C, P), np.float32)          # data transform B^T d B
    coords = []; b = 0
    for n in range(N):
        for ti in range(tH):
            for tj in range(tW):
                for c in range(C):
                    tile = Dp[n, c, ti*m:ti*m+alpha, tj*m:tj*m+alpha]
                    V[:, :, c, b] = BT @ tile @ BT.T
                coords.append((n, ti, tj)); b += 1

    M = np.zeros((alpha, alpha, K, P), np.float32)          # alpha^2 GEMMs, 1 real mult/input
    for xi in range(alpha):
        for nu in range(alpha):
            M[xi, nu] = U[xi, nu] @ V[xi, nu]

    Y = np.zeros((N, K, tH*m, tW*m), np.float32)            # single inverse transform A^T M A
    for k in range(K):
        for b, (n, ti, tj) in enumerate(coords):
            Y[n, k, ti*m:ti*m+m, tj*m:tj*m+m] = AT @ M[:, :, k, b] @ AT.T
    return Y[:, :, :Ho, :Wo]

rng = np.random.default_rng(0)
Dtest = rng.normal(size=(2, 3, 7, 8)).astype(np.float32)
Wtest = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
np.testing.assert_allclose(conv_layer(Dtest, Wtest), direct_conv_valid(Dtest, Wtest),
                           rtol=1e-5, atol=2e-5)
```
