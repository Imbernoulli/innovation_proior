# Winograd minimal filtering for convolutional networks

## Problem

A 3x3 convolution layer correlates K filters (C channels, r x r) against N images (C channels,
H x W). Computed directly, each output is a length-r² dot product, so the layer costs
r²·N·H·W·C·K multiplications. On networks built from small 3x3 filters this multiplication count
dominates training and inference. FFT-based convolution reduces it but only with large tiles,
complex arithmetic (~1.5–2 real multiplies/input), heavy workspace, and large batches — the
wrong regime for small filters and small batches. The goal: compute the *exact* small
convolution with the minimum number of multiplications, on small tiles, in real arithmetic,
with overhead cheap enough that it pays off at any batch size.

## Key idea

A small convolution is generated from a polynomial product: multiply a length-m data polynomial
by a length-r filter polynomial, evaluate/interpolate that degree-(m+r−2) product, then transpose
the linear-convolution maps to get the FIR filtering maps. The product has m+r−1 coefficients,
hence is determined by its values at m+r−1 distinct points, and the value of a product is the
product of the values. So compute F(m,r) in three stages: **evaluate** the short data side and g
at m+r−1 points (linear maps = additions and fixed constants), **multiply** the m+r−1 value pairs
pointwise (the only data-dependent multiplications), **interpolate** y from its samples (a linear
map, inverse Vandermonde), then transpose into filtering form. This is the minimal number of
multiplications,

    μ(F(m,r)) = m + r − 1,

one per input. In matrix form, with data transform B^T, filter transform G, inverse transform
A^T:

    Y = A^T [ (G g) ⊙ (B^T d) ]        (⊙ = elementwise product).

**Nesting to 2D.** The 1D algorithm nests with itself: F(m×m, r×r) uses

    Y = A^T [ (G g G^T) ⊙ (B^T d B) ] A,

with (m+r−1)² pointwise multiplies. F(2×2,3×3): 16 vs 36 direct = 2.25× fewer. F(4×4,3×3):
36 vs 144 = 4× fewer. Tiles are (m+r−1)×(m+r−1) with r−1 overlap. Larger tiles raise the
asymptotic reduction but the transform additions grow ~quadratically and the transform-entry
magnitudes grow (eroding accuracy), so small tiles F(2,3)/F(4,3) are the sweet spot.

**F(2,3), points {0,1,−1,∞}** — 4 multiplies:

    m1=(d0−d2)g0,  m2=(d1+d2)(g0+g1+g2)/2,  m3=(d2−d1)(g0−g1+g2)/2,  m4=(−d1+d3)g2
    y0 = m1+m2+m3 = d0g0+d1g1+d2g2,   y1 = m2−m3+m4 = d1g0+d2g1+d3g2
    Equivalently, with m4'=(d1−d3)g2, y1=m2−m3−m4'.

    B^T=[[1,0,−1,0],[0,1,1,0],[0,−1,1,0],[0,−1,0,1]]
    G =[[1,0,0],[1/2,1/2,1/2],[1/2,−1/2,1/2],[0,0,1]]
    A^T=[[1,1,1,0],[0,1,−1,1]]

**F(4,3), points {0,1,−1,2,−2,∞}** — 6 multiplies (via the Chinese-Remainder / Cook-Toom
construction on m(x)=x(x−1)(x+1)(x−2)(x+2)):

    B^T=[[4,0,−5,0,1,0],[0,−4,−4,1,1,0],[0,4,−4,−1,1,0],[0,−2,−1,2,1,0],[0,2,−1,−2,1,0],[0,4,0,−5,0,1]]
    G =[[1/4,0,0],[−1/6,−1/6,−1/6],[−1/6,1/6,−1/6],[1/24,1/12,1/6],[1/24,−1/12,1/6],[0,0,1]]
    A^T=[[1,1,1,1,1,0],[0,1,−1,2,−2,0],[0,1,1,4,4,0],[0,1,−1,8,−8,1]]

## Layer algorithm (the GEMM reformulation)

Pull the linear inverse transform outside the channel sum (amortizing it over C), and observe
that the channel reduction at each fixed transform coordinate (ξ,ν) is a matrix multiply:

    M^{(ξ,ν)}_{k,b} = Σ_c U^{(ξ,ν)}_{k,c} V^{(ξ,ν)}_{c,b},  i.e.  M^{(ξ,ν)} = U^{(ξ,ν)} V^{(ξ,ν)},

with U^{(ξ,ν)} a K×C matrix and V^{(ξ,ν)} a C×P matrix (P = N⌈H/m⌉⌈W/m⌉ tiles). The elementwise
stage becomes (m+r−1)² dense GEMMs over filters × channels × tiles — efficient even at N=1,
because P is large.

1. Filter transform (once per filter): u = G g_{k,c} G^T; scatter U^{(ξ,ν)}_{k,c} = u_{ξ,ν}.
2. Data transform (per tile): v = B^T d_{c,b} B; scatter V^{(ξ,ν)}_{c,b} = v_{ξ,ν}.
3. For each (ξ,ν): M^{(ξ,ν)} = U^{(ξ,ν)} V^{(ξ,ν)}.
4. Gather m_{ξ,ν} = M^{(ξ,ν)}_{k,b}; inverse transform Y_{k,b} = A^T m A (once per tile).

Direct convolution is the m=1 corner (F(1×1,R×S)). Backward pass: the input gradient is the
same F(m,r) on the flipped filter; the weight gradient F(R×S,H×W) is decomposed into a direct
sum of small tiles, e.g. F(3×3,2×2) (4×4 tiles, overlap 2), again 2.25×.

## Code

The transforms are *generated* from the interpolation points (symbolically, so the constants
are exact), then applied numerically.

```python
import operator
from functools import reduce
from sympy import IndexedBase, Matrix, Poly, simplify, symbols, zeros

def At(a, m, n):  return Matrix(m, n, lambda i, j: a[i] ** j)
def A(a, m, n):   return At(a, m - 1, n).row_insert(
                      m - 1, Matrix(1, n, lambda i, j: 1 if j == n - 1 else 0))
def Lx(a, n):
    x = symbols("x")
    return Matrix(n, 1, lambda i, j: Poly(
        reduce(operator.mul, ((x - a[k] if k != i else 1) for k in range(n)), 1
              ).expand(basic=True), x).as_expr())
def F(a, n):  return Matrix(n, 1, lambda i, j: reduce(
                  operator.mul, ((a[i] - a[k] if k != i else 1) for k in range(n)), 1))
def L(a, n):
    x = symbols("x"); lx, f = Lx(a, n), F(a, n)
    return Matrix(n, n, lambda i, j: lx[i, 0].coeff(x, j) / f[i]).T
def T(a, n):  return Matrix(Matrix.eye(n).col_insert(
                  n, Matrix(n, 1, lambda i, j: -(a[i] ** n))))
def Bt(a, n): return L(a, n) * T(a, n)
def B(a, n):  return Bt(a, n - 1).row_insert(
                  n - 1, Matrix(1, n, lambda i, j: 1 if j == n - 1 else 0))
def Fdiag(a, n):
    f = F(a, n); return Matrix(n, n, lambda i, j: (f[i, 0] if i == j else 0))
def FdiagPlus1(a, n):
    f = Fdiag(a, n - 1).col_insert(n - 1, zeros(n - 1, 1))
    return f.row_insert(n - 1, Matrix(1, n, lambda i, j: (1 if j == n - 1 else 0)))

def cookToomFilter(a, n, r):
    """Transforms (AT, G, BT, f) for F(n=m, r). Constant fractions absorbed onto G."""
    alpha = n + r - 1
    f = FdiagPlus1(a, alpha)
    if f[0, 0] < 0:
        f[0, :] *= -1
    AT = A(a, alpha, n).T
    G  = (A(a, alpha, r).T * f ** (-1)).T
    BT = f * B(a, alpha).T
    return AT, G, BT, f

def filterVerify(n, r, AT, G, BT):
    alpha = n + r - 1
    di, gi = IndexedBase("d"), IndexedBase("g")
    d = Matrix(alpha, 1, lambda i, j: di[i]); g = Matrix(r, 1, lambda i, j: gi[i])
    return simplify(AT * (G * g).multiply_elementwise(BT * d))

# F(2,3): {0,1,-1}+inf -> 4 mults; F(4,3): {0,1,-1,2,-2}+inf -> 6 mults
AT, G, BT, _ = cookToomFilter((0, 1, -1), 2, 3)
AT4, G4, BT4, _ = cookToomFilter((0, 1, -1, 2, -2), 4, 3)

di, gi = IndexedBase("d"), IndexedBase("g")
assert filterVerify(2, 3, AT, G, BT) == Matrix(2, 1, lambda i, j:
    di[i]*gi[0] + di[i+1]*gi[1] + di[i+2]*gi[2])
assert filterVerify(4, 3, AT4, G4, BT4) == Matrix(4, 1, lambda i, j:
    di[i]*gi[0] + di[i+1]*gi[1] + di[i+2]*gi[2])
```

```python
import numpy as np

def fast_conv_tile(d, g, AT, G, BT):
    """F(m x m, r x r): Y = A^T [ (G g G^T) (elementwise) (B^T d B) ] A."""
    U = G @ g @ G.T
    V = BT @ d @ BT.T
    return AT @ (U * V) @ AT.T

def conv_layer(D, g, AT, G, BT, m, r):
    """D: (N,C,H,W); g: (K,C,r,r). Channel reduction in transform space = alpha^2 GEMMs."""
    N, C, H, W = D.shape
    K, alpha = g.shape[0], m + r - 1
    tiles = [(i, x, y)
             for i in range(N)
             for x in range(0, H - r + 1, m)
             for y in range(0, W - r + 1, m)]
    P = len(tiles)
    U = np.zeros((alpha, alpha, K, C)); V = np.zeros((alpha, alpha, C, P))
    for k in range(K):
        for c in range(C):
            U[:, :, k, c] = G @ g[k, c] @ G.T          # filter transform (hoisted)
    for c in range(C):
        for b, (i, x, y) in enumerate(tiles):
            V[:, :, c, b] = BT @ D[i, c, x:x+alpha, y:y+alpha] @ BT.T   # data transform
    Mt = np.einsum('xykc,xycb->xykb', U, V)            # one GEMM per (xi,nu)
    Y = np.zeros((K, P, m, m))
    for k in range(K):
        for b in range(P):
            Y[k, b] = AT @ Mt[:, :, k, b] @ AT.T       # inverse transform, once/tile
    return Y, tiles
```
