Deep recognition networks are now built almost entirely from small $3\times 3$ convolution layers, and those layers are where essentially all the compute goes — paid once across the GPU-days of training and again at every inference. A layer correlates $K$ filters of $C$ channels and size $R\times S$ against $N$ images of $C$ channels and size $H\times W$, and the direct way of doing it spends $R\cdot S$ multiplications per output pixel per channel — nine multiplies for a $3\times 3$ filter — so the whole layer costs on the order of $R\cdot S\cdot N\cdot H'\cdot W'\cdot C\cdot K$ multiplications. That multiply count is the wall. The two regimes that matter most are the worst for the existing fast methods: very small batches (a single image for low-latency detection, small per-node minibatches when training is sharded) and small filters ($3\times 3$ almost exclusively). The natural way to make a layer fast on a GPU is to cast it as a dense matrix multiply (im2col + GEMM), which is exact and needs no workspace but is minimal for nothing — it pays all nine multiplies. The FFT route turns convolution into pointwise multiplication in the Fourier domain and amortizes the transforms across many feature maps, but it is built for the opposite corner of the problem: its pointwise stage multiplies *complex* numbers (four real multiplies, three with a fast-complex-multiply trick, roughly halved again by Hermitian symmetry, but never below one real multiply per input), it needs large tiles to amortize the transform (a $64\times 64$ tile turns a nine-element filter into 4096 stored units and a multi-gigabyte workspace), and it needs a large batch to generate enough tiles to fill an efficient pointwise stage — exactly why a vendor library that flips to FFT at moderate batch can collapse below 2 TFLOPS. Strassen recursion on the layer cuts the *number* of convolutions by $8/7$ per step but halves all three matrix dimensions, shrinking precisely what a GEMM needs to be large; and low-precision quantization changes the datatype, not the multiply count. None of these reduces the arithmetic of a *small* filter while keeping the heavy stage a GEMM that stays efficient at batch one.

The instinct that nine multiplies per window is forced rests on computing a length-9 dot product the sliding way, forming each product and summing. But the outputs are bilinear forms in the data and the filter, and a bilinear form need not be realized by literally forming each monomial: one can take linear combinations of the inputs, multiply *those*, and arrange for the unwanted cross-terms to cancel in the recombination — exactly as $(a+ib)(c+id)$ needs only three real multiplies instead of four. On multiply-bound hardware where an add is nearly free, trading multiplies for adds is a pure win. I propose the Winograd minimal-filtering convolution, which realizes this trade at the proven arithmetic minimum. The load-bearing fact is that linear convolution *is* polynomial multiplication: writing the filter as $g(x)=g_0+g_1 x+g_2 x^2$ and the data as $d(x)$, the coefficients of $g(x)\,d(x)$ are the linear convolution of the two sequences. Computing $m$ outputs of an $r$-tap filter, denoted $F(m,r)$, therefore needs a minimum of $\mu(F(m,r)) = m+r-1$ general multiplications — one per distinct input value the $m$-output window touches — against the $m\cdot r$ of direct computation. The minimum is reached by Cook–Toom: a degree-$(m+r-2)$ result is pinned down by its values at $m+r-1$ points, so evaluate $g$ and $d$ at $m+r-1$ chosen points (small integers $0,\pm 1,\pm 2,\dots$ plus the "point at infinity," which simply reads off the top coefficient), multiply the values pointwise — these are the only general multiplies — and Lagrange-interpolate the product back. Choosing small integer points makes the evaluations and the interpolation cheap adds and scalings by small constants that depend only on the points, never on the data, so the matrices are built once. Asking for the full linear convolution would re-derive the direct count, because filtering wants only the $m$ valid outputs, not all $m+r-1$ product coefficients; the modified Cook–Toom move of taking the point at infinity reduces the degree to exactly what is needed, landing on $m+r-1$.

The piece that makes it a *filtering* algorithm rather than a *linear-convolution* one is the transpose. Filtering — a fixed filter slid to produce $m$ valid outputs — is the transpose of the linear-convolution map. Factor linear convolution as $T = C\,H\,D$ with $D$ a pre-addition matrix forming the input combinations, $H$ diagonal (its nonzero count is the multiply count, $m+r-1$ by construction), and $C$ the interpolation recombination; by Winograd's matrix-exchange theorem the filtering problem is computed by transposing this factorization, so the data and recombination roles swap and transpose. The end form in 1D is
$$Y = A^{\mathsf T}\,\big[\,(G\,g)\odot(B^{\mathsf T} d)\,\big],$$
where $\odot$ is elementwise multiply: the filter transform $G$ takes $g$ into the point-value domain, the data transform $B^{\mathsf T}$ takes $d$ into the same domain, the $\alpha=m+r-1$ elementwise products are the only real multiplies, and $A^{\mathsf T}$ interpolates them back to the $m$ outputs. For $F(2,3)$ with points $\{0,1,-1,\infty\}$ the transforms are $B^{\mathsf T}=\big[\begin{smallmatrix}1&0&-1&0\\0&1&1&0\\0&-1&1&0\\0&-1&0&1\end{smallmatrix}\big]$, $G=\big[\begin{smallmatrix}1&0&0\\1/2&1/2&1/2\\1/2&-1/2&1/2\\0&0&1\end{smallmatrix}\big]$, $A^{\mathsf T}=\big[\begin{smallmatrix}1&1&1&0\\0&1&-1&1\end{smallmatrix}\big]$, giving four multiplies versus the direct six, and expanding $A^{\mathsf T}[(Gg)\odot(B^{\mathsf T}d)]$ recovers exactly $y_0=d_0g_0+d_1g_1+d_2g_2$ and $y_1=d_1g_0+d_2g_1+d_3g_2$, with the price paid in additions. Because $G\,g$ depends only on the filter, and a convnet filter is reused across every tile of the image and the whole batch, the filter transform is computed once and amortized — only the data-side transform scales with the image.

Minimal 1D algorithms nest into minimal 2D ones: applying $F(m,r)$ along rows and $F(n,s)$ along columns gives $F(m\times n, r\times s)$ with $(m+r-1)(n+s-1)$ multiplies. For square $3\times 3$ filters,
$$Y = A^{\mathsf T}\,\big[\,(G\,g\,G^{\mathsf T})\odot(B^{\mathsf T} d\,B)\,\big]\,A,$$
where $g$ is the $r\times r$ filter, $d$ an $\alpha\times\alpha$ tile, $\alpha=m+r-1$: the filter transform $G\,g\,G^{\mathsf T}$ and the data transform $B^{\mathsf T} d\,B$ each produce $\alpha\times\alpha$ arrays, the elementwise product is $\alpha^2$ real multiplies, and $A^{\mathsf T}[\cdot]A$ interpolates back to an $m\times m$ output tile. $F(2\times 2,3\times 3)$ uses $4\times 4=16$ multiplies against the direct $36$ — a $2.25\times$ reduction — on a tiny $4\times 4$ tile, the opposite of the FFT's $64\times 64$. The layer-level payoff comes from the channel sum. Computing $A^{\mathsf T}[\cdot]A$ per (filter, channel, tile) and then summing over channels would pay the inverse transform $C$ times per output tile; but the channel sum and the inverse transform are both linear, so the sum can be pushed *inside* the transform domain and the inverse transform applied just once per output tile,
$$Y_{i,k,\mathrm{tile}} = A^{\mathsf T}\,\Big[\,\textstyle\sum_c (G\,g_{k,c}\,G^{\mathsf T})\odot(B^{\mathsf T} d_{c,i,\mathrm{tile}}\,B)\,\Big]\,A.$$
Writing $U_{k,c}=G\,g_{k,c}\,G^{\mathsf T}$ and $V_{c,b}=B^{\mathsf T} d_{c,b}\,B$ and fixing a single transform-domain coordinate $(\xi,\nu)\in\{0,\dots,\alpha-1\}^2$, the inner sum becomes
$$M^{(\xi,\nu)}_{k,b}=\sum_c U^{(\xi,\nu)}_{k,c}\,V^{(\xi,\nu)}_{c,b}=U^{(\xi,\nu)}\,V^{(\xi,\nu)},$$
a $(K\times C)\cdot(C\times P)$ dense matrix multiply. So the entire heavy stage of the layer is $\alpha^2$ independent GEMMs — 16 of them for $F(2\times 2,3\times 3)$ — whose reduction dimension is *channels*, not batch. That is the property I was chasing: the GEMMs stay efficient even at batch one, because a single image already supplies full-width matrices over $C$, where the FFT's pointwise stage starves without a large batch. And the multiply stage is exactly one real multiply per input, which strictly beats the FFT's complex-multiply stage — with Hermitian symmetry and the fast three-real-multiply complex product, $3(\lfloor\alpha/2\rfloor+1)/\alpha > 1.5$ real multiplies per input for every tile size — on multiplies, on workspace memory, and on the small-batch regime simultaneously.

The tile size cannot be cranked up freely to drive the normalized cost $\alpha'=(m+r-1)^2/m^2$ toward one. Two costs grow with the tile: the number of additions and constant-multiplications in the transforms grows quadratically, and — the sharper limit — the magnitudes of the transform-matrix entries grow, because adding interpolation points $2,-2,3,-3,\dots$ makes the Lagrange denominators $\prod_{k\neq i}(\beta_i-\beta_k)$ blow up. Carrying the CRT construction out to $F(4,3)$ with points $\{0,1,-1,2,-2,\infty\}$ already produces denominators $4, 6, 24$ — the filter transform $G$ acquires $1/24$-scale entries — and $F(6,3)$ throws up $1/720$; in finite precision such matrices lose bits. So there is a sweet spot at *small* tiles, not a limit at one: $F(2\times 2,3\times 3)$ (tile 4, transform entries only $0,\pm 1,\pm\tfrac12$, best conditioned, $2.25\times$) is the safe default, and $F(4\times 4,3\times 3)$ (tile 6, $36$ vs $144$, the full $4\times$) is available when the precision budget — convnets being known to tolerate surprisingly few bits — can absorb the larger roundoff. The same machinery covers backprop: the input gradient is a convolution with spatially flipped filters, so the forward algorithm applies unchanged; the weight gradient is $F(R\times S, H\times W)$, whose huge $H\times W$ "filter" is exactly the large tile to avoid, so it is decomposed into a sum of small convolutions — e.g. $F(3\times 3, 2\times 2)$, the tile-4 sibling of $F(2\times 2,3\times 3)$, overlap-added over the large operands. I deliberately do not stack Strassen on top: its $8/7$ gain comes at the cost of halving $K$, $C$, and $P$, the dimensions my $\alpha^2$ GEMMs need to be large, so it is a net loss here except possibly for layers with very large $C$, $K$, $P$.

In code, two layers. First the transform generator — the Cook–Toom construction from interpolation points, which produced every $G$, $B^{\mathsf T}$, $A^{\mathsf T}$ above and verifies them symbolically rather than by transcription:

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

Then the numeric $F(2\times 2,3\times 3)$ layer — tile, transform, the $\alpha^2$ GEMMs that came out of the channel sum, and the single inverse transform — matching direct convolution to fp32 rounding:

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
