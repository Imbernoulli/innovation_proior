OK, let me think this through from scratch. I have a deep network that is almost entirely 3x3 convolution layers, and the thing eating all the compute is exactly those layers. A layer correlates K filters of C channels and size 3x3 against N images of C channels and size HxW. Count the multiplications the obvious way: every output pixel, every filter, every channel costs 3x3 = 9 multiplies, so the whole layer is 9·N·H·W·C·K multiplies. That count is the wall. I want to make it smaller — not by approximating the convolution, by computing the *same* convolution with fewer multiplications.

My gut says 9 per output window is forced. The filter has 9 taps, each output is a dot product of those 9 taps with 9 image values, and a dot product of length 9 is 9 multiplies. How could you possibly get that inner product without forming the 9 products? But I've been burned by that intuition before. "Every product is needed" is a statement about *one particular way* of computing — the sliding dot-product way. The outputs are bilinear forms in the image values and filter values. Nothing forces me to realize a bilinear form by literally forming each monomial and summing. Maybe I can take combinations of the inputs, multiply *those*, and have the cross-terms I don't want cancel. That's the only loophole, but it's a real one, and there's precedent.

There's an idea worth tracking: the FFT already exploits this loophole. The convolution theorem says convolution becomes pointwise multiplication in the Fourier domain. Transform the image tile, transform the filter, multiply elementwise, transform back. The pointwise stage is one multiply per input instead of nine. So in principle the cost of convolution is one-multiply-per-input, not r-multiplies-per-input. Why doesn't that just solve my problem? Because the FFT pays in the wrong currency for my regime. The pointwise products are *complex* — four real multiplies per complex multiply, and even after exploiting that a real signal's DFT is Hermitian-symmetric (so I only need about half the products) and the three-real-multiply trick for a complex product, I'm at roughly 1.5 real multiplies per input, not one. Worse, the FFT's transform overhead only amortizes when the tile is *large* — tile sizes of 16, 32, 64 — and large tiles are a disaster for 3x3 networks: a 3x3 filter blows up into a 64x64 transformed tile (9 numbers become 4096), the workspace memory goes to gigabytes, images whose size isn't near a multiple of the tile waste huge fractions of the computation on pixels I throw away, and with a small batch there are too few tiles to make the pointwise stage's matrix multiplies efficient. So FFT is the right *shape* of idea — transform, multiply pointwise, transform back — but the wrong instrument for small filters and small batches. I want the same shape with (a) real arithmetic, one multiply per input, and (b) tiles small enough that the transforms stay cheap.

So let me forget Fourier and ask the bare question: for a 1-D filter, compute m outputs of an r-tap FIR filter — call it F(m,r) — with as few multiplications as possible. Direct cost is m·r. Take the smallest nontrivial case, F(2,3): two outputs, three taps,

    y_0 = d_0 g_0 + d_1 g_1 + d_2 g_2
    y_1 = d_1 g_0 + d_2 g_1 + d_3 g_2.

Six multiplies the direct way. I want fewer. Let me look for the algebraic structure instead of staring at the dot products.

Convolution is polynomial multiplication — but I have to be precise about which convolution. If I multiply a length-m polynomial d(x) = d_0 + … + d_{m-1}x^{m-1} by an r-tap filter polynomial g(x) = g_0 + … + g_{r-1}x^{r-1}, the product has degree m+r−2 and its m+r−1 coefficients are the full linear convolution. My FIR filtering problem looks slightly different: it takes a length-(m+r−1) data tile and returns only m adjacent correlation outputs. The way to connect them is the transposition principle for bilinear algorithms. Derive a minimal linear-convolution algorithm for the length-m by length-r product, then transpose the data/output linear maps; the number of pointwise products is unchanged, and the result is an FIR filtering algorithm F(m,r). Good. The polynomial product is still the structural object, but the filtering matrices will not be raw evaluations of the long data tile.

A polynomial of degree n−1 is *uniquely determined by its values at n distinct points*. The linear-convolution product y(x) = g(x)d(x) has degree (r−1)+(m−1) = m+r−2, so it has m+r−1 coefficients, so it is pinned down by its values at any m+r−1 distinct points. And the value of a product at a point is just the product of the values: y(x_j) = g(x_j)·d(x_j). So I can compute the whole product in three moves. Evaluate g and the length-m data block at the same points — those are linear combinations, sums and scalings, no data-dependent multiplications. Then multiply the values pairwise: m+r−1 real multiplications, the only true multiplications. Then interpolate y from its m+r−1 sampled values — again a linear map, the inverse of the Vandermonde evaluation matrix, pure additions and constants. Transpose the resulting bilinear algorithm, and the FIR filter F(m,r) also uses m+r−1 multiplications. For F(2,3): m+r−1 = 4. Four multiplies, not six.

Let me make sure four is really the floor and not just where I happened to stop. The length-m by length-r product polynomial has m+r−1 coefficients; recovering that many independent bilinear quantities needs at least m+r−1 essential multiplications. Equivalently, fewer than m+r−1 sample points cannot pin down a degree-(m+r−2) polynomial — two different products would be indistinguishable. The transposed filtering problem has the same bilinear rank, so μ(F(m,r)) = m+r−1 exactly, and it equals the number of input samples I touch: one multiply per input. That's the same one-multiply-per-input the FFT promised, but I'm going to get it in *real* arithmetic.

Now actually build F(2,3). I need m+r−1 = 4 points for the length-2 by length-3 product. The product has degree 3, so four samples. Which points? This is a real design choice, not a formality, because the points become the entries of my transform matrices: evaluating at point x means forming Σ_k (coeff_k)·x^k, so the powers of the chosen points *are* the constants I multiply by. I want those constants tiny and simple, because every one of them is an extra addition or a constant multiply in the transform, and I'm trying to keep the overhead cheap. So pick the smallest things: 0, 1, −1, and then I need a fourth. A fourth small integer like 2 would work, but the leading coefficient is available as the point at infinity: the limit of y(x)/x^3 as x→∞. So my four points are {0, 1, −1, ∞}. After transposition, that infinity product is still the row that isolates the highest filter coefficient g_2, but the data side is a short linear combination of the four-sample filtering tile, not just a raw data endpoint.

Let me grind it out without pretending the filtering tile is a four-coefficient product polynomial. The linear-convolution algorithm evaluates the length-2 data side at 0, 1, −1, ∞, multiplies by the corresponding filter evaluations, and interpolates four coefficients. Transposing that algorithm gives four filtering products whose data rows are folded combinations of d_0,d_1,d_2,d_3. That is exactly the bookkeeping I need: it computes y_0 and y_1 directly instead of reconstructing the whole long product of the filtering tile.

Stack the four pointwise products. With the sign placement generated by the CRT normalization I will use in code, the data transform B^T times d = (d_0,d_1,d_2,d_3)^T is

    B^T = [ 1  0  -1  0 ]
          [ 0  1   1  0 ]
          [ 0 -1   1  0 ]
          [ 0 -1   0  1 ]

Let me read those rows. Row 1 gives d_0 − d_2; row 2 gives d_1 + d_2; row 3 gives −d_1 + d_2; row 4 gives −d_1 + d_3. These aren't literally d(0), d(1), d(−1), d(∞) on the nose — they're those evaluations folded together with the interpolation so the four products land as the right FIR outputs, but the *structure* is exactly "small-integer combinations of the data, no multiplications." Good. The filter evaluations, G times g = (g_0,g_1,g_2)^T:

    G = [  1     0     0  ]
        [ 1/2   1/2   1/2 ]
        [ 1/2  -1/2   1/2 ]
        [  0     0     1  ]

Row 1 is g_0; row 2 is (g_0+g_1+g_2)/2; row 3 is (g_0−g_1+g_2)/2; row 4 is g_2. There are the evaluations at 0, 1, −1, ∞ of the filter — with a factor of 1/2 absorbed into the filter side. Why 1/2, and why on the filter rather than the data? Because the interpolation has to divide by the products of point-differences (the Vandermonde determinant pieces), and those constant factors have to live *somewhere*; putting them on the filter transform G is the right place because in a convnet the filters are transformed far less often than the data — a filter is reused across all image tiles — so I'd rather pay the constant multiplications once per filter than once per tile. The four pointwise products are then

    m_1 = (d_0 − d_2)·g_0
    m_2 = (d_1 + d_2)·(g_0+g_1+g_2)/2
    m_3 = (d_2 − d_1)·(g_0−g_1+g_2)/2
    m_4 = (−d_1 + d_3)·g_2

— four multiplications, each one (a combination of data)·(a combination of filter). And the inverse transform A^T recombines them:

    A^T = [ 1  1  1  0 ]
          [ 0  1 -1  1 ]

so, with m_4 = (−d_1+d_3)g_2, y_0 = m_1 + m_2 + m_3 and y_1 = m_2 − m_3 + m_4. This is the same algorithm as the handwritten convention m_4' = (d_1−d_3)g_2, y_1 = m_2 − m_3 − m_4'; the minus sign can live in either the data row or the inverse row. Let me verify by expanding, because a sign error here would be silent. y_0 = m_1+m_2+m_3 = (d_0−d_2)g_0 + (d_1+d_2)(g_0+g_1+g_2)/2 + (d_2−d_1)(g_0−g_1+g_2)/2. Collect the g_0 terms: (d_0−d_2)g_0 + [(d_1+d_2)+(d_2−d_1)]g_0/2 = (d_0−d_2)g_0 + (2d_2)g_0/2 = d_0 g_0. The g_1 terms: [(d_1+d_2) − (d_2−d_1)]g_1/2 = (2d_1)g_1/2 = d_1 g_1. The g_2 terms: [(d_1+d_2)+(d_2−d_1)]g_2/2 = d_2 g_2. So y_0 = d_0 g_0 + d_1 g_1 + d_2 g_2. For the second output, m_2−m_3 gives d_1g_0+d_2g_1+d_1g_2, and adding m_4 = (−d_1+d_3)g_2 leaves d_1g_0+d_2g_1+d_3g_2. It checks. Four multiplications, and the rest is: four additions on the data, three additions plus two constant multiplies on the filter (and g_0+g_2 can be shared), four additions to reduce the products. The whole thing is

    Y = A^T [ (G g) ⊙ (B^T d) ],     ⊙ elementwise.

That matrix form is the object to hold onto. B^T is the data transform, G the filter transform, ⊙ the four real multiplies, A^T the inverse transform.

Now, this is one dimension and the win is 6→4, a factor 1.5. My problem is 2-D, and that's where the leverage compounds. Naively a 2-D minimal algorithm sounds like it needs its own derivation, but it doesn't — a 1-D algorithm *nests* with itself. Think of the 2-D tile as a 1-D filtering along rows whose "scalars" are themselves columns being filtered along the other axis. Applying the row transform and the column transform separately: the 2-D minimal algorithm for an m×m output from an r×r filter is

    Y = A^T [ (G g G^T) ⊙ (B^T d B) ] A,

where g is the r×r filter, d is the (m+r−1)×(m+r−1) tile, GgG^T transforms the filter in both dimensions, B^T d B transforms the tile in both dimensions, ⊙ is the elementwise product over the (m+r−1)×(m+r−1) grid, and A^T(·)A inverts in both dimensions. The multiplication count multiplies too: (m+r−1)·(m+r−1) products. For F(2x2,3x3): 4×4 = 16 multiplications, where the direct 2-D convolution uses 2·2·3·3 = 36. That's 36/16 = 2.25× fewer multiplications. Suddenly the 1.5× per-axis win has squared into 2.25×. *That's* the leverage — the savings compound across the two spatial dimensions, exactly the way recursion compounds Karatsuba.

And it keeps paying if I make m bigger. F(4,3): four outputs, three taps, m+r−1 = 6 points, 6 multiplies versus direct 12. Nested, F(4x4,3x3) uses 6×6 = 36 multiplications against direct 4·4·3·3 = 144 — a 4× reduction. The same 3x3 filter, four times fewer multiplications. Let me actually construct F(4,3) and not wave at it, because the points now need a real choice and the transform entries start to grow.

For F(4,3), the dual linear-convolution product has degree m+r−2 = 5, so six coefficients, so I need six points. I'll take {0, 1, −1, 2, −2} and the point at infinity. Five finite points plus infinity. Now the cleanest way to organize the interpolation when there are this many points is the Chinese-Remainder / modulus view, which is just evaluation–interpolation written algebraically. Pick the modulus polynomial whose roots are my finite points: p(x) = x(x−1)(x+1)(x−2)(x+2), degree 5, and the sixth "root" is the point at infinity handling the leading coefficient. Computing y(x) = g(x)d(x) mod p(x) plus the leading term recovers all six coefficients of the length-4 by length-3 product. On the short data side the finite residues are d(0)=d_0, d(1)=d_0+d_1+d_2+d_3, d(−1)=d_0−d_1+d_2−d_3, d(2)=d_0+2d_1+4d_2+8d_3, and d(−2)=d_0−2d_1+4d_2−8d_3. Same for g at the five points, giving g_0; g_0+g_1+g_2; g_0−g_1+g_2; g_0+2g_1+4g_2; g_0−2g_1+4g_2. Six pointwise products, including the leading-coefficient product, then interpolate. Transposing the CRT interpolation into filtering folds those short-data evaluations into rows over the six-sample filtering tile. Carrying the CRT interpolation through — solving for the per-root inverse factors, the cofactors of each p^{(i)} = p(x)/(x−x_i), and dividing by the Vandermonde difference products — produces the constants, and they're the reason this is getting heavier: the inverse factors come out as 1/4, −1/6, 1/24, 1/12, and so on. Concretely the transforms come out

    B^T = [ 4  0  -5   0   1  0 ]      G = [ 1/4     0      0   ]
          [ 0 -4  -4   1   1  0 ]          [ -1/6  -1/6   -1/6 ]
          [ 0  4  -4  -1   1  0 ]          [ -1/6   1/6   -1/6 ]
          [ 0 -2  -1   2   1  0 ]          [ 1/24   1/12   1/6 ]
          [ 0  2  -1  -2   1  0 ]          [ 1/24  -1/12   1/6 ]
          [ 0  4   0  -5   0  1 ]          [  0      0      1  ]

    A^T = [ 1  1   1   1   1  0 ]
          [ 0  1  -1   2  -2  0 ]
          [ 0  1   1   4   4  0 ]
          [ 0  1  -1   8  -8  1 ]

and the finite columns of A^T are the powers of my points {0,1,−1,2,−2}, with the infinity column contributing only to the highest recovered output row. That's the Vandermonde structure showing through: interpolation is reading off polynomial values at powers of the points. Six multiplies, and the transforms cost a handful of adds and constant multiplies per tile.

Here's the thing I have to stare at, though, because it tells me where to stop. The constants are no longer just ±1 and 1/2; they're 1/24, 5, 8. As I push m larger and pick more spread-out points, two bad things happen together. First, the number of additions and constant multiplications in the transforms grows roughly quadratically with the tile size — so the "cheap" overhead stops being cheap; past some tile size the transform arithmetic overwhelms the multiplications I saved. Second, and more dangerous, the *magnitudes* of the transform entries grow with the points (the 8s and the 1/24s), and large-magnitude transform matrices mean cancellation and lost precision — the transforms literally can't be computed accurately for large tiles. So this is not "make m as large as possible." F(2x2,3x3) at 2.25× has tiny ±1, ±1/2 transforms; F(4x4,3x3) at 4× has manageable but growing constants; if I push to F(6x6,3x3) for a 5.06× count I'm into entries like 21/4 and 1/90 and the accuracy is visibly eroding. The sweet spot is small tiles — F(2,3) and F(4,3) — where one-multiply-per-input is real *and* the transforms are cheap and accurate. The one thing that makes even F(4,3) tolerable in a convnet is that convnets are unusually forgiving of low precision (they already train in fp16), so I can spend a little accuracy to buy multiplications. But I will not chase the asymptotic multiplication count off a cliff; the tile stays small.

So I have a cheap, accurate small-convolution primitive. Now the real question for a *network*: does this actually make the layer fast, or does the overhead of transforming every tile of every channel eat the win? This is where I almost talk myself out of it. Per output tile I now do: transform the filter (GgG^T), transform the data tile (B^T d B), 36 elementwise multiplies, inverse transform. The filter transform I can hoist — a filter is reused across every image tile, so transform it once. But the data transform and especially the inverse transform happen per tile, and there are a *lot* of tiles. If I pay a full inverse transform per channel per tile, the overhead could swamp the 4× multiply savings.

Let me look at the layer sum and see if the structure rescues me. For one image, one filter, one output-tile position, the layer sums the small convolutions over channels:

    Y = Σ_c A^T [ (G g_{k,c} G^T) ⊙ (B^T d_{c} B) ] A.

The inverse transform A^T(·)A is linear, so I can pull it *outside* the channel sum:

    Y = A^T [ Σ_c (G g_{k,c} G^T) ⊙ (B^T d_{c} B) ] A.

That's the first rescue: the inverse transform is applied once per output tile, not once per channel — its cost amortizes over all C channels. Write U_{k,c} = G g_{k,c} G^T and V_{c,b} = B^T d_{c,b} B for tile index b, and look at the inner accumulation

    M_{k,b} = Σ_c U_{k,c} ⊙ V_{c,b}.

The ⊙ is elementwise over the 6×6 transform grid, so this decouples coordinate by coordinate. Fix a transform coordinate (ξ,ν) and write out that one scalar:

    M^{(ξ,ν)}_{k,b} = Σ_c U^{(ξ,ν)}_{k,c} · V^{(ξ,ν)}_{c,b}.

Stare at that. Index k runs over filters, b over (image, tile) positions, c over channels, and it's a sum over c of (something indexed k,c) times (something indexed c,b). That is *exactly* a matrix multiplication: M^{(ξ,ν)} = U^{(ξ,ν)} V^{(ξ,ν)}, with U^{(ξ,ν)} a K×C matrix (filters by channels) and V^{(ξ,ν)} a C×P matrix (channels by all tiles). One matrix multiply per transform coordinate, (m+r−1)² of them — 36 GEMMs for F(4x4,3x3), 16 for F(2x2,3x3).

That's the move that makes the whole thing practical, and it lands precisely where the FFT failed me. The elementwise stage isn't a scattered pile of independent tiny multiplies — collected across all filters and all tiles, it's a batch of dense matrix multiplications, the one operation that runs near peak on a GPU. And crucially, the dimension that's large is P, the number of tiles per channel, which is huge — H·W/m² per image, large even at batch size N=1. So the GEMMs are well-shaped even for a *single image*, which is exactly the small-batch regime where FFT's large tiles left too few tiles to fill a matrix. Real arithmetic, one multiply per input, tiny tiles, and the pointwise stage reorganizes into dense GEMM that doesn't care that the batch is small.

So the algorithm for a whole layer falls out. Transform every filter once: u = G g_{k,c} G^T, and scatter its entries across the (ξ,ν) matrices so that U^{(ξ,ν)}_{k,c} = u_{ξ,ν}. Transform every data tile: v = B^T d_{c,b} B, scatter into V^{(ξ,ν)}_{c,b} = v_{ξ,ν}. For each of the (m+r−1)² coordinates, do the GEMM M^{(ξ,ν)} = U^{(ξ,ν)} V^{(ξ,ν)}. Then for each filter k and tile b, gather the (m+r−1)² values M^{(ξ,ν)}_{k,b} back into a small tile m and apply the inverse transform Y_{k,b} = A^T m A. The tiles overlap by r−1 and tile the image with P = N⌈H/m⌉⌈W/m⌉ positions.

One loose end I should close, because training needs it: the backward pass. The gradient with respect to the layer's inputs is a convolution of the next layer's error with the flipped filter — same shape as forward propagation, so the same F(m,r) algorithm computes it. The gradient with respect to the *weights* is different: it's a convolution of the inputs with the backpropagated errors, producing only an r×r result per filter/channel, i.e. F(R×S, H×W) — and H×W is far too large to be a tile for my fast algorithm (large tiles are exactly what I rejected). So I don't run it as one giant convolution; I decompose it into a direct sum of *small* convolutions. Use F(3x3, 2x2): now the roles flip, the filter is the 2×2 piece and the output is 3×3, the 4×4 tiles overlap by 2 in each dimension, and I sum the 3×3 outputs over all tiles to assemble F(3×3, H×W). That's the same trade — (3+2−1)² = 16 multiplies versus 3·3·2·2 = 36, again 2.25× — just with the output and filter sizes swapped, which I get by transposing the transform roles (the duality that turns a filtering algorithm into a convolution algorithm and back). So forward, input-gradient, and weight-gradient all reduce to small minimal-convolution tiles.

Let me also sanity-check the boundary of the idea against direct convolution, to be sure I haven't fooled myself. Set m=1: F(1,r) means one output, the tile equals the filter size, m+r−1 = r multiplies — which is just the direct dot product. So direct convolution is the m=1 corner of this same family, the minimal algorithm for F(1×1, R×S). The fast algorithms are what I get by taking m>1, paying a little transform overhead, and buying down the multiplication count from R² toward roughly 1 per output. The maximum speedup over direct is R² divided by the per-output multiplication count (m+r−1)²/m², and the whole layer cost is that multiplication term times (1 + transform-overhead terms), where the overhead terms are the data transform divided by K, the filter transform divided by P, and the inverse transform divided by C. Each overhead is small precisely when the corresponding dimension — filters, tiles, channels — is large, which for a real 3x3 layer it is. So the win survives contact with the full layer.

Now to real code. The transforms aren't something to hand-tabulate per (m,r); I want to *generate* them from the points, because that's the whole content — pick m+r−2 finite interpolation points plus infinity, build the Vandermonde/Lagrange pieces, transpose the linear-convolution maps into filtering maps, and absorb the constant factors onto the filter side. Lagrange interpolation gives the rows of the inverse directly. Let me write the generator and then the numeric apply, and let me make the generator *symbolic* so the constants (the 1/24s) are exact — using floating point to derive the transforms would inject rounding error into the very matrices whose job is to be exact.

```python
import operator
from functools import reduce
from sympy import IndexedBase, Matrix, Poly, simplify, symbols, zeros

# --- build the Cook-Toom transforms from interpolation points ---------------
# For F(m, r) pick m+r-2 finite points; the (m+r-1)th is the point at infinity,
# supplied by the trailing [0,...,0,1] row that reads off the leading coefficient.

def At(a, m, n):                         # Vandermonde block: A[i,j] = a[i]**j
    return Matrix(m, n, lambda i, j: a[i] ** j)

def A(a, m, n):                          # ... plus the point-at-infinity row
    return At(a, m - 1, n).row_insert(
        m - 1, Matrix(1, n, lambda i, j: 1 if j == n - 1 else 0))

def Lx(a, n):                            # nodal polynomials prod_{k!=i}(x - a[k])
    x = symbols("x")
    return Matrix(n, 1, lambda i, j: Poly(
        reduce(operator.mul, ((x - a[k] if k != i else 1) for k in range(n)), 1
              ).expand(basic=True), x).as_expr())

def F(a, n):                             # Vandermonde difference products
    return Matrix(n, 1, lambda i, j: reduce(
        operator.mul, ((a[i] - a[k] if k != i else 1) for k in range(n)), 1))

def L(a, n):                             # Lagrange interpolation matrix
    x = symbols("x")
    lx, f = Lx(a, n), F(a, n)
    return Matrix(n, n, lambda i, j: lx[i, 0].coeff(x, j) / f[i]).T

def T(a, n):                             # reduce mod the nodal modulus (CRT)
    return Matrix(Matrix.eye(n).col_insert(
        n, Matrix(n, 1, lambda i, j: -(a[i] ** n))))

def Bt(a, n):                            # data transform  B^T = L * T
    return L(a, n) * T(a, n)

def B(a, n):
    return Bt(a, n - 1).row_insert(
        n - 1, Matrix(1, n, lambda i, j: 1 if j == n - 1 else 0))

def Fdiag(a, n):
    f = F(a, n)
    return Matrix(n, n, lambda i, j: (f[i, 0] if i == j else 0))

def FdiagPlus1(a, n):                    # difference products, padded for inf point
    f = Fdiag(a, n - 1).col_insert(n - 1, zeros(n - 1, 1))
    return f.row_insert(n - 1, Matrix(1, n, lambda i, j: (1 if j == n - 1 else 0)))

def cookToomFilter(a, n, r):
    """Transforms (AT, G, BT, f) for F(n, r), n = output size m, r = filter size.
    The constant fractions (the 1/2, 1/24, ...) are absorbed onto G, the filter
    transform, because filters are transformed far less often than data tiles."""
    alpha = n + r - 1                    # tile size m + r - 1
    f = FdiagPlus1(a, alpha)
    if f[0, 0] < 0:
        f[0, :] *= -1
    AT = A(a, alpha, n).T                # inverse transform: interpolate (Vandermonde)
    G  = (A(a, alpha, r).T * f ** (-1)).T  # filter transform: evaluate, fractions here
    BT = f * B(a, alpha).T               # data transform after transposition
    return AT, G, BT, f

def filterVerify(n, r, AT, G, BT):
    """Symbolic check: AT*((G g) . (BT d)) must equal the exact FIR outputs."""
    alpha = n + r - 1
    di, gi = IndexedBase("d"), IndexedBase("g")
    d = Matrix(alpha, 1, lambda i, j: di[i])
    g = Matrix(r, 1, lambda i, j: gi[i])
    return simplify(AT * (G * g).multiply_elementwise(BT * d))

# F(2,3): points {0, 1, -1}; the 4th point is infinity.  Gives the 4-mult algorithm.
AT, G, BT, _ = cookToomFilter((0, 1, -1), 2, 3)
assert filterVerify(2, 3, AT, G, BT) == Matrix(2, 1, lambda i, j:
    IndexedBase("d")[i]*IndexedBase("g")[0]
  + IndexedBase("d")[i+1]*IndexedBase("g")[1]
  + IndexedBase("d")[i+2]*IndexedBase("g")[2])
# F(4,3): points {0, 1, -1, 2, -2}; the 6th is infinity.  Gives the 6-mult algorithm.
AT4, G4, BT4, _ = cookToomFilter((0, 1, -1, 2, -2), 4, 3)
assert filterVerify(4, 3, AT4, G4, BT4) == Matrix(4, 1, lambda i, j:
    IndexedBase("d")[i]*IndexedBase("g")[0]
  + IndexedBase("d")[i+1]*IndexedBase("g")[1]
  + IndexedBase("d")[i+2]*IndexedBase("g")[2])
```

```python
import numpy as np

# --- apply the 2D nested algorithm and assemble a convnet layer -------------
def fast_conv_tile(d, g, AT, G, BT):
    """F(m x m, r x r) on one (alpha x alpha) tile and (r x r) filter:
       Y = A^T [ (G g G^T) (elementwise) (B^T d B) ] A."""
    U = G @ g @ G.T                      # filter transform, both dims (hoist per filter)
    V = BT @ d @ BT.T                    # data transform, both dims
    M = U * V                            # the alpha^2 pointwise multiplies
    return AT @ M @ AT.T                 # inverse transform, both dims

def conv_layer(D, g, AT, G, BT, m, r):
    """D: (N, C, H, W); g: (K, C, r, r). Reduce over channels in transform
    space so the elementwise stage becomes alpha^2 dense matrix multiplies."""
    N, C, H, W = D.shape
    K, alpha = g.shape[0], m + r - 1
    tiles = [(i, x, y)
             for i in range(N)
             for x in range(0, H - r + 1, m)
             for y in range(0, W - r + 1, m)]
    P = len(tiles)
    # filter transform U[xi,nu] : K x C ; data transform V[xi,nu] : C x P
    U = np.zeros((alpha, alpha, K, C)); V = np.zeros((alpha, alpha, C, P))
    for k in range(K):
        for c in range(C):
            U[:, :, k, c] = G @ g[k, c] @ G.T
    for c in range(C):
        for b, (i, x, y) in enumerate(tiles):
            V[:, :, c, b] = BT @ D[i, c, x:x+alpha, y:y+alpha] @ BT.T
    # one GEMM per transform coordinate; reduces over channels in transform space
    Mt = np.einsum('xykc,xycb->xykb', U, V)        # M[xi,nu] = U[xi,nu] @ V[xi,nu]
    Y = np.zeros((K, P, m, m))
    for k in range(K):
        for b in range(P):
            Y[k, b] = AT @ Mt[:, :, k, b] @ AT.T   # inverse transform, once per tile
    return Y, tiles
```

To recap the chain. The multiplication wall in a 3x3 layer comes from computing each output as a length-r dot product; but the dual length-m by length-r linear convolution is a polynomial product, and a degree-(m+r−2) product is fixed by its values at m+r−1 points. I evaluate the short data side and the filter at those points, do m+r−1 real multiplies, interpolate, and transpose the linear maps into the FIR filtering form — one multiply per input, the provable minimum. Nesting the 1-D algorithm in both spatial dimensions squares the per-axis saving into 2.25× (F(2,3)) or 4× (F(4,3)) fewer multiplies, and small tiles keep the transform constants tiny and accurate, which is why I stay at F(2,3)/F(4,3) rather than chasing the asymptotic count. Pulling the linear inverse transform outside the channel sum amortizes it over channels, and the resulting per-coordinate channel reduction is literally a matrix multiply over filters × channels × tiles — so the elementwise stage becomes a batch of dense GEMMs that run efficiently even at batch size one, which is exactly where the large-tile FFT route fell down.
