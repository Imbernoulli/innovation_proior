# Context: minimal-multiplication convolution for small filters

## Research question

A convolutional layer correlates a bank of K filters, each with C channels and spatial size
R x S, against a minibatch of N images with C channels and spatial size H x W. For a single
image i and filter k the output is

    Y_{i,k,x,y} = Σ_{c=1}^C Σ_{v=1}^R Σ_{u=1}^S D_{i,c,x+u,y+v} G_{k,c,u,v},

i.e. Y_{i,k} = Σ_c D_{i,c} ∗ G_{k,c}, a sum of 2D correlations. State-of-the-art image
recognition networks are built almost entirely from small 3 x 3 filters, stacked deep, because
deep stacks of small filters reach better accuracy with fewer weights than shallow stacks of
large filters. These layers dominate both the multi-day training cost and the inference cost,
and low-latency applications (a single image, or a small batch) make the per-call cost the
binding constraint.

The precise question: a length-r filter sliding over a signal to produce m outputs, computed
the obvious way, costs m·r scalar multiplications; in 2D an m x m output from an r x r filter
costs (m·r)² multiplications. Is that multiplication count intrinsic, or can a small
convolution be computed with strictly fewer multiplications — few enough, and with cheap
enough overhead, that it pays off on the small filters and small batches real networks use?
The constraint that makes this hard is exactly the regime: the existing fast method (FFT)
becomes efficient only with large filters and large batches, the opposite of what 3 x 3
networks present.

## Background

**Convolution is polynomial multiplication.** Encode a length-r filter as a polynomial
g(x) = g_0 + g_1 x + … + g_{r-1} x^{r-1} and a length-m signal block as
d(x) = d_0 + … + d_{m-1} x^{m-1}. The coefficients of the product y(x) = g(x)d(x) are exactly
the length-(m+r-1) linear-convolution outputs. FIR filtering F(m,r) computes m adjacent
correlation outputs from a length-(m+r-1) signal tile; it is the transposed, or dual, bilinear
problem to that length-m by length-r polynomial product. So "compute a small convolution/filter"
is "compute the coefficients of a low-degree polynomial product, then transpose the linear maps
when the desired object is filtering rather than full linear convolution."

**Evaluation–interpolation (Toom 1963, Cook 1966).** A polynomial of degree n−1 is uniquely
determined by its values at n distinct points. The product y(x) of a degree-(r−1) and a
degree-(m−1) polynomial has degree m+r−2, so m+r−1 coefficients, so it is pinned down by its
values at m+r−1 distinct points x_1,…,x_{m+r-1}. And the value of a product at a point is the
product of the values: y(x_j) = g(x_j)·d(x_j). This decouples the computation into three
stages — *evaluate* g and d at the chosen points (linear maps in the inputs), *multiply* the
m+r−1 pairs of values pointwise (the only true multiplications), *interpolate* y from its
values (a linear map, the inverse of a Vandermonde matrix). Evaluating at a point is a sum of
scaled inputs; interpolation is solving a Vandermonde system. Both are additions and fixed
constant multiplies, not data-dependent multiplications. Generalizing Karatsuba (1962), which
is the k=2 case, this is the Toom–Cook family: a product of polynomials split into k parts
takes 2k−1 pointwise multiplications instead of k². Small integer points {0, 1, −1, 2, −2, …}
keep the evaluation/interpolation matrices small and simple; a "point at infinity" supplies the
leading coefficient directly (the limit of y(x)/x^{deg} as x→∞), which lets the highest-order
term be read off without adding another finite evaluation point.

**The multiplication lower bound.** Counting distinct points gives both an algorithm and a
floor: a degree-(m+r−2) product has m+r−1 coefficients, and recovering m+r−1 unknowns from
bilinear products provably needs at least m+r−1 multiplications. So the minimal number of
multiplications to compute m FIR outputs from an r-tap filter is

    μ(F(m,r)) = m + r − 1,

equal to the number of input samples accessed — one multiplication per input. This was
established for one dimension and shown to nest to multiple dimensions: the minimal 2D
algorithm for m x n outputs from an r x s filter uses (m+r−1)(n+s−1) multiplications, again the
number of inputs in the (m+r−1) x (n+s−1) input tile. The general construction (linear
convolution via the Chinese Remainder Theorem on a modulus polynomial m(x), then a duality that
turns a linear-convolution algorithm into an FIR-filtering algorithm) produces F(m,r) for any
m, r. There is a cost to spending those few multiplications: the number of additions and
constant multiplications in the transforms grows roughly quadratically with the tile size, and
the magnitudes of the transform-matrix entries grow with tile size, which erodes numeric
accuracy. So the trade is favorable only for *small* tiles.

**The matrix form.** A fast filtering algorithm can be written

    Y = A^T [ (G g) ⊙ (B^T d) ],     ⊙ = elementwise product,

where B^T is the transposed data map acting on the filtering tile, G evaluates the filter, ⊙ does
the m+r−1 pointwise multiplications, and A^T recombines the products. For 2D the same 1D
algorithm is nested with itself:

    Y = A^T [ (G g G^T) ⊙ (B^T d B) ] A,

with g an r x r filter and d an (m+r−1) x (m+r−1) tile, costing (m+r−1)² pointwise products.

**The convnet layer structure.** Each channel of each image is cut into overlapping tiles of
size (m+r−1) x (m+r−1) (overlap r−1), giving P = ⌈H/m⌉⌈W/m⌉ tiles per channel. The layer sums
the per-tile, per-channel filtered results over all C channels. Two facts about this sum will
matter for any fast scheme: the inverse transform A^T(·)A is linear and can be pulled out of
the channel sum (apply it once to the accumulated tile rather than once per channel), and at a
fixed transform coordinate (ξ,ν) the channel-sum of elementwise products is exactly an
inner product over c — i.e. a matrix multiplication across filters, channels, and tiles.

**Numerical and practical context.** The direct algorithm is backward stable. A scheme that
trades multiplications for additions and constant scalings introduces extra cancellation, so it
is expected to be somewhat less accurate, the more so as tile size and transform-entry
magnitude grow. Separately, convnets are known to tolerate low numeric precision (training and
inference work in fp16 / low-precision arithmetic), which loosens the accuracy budget a fast
filtering stage must meet.

## Baselines

**Direct convolution (im2col + GEMM).** Lower the convolution to a matrix multiply: gather each
sliding window into a column, multiply by the filter matrix. Uses m·r multiplications per
output in 1D, (m·r)² in 2D — R²·HWCK multiplies for a layer. It is the workhorse, maps onto
dense GEMM, and is backward stable, but it performs the full multiplication count; it leaves
open whether those products are necessary or merely sufficient.

**FFT-based convolution (Mathieu et al. 2013; Vasilache et al. 2014, fbfft; cuDNN).** Apply the
convolution theorem: transform tiles and filters with the FFT, multiply elementwise in the
frequency domain (cyclic convolution, via overlap-and-save to recover linear convolution), and
inverse-transform. The pointwise stage is again one multiply per input, but on *complex*
numbers — 4 real multiplies per complex multiply naively, reduced toward ~1.5 with Hermitian
symmetry of real-input DFTs plus a 3-multiply complex-product trick. Its gap: the transform
overhead and complex arithmetic only amortize at *large* tile sizes (tile 16–64 to reach the
multiplication-count of a size-6 real minimal-filtering tile), large tiles waste computation on
images whose size is not near a multiple of the tile, the transformed filters and data balloon memory
(a 3 x 3 filter expands to a 64 x 64 = 4096-unit tile; workspaces of gigabytes), and small
batches yield too few tiles to make the pointwise GEMMs efficient. Exactly the small-filter,
small-batch regime where it is weakest.

**Strassen recursion on the layer matmul (Cong & Xiao 2014).** Treat the layer as a matrix
multiplication over filters × channels × tiles and apply Strassen's 7-multiply 2 x 2 identity
to cut the number of sub-convolutions, an 8/7 reduction per recursion. It shrinks all three
matrix dimensions each step, so it degrades the GEMM efficiency, and 8/7 ≈ 1.14 per recursion
is a modest, slowly-compounding win. It is orthogonal to exploiting the algebraic structure of
the convolution itself, and that work noted that further arithmetic-complexity techniques
ought to apply.

## Evaluation settings

The primary yardstick is **arithmetic operation count** as a function of the layer dimensions:
the number of true multiplications in the pointwise stage, plus the additions and constant
multiplications in the data, filter, and inverse transforms, normalized per output. Correctness
is checked symbolically — the transforms must reproduce the exact convolution as a polynomial
identity in indeterminate inputs — and numerically against a direct convolution with a
high-precision accumulator on random data and filters (uniform on [−1,1]), measuring maximum
absolute element error, in both fp32 and fp16. For a deployed kernel the relevant secondary
measures are achieved throughput against a tuned direct/FFT baseline across batch sizes
(including N=1), and workspace memory. The natural benchmark layers are the 3 x 3 convolution
stack of a deep image-recognition network across its spatial resolutions and channel counts.

## Code framework

The existing ingredients are a direct convolution (ground truth) and a Lagrange/Vandermonde
interpolation toolkit for working with polynomials symbolically. The unresolved slot is *how to
compute a small convolution with fewer multiplications* — what linear maps to apply to the data
and filter, how to combine the pointwise products, and how to assemble a whole convnet layer
out of that primitive.

```python
import numpy as np
from sympy import Matrix, symbols

def direct_conv2d(D, G):
    """Ground-truth 2D correlation of an (alpha x alpha) tile with an (r x r) filter,
    producing an (m x m) output for checking any faster scheme."""
    r = G.shape[0]
    m = D.shape[0] - r + 1
    Y = np.zeros((m, m))
    for x in range(m):
        for y in range(m):
            Y[x, y] = np.sum(D[x:x+r, y:y+r] * G)
    return Y

def vandermonde(points, n):
    """Evaluation matrix whose (i, j) entry is points[i] ** j: maps the n coefficients
    of a polynomial to its values at the given points. Its inverse interpolates."""
    return Matrix(len(points), n, lambda i, j: points[i] ** j)

def small_conv_transforms(m, r):
    """
    Return linear maps that turn an F(m, r) small convolution into a stage of independent
    pointwise multiplications. A fast algorithm should take the form
        Y = A^T [ (G g) (elementwise) (B^T d) ]
    where B^T acts on the (m+r-1)-sample data tile, G acts on the r-tap filter, and A^T
    recombines the m+r-1 pointwise products into the m outputs.

    TODO: choose where the work goes so that the only data-dependent multiplications are
    the m+r-1 pointwise products, and B^T, G, A^T contain only additions and fixed
    constants. Return (AT, G, BT, normalization).
    """
    pass  # TODO

def fast_conv2d(D, G, AT, Gm, BT):
    """
    Compute an (m x m) output tile from an (alpha x alpha) data tile and an (r x r) filter
    by nesting the 1D transforms in both spatial dimensions.

    TODO: apply the filter transform to G, the data transform to D, multiply elementwise,
    and apply the inverse transform -- the 2D analogue of the 1D form above.
    """
    pass  # TODO

def conv_layer(D, G, AT, Gm, BT):
    """
    Whole convnet layer: D is (N, C, H, W), G is (K, C, r, r). Cut each channel into
    overlapping (alpha x alpha) tiles, run the small-convolution primitive per tile/filter,
    and sum over channels.

    TODO: arrange the per-tile work so the channel reduction at each fixed transform
    coordinate becomes a single dense matrix multiply over (filters, channels, tiles),
    and the inverse transform is applied once per output tile rather than once per channel.
    """
    pass  # TODO
```
