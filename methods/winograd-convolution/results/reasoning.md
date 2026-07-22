OK, let me think this through from scratch. I have a deep network that is almost entirely 3×3 convolution layers, and the thing eating all the compute is exactly those layers. A layer correlates K filters of C channels and size 3×3 against N images of C channels and size H×W. Count the multiplications the obvious way: every output pixel, every filter, every channel costs 3×3 = 9 multiplies, so the whole layer is 9·N·H·W·C·K multiplies. That count is the wall. I want to make it smaller — and not by approximating, by computing the *same* convolution with fewer multiplications.

My gut says 9 per output window is forced. The filter has 9 taps, each output is a dot product of 9 taps with 9 image values, and a dot product of length 9 is 9 multiplies. How could I get that inner product without forming the 9 products? But I've been burned by that intuition. "Every product is needed" is a statement about *one particular way* of computing — the sliding dot-product way. The outputs are bilinear forms in the image values and the filter values; nothing forces me to realize a bilinear form by literally forming each monomial and summing. I could take linear combinations of the inputs, multiply *those*, and arrange for the cross-terms I don't want to cancel in the recombination. That's the loophole, and there's precedent: complex multiplication (a+ib)(c+id) looks like 4 real multiplies, but ac−bd = a(c−d)+d(a−b) and ad+bc = b(c+d)+d(a−b) does it in 3, paying a few extra adds. So the move is real: trade multiplies for adds. And on this hardware that trade is enormously favorable — a multiply is the expensive operation, an add is nearly free, and the layer is multiply-bound. If I can shave multiplies and only pay adds, I win even if the add count goes up.

Before I get clever, what does fast convolution already look like in this space? The FFT route. Convolution is pointwise multiply in the Fourier domain, so I transform each image channel and each filter once, multiply pointwise, inverse-transform, and reuse each transformed map across all the filter pairings — the transform cost amortizes when there are many feature maps. It genuinely helps for *large* filters. But stare at what it costs me here. The pointwise products are between *complex* numbers, and a complex multiply is 4 real multiplies; with a fast complex-multiply trick I get it to 3, and exploiting that a real signal's transform is Hermitian-symmetric I roughly halve the count, but I'm still paying something like 1.5–4 real multiplies *per input*, never below 1. And to amortize a 2D FFT I need a big tile — only m×n of an (m+r-1)×(n+s-1) cyclic convolution is valid, so I overlap-and-save, and to make the transform overhead worthwhile the tile wants to be 16, 32, 64 on a side. A 64×64 tile turns my 3×3 = 9-element filter into 64×64 = 4096 stored units, demands a fat memory workspace, and only generates enough tiles to fill an efficient pointwise matrix multiply when the batch is large. My pain is *small* filters at *small* batch. FFT is built for the opposite corner. I even see the symptom in practice: a library that flips to FFT at moderate batch sizes collapses to under 2 TFLOPS, because its pointwise stage is starved of tiles. So FFT is the wrong transform for this regime. I want a transform whose pointwise stage costs exactly *one real multiply per input* and whose tile is *small*.

That phrasing — "one real multiply per input" — is suggestive. Is there a known minimum number of multiplications to filter? Let me set up the smallest nontrivial 1D case and just try to beat the direct count by hand. Compute 2 outputs of a 3-tap filter; call it F(2,3). Direct: each output is 3 multiplies, so 6 total. The two outputs are

  y0 = d0·g0 + d1·g1 + d2·g2
  y1 = d1·g0 + d2·g1 + d3·g2,

over the data window d0,d1,d2,d3 and filter g0,g1,g2. Six multiplies the naive way. Can I do it in fewer? Let me think about *why* there might be a floor and where it is. The window F(2,3) touches 4 distinct data values d0..d3. The intuition "one multiply per input" would say 4. Let me see if 4 is even reachable before I trust it.

The structural fact I need is that linear convolution *is* polynomial multiplication. Write the filter as a polynomial g(x) = g0 + g1·x + g2·x² and the data as d(x) = d0 + d1·x + d2·x² + d3·x³. The coefficients of the product g(x)·d(x) are exactly the linear convolution of the two sequences. The product has degree 2+3 = 5, so 6 coefficients — and a degree-5 polynomial is completely pinned down by its values at 6 distinct points. So instead of multiplying the polynomials coefficient-by-coefficient (the O(LN) way), I can *evaluate* g and d at 6 chosen points, multiply the values pointwise — 6 scalar products — and then *interpolate* the product polynomial back from those 6 values. That's Lagrange interpolation, and it's the Cook–Toom idea: the only real multiplies are the 6 pointwise products s(β_i) = g(β_i)·d(β_i); everything else — evaluating g and d at the points, and interpolating back — is additions and multiplications by *constants that depend only on the chosen points*, not on the data. If I pick the points to be small integers like 0, 1, −1, 2, −2, those evaluations and the interpolation are cheap adds and scalings by small numbers.

But wait — 6 points gives me 6 multiplies, which is just the direct count. That re-derives 6, not beats it. The reason is that I asked for the *full* linear convolution, all 6 output coefficients. I don't need all 6. Filtering F(2,3) only wants y0 and y1 — the two "valid" outputs where the 3-tap filter sits fully inside the data. So I'm computing more than I need. Let me reduce the degree of the problem to match what I actually want.

Two outputs means I want a length-2 result. The cleanest way to see the real minimum: the *modified* Cook–Toom move. The product s(x) has top coefficient g2·d3, which I can compute with a single dedicated multiply and subtract off, leaving a lower-degree polynomial to interpolate. Equivalently: take the point "at infinity," which just reads off the highest-order coefficient directly. Each point I spend buys one multiply; the point at infinity is one of them, contributing the g2·d3-type product, and it costs no growing constants. So the count is *the number of points I truly need*. Counting: the F(2,3) window touches 4 distinct data values d0..d3, so the heuristic "one multiply per input it touches" says 4 — three finite points plus the one at infinity. I don't actually believe that number yet; "one per input" is the same intuition that told me 9 was forced, and I was wrong about that, so I want to see 4 interpolation points produce y0 and y1 *exactly* before I treat 4 as the floor. The recipe to test: choose 4 points, three finite and one at infinity, and grind the algebra all the way to the outputs.

Let me actually pick points and grind it out, because I don't trust an algorithm I haven't watched produce the right numbers. Take β = {0, 1, −1, ∞}. The finite-point evaluations:

  g(0)  = g0,                       d(0)  = d0
  g(1)  = g0+g1+g2,                 d(1)  = d0+d1+d2+d3
  g(−1) = g0−g1+g2,                 d(−1) = d0−d1+d2−d3
  ∞: read off top coefficients      g2 and d3.

So the four pointwise products — the only real multiplies — are
  m1 = g(0)·d(0)   = g0·(d0)            ... but I want this organized by the *output window*, not the raw data polynomial. Let me re-derive in the form that filters cleanly.

This is where the subtlety lives. What I just set up, evaluate-multiply-interpolate, computes a *linear convolution* — a full product polynomial. Filtering is not quite that; filtering takes a fixed filter and slides it to produce m valid outputs, and as a linear map from the data it is the *transpose* of the linear-convolution map. So the clean way to get the filtering algorithm is: build the minimal *linear convolution* algorithm and then transpose it. Concretely, suppose I've written linear convolution as a factorization

  s = C · (H · (D·x)),

where D is a pre-addition matrix (it forms the input combinations like d0+d1+d2+d3), H is *diagonal* (its nonzero entries are exactly the pointwise multiplies — and the count of nonzeros is the multiply count), and C is a post-addition matrix (the interpolation recombination). The convolution matrix T factors as T = C·H·D, and because H is diagonal, the number of multiplies is just the number of nonzero diagonal entries — m+r−1, by construction. Now, the matrix-exchange theorem: the *filtering* problem, being the transpose, is computed by transposing this factorization. If for linear convolution the filter side gets transform G, the data side gets A, and the recombination is B, then for filtering the data and recombination roles swap and transpose — I get the data transform Bᵀ, the same filter transform G, and the inverse transform Aᵀ. The end form is

  Y = Aᵀ [ (G·g) ⊙ (Bᵀ·d) ],

where ⊙ is elementwise multiply. The filter is transformed by G into the point-value domain; the data is transformed by Bᵀ into the same domain; I multiply elementwise — that's the m+r−1 real multiplies, one per transformed component; and Aᵀ interpolates the elementwise products back down to the m outputs. The transform matrices depend only on the chosen points, never on the data — so I build them once.

Let me write out the F(2,3) matrices from β = {0,1,−1,∞} and *check* them by hand, because if the elementwise products don't reduce to y0,y1 I've fooled myself. With the standard construction the data transform is

  Bᵀ = [[1, 0,  −1, 0],
        [0, 1,   1, 0],
        [0, −1,  1, 0],
        [0, −1,  0, 1]],

the filter transform is

  G = [[1,    0,    0  ],
       [1/2,  1/2,  1/2],
       [1/2, −1/2,  1/2],
       [0,    0,    1  ]],

and the inverse transform is

  Aᵀ = [[1, 1,  1, 0],
        [0, 1, −1, 1]].

Read off the four transformed-filter components Gg = (g0, (g0+g1+g2)/2, (g0−g1+g2)/2, g2) and the four transformed-data components Bᵀd = (d0−d2, d1+d2, −d1+d2, d3−d1). The four products are

  m1 = (d0−d2)·g0
  m2 = (d1+d2)·(g0+g1+g2)/2
  m3 = (d2−d1)·(g0−g1+g2)/2
  m4 = (d3−d1)·g2.

Four multiplies — that's the bound m+r−1 = 4, beating the direct 6. Now interpolate with Aᵀ: y0 = m1+m2+m3 and y1 = m2−m3+m4. Let me actually expand y0 to be sure nothing leaks. m1 = d0·g0 − d2·g0. m2+m3 = (d1+d2)(g0+g1+g2)/2 + (d2−d1)(g0−g1+g2)/2. Group: the (g0+g2)/2 part gives (d1+d2 + d2−d1)/2·(g0+g2) = d2·(g0+g2); the g1/2 part gives (d1+d2 − (d2−d1))/2·g1 = d1·g1. So m2+m3 = d2·g0 + d2·g2 + d1·g1. Add m1: y0 = d0·g0 − d2·g0 + d2·g0 + d2·g2 + d1·g1 = d0·g0 + d1·g1 + d2·g2. Exactly y0. And y1 = m2 − m3 + m4: m2−m3 = (d1+d2)(g0+g1+g2)/2 − (d2−d1)(g0−g1+g2)/2; the (g0+g2)/2 part is (d1+d2 − d2+d1)/2·(g0+g2) = d1·(g0+g2), the g1/2 part is (d1+d2 + d2−d1)/2·g1 = d2·g1; so m2−m3 = d1·g0 + d1·g2 + d2·g1, plus m4 = d3·g2 − d1·g2 gives y1 = d1·g0 + d2·g1 + d3·g2. Exactly y1. It works. I've computed a 2-output, 3-tap filter in 4 multiplies, paying the price in additions: 4 additions on the data, a few additions and constant-scalings on the filter (and g0+g2 is shared, so it's cheap), and 4 additions to recombine. And every one of those extra ops is an add, which is the cheap operation. That's the whole bargain, and on multiply-bound hardware it's a great one.

One more thing about the filter side I shouldn't gloss over: G·g depends only on the filter, not the data. In a convnet a filter is reused across the entire image (all its tiles) and across the whole batch, so I can compute the filter transform *once* per filter and amortize it — exactly the way precomputed FIR coefficients are stored once and reused. That makes the data-side transform the only per-tile transform cost that scales with the image.

Now lift it to 2D, because my filters are 3×3, not 3×1. The beautiful fact: minimal 1D algorithms *nest*. If I have a minimal F(m,r) along rows and a minimal F(n,s) along columns, I get a minimal 2D algorithm F(m×n, r×s) by applying one along each axis, and the multiply count multiplies: (m+r−1)(n+s−1). The 2D form drops right out of the 1D form by applying the transforms on both sides of the tile. For a square case, with g now an r×r filter and d an (m+r−1)×(m+r−1) image tile,

  Y = Aᵀ [ (G g Gᵀ) ⊙ (Bᵀ d B) ] A.

Filter transform G g Gᵀ takes the r×r filter to an α×α array (α = m+r−1); data transform Bᵀ d B takes the α×α tile to α×α; elementwise multiply — that's α² real multiplies; and Aᵀ[·]A interpolates back to the m×m output tile. For F(2×2, 3×3): α = 4, so 4×4 = 16 multiplies, versus direct 2·2·3·3 = 36. That's 36/16 = 2.25× fewer multiplies. The tile is only 4×4 — tiny, the opposite of the FFT's 64×64. The transforms are cheap: counting floating-point instructions, the 2D data transform is 32 additions, the filter transform 28 instructions, the inverse transform 24 additions. All adds and small scalings; the heavy multiplies are gone.

Now I have to make this an actual *layer*, not a single tile, and this is where the design has to be careful or I'll lose the win to overhead. A channel of size H×W is cut into α×α = 4×4 tiles that overlap by r−1 = 2 pixels (because neighboring 2-output windows share input columns), giving ⌈H/m⌉⌈W/m⌉ tiles per channel. For each (filter k, channel c, tile b) I'd compute Aᵀ[(G g_{k,c} Gᵀ) ⊙ (Bᵀ d_{c,b} B)]A and sum over channels c. If I apply Aᵀ…A per (k,c,b) and then sum over c, I pay the inverse transform C times per output tile. That's wasteful. But the sum over channels is linear and the inverse transform is linear, so I can push the channel-sum *inside*, into the transform domain, and apply Aᵀ…A only once per output tile:

  Y_{i,k,tile} = Aᵀ [ Σ_c (G g_{k,c} Gᵀ) ⊙ (Bᵀ d_{c,i,tile} B) ] A.

The inverse transform is now amortized over all C channels. Look hard at that inner sum. Write U_{k,c} = G g_{k,c} Gᵀ (filter, transformed) and V_{c,b} = Bᵀ d_{c,b} B (data tile, transformed), and pull out a single transform-domain coordinate (ξ,ν) ∈ {0..α−1}². Then

  M_{k,b}^{(ξ,ν)} = Σ_c U_{k,c}^{(ξ,ν)} · V_{c,b}^{(ξ,ν)}.

Stare at that for a second. For a *fixed* coordinate (ξ,ν), this is summing over c a product of something indexed by (k,c) and something indexed by (c,b) — that is the contraction pattern of a matrix multiply: M^{(ξ,ν)} = U^{(ξ,ν)} · V^{(ξ,ν)}, where U^{(ξ,ν)} is K×C (filters × channels) and V^{(ξ,ν)} is C×P (channels × tiles), giving K×P. So the entire heavy stage of the layer comes out as α² independent dense matrix multiplies — 16 of them for F(2×2,3×3). That is a better landing than I expected: the work is GEMMs, which are efficient on GPUs precisely because of their high arithmetic intensity. And look at which dimension does the reduction — it's C (channels), not N. So the GEMMs stay full-width even when the *batch* is tiny: a single image already supplies a C-deep reduction. FFT needed a big batch to fill its pointwise stage; here the matrix multiply is fed by channels, so batch 1 is already fine. That removes the exact failure I saw FFT hit in the small-batch regime.

So the layer is: transform every filter (once, U = G g Gᵀ, scatter into the α² matrices U^{(ξ,ν)}); transform every data tile (V = Bᵀ d B, scatter into V^{(ξ,ν)}); do α² batched matrix multiplies M^{(ξ,ν)} = U^{(ξ,ν)} V^{(ξ,ν)}; gather the results and apply the single inverse transform Y = Aᵀ M A per output tile. The multiply stage is α² GEMMs; the transforms are the cheap add-heavy boundary work.

Let me sanity-check that this analysis recovers the right limits and tells me how far to push. Per output, the multiplication complexity normalizes to α′ = (m+r−1)²/m². When m=1 (no fast filtering, a 1×1 output tile), α′ = R², the direct count — good, direct convolution is the degenerate F(1×1,R×S). For F(2×2,3×3), α′ = 16/4 = 4 multiplies-per-output, versus direct's R² = 9, the 2.25× again. The maximum possible speedup of a fast algorithm over direct is R²/α′. To make α′ small I want m large relative to r. So why not just crank m way up — F(6×6,3×3), F(10×10,3×3) — and drive α′ toward 1?

Then I hit the wall. Two costs grow with the tile size m+r−1, and they fight the multiply savings. First, the number of additions and constant-multiplications in the transforms grows *quadratically* with the tile size, so for a large tile the transform work overwhelms whatever I saved on multiplies. I can see it numerically: as I tabulate the normalized data/filter/inverse transform complexities β′, γ′, δ′ against tile size, they climb fast while α′ only inches toward 1. Second — and this is the sharper limit — the *magnitudes of the transform-matrix entries grow* with tile size. Let me see why from the construction: as I add interpolation points 2, −2, 3, −3, … the Vandermonde/Lagrange entries involve powers and reciprocals of those points (1/24, 1/120, and worse), so the transform matrices acquire large and finely-scaled coefficients. Multiplying data by such matrices in finite precision loses bits; for a large enough tile the transforms simply can't be computed accurately. So there's a sweet spot, not a limit at 1: small tiles. F(2×2,3×3) (tile 4) and F(4×4,3×3) (tile 6) are the practical operating points.

Let me grind F(4,3) so the construction is honest at a less trivial size — the 1/24-style coefficients are where the accuracy worry becomes concrete, and I want to see them appear from the CRT machinery rather than take them on faith. F(4,3): 4 outputs, 3 taps, α = 4+3−1 = 6, so I need 6 interpolation points; take {0,1,−1,2,−2,∞}. Going the polynomial-CRT route: g(x) = g2x²+g1x+g0, d(x) = d3x³+d2x²+d1x+d0, the product s(x) = g(x)d(x) has degree 5. The five finite factors are m^(0)=x, m^(1)=x−1, m^(2)=x+1, m^(3)=x−2, and m^(4)=x+2; the sixth residue is the ∞ slot that carries the degree-5 top coefficient. Reduce g and d modulo each finite factor — that's just evaluating at the point: g mod (x−β) = g(β). So g^(0)=g0, g^(1)=g0+g1+g2, g^(2)=g0−g1+g2, g^(3)=g0+2g1+4g2, g^(4)=g0−2g1+4g2, g^(5)=g2 (the ∞ residue, top coefficient). Likewise the data residues d^(i) = d(β_i). Stacking the d^(i) as a matrix in the data values gives the rows [1 0 0 0], [1 1 1 1], [1 −1 1 −1], [1 2 4 8], [1 −2 4 −8], [0 0 0 1] — the Vandermonde rows at the chosen points (last row = the ∞ row reading off the top coefficient).

Now CRT reconstruction. Define M^(i)(x) = m(x)/m^(i)(x), the product of all the *other* finite factors. To recombine the residues into s(x) I need, per point, the cofactor N^(i)(x) ≡ (M^(i))^{-1} mod m^(i), found by solving n^(i)(x)m^(i)(x) + N^(i)(x)M^(i)(x) = 1. Since each finite m^(i) is linear, N^(i) is just the scalar 1/M^(i)(β_i) = 1/Π_{j≠i}(β_i−β_j). Compute those denominators: at β=0 the product of (0−1)(0+1)(0−2)(0+2) = (−1)(1)(−2)(2) = 4, so N^(0) = 1/4; at β=1 the product (1)(2)(−1)(3) = −6, so N^(1) = −1/6; at β=−1 the product (−1)(−2)(−3)(1) = −6, so N^(2) = −1/6; at β=2 the product (2)(1)(3)(4) = 24, so N^(3) = 1/24; at β=−2 the product (−2)(−3)(−1)(−4) = 24, so N^(4) = 1/24. There it is — absolute denominators 4, 6, and 24, with the signs that fold into the filter transform G. Setting row G_i to the residue g^(i) scaled by N^(i) reproduces

  G = [[1/4,    0,     0  ],
       [−1/6,  −1/6,  −1/6],
       [−1/6,   1/6,  −1/6],
       [1/24,   1/12,  1/6],
       [1/24,  −1/12,  1/6],
       [0,      0,     1  ]],

and assembling the cofactor columns into the inverse/data transforms gives

  Bᵀ = [[4, 0, −5, 0, 1, 0],
        [0, −4, −4, 1, 1, 0],
        [0, 4, −4, −1, 1, 0],
        [0, −2, −1, 2, 1, 0],
        [0, 2, −1, −2, 1, 0],
        [0, 4, 0, −5, 0, 1]],

  Aᵀ = [[1, 1, 1, 1, 1, 0],
        [0, 1, −1, 2, −2, 0],
        [0, 1, 1, 4, 4, 0],
        [0, 1, −1, 8, −8, 1]].

That's 6 multiplies for F(4,3). Now the 1/24 in G and the entries like −5 in Bᵀ and ±8 in Aᵀ are exactly the growing magnitudes I worried about — at this tile size still fine, but I can feel them growing, and that's the accuracy ceiling on large tiles made concrete. Nest F(4,3) with itself: F(4×4,3×3) uses 6×6 = 36 multiplies versus direct 4·4·3·3 = 144 — a 4× reduction, tile size 6.

So which operating point? F(2×2,3×3) gives 2.25× with a tiny 4×4 tile and the smallest, best-conditioned transforms; its transform entries are only 0, ±1, and ±1/2, so the finite-precision risk is low. F(4×4,3×3) gives the full 4× but with the 1/24-scale entries, so I should expect more roundoff. Here the quantization evidence pays off: convnets are known to train and infer with surprisingly little numeric precision, so I can spend some accuracy for the bigger speedup if the measured error stays within the network's tolerance. So both are plausible operating points; the small tile is the safe default and the larger tile is available when the precision budget allows. I will *not* chase F(6×6,3×3) and beyond as a default: carrying the same CRT construction out to F(6,3) already throws up coefficients like 1/720, and the transform magnitudes and add-counts make it fragile.

There's still the FFT comparison to settle quantitatively, because both algorithms have the same outer shape — transform, pointwise-multiply (a GEMM per transform-domain coordinate), inverse-transform; FFT is overlap-and-save with FFT/iFFT as the transforms. If I can't show a real gap I'm just reinventing FFT with worse transforms. The decisive difference has to be the multiply stage, so let me count it carefully and not hand-wave, because there are two different denominators floating around — per *input* and per *output* — and conflating them is exactly how I'd fool myself.

Per *input* the comparison is lopsided and almost trivial: my Winograd stage does α² real multiplies over an α² tile, so it is exactly 1 real multiply per transformed input, flat for every tile size. The FFT stage does 1 *complex* multiply per input. With Hermitian symmetry a real α×α tile's transform carries only α×(⌊α/2⌋+1) complex values; with the textbook 4-real-multiply complex product that is 4(⌊α/2⌋+1)/α² real multiplies *per input*, and with the fast 3-real-multiply product — (x0+ix1)(y0+iy1) via three real products — it is 3(⌊α/2⌋+1)/α². Let me actually evaluate that fraction rather than eyeball it. At α = 4, 6, 8, 16, 32, 64 the fast-complex figure is 2.25, 2.00, 1.875, 1.69, 1.59, 1.55, and the direct-complex figure is 3.00, 2.67, 2.50, 2.25, 2.13, 2.06. It decreases with α but it is asymptoting toward 3/2, not toward 1 — even at a ruinous 64×64 tile it is 1.55, still half again my 1. So on a per-input basis FFT can *never* reach Winograd; there is no tile size that closes the gap. Good, but this is the wrong denominator to brag about, because FFT amortizes one large transform over many valid outputs while my small tile produces few outputs per transform — the honest comparison is per *output*.

So redo it per output. Winograd F(2×2,3×3) does 16 multiplies for 4 outputs = 4.0 per output; F(4×4,3×3) does 36 for 16 = 2.25 per output. An α×α FFT tile of a 3×3 filter yields (α−2)² valid outputs (overlap-save), so its fast-complex cost per output is 3·α(⌊α/2⌋+1)/(α−2)². Evaluate that against my 2.25: α=8 gives 3.33, α=16 gives 2.20, α=32 gives 1.81, α=64 gives 1.65; with the direct 4-multiply complex product the same tiles give 4.44, 2.94, 2.42, 2.20. So FFT only *reaches* Winograd F(4×4,3×3)'s 2.25-per-output figure at tile size ≈16 with the fast complex multiply, or ≈64 with the direct one — and only by then does it cross. This is the number I half-remembered as "16 or 64," and now I can see it falls straight out of the (α−2)² output count rather than having to trust it. The cost of getting there is the punchline: a 16×16 or 64×64 tile transforms my 3×3 = 9-element filter channel into 256 or 4096 stored units, so the workspace balloons (the 64×64 case runs to gigabytes), and the tile only produces enough independent tiles to fill the pointwise GEMM when the batch is large. My F(2×2,3×3) tile is 4×4: it fuses into on-chip registers and shared memory, fills its GEMMs from the channel dimension at batch 1, and keeps the workspace in megabytes. So the gap is real and it is structural, not a transform-quality artifact: in the small-filter, small-batch corner, minimal filtering wins on multiplies-per-output at a tile two-to-ten times smaller, and simultaneously on memory and on the batch regime — and for FFT to even tie the multiply count it has to pay the memory and batch costs that defined the problem in the first place.

I should also handle training, because a layer needs gradients. The gradient with respect to the *inputs* is a convolution of the next layer's backpropagated error with the spatially-flipped filters — same shape of problem, so the same forward algorithm applies. The gradient with respect to the *weights* is trickier: it's a convolution of the layer inputs with the backpropagated errors that produces R×S outputs per (filter,channel), i.e. F(R×S, H×W) — and H×W is huge, far too large a "filter" for my small-tile algorithms, and a large tile is exactly what I just argued against. So I won't compute it as one giant convolution. Instead decompose it into a sum of small convolutions: tile the large operands and accumulate, e.g. compute it as F(3×3, 2×2) over 4×4 tiles overlapped by 2, summing the 3×3 outputs over all tiles to assemble F(3×3, H×W). F(3×3,2×2) is the transpose-flavored sibling of F(2×2,3×3) — by symmetry of m+r−1 it also has tile size 4 and the same 36/16 = 2.25× reduction, with transforms

  Bᵀ = [[1,0,−1,0],[0,1,1,0],[0,−1,1,0],[0,−1,0,1]],
  G = [[1,0],[1/2,1/2],[1/2,−1/2],[0,1]],
  Aᵀ = [[1,1,1,0],[0,1,−1,0],[0,1,1,1]].

So the same machinery covers forward, input-gradient, and weight-gradient.

And I should be disciplined about *not* over-reaching. Could I stack Strassen on top to also cut the *number* of convolutions? Each Strassen recursion gives only an 8/7 reduction while halving all three matrix dimensions — and those dimensions (K, C, P) are exactly what my α² GEMMs need to be large to run efficiently. Fast filtering already gives ≥2.25× while shrinking only the largest dimension P (it doesn't touch K or C). So Strassen on top is a net loss in this regime except possibly for layers with very large C, K, and P; I'll leave it out and keep the GEMMs fat.

Now let me make this concrete in code. The transform *generator* is the Cook–Toom construction itself — evaluate filter and data polynomials at the chosen points (with the "point at infinity" row reading off the top coefficient), divide by the interpolation denominators ∏_{k≠i}(β_i−β_k) (the 1/4, 1/6, 1/24 scalings that land in G), and transpose the linear-convolution factorization into the filtering form Aᵀ[(Gg)⊙(Bᵀd)] — and it is what produced every G/Bᵀ/Aᵀ above; I verified each by expanding Aᵀ[(Gg)⊙(Bᵀd)] symbolically (the F(2,3) expansion back to y₀=d₀g₀+d₁g₁+d₂g₂ and y₁=d₁g₀+d₂g₁+d₃g₂ is the worked check I did by hand earlier) rather than trusting transcription. With the transform triples in hand, the layer itself is just the four numeric stages — tile, transform, the α² GEMMs that came out of the channel-sum, and the single inverse transform.

So I land the F(2×2,3×3) layer as one self-contained C++17 program. It reads a convolution layer from stdin — `N C H W K`, then the N·C·H·W data values in index order [n][c][h][w], then the K·C·3·3 filter values [k][c][u][v] — and writes the valid output Y[n][k][H−2][W−2] to stdout, one row of W−2 space-separated values per line. The transform matrices are baked in as the F(2,3) triple (entries only 0, ±1, ±1/2, the best-conditioned operating point); the heavy stage is the 16 dense GEMMs whose reduction dimension is channels, so it stays full-width at batch 1; each output costs 4 real multiplies versus the direct 9, and the result matches a direct convolution to fp32 rounding.

The part I would most easily get wrong under contest pressure is the F(2×2,3×3) Winograd transform layout and the 16-coordinate GEMM scatter/gather reuse; if I were not confident I could implement that cleanly in time, I would fall back to the direct valid 3×3 convolution I already used as the verified reference.
