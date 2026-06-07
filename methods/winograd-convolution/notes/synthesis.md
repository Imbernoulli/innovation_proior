# Synthesis — Winograd / minimal-filtering fast convolution for convnets

## Sources actually read this run (all in refs/)
- `lavin_gray_fulltext.txt` — Lavin & Gray 2016 "Fast Algorithms for Convolutional Neural Networks" (arXiv:1509.09308v2), full main text. PRIMARY (CNN application).
- `lavin_gray_supplementary.txt` — same paper's supplementary appendix: full CRT derivation of F(4,3), plus F(3,3), F(6,3), F(3x3,3x3), F(6x6,3x3) transforms. PRIMARY (the worked derivation).
- `parhi_fast_convolution.txt` — Parhi, "Fast Convolution" (Chap. 8, VLSI DSP). PRIMARY theory: Cook-Toom (Lagrange interpolation), the matrix factorization T = C·H·D, the matrix-exchange/transpose technique, Modified Cook-Toom, Winograd CRT. The Toom-Cook / CRT lineage.
- `mathieu_fft.txt` — Mathieu, Henaff, LeCun 2013 (arXiv:1312.5851), FFT convolution for convnets. ANTECEDENT / baseline reacted against.
- `vasilache_fbfft_1412.7580.pdf` — Vasilache et al fbfft (refinement of FFT-conv). ANTECEDENT.
- `barabasz_error_analysis.txt` — Barabasz & Gregg 2018 (arXiv:1803.10986), error analysis of Winograd convolution. ANALYSIS (numeric accuracy of the transforms).
- code/wincnn/wincnn.py — Andrew Lavin's canonical Winograd/Cook-Toom codegen (sympy). Verified F(2,3) and F(4,3) reproduce the paper's G/B/A matrices and pass symbolic FIR check this run.

GAP: Winograd 1980 book ("Arithmetic Complexity of Computations", SIAM) is access-restricted on archive.org — no free full text. Its content (the bound µ=m+r-1, the CRT construction, the nesting theorem) is reconstructed from the two PRIMARY derivations that re-derive it: Lavin & Gray supplementary (CRT for F(4,3)) and Parhi Chap 8 (Cook-Toom + CRT + matrix exchange). Winograd's "On multiplication of polynomials modulo a polynomial" SIAM J. Comput 9(2) 1980 (ref [14], the nesting/tensor theorem) is likewise not freely available; its statement is captured in Lavin eq (4) and Parhi's iterated-convolution section.

## Research question (in-frame)
Convnets are bottlenecked by 3x3 conv layers at small batch. FFT conv is fast only for large filters / large tiles / large batch (big workspace; only m×n of the (m+r-1) cyclic outputs are valid). Direct conv does R²=9 mults per output. Want: an algorithm that does FAR fewer multiplications per output for SMALL filters, whose multiply stage is still a big dense matmul (efficient even at batch 1), with small memory.

## Key derivation chain
1. **Minimal-filtering bound.** Computing m outputs of an r-tap FIR filter = F(m,r). Lower bound µ(F(m,r)) = m+r-1 multiplications (= number of distinct data values touched = "one mult per input"). Direct uses m·r. For F(2,3): 4 vs 6.
2. **Linear convolution = polynomial product.** g(x)=Σg_i x^i (deg r-1), d(x) (deg m+r-2 for the linear-conv view), product s(x)=g(x)d(x) has m+r-1 coefficients → determined by its values at m+r-1 points. Cook-Toom: pick m+r-1 points β_i; evaluate g,d there (cheap adds + small-const mults if β∈{0,±1,±2,∞}); multiply pointwise (m+r-1 mults — the only real mults); Lagrange-interpolate back. Factorize the convolution matrix T = C·H·D (post-add · diagonal · pre-add). H diagonal ⇒ #mults = #nonzero diag = m+r-1.
3. **"Modified" Cook-Toom / the ∞ point.** Using m+r-2 finite points plus the point at infinity (handled by appending the highest-order coefficient directly — the row [0…0 1] in A) keeps it minimal while cutting adds. wincnn's A()/B() append exactly that row.
4. **Matrix-exchange / transpose to FILTERING.** Cook-Toom gives s=C·H·D·x for *linear convolution* (full output). FILTERING F(m,r) (m correlation outputs) is the transpose problem. Winograd's matrix-exchange: if linear conv is B·[(G·g)⊙(A·d)], the filtering algorithm is Aᵀ·[(G·g)⊙(Bᵀ·d)] — same G; A and B swap/transpose roles. Hence the paper's form Y = Aᵀ[(Gg)⊙(Bᵀd)]: G filter-transform (r→α), Bᵀ data-transform (α→α), elementwise, Aᵀ inverse (α→m). α=m+r-1.
5. **F(2,3) concrete (β={0,1,-1,∞}).** Bᵀ=[[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,-1,0,1]]; G=[[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]]; Aᵀ=[[1,1,1,0],[0,1,-1,-1]]. m1=(d0-d2)g0, m2=(d1+d2)(g0+g1+g2)/2, m3=(d2-d1)(g0-g1+g2)/2, m4=(d1-d3)g2. y0=m1+m2+m3, y1=m2-m3-m4. 4 mults. VERIFIED via wincnn this run.
6. **Nesting to 2D.** Minimal 1D F(m,r) nests with F(n,s) → minimal 2D F(m×n,r×s), µ=(m+r-1)(n+s-1). 2D form: Y = Aᵀ[(G g Gᵀ) ⊙ (Bᵀ d B)]A. F(2x2,3x3): 4·4=16 mults vs 2·2·3·3=36 → 2.25x. F(4x4,3x3): 6·6=36 vs 144 → 4x.
7. **Why this beats FFT for convnets.** Winograd multiply stage is always 1 *real* mult/input. FFT: 1 *complex* mult/input = 4 real (3 with fast CGEMM, ~2 with Hermitian symmetry; still >1.5). FFT needs tile ≥16-64 to amortize → huge workspace (64×64=4096-unit filter expansion), only efficient for large batch. Winograd's 4×4 tile (F(2x2,3x3)) → tiny transforms, fuse in registers, 16MB workspace.
8. **Reduce-in-transform-space = a GEMM.** Sum over C channels happens in transform space BEFORE inverse transform (amortizes Aᵀ over C). For each of the α² transform coordinates (ξ,ν): M^(ξ,ν) = U^(ξ,ν) V^(ξ,ν) is a (K×C)·(C×P) matrix multiply. The whole layer = α² independent batched GEMMs → efficient even at batch 1. The practical payoff.
9. **Walls.** (a) Transform adds/const-mults grow quadratically with tile size [Madisetti DSP handbook p.211]; transform-matrix entries grow in magnitude with tile size → accuracy degrades for large tiles [Winograd p.28]. Can't just take huge m; F(2x2) and F(4x4) are the sweet spot. (b) Convnets tolerate low precision (Courbariaux; Gupta) → can afford F(4x4,3x3)'s larger error. (c) Backprop wrt weights needs F(R×S,H×W) with H×W huge → impractical; decompose into a sum of small F(3×3,2×2) overlap-add.

## Design decisions → why
- β = {0,1,-1,2,-2,...}: small integers ⇒ evals are adds + tiny-constant mults, not real mults; symmetric ± pairs ⇒ shared adds.
- Point at infinity (Modified Cook-Toom): one fewer finite point's coefficient growth, fewer adds, still minimal.
- Same G reused via GgGᵀ in 2D: filter transform precomputable once per filter (offline-amortizable, like FIR coefficients H precomputed in Parhi).
- Tile F(2x2,3x3), not bigger: keep transform cheap + accurate + fits on-chip. F(4x4,3x3) only when precision budget allows (4x vs 2.25x).
- Reduce over channels in transform space → matmul → high arithmetic intensity → fast even at N=1. The whole reason it beats FFT at small batch.
- No Strassen on top: each recursion halves all 3 matrix dims for only 8/7 gain and shrinks the dims a GEMM needs; Winograd alone gives ≥2.25x shrinking only the P dim.

## Code grounding
Final code = wincnn.py cookToomFilter (build A,G,B from interpolation points via Vandermonde A()/At(), Lagrange L(), the f diagonal, transpose) + filterVerify + an explicit F(2,3) numeric apply + the 2D nesting GgGᵀ / BᵀdB / Aᵀ[...]A + the batched-GEMM layer loop (Algorithm 1).
