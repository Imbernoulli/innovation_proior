# Synthesis — Cooley–Tukey FFT

## Pain point / research question
Compute the DFT X(j) = sum_{k=0}^{N-1} A(k) W^{jk}, W = exp(-2πi/N), for N samples.
Naive = matrix-vector multiply by the N×N matrix [W^{jk}] → N^2 complex multiplies.
For N=10^6 this is ~weeks of CPU vs ~seconds. Whole disciplines (spectroscopy, radar,
seismic, X-ray crystallography, time-series) were bottlenecked. Goal: sub-quadratic.

## Object being optimized
The DFT is exactly the evaluation of a degree-(N-1) polynomial at the N-th roots of unity,
OR equivalently a structured matrix-vector product. The structure to exploit: W^{jk} entries
are powers of a single root of unity → enormous redundancy (periodicity W^{N}=1).

## The leap (in-frame derivation order)
1. Look at the matrix. Entries depend only on (jk mod N). Redundant.
2. Periodicity of roots of unity is the lever. W^{j(k+N/2)} = W^{jk}·W^{jN/2} = (-1)^j W^{jk}
   (for even N). So splitting k by parity collapses the k-sum's high half onto the low half.
3. Decimation-in-time: split input by parity of index k. k=2m and k=2m+1.
   X(j) = sum_m A(2m) W^{j·2m} + sum_m A(2m+1) W^{j(2m+1)}
        = sum_m A(2m) (W^2)^{jm} + W^j sum_m A(2m+1) (W^2)^{jm}.
   W^2 = exp(-2πi/(N/2)) = the (N/2)-point root. So both inner sums are N/2-point DFTs.
   → X(j) = E(j) + W^j O(j),  where E,O are N/2-point DFTs of even/odd subsequences.
4. E, O are periodic with period N/2. So for j and j+N/2 we reuse the SAME E(j),O(j):
   X(j)       = E(j) + W^j O(j)
   X(j+N/2)   = E(j) - W^j O(j)   (since W^{j+N/2}=-W^j).
   This is the BUTTERFLY. One complex mult (W^j·O(j)) + two adds produces TWO outputs.
   W^j = twiddle factor.
5. Recurse. T(N) = 2 T(N/2) + O(N). Master theorem → T(N) = O(N log N).
   For N=2^L: L = log2 N stages, each O(N) work.
6. Bit reversal: recursively splitting on parity = sorting by least-significant bit first.
   After L splits, the single-point transforms sit in bit-reversed index order.
   One-point DFT = identity. Combine up: stage s combines length-2^s blocks.

## Mixed-radix / original Cooley–Tukey general form (grounded: Medium + Gauss-history eq 4)
N = r1·r2. Reindex j = j1·r1 + j0 (j0∈[0,r1), j1∈[0,r2)),  k = k1·r2 + k0 (k0∈[0,r2), k1∈[0,r1)).
W^{jk} factors; cross term W^{j0·k0} survives as the twiddle factor; W^{j1·r1·k1·r2}=W^{N·j1k1}=1 drops.
Inner sum over k1 = r2 transforms of length r1; multiply by twiddle W^{j0 k0}; outer sum over k0 =
r1 transforms of length r2. Cost N(r1+r2) instead of N^2. Recurse on composite r_i →
N·sopfr(N) = N·(sum of prime factors). For N=2^L this is 2N log2 N.

## Lineage (load-bearing ancestors; grounded: Gauss-history Heideman/Johnson/Burrus 1985)
- Gauss 1805 ("Theoria Interpolationis Methodo Nova Tractata", pub. posthumously 1866):
  asteroid (Pallas, Juno) orbit interpolation. N=N1·N2 sample decomposition; computed N2 sets of
  length-N1 series, then combined via length-N2 series with a shift correction = the twiddle factor.
  Worked N=12 (=4·3 and 3·4) and N=36 (=6·6). Art.27: states generalization to ≥3 factors.
  Predates Fourier 1807. Equivalent to decimation-in-frequency. Real-trig notation hid it.
- Runge 1903/1905 (+ Runge & König book): "doubling" — build 2N-point transform from two
  N-point ones in ~N auxiliary ops. Only doubling, not general.
- Stumpff 1939: doubling AND tripling; suggests arbitrary-multiple generalization.
- Danielson & Lanczos 1942 ("Some improvements in practical Fourier analysis and their
  application to X-ray scattering from liquids", J. Franklin Inst.): the cleanest derivation —
  a length-N DFT = sum of two length-N/2 DFTs (even/odd), used recursively (doubling). Cut
  a 64-point hand computation drastically (referenced as ~140 min for 8→64 work range in lore).
  Cited Runge.
- Good 1958: prime-factor algorithm (PFA) — the ONLY work Cooley–Tukey originally cited;
  but PFA needs COPRIME factors and uses CRT index map, NO twiddle factors. Distinct from C–T.
- Cooley & Tukey 1965: general composite N (not just 2^k, not just coprime), twiddle-factor
  recombination, made it an actual machine algorithm; IBM 7094, N=2048 in ~0.02 min.

## Code grounding (verified vs numpy, code/fft_reference.py, max err ~1e-15)
- recursive radix-2 DIT: even=FFT(x[0::2]); odd=FFT(x[1::2]); butterfly X[k]=E+W^k O, X[k+N/2]=E-W^k O.
- iterative in-place: bit-reverse-copy, then stages length=2,4,...,N; wlen=exp(-2πi/length);
  butterfly u=A[i+j], t=w*A[i+j+len/2], A[i+j]=u+t, A[i+j+len/2]=u-t, w*=wlen. (= NR four1.)
- naive DFT for baseline / small-N base case.

## Design decisions → why
- Decimation in TIME (split input by parity) vs in FREQUENCY (split output): dual; DIT needs
  bit-reversed input → natural output; DIF the reverse. Pick DIT for the recursion clarity.
- Radix 2 vs higher radix: power-of-2 makes every split exact & recursion uniform; NR
  "categorically recommend N a power of 2; pad with zeros". Radix-4/8 save ~20-30% by
  eliminating ±1,0 multiplies but complicate indexing.
- Twiddle applied to ODD half only (one mult per butterfly) — periodicity gives the second
  output for free via the sign flip. This halving of multiplies per level is the whole point.
- Bit reversal not extra-order: O(N) swaps; it's the trace of which LSBs were peeled.
- In-place: butterfly reads & writes the same two slots → no extra storage; pairs swap symmetric.
- sign convention: forward uses W=exp(-2πi/N); isign=-1 → inverse (unnormalized). NR four1 uses
  this sign flag.
