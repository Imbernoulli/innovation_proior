# Antecedents — grounded notes (retrieved this run)

## A. Schoolbook division cost (Knuth, TAOCP Vol.2 §4.3.1, Algorithm D)
Source: search of TAOCP Algorithm D descriptions + ridiculousfish/skanthak implementations.
- Multiprecision long division is O(n²): for an n-word divisor and quotient, each of the ~n
  quotient digits costs an O(n) multiply-and-subtract of the divisor.
- Per position, the quotient digit q̂ is *estimated* from the two leading divisor words and is
  then corrected: q̂ may be too large by at most 2 (with probability ≈ 3/b ≈ tiny for word-size b),
  fixed by an "add-back" of the divisor.
- Key structural point for the Barrett contrast: in Algorithm D the divisor changes per problem,
  so the trial-quotient estimate is recomputed at every position. In RSA the modulus m is FIXED
  for the whole exponentiation, so the reciprocal can be computed ONCE and reused. That is the
  opening Barrett exploits.

## B. Montgomery 1985 — the contemporaneous alternative
Source: refs/montgomery1985.pdf (full 3-page paper, Math. Comp. 44(170):519–521, Apr 1985;
received Dec 19 1983). Verbatim in refs/montgomery1985.txt.
- Idea: represent residues in a NON-STANDARD form. Pick radix R coprime to N, R > N, computations
  mod R cheap (R = power of base b → R a shift). Precompute N' with R·R^{-1} − N·N' = 1.
- REDC(T) for 0 ≤ T < RN:  m ← (T mod R)·N' mod R;  t ← (T + mN)/R;  if t ≥ N return t−N else t.
  Validity: mN ≡ −T (mod R) so t is an integer; tR ≡ T (mod N) so t = T·R^{-1} mod N; 0 ≤ t < 2N
  → at most one conditional subtraction.
- Cost replaces division by N with: one mult mod R (cheap), one mult by N, an add, a shift by R.
  Multiprecision REDC (§2): n single-precision mults mod R + n mults by N + adds; carry c ∈ {0,1}.
- Trade-off vs Barrett: Montgomery needs an N-residue domain (convert in/out via xR mod N), needs
  N ODD (coprimality), and the answer is x·R^{-1} mod N not x mod N — natural inside a long
  exponentiation (stay in the domain), awkward for a single isolated reduction. Barrett stays in
  the ordinary integer domain, handles any modulus (no oddness/coprimality requirement), at the
  price of looking at the TOP words of x (needs x < b^{2k}) and a slightly looser ≤2-subtraction tail.

## C. Fixed-point reciprocal arithmetic (the scale-and-round move)
Source: Barrett 1986 paper itself (refs/barrett1986.txt, p.317–318) + Wikipedia error-bound page.
- R = 1/M is a real number < 1; to use the DSP's cheap multiplier you must SCALE by a power of the
  base and ROUND to an integer: R := ⌊b^{2n}/M⌋ (an (n+1)-digit integer for an n-digit M).
- Because ⌊·⌋ rounds DOWN, the scaled reciprocal under-estimates 1/M, so the estimated quotient is
  never too big — only too small — which is exactly why the correction is a *subtraction* loop, never
  an add-back. Trade-off: more reciprocal precision ⇒ longer multiplies but smaller final error.
