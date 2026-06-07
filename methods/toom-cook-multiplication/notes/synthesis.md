# Synthesis — Toom–Cook multiplication

## Sources actually read this run
- **Primary**: Cook 1966 PhD thesis (Harvard), Chapter III "Positive Results", §16 "The Toom
  algorithm for multiplication", §17 "A Turing machine which incorporates the algorithm",
  §18 "Solution of the linear equations". Scanned JPGs (Bernstein scan) in
  `refs/cook_thesis_pages/`, OCR in `refs/cook-1966-thesis-ocr-sec16-18.txt`. Pages read:
  01–06 (setup + theorem), 13–16 (interpolation = Gaussian elimination on the Vandermonde
  matrix via P(X)-P(Y)=(X-Y)Q(X)). This is THE primary.
- **Primary antecedent (named by Cook as [14])**: A. L. Toom 1963, "The complexity of a
  scheme of functional elements realizing the multiplication of integers", Soviet Math.
  Doklady 3, 714–716 (orig. Doklady AN SSSR 150 (1963) 496–498). Could not obtain a free
  full-text PDF this run; its content is faithfully relayed by Cook §16 (Toom's bound
  C1·n·C2^√log n components; the polynomial-evaluation scheme that Cook makes "slightly
  modified" and turns into a TM algorithm). Noted as a gap (secondary access only).
- **Direct ancestor (already in repo)**: Karatsuba–Ofman 1962 / the k=2 case
  (`methods/karatsuba-multiplication/`). Toom–Cook = generalization to k parts.
- **Knuth TAOCP Vol 2 §4.3.3** named as primary; consulted via standard restatements
  (Wikipedia Toom-Cook, GMP manual) since no free copy — the worked Toom-3 with 5 points and
  the interpolation. Noted as gap (no direct copy read).
- **Standard implementation / worked Toom-3**: Wikipedia Toom–Cook (points {0,1,-1,-2,∞},
  Bodrato interpolation sequence + 5×5 Vandermonde inverse) and GMP manual (points
  {0,1,-1,2,∞}). Bodrato sequence numerically verified in `code/` over 2000 random polys.

## Self-account?
No first-person Toom or Cook discovery account found (searched lectures/interviews/memoirs;
SELF_ACCOUNT_SOURCES.md has Knuth entries for KMP but nothing for Toom-Cook). Cook's thesis
prose itself is mildly first-person about the *why* ("It seems especially interesting that
this type of multiplication can be performed so rapidly, since it is so easy to visualize
the dependence of the output on the input... if each output calculation were independent of
the others, the computing time would grow as n²"). That motivating remark is the closest
thing to omitted reasoning and is used as a backbone hint. Recorded as: no dedicated
self-account; reconstruct from primary + Karatsuba antecedent.

## The pain point / research question
Karatsuba (k=2) gives n^{log2 3}=n^1.585, beating the n² that Kolmogorov conjectured was a
floor. Natural question: is 1.585 special, or just the k=2 instance of something? If
splitting into 2 saves 1 of 4 products (4→3), splitting into k parts ought to save more.
Goal: drive the exponent toward 1, i.e. n^{1+ε} for any ε>0.

## Core derivation (the spine of reasoning.md)
1. **Reframe multiplication as polynomial multiplication.** Write x in base B^m by cutting it
   into k limbs: x = Σ x_i B^{mi} = p(B^m) where p(t)=Σ x_i t^i has degree k-1. Same for y,
   q(t). Then x·y = (p·q)(B^m). So the whole problem is: compute the product polynomial
   r=p·q, then evaluate it at t=B^m (a shift-and-add, O(n)). (Cook §16: M=P(b^q),
   product = coefficients of P·Q evaluated at b^q.)
2. **Schoolbook on polynomials = k² coefficient multiplications** (the convolution). That is
   the k² we must beat — exactly the analogue of the digit-pair argument.
3. **Key insight: a degree-(2k-2) polynomial r is determined by its values at 2k-1 distinct
   points.** So instead of computing r's coefficients by convolution, compute r's *values*:
   r(s_j) = p(s_j)·q(s_j) at 2k-1 points s_j. Each is ONE multiplication of two numbers each
   ~1/k the size. That is 2k-1 sub-multiplications (not k²), each on operands of length n/k.
4. **Recover coefficients = interpolation** through the 2k-1 points. (Cook §16 eqns 16.4–16.5:
   evaluate at k=0..2r, multiply m_k n_k, solve the Vandermonde system Σ γ_j s^j = m_k n_k.)
5. **Recurrence** T(n) = (2k-1)·T(n/k) + O(n) ⇒ Θ(n^{log_k(2k-1)}).
   - k=2: log2 3 = 1.585 (Karatsuba falls out as a special case — the unification).
   - k=3: log3 5 = 1.465.
   - As k→∞, log_k(2k-1) → 1, so exponent → 1+ε. (Cook's near-optimal r=2^{[√log q]} drives
     the bound to n·2^{5√log n} = n^{1+o(1)}.)
6. **Why interpolation is exact / why evaluation points matter.** Solving the Vandermonde
   system needs the points distinct AND the determinant invertible in the ring. Over Z this is
   fine for distinct integers; the divisions (by Vandermonde minors / by 3, by 2) come out
   exact because r's coefficients are integers. Cook §18: do Gaussian elimination keeping
   entries in R, using the identity P(X)-P(Y)=(X-Y)Q(X) so each elimination step's "division"
   is a clean polynomial division of monic polys — no fractions accumulate. (For polynomials
   over Z_b this needs b prime so Z_b is an integral domain and 16.5 has a unique solution.)
7. **Choice of points.** Small integers {0,±1,±2,...} and ∞ (the leading-coeff "point") keep
   the evaluations cheap (just adds/shifts, no real multiply) and the interpolation constants
   small. ∞ means: the top coefficient r_{2k-2}=x_{k-1}·y_{k-1} read off directly. 0 gives
   r_0=x_0 y_0 directly. The rest are small dot products.

## Toom-3 concrete (the landing code)
Split into 3: p(t)=x2 t²+x1 t+x0, deg 2; product deg 4 ⇒ 5 points {0,1,-1,-2,∞}.
Evaluations:
  v0   = x0·y0
  v1   = (x0+x1+x2)(y0+y1+y2)
  v-1  = (x0-x1+x2)(y0-y1+y2)
  v-2  = (x0-2x1+4x2)(y0-2y1+4y2)
  vinf = x2·y2
Interpolation (Bodrato, VERIFIED in code over 2000 random polynomials):
  r0 = v0
  r4 = vinf
  r3 = (v-2 - v1)/3
  r1 = (v1 - v-1)/2
  r2 = v-1 - v0
  r3 = (r2 - r3)/2 + 2·vinf
  r2 = r2 + r1 - vinf
  r1 = r1 - r3
Recompose r(B) = Σ r_i B^i with B=base^m.
Recurrence T(n)=5T(n/3)+O(n) ⇒ Θ(n^{log3 5}) ≈ n^1.465.

## Design decisions → why
- **Why polynomials at all?** Decouples "what to compute" (a convolution) from "how" — lets
  the evaluate/interpolate trick apply. The base-B value is recovered by one shift-eval.
- **Why 2k-1 points?** deg(p·q)=2k-2, so 2k-1 values pin it down uniquely (interpolation).
- **Why this beats k² convolution?** 2k-1 grows linearly in k while k² grows quadratically;
  recursively the branching 2k-1 with shrink factor k gives exponent log_k(2k-1)<2 for k≥2,
  decreasing in k.
- **Why does Karatsuba = k=2?** 2·2-1=3 points (0,∞,1 or 0,∞,-1): the 3 products are exactly
  x0y0, x1y1, and (x0+x1)(y0+y1) — the Karatsuba three. The middle coefficient via
  interpolation IS the (sum-of-products − two corners) identity. Unification lands here.
- **Why include ∞ as a point?** Gives the leading coefficient for free (no multiply of a
  combination) and keeps the interpolation matrix nonsingular with minimal-magnitude entries.
- **Why small integer points?** Evaluations are adds/small-shifts (cheap, O(n)); large points
  would make p(s) itself large and inflate the recursive multiply sizes and the interpolation
  constants.
- **Why exact division works (no floats)?** r has integer coeffs; the Vandermonde system over
  Z has integer solution; Cook's monic-poly elimination keeps everything in R. In code we use
  Fraction only as a guard and cast back to int — the divisions by 3 and 2 are always exact.
- **Why a threshold base case?** Recursion must bottom out on operands small enough that direct
  multiply is O(1); without it (or with true division on the split) it never terminates.
- **m = n//3 + 1**: ensures each operand splits into at most 3 limbs in base B=base^m.

## Anti-pattern guard
reasoning.md: no headers, first-person present, discovery order (insight before formula),
Karatsuba "aha" emerges from the recurrence, no hindsight, no reference to "the paper".
