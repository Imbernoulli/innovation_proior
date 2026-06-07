# Synthesis — Barrett reduction discovery path (grounded in refs/)

## The actual situation (from refs/barrett1986.txt — the primary source)
Oxford / Computer Security Ltd, 1986. Goal: implement RSA (512-bit modulus+exponent) fast on an
OFF-THE-SHELF chip. Options ranked and rejected: 8-bit micro (~4 min), 16-bit micro (~50 s),
discrete logic (messy), bit-slice (expensive), custom chip (dev cost + inflexible), a MAC alone
(needs custom feeder hardware). Answer: the new Texas Instruments TMS320 DSP = a fast 16×16 MAC +
microprocessor on one chip, 200 ns cycle. Target: 5 s, achieved 2.6 s avg.

Method chosen for correctness: Gries/Dijkstra guarded-command stepwise refinement (provably correct).
Top level = Knuth square-and-multiply (fastexp), with two subroutines:
  longmult: w = u·v
  longmod:  v = w mod m,  precondition 0 ≤ w < m².  ← the heart.

## longmult cost (paper's own analysis, p.315–316)
Schoolbook n×n: ~6n² instructions (n mults, 2n fetches, n stores, n carry/add per row + final adds).
Column method saving carries/fetches: ~4n² (33% saving). TMS320 auto-inc/dec pointers (MPY *+ / LTA *-)
fetch data for free → another 50%. So multiply is CHEAP on this chip.

## longmod — the wall and the move (p.317–319)
Want X = W mod M = W − M·(W div M). "Division on a DSP is hard (expensive in time)."
But across one whole exponentiation M is FIXED and multiply is cheap. So compute ONCE R = reciprocal
of M and get X by X = W − M·(W·R) — two multiplies + a subtract, no division.
Wall: R = 1/M is real, < 1. Fix: scale and round → integer reciprocal.
  R := ⌊b^{2n}/M⌋  (M is n base-b digits ⇒ W is 2n digits; R has n+1 digits).
Estimate quotient: multiply the top n+1 digits of W by R, then the top n digits of THAT by M,
subtract the low n+1 digits from the corresponding part of W.
Result X is in range 0 .. 3M−1 ⇒ at most TWO further subtractions of M.
Empirically (paper): ~90% of (W,M) need 0 corrections, only ~1% need 2.
Cost ≈ two long multiplies (and ~half that if you only compute the needed halves of each product).
Reciprocal can be precomputed and stored with M as part of the key.

## Why floor → under-estimate → subtract (never add-back)
⌊b^{2n}/M⌋ rounds the scaled reciprocal DOWN, so the estimated quotient ≤ true quotient. The estimate
is never too big, so corrections only ever ADD residue back by SUBTRACTING M. (Contrast Knuth Alg D,
whose trial digit can be too big and needs an add-back.)

## Exact bound (HAC §14.3.3, refs/HAC_14.3.3_excerpt.md) — k = paper's n
µ = ⌊b^{2k}/m⌋. q1=⌊x/b^{k-1}⌋ (top k+1 digits), q2=q1·µ, q3=⌊q2/b^{k+1}⌋.
Fact 14.43: Q−2 ≤ q3 ≤ Q (true quotient Q). Two floors each lose < 1; the x mod b^{k-1} dropped at the
top loses < 1 more ⇒ total deficit ≤ 2.
r1=x mod b^{k+1}, r2=q3·m mod b^{k+1}, r=r1−r2 (+b^{k+1} if <0). Then r=(Q−q3)m+R, 0≤r<3m<b^{k+1}
(needs m<b^k and b>3) ⇒ ≤2 subtractions. Efficiency: divisions are shifts; q3 partial-mult
(k²+5k+2)/2 single mults; r2 partial-mult. Worked Example 14.46: b=4,k=3,x=3561,m=47→36. ✓

## Montgomery (refs/montgomery1985.txt) — the contemporaneous alternative
REDC: residues in R-residue domain, R coprime to N, t=(T+mN)/R, 0≤t<2N → one cond. subtraction.
Needs domain conversion (xR mod N), N odd, output is TR^{-1} mod N. Better when you stay in the domain
through a whole exponentiation; Barrett needs no domain change, no oddness, works in plain integers,
but inspects the top words of x and tolerates a ≤2-subtraction tail.

## Canonical code (refs → code/barrett-reducer.py, nayuki)
shift = 2·bitlen(mod); factor = (1<<shift)//mod; reduce(x): t = x − ((x*factor)>>shift)*mod;
return t if t<mod else t−mod. (Single conditional subtraction suffices when shift = 2·bitlen and
x<mod²; the 3m bound collapses toward one subtraction with the bit-granular k.)

## refs/ inventory (all retrieved & read THIS run)
- refs/barrett1986.pdf + .txt  — PRIMARY, full CRYPTO '86 text (13 pp).
- refs/montgomery1985.pdf + .txt — antecedent, full Math.Comp. paper (3 pp).
- refs/hac_ch14.pdf + .txt, refs/HAC_14.3.3_excerpt.md — analysis, Alg 14.42/Fact 14.43/Notes/Example.
- refs/antecedents_notes.md — schoolbook division (Knuth Alg D), Montgomery, fixed-point reciprocal.
- code/barrett-reducer.py — canonical implementation.
