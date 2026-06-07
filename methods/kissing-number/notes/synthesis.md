# Synthesis — Kissing Number / Delsarte LP bound

## The task
tau_n = max number of non-overlapping unit spheres touching a central unit sphere in R^n.
Equivalently: max number of unit vectors x_i in S^{n-1} with <x_i,x_j> <= 1/2 for i!=j
(touching unit spheres at distance 2 apart => centers at angle >= 60deg => inner product <= 1/2).
So tau_n = A(n, 1/2), A(n,s) = max code on sphere with max inner product <= s.

Known exact: tau_1=2, tau_2=6, tau_3=12 (Schutte-vdW 1953), tau_4=24 (Musin 2008),
tau_8=240, tau_24=196560 (Levenshtein 1979; Odlyzko-Sloane 1979). Only these six.

## Pain point / prior art before the LP method
- Lower bounds: easy, by exhibiting configurations (lattice min vectors). tau_8>=240 from E8,
  tau_24>=196560 from Leech, tau_3>=12 from icosahedron.
- Upper bounds: HARD. Newton vs Gregory 1694 dispute over tau_3 (12 vs 13). Even tau_3<=12
  resisted rigorous proof until 1953 (Schutte-vdW), via heavy spherical-trigonometry case analysis
  (areas of spherical triangles, Toth-type packing of caps). Does not generalize / scale.
- Coxeter-Boroczky bound (1963, via Schlafli function, Boroczky 1978): general but weak; weaker
  than LP for kissing. Good only near s->1 (e.g. A(4,cos pi/5)=120, the 600-cell).
- Fejes Toth bound: bound on min distance D(n,M); attained only M=3,4,6,12.

## The methodological leap: Delsarte LP (Delsarte-Goethals-Seidel 1977; KL 1978)
Analogy from finite-field coding theory: Delsarte 1973 used Krawtchouk polynomials + LP
to bound binary codes. DGS transplant it to the SPHERE; the right orthogonal polynomials are
Gegenbauer (ultraspherical) polynomials.

### The engine: Schoenberg / addition theorem (positive-definite kernels)
Gegenbauer P_k^{(n)} (G_k), normalized G_k(1)=1, recurrence:
  P_0=1, P_1=t, (k+n-2)P_{k+1}(t) = (2k+n-2) t P_k(t) - k P_{k-1}(t).
Orthogonal wrt weight (1-t^2)^{(n-3)/2} on [-1,1] (the measure from integrating over S^{n-1}).
n=3 -> Legendre; n=4 -> Chebyshev 2nd kind.

ADDITION THEOREM (Lemma 1, goes back to Herglotz/Muller):
  G_k(<x,y>) = (omega_n / m) * sum_{l=1}^m S_{k,l}(x) S_{k,l}(y),
where {S_{k,l}} is an orthonormal basis of degree-k spherical harmonics, dim m=m(k,n).
=> For any finite set X on the sphere:
  sum_{x,y in X} G_k(<x,y>) = (omega_n/m) sum_l ( sum_{x} S_{k,l}(x) )^2 >= 0.
This is SCHOENBERG'S LEMMA: G_k is a positive-definite kernel on the sphere; the Gram-style
sum of all entries of [G_k(<x_i,x_j>)] is >=0. (k=0 gives G_0=1, sum = N^2.)

### The bound (Theorem / DGS) — the two-way count
Let f(t) = sum_{k=0}^d c_k G_k(t), with:
 (A2) c_0 > 0, c_k >= 0 for k>=1   [Gegenbauer coefficients nonneg]
 (A1) f(t) <= 0 for all t in [-1, 1/2]   [<-1,s]; for kissing s=1/2]
Then count S = sum_{i,j=1}^N f(<x_i,x_j>) two ways.
 Lower:  S = sum_k c_k sum_{i,j} G_k(<x_i,x_j>) >= c_0 * sum_{i,j} G_0 = c_0 N^2
         (drop all k>=1 terms, each >=0 by Schoenberg).
 Upper:  S = N f(1) + sum_{i!=j} f(<x_i,x_j>) <= N f(1)
         (each off-diagonal term <=0 since <x_i,x_j> in [-1,1/2] and (A1)).
=> c_0 N^2 <= N f(1) => N <= f(1)/c_0. So tau_n = A(n,1/2) <= f(1)/c_0.
The whole problem becomes: MINIMIZE f(1)/c_0 over admissible f. This is a linear program
(infinitely many constraints f(t)<=0; finite once degree fixed).

### Why this is principled, not search
Best polynomial is FORCED, not guessed. For the bound to be TIGHT (N = f(1)/c_0) both
inequalities must be equalities:
 - Upper tight: f(<x_i,x_j>) = 0 for EVERY inner product that actually occurs (i!=j) in the
   optimal configuration. So f must VANISH at exactly the inner-product set A(X) of the candidate.
 - Lower tight: c_k sum_{i,j} G_k = 0 for all k>=1 -> the configuration is a spherical design
   (harmonic moments vanish up to degree d).
=> Read off the candidate's inner-product set, write f with those as roots, then check (A2).

## d=8 (E8) — the magic polynomial
E8 min vectors (length sqrt8), normalized to S^7: inner products that occur = {-1, -1/2, 0, 1/2}.
Force f to vanish there, with even multiplicity where the constraint f<=0 must only touch 0:
  f_8(t) = (t+1)(t+1/2)^2 t^2 (t-1/2).
Degree 6. On [-1,1/2]: factors (t+1)>=0, squares >=0, (t-1/2)<=0 => f<=0. Good (A1).
Expand in Gegenbauer for n=8: all c_k >= 0 (check), c_0>0. Good (A2).
f_8(1)/c_0 = 240. Matches E8 lower bound => tau_8 = 240 exactly.

## d=24 (Leech) — same recipe
Leech min vectors (length sqrt32 = 4sqrt2) normalized: inner products = {-1,-1/2,-1/4,0,1/4,1/2}.
  f_24(t) = (t+1)(t+1/2)^2 (t+1/4)^2 t^2 (t-1/4)^2 (t-1/2).
Degree 10. f<=0 on [-1,1/2], Gegenbauer coeffs (n=24) nonneg.
f_24(1)/c_0 = 196560. Matches Leech => tau_24 = 196560 exactly.

## Levenshtein's universal polynomials (1979)
Don't want to guess per dimension. Levenshtein gives the optimal LP polynomial in closed form
via Jacobi polynomials P_k^{(a+(n-3)/2, b+(n-3)/2)}, a,b in {0,1}. Let t_k^{a,b} = largest zero.
Intervals I_m partition [-1,1): I_{2k-1}=[t_{k-1}^{1,1}, t_k^{1,0}], I_{2k}=[t_k^{1,0}, t_k^{1,1}].
For s in I_m use
  f_m^{(n,s)}(t) = (t-s) (T_{k-1}^{1,0}(t,s))^2          if m=2k-1
                 = (t+1)(t-s)(T_{k-1}^{1,1}(t,s))^2      if m=2k
(the T are kernel/Christoffel-Darboux type). Levenshtein proved these satisfy (A1),(A2) and give
the universal bound L_m(n,s) (Theorem 2, explicit binomials). For s=1/2:
  tau_8 <= L_6(8,1/2)=L_7=240, tau_24<=L_10(24,1/2)=L_11=196560 — recovers the exact cases.
Sidelnikov: L_m is the BEST pure-LP bound among polynomials of degree <= m (later m+2, BDD).

## d=3, d=4 — why pure LP fails, and Musin's extension
Pure LP: tau_3 <= L_5(3,1/2) ~ 13.285, improvable only to ~13.18 — not < 13. tau_4 <= 25.55,
so tau_4 in {24,25}; Arestov-Babenko: 25 is the best pure-LP can do. The obstacle: optimal
config not unique / not a tight design, so f can't simultaneously vanish on a whole RANGE.
Musin's trick (2003/2008): RELAX (A1). Allow f(t)>0 near t=-1 (the cap opposite a point):
 (B1) f<=0 on [t0, 1/2] with -1<=t0<-1/2; (B2) f decreasing on [-1,t0]; (B3) Gegenbauer coeffs >=0.
Then in the upper count you can't drop the off-diagonal terms with <x_i,x_j> in [-1,t0]; instead
bound them: at most mu points fit in the cap {y: <e1,y> <= t0} with pairwise <= 1/2, so
  sum_{i,j} f <= max_{0<=m<=mu} [ f(1) + max over m caps of sum_j f(<e1,y_j>) ] = max h_m.
=> tau_n <= max{h_0,...,h_mu}/c_0. The h_m are nonconvex sub-optimizations (needs S(n,M) bounds).
n=3: mu=4, t0=-0.5907, degree-9 polynomial -> tau_3 = 12. n=4: mu=6, t0=-0.608, deg 9 -> tau_4=24.
(SDP of Bachoc-Vallentin 2008 / three-point distributions A_{u,v,t} pushed n=5,6,7,9,10 further.)

## Lower bounds / constructions (where d=11 lives)
Upper bound is the LP story; the matching LOWER bound is a CONSTRUCTION.
Construction A (Leech-Sloane): from binary (n,M,d) code C, lattice = integer vectors whose
mod-2 reduction is in C. Min-distance contacts: 2^d A_d(x) if d<4, 2n+16A_4 if d=4, 2n if d>4.
Construction B: even-weight codewords + sum divisible by 4; contacts 2^{d-1}A_d (d<8),
 2n(n-1)+128A_8 (d=8), 2n(n-1) (d>8). B on the right code gives the even Leech in n=24.
Cross-sections / laminations build neighbours. Concatenation (Dodunekov-Ericson-Zinoviev) gives
most record kissing lower bounds below 24. d=11: laminated lattice Lambda_11 min vectors give
tau_11 >= 582 (Conway-Sloane Table 1.5); upper bound 870. The gap (582 vs 870) is why 11 is
an open intermediate case where explicit constructions decide the record. (Recently improved
>582, e.g. 592, via optimized spherical codes.)

## Design-decision -> why
- Gegenbauer not arbitrary basis: only nonneg combos of Gegenbauer stay PSD on rank-n PSD Gram
  matrices with ones on diagonal (Schoenberg). The recurrence/weight come from S^{n-1} geometry.
- c_k>=0 (k>=1): needed to DROP those terms in the lower bound (they are >=0).
- f<=0 on [-1,1/2]: needed to drop off-diagonal in the upper bound (the kissing constraint window).
- minimize f(1)/c_0: that IS the bound; normalize c_0=1 WLOG.
- magic polynomials: roots = the candidate config's inner products (forces upper-tightness),
  squared at interior roots so f only touches 0 from below (keeps (A1)); endpoint -1 simple,
  s=1/2 simple. Degree = #distinct inner products counted with the tightness multiplicities.
- Musin t0: chosen so the cap opposite a point holds few (mu) points; relaxing f>0 there buys
  a smaller f(1)/c_0 at the cost of a nonconvex max over cap configs.

## Code
The LP is solved numerically: choose degree d, sample t in [-1,1/2], variables c_0..c_d>=0,
maximize c_0 s.t. f(1)=1 (i.e. minimize f(1)/c_0) and f(t)<=0 on grid. Use cvxpy/scipy linprog,
Gegenbauer via the recurrence. For d=8 just symbolically expand f_8 in Gegenbauer and verify.
Reference numeric path: numpy + scipy.optimize.linprog or cvxpy.

## Key sources read
- Boyvalenkov-Dodunekov-Musin, "Kissing numbers - a survey", arXiv:1507.03631 (LaTeX source read in full):
  Theorems 1-4, magic polynomials, Levenshtein bound, Construction A/B, the n<=24 table.
- Pfender-Ziegler, "Kissing Numbers, Sphere Packings, and Some Unexpected Proofs", Notices AMS Sep 2004:
  addition theorem, Schoenberg lemma proof, two-way count, explicit E8/Leech vectors, Musin's trick.
- dustingmixon.wordpress.com "Spherical codes and designs": clean LP-bound proof, f_8 = (320/3)(...).
- Musin, "The kissing problem in three dimensions", arXiv:math/0410324: n=3 details, t0=-0.5907.
