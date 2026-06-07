# Synthesis: Finite-Field Kakeya via the Polynomial Method

## The pain point (research question)
- Kakeya set in F_q^n: contains a full line in every direction. Conjecture (Wolff 1999):
  |K| >= c_n q^n. This is the finite-field model of the Euclidean Kakeya conjecture
  (Hausdorff dim n), introduced to strip away measure-theoretic technicalities.
- Prior art (additive combinatorics): Wolff's bound C_n q^{(n+2)/2}; improved via
  sum-product / Bourgain-Katz-Tao / Mockenhaupt-Tao / Rogers to ~ C_n q^{4n/7} for
  general n. All these are "geometric/incidence + additive number theory": bound
  sizes of sumsets A + r·B, hairbrush/bush arguments. They never reach q^n and the
  constants/exponents degrade with n.

## The key objects (background)
- Dimension counting: space of polynomials of degree <= d in n vars has dimension
  C(n+d, n) (monomial count). VERIFIED on small fields (code/verify_kakeya.py Claim A).
- Homogeneous degree-d monomials: C(d+n-1, n-1). VERIFIED.
- Schwartz-Zippel: nonzero deg-d poly over F_q has <= d q^{n-1} zeros. VERIFIED Claim C.
- A univariate poly of degree <= q-1 with q roots in F_q is the zero polynomial.

## The methodological leap (Dvir 2008)
ASSUME K is small. Then |K| < dim of some polynomial space => a NONZERO low-degree
poly vanishes on K (linear algebra: more unknowns than equations). Then USE the
Kakeya structure (a line in every direction) to force the polynomial to be 0 =>
contradiction. The leap: turn a combinatorial/geometric size question into "a
polynomial vanishes too much, hence is zero."

## Two flavours of the argument (both grounded in primary text src/kakeya_arxiv.tex)

### (A) Dvir's original: homogeneous poly + reconstruction => q^{n-1}, then product trick.
- Theorem (strong, (delta,gamma)-Kakeya): |K| >= C(d+n-1, n-1), d = floor(q min{delta,gamma}) - 2.
- If |K| < C(d+n-1,n-1) = #(homogeneous degree-d monomials), there is a nonzero
  HOMOGENEOUS degree-d poly g vanishing on K.
- K' = {c x : x in K, c in F}. g homogeneous => g(cx)=c^d g(x) => g vanishes on K'.
- For each direction y in L (>= delta q^n of them): a line z+a y meets K in >= gamma q
  >= d+2 points. Take d+1 nonzero a_i; set b_i = a_i^{-1}; then w_i = b_i z + y in K'
  and g(w_i)=0. The w_i are d+1 distinct collinear points; g restricted to that line is
  univariate degree <= d with d+1 roots => identically 0 on it => g(y)=0.
- So g vanishes on all of L (>= delta q^n points). By Schwartz-Zippel a nonzero deg-d
  poly vanishes on <= d q^{n-1} = (d/q) q^n points. Since d/q < delta, contradiction => g=0.
- Theorem 1: |K| >= C(q-2+n-1, n-1) ~ q^{n-1}/(n-1)!. Product trick K^r is Kakeya in
  F^{nr} => bumps to q^{n - 1/r}, i.e. q^{n - eps}.

### (B) Alon-Tao improvement: drop "homogeneous", use leading-form-at-infinity => q^n.
- If |K| < C(q+n-2, n), there is a nonzero P (not nec. homogeneous) of degree <= q-1
  vanishing on K. Write P = sum_{i=0}^{q-1} P_i, P_i = homogeneous part of degree i.
- Fix direction y. Some b with P(b + a y) = 0 for all a in F (the line in K). As a poly
  in a it has degree <= q-1 with q roots => identically zero => all its a-coeffs vanish.
  Coeff of a^{q-1} is exactly P_{q-1}(y). y arbitrary => P_{q-1} = 0 (as a function on
  F^n; but a degree-(q-1) homog poly that is the zero function is the zero polynomial by SZ).
  VERIFIED: coeff of a^{q-1} in P_top(b+ay) == P_top(y) (code Claim B).
- Strip top part; repeat for P_{q-2}, ..., P_1; left with constant P_0, which is 0
  because P vanishes somewhere. So P=0, contradiction. Gives |K| >= C(q+n-2, n) ~ q^n/n!.

## The sharpening (DKSS 2009, method of multiplicities) -- context only
- q^n/n! is off from q^n by n!; even tightening, the simple argument loses a 2^n.
  Reason: SZ "degree d vanishing on > (d/q) fraction" forces d >= ~q, but degree d
  poly only has ~ d^n/n! coefficients => loses n!.
- Fix: require P to vanish to MULTIPLICITY m on K (all Hasse derivatives of order < m).
  Order-m vanishing = C(m+n-1, n) linear conditions per point. Existence when
  C(m+n-1,n)|K| < C(d+n, n).
- Multiplicity Schwartz-Zippel: sum_{a in F^n} mult(P,a) <= d q^{n-1}.
- Leading-form propagation: the top homogeneous part H_P vanishes to order ell at EVERY
  point of F^n; with d ~ ell q, ell q^n > (ell q - 1) q^{n-1} forces H_P = 0.
- Optimize m ~ 2 ell, d ~ ell q, ell -> infinity: |K| >= q^n / 2^n, tight within 2+o(1).

## Numeric ground truth (code/end_to_end.py)
- #monomials deg<=q-1 in 2 vars = C(q+1,2). Confirmed.
- Genuine small Kakeya sets (quadratic construction) in F_q^2 have size ~q^2/2
  (ratios .68,.63,.59 for q=5,7,11 -> 0.5). They STILL admit NO nonzero deg<=q-1 poly
  vanishing on them (nullspace dim 0). A set smaller than the monomial count DOES.
  This is the contradiction engine made concrete.

## Design-decision -> why
- Why polynomials of degree exactly q-1 (not less)? Because the univariate line
  restriction must have degree <= q-1 to be killed by "q roots in F_q => zero". q-1 is
  the largest degree forced to vanish by all q field elements; pushing the budget to q-1
  maximizes the monomial count C(q+n-2,n) ~ q^n/n!, hence the bound.
- Why homogeneous in (A) vs full poly in (B)? Homogeneous lets g(cx)=c^d g(x) extend
  vanishing from K to the cone K'; but it caps the bound at q^{n-1} (homog count
  ~ q^{n-1}). Dropping homogeneity and instead peeling leading forms one degree at a
  time (B) uses the FULL monomial count ~ q^n/n!, gaining a factor of q.
- Why "vanish in every direction at infinity"? The coeff of the top power a^{deg} in the
  line restriction is precisely the leading form evaluated at the direction = the value
  of the projective closure on the point at infinity of that line. Kakeya => a line in
  every direction => leading form vanishes at every projective point at infinity =>
  (being a poly of degree < q on all of F^n) it is zero. This "point at infinity" is
  the geometric content of taking the top coefficient.
- Why multiplicities (context)? Higher multiplicity packs more vanishing per point
  (C(m+n-1,n) conditions) while the zero-bound only grows linearly (sum of mult <=
  d q^{n-1}); the ratio of "conditions available" to "degree budget" improves, removing
  the n! and leaving only 2^n.

## Sources (retrieved this run)
- Primary: src/kakeya_arxiv.tex (arXiv:0803.2336 full LaTeX source).
- Tao blog: terrytao.wordpress.com/2008/03/24/dvirs-proof-of-the-finite-field-kakeya-conjecture
- DKSS: arXiv:0901.2529 (ar5iv) + cs.princeton.edu/~zdvir/papers/DKSS09.pdf.
- Wikipedia "Polynomial method in combinatorics" (lineage: Stepanov, Segre, Ellenberg-Gijswijt, Guth-Katz).
