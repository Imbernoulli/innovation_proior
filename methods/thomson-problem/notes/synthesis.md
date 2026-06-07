# Synthesis: principled method behind rigorous Thomson / universal-optimality results

## Pain point
Thomson problem: place N points on S^{n-1} minimizing sum_{x!=y} f(|x-y|^2),
f = Coulomb 1/r (i.e. f(R)=R^{-1/2}) or Riesz 1/r^s. Numerical relaxation
(steepest descent on the sphere, simulated annealing) finds candidate minima but
PROVES nothing — # local minima grows exponentially; the global minimum for N=8,20
is NOT the Platonic solid. Anti-pattern = the numerical record (e.g. 20 points,
Coulomb, energy 301.763..., not a dodecahedron). We want a PROOF.

## The principled leap: a dual certificate, not a search
Don't search the primal (configurations). Build a single function that lower-bounds
EVERY configuration's energy at once, and make that bound equal the candidate's energy.

### LP / positive-definite bound (Delsarte-Yudin; Cohn-Kumar Prop. on sphere)
Replace f(2-2t) (t = inner product) from above by a polynomial h(t):
  - (i)  h(t) <= f(2-2t) for all t in [-1,1)   [pointwise domination]
  - (ii) h(t) = sum_i alpha_i C_i^{lambda}(t), alpha_i >= 0, lambda=n/2-1
         [positive definite: Gegenbauer/ultraspherical coefficients nonneg]
Then for ANY N points:
  sum_{x!=y} f >= sum_{x!=y} h(<x,y>) = -N h(1) + sum_{all x,y} h(<x,y>)
                = -N h(1) + sum_i alpha_i sum_{x,y} C_i(<x,y>)
                >= -N h(1) + alpha_0 N^2  =  N^2 alpha_0 - N h(1).
Key fact (Schoenberg): C_i^{n/2-1}(<x,y>) is a positive-definite KERNEL on the
sphere => sum_{x,y} C_i(<x,y>) >= 0; the i=0 term gives alpha_0 N^2 exactly
(C_0=1). Choosing h = solving an infinite-dim LP. This is the generalization of
the Kabatiansky-Levenshtein / Delsarte-Goethals-Seidel LP bound for codes
(recover it: f = +inf above cos theta, 0 below).

### Why Gegenbauer? Schoenberg's theorem.
L^2(S^{n-1}) = direct sum of spherical-harmonic spaces V_l (O(n)-irreps). The
reproducing kernel of V_l is K_l(x,y)=C_l^{n/2-1}(<x,y>) (positive-definite kernel
since it's <ev_x,ev_y>). Schoenberg: continuous distance-only PD kernels are
EXACTLY nonneg combos of these C_l. So "positive-definite function" <=> nonneg
Gegenbauer coefficients. Normalization: C_0=1, C_1=2*lambda*t, recurrence
i C_i = 2(i+lambda-1) t C_{i-1} - (i+2 lambda-2) C_{i-2}. Orthogonal wrt
(1-t^2)^{(n-3)/2} dt.

### When is the bound SHARP (= equals candidate energy)? Two conditions.
1. h(t) = f(2-2t) at every inner product t_1..t_m occurring in S (so the energy
   estimate is tight pair-by-pair).
2. For every i>0 with alpha_i>0: sum_{x,y in S} C_i(<x,y>)=0, i.e. S is a spherical
   deg(h)-design (Delsarte-Goethals-Seidel Thm 5.5: M-design iff these sums vanish
   for 1<=i<=M).

### Constructing the magic auxiliary function (the methodological core)
Want h with (a) h<=a:=f(2-2.) ; (b) double contact at each t_i (so h=a and h'=a'
there => h <= a near each contact and no sign change); (c) positive-definite.
Construction: h = Hermite interpolation of a to ORDER 2 at the m inner products
t_1..t_m of the candidate (degree 2m-1). Then:
 - (a)/(b): remainder formula a(t)-h(t) = a^{(2m)}(xi)/(2m)! * prod (t-t_i)^2 >= 0
   because a absolutely monotonic (a^{(k)}>=0) <- f completely monotonic. So h<=a. SHARP at t_i.
 - design: a sharp configuration is a spherical (2m-1)-design by DEFINITION
   (m distances, (2m-1)-design), exactly deg(h)=2m-1 => condition 2 holds.
 - positive-definite: the hard part. F(t)=prod(t-t_i). Show F^2 "conductive":
   for absolutely monotonic a, H(a,g) is positive definite. Built from: F strictly
   positive definite (its low Gegenbauer coeffs = design-strength sums, vanish; leading
   positive); partial products prod_{i<=j}(t-t_i) are p.d. via the Jacobi-orthogonal-
   polynomial relation F=p_m+alpha p_{m-1} (p_k monic orthog wrt (1-t)dmu, nonneg
   Gegenbauer combos); products of p.d. functions are p.d. (Schur/Gegenbauer linearization
   nonneg). Conductivity multiplies up: H(a,g1 g2)=H(a,g1)+g1 H(Q(a,g1),g2),
   Q(a,g) absolutely monotonic (Prop). => h=H(a,F^2) is positive definite. DONE.

### "sharp configuration" definition
m inner products between distinct points AND a spherical (2m-1)-design. Examples
(Table): regular simplex, cross-polytope, icosahedron (3,12,M=5), E8 roots (8,240,M=7),
Leech min vectors (24,196560,M=11), 600-cell handled separately (only an 11-design,
needs degree-15 polynomial -> extra trick: force 12th-14th Gegenbauer coeffs >=0/=0).
This list = exactly where LP bound for spherical codes is sharp (Levenshtein).

### Universal optimality
Because the construction works for EVERY completely monotonic f at once (only used:
a absolutely monotonic), sharp configs minimize energy for ALL completely monotonic
potentials simultaneously => "universally optimal". By Bernstein/Widder, completely
monotonic = Laplace transform of nonneg measure / nonneg combo of (4-r)^k; so testing
on f=(4-r)^k or f=1/r^s suffices. Includes Coulomb 1/r, all Riesz 1/r^s, Gaussians.

### Euclidean culmination (E8, Leech)
Same dual idea in R^n (not sphere): Cohn-Elkies-type LP bound for energy. Lower bound
on energy of a periodic/general config from a radial function f with sign conditions
on f and its Fourier transform f^, made tight via Poisson summation. The "magic"
auxiliary function is built by a NEW interpolation theorem: a radial Schwartz function
is reconstructed from values+derivatives of f and f^ at radii sqrt(2k) (the vector
lengths of E8 / Leech), using integral transforms of (quasi)modular forms. Same shape:
domination + Fourier-positivity (the Euclidean analog of Gegenbauer-nonneg) + double
roots at the lattice distances => sharp. Proves E8 (n=8), Leech (n=24) universally
optimal. The d=2 triangular lattice is conjectured but open.

## Design decisions -> why
- Squared distance as the variable: |x-y|^2=2-2t linear in inner product; makes
  Gegenbauer machinery apply; completely-monotonic-in-R is the right (broader) class.
- Order-2 (double) Hermite interpolation, not order-1: avoids sign change of h-a at
  each t_i so domination holds on the whole interval (at t=-1 first order would do).
- Degree 2m-1 (not 2m+1): a (2m+1)-design with only m distances is impossible
  (the nonneg poly (1-<x,y>)prod(<x,y>-t_i)^2 vanishes on S but has positive integral)
  => 2m-1 is the max design strength, matched exactly.
- Positive-definite (alpha_i>=0) is forced: only then sum_{x,y}C_i>=0 makes the bound
  valid for ALL configs (Schoenberg). Drop it and the bound is false.
- Complete monotonicity of f (not just convex/decreasing): needed so a is absolutely
  monotonic so the remainder a^{(2m)}(xi)>=0 gives domination AND Q(a,.) stays abs.
  monotonic through the conductivity induction. Pure convexity only handles the simplex
  (1 distance) case.

## Code grounding
LP bound is a finite LP: variables alpha_0..alpha_d>=0, constraints h(t_grid)<=f(2-2t),
maximize N^2 alpha_0 - N h(1). scipy.linprog + scipy.special.gegenbauer. Verified:
S^2, N=12, Coulomb -> bound 98.333 == icosahedron energy 98.331 (grid-limited). The
Hermite-interpolation construction gives the sharp certificate in closed form for sharp
configs. Numerical steepest descent (the anti-pattern) is the primal search; we use it
only to FIND candidates, never to prove.

## Sources
- Cohn & Kumar, "Universally optimal distribution of points on spheres", JAMS 2007
  (arXiv math/0607446) -- primary, read in full.
- Cohn, Kumar, Miller, Radchenko, Viazovska, "Universal optimality of E8 and Leech...",
  Annals 2022 (arXiv 1902.05438).
- Delsarte-Goethals-Seidel (designs/LP), Kabatiansky-Levenshtein (LP codes),
  Yudin / Kolushov-Yudin / Andreev (early energy bounds), Schoenberg (PD kernels),
  Bernstein-Widder (completely monotonic = Laplace transform).
- Ballinger-Blekherman-Cohn et al, "Experimental study..." (arXiv math/0611451).
- Wikipedia Thomson problem; Quanta "magic function" article.
