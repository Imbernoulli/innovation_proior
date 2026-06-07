# Synthesis: Tammes problem (max-min distance on S^2), principled exact-optimality method

## The task
Place N points on S^2 maximizing the minimum pairwise (geodesic/Euclidean) distance.
d_N = max over |X|=N of psi(X), psi(X)=min angular distance.
Equivalent to spherical-code / packing of N congruent caps. Lower bound = a construction (easy);
the HARD half is the UPPER bound + uniqueness: certify NO configuration beats a given value, and the
optimum is unique up to isometry. That is what "solving" means (exact, with proof), not a numeric record.

## Why this is the right framing (principled, not brute-force)
A numeric record is found by SDP/numerical optimization (anti-pattern). The principled method that
actually PROVES optimality for small N is the contact-graph / irreducible-graph approach
(Schütte–van der Waerden, Fejes Tóth, Danzer; completed computationally by Musin–Tarasov for N=13,14).
LP/Delsarte bounds and Fejes Tóth's area bound are the survey of tools that ALMOST work but leave gaps;
the methodological leap is to discretize the continuous configuration space into a FINITE list of
combinatorial graphs and eliminate each.

## Pillar A — global analytic upper bounds (surveyed, found insufficient for exact small N)
1. Fejes Tóth area bound (1943): d_N <= arccos( c_N/(1-c_N) ), c_N = cos(pi N/(3N-6)).
   Derivation: each point owns a Voronoi/Delaunay region; sum of areas of N spherical triangles in a
   triangulation with 2N-4 faces covers 4 pi; a triangle with all sides >= d has area >= area of the
   equilateral one; equilateral spherical triangle with side d has angle alpha(d)=arccos(cos d/(1+cos d)),
   so its area (angle excess) = 3 alpha(d) - pi; (2N-4)(3 alpha - pi) <= 4 pi gives alpha <= ...; invert.
   VERIFIED: N=14 -> 58.6809 deg. Sharp only at N=3,4,6,12 (when sphere tiles by equilateral triangles).
2. Delsarte LP bound for spherical codes (Delsarte–Goethals–Seidel 1977; Kabatiansky–Levenshtein):
   Gegenbauer polys P_k^(n)(t), recurrence (k+n-2)P_{k+1}=(2k+n-2)t P_k - k P_{k-1}, P_0=1,P_1=t, P_k(1)=1.
   n=3 -> Legendre. Schoenberg 1942: f(cos theta) positive-definite on S^{n-1} iff f = sum f_k P_k^(n), f_k>=0.
   Addition theorem (key identity): for any finite C, |C| f(1) + sum_{x!=y} f(<x,y>) = |C|^2 f_0 + sum_{k>=1}(f_k/r_k) sum_j (sum_x v_kj(x))^2.
   Theorem: if f in A_{n,s} (f(t)<=0 on [-1,s], f_k>=0 for k>=1, f_0>0) then any (n,M,s)-code has M <= f(1)/f_0.
   Gives kissing/code SIZE bounds; tight only n=8,24. For S^2 it bounds N given angle, not d given N exactly;
   Bachoc–Vallentin SDP: d13<58.5, d14<56.58. Musin's modified LP solved k(3)=12, k(4)=24.
   But LP/SDP give a numeric gap, NOT the exact d_N nor uniqueness for general small N.

## Pillar B — THE METHOD: irreducible contact graphs + LP elimination + rigidity (Musin–Tarasov)
Definitions:
- Contact graph CG(X): vertices X, edge (x,y) iff dist(x,y)=psi(X) (the "taut"/minimal-distance pairs).
- Shift: vertex x movable to increase its distance to the rest -> not optimal locally.
- Danzer's flip: x has two contacts y,z; reflect x across great circle yz to x'; if dist(x',rest)>psi then can improve.
- Irreducible (jammed): no shift, no Danzer flip. KEY: for N>6 the maximal X has CG(X) irreducible (Prop). 
Combinatorial consequences (Danzer, Böröczky–Szabó):
- CG(X) is planar (shortest equal-length arcs don't cross).
- Degrees in {0,3,4,5} (isolated, or 3/4/5; <3 would allow shift, >5 impossible at this distance).
- Faces are convex equilateral spherical polygons with 3..floor(2pi/d_N) <=6 vertices (triangle/rhombus/pentagon/hexagon).
- Isolated vertex (deg 0) only inside a hexagon, at most one per hexagon.
Geometric (metric) constraints per face, parameters = face angles u_ki and d=psi:
- sum of angles around each vertex = 2pi.
- u_ki >= alpha(d) (every angle at least the equilateral-triangle angle).
- triangle: equilateral, all angles = alpha(d).
- rhombus (m=4): u1=u3,u2=u4, cot(u1/2)cot(u2/2)=cos d (spherical Pythagoras) => u2=rho(u1,d)=2 cot^{-1}(tan(u1/2) cos d).
- pentagon (m=5)/hexagon (m=6): determined by d and (m-3) angles; non-flip condition gives extra inequalities
  zeta_ij(...) >= d (mirror images don't get too close); hexagon-with-isolated: lambda(...) >= d.

THE LEAP (two-part proof):
LEMMA 1 (finite enumeration). The maximal graph G_N is one of finitely many planar graphs.
  (I) Generate ALL planar graphs on N vertices with the degree/face constraints, isomorph-free, using
      `plantri` (Brinkmann–McKay). N=13: 94,754,965 candidates; N=14: ~1.5 billion.
  (II) For each candidate graph write the (nonlinear) metric constraints; LINEARIZE them (interval bounds on
      alpha, rho, etc.) and run LINEAR PROGRAMMING to test feasibility of the parameter domain D. Empty D =>
      graph cannot embed at d in the target window => eliminate. Iterate l=1,2,...: split D, tighten linear
      bounds (nested convex regions D_m subset ... subset D_1). Survivors checked by nonlinear solver (ipopt).
      Almost all killed at l=1. Survivors -> handful of subgraphs Gamma_N^(i).
LEMMA 2 (identify the maximizer & uniqueness). Among survivors Gamma_N^(0..k), show only Gamma_N^(0)=CG(P_N)
  reaches the max and the rest give psi<delta_N. Two techniques:
  - Geometric/analytic: use graph symmetry to reduce to 1-2 free parameters; e.g. N=13 build chain u5=rho(u1,d),
    ... extra vertex equation makes d a function of (u1,u2); show monotone functions u18(d) etc. force d<delta unless graph=Gamma^(0). delta_13: solve 2 tan(3pi/8 - a/2)=(1-2cos a)/cos a, cos d=cos a/(1-cos a), a13=alpha(delta13)~69.4051, delta13~57.1367 deg.
  - Stress-matrix / infinitesimal rigidity (Connelly), used for hard N=14 cases i=3,4: maximal X is infinitesimally
    rigid; an equilibrium stress (omega_ij>=0 on edges, 0 off, sum_j omega_ij e_ij=0 at each vertex, e_ij=unit tangent
    toward x_j) must exist (KKT: omega = Lagrange multipliers, >=0 since distance can't increase). Using interval
    enclosures of e_ij=(c_ij,s_ij), the equilibrium equations become linear inequalities in omega; LP shows the
    system (with normalization sum omega_ij=1) is INFEASIBLE => that graph cannot be the maximizer.

Known optimal values / configs (context only, the record, not the goal):
- d13 = 57.1367 deg (P13), unique; d14 = 55.67057 deg (P14), unique. Earlier exact: N=3,4,6,12 (Fejes Toth),
  5,7,8,9 (Schütte–vdW), 10,11 (Danzer), 24 (Robinson). N=10 closed-form by Sugimoto–Tanemura.

## Code grounding
Real implementations: `plantri` (planar graph generation), LP solver (scipy.linprog / GLPK), interval arithmetic
for the alpha/rho enclosures, nonlinear solver ipopt for survivors. The repo has no single canonical script; the
genuine pipeline is plantri -> per-graph LP elimination -> survivor check. I will write grounded Python mirroring
this: gegenbauer recurrence + LP bound (Pillar A), and the irreducible-graph constraint generator + LP feasibility
elimination (Pillar B), using real formulas (alpha, rho, angle-sum, Fejes Toth window).

## Design decisions -> why
- Contact graph (only minimal-distance edges, not all pairs): because at the optimum only the taut pairs constrain
  the configuration; the rest are slack and irrelevant to local optimality. Edge length is EXACTLY d => faces are
  equilateral, which is what makes angles the right coordinates.
- Irreducibility (no shift, no flip): the optimum is locally jammed; shifts/flips are exactly the two ways a point
  could move to strictly improve, so excluding them captures local optimality with a FINITE combinatorial signature.
- Degrees {0,3,4,5} & faces <=6: forced by d_N>~55deg (2pi/d < 7), so the configuration space is a FINITE set of
  planar graphs -> discretizes a continuum.
- LINEARIZE + LP: the metric constraints are transcendental; can't solve directly for ~10^8 graphs. Linear outer
  approximations + LP feasibility is cheap and only needs to prove EMPTINESS to eliminate; iterative refinement
  (nested regions) handles the near-degenerate survivors.
- plantri not hand enumeration: 10^8-10^9 graphs, must be isomorph-free and fast.
- Stress matrix for the last cases: when angle-chasing can't separate a near-optimal subgraph, infinitesimal
  rigidity gives a certificate (equilibrium stress exists iff jammed); LP-infeasibility of the stress system kills it.
- alpha(d), rho(u,d): exact spherical-trig identities (equilateral triangle angle; rhombus diagonal relation via
  spherical Pythagoras) — these are the nonlinear constraints being enclosed.
