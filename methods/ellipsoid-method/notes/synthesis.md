# Synthesis — Ellipsoid Method for LP feasibility (Khachiyan 1979/1980)

## Primary sources read this run (refs/)
- `khachiyan_1979_russian_dan.pdf` + OCR `khachiyan_1979_ocr_rus.txt` / `khachiyan_1979_russian.txt` — THE primary source. L.G. Khachiyan, "Полиномиальный алгоритм в линейном программировании", Dokl. Akad. Nauk SSSR 244:5 (1979) 1093–1096. Presented by acad. A.A. Dorodnitsyn, 4 X 1978. (English: Soviet Math. Doklady 20 (1979) 191–194.) Read fully, all 4 pages.
- `gls_combinatorica_1981.pdf` + `gls_1981.txt` — Grötschel, Lovász, Schrijver, "The ellipsoid method and its consequences in combinatorial optimization", Combinatorica 1(2) (1981) 169–197. The separation⇔optimization equivalence Theorem (3.1); the rounded-arithmetic ellipsoid algorithm (Ch. 2).
- `lovasz_ellipsoid_chapter.pdf` / `lovasz-ch3.pdf` — GLS book "Geometric Algorithms and Combinatorial Optimization" (1993), Ch. 3. Antecedent history in authors' own words; E(A,a) def; vol = √detA·Vₙ.
- `goemans_mit_ellipsoid_notes.pdf` — MIT 18.433 Goemans. The cleanest full derivation: unit-sphere half-ellipsoid formula, vol ratio < e^{−1/(2(n+1))} via 1+x≤eˣ, affine pullback giving rank-1 update, encoding-length termination, separation⇒optimization (GLS Thm 3).
- `boyd-ellipsoid.txt` — Boyd EE364b. Canonical code (P-form and H-form), deep-cut variant, bisection-in-n-dim interpretation, quasi-Newton view, Newman/Levin precursor note.
- `cmu-lecture08.pdf`, `princeton-ellipsoid.pdf`, `nemirovski_ellipsoid_redux_2301.07632.pdf` — supporting analyses.

## The exact problem (Khachiyan's own framing, from OCR p.1)
Decide feasibility in ℝⁿ of a system of m≥2 linear inequalities Aᵢx ≤ bᵢ, i=1..m, n≥2, with INTEGER coefficients aᵢⱼ, bᵢ. Input/encoding length
  L = ⌈Σ_{i,j} log₂(|aᵢⱼ|+1) + Σ_i log₂(|bᵢ|+1) + log₂(nm)⌉ + 1   (his eq (2))
= number of 0/1 symbols to write (1) in binary. He proves: a poly(L) algorithm; memory O(nm+n²) numbers each O(nL) bits; O(n²(n²+m)L) arithmetic operations (+,−,×,÷,√,max) at O(nL)-bit precision. ⇒ feasibility of linear inequalities is in P. ⇒ LP (maximize integer linear form s.t. (1)) is in P. He explicitly notes Karp [1] posed whether this feasibility problem is NP-complete; so EITHER it is not complete OR P=NP.

## The two localization lemmas (Khachiyan p.1–2)
- Lemma 1: if (1) is feasible there is a solution x° in the ball S = {x : ‖x‖ ≤ 2^L}. (So we only search a ball of radius 2^L.)
- Define residual σ(x) = maxᵢ {aᵢ₁x₁+…+aᵢₙxₙ − bᵢ}. x is a solution iff σ(x) ≤ 0.
- Lemma 2: if (1) is INFEASIBLE then for every x∈ℝⁿ, σ(x) ≥ 2^{−L}. (A "gap": infeasible systems are violated by a margin 2^{−L}.)
- ⇒ Decision rule (his §1): it suffices to find x with σ(x) ≤ σ_S + 2^{−L} where σ_S = min residual over the ball S. Then either σ(x) < 2^{−L} (feasible) or σ(x) ≥ 2·2^{−L} → ... actually: either σ_S<0 → feasible point exists; reaching σ(x) within 2^{−L} of σ_S decides it, because feasible⇒σ_S≤0 and infeasible⇒σ_S≥2^{−L}.

## The geometric construction (Khachiyan §2, Lemma 3)
Ellipsoid E~(x,Q): center x, (n×n) matrix Q, E = {y : y = x + Qz, ‖z‖≤1} (image of unit ball under affine map z↦x+Qz). Nondegenerate iff detQ≠0. vol E = |detQ|·vol(unit ball).
Half-ellipsoid ½E_R = E ∩ {y : Rᵀ(y−x) ≥ 0} (cut by hyperplane through center with normal R).
Lemma 3: given E~(x,Q) and R≠0, the ellipsoid E^~(x^,Q^) with
  x^ = x − (1/(n+1))·QQᵀR/‖QᵀR‖
  Q^ = 2^{1/(2n²)}·Q·ORT(QᵀR)·Δₙ
where ORT(QᵀR) is an orthogonal matrix whose first column is QᵀR/‖QᵀR‖, and
  Δₙ = diag( n/(n+1), 1/√(1−1/n²), …, 1/√(1−1/n²) )  (first entry special, rest equal)
contains ½E_R, and the volume ratio det Q^/det Q ∈ (2^{−1/(2n)}, 2^{1/(2n)})·... his (2,5): 2^{−1/(2n)} ≤ detQ^/detQ ≤ 2^{−1/(4n)} approx — the key shrink ~ 2^{−1/(...)} per step; with the 2^{1/(2n²)} fudge factor inserted to absorb rounding so that approximate (δ-accurate) computation still yields a valid enclosing ellipsoid. δ = 2^{−8nL} accuracy.

NOTE the equivalence to the modern (Goemans/Boyd) PSD form: with A = QQᵀ (so E={y:(y−x)ᵀA⁻¹(y−x)≤1}) and cut normal a (here a = Aᵢ, the violated row), the exact (no-rounding) update is
  b = A a / √(aᵀ A a)        (step direction to boundary)
  x⁺ = x − (1/(n+1)) b
  A⁺ = (n²/(n²−1)) ( A − (2/(n+1)) b bᵀ )
This is the rank-1 update. Volume ratio (exact): vol E⁺/vol E = (n/(n+1))·(n²/(n²−1))^{(n−1)/2} < e^{−1/(2(n+1))}.

## The volume-shrink proof (Goemans, lived in reasoning)
Reduce by affine invariance to E = unit ball, cut x₁≥0. Claim the min-volume enclosing ellipsoid of {‖x‖≤1, x₁≥0} is
  E⁺ = { x : ((n+1)/n)²(x₁ − 1/(n+1))² + ((n²−1)/n²)Σ_{i≥2}xᵢ² ≤ 1 }.
Verify containment: for x in the half-ball, LHS = (2n+2)/n² · x₁(x₁−1) + 1/n² + (n²−1)/n²·Σxᵢ² ≤ 1/n² + (n²−1)/n² = 1 (since x₁(x₁−1)≤0 on [0,1] and Σ_{i}xᵢ²≤1). Semi-axes: a₁ = n/(n+1), a_{i≥2} = √(n²/(n²−1)) = n/√(n²−1). vol ratio = a₁·a₂^{n−1} = (n/(n+1))·(n²/(n²−1))^{(n−1)/2}. Take log, use ln(1+t)≤t: ln ratio = −1/(n+1)·... < −1/(2(n+1)). General cut and general E by pulling back/pushing forward under the affine map y=(B⁻¹)ᵀ(x−a), A=BᵀB.

## Termination / encoding-length bookkeeping (Goemans + Khachiyan)
- Start E₀ = ball of radius 2^L (Khachiyan: Q₀=diag(2^L); x₀=0). vol E₀ ≤ (2^L)ⁿ·Vₙ ⇒ log vol E₀ = O(nL).
- If feasible region has positive volume, it is ≥ 2^{−poly(L)} (full-dim polytope with integer data: vol ≥ 1/n!·(1/(2n c_max))ⁿ in the {0,1}ⁿ combinatorial setting; in general ≥ 2^{−O(nL)} by Cramer/det bounds). log vol P' = −O(nL).
- #iterations ≤ 2(n+1)·ln(vol E₀/vol P') = O(n²L). Khachiyan fixes M = 16n²L iterations.
- After M steps without finding σ<2^{−L} (his rule via Lemma 2), declare infeasible. Each iteration: O(n²) ops for the rank-1 update, O(nL)-bit precision.
- Rational/rounding care: square roots make entries irrational ⇒ round to p bits (GLS: p=5N; Khachiyan: tape with 23L+38nL bits, δ=2^{−8nL}); the deliberately-LARGER shrink factor (GLS use 2n²/(2n²+3) instead of n²/(n²−1); Khachiyan inserts 2^{1/(2n²)}) guarantees the rounded ellipsoid still contains the half-ellipsoid. This is the self-correction: naive exact-arithmetic ellipsoid would blow up bit-length unboundedly; rounding + slack fixes it.

## Antecedents (refs: lovasz_ellipsoid_chapter.pdf p.65, gls_1981.txt, boyd notes; web)
- Newman (1965) / Levin (1965): cutting-plane localization where the localization set is a POLYHEDRON; its #inequalities grows each iteration ⇒ per-iteration cost grows unboundedly. The wall ellipsoids fix: keep the localization set a fixed-complexity ellipsoid (n + n² numbers), constant work per step.
- Shor (1970a,b): subgradient method with SPACE DILATION (gradient projection with dilation of the space) for convex nondifferentiable optimization. "First explicit statement of the ellipsoid method as we know it today is due to Shor (1977)" (GLS book). In nonlinear language: a rank-one update algorithm, analogous to a variable-metric quasi-Newton method (Goffin 1984).
- Yudin & Nemirovski (1976a,b): observed Shor's algorithm answers an informational-complexity question of Levin (1965); gave an outline of the ellipsoid method; "modified method of central sections". (Khachiyan's ref [2], Ekonomika i mat. metody 12(2) 357, 1976.)
- Khachiyan (1979) [Dokl] adapted this convex-optimization machinery to LP FEASIBILITY with integer data, supplying the encoding-length analysis (Lemmas 1,2; rational arithmetic) that turns geometric convergence into a POLYNOMIAL-TIME decision procedure. Proofs in Khachiyan (1980); also Gács–Lovász (1979/1981).
- Open question of the time: simplex is exponential in worst case (Klee–Minty 1972); LP known to be in NP∩co-NP (LP duality / Farkas) but NOT known to be in P. "Is LP in P?" was a major open problem. Khachiyan resolved it: YES.

## The deep payoff (GLS 1981 Thm (3.1); Goemans Thm 3)
The ellipsoid algorithm needs only a SEPARATION ORACLE: given x, either assert x∈P or return a violated inequality aᵀx ≤ b (aᵀx > b). It never needs P listed explicitly. So a P with exponentially many facets (matchings polytope, perfect-graph stable-set polytope, submodular function minimization, …) is still optimized in poly time if separation is poly. GLS Theorem (3.1): for a class of (well-described) convex bodies, the (weak) optimization problem is polynomially solvable IFF the (weak) separation problem is. Feasibility→optimization by binary search on the objective value d (P' = P ∩ {cᵀx ≤ d+½}); O(log(n c_max)) feasibility queries.

## Design decisions → why (table)
- Enclose feasible set in an ellipsoid (not polyhedron): keeps localization-set description constant-size (n+n² numbers) so per-step work is O(n²), unlike Newman/Levin polyhedral cutting planes whose description grows.
- Cut THROUGH the center (central cut, normal = violated row): guarantees we discard ≥ half the ellipsoid's volume's worth of the region; the min-vol reenclosing ellipsoid then has vol ratio e^{−1/(2(n+1))} independent of geometry. Deep cuts possible if a value bound is known (Boyd) but central cut suffices for the poly bound.
- Step 1/(n+1) toward boundary, shrink n²/(n²−1): these are exactly the parameters of the minimum-volume ellipsoid covering a half-ball (derived, not chosen). Any other re-enclosure is bigger ⇒ slower or non-convergent.
- Start radius 2^L: Lemma 1 guarantees a solution lives in this ball if one exists. Bigger wastes iterations (log of vol ratio); smaller could miss solutions.
- Stop at M=16n²L (= O(n²·log(volE₀/volP))): the volume floor from integer data (Lemma 2 / det bounds) means a feasible region, if any, has vol ≥ 2^{−O(nL)}; once vol Eₖ drops below that, nonemptiness is decided.
- Round to finite precision + use a slightly LARGER shrink factor: exact arithmetic introduces √· ⇒ irrationals with growing bit-length; rounding controls bit-length, and the slack restores the containment guarantee the rounding would otherwise break. Without the slack the rounded ellipsoid can fail to contain the half-ellipsoid (a real wall).
- Integer coefficients + L: the whole polynomiality hinges on the 2^{−L} infeasibility gap (Lemma 2); without integrality there is no gap and no finite termination.

## Code grounding
Boyd P-form / H-form and Goemans A-form. Final code: feasibility oracle returns a violated row; rank-1 PSD update A⁺ = (n²/(n²−1))(A − (2/(n+1)) b bᵀ), b = Aa/√(aᵀAa); loop ≤ ⌈2(n+1)·(L·log... )⌉ ~ 16n²L; binary-search wrapper for optimization; separation-oracle abstraction so an implicit P (e.g. exponentially many constraints) plugs in.
