# Synthesis — Interior-Point Methods for Linear Programming

## Refs retrieved & read this run
- `refs/karmarkar-1984.txt` (full primary: Karmarkar, "A new polynomial-time algorithm for LP", Combinatorica 4(4) 1984, 373–395). Read in full: projective transformation T(a,a0), potential f(x)=Σln(c'x/x_j) [in canonical form n·ln(cᵀx) − Σln x_j], inscribed-sphere step, primal-dual combination (§5 Step 1), incremental inverse (§6), O(n^3.5 L).
- `refs/nemirovski-ipm-history.txt` (Nemirovski–Todd, Acta Numerica 2008). Read §1–§2.3 in full: history (Frisch 1955 log barrier, Fiacco–McCormick 1968 SUMT, Karmarkar 1984, Gill et al 1986 barrier equivalence, Renegar 1988 & Gonzaga 1989 path-following √-iteration, Megiddo 1989 / Monteiro–Adler 1989 / Kojima–Mizuno–Yoshise 1989 primal-dual), self-concordance def (2.3),(2.4), Dikin ellipsoid (2.5), Newton decrement λ, damped Newton (2.7)-(2.8), central path x*(t)=argmin tcᵀx+F(x), gap ϑ/t (2.12), path-following √ϑ complexity (2.14)-(2.15), conic primal-dual central path (2.20), Karmarkar potential as conic generalization φ(x)=ϑln(cᵀx)+F(x), Tanabe–Todd–Ye primal-dual potential (2.21).
- `refs/boyd_ch11_central_path.txt`, `refs/boyd_ch11_lp.txt`, `refs/boyd_ch10_newton.txt` (Boyd–Vandenberghe, Convex Optimization, ch.10–11). Log barrier φ=−Σln(−fᵢ), central path KKT (11.7), dual point λᵢ*(t)=−1/(t fᵢ), duality gap m/t (11.10),(11.11) KKT deformation −λᵢfᵢ=1/t, force-field interpretation, barrier method outer/inner loop.
- `refs/karmarkar-focm-2014.txt` (Karmarkar, "Towards a Broader View of Theory of Computing", FOCM 2014, arXiv 1412.3335). This is his LATER research program (continuum computing); only weak retrospective on the 1984 genesis — opens "Beginning with the projectively invariant method for linear programming…". NOT a strong self-account of the original discovery.

## Code (landing artifact)
`code/mpcSol.m`, `newtonDirection.m`, `stepSize.m`, `initialPoint.m` — Yiming Yan's Mehrotra predictor-corrector primal-dual IPM (canonical). Structure:
- initialPoint: least-norm x,s then shift into positive orthant (Mehrotra heuristic).
- main loop: residuals Rb=Ax−b, Rc=Aᵀy+s−c, Rxs=x∘s, μ=mean(x∘s).
- predictor (affine, σ=0): solve KKT/augmented system for (dx,dy,ds).
- σ = (μ_aff/μ)³ (Mehrotra adaptive centering).
- corrector: rhs Rxs += dx_aff∘ds_aff − σμe; reuse factorization.
- step: fraction-to-boundary η=0.9995.
Final code = NumPy port of this, grounded piece-for-piece.

## Self-account?
No strong first-person genesis account of the 1984 algorithm found. FOCM 2014 lecture is later-program, not retrospective-on-discovery. Recorded in notes/. Reconstruction is from primary + antecedents (Frisch/Fiacco–McCormick barrier, Dikin, Khachiyan ellipsoid, Newton, self-concordance, Gill et al equivalence).

## The pain point (research question)
Solve LP min cᵀx s.t. Ax=b, x≥0 in PROVABLE polynomial time AND fast in practice. Two unsatisfactory poles in 1983:
- Simplex (Dantzig 1947): walks vertices of polytope; superb in practice but Klee–Minty (1972) shows exponential worst case — not poly-time.
- Ellipsoid (Khachiyan 1979): first poly-time LP, O(n⁶L²) — but hopeless in practice (slow, accumulates round-off, always near worst case).
Goal: an algorithm that is BOTH poly-time AND practical, by moving through the INTERIOR (avoiding the combinatorial vertex structure entirely).

## Antecedents → gap each leaves
1. **Simplex / Dantzig** — vertex-following; gap: exponential worst case (Klee–Minty), and it's wedded to the boundary's combinatorics.
2. **Khachiyan ellipsoid** — poly-time via shrinking ellipsoids; gap: huge constants, n⁶, round-off accumulates, useless in practice.
3. **Frisch (1955) log barrier / Fiacco–McCormick (1968) SUMT** — replace inequalities x≥0 by −Σ ln xᵢ penalty, minimize tcᵀx − Σln xᵢ for increasing t (the barrier family). Gap: classical theory predicted SLOWDOWN as t→∞ (subproblems ill-conditioned near boundary); fell out of favor late 1960s–70s; NO polynomiality claimed.
4. **Newton's method** — quadratic local convergence for smooth convex min; gap: classical convergence region is "frame-dependent" (depends on condition number of Hessian and Lipschitz constant of Hessian — ad hoc Euclidean structure), so no clean global/poly statement for the barrier subproblems.
5. **Dikin (1967) affine scaling** — ellipsoid x:Σ(xᵢ−x̄ᵢ)²/x̄ᵢ²≤1 inscribed at current point, move in scaled steepest descent; the local rescaling idea. Gap: not proven poly-time.

## Design-decision → why table
- **Move through interior, not vertices**: vertices are the source of simplex's combinatorial blowup; the interior is smooth, amenable to continuous/Newton methods.
- **Logarithmic barrier −Σ ln xᵢ (not inverse 1/xᵢ, not quadratic penalty)**: log is the unique self-concordant building block; −ln x: R₊₊→R is a 1-self-concordant barrier *directly* (factor 2 in def chosen so it works without scaling). Barrier → ∞ at boundary keeps iterates strictly feasible. The gradient −1/xᵢ gives the central-path equations cleanly.
- **Central path x*(t)=argmin tcᵀx+φ(x)**: the minimizer of barrier family; characterized by ∇φ(x*(t)) = −tc, i.e. cᵀx − cᵀx*(t) ≤ m/t. Following it → optimum. WHY a path not single barrier solve: a single large t is ill-conditioned (the old SUMT failure); tracing the path with short Newton steps avoids it.
- **Duality gap = m/t exactly** (LP): λᵢ*(t)=1/(t·s? ) — for standard form x≥0, sᵢ=−1/(t·?), gives xᵢsᵢ=1/t per coordinate, gap = Σxᵢsᵢ = n/t. This is WHY the path works: it's a deformation of KKT (complementarity xᵢsᵢ=0 → xᵢsᵢ=μ=1/t).
- **Newton step on the barrier (KKT system)**: the central-path conditions are smooth nonlinear equations; Newton's method solves them, and for a self-concordant barrier the convergence is frame-independent → quadratic in λ once λ≤1/4.
- **Why short steps in t (t←(1+0.1/√ϑ)t)**: keeps the current iterate inside the Newton quadratic-convergence region of the next center; gives √ϑ-iteration complexity. WHY √ϑ not ϑ: the path-following barrier uses the half-derivative bound (2.4) → step factor 1/√ϑ; Karmarkar's potential reduction gives only ϑ.
- **Self-concordance |f'''|≤2(f'')^{3/2}**: the intrinsic property making Newton frame-independent; measure Hessian Lipschitzness in the local norm the Hessian itself defines. Why 3/2 & 1/2 powers: homogeneity in h. Why factor 2: makes −ln x a 1-scb without rescaling.
- **Karmarkar projective transformation x→D⁻¹x/(eᵀD⁻¹x)** with D=diag(current point): maps current interior point to the center of the simplex, where the inscribed/circumscribed ball ratio R/r=n−1 is controllable → guaranteed constant reduction per step. Why projective not affine: projective fixes simplex vertices, keeps the homogeneous structure, and makes the metric (cross-ratio) invariant; the potential transforms into a function of the same form.
- **Karmarkar potential f(x)=n·ln(cᵀx) − Σ ln xᵢ** (= ϑln(cᵀx)+F(x) with ϑ=n): linear objective isn't projective-invariant, ratios of linear functions are; the log-of-ratio potential IS transformed into same form by T, and constant reduction in it ⇒ geometric reduction in cᵀx. This is φ(x)=G(p(x)) under projective map p(x)=x/cᵀx; minimizing it = running to ∞ along the ray, i.e. cᵀx→0.
- **Inscribed-sphere optimization / step length αr, α=1/4**: at the center, optimize linear approx over a ball of radius αr; α<1 so linear approx of potential is accurate and round-off has margin. δ=1/8 reduction when α=1/4.
- **Primal-dual (Mehrotra predictor-corrector)**: solve P and D simultaneously; track xᵢsᵢ→μ. WHY primal-dual beats pure primal: symmetric, adaptive long steps, observable duality gap sᵀx as stopping test, far better in practice. Predictor (σ=0, affine) probes; σ=(μ_aff/μ)³ adaptively centers; corrector adds the 2nd-order dx∘ds term. Fraction-to-boundary η=0.9995 keeps strict positivity.
</content>
</invoke>
