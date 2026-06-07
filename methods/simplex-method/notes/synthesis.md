# Synthesis — Simplex Method (grounded in Dantzig's own account)

## Refs read in full
- `refs/dantzig-1987-origins-of-simplex-SOL-87-5.txt` (Stanford SOL 87-5, May 1987; also in *A History of Scientific Computing*, ACM 1990) — Dantzig's OWN retrospective. THE backbone.
- `refs/dantzig-1982-reminiscences-springer.txt` (Reminiscences About the Origins of LP, 1983 Springer volume; ORL 1982) — the WWII planning / Leontief / von Neumann / homework-thesis narrative.
- `code/scipy_linprog_simplex.py` — canonical full-tableau implementation (Dantzig pricing, min-ratio test, pivot, Bland's rule, Phase 1/2).

## The discovery path IN DANTZIG'S WORDS (the load-bearing facts)

1. **Origin = WWII Air Force "programming."** Five war-time years as a practical *program planner* using desk calculators. 1946: Mathematical Advisor to the USAF Comptroller. Colleagues Hitchcock & M. Wood challenged him to *mechanize the planning process* — compute a time-staged deployment/training/supply program faster. "Programming" = the military word for a proposed schedule of training, supply, deployment. (This is why it's "linear PROGRAMMING".)

2. **Model from generalizing Leontief.** He admired Leontief's 1932 Interindustry Input-Output Model (simple matrix structure, one-to-one process↔item). He saw it had to be generalized: needed (a) dynamic (time-staged) not steady-state, (b) MANY ALTERNATIVE activities per item, not one-to-one, (c) large scale (hundreds of items/activities), (d) computable. Result: "a time-staged, dynamic linear program with a staircase matrix structure."

3. **The novel move: make the objective EXPLICIT.** Initially there was NO objective function — planners used a pile of ad-hoc "ground rules"/edicts instead of a goal. The assignment-problem example (70 men/70 jobs → 70! ≈ >10^100 solutions; even Earths-full of nanosecond computers from the Big Bang couldn't enumerate) shows WHY you can't enumerate and WHY you need both an explicit objective and a clever algorithm. By mid-1947 he decided "the objective had to be made explicit." Axiomatized: items vs activities; activities consume/produce items in fixed non-negative proportions. "The resulting mathematical system to be solved was the minimization of a linear form subject to linear equations and inequalities. The use of the linear form as the objective function to be extremized was the novel feature."

4. **Problem statement (his eq 1):** min z, x ≥ 0 s.t. Ax = b, cx = z(min). A is m×n.

5. **The obvious algorithm, REJECTED on intuition.** Summer 1947, first idea "that would occur to any trained mathematician": step-by-step descent (w.r.t. objective) along EDGES of the convex polyhedral set, vertex to adjacent vertex. He "rejected this algorithm outright on intuitive grounds — it had to be inefficient because it proposed to solve the problem by wandering along some path of outside edges until the optimal vertex was reached." So he went looking for methods through the INTERIOR instead. KEY: the famous vertex-walking idea was his FIRST thought and he initially THREW IT AWAY.

6. **Antecedents he didn't know about.** Four/five isolated prior papers on special cases, all proposing edge-descent (Fourier 1824, de la Vallée Poussin 1911, Hitchcock 1941; Monge 1781; Kantorovich 1939 was the exception — different). Zero influence on each other, unknown to him. He stresses: ANY well-trained mathematician in the ROW geometry would immediately propose vertex-descent (as Fourier/Poussin/Hitchcock did) — and would reject it as a "very stupid idea" once forced to consider it practically.

7. **The thesis / column geometry — the real insight.** His Berkeley PhD (advisor Neyman) was the two famous unsolved problems he "mistakenly thought was a homework assignment and solved" (one became the Dantzig–Wald paper on the Neyman–Pearson Lemma). In today's terms: existence of Lagrange multipliers (dual variables) for a semi-infinite LP over a continuum of variables each in [0,1] under Lebesgue-integral constraints, with a linear objective. **The geometry he used was the dimension of the COLUMNS not the rows.** Because the number of discrete points n could be infinite, it was more convenient to work in the finite (m+1)-dimensional space of a column. "This column geometry gave me the insight that made me believe the Simplex Method would be a very efficient solution technique." He proposed it summer 1947 and "by good luck it worked."

8. **Column geometry, concretely (SOL 87-5).** Drop the leading "1" coordinate (from convexity constraint Σxⱼ=1). Each column (A.ⱼ, cⱼ) → a point (yⱼ, zⱼ) in R^m. Assign weights xⱼ ≥ 0 to the n points so the "center of gravity" lies on the vertical "requirement line" (b, z), with z as small as possible.
   - **A step:** current basis = m points forming an (m−1)-simplex; requirement line pierces the simplex's solution plane z = πy + π₀ at (b, zₜ). Plane found by solving πA.ⱼᵢ + π₀ = cⱼᵢ for the m basic columns.
   - **Entering (pricing):** find the point most BELOW the solution plane: s = argmin_j [cⱼ − (πA.ⱼ + π₀)]. If that min ≥ 0 → STOP, optimal. (This is the reduced cost / "pricing out".)
   - **Leaving:** form the new m-simplex (tetrahedron) as convex combo of new point s and the old (m−1)-simplex; requirement line now pierces it at (b, zₜ₊₁) with zₜ₊₁ < zₜ; the face containing (b,zₜ₊₁) is the new (m−1)-simplex; "operationally point s replaces some r." (The leaving column.) → min ratio test.
   - **2-D picture (Fig 2):** the "underbelly" of the convex hull; like a secant method — given a slope it's cheap to find the support point of that slope, but you can't directly compute f(b).

9. **Why it looked efficient (his conjecture + convergence theorem).** "In practical applications most points lie ABOVE the underbelly of their convex hull, so very few j are extreme points of the underbelly; the algorithm only chooses from those rare extreme points → I conjectured ~m steps in practice."
   - **Convergence theorem (SOL 87-5):** Assume basic variable values satisfy x ≥ θ > 0 at every iteration. With Δₜ = zₜ − z*, then Δₜ ≤ (1−θₜ)Δₜ₋₁ ≤ e^{−θₜ}Δₜ₋₁, so Δₜ/Δ₀ ≤ ∏(1−θᵢ) ≤ e^{−Σθᵢ}. Derivation: Δₜ₋₁ = zₜ₋₁ − z* = Σ(−c̄ⱼ)xⱼ* ≤ (−c̄ₛ)Σxⱼ* = (−c̄ₛ); and zₜ₋₁ − zₜ = (−c̄ₛ)θₜ ≥ Δₜ₋₁·θₜ; rearrange → Δₜ ≤ (1−θₜ)Δₜ₋₁.
   - **Corollary:** if θ averages 1/m, an e^{−k}-fold decrease takes < km iterations; a 1000-fold decrease in < 7m iterations (e^{−7}<.001). "It was considerations such as these that led me back in 1947 to believe the simplex method would be very efficient."

10. **The punchline he himself states.** He developed a variant WITHOUT the convexity constraint, had the Bureau of Standards test it on Stigler's nutrition (diet) problem in fall 1947. "I soon observed that what appeared in the column geometry to be a new algorithm was, in the row geometry, the vertex-descending algorithm that I had rejected earlier." → THE SAME ALGORITHM he threw away on intuition; the column geometry is what convinced him it was fast. His 3 contributions: (1) independently proposing the algorithm, (2) initiating the software, (3) observing via column geometry that walking the OUTSIDE of the polyhedron is actually efficient, contrary to intuition.

11. **von Neumann / duality (Oct 3 1947).** After proposing simplex but before realizing how efficient, visited von Neumann at IAS. "Get to the point." Dantzig slapped the geometric+algebraic problem on the board in under a minute; vN: "Oh that!" and lectured 1.5 hrs on LP theory — conjecturing the LP problem ≡ the theory of games he'd just finished with Morgenstern. Thus Dantzig learned Farkas' Lemma and DUALITY for the first time. (Mini-Max theorem, vN 1928, forerunner of LP Duality.)

12. **Degeneracy story.** Koopmans (editing the 1949/1951 proceedings) asked him to drop the non-degeneracy assumption. Dantzig thought degeneracy had probability zero ("what's the prob of four planes in 3-space meeting at a point?") — but EVERY practical Air Force problem turned out degenerate. "Degeneracy couldn't happen but it did. It was the rule not the exception." Fix: perturbation of the RHS → later lexicographic (Wolfe/Orden/Dantzig); Bland's rule is the modern anti-cycling pivot rule.

13. **Naming.** "Programming in a Linear Structure" → Koopmans on the Santa Monica beach (summer 1948): "Why not shorten to Linear Programming?" "Simplex Method" from Motzkin: the column-geometry move is "a movement from one simplex to a neighboring one." "Primal" coined ~1954 by Tobias Dantzig (George's father) as the Latin antonym of dual.

## The standard tableau algorithm (canonical impl, scipy `_linprog_simplex.py`)
- Standard form: min cᵀx s.t. Ax = b (b≥0), x ≥ 0. Add slacks for ≤ rows; artificials + Phase 1 for = / ≥ rows.
- Tableau rows = constraints + objective row of reduced costs; basis = list of basic column indices.
- **Pricing (`_pivot_col`):** entering col = most negative reduced cost in objective row (Dantzig rule); none negative → optimal. Bland: first negative (anti-cycling).
- **Ratio test (`_pivot_row`):** among rows with positive pivot-col entry, min of b/aᵢ; none positive → unbounded. Ties → Bland (lowest basic index).
- **Pivot (`_apply_pivot`):** divide pivot row by pivot value; eliminate pivot col from all other rows (incl objective). basis[pivrow] = pivcol.
- Phase 1 minimizes sum of artificials to get an initial BFS; Phase 2 optimizes real objective.

## Design decisions → why
- **Linear objective form (not ground rules):** ground rules confuse means with ends and leave astronomically many feasible solutions; an explicit linear objective lets you compare and pick the best. (Dantzig contribution #2.)
- **Equalities Ax=b standard form via slacks:** inequalities ↔ equalities + nonneg slack; uniform algebra.
- **Vertex optimum:** linear objective over a polyhedron attains its optimum at a vertex (an extreme point / basic feasible solution) — so search vertices, not the continuum.
- **Move along edges (pivoting) not interior:** at a vertex, the reduced costs tell you which edge decreases the objective; follow it to the adjacent vertex. The whole worry was that this is a boundary walk and could be long.
- **Dantzig pricing (most negative reduced cost):** greedy steepest decrease per unit of entering variable; cheap, and empirically ~m steps. Alternatives: Bland (first negative) trades speed for guaranteed termination.
- **Min ratio test:** the entering variable rises until the first basic variable hits 0; that one leaves. Going further violates x≥0. Ratio = b/(positive pivot entries) only — negative/zero entries impose no upper bound.
- **Bland's rule for ties / anti-cycling:** degeneracy (the "rule not the exception") can cause cycling; choosing lowest index for both entering and leaving provably terminates.
- **Sparsity & near-triangular basis (SOL 87-5):** practical bases are <0.2% dense and rearrangeable to near-triangular → keep B as LU product of elementary matrices, cheap to solve By=A.ⱼ. This is why simplex scales (his "good luck" factor #3). Pricing is cheap because columns have ~4–5 nonzeros.

## In-frame anchors (the narrator IS Dantzig, summer 1947)
- I'm an Air Force program planner; planners use ad-hoc ground rules; I want to mechanize.
- Generalize Leontief; make the objective a linear form — that's the new thing.
- The obvious vertex-walk feels like a stupid boundary wander; I almost discard it.
- The column geometry from my thesis makes the same walk look fast (few extreme points of the underbelly; ~m steps; the e^{−Σθ} convergence bound).
- Land on the tableau: pricing → ratio test → pivot; Phase 1 for a start; Bland for degeneracy.
- NO hindsight: no Khachiyan/Karmarkar polynomial-time, no Klee-Minty worst case (those are posterior to the discovery), no "this paper".
