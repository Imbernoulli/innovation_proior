# Synthesis — Gilmore–Gomory column generation for cutting stock

## Sources retrieved & read (this run)
1. PRIMARY: Gilmore & Gomory, "A Linear Programming Approach to the Cutting-Stock Problem," Operations Research 9:6 (1961) 849–859 — full text read (refs/gilmore-gomory-1961.pdf). Part II: OR 11 (1963) 863–888 (referenced; the 1961 paper is the load-bearing one re-derived here).
2. BACKGROUND: the integer/LP model, simplex pricing/reduced cost, LP duality, knapsack, Eisemann "The Trim Problem" (1957) framing, Ford–Fulkerson (1958, ref 2 — shortest-path column generation precursor), Dantzig knapsack DP (ref 4), Dantzig–Wolfe decomposition (1960).
3. EXPLAINER: Desrosiers & Lübbecke, "A Primer in Column Generation" (in Column Generation, Desaulniers–Desrosiers–Solomon eds., Kluwer 2005), §4 cutting stock — full text read (refs/primer-column-generation.pdf).
4. CODE: scipy.optimize.linprog implementation (Leite, TDS) — RMP + dual marginals + integer knapsack pricing + loop; and OR-Tools pywraplp implementation (mingcaixiao GitHub, code/ortools_cutstock.py).

## Problem (in-frame)
Stock rolls/lengths of width L (cost c per length, or count rolls). Order: N_i pieces of length ℓ_i, i=1..m. Fill the order at minimum cost (min number/cost of stock pieces). A "cutting pattern"/"activity" = a way to cut one stock length into ordered pieces. Pattern r has a_{ir} = number of pieces of length ℓ_i it yields, feasible iff Σ_i a_{ir} ℓ_i ≤ L.

## The natural model and why it's intractable
Assign x_j = number of times pattern j is run. Min Σ c_j x_j s.t. Σ_j a_{ij} x_j ≥ N_i, x_j ≥ 0 integer. Two troubles (1961 paper, p.850):
- n (number of patterns) is astronomically large — every feasible knapsack-packing of L is a column. Can't even write the LP.
- integrality. The paper DROPS integrality (solves LP relaxation), rounds/branches at the end. Their purpose is the *first* factor (huge n). Note: with integrality dropped, slack variables can be dropped (any over-fulfilling solution has an equal-cost solution with no slack) — paper proves this p.851; but they keep slacks because a min solution may then use < m activities, helping the final rounding.

## The key idea — delayed column generation (G&G's own words)
"When, in the simplex method, we reach the stage of 'pricing out'... instead of looking over a vast existing collection of columns to pick out a useful one, we simply create a useful column by solving an auxiliary problem." (p.849). Same idea as implicit in Ford–Fulkerson (shortest path) and a Dantzig–Wolfe LP; here the auxiliary problem is of knapsack type.

## Reduced-cost derivation (paper p.852, exact)
Basis B = patterns P_1..P_m (m×m, P_i^T = (a_{1i},...,a_{mi})), costs c_1..c_m. New candidate activity P=(a_1,...,a_m) cutting from stock length L at cost c. Solve A·U = P (U = B^{-1}P). The new activity improves the basic solution iff C·U > c, where C = (c_1,...,c_m). If row vector C·A^{-1} = (b_1,...,b_m) (these are the simplex multipliers / dual prices), then a profitable activity from L exists iff ∃ nonneg integers a_1..a_m with:
  (6) L ≥ ℓ_1 a_1 + ... + ℓ_m a_m   (knapsack feasibility)
  (7) b_1 a_1 + ... + b_m a_m > c    (improvement / negative reduced cost)
So: max Σ b_i a_i s.t. Σ ℓ_i a_i ≤ L, a_i ≥ 0 integer; if the max > c then column is profitable, else (for all stock lengths) current LP solution is optimal.

NOTE on sign conventions: the 1961 paper is a *maximization-of-improvement* pricing using simplex multipliers b_i = (C A^{-1})_i, condition CU>c. The primer (modern min form, cost-1-per-roll, "≥ n_i" demand): min Σλ_r, dual DCS max Σ n_i π_i s.t. Σ_i a_{ir} π_i ≤ 1, π_i ≥ 0. Reduced cost of pattern r = 1 − Σ_i π_i x_i. Pricing: max Σ_i π_i x_i s.t. Σ ℓ_i x_i ≤ L, x_i ∈ Z+. If max > 1 → reduced cost < 0 → add column; else LP optimal. Both are the same knapsack pricing; the modern min-rolls form (cost c_j=1) is what the code uses. I will derive in the modern min-number-of-rolls form (cleaner) but keep G&G's general cost-c_j version visible.

## Knapsack pricing solved by DP (paper p.853)
Bounded/unbounded knapsack. F_s(x) = max of b_1 a_1+...+b_s a_s s.t. x ≥ ℓ_1 a_1+...+ℓ_s a_s. Recursion:
  F_{s+1}(x) = max_{0≤r≤[x/ℓ_{s+1}]} { r b_{s+1} + F_s(x − r ℓ_{s+1}) }.
One DP pass over the largest stock length L_1 simultaneously gives F_m(L_2),...,F_m(L_k). Also an ad-hoc greedy (Dantzig): sort by b_i/ℓ_i descending, fill greedily — try this first, only fall to DP when greedy gives no profitable pattern for any stock length.

## Algorithm loop (paper p.853–855 routine; primer §4)
1. Initialize: m starting patterns — for each i pick a stock length L_j ≥ ℓ_i, pattern = [L_j/ℓ_i] copies of ℓ_i. (Diagonal "homogeneous" patterns: pattern i cuts only item i, a_{ii} = floor(L/ℓ_i).) Gives a feasible basis (B is upper-triangular-ish, invertible).
2. Solve RMP (LP relaxation over current patterns), get duals π (= simplex multipliers b_i).
3. Pricing: solve knapsack max Σ π_i x_i s.t. Σ ℓ_i x_i ≤ L. If objective ≤ 1 (resp. ≤ c) for all stock lengths → LP optimal, stop. Else add the new pattern column, re-solve RMP. Repeat.
4. Round/branch the fractional LP solution to integers.

## Worked example (paper p.856-858) — REAL numbers from primary text
Order: 20 of length 2, 10 of length 3, 20 of length 4. Stock lengths 5,6,9 with costs 6,7,10. Optimal LP cost = 170: cut 10 pieces of stock-6 into (one 4 + one 2) each, and 10 pieces of stock-9 into (one 2 + one 3 + one 4) each. (Integers came out "fortuitously.") This is the only result number I may use, and only as a derived/lived computation, not a benchmark claim.

## Code structure (for scaffold ↔ final correspondence)
- solve_master(patterns, demand) -> (objective, duals): build LP min Σ x_j s.t. Σ a_{ij} x_j ≥ d_i, x≥0; return primal value + dual marginals on demand constraints.
- solve_pricing(L, lengths, duals) -> (pattern, reduced_cost): integer knapsack max Σ duals_i a_i s.t. Σ ℓ_i a_i ≤ L; reduced cost = 1 − value.
- column_generation loop: init diagonal patterns → loop{ solve_master → solve_pricing → if reduced_cost ≥ −eps break else append column }.
- final integer rounding / round-up heuristic.
scipy version: linprog(c, A_ub=-A, b_ub=-d), duals = -sol.ineqlin.marginals, knapsack via linprog(-duals, A_ub=[w], b_ub=[W], integrality=1). OR-Tools version: pywraplp CLP for master (RowConstraint(demand, inf)), reads dual via constraint.dual_value(), pricing knapsack via SCIP/CBC integer solver.

## Design decisions → why
- Drop integrality first: the hard part they target is huge n, not integrality; LP relaxation gives a strong bound and near-integer answer for large N (rounding error small in %). Integers handled by rounding/branching after.
- Column = pattern (Dantzig–Wolfe / extensive reformulation) NOT item-indexed (Kantorovich compact form): the pattern LP relaxation is far tighter than Kantorovich's (whose LP bound is just Σ ℓ_i n_i / L and is symmetric/weak). The strength is *why* you reformulate into patterns.
- Don't enumerate columns; generate on demand: simplex only needs the most-negative-reduced-cost column each iteration → solve a subproblem to *produce* it. This is the whole idea.
- Pricing is exactly knapsack: reduced cost minimization over feasible patterns = "pack value (dual prices) into width L" = knapsack. Integer because a pattern must be an integer packing.
- Demand "≥" not "=": makes duals π_i ≥ 0 (dual-optimal inequalities), accelerates, and lets slacks drop. G&G proved the equal-cost slack-free solution exists.
- Greedy-then-DP for pricing: greedy (ratio b_i/ℓ_i) is cheap and often suffices; DP is exact fallback and one pass covers all stock lengths.
- Initial diagonal patterns: trivially feasible (each item from its own roll), invertible basis to start simplex.
- Stop when max reduced-cost ≥ 0: standard simplex optimality (no improving column) ⇒ LP relaxation optimal even though we never enumerated all columns.

## In-frame cautions
- Never name "the paper"/authors as artifact. Method name "Gilmore–Gomory column generation"/"delayed column generation" OK in answer.md as the thing being built.
- Only result number allowed: the 170-cost worked example, lived as a computation. No fabricated waste/efficiency %.
- Ancestors cited by author/year OK: Eisemann 1957 (trim problem), Ford–Fulkerson 1958, Dantzig (knapsack DP), Dantzig–Wolfe 1960, Kantorovich 1939/1960 (as a *prior* compact model to react to — careful: Kantorovich predates, so it's background/baseline, fine to cite).
