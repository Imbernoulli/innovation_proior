# NSGA-II synthesis (from primary text + pymoo + ancestor research)

## Sources actually read this run
- PRIMARY: Deb, Pratap, Agarwal, Meyarivan, "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," IEEE TEC 6(2):182–197, 2002 — refs/nsga2_deb.pdf (ugr mirror; full 16-page IEEE article, pages 1-5 read text + visual). Contains: 3 criticisms, naive O(MN^3) argument, fast-non-dominated-sort box (S_p, n_p), crowding-distance-assignment box (cuboid perimeter), crowded-comparison ≺_n partial order, main loop box, per-iteration complexity table, O(N^2) space.
- SECONDARY: Seshadri "A Fast Elitist Multiobjective GA: NSGA-II" — refs/seshadri_nsga2.pdf. Reproduces fast sort, crowding distance, ≺_n, SBX (eta_c), polynomial mutation (eta_m), R_t merge/elitism. Cross-checks primary.
- CODE: pymoo (anyoptimization) cloned to code/pymoo. Read: algorithms/moo/nsga2.py (binary_tournament, NSGA2 defaults SBX eta=15 prob=0.9, PM eta=20, pop=100), operators/survival/rank_and_crowding/classes.py (RankAndCrowding._do), .../metrics.py (calc_crowding_distance), util/nds/fast_non_dominated_sort.py, util/dominator.py.
- Ancestor research via WebSearch: VEGA (Schaffer 1985), MOGA (Fonseca & Fleming 1993), NSGA (Srinivas & Deb 1994), SPEA (Zitzler & Thiele 1999), PAES (Knowles & Corne), NPGA (Horn et al.), Rudolph elitist GA. Complexity claims cross-checked.

## The pain (research question)
Multiple conflicting objectives → a *set* of Pareto-optimal trade-offs, not one optimum. Classical scalarization (weighted sum etc.) converts to single-objective, emphasizes ONE point per run, must be re-run many times, and on non-convex fronts a weighted sum can never reach concave regions. EAs work on a population → can return the whole front in ONE run. Goal: (a) converge to the true Pareto front, (b) spread solutions uniformly across it.

## Pareto definitions (pre-method, established)
Dominance: x dominates y (x ≺ y) iff x is no worse in all objectives AND strictly better in at least one. Pareto-optimal set = nondominated set over the whole feasible space. Nondominated front = the boundary.

## Ancestors and their exact gaps
- **VEGA (Schaffer 1985):** split population into M subpopulations, each selected by ONE objective, then shuffle+crossover. Equivalent to a weighted-sum with each subpop pulling one direction → "speciation"/bias to objective extremes; middle of front under-sampled. No notion of dominance.
- **MOGA (Fonseca & Fleming 1993):** rank(i) = 1 + (# solutions dominating i). Fitness from rank. Diversity via fitness sharing + mating restriction. Introduced dominance ranking but kept sharing → σ_share dependence.
- **NSGA (Srinivas & Deb 1994):** the direct ancestor. (1) sort population into nondomination fronts; (2) dummy fitness by front; (3) within a front, *fitness sharing* with σ_share to spread. THREE criticisms (primary, abstract+intro): (i) sorting is O(MN^3) → expensive for large N; (ii) NON-elitist → good solutions can be lost across generations; (iii) needs σ_share — performance hinges on its value, set by user/guidelines [4].
- **SPEA (Zitzler & Thiele 1999):** external archive of all nondominated found so far; strength = #dominated; fitness = sum of strengths of archive members dominating you; deterministic clustering for diversity. Elitist, but archive bookkeeping; clustering. Primary says naive impl O(N^3), with bookkeeping O(N^2). A baseline NSGA-II is compared against.
- **PAES (Knowles & Corne):** (1+1)-ES with adaptive grid archive; worst case O(aMN) with archive a; overall O(MN^2). Diversity via grid.
- **Rudolph elitist GA:** combine parent+offspring nondominated; proved convergence but NO explicit diversity mechanism. (This is the seed of the R_t = P_t ∪ Q_t idea but lacks spread.)

## Derivation chain (reasoning.md must re-derive in order)
1. Scalarization loses the trade-off surface; non-convex fronts unreachable by weighted sum. Want whole front in one run → population + dominance.
2. To rank a population by dominance need nondominated SORTING. Naive: to find front-1, compare each of N solutions with all others (M-objective dominance check = O(M)) → O(MN) per solution → O(MN^2) for one front. Peel it off, repeat. Worst case N fronts of 1 each → O(MN^3). Storage O(N). This is exactly NSGA's cost → criticism (i).
3. FAST NONDOMINATED SORT. Key: do all pairwise comparisons ONCE. For each p compute n_p (# that dominate p) and S_p (set p dominates) — O(MN^2) total, O(N^2) storage. Front 1 = {p: n_p=0}. To get next front, for each p in current front, for each q in S_p decrement n_q; when n_q hits 0, q is in next front. Complexity argument (two ways): (a) finding each of ≤N fronts is the cost; (b) tighter: the front-building loop body runs: outer (each p ∈ F_i) executes exactly N times total (each individual in one front), inner (each q ∈ S_p) at most N−1 times, each domination check ≤M comparisons in the INITIAL pass → O(MN^2). The decrement phase is O(N^2). So overall O(MN^2). Trade: storage O(N) → O(N^2).
4. ELITISM. Combine R_t = P_t ∪ Q_t (size 2N). Sort R_t. Fill P_{t+1} front by front. Best fronts (incl. all parents' best) always survive → elitism for free. No external archive (unlike SPEA). (Rudolph had the merge idea but no diversity.)
5. DIVERSITY WITHOUT A PARAMETER. Sharing needs σ_share (criticism iii) AND sharing is O(N^2) per front (every-pair comparison). Replace with **crowding distance**: per front, per objective m, sort by f_m; boundary points get ∞ (always kept, preserve extent); interior point i gets += (f_m of i+1 − f_m of i−1)/(f_m^max − f_m^min). Sum over m. = normalized perimeter of the cuboid whose sides are the gaps to the two nearest neighbors. Larger ⇒ more isolated ⇒ favored. Normalization makes objectives commensurable. NO parameter. Cost: M sorts of ≤N → O(MN log N), beating sharing's O(N^2) AND removing σ_share.
6. CROWDED-COMPARISON OPERATOR ≺_n: i ≺_n j iff (i_rank < j_rank) OR (i_rank = j_rank AND i_distance > j_distance). Lexicographic: rank first (push toward front), crowding-distance tie-break (push toward sparse regions). Used in BOTH binary tournament selection AND the population reduction (sort the splitting last front by ≺_n, take the best N−|P_{t+1}|).
7. MAIN LOOP (per-iteration complexity table): nondominated sort O(M(2N)^2); crowding assignment O(M(2N)log(2N)); sort on ≺_n O(2N log 2N). Overall O(MN^2), governed by the sort. Space O(N^2). Early-stop the sort once enough fronts to fill N.
8. OPERATORS (real-coded): SBX crossover (distribution index η_c; pymoo eta=15) + polynomial mutation (η_m; pymoo eta=20). Constraint handling: constrained-domination (feasible beats infeasible; among infeasible, smaller constraint violation wins) — pymoo's binary_tournament compares CV first.

## pymoo correspondence (final code grounded here)
- fast_non_dominated_sort.py: is_dominating = S_p list-of-lists, n_dominated = n_p, peel fronts by decrement. EXACT.
- calc_crowding_distance: argsort per objective, dist to last/next, divide by (max−min) norm, boundaries ∞ via ±inf padding, sum over obj / n_obj.
- RankAndCrowding._do: nds.do with n_stop_if_ranked (early stop), fill survivors front by front, on splitting front compute crowding and randomized_argsort descending, drop n_remove. set rank+crowding on individuals.
- binary_tournament: feasibility/CV first, then dom (or rank), then crowding (larger better).
- NSGA2 defaults: pop_size=100, SBX(eta=15,prob=0.9), PM(eta=20), RankAndCrowding survival, comp_by_dom_and_crowding tournament.

## Evaluation settings (pre-method, context only — NO outcomes)
Test problems from literature: Schaffer SCH, Fonseca FON, Poloni POL, Kursawe KUR, and ZDT1–ZDT6 (Zitzler-Deb-Thiele, two-objective, convex/nonconvex/discontinuous/multimodal/nonuniform). DTLZ (Deb-Thiele-Laumanns-Zitzler) scalable many-objective. Metrics that existed: convergence metric γ (mean distance to true front), diversity/spread metric Δ. Compared against PAES, SPEA. Real-coded with SBX + polynomial mutation; or binary with single-point crossover + bitwise mutation.

## Design-decision → why (no holes)
- Population not single point → whole front in one run (single-obj EAs lose it).
- n_p + S_p instead of repeated scanning → amortize all comparisons to one O(MN^2) pass; the decrement scheme reuses S_p so no re-comparison. Cost: O(N^2) memory.
- R_t = P_t∪Q_t merge → elitism without an external archive; parents and children compete on equal footing.
- Crowding distance vs sharing → removes σ_share (param-less), cheaper (O(MN log N) vs O(N^2)), and as a *perimeter* it directly measures local emptiness.
- Normalize each objective by (f^max−f^min) → objectives with different scales would otherwise dominate the distance.
- Boundary = ∞ → never discard the extremes; preserves the front's extent/range.
- Crowded-comparison lexicographic (rank then distance) → convergence pressure must dominate diversity pressure; only break rank ties by spread.
- Use ≺_n in BOTH selection and reduction → consistent pressure toward converged+spread solutions.
- Early-stop sort once N filled → don't sort fronts you'll discard.
- SBX η_c / poly-mut η_m → real-coded spread control; pymoo eta=15/20.
- Constrained domination → no penalty parameter; feasibility is just a pre-emptive layer over dominance.
