# Synthesis — Selinger-style cost-based join-order optimization (System R)

## Three sources (all read this run)
1. PRIMARY: Selinger, Astrahan, Chamberlin, Lorie, Price (1979) "Access Path Selection in a Relational
   Database Management System", SIGMOD. refs/selinger1979.pdf (16pp, read in full incl. all figures/tables).
2. THIRD-PARTY EXPLAINER: MIT 6.830 (Madden) "Selinger Optimizer" lecture, refs/mit_selinger.pdf (read in full).
   Clean `optjoin` DP recurrence, cache table, O(n·m·2^n) complexity, interesting-orders worked example.
3. BACKGROUND + canonical code: DuckDB join_order optimizer source (code/duckdb_cost_model.cpp,
   duckdb_plan_enumerator.cpp, duckdb_cardinality_estimator.cpp). Modern descendant: additive cost recurrence
   cost = join_card + left.cost + right.cost; DP table `plans[set]` updated when new_cost < old_cost; DPhyp
   exact enumerator + greedy fallback. Plus DuckDB internals explainer (alibabacloud blog) for cardinality
   |t1⋈t2| ≈ |t1|·|t2| / max(domain(t1.A), domain(t2.A)).

## Pain point / research question
SQL is non-procedural: user states WHAT not HOW. The system must pick, for each table, an access path (which
index or a scan) AND, for joins, the ORDER of joining and the JOIN METHOD — to minimize total resource cost.
Join order matters enormously: the *result* of joining n relations is the same regardless of order, but the
*cost* differs by orders of magnitude (intermediate cardinalities explode or stay small). Number of orders is
combinatorial: n! orderings of relations, and for each ordering (n-1)! tree shapes → for a 20-way join,
20!·19! ≈ 2.9×10^35. Cannot enumerate.

## Cost model (primary, §4 + Tables 1,2)
COST = PAGE FETCHES + W * (RSI CALLS).   W = weight, I/O vs CPU. RSI CALLS ≈ tuples returned ≈ CPU proxy.
Selectivity factor F per boolean factor (Table 1):
 - column = value: F = 1/ICARD(index) if indexed (uniform-distribution assumption), else 1/10.
 - col1 = col2: F = 1/MAX(ICARD1, ICARD2) if both indexed; 1/ICARD(i) if one indexed; else 1/10.
 - col > value: F = (high - value)/(high - low) if arithmetic & value known; else 1/3.
 - col BETWEEN v1 AND v2: F = (v2 - v1)/(high - low); else 1/4.
 - p1 OR p2: F = F1 + F2 - F1·F2.   p1 AND p2: F = F1·F2 (independence).   NOT p: 1 - F.
 - column IN list: F = (#items)·F(=value), capped at 1/2.
QCARD = (product of relation cardinalities in FROM) × (product of all selectivity factors). Output card.
RSICARD = (product of relation cards) × (product of *sargable* boolean factor selectivities). Tuples crossing
RSI interface (those not filtered inside RSS by search arguments).

Single-relation access-path costs (Table 2), in pages:
 - unique index matching equal predicate: 1 + 1 + W
 - clustered index I matching booleans: F·(NINDX(I)+TCARD) + W·RSICARD
 - non-clustered index I matching booleans: F·(NINDX(I)+NCARD) + W·RSICARD  [or F·(NINDX+TCARD) if fits buffer]
 - clustered, no match: (NINDX+TCARD) + W·RSICARD
 - non-clustered, no match: (NINDX+NCARD) + W·RSICARD  [or NINDX+TCARD if fits]
 - segment (sequential) scan: TCARD/P + W·RSICARD
Statistics maintained: NCARD(T), TCARD(T), P(T)=TCARD/(#nonempty pages); per index ICARD(I) distinct keys,
NINDX(I) pages.

## Join methods (primary §5)
- Nested-loop join: scan outer, for each outer tuple scan inner for matching tuples (any access path on each).
  C-nested-loop-join(path1,path2) = C-outer(path1) + N · C-inner(path2),
  N = cardinality of outer composite so far = (product of cards of relations joined so far) × (product of
  selectivity factors of all applicable predicates).
- Merge (sort-merge) scan join: both inputs in join-column order; scan in lockstep, remember matching groups.
  C-merge(path1,path2) = C-outer(path1) + N · C-inner(path2). For a sorted temp inner:
  C-inner(sorted list) = TEMPPAGES/N + W·RSICARD (each inner page fetched once). Plus C-sort(path) for the
  sort. Sorting may require materializing the composite to a temp relation.
  Key observation in paper: nested-loop and merge cost FORMULAS are essentially the same; merge wins when the
  inner scan cost is much smaller (inner already sorted / clustered on join column → fewer page fetches).

## The DP insight (primary §5 "Computation of costs"; MIT optjoin)
Restrict to left-deep trees only: at each step a *base* relation (the inner) is joined to the running
composite (the outer). This drops (n-1)! tree shapes; left-deep also pipelines — the composite need not be
materialized unless a sort is required.
Principle of optimality: "once the first k relations are joined, the method to join the composite to the
(k+1)-st relation is independent of the order of joining the first k; the applicable predicates are the same,
the set of interesting orderings is the same, the possible join methods are the same." → the best plan for a
SET S of relations (for a given output order) depends only on S, not on how S was built.
DP recurrence (MIT form):
  optjoin(S) = min over a∈S of [ cost(optjoin(S − {a})) + join-cost(optjoin(S−{a}), a) + access-cost(a) ]
optjoin(S−{a}) is looked up from the previous size level (cached). Build bottom-up over subset SIZE:
size 1 (best access path per single relation, per interesting order + unordered), size 2, …, size n.
Complexity: subsets total Σ C(n,k) = 2^n; each subset tries up to n removals × m join methods →
O(n·m·2^n) plan evaluations. For n=20, m=2 → ~4.1×10^7 (vs 2.9×10^35). Storage ≤ 2^n × (#interesting orders).

## Don't-consider-cross-products heuristic (primary §5)
Only extend the composite with a relation that has a join predicate to some relation already in it
(unless no such relation exists). Pushes Cartesian products as late as possible — prunes the subset lattice
further (only *connected* subsets matter). DuckDB's DPhyp formalizes this as enumerating connected
subgraph–complement pairs over the query (hyper)graph.

## Interesting orders (primary §4 "interesting order" + §5; MIT worked example)
An output order is "interesting" if some later operator can exploit it: an ORDER BY / GROUP BY clause, or a
join column used by a merge-scan join downstream. The cheapest *unordered* plan for S may NOT dominate: a
costlier plan that happens to produce an interesting order can save a later sort or enable a cheap merge join.
So the DP table keeps, per subset S, the cheapest unordered plan AND the cheapest plan for each interesting
order. Equivalence classes of orders (E.DNO=D.DNO=F.DNO collapse to one class) limit how many to track.
At the end: compare (cheapest unordered plan + cost to sort into required order) vs (cheapest plan already in
that order); pick min. MIT example: SELECT A.f3,B.f2 ... A.f3=B.f4 ORDER BY A.f3 → compare cost(sort)+156
(BA-hash, cheapest) vs 180 (AB-merge already in A.f3 order). Increases complexity by factor (k+1), k =
#interesting orders.

## Empirical / motivating facts (→ context Background; recall-and-reason in reasoning, never measure)
- Join order changes cost by orders of magnitude though result cardinality is order-independent (primary §5).
- 20-way join naive count 20!·19! ≈ 2.9×10^35 (MIT). DP brings it to ~4.1×10^7. Optimization of an 8-table
  join took "a few seconds" / "a few tenths of a second of 370/158 CPU time", a few thousand bytes (primary).
- Blasgen & Eswaran 1976: for non-tiny relations one of the two join methods is always optimal or near optimal
  → System R uses just nested-loop + merge-scan (primary §5, ref <4>).
- Primary conclusion (motivating, pre-method-result, fine for context/reasoning): predicted costs are often
  not accurate in absolute value, but the true optimal path is picked in a large majority of cases, and the
  *ordering* among estimated costs is frequently exactly the ordering among actual measured costs. This is the
  diagnostic that licenses cost-based optimization: you don't need accurate costs, only correct *rankings*.
  (NOT the proposed method's benchmark win — it's the rationale for why ranking-by-estimate works.)

## Baselines / lineage
- Codd 1970 (relational model), Date 1975 — relational algebra; order-independence of join result is the
  algebraic fact the DP rests on.
- INGRES decomposition (Wong & Youssefi 1976; Stonebraker et al 1976): query decomposition / variable
  substitution + tuple substitution — a heuristic, dynamic, runtime strategy; no global cost-based search,
  re-decides at runtime, no compiled plan. Gap: no principled minimization over the whole join order.
- Blasgen & Eswaran 1976: analyzed 2-way join methods → justifies the 2-method restriction.
- ASL / Lorie & Nilsson 1978, code generator Lorie & Wade 1977: the plan representation the optimizer emits.

## Design-decision → why table
- Restrict to left-deep: kills (n-1)! tree shapes; enables pipelining (no materialization unless sort needed);
  bushy could be cheaper sometimes but the search blows up and the inner being a base relation lets you use
  its indexes directly. (DuckDB later does bushy via DPhyp; at System R time left-deep is the win.)
- DP over subsets (not branch-and-bound / not greedy): principle of optimality holds because cost of best plan
  for S depends only on S → memoize per subset; 2^n ≪ n!·(n-1)!.
- Build by subset SIZE bottom-up: guarantees optjoin(S−{a}) is already computed when needed.
- Cost = page fetches + W·RSI calls: captures both I/O and CPU; W tunable to the machine's balance.
- Selectivity F with 1/10, 1/3, 1/4 defaults: crude but monotone; absolute accuracy not needed, only ranking.
  Uniform-distribution / independence assumptions = tractable, no histograms in 1979.
- Two join methods only: Blasgen-Eswaran says more is wasted effort.
- Don't consider cross products: a Cartesian product blows N up multiplicatively with no selectivity to tame
  it → always defer; prunes to connected subsets.
- Keep interesting orders: a sort is expensive; a plan that arrives pre-sorted can dominate end-to-end even if
  its own join cost is higher → must not prune purely on join cost.
- Equivalence classes for orders: columns transitively equated by join predicates produce the same useful
  order → track one representative, bound the bookkeeping.
- N (outer card) = product of cards × product of applicable-predicate selectivities — the running estimate
  that drives nested-loop cost; same engine as QCARD.

## Final code plan (grounded in DuckDB recurrence + System R formulas + MIT optjoin)
Self-contained Python Selinger-style DP optimizer:
- Catalog: Relation(card, pages), Index(icard, npages, clustered), Predicate (selectivity).
- Selectivity model: Table-1 rules. Cardinality of a subset = product cards × product applicable F.
- Single-relation access paths: scan + each index, cost per Table 2; produce (cost, output_order).
- DP table keyed by frozenset(subset) → dict {order: best Plan}; Plan has cost, card, order, structure.
- Bottom-up by size: size1 = access paths; size k = for each connected subset, for each way to split off one
  base relation `a`, for each join method (nested-loop, merge), combine plans[S-a] × access(a) →
  cost = left.cost + N·inner_scan (nested) / merge variant; keep cheapest per interesting order + unordered.
  DuckDB-style update: replace plans[S][order] iff new_cost < old_cost.
- connected(): only join when a join predicate links a to S (else skip = don't-consider-cross-products),
  unless forced.
- Finalize: for required ORDER BY, min(cheapest-unordered + sort_cost, cheapest-in-order).
