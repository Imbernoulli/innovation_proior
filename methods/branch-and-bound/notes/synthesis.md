# Synthesis: Branch and Bound (Land-Doig 1960 / Dakin LP-relaxation tree search)

## Sources retrieved & read
- PRIMARY: Land & Doig 1960, "An Automatic Method of Solving Discrete Programming Problems",
  Econometrica 28(3):497-520. Full PDF read (refs/land_doig_1960.pdf). Their actual method:
  parametric-LP — trace min x_k(y) and max x_k(y) functions of the falling objective hyperplane,
  find first integer of x_k as y decreases, build a tree of subproblems P(j) (j of the x vars
  constrained integer), labelled vertices = nonincreasing sequence of upper bounds y0>=y1>=...,
  prune when a discrete solution's value >= every branch's bound. "Cut off" value = early incumbent.
  They branch by creating a branch for EACH integer value of x_k in [l_k,u_k].
- PRIMARY (variant): Dakin 1965, Computer Journal 8:250-255. The dichotomous branching
  x_k <= floor(x*_k) OR x_k >= ceil(x*_k) — replaces Land-Doig's many-children-per-value split.
  Modern MILP B&B is Dakin's rule. (Confirmed via GERAD retrospective + DTU lecture.)
- PRIMARY (coining): Little, Murty, Sweeney & Karel 1963, Operations Research 11(6):972-989 —
  coined "branch and bound" for the asymmetric TSP: split tours into subsets (branch), compute a
  lower bound on each subset (bound), prune.
- BACKGROUND: simplex (Dantzig 1947/1951) — moves extreme point to extreme point via pivots;
  LP optimum at a vertex; relaxation gives a bound. LP duality: any dual-feasible solution gives a
  valid bound on the LP optimum. Gomory 1958 cutting planes — add valid inequalities that cut off
  fractional LP optima without removing integer points; convergent but slow in practice. ILP/MILP
  framing (Dantzig 1960). Convex hull / relaxation idea.
- THIRD-PARTY: Laporte 2023 GERAD retrospective (refs/gerad_landdoig.pdf) — full lineage, Dakin,
  branch-and-cut (Miliotis 1976 combined B&B + Gomory cuts), best/depth-first, incumbent/fathom.
  DTU LP-based B&B lecture (Larsen, refs/dtu_bab4.pdf) — clean modern statement: divide-and-conquer
  S = S1 u ... u SK, z-bar = max z-bar_k upper bound, z = max z_k lower bound, prune by
  optimality / bound / infeasibility, branch on fractional var x_j <= floor / x_j >= ceil.

## The intellectual move (discovery order)
1. Pure integer program: max c'x s.t. Ax<=b, x integer >=0. Feasible set = lattice points in a
   polytope. NON-CONVEX (discrete points). Enumerate? Exponential — 2^n for 0-1, unbounded for
   general integer. Need something smarter.
2. Drop integrality -> LP RELAXATION. Convex polytope, simplex solves it in practice fast, optimum
   at a vertex. KEY: relaxation enlarges the feasible set, so for MAXIMIZATION its optimum value is
   an UPPER BOUND on the integer optimum (z_LP >= z_IP). (For min: lower bound.)
3. If the LP optimum x* is already integer -> done, it's optimal for the IP (feasible + achieves the
   upper bound). Usually some x*_j is fractional.
4. BRANCH on a fractional x*_j: the true integer optimum has x_j <= floor(x*_j) OR x_j >= ceil(x*_j)
   — the open strip floor < x_j < ceil contains no integer, so we lose no integer solutions. Two
   child subproblems, each an IP with one tightened bound. Their union covers all integer feasible
   points; x* is feasible in NEITHER child (it's cut out of the strip), so each child's relaxation
   is strictly worse-or-equal — the tree makes progress.
5. RECURSE: each child gets its LP relaxation = its own upper bound. Tree of LPs.
6. INCUMBENT: best integer-feasible solution found so far, value z_inc (= global LOWER bound for max).
   When a node's relaxation is integral, it's a candidate; update incumbent if better.
7. PRUNE (fathom) a node without exploring its subtree when:
   - infeasible (relaxation has no solution), OR
   - bound: relaxation upper bound <= z_inc (subtree can't beat incumbent), OR
   - integral: relaxation is integer (it's a leaf -> candidate incumbent, no need to branch).
   The BOUND does the pruning — that's what makes it better than enumeration.
8. OPTIMALITY GAP: z_inc <= z* <= best remaining upper bound (max over open nodes' bounds). Gap =
   (best_bound - z_inc). Zero gap = proven optimal. Can stop early at a tolerance.
9. Node selection (which open node next):
   - BEST-FIRST: expand node with best (largest, for max) relaxation bound — minimizes tree size,
     fewest nodes, but stores many open nodes (memory).
   - DEPTH-FIRST: dive to leaves — small memory (one path's bounds), finds an incumbent fast so
     pruning kicks in early; may explore more nodes. Land & Doig (by hand, paper storage) pursued
     each branch until its bound was no longer best (best-first-ish); on a computer depth-first
     dominates for memory. Modern solvers mix (best-first + diving).
10. BRANCH-AND-CUT: when a node's LP bound is loose, ADD cutting planes (e.g. Gomory cuts) to the
    node's LP to tighten the bound BEFORE branching — shave the fractional vertex off without
    removing integer points. Gomory alone is slow; embedded in B&B (Miliotis 1976) it's the basis
    of modern MILP solvers (CPLEX, Gurobi).

## Bounding direction (CRITICAL, easy to flip)
- MAXIMIZE: relaxation value is an UPPER bound; incumbent is a LOWER bound; prune if node bound
  <= incumbent. Gap = best_open_upper_bound - incumbent >= 0.
- MINIMIZE: relaxation value is a LOWER bound; incumbent is an UPPER bound; prune if node bound
  >= incumbent.
The code maximizes; linprog minimizes, so feed -c and negate the returned fun.

## Branching constraints (CRITICAL)
Fractional x*_j: down child x_j <= floor(x*_j), up child x_j >= ceil(x*_j). In a bounded-variable
LP these are just tightened variable BOUNDS — no new rows needed. Most-fractional rule: pick j
maximizing |x*_j - round(x*_j)| (closest to 0.5).

## Design choices -> why
- LP (not Lagrangian/combinatorial) relaxation: cheapest tight convex bound available; simplex is
  mature; warm-starts down the tree (child differs from parent by one bound) via dual simplex.
- Dakin dichotomy vs Land-Doig per-value split: two children keep the tree binary and each child is
  still a clean LP with just a tightened bound; per-value split makes a wide bushy node and (for
  general integer vars with large domains) many children. Dakin is more efficient and is what
  solvers use.
- <= floor / >= ceil (not = each integer): the disjunction is exhaustive over integers and the
  removed strip (floor, ceil) has no integer, so completeness holds with only two children.
- most-fractional branching: a variable near 0.5 is maximally "undecided"; both children move the
  relaxation substantially, so the bound tightens fastest. (Pseudocost/strong branching are better
  but need history; most-fractional is the simple default.)
- depth-first default in code: O(depth) memory, gets an incumbent fast to enable bound-pruning.
- incumbent + bound pruning is the whole point: without it B&B = enumeration; the relaxation bound
  proves whole subtrees can't contain the optimum, so they're never opened.

## Code
scipy.optimize.linprog (HiGHS) for each node's LP relaxation; branching = tighten the per-variable
`bounds`; explicit stack (depth-first); incumbent + most-fractional + prune by
infeasible/bound/integral; report optimality gap; demonstrate on 0-1 knapsack + general integer LP;
check against brute force. Verified: knapsack and DTU general-integer instance both match brute
force. (code/bnb.py, code/test2.py)
