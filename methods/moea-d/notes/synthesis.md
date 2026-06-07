# MOEA/D synthesis

## Pain point / research question
Multi-objective: want the whole Pareto front in one run. Population EA holds many solutions. But how to give scalar fitness to vectors? NSGA-II's answer: Pareto-dominance ranking + crowding. Diagnostic limitation: dominance-based ranking weakens selection pressure as M grows (most pairs become mutually nondominated -> almost everything is rank 1 -> selection becomes near-random on convergence; "dominance resistance"). Also nondominated sort is O(MN^2)/gen and crowding is a heuristic with no direct link to a uniform spread target.

## Intellectual move
Go back to classical scalarization (Miettinen): a Pareto-optimal point is the optimum of a scalar aggregation g(x|w). Sweeping weights is expensive (one run per point) and weighted-sum misses concave fronts. BUT: in a population EA the "many runs" objection vanishes — keep N subproblems, one per weight vector, and solve them all *simultaneously in one population*, sharing information. So decompose into N scalar subproblems and co-evolve.

## Three scalarizations (verified vs survey 2404.14571 eqs 2-4 AND pymoo code)
- WS: g^ws(x|w) = Σ_i w_i f_i(x).  Convex fronts only.
- TCH (Tchebycheff): g^tch(x|w,z*) = max_i { w_i |f_i(x) - z_i*| }.  z* = ideal point. Recovers concave fronts; weight w<->Pareto point is one-to-one; non-smooth (max).
  - convention: w_i can be 0 -> replace with tiny eps or use 1/w handling; pymoo uses utopian z*-eps. Standard paper form uses w_i directly with max.
- PBI: g^pbi(x|w,z*) = d1 + θ d2, d1 = ||(F(x)-z*)^T w|| / ||w||  (projection length along w), d2 = ||F(x) - (z* + d1 w)|| (perpendicular distance to the line). θ>0 penalty, typically 5. Smooth, gives even spread, but needs θ and normalization.

## Weight vectors: Das & Dennis simplex lattice
Divide each coord into H equal parts; pick all (a_1..a_m) with a_i in {0,1/H,...,H/H} summing to 1. Count N = C(H+m-1, m-1). Recursive generator (pymoo das_dennis_recursion). For m=2 just w = (i/H, 1-i/H), i=0..H -> N=H+1.

## Neighborhood
B(i) = indices of the T weight vectors closest (Euclidean) to w^i, including i itself. T~20. Idea: neighboring weight vectors -> similar subproblems -> similar optima, so a good solution for one is promising for its neighbors. Mating restricted to neighborhood; offspring updates neighborhood.

## Vanilla MOEA/D loop (verified vs 2108.09588 + pymoo)
Init: N weight vectors + B(i) by T-NN; population x^1..x^N random, FV^i=F(x^i); z* = min over pop per objective.
For each generation, for i=1..N:
  - pick two parents k,l randomly from B(i); reproduce y by SBX+polynomial mutation; (optional repair).
  - evaluate F(y); update ideal: for each j, if f_j(y)<z_j then z_j=f_j(y).
  - update neighbors: for each index k in B(i): if g(y|w^k,z*) <= g(x^k|w^k,z*) then x^k=y, FV^k=F(y).
  - update EP: remove from EP all vectors dominated by F(y); add F(y) if not dominated by any EP member.
Output EP (or population).

pymoo nuance: iterate k in random permutation of pop; with prob prob_neighbor_mating (0.9) draw parents from neighborhood else whole pop; replace where off_FV < FV (strict). Default decomp Tcheb for m<=2, PBI for m>2. n_neighbors=20.

## Complexity
Selection/update per offspring is O(T) (only neighborhood touched), O(N·T) per generation — cheap, no global sort, no global nondominated comparison. vs NSGA-II O(MN^2)/gen sort.

## Why decomposition beats dominance on selection pressure
Each subproblem has a *total order* (scalar g), so selection pressure never degrades as M grows: the comparison g(y) vs g(x^k) is always decisive. Dominance gives a partial order that flattens with M.

## Design choices -> why
- Tchebycheff over WS: WS supporting-hyperplane only touches convex hull -> concave front regions unreachable; TCH max-form supporting "L-shaped" contours touch concave parts; bijection weight<->PO point.
- z* ideal point not fixed: true ideal unknown; estimate by running min over all evaluated F; TCH measures weighted L_inf distance to z*, so a moving z* keeps contours anchored to the (estimated) best corner.
- neighborhood T: parents from neighbors -> recombine similar good solutions (local information); update only neighbors -> an offspring good for w^i is likely good for nearby w^k. Too large T -> loses locality (becomes global, weak), too small -> little info sharing. 
- mating sometimes from whole pop (prob 0.9 neighbor): escape local stall / diversity.
- EP: population stores one-per-subproblem (might be dominated due to weighting); EP keeps the true nondominated archive to report.
- Das-Dennis lattice: uniform spread of weights -> uniform spread of PO solutions when PF is simplex-like.

## Canonical code = pymoo MOEAD (algorithms/moo/moead.py) + decomposition/{tchebicheff,pbi,weighted_sum}.py + util/reference_direction.py das_dennis. Tcheb: v=|F-z*|·w; g=max(v). Replace where off_FV<FV. neighbors=argsort(cdist(W,W))[:, :T].

## ZDT1 (m=2, n=30): f1=x1; g=1+9*sum(x2..xn)/(n-1); f2=g*(1-sqrt(f1/g)); PF: f2=1-sqrt(f1), convex, x_i=0 for i>=2. x in [0,1]^30.
DTLZ2 alt. Use ZDT1.

## House style: sibling nsga-ii. Position MOEA/D as the decomposition alternative to dominance ranking. In-frame: never cite Zhang-Li as artifact; may cite Miettinen scalarization, Das&Dennis, NSGA-II ancestors, Schaffer VEGA, MOGLS (Ishibuchi/Jaszkiewicz).
