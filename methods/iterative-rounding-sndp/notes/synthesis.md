# Synthesis — Iterative Rounding for SNDP (Jain factor-2)

## Problem
Undirected G=(V,E), edge cost c_e ≥ 0, integer requirement r(uv) for each pair. Find min-cost
subgraph H with ≥ r(uv) edge-disjoint u-v paths. By Menger, edge-disjoint paths = min cut, so the
requirement is a cut condition: for every S separating some pair, |δ_H(S)| ≥ max_{u∈S,v∉S} r(uv).
Define f(S) = max_{u∈S,v∉S} r(uv); f(∅)=f(V)=0. Goal: min-cost subgraph covering f.

## Key abstraction: cover a skew/weakly-supermodular function by a graph
f: 2^V → Z is weakly supermodular (= skew-supermodular) if for all A,B at least one of
  f(A)+f(B) ≤ f(A∪B)+f(A∩B)   OR   f(A)+f(B) ≤ f(A\B)+f(B\A).
f(S)=max r(uv) over the cut is weakly supermodular (verify by cases on where the maximizing pair
sits). Problem becomes: min Σ c_e x_e s.t. x(δ(S)) ≥ f(S) ∀S, 0≤x_e≤1.

## Why not the obvious things
- Doubling / k-ECSS doubling gives factor 2 only for uniform/spanning; for general pairwise r it
  doesn't even give feasibility cheaply.
- Primal-dual (Williamson-Goemans-Aronson-Tardos / Goemans et al. 1994) gives 2·H(r_max) — factor
  grows with max requirement, because it works "layer by layer" (augment connectivity one unit at a
  time, each layer a 0/1 cut-cover). Want a guarantee independent of r_max.
- Pure LP rounding: round all x_e ≥ τ. Need a τ that (a) always exists and (b) gives small factor.
  Threshold 1/2 ⇒ factor 2 IF such an edge always exists in an extreme point.

## The structural theorem (the crux)
THEOREM (Jain). In any basic feasible (extreme point) solution x to the LP with f weakly
supermodular, there is an edge e with x_e ≥ 1/2.

### Step A — uncrossing → laminar basis
An extreme point in R^E is determined by |E| linearly independent tight constraints. Drop edges with
x_e=0; remove x_e=1 (handled separately) — WLOG 0<x_e<1 for all e. Tight sets are those with
x(δ(S))=f(S). The cut function x(δ(·)) is submodular AND posimodular:
  x(δ(S))+x(δ(T)) ≥ x(δ(S∪T))+x(δ(S∩T))            (submodular)
  x(δ(S))+x(δ(T)) ≥ x(δ(S\T))+x(δ(T\S))            (posimodular, since symmetric submodular)
Characteristic-vector identities:
  χ(δ(S))+χ(δ(T)) = χ(δ(S∪T))+χ(δ(S∩T)) + 2·χ(E(S\T,T\S))
  χ(δ(S))+χ(δ(T)) = χ(δ(S\T))+χ(δ(T\S)) + 2·χ(E(S∩T, V\(S∪T)))
If S,T both tight and they cross: pick whichever of the two weakly-supermodular inequalities f
satisfies. Say f(S)+f(T) ≤ f(S∪T)+f(S∩T). Then
  f(S)+f(T)=x(δ(S))+x(δ(T)) ≥ x(δ(S∪T))+x(δ(S∩T)) ≥ f(S∪T)+f(S∩T) ≥ f(S)+f(T),
so equality throughout ⇒ S∪T, S∩T are also tight, AND the cross term χ(E(S\T,T\S)) has zero
x-weight ⇒ since all x_e>0, E(S\T,T\S)=∅ ⇒ χ(δ(S))+χ(δ(T))=χ(δ(S∪T))+χ(δ(S∩T)). So the span of
tight constraints is preserved by replacing {S,T} with {S∪T, S∩T}, which are uncrossed. Iterating
(potential function Σ|S|^2 strictly increases) yields a laminar family L of tight sets that still
spans, with a maximal independent subfamily of size |E|. Hence: there is a laminar L with
x the unique solution of {x(δ(S))=f(S): S∈L}, χ(δ(S)) independent, |L|=|E|.

### Step B — counting argument: assume x_e<1/2 ∀e and derive |E|>|L| (contradiction)
Two presentations, both grounded:

(B1, NRS fractional token, cleanest). Give each edge 1 token. Edge e=(u,v) redistributes:
 Rule 1: to the smallest set of L containing u, and to the smallest containing v: x_e each.
 Rule 2: to the smallest set T∈L containing BOTH u and v: 1−2x_e.
All of x_e, 1−x_e, 1−2x_e > 0 since 0<x_e<1/2.
For S∈L with children R_1..R_k, subtract child equalities from parent:
  x(δ(S)) − Σ x(δ(R_i)) = f(S) − Σ f(R_i) ∈ Z.
Classify edges by (#endpoints in ∪R_i, #endpoints in S):
  A: 0 in ∪R_i, 1 in S; B: 1 in ∪R_i, 2 in S; C: 2 in ∪R_i, 2 in S. Then
  x(A) − x(B) − 2x(C) = f(S) − Σf(R_i).
Tokens S collects: x_e for e∈A (Rule1), (1−x_e) for e∈B (Rule1+Rule2), (1−2x_e) for e∈C (Rule2):
  Σ_A x_e + Σ_B(1−x_e) + Σ_C(1−2x_e) = x(A) + |B|−x(B) + |C|−2x(C)
   = |B|+|C| + [f(S) − Σf(R_i)]  ∈ Z, and > 0 because A∪B∪C ≠ ∅ (else χ(δ(S))=Σχ(δ(R_i)),
   contradicting independence). A positive integer ⇒ ≥ 1. So every S∈L gets ≥1 token.
Finally a maximal set R∈L: any e∈δ(R) has its Rule-2 token unassigned (no set contains both ends).
So strictly more than |L| tokens distributed-but-not-fully-collected ⇒ |E|>|L|. Contradiction.
COROLLARY: some x_e ≥ 1/2. (And if f is even-valued ⇒ some x_e=1; gives STSP/2-ECSS as special case
with the 1−x_e, x_e/2 token split — that's the integral-edge variant, Boyd–Pulleyblank flavor.)

(B2, Chekuri combinatorial, for intuition). Warmups: if L all disjoint, x_e<1/2 ⇒ |δ(S)|≥3 ⇒
≥3m endpoints among m disjoint sets but only 2m endpoints exist. With internal nodes (≥2 children)
get x_e≥1/3 by leaf-counting (4k endpoints, k>m/2). Unique-child handled by Lemma 5 (a unique child
forces ≥2 endpoints "owned" by parent, else χ(δ(S)),χ(δ(C)) dependent). The 1/2 bound needs the
invariant Claim: f(S) ≥ α(S) − β(S) (α = #sets of L inside S incl S; β = #edges with both ends in
S), proven by induction with the γ(S) edge bookkeeping (Case γ=0 uses independence ⇒ E_po≠∅; Case
γ≥1 uses x_e<1/2). Summing over roots reduces to the disjoint case.

## Residual stays in the class (so we can iterate)
LEMMA (Jain). f weakly supermodular, F⊆E ⇒ g(S)=f(S)−|δ_F(S)| is weakly supermodular. Proof: |δ_F|
is symmetric + submodular (cut function), symmetric submodular ⇒ posimodular too; f minus a
symmetric submodular function is weakly supermodular (check both inequality cases). This is what lets
the residual problem be the SAME kind of problem.

## The algorithm
F ← ∅.
while f' := f − |δ_F| is not all ≤ 0:
   solve LP: min Σ_{e∉F} c_e x_e  s.t.  x(δ(S)) ≥ f'(S) ∀S, 0≤x_e≤1, to an EXTREME POINT.
   by the theorem ∃ e with x_e ≥ 1/2 (also set any x_e=1 to 1); add all such e to F.
return F.

## Factor-2 analysis (induction on rounds)
Let OPT_LP = initial LP optimum. In a round, let x* be the extreme point, R={e: x*_e ≥ 1/2}.
cost(R)=Σ_{e∈R} c_e ≤ Σ_{e∈R} c_e (2 x*_e) ≤ 2 Σ_e c_e x*_e = 2·LP(current). Restricting x* to the
remaining edges (and using x*_e on them) is feasible for the residual instance with f' = f−|δ_R|,
because residual requirements drop by exactly what R covers and g stays
skew-supermodular; the residual LP optimum ≤ LP(current) − cost(R)/ (… ) — cleaner: by induction the
total cost ≤ 2·OPT_LP ≤ 2·OPT_IP. Standard statement: at termination
  cost(F) = Σ_rounds cost(R_round) ≤ 2 Σ_rounds (LP_round − LP_{round+1}) ≤ 2·OPT_LP,
using LP monotonicity LP_{round+1} ≤ LP_round − cost(R_round) (x* minus its rounded part is feasible
for the next residual LP). Since OPT_LP ≤ OPT_IP = OPT, cost(F) ≤ 2·OPT.

## Solving the LP (separation oracle)
Exponentially many cut constraints. Separation: given x, for each pair u,v compute min u-v cut value
λ_x(uv) in G weighted by x (max-flow). Violated iff λ_x(uv) < r(uv); the min-cut set S is the
violated constraint. All pairs via a Gomory–Hu tree (n−1 max-flows). Ellipsoid (poly) or
cutting-plane + LP solver in practice. Extreme point: solve LP then if solution not a vertex, move to
a vertex of the optimal face (or use a solver returning a basic optimal solution).

## Code grounding
No single canonical repo; implement faithfully from the math with NetworkX (max-flow / min-cut,
Gomory–Hu) + an LP solver via cutting planes (scipy.optimize.linprog or PuLP). Structure:
 - requirement function f via r(uv)
 - separation oracle: Gomory–Hu / per-pair max-flow → violated cut S
 - LP solve with cutting planes to a basic optimal solution
 - round edges x_e ≥ 1/2, fix, recompute residual r' (subtract connectivity already provided by F)
 - loop. The residual update in code: when F fixed, the *effective* requirement on the remaining
   graph for pair uv is r(uv) − (edge-disjoint paths already in F) — equivalently keep f and use
   f'(S)=f(S)−|δ_F(S)| inside the oracle.

## Design decisions → why
- Threshold exactly 1/2: it's the largest threshold the structure theorem can guarantee always
  exists (token argument is tight), and 1/τ = 2 is the resulting factor. 1/3 is easy but worse;
  >1/2 not guaranteed.
- Round ALL e≥1/2 each round (not one): fine, still feasible-residual + same 2·LP charge; fewer LP
  solves. Rounding one at a time also works.
- Extreme point (not any optimal x): the 1/2 guarantee is a property of *vertices* of the polytope;
  an interior optimal point can be all-fractional (e.g. all x_e=1/2−ε would violate, but fractional
  optima exist that have no big coordinate unless we go to a vertex). Must take a basic solution.
- f = max over cut (not sum): cut form is exactly Menger's reduction of edge-disjoint paths; and max
  is what's weakly supermodular (sum would not encode the requirement correctly).
- Residual via f−|δ_F|: keeps the problem inside the skew-supermodular class (the closure property),
  which is the only reason iteration is legitimate.
