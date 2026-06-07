# Synthesis — Energy-function (soft-min) method for packing equal circles in a square

## Method chosen and why
The strongest PRINCIPLED method for this task is the **energy-function minimization with
exponent annealing** (Nurmela & Östergård 1997, building on Clare–Kepert 1986 and
Kottwitz 1991 spherical-code energy minimization). It is genuinely insight-driven, not blind
search:
- It turns the non-smooth max-min (L∞) objective into a smooth one via the L^m → L∞ identity.
- It uses a **soft-min relaxation**: minimize Σ (λ/d_ij²)^m, an inverse-power repulsion energy.
- It makes the box constraint **vanish** via the sin() coordinate change (unconstrained).
- It anneals m from small (10–100) to large (10^6 … 10^50), a **homotopy/continuation**:
  small m = soft, convex-ish, easy to optimize; large m = sharp, recovers the true max-min.
- It rescales λ each stage to keep the energy in a workable numeric range.
- It finishes with multistart + Newton polish + **structure detection** (identify equal/contact
  distances, build a polynomial/nonlinear system, solve to arbitrary precision).

The Lubachevsky–Stillinger / Graham–Lubachevsky billiard simulation is the principled
*physical* alternative (event-driven MD of growing disks to a jammed state) — covered as a
baseline. Packomania records / Boll-Donovan perturbation search / AlphaEvolve-style search are
the ANTI-PATTERN (blind search outputs) — context only.

## Exact problem formulation
- Maximize the radius r_n of n non-overlapping equal circles in the unit square.
- Equivalent point form: place n points in [0,1]^2 maximizing d_n = min_{i<j} ||s_i − s_j||.
- Exact relation (Nurmela-Östergård p.112):  r_n = d_n / (2(d_n + 1)).
  Derivation: shrink the square by r on all sides → centers live in a (1−2r) square; scaling so
  centers are unit-square points means the center-square side is 1, and on that scale the
  inter-center spacing is d_n while the real side is 1/(1−2r)... equivalently real side
  L = 2 + 2/d_n with unit-radius circles (Wikipedia). Both give r_n = d_n/(2(d_n+1)).
- Wall constraint: a circle center must be ≥ r from each wall.

## The L^m identity (the seed)
min_{i<j} ||s_i − s_j|| = lim_{m→ -∞} ( Σ_{i<j} ||s_i−s_j||^m )^{1/m}.
So for large negative power, the sum is dominated by the SMALLEST distance. Drop the 1/m
exponent (monotone) and write the energy with positive m on the inverse square:
E = Σ_{i<j} (λ / d_ij²)^m,  d_ij² = (x_i−x_j)²+(y_i−y_j)².
Minimizing E ⇔ pushing up the smallest d_ij (since the term with smallest d² dominates and we
want to shrink the largest term). As m→∞ only the minimum distance matters → recovers max-min.

## sin() transform
x_i = sin(x̃_i), y_i = sin(ỹ_i), x̃,ỹ ∈ R. Then automatically −1 ≤ x_i,y_i ≤ 1 (a box of side 2;
unit square is just a rescale). Unconstrained in (x̃,ỹ). Chain rule: ∂x_i/∂x̃_i = cos(x̃_i).
This is smoother / less prone to "sticking on the boundary" than clamping, and it removes the
need for inequality constraints / penalty walls.

## Gradient (verified numerically, rel err 5e-9)
Let T_ij = (λ/d_ij²)^m. ∂T_ij/∂d_ij² = −m T_ij / d_ij². For point i:
∂E/∂x_i = Σ_{j≠i} (−m T_ij/d_ij²)·2(x_i−x_j),  similarly y.
Then ∂E/∂x̃_i = (∂E/∂x_i)·cos(x̃_i).

## λ rescaling
After each m-stage set λ = (shortest distance)² (Nurmela-Östergård p.114). Keeps the dominant
term (λ/d²_min)^m ≈ 1 instead of overflowing/underflowing. Use λ slightly below d²_min in code
to avoid >1 base with huge m.

## Annealing schedule
Start m in [10,100]. Optimize to a local min (steepest descent + Goldstein–Armijo backtracking
line search early; modified Newton late). Double m, re-optimize, warm-started from previous
solution. Continue to m ~ 10^6 (sometimes up to 10^50). ≥ 50 random restarts per n.

## Local-optimum escapes
- A circle touching a wall that could move inward without overlap (Fig.2 in NO97): nudge it off
  the wall and re-optimize with large m.
- Multistart (≥50) to dodge poor basins.

## Structure detection / exact refinement
Sort all pairwise distances; find the sudden jump separating "contact" distances from gaps.
Treat those as equalities; treat near-wall points as on-wall. Build nonlinear system → solve by
Newton–Raphson to arbitrary precision (overlaps/gaps < 1e-33). Reduces in principle to a single
univariate polynomial whose root is the diameter.

## Baselines / ancestors
- Fejes Tóth (1972) seminal packing theory; Graham-Meir-Schaer (1960s) exact n≤9.
- de Groot–Peikert–Würtz (1990–92): maximize-radius directly via simplex + BFGS quasi-Newton;
  also Langevin (stochastic) formalism. Fails n=14,15,17 — the max-min objective is non-smooth
  (subgradient at the active min), so smooth optimizers stall.
- Clare-Kepert (1986), Kottwitz (1991): energy minimization for circles ON A SPHERE; quasi-Newton
  (Fletcher-Powell-Davidon) / gradient+Newton; refine by Newton-Raphson on contact equations.
  Direct ancestor of the energy approach; NO97 adapt it to the square + add the m-annealing.
- Graham-Lubachevsky (1995/96): event-driven billiard simulation of growing disks (physical
  jamming) for triangle/square. LS algorithm (Lubachevsky-Stillinger 1990): random point
  particles + velocities, grow common radius linearly in time, event-driven collisions, stop at
  jamming; rattlers are loose. Principled but stochastic and gets stuck without restarts.
- Boll-Donovan-Graham-Lubachevsky (2000): N/S/E/W perturbation search, s←s/1.5 — pure local
  search; the anti-pattern (blind). Packomania (Specht) collects best-known records — context only.
- Proven optimal: n ≤ 30 (and 36) via hand proofs + interval branch-and-bound (Markót-Csendes,
  Nurmela-Östergård interval verification). Square lattice optimal for n=k² exactly up to n=36.

## Evaluation settings (pre-method)
Yardstick: d_n / r_n vs known exact values (n≤9 by hand; later n≤20, n≤30,36 proven). Density =
n·π·r_n² / 1. Number of contacts, symmetry group order. Compare to square-lattice and hexagonal
densities. Packomania tables of best-known d_n.

## Code framework
scipy.optimize.minimize (CG / Newton-CG) with analytic jac; numpy vectorized pairwise distances;
multistart loop; m-annealing loop; λ update; final contact-system polish (Newton).

## Source URLs actually read
- Nurmela & Östergård, "Packing up to 50 Equal Circles in a Square", DCG 18:111-120 (1997) —
  full PDF read (refs/no97.txt). PRIMARY for the energy method.
- Szabó, "Global Optimization in Geometry — Circle Packing into the Square" survey PDF
  (inf.u-szeged.hu/~pszabo/Pub/45survey.pdf) — full energy/billiard/MBS/perturbation sections.
- Wikipedia "Circle packing in a square" — r_n=d_n/(2(d_n+1)), L=2+2/d_n, proven-optimal n≤30.
- Wikipedia "Lubachevsky–Stillinger algorithm" — billiard mechanics, jamming, rattlers.
