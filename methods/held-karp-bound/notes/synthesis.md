# Synthesis — Held–Karp 1-tree lower bound (Lagrangian) via subgradient ascent

## In-frame slug discipline
This slug IS the **Held–Karp 1-tree LOWER BOUND** (a Lagrangian dual bound on the symmetric
TSP) and the **subgradient / iterative-ascent** method that computes it. It is NOT the
O(2^n n^2) Held–Karp dynamic program for exact TSP — keep them distinct. The reasoning is
about a *cheap, tight lower bound to drive branch-and-bound*, not an exact DP.

## Primary sources (full text in refs/)
- `hk1971_partII.pdf` / `.txt` — Held & Karp, "The Traveling-Salesman Problem and Minimum
  Spanning Trees: Part II," *Mathematical Programming* 1 (1971) 6–25. THIS is the in-frame
  paper: it contains the 1-tree, the bound (eqs 1–2), w(π), the subgradient iteration (eq 3),
  Lemma 1 (subgradient inequality), Lemma 2 (step-size shrinks distance to optimum), Lemma 3
  (relaxation/Fejér-monotone convergence), Theorem 1, branch-and-bound (§3), and the
  assignment-problem warm start (§4a). Read in full.
- Part I (Held & Karp 1970, *Operations Research* 18:1138–1162) is **CLOSED access** —
  verified via Unpaywall (`oa_status: closed`, no oa_locations), Semantic Scholar
  (`openAccessPdf: CLOSED`), and ~10 direct mirror attempts (all 404/paywall HTML). Its
  load-bearing content (1-tree definition, the bound, node-potential transformation
  c_ij→c_ij+π_i+π_j, the original column-generation LP + steepest-ascent attempts) is
  **fully reproduced in Part II §1**, which "briefly review[s] the approach taken in [7]." So
  the primary content is captured; the Part I PDF itself is an honest GAP.

## Antecedents (full text in refs/)
- `agmon1954_relaxation.pdf/.txt` — Agmon, "The Relaxation Method for Linear Inequalities,"
  *Canad. J. Math.* 6 (1954) 382–392. The projection-into-halfspace iteration with relaxation
  parameter 0<λ<2 (Lemma 2.1): x → x+λ(x_r−x) gets closer to the solution set. HK's ref [1];
  Motzkin–Schoenberg (ref [12]) is the companion. Hoffman pointed HK to these.
- `geoffrion_lagrangian_relaxation.pdf/.txt` — Geoffrion, "Lagrangian Relaxation for Integer
  Programming" (commentary + Ch. 9 of *50 Years of IP*). Dualize the "complicating"
  constraints with multipliers; the dual is concave; HK's degree-2 dualization is the
  archetype that "birthed" the Lagrangian approach.
- `held-wolfe-crowder1974_validation.pdf/.txt` — Held, Wolfe, Crowder, "Validation of
  Subgradient Optimization," *Math. Prog.* 6 (1974) 62–88. Canonical Polyak step
  t_j = λ_j (w̄ − w(π^j))/‖v(π^j)‖², ε≤λ_j≤2 (eq 2.8); subgradient inequality (2.4); narrates
  the discovery — steepest-ascent and column-generation simplex were "dishearteningly slow,"
  so HK "invented a subgradient method." Used to ground derivation-time *reasoning*, not as
  posterior to cite in-frame.

## Analysis (full text / capture in refs/ + notes/)
- `goffin1977_convrates_subgradient.pdf/.txt` — Goffin, "On Convergence Rates of Subgradient
  Optimization Methods," *Math. Prog.* 13 (1977) 329–347. Geometric/linear rate under the
  Polyak step; conditioning of the nonsmooth max.
- `umd_heldkarp_implementation_report.txt` — Taylor-Moore (UMD) implementation report:
  concrete recipe — min 1-tree = MST (Kruskal/Prim) + two cheapest edges at the special node;
  π_i^(m+1)=π_i^m+t_m(d_i^m−2); VJ / Valenzuela–Jones step schedule.
- scientific-python blog "A Closer Look at the Held-Karp Relaxation" (WebFetch capture):
  c̄_ij=c_ij+π_i+π_j, w(π) piecewise-linear concave, degree residuals d_i−2 as subgradients,
  out-of-kilter vertices.

## Code (canonical, in code/)
- `one_tree_lower_bound.h` — genuine google/or-tools snapshot (Copyright 2010–2025 Google
  LLC; "Held-Karp symmetric TSP lower bound … minimum 1-trees"). Volgenant–Jonker (default)
  and Held–Wolfe–Crowder step rules.
- `held_karp_bound.py` — Python mirror of the OR-Tools structure; runs (5-city: plain
  1-tree 1.87 → HK bound 2.70, correctly tighter). min 1-tree via Prim on n−1 ordinary nodes
  + two cheapest edges from the left-out node; w(π)=cost(1-tree)+Σπ_i(deg_i−2); ascent
  π+=step·(deg−2); VJ vanishing schedule and HWC Polyak step with λ-halving.

## The derivation spine (for reasoning.md, discovery order)
1. Pain: exact TSP via branch-and-bound needs a **tight, cheap lower bound** at every node to
   prune; weak bounds (assignment relaxation alone, simple LP) leave huge trees.
2. A tour = a connected 2-regular spanning subgraph. Drop "2-regular," keep "spanning tree on
   the rest + one extra vertex with two edges" → **1-tree** (spanning tree on {2..n} + 2 edges
   at node 1). Min 1-tree is cheap: MST (greedy) + two cheapest edges at node 1. A tour IS a
   1-tree with all degrees 2, so min-1-tree cost ≤ OPT. A bound, but loose.
3. Wall: how to *tighten* without solving TSP? Observation (iii): adding π_i+π_j to every edge
   adds exactly 2Σπ_i to **every tour** (each vertex degree 2) — a constant shift — so argmin
   tour is invariant, but the min **1-tree** changes (its degrees aren't all 2). So
   min_{1-tree}[c_k + Σπ_i d_ik] − 2Σπ_i ≤ OPT for every π.
4. This is **Lagrangian relaxation**: dualize the n degree-2 constraints d_i=2 with multipliers
   π_i. w(π)=min_k[c_k + π·v_k], v_k,i = d_ik−2. C* ≥ w(π) ∀π; best bound = max_π w(π). w is a
   min over finitely many linear functions of π ⇒ concave, piecewise linear, **nonsmooth** at
   the kinks where the optimal 1-tree changes.
5. Maximize a concave nonsmooth function? First tries (Part I): column-generation LP and a
   steepest-ascent that increases w each step — both slow / iterations blow up with n. Wall.
6. Self-correction: at π, the active 1-tree k(π) gives v_k(π) with w(τ)−w(π) ≤ (τ−π)·v_k(π)
   (Lemma 1) — v_k(π) is a **subgradient**: degree residuals d_i−2. Step π ← π + t·(d−2): raise
   potentials at over-degree vertices (their edges get costlier → fewer chosen), lower at
   degree-1 vertices. Need NOT increase w, but (Lemma 2) for 0<t<2(w(τ)−w(π))/‖v‖² it moves
   **closer to the maximizer** — Fejér-monotone. That's the Agmon–Motzkin–Schoenberg relaxation
   method (Hoffman's pointer).
7. Step size: constant t (naive, what they ran) works (Thm 1: sup w(π^m) ≥ max w −
   ½t·limsup‖v‖², and ‖v‖²→small integer as 1-trees become tour-like). Better: Polyak
   t = λ(w̄−w(π))/‖d−2‖², ε≤λ≤2, w̄ an upper bound (heuristic tour). Warm start π^0 = −½(u+v)
   from the assignment-problem dual.
8. Land on code: min-1-tree + subgradient ascent loop; embed in branch-and-bound (include/
   exclude edges, branch when ascent stalls for p iterations).
