# Synthesis — Lin–Kernighan TSP heuristic

## Pain point / research question
Symmetric TSP: given an n×n symmetric distance matrix, find a min-length Hamiltonian tour.
Exact methods (Held–Karp branch-and-bound DP) blow up; largest reported ~64 cities. Heuristics
needed for larger. The state of the art in local search was Lin's 3-opt (1965): take a random
tour, repeatedly swap k=3 edges for 3 others if it shortens the tour, restart from many random
tours, keep the best. Croes 1958 = k=2 (inversion / 2-opt).

## The precise object
Local search over the space of tours. A "k-opt move" removes k edges of the current tour T and
adds k different edges so T stays a tour, accepting if shorter. A tour is k-opt(λ-opt in Lin's
term) if no such improving move exists. Higher k → stronger local optimum, but the cost of one
sweep is ~Θ(n^k) (you must consider all k-subsets). So you face: fixed small k = cheap but weak;
fixed large k = strong but Θ(n^k) is unaffordable, and you must pick k in advance with no way to
know the right value.

## The central difficulties (from the primary, §front matter, §intro, §1)
1. "Having to specify the value of k in advance is a serious drawback." Computational effort
   rises rapidly with k; can't know the best compromise. → want k chosen adaptively, per move.
2. If you build the swap element-by-element (decide x1,y1 then x2,y2 ...) the branching is huge.
   Need a strong pruning rule.

## The key ideas (the contribution), in derivation order
- **Generalize the fixed-k interchange to variable depth.** Don't fix k. Build the exchange one
  pair at a time (x_i out, y_i in), "element by element," choosing the most-out-of-place pair
  each step, and let a stopping rule decide the depth k. (Primary §intro: "substantial
  generalization of the interchange transformation.")
- **Make the move sequential / a chain.** Number the broken edges x_i and added edges y_i so they
  share endpoints: x_i=(t_{2i-1},t_{2i}), y_i=(t_{2i},t_{2i+1}), and the last added edge closes
  back to t_1: x_{i+1}=(t_{2i+1}, t_{2i+2}) etc. This forms an alternating trail. Performing such
  a numbered sequence converts T→T' as one move (Fig 1). Some moves can't be so numbered (Fig 2,
  nonsequential) but those are rare in practice.
- **Define per-step gain** g_i = |x_i| − |y_i| = c(broken) − c(added). Total gain
  G = Σ_{i=1}^k g_i = f(T) − f(T'). Individual g_i may be negative.
- **Positive-gain (cumulative) criterion.** This is the heart. Lemma: if Σ a_i > 0 then some
  cyclic permutation has *all partial sums positive*. Proof: pick the largest index where the
  prefix sum is minimal; rotating to start just after it makes every partial sum positive
  (two cases, j>that index and j≤it). Consequence: when searching for a profitable sequence we
  *only need to consider sequences whose every partial sum G_i = Σ_{j≤i} g_j is positive* — any
  profitable closed move has such a cyclic ordering, so requiring G_i>0 at every step loses no
  attainable improvement while pruning the search enormously. This is also the stopping rule:
  stop extending when G_i ≤ 0 (the running gain went non-positive) or G_i ≤ G* (best seen).
- **Feasibility / closing-up.** At each step, x_i is chosen so that the configuration can "close
  up" to a valid tour by joining the loose end t_{2i} back to t_1 — given y_{i-1}, the broken
  edge x_i that keeps closability is *uniquely determined*. y_i is then any near edge from the new
  endpoint with positive cumulative gain. Maintaining closability means at every depth we have an
  actual candidate tour T' to compare; G* tracks the best closing gain seen, and the realized move
  is the prefix k that maximized G*.
- **Choose y_i among nearest neighbors.** To make a large reduction, |y_i| should be small → in
  practice scan the 5 nearest neighbors of t_{2i} (primary §2B). Strong gain criterion plus this
  candidate list = small branching.
- **Disjointness:** x's and y's are kept disjoint (an edge once broken isn't re-added in the same
  move and vice versa) — "largely pragmatic," avoids loops/bugs, speeds it up.
- **Limited backtracking.** Only at levels i=1 and i=2: try alternative x_1 (the other tour
  neighbor of t_1), alternative y_1 (next nearest), alternative x_2 (allowed to be infeasible only
  at i=2 — the configuration may temporarily be two subtours that a later step repairs). Measured
  mean choice number is 1.2 (level1), 1.8 (level2); considering ~5 each is enough; backtracking
  deeper costs a lot for little gain. (Primary §"Backtracking".)
- **Result strength:** the local optima are at least 3-opt ("necessarily 3-opt in the sense of
  [Lin 1965]") yet obtained in far less time, because variable depth + gain pruning explores the
  high-k moves that matter without paying Θ(n^k).
- **Running time** grows ~ n^2.2 empirically per local optimum; ~n^2 in the abstract claim. 100
  city: <25 s one case; ~3 min for optimum w/ >95% confidence. (Front matter + §3.)
- The framework is generic: "find from set S a subset T minimizing f subject to criterion C" —
  same engine already applied to graph partitioning (Kernighan–Lin 1970).

## Why sequential + gain criterion beats fixed 2-opt/3-opt
- Fixed k forces an a-priori cost/quality tradeoff and Θ(n^k) per sweep. Variable depth lets the
  *first few* moves be deep (k often large with zero overshoot) and later ones shallow (2–7), per
  the data, so you get >=3-opt strength but pay roughly the 2-opt-ish branching because the gain
  criterion + neighbor lists keep the per-node fanout near 1.
- A single 2-opt or 3-opt move can't escape some basins that a depth-6 chain crosses; the chain is
  a path through intermediate non-improving (G_i may dip but stays positive) states that no fixed-k
  move can reach in one step, yet the gain criterion guarantees we never wander into clearly-losing
  territory.

## Antecedents (load-bearing)
- **Croes 1958** "A method for solving traveling-salesman problems," Oper. Res. 6:791–812. The
  "inversion" transformation = 2-opt: reverse a segment if it shortens the tour; iterate to an
  inversion-free tour; then ad-hoc manual "adjustment" steps (not mechanizable) to push toward
  optimum. Gap: weak (only 2-opt), and the final polishing needs human inspection.
- **Lin 1965** "Computer solutions of the TSP," BSTJ 44:2245–2269. Introduces **λ-optimality**:
  a tour is λ-opt if no λ edges can be replaced by λ others to shorten it. Theorems: 2-opt ⇔
  optimal-rel-to-inversion ⇔ non-self-intersecting (Euclidean); λ-opt ⊃ (λ+1)-opt classes
  (C_1 ⊇ C_2 ⊇ … ⊇ C_n); n-opt ⇔ optimal. Implements **3-opt** (remove a length-k section,
  reinsert as-is or inverted between two other cities; equivalently swap 3 links): every 3-opt tour
  is inversion-free, 3-opt is much stronger than 2-opt, P(3-opt tour is optimal) nontrivial. Method:
  generate many 3-opt tours from random starts, keep the best; a *reduction* scheme fixes links
  common to many local optima to shrink the problem. 4-opt: much more compute, little extra optimum
  probability. Per-local-optimum time ~Θ(n^3) (the (n choose 3) check-out). **Gap that motivates
  LK:** k fixed in advance; cost ~Θ(n^k); raising k helps quality but is unaffordable and you can't
  know the right k.
- **Held–Karp 1962/1970** exact DP / 1-tree bounds; cited as the exact-but-limited alternative
  (~64 cities). Krolak et al. man-machine interaction; cited as the human-in-loop alternative.

## Self-account
Kernighan CHM oral history (2017): partnership lineage (graph partitioning 1968 → TSP; "Shen
amazing for insight into how combinatorial problems ought to work; I programmed better"),
NP-completeness timing, FORTRAN. Confirms variable-depth interchange came from the
graph-partitioning work first. Does NOT contain the k-opt/gain derivation — that is reconstructed
from primary + Lin 1965.

## Code grounding
- Canonical simplified-LK structure (Mahéo blog; kikocastroneto/lk_heuristic): main loop over t1,
  the two tour-neighbors t2 (=x1), nearest-neighbor t3 (=y1, g1>0), then recursive chooseX/chooseY
  building X (broken) and Y (added) with cumulative gain Gi>0, closing test (relink t_2i→t1), apply
  if improving tour, else extend. Backtracking via the loops. Tour reconstruction = segment reversal
  (2-opt style step composing the chain).
- Final code in deliverables: a self-contained simplified LK (the "Or-2opt"/sequential-chain core),
  Euclidean, tour as array + position index, segment reversal for the 2-opt-style step, neighbor
  lists, positive-gain chain with limited depth. Faithful to that structure, runnable.
