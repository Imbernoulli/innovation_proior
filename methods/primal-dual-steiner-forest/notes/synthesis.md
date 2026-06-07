# Synthesis — Primal–Dual 2-approximation for Steiner Forest

## Sources actually read this run (refs/)
- `agrawal-klein-ravi-sicomp95.txt` — PRIMARY. The first 2-approx for generalized Steiner (R-join).
  Combinatorial presentation: grow BFS trees simultaneously from all sites, accumulate level-cuts
  ("requirement cuts") as a 2-packing, merge trees when they collide ("when trees collide"),
  deactivate+contract a tree when all its site-pairs are internally satisfied. Performance bound
  2(1 - 1/k) via charging network cost to cuts; Lemma 3.2 lower bound (min network >= 1/2 max
  2-packing of requirement cuts); approximate min-max equality (Thm 1.3). Does NOT use LP duality
  explicitly. Bipartite transform (split each edge by a midpoint) to make cuts edge-disjoint.
- `goemans-williamson-1995-constrained-forest.txt` — PRIMARY (the clean primal-dual reframing).
  IP (cut-cover) with proper function f; LP relaxation; dual (D) = max sum f(S) y_S subject to
  sum_{S: e in delta(S)} y_S <= c_e. Main algorithm (Fig 1): F empty; all components singletons;
  d(v)=0; while exists active component (f(C)=1): pick edge e=(i,j) across two components
  minimizing reduced cost eps = (c_e - d(i) - d(j))/(f(C_p)+f(C_q)); add e; raise d(v) by
  eps*f(C) for v in active C (== raising y_C by eps for active C); merge. Final edge-removal
  (reverse-delete-style cleanup): keep only edges needed so every component N has f(N)=0.
  Analysis Thm 2.4: cost(F') = sum_e c_e = sum_S y_S |F' ∩ delta(S)|; show by induction that
  this <= (2 - 2/|A|) sum_S y_S. Per-iteration: build forest H on current components with edges
  F' ∩ delta(C); no leaf is inactive (else that edge would have been removed); so
  sum over active components of |F' ∩ delta(C)| = sum of active-vertex degrees in a forest
  <= 2(#vertices) - 2(#inactive, since each inactive has degree >= 2) <= 2|active|. "Moats" of
  Junger-Pulleyblank = the dual variables drawn as rings around components.
  O(n^2 log n) implementation via time-T priority queue.
- `williamson-primal-dual-survey.txt` — SELF-ACCOUNT / reflective (David Williamson, co-inventor of
  the GW framework). Narrates the development AS a sequence of motivated modifications to the
  Bar-Yehuda–Even primal-dual skeleton:
    (1) hitting-set IP + dual; Hochbaum's "take all tight edges" -> f-approx;
    (2) Bar-Yehuda–Even: don't need OPTIMAL dual, grow any feasible dual (Fig 1);
    (3) reverse-delete (GW introduced it; refined by Klein–Ravi and Saran–Vazirani–Young to
        delete in REVERSE order of addition) -> Theorem 3 bounds by minimal augmentations;
    (4) shortest path / branching come out as minimal-violated-set = the component containing s;
    (5) THE STEINER WALL: single-violated-set choice gives only k-approx (star example
        s=s1=...=sk; D = all k star edges hits delta({s}) k times). Fix: average over the k+1
        minimal violated sets -> 2k/(k+1) ≈ 2. "This leads to the following idea: choose multiple
        violated sets and increase their dual variables simultaneously and uniformly." (Fig 3).
    Theorem 4 + Theorem 5 = the degree-counting analysis (red=active, blue=inactive; no blue leaf).
  This is the omitted-reasoning backbone: WHY reverse-delete, WHY uniform simultaneous growth.
- `goemans-williamson-survey-book-ch4.txt` — reflective survey chapter (same lineage, fuller moat
  picture and tightness).
- `dartmouth_steiner.txt` (Hulse/Chakrabarty) — clean modern moat exposition: blocking sets B,
  primal/dual, PD-STEINER-FOREST pseudocode (active set A of blocking sets, raise y_S uniformly
  until edge tight, merge, reverse-delete), and the F[A_t]-is-a-forest proof.
- `jhu_dinitz_steiner.txt` — Dinitz lecture, same algorithm/proof, different framing.

## Self-account? YES (reflective).
The GW survey + Williamson survey are reflective/expository self-accounts by the framework's authors;
they explicitly narrate the motivating reasoning the theorem-first papers compress: the moat picture,
why reverse-delete, and especially the "single violated set gives k, average gives 2 -> grow all
active duals at once" turn. AKR itself is the combinatorial primary; GW is the primal-dual reframing.
No dramatic anecdote; rich omitted-reasoning. Added an entry to SELF_ACCOUNT_SOURCES.md.

## The problem
Steiner Forest: undirected G=(V,E), costs c_e >= 0, k terminal pairs (s_j,t_j). Find min-cost F⊆E
with s_j–t_j connected for all j. NP-hard (generalizes Steiner tree). Want factor 2 in poly time.

## Pain points / landscape at the time
- Steiner TREE has 2-approx heuristics (MST-based, Takahashi–Matsuyama, Kou–Markowsky–Berman) but
  they assume ONE connected terminal set. Steiner FOREST's optimal solution can be a disconnected
  forest; an MST/Steiner-tree heuristic over all terminals can be arbitrarily worse than OPT.
- LP-relaxation + rounding: the cut-cover LP has exponentially many constraints; Hochbaum-style
  "solve LP, take tight edges" needs to actually solve a huge LP, and gives f-approx where f =
  max cut size = bad (k). Want to avoid solving the LP at all.
- Bar-Yehuda–Even primal-dual: grow ONE dual at a time, choose a single violated set. For Steiner
  forest, choosing one violated component gives only k-approx (the star counterexample).
- Goemans–Bertsimas tree heuristic handled only r_ij = min(r_i,r_j) special form, not arbitrary
  0/1 pairs.

## Design-decision -> why
- LP cut-cover formulation (constraint sum_{e in delta(S)} x_e >= 1 for every S separating a pair):
  connectivity of s_j–t_j <=> every separating cut has an edge (max-flow/min-cut), so this exactly
  captures feasibility. WHY not flow LP: flow LP is compact but the cut LP's DUAL is the moat-packing
  that drives the algorithm and the analysis. The exponential primal is fine because we never solve
  it; we grow the dual.
- Dual y_S = "moat" widths; constraint sum_{S: e in delta(S)} y_S <= c_e: a feasible dual is a
  lower bound (weak duality) so beating 2*dual beats 2*OPT. We construct dual + primal together.
- Grow ONLY active components' duals (a component C is active iff it separates some still-unconnected
  pair, i.e. f(C)=1). WHY: raising an inactive component's dual is wasted — it can't help connect
  anything and only eats edge budget; the analysis needs inactive components to NOT be charged.
- Grow ALL active duals SIMULTANEOUSLY and UNIFORMLY by the same eps. WHY (the key turn): growing one
  at a time / choosing a single violated set gives k-approx (star example). Averaging the charge over
  ALL minimal violated sets gives 2k/(k+1) < 2. Simultaneous uniform growth makes every active moat
  share the charge, which is exactly what the degree-counting (avg degree in a forest < 2) bounds.
- eps = min reduced cost so exactly one edge goes tight per iteration; add that edge, merge its two
  components. (Continuous "moat growth" discretized to events.)
- Reverse-delete cleanup: after F connects every pair, scan edges in REVERSE order of addition and
  drop any whose removal keeps all pairs connected. WHY reverse order: it guarantees that for the
  cut delta(S) of an active component at iteration t, the kept edges form a MINIMAL augmentation, so
  the induced graph on the components-at-time-t is a FOREST, giving |F' ∩ delta(C)| summed over
  active = avg degree * #active < 2|active|. Without cleanup the star solution keeps k edges across
  one moat and the bound fails.
- Output is a forest (break any cycle).

## The 2-approx proof (must be lived in reasoning.md)
cost(F') = sum_{e in F'} c_e = sum_{e in F'} sum_{S: e in delta(S)} y_S   (each kept edge is tight)
         = sum_S y_S * |F' ∩ delta(S)|   (swap order)
Write y_S = sum over iterations t with S active of eps_t. So
cost(F') = sum_t eps_t * sum_{S active at t} |F' ∩ delta(S)|.
dual value = sum_S y_S = sum_t eps_t * |A_t|   (|A_t| = #active components at t).
So suffices: for each t, sum_{S in A_t} |F' ∩ delta(S)| <= 2|A_t|.
Per iteration build H = contract each current component to a node, edges = F' edges across
components. H is a forest (reverse-delete => minimal augmentation => no cycle). Color a node red
if its component is active (in A_t), blue if inactive. |F' ∩ delta(C)| = deg_H(C).
No blue node is a leaf: a degree-1 blue (inactive) node's single incident kept edge e would be
removable (the pair it helps was already connected through the rest), contradicting reverse-delete's
minimality. So every blue node has degree >= 2. Then
sum_{red} deg = sum_all deg - sum_blue deg <= 2(#nodes - 1) - 2(#blue) <= 2(#red).
Hence cost(F') <= sum_t eps_t * 2|A_t| = 2 * dual <= 2 * OPT.  Factor exactly 2 (AKR's 2-2/k is the
finer bound counting that H has <= |nodes|-1 edges).

## Code grounding (canonical structure from GW Fig 1 + Dartmouth/Dinitz pseudocode)
Maintain: union-find of vertices into components; per-component active flag (separates some
unconnected pair); per-component dual sum y_C (moat width); per-edge "tightness": slack_e = c_e -
sum of moat widths of components touching e. Each iteration: among edges crossing two distinct
components, the time to tightness = slack_e / (#active endpoints). Pick min, advance all active
moats by that eps, add the now-tight edge, merge. Recompute active flags (a merged component is
active iff it still separates some unconnected pair). Stop when no active component. Reverse-delete.
This is a faithful, real, runnable implementation — not invented.
