# HDBSCAN synthesis (for Phase 2)

## Method identity
- Target: HDBSCAN* — "Density-Based Clustering Based on Hierarchical Density Estimates",
  Campello, Moulavi, Sander, PAKDD 2013 (LNCS 7819, pp.160-172). Extended journal version
  (the definitive 51-page treatment, used as primary here): Campello, Moulavi, Zimek,
  Sander, "Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier
  Detection", ACM TKDD 10(1), Article 5, 2015. No arXiv (pre-arXiv classic / paywalled
  journal; open copy from USP repository).
- The MLS-Bench baseline calls `sklearn.cluster.HDBSCAN(min_cluster_size, min_samples=5,
  cluster_selection_method="eom")`. So the canonical method to reconstruct is HDBSCAN with
  Excess-of-Mass flat extraction.

## Pain point / research question
- DBSCAN (Ester et al. 1996) and density-based flat methods need a *global* density
  threshold (radius eps). One eps cannot simultaneously capture clusters of very different
  densities; the choice is critical and the result is brittle.
- A single horizontal cut through any density hierarchy = a single global threshold, with
  the same limitation. Need a *flat* partition that draws clusters from *local* cuts at
  *different* density levels, automatically, with as few parameters as possible.

## Ancestors (load-bearing)
1. **Single-linkage / MST hierarchical clustering** (Johnson 1967; Jain & Dubes 1988).
   Remove edges from complete graph in decreasing weight order = build dendrogram; fastest
   via MST (remove MST edges in decreasing weight). GAP: chaining — a thin bridge of points
   merges genuinely separate dense clusters; no noise notion.
2. **DBSCAN / DBSCAN*** (Ester, Kriegel, Sander, Xu 1996). Core point: |N_eps(p)| >= MinPts.
   Clusters = density-connected components of core points; non-core = noise (DBSCAN also has
   border objects; DBSCAN* drops them for a clean level-set interpretation). GAP: one global
   (eps, MinPts); border objects break the symmetric/statistical interpretation.
3. **OPTICS** (Ankerst, Breunig, Kriegel, Sander 1999). core-dist(p) = MinPts-th NN distance;
   reachability-dist(o,p) = max(core-dist(p), d(p,o)) — ASYMMETRIC (depends on p). Outputs a
   reachability plot (ordering + bar heights); clusters = valleys. GAP: reachability is
   directional, so it only *approximately* relates to DBSCAN; extracting a flat clustering
   from the plot (steep areas / xi) is heuristic, and a single threshold ~ DBSCAN again.
4. **Hartigan density-contour clusters / tree** (Hartigan 1975). Density-contour cluster at
   level lambda = maximal connected component of the level set {x : f(x) >= lambda}. Vary
   lambda -> nested tree. The formal model HDBSCAN realizes for *all* lambda at once.
5. **Excess of mass** (Hartigan 1987; Muller & Sawitzki 1991; Stuetzle & Nugent 2010).
   E(C) = integral over C of (f(x) - lambda_min(C)) dx. Monotone along a branch (parent
   contains children), so can't compare nested clusters directly -> relative excess of mass.
6. **Mutual reachability distance** (Lelis & Sander 2009): the *symmetric* version of OPTICS
   reachability — d_mreach(p,q) = max{core(p), core(q), d(p,q)}. This is what makes the
   single-linkage ↔ DBSCAN* equivalence exact.

## Core derivation chain (what reasoning.md must walk)
1. **Core distance** d_core(x_p) = distance from x_p to its m_pts-NN (including itself).
   = min radius eps s.t. |N_eps(x_p)| >= m_pts. Density estimate f ≈ 1/d_core (K-NN density,
   K=m_pts). So lambda = 1/eps is a density level.
2. **Want every DBSCAN* clustering for all eps at once.** DBSCAN* w.r.t. (m_pts, eps): two
   core objects p,q are eps-reachable iff p∈N_eps(q) and q∈N_eps(p), i.e. iff
   max{core(p),core(q),d(p,q)} <= eps. Define **mutual reachability distance**
   d_mreach(p,q) = max{core(p), core(q), d(p,q)}. Then "p,q eps-reachable" ⟺ d_mreach <= eps.
3. **Mutual reachability graph** G_mpts: complete graph, edge weight = d_mreach.
   Removing edges with weight > eps and taking connected components of core objects gives
   exactly DBSCAN*(m_pts, eps) clusters (Prop 3.4). Removing edges in decreasing weight
   order = single-linkage = nested family of all DBSCAN* clusterings. KEY UNIFICATION:
   DBSCAN* over all eps = single-linkage on the mutual-reachability-transformed space.
4. **Build via MST**, not full graph: single-linkage = remove MST edges in decreasing
   weight order. To also encode WHEN an isolated object goes from core (dense) to noise, add
   a **self-edge** to each vertex with weight = its core distance (extended MST, MST_ext).
   Self-loop weight <= all incident edges (since d_mreach >= core of both endpoints).
5. **Hierarchy** (Alg 1): sort 2n-1 edges, remove in decreasing weight; relabel components.
   This is the full dendrogram of all DBSCAN* clusterings.
6. **Simplification / condensation** with **min_cluster_size** (m_clSize): when a cluster
   splits, a child with < m_clSize points is "spurious" — it's points *falling out as
   noise* (cluster shrinks), NOT a true split. A true split = >=2 children each >= m_clSize.
   Only true splits / disappearances are kept -> condensed cluster tree (few significant
   clusters). Setting m_clSize = m_pts makes m_pts a single parameter (smoothing + min size).
7. **Cluster stability = relative excess of mass.** lambda = 1/eps. Cluster C_i is born at
   lambda_min(C_i) (when it first appears). Each point x_j leaves C_i at lambda_max(x_j,C_i)
   (becomes noise or falls into a child). Discrete relative excess of mass:
   S(C_i) = Σ_{x_j∈C_i} (lambda_max(x_j,C_i) - lambda_min(C_i))
          = Σ_{x_j∈C_i} (1/eps_min(x_j,C_i) - 1/eps_max(C_i)).
   Implementation: stability(parent) = Σ over child edges of (lambda_edge - lambda_birth) *
   child_size, lambda_birth = lambda at which the cluster appears (min lambda of its edges).
   Relative (subtract birth) so nested clusters are comparable; raw excess of mass is
   monotone along a branch and can't compare parent vs children.
8. **Optimal flat extraction = EOM** (Alg 3, dynamic program). Maximize total stability of a
   set of clusters s.t. exactly one selected per root-to-leaf path (no nesting). DP recursion
   Ŝ(C_i) = S(C_i) if leaf; else max{ S(C_i), Ŝ(C_il)+Ŝ(C_ir) }.
   Bottom-up: if sum of children's propagated stability > node's own stability, DESELECT node
   (keep descendants), set node's propagated stability = children sum; else SELECT node and
   deselect its whole subtree. Globally optimal, O(num_clusters).
9. (Extensions, mention but not the focus) GLOSH outlier score
   = 1 - eps_max(x_i)/eps(x_i); semi-supervised extraction by replacing stability with
   fraction of satisfied should-link/should-not-link constraints + virtual noise nodes.

## Canonical code grounding (scikit-learn-contrib/hdbscan + sklearn.cluster.HDBSCAN)
- core_distances = np.partition(D, min_points, axis=0)[min_points]  (m_pts-th smallest col).
- mutual_reachability: np.maximum(D, core[None,:]) then np.maximum(result, core[:,None]).
- MST: Prim's (mst_linkage_core), then sort edges, union-find `label` -> single-linkage tree
  in scipy [left,right,dist,size] format.
- condense_tree(hierarchy, min_cluster_size): the 4-case split logic (both>=mcs -> true
  split, new labels; both<mcs -> whole node dies, points fall out; one<mcs -> survivor keeps
  parent label, small side's points fall out at lambda=1/dist).
- compute_stability: birth[cluster] = min lambda of its edges; stability += (lambda - birth) *
  child_size.
- get_clusters EOM: for node in sorted-desc (exclude root): subtree = Σ children stability;
  if subtree > stability[node]: deselect node, stability[node]=subtree; else select node,
  deselect subtree.
- sklearn.cluster.HDBSCAN defaults: min_cluster_size=5, min_samples=None(->min_cluster_size),
  cluster_selection_method='eom', metric='euclidean', alpha=1.0. Noise label = -1.

## Design-decision → why
- **Mutual reachability (max of three) vs raw distance**: pushes sparse points apart (each
  point at least its core distance from anything), so single-linkage chaining through sparse
  noise bridges is suppressed; dense points unchanged. SYMMETRIC (unlike OPTICS) -> exact
  DBSCAN* equivalence.
- **Self-edges = core distance**: encode the level at which an isolated dense object becomes
  noise; without them single-linkage on transformed space loses the core/noise distinction.
- **min_cluster_size**: separate "shrink (lose noise)" from "true split"; prevents the
  hierarchy from registering every single-point departure as a new cluster; standard runt
  pruning (Stuetzle).
- **lambda = 1/eps**: density level; makes "longer-lived = denser = more prominent" precise.
- **relative excess of mass (subtract birth)**: raw EoM monotone along branch (parent >=
  children always), useless for nesting; subtracting birth gives a comparable per-cluster
  prominence.
- **EOM DP over horizontal cut**: lets selected clusters come from DIFFERENT density levels
  (local cuts), which is the whole point — a global cut can't capture varied-density clusters.
- **m_pts the only required parameter**: it's a classic smoothing factor for K-NN density;
  the method is robust to it because density is only used to discriminate noise vs non-noise.

## Sources captured
- refs/hdbscan-tkdd-2015.pdf (+ .txt.part1/part2) — PRIMARY (full text read for §2,3,5,6).
- refs/dbscan-kdd-1996.pdf (+ .txt) — ancestor, full read.
- OPTICS definitions via Wikipedia explainer (core-dist, reachability-dist asymmetry).
- hdbscan.readthedocs how_hdbscan_works — third-party explainer (mutual reachability,
  condensed tree, lambda stability, EOM).
- code/: hdbscan_.py, _hdbscan_linkage.pyx, _hdbscan_tree.pyx, _hdbscan_reachability.pyx
  (canonical scikit-learn-contrib/hdbscan); sklearn.cluster.HDBSCAN signature verified.
- No author self-account found (checked SELF_ACCOUNT_SOURCES.md; none for clustering/density).
