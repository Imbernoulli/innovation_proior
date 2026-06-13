# DBSCAN — synthesis (Phase 1.5)

## Method identity (from edit.py + task)
- Task `ml-clustering-algorithm`, baseline `dbscan`. edit.py wraps `sklearn.cluster.DBSCAN`
  with eps/min_samples heuristics. The CANONICAL method is:
  **DBSCAN — A Density-Based Algorithm for Discovering Clusters in Large Spatial
  Databases with Noise**, Ester, Kriegel, Sander, Xu, KDD-96, pp. 226-231.
  Pre-arXiv classic (no arXiv id).
- The trace is the PAPER derivation (density-based clustering), NOT the task scaffold's
  eps grid-search tricks. The scaffold's eps=0.22 etc. is posterior task tuning, out of frame.

## Sources retrieved this run (all read)
1. PRIMARY: refs/dbscan-1996-kdd.pdf — KDD'96 paper, read all 6 pages (full). Has Defs 1-6,
   Lemmas 1-2, ExpandCluster pseudocode, k-dist heuristic.
2. RETROSPECTIVE (authors' own re-explanation): refs/dbscan-revisited-2017.pdf
   (Schubert, Sander, Ester, Kriegel, Xu, TODS 2017) — the WHY, parameter guidance,
   complexity correction. Third-party-style explainer + author retrospective.
3. CANONICAL CODE: code/sklearn_dbscan.py + code/sklearn_dbscan_inner.pyx (BFS/DFS labeling).
4. Explainers: sklearn plot_dbscan demo (notes/), WebSearch summaries (Frey notes, DataCamp,
   STHDA), Wikipedia k-medoids (for PAM/CLARANS baseline math).
NO author-self-account / Nobel-style narrative exists; the 2017 retrospective is the closest
(used for the omitted reasoning, in-frame). Noted.

## Pain point / research question
- Cluster spatial DB (earth-observation, astronomy, etc.); millions of points.
- Requirements the paper states (all three at once, none of the prior art has all):
  (1) MINIMAL domain knowledge to set parameters (don't need to know k in advance);
  (2) clusters of ARBITRARY shape (spherical, linear, elongated, drawn-out, non-convex) —
      because spatial clusters genuinely are these shapes;
  (3) GOOD efficiency on large DBs (>> a few thousand points).
- Implicit: handle NOISE/outliers explicitly (real spatial data has noise).

## Background / load-bearing concepts
- "What makes us recognize clusters" = within a cluster, the local density of points is
  considerably higher than outside; density in noise areas is lower than in any cluster.
  This is the KEY OBSERVATION the whole method is built to formalize.
- Distance function dist(p,q): the neighborhood shape follows the metric (Euclidean → ball;
  Manhattan → rectangle). Works in any k-dim feature space, any dist.
- Spatial access methods: R*-tree (Beckmann et al. 1990) supports efficient region queries;
  height O(log n); a "small" region query traverses few paths. This is the index that makes
  neighborhood queries cheap. (RETROSPECTIVE correction: O(log n) per query is an *informal
  average-case* claim, never a worst-case guarantee — worst case Θ(n) per query.)

## Baselines (prior art + the gap each leaves)
- **k-means (Lloyd 1957/MacQueen 1967), k-medoid/PAM (Kaufman & Rousseeuw 1990).**
  Partitioning: minimize sum of distances to k representatives. k-means rep = centroid
  (gravity center); k-medoid rep = a data object (medoid). Assign each point to nearest rep.
  Objective E = Σ_clusters Σ_{p in cluster} dist(p, rep). Two steps: pick k reps minimizing
  objective; assign each object to closest rep → partition = Voronoi diagram, each cluster in
  one Voronoi cell. GAP: (a) need k in advance; (b) the partition is a Voronoi tessellation,
  so every cluster is CONVEX — cannot represent a non-convex / elongated / nested shape;
  (c) no noise concept — every point is forced into a cluster, outliers distort the reps.
- **CLARANS (Ng & Han 1994).** Improved k-medoid for larger DBs via randomized search:
  view clustering as searching a graph whose nodes are sets of k medoids; neighbors differ in
  one medoid; do randomized hill-climbing (sample a bounded number of neighbor swaps, move to
  a better one, restart). More effective/efficient than PAM/CLARA. To find a "natural" k,
  Ng&Han propose running CLARANS for each k=2..n and picking max silhouette coefficient
  (Kaufman & Rousseeuw 1990). GAP: (a) still partitioning → convex clusters only;
  (b) run time prohibitive on large DBs (silhouette-for-each-k implies O(n) CLARANS calls;
  CLARANS itself is ~O(n^2)); (c) assumes all objects fit in main memory; (d) no noise model.
- **Hierarchical (agglomerative/divisive; Kaufman & Rousseeuw 1990).** Build a dendrogram by
  iteratively merging/splitting; no k needed, but need a TERMINATION condition (e.g. Dmin
  between clusters) — hard to set: small enough to separate all clusters yet large enough not
  to split a cluster. Ejcluster (Garcia et al. 1994): two points in same cluster if you can
  "walk" between them by sufficiently-small steps; very good on non-convex clusters, derives
  its own termination, BUT O(n^2) due to all-pairs distance — only acceptable for small data.
- **Grid/density-histogram (Jain 1988).** Partition the space into nonoverlapping cells,
  build multidim histograms; high-count cells are candidate cluster centers, cluster
  boundaries fall in histogram "valleys". CAN find arbitrary shapes. GAP: storing/searching
  multidim histograms is enormous in space & run time; performance crucially depends on the
  (arbitrary) cell size.
- **Summary gap that motivates the move (state as OBSERVED limitation, not prescription):**
  partitioning methods (k-means/medoid/CLARANS) need k and produce convex Voronoi cells with
  no noise; hierarchical/Ejcluster avoid k but cost O(n^2) or need a hard-to-set termination;
  grid-histograms find shapes but are space/time heavy and cell-size sensitive. None
  simultaneously: minimal params + arbitrary shape + efficient + explicit noise.

## The method (the discovery in reasoning.md)
Formalize "high local density within a cluster, low between". A point's local density =
how many points lie within radius Eps of it = |N_Eps(p)|, N_Eps(p)={q in D : dist(p,q)<=Eps}.
- DEF 1: Eps-neighborhood N_Eps(p) = {q∈D | dist(p,q) ≤ Eps}.
- Naive "every point in a cluster has ≥ MinPts neighbors" FAILS: border points of a cluster
  genuinely have fewer Eps-neighbors than interior points. So distinguish CORE points
  (|N_Eps| ≥ MinPts) from BORDER points (fewer, but inside a cluster). If you raise MinPts
  to include borders, you'd never get a stable threshold (border density isn't characteristic).
- Fix: require, for every cluster point p, that there EXISTS a point q in the cluster with p
  in N_Eps(q) AND |N_Eps(q)| ≥ MinPts. (a border point "belongs" via a nearby core point.)
- DEF 2: p directly density-reachable from q (wrt Eps,MinPts) if (1) p ∈ N_Eps(q) and
  (2) |N_Eps(q)| ≥ MinPts (core condition on q). Symmetric for two core points; asymmetric
  if q core, p border (core→border yes, border→core no).
- DEF 3: p density-reachable from q if a chain p_1=q,...,p_n=p with p_{i+1} directly
  density-reachable from p_i. Transitive, NOT symmetric (chain of core points; endpoints can
  be border). Canonical extension of DDR.
- Two border points of the same cluster may NOT be density-reachable from each other (neither
  is core). Need a symmetric relation linking them → density-connectivity.
- DEF 4: p density-connected to q if ∃ core point o with both p and q density-reachable from
  o. Symmetric; reflexive for density-reachable points.
- DEF 5: a CLUSTER C wrt Eps,MinPts is a non-empty subset of D s.t.
  (1) Maximality: ∀p,q: if p∈C and q density-reachable from p, then q∈C.
  (2) Connectivity: ∀p,q∈C: p density-connected to q.
- DEF 6: NOISE = points in D belonging to no cluster.
- Note: a cluster contains ≥ MinPts points (it has at least one point p, density-connected to
  itself via some core o, so o satisfies the core condition → |N_Eps(o)| ≥ MinPts).
- LEMMA 1 (a cluster is GENERATED by any of its core points): let p∈D, |N_Eps(p)|≥MinPts.
  Then O = {o | o density-reachable from p} is a cluster wrt Eps,MinPts. (proof: O nonempty
  (p∈O since DR is reflexive when p core); maximality from transitivity of DR; connectivity
  via o=p.) ⇒ each cluster is exactly the set of points density-reachable from any one of its
  core points; the cluster is uniquely determined by any of its core points.
- LEMMA 2: let C be a cluster, p∈C with |N_Eps(p)|≥MinPts. Then C = {o | o density-reachable
  from p wrt Eps,MinPts}. ⇒ ALGORITHM: pick any unprocessed point; if it's core, retrieve all
  points density-reachable from it (one region query at a time, growing a seed list) = the
  cluster; if it's not core (border or noise), mark NOISE provisionally and move on (it may
  later be reclassified into a cluster when reached from a core point's expansion).

## Algorithm (ExpandCluster, from paper) → maps 1:1 to sklearn
- Global Eps, MinPts (same for all clusters — simplicity; "thinnest cluster" sets them).
- For each unclassified point p: regionQuery(p,Eps). If |seeds| < MinPts → mark p NOISE,
  return False (not a cluster seed). Else assign all seeds the new ClId, remove p from seeds,
  and WHILE seeds nonempty: take currentP, result=regionQuery(currentP,Eps); if
  |result|>=MinPts (currentP is core) then for each q in result with ClId in
  {UNCLASSIFIED,NOISE}: if UNCLASSIFIED append q to seeds; set q.ClId=ClId. Pop currentP.
- Exactly ONE region query per point (key for complexity). Runtime = O(n · cost of region
  query). With R*-tree and small queries the paper claims average O(n log n); worst case
  O(n^2) (RETROSPECTIVE). Border-point ambiguity: a border reachable from two clusters is
  assigned to whichever discovers it first (rare; little interest).
- sklearn: radius_neighbors for all points (bulk), mark core (n_neighbors>=min_samples),
  dbscan_inner does DFS over core points labeling connected components, absorbs border points
  (labels them) but does NOT expand non-core neighborhoods. Noise stays -1. EXACT match.

## Parameter heuristics (the WHY for the defaults)
- MinPts smooths the density estimate. Paper: eliminate MinPts by fixing MinPts=4 for all 2D
  databases (k-dist graphs for k>4 don't differ much from 4-dist and cost more). RETROSPECTIVE:
  general rule MinPts = 2·dim (Sander et al.); increase for noisy/high-dim/duplicate data.
- Eps from the SORTED k-dist graph: for each point compute distance to its k-th NN (k=MinPts).
  Sort all points descending by k-dist. The first "valley"/knee = threshold: points with
  larger k-dist (left of knee, sparse) → noise; smaller (right) → in some cluster. Set Eps =
  k-dist of the threshold point. Choosing the THINNEST cluster's density as the global Eps,MinPts
  means everything denser is captured. Interactive: hard to auto-detect the valley.
- CAUTION (k vs MinPts off-by-one): the k-NN distance does NOT count the query point, but the
  Eps-neighborhood (region query) DOES include the query point. So k corresponds to MinPts-1,
  i.e. MinPts = k+1. (RETROSPECTIVE 4.1 footnote.) Choose Eps as SMALL as possible.
- Red flags for degenerate result (RETROSPECTIVE 4.2): noise fraction should be 1-30%;
  largest connected component should not exceed ~20-50% of clustered points (too-large Eps).

## Cluster model = density-level-set / KDE view (RETROSPECTIVE 2.1)
- DBSCAN = a simple minimum-density-level-set estimate: uniform kernel KDE with bandwidth
  h=Eps and density threshold MinPts/n; core points = points where the estimate exceeds the
  threshold; clusters = connected components of the super-level set; valleys (low density) =
  noise. This is the deeper "why it works".

## Design decisions → why
- Density via count-in-radius (not distance-to-k-NN, not histogram): cheapest, metric-agnostic,
  one region query reuse for both core test and expansion.
- Core/border/noise trichotomy: needed because cluster boundaries are genuinely lower density;
  a single threshold on all points would either fragment clusters (high MinPts) or merge noise
  (low MinPts). The core condition anchors clusters; border points inherit membership.
- density-reachable (asymmetric, chain): lets a cluster be ANY connected high-density region —
  arbitrary shape — by transitively walking core→core; convexity never assumed.
- density-connected (symmetric): needed so two border points of one cluster count as one
  cluster (DEF 5 connectivity); pure density-reachability would not relate them.
- Global Eps,MinPts: simplicity; sets the THINNEST cluster's density. Cost: two clusters of
  different density that are < Eps apart can merge; recursive DBSCAN with higher MinPts can
  separate (rare, easily detected). Default-able.
- One region query per point + index: gives the efficiency requirement (vs O(n^2) Ejcluster
  / CLARANS), the whole reason to formalize density this way and lean on R*-trees.
- No k needed; #clusters falls out of the data. Explicit noise (label -1).

## Final code grounding
sklearn DBSCAN (BaseEstimator, ClusterMixin): NearestNeighbors(radius=eps).radius_neighbors,
core = n_neighbors>=min_samples, dbscan_inner DFS labeling, labels -1 = noise. I will write a
self-contained NumPy/sklearn version filling the clustering-estimator scaffold (fit→labels_,
predict). custom_distance = Euclidean (the metric that shapes N_Eps).

## In-frame discipline
- Never name "DBSCAN" as a paper; may name the method as the thing being built (mainly answer.md).
- Cite ancestors (k-means/PAM, CLARANS Ng&Han 1994, Ejcluster Garcia 1994, Jain 1988,
  R*-tree Beckmann 1990, Kaufman&Rousseeuw 1990) freely.
- Context scaffold = generic clustering estimator (fit/predict on X, sets labels_), ONE empty
  slot for the assignment rule; no density/core/eps naming pre-given.
