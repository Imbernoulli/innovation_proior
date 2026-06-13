# Synthesis — k-means (Lloyd batch iteration + k-means++ seeding)

## Method identification (from the MLS-Bench edit)
The `kmeans.edit.py` baseline calls `sklearn.cluster.KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)`,
and `custom_distance` is plain Euclidean. sklearn's `KMeans` default `init='k-means++'`. So the canonical
published method = **Lloyd's batch k-means iteration** (the assignment/centroid-update alternation, minimizing
within-cluster sum of squares / inertia) **seeded by k-means++ (D² weighting)**, with `n_init` restarts keeping
the lowest-inertia run. The trace lands on exactly this.

## Three required sources (all retrieved & read this run)
1. PRIMARY (the discovery being reconstructed) — **k-means++: The Advantages of Careful Seeding**, Arthur &
   Vassilvitskii, SODA 2007. `refs/kmeanspp_arthur.pdf`, text in `notes/kmeanspp_text.txt`. Read fully incl.
   Lemmas 2.1, 3.1, 3.2, 3.3, Theorem 3.1 (E[φ] ≤ 8(ln k + 2) φ_OPT), the matching Ω(log k) lower bound (§4),
   and the D^ℓ generalization (§5). (Empirical §6 = proposed-method eval, excluded.)
2. ANCESTORS:
   - **Lloyd, "Least Squares Quantization in PCM"** (IEEE Trans. IT, 1982; presented 1957). `refs/lloyd1982.pdf`.
     Necessary conditions for an optimum quantizer: (14) each quantum = center of mass of its cell; (15)/(16)/(17)
     each cell = nearest-quantum (Voronoi) region with boundaries bisecting adjacent quanta. "Method I": impose
     (14) and (17) alternately → monotone-decreasing noise sequence (20) N(ρ^(1)) ≥ N(ρ^(2)) ≥ ... This IS the
     two-step alternation. "Moment of inertia minimized at center of mass" = the key identity.
   - **MacQueen, "Some methods for classification and analysis of multivariate observations"** (5th Berkeley
     Symp., 1967). `refs/macqueen1967.pdf`. Coins "k-means"; objective W(x)=Σ_i ∫_{S_i}|z−x_i|² dp(z) (within-
     class variance); minimum-distance partition S_i(x) (eq 2.1/2.2); ORIGINAL k-means is the *online/sequential*
     running-mean update x_i^{n+1}=(x_i^n w_i^n + z)/(w_i^n+1). Notes Forgy(1965)/Jennrich independently proposed
     the simple *batch* low-within-class-variance method (= Lloyd/Forgy batch form). Name reason: "at each stage
     the k-means are, in fact, the means of the groups they represent."
   - Conditional-mean-minimizes-squared-error is also in MacQueen's information-structure framing (B should pick
     the conditional mean of p over S_i).
3. THIRD-PARTY EXPLAINERS — sklearn user guide `modules/clustering.html` (inertia objective Σ_i min_{μ_j}‖x_i−μ_j‖²;
   inertia assumes convex+isotropic clusters, fails on elongated/non-convex, curse of dimensionality concentrates
   distances, PCA pre-step; non-convex → local minima → n_init restarts; k-means++ spreads centers); Wikipedia
   k-means++ (D² steps, clumping failure of uniform init); search corroboration of D² intuition.
CODE: canonical impl = sklearn `_kmeans.py` (`code/sklearn_kmeans.py`): `_kmeans_plusplus` (greedy
n_local_trials = 2 + int(log k)) and `_kmeans_single_lloyd` (batch loop, tol convergence on labels/center shift,
final E-step). This grounds the final code.

## Core objective & central difficulty
Objective φ(C) = Σ_{x∈X} min_{c∈C} ‖x−c‖² (inertia / WCSS). Choosing C to minimize φ is NP-hard even for k=2
(Aloise/Dasgupta; cited [10] in primary). So no exact poly algorithm; must locally search. The hardness is what
forces (a) a local-search heuristic and (b) the seeding question.

## Derivation spine (problem-first)
1. Want to partition unlabeled points into k groups by squared-distance compactness; define φ. NP-hard exactly.
2. Two-variable objective φ(C, assignment). Coordinate descent / alternating minimization:
   - Fix centers → optimal assignment is nearest-center (each term min over c is achieved pointwise). = Lloyd (15),
     MacQueen S_i.
   - Fix assignment → optimal center per cluster = centroid (mean). PROOF via the bias-variance identity:
     Σ_{x∈S}‖x−z‖² = Σ‖x−c(S)‖² + |S|‖c(S)−z‖², minimized at z=c(S). = Lloyd (14)/Lemma 2.1. This is "moment of
     inertia minimized at center of mass."
   - Each step weakly decreases φ; φ≥0; finitely many partitions (≤ k^n) → terminates at a fixed point. = monotone
     sequence (20).
3. Online vs batch: MacQueen's running-mean (process one point at a time, nudge its nearest mean) vs Forgy/Lloyd
   batch (recompute all means each sweep). Batch is the alternating-minimization form; what sklearn ships.
4. Empty-cluster degeneracy: a center can win no points → undefined mean; handle (reseed farthest point). Real
   implementations handle it; mention as a wall.
5. Local-minimum problem: φ non-convex in centers; arbitrary/uniform init → can get arbitrarily bad clusterings
   (φ/φ_OPT unbounded even for fixed n,k) and uniform init clumps centers inside one true cluster, missing far
   clusters. WALL: Lloyd's quality is hostage to its seeding.
6. SEEDING — the actual contribution. Want initial centers (a) spread out, (b) still data-adaptive/robust to
   outliers, (c) with a provable guarantee. Reject: uniform (clumps), farthest-point/maximin (an outlier is the
   farthest point → seeds outliers). Compromise: SAMPLE with probability ∝ D(x)² (D = dist to nearest chosen
   center). Squared → strongly biased to far/under-covered regions (spread), but random → an outlier is only ONE
   point so its total mass is small (robust). D² (not D¹) because the objective is squared distance, so the
   sampling matches the cost each point contributes (φ(A)=Σ D²).
7. PROOFS (lived out in reasoning):
   - Lemma 2.1 (bias-variance / parallel-axis): Σ_{x∈S}‖x−z‖²−Σ‖x−c(S)‖²=|S|‖c(S)−z‖². Derive by expanding.
   - Lemma 3.1: first center uniform from an OPT cluster A ⇒ E[φ(A)]=2φ_OPT(A). (Pay factor 2 for not knowing
     the true centroid; uniform point is on average √2× the centroid's radius in squared terms.)
   - Lemma 3.2: a D²-sampled center FROM A ⇒ E[φ(A)] ≤ 8 φ_OPT(A). Uses triangle ineq D(a_0)≤D(a)+‖a−a_0‖ and
     power-mean (a+b)²≤2a²+2b² ⇒ D(a_0)²≤2D(a)²+2‖a−a_0‖²; sum, substitute, Lemma 3.1 ⇒ factor 8.
   - Lemma 3.3 (induction over t centers, u uncovered OPT clusters): E[φ'] ≤ (φ(X_c)+8φ_OPT(X_u))(1+H_{t}) +
     (u−t)/u · φ(X_u), H_t = 1+1/2+...+1/t.
   - Theorem 3.1: specialize t=u=k−1 + Lemma 3.1 ⇒ E[φ] ≤ 8(ln k + 2) φ_OPT. (H_{k−1} ≤ 1+ln k.)
   - Tightness (§4): Ω(log k) lower bound — D² seeding no better than 2 ln k on a simplex-of-simplices instance.
   - Generalization (§5): D^ℓ for φ^[ℓ]=Σ min‖x−c‖^ℓ; ℓ=1 is k-median; E[φ^[ℓ]] ≤ 2^{2ℓ}(ln k+2)φ_OPT^[ℓ].
8. Practical greedy: try n_local_trials = 2+⌊log k⌋ candidate points per new center, keep the one that most
   reduces φ — variance reduction on the seeding (sklearn default). Then run Lloyd from the seeds; n_init=10
   restarts (each its own k-means++ seed), keep lowest inertia. max_iter=300 cap.

## Design-decision → why table
- min_{c}‖x−c‖² objective (squared, not abs): squared error ⇒ optimal representative is the MEAN (closed form,
  Lemma 2.1); L1 ⇒ median (k-medoids/medians, no closed-form mean). Mean is cheap and differentiable-in-spirit.
- assignment = nearest center: exact minimizer of φ over assignments with centers fixed (pointwise min).
- center = mean of cluster: exact minimizer over centers with assignment fixed (Lemma 2.1). Together = alt. min.
- monotone + finite partitions ⇒ convergence to fixed point (local min / saddle). No step size, no tuning.
- k-means++ over uniform: uniform clumps, gives unbounded φ/φ_OPT; D² spreads with a provable O(log k).
- D² over D¹ / over farthest-point: D² matches the squared cost (Lemma 3.2 needs the square), beats D¹'s weaker
  spread; randomized beats deterministic farthest-point which seeds outliers.
- greedy n_local_trials=2+log k: cheap variance reduction of the seeding; sklearn's shipped default.
- n_init=10 restarts, keep min inertia: φ non-convex/local minima ⇒ best of several seeds.
- max_iter=300: practical cap; Lloyd usually converges in far fewer (label-equality / center-shift<tol).
- empty clusters: reseed the point farthest from its center (sklearn relocates); keeps k clusters alive.

## In-frame discipline
context.md: never name "k-means++"/"Lloyd" as a paper; frame baselines (uniform init, farthest-point, online
MacQueen update, the alternating loop) as prior art with observed limitations; scaffold = a generic
"initialize centers → refine" clustering harness with one empty seeding slot + neutral TODO. reasoning.md:
first-person present, derive φ and both update steps and all k-means++ lemmas, hit the uniform-init wall, land on
the D² seeding + greedy + Lloyd + restarts code. answer.md: name k-means/k-means++ as the thing built, faithful
sklearn-style code, no citation header.
