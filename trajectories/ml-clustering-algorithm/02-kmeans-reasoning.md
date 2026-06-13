DBSCAN's numbers say exactly what I feared and exactly what to do next. On moons it was excellent —
ARI 0.972, NMI 0.944 on the one seed that ran the non-convex settings — vindicating the whole
density-backbone story: two interleaving arcs are precisely the follow-the-dense-path case, and DBSCAN
peeled the noise halo and kept the arcs apart. On blobs it was respectable but plainly capped: ARI
ran 0.59–0.76 across seeds (mean 0.70), and the spread itself is the tell — a single global `eps = 0.22`
fits the tight blobs and starves the loose ones, so the varying `cluster_std` cost me, seed-to-seed,
exactly the points a method that *modeled* each blob would keep. But digits is the catastrophe, and it
is unambiguous: ARI 0.000302, NMI 0.0107, silhouette **−1.0**. That −1.0 is not "a bit worse" — it is
the harness's signature for a *degenerate* labeling, fewer than two real clusters. In 64 dimensions the
`k`-distance curve flattened (distances concentrate, every pair creeps toward the same value), the knee
I read off it was meaningless, the derived `eps` came out too small, and DBSCAN swept essentially the
whole set into noise plus one residual blob. Global density thresholding did not survive the curse of
dimensionality. So the diagnosis is sharp: DBSCAN owns the non-convex setting but *throws away* a whole
geometry, and on the task's geometric-mean aggregate a single collapsed setting is ruinous. I need a
rung that is at least *non-degenerate everywhere* — that returns `k` honest clusters on the
high-dimensional digits where DBSCAN returned nothing — even if I have to give up some of DBSCAN's
shape flexibility to get it. The harness hands me the true cluster count as `n_clusters`; DBSCAN
ignored it, and on digits that refusal is exactly what killed it. The obvious move is to *use* it.

So let me build the method that uses `k` and cannot collapse: I will partition into exactly `k` groups
by a compactness objective, accepting convex cells as the price of robustness in high dimensions. Start
from what I want and write down a single number that scores a grouping, because "tight and separated"
is an intuition, not a procedure. Summarize each group by one representative point `c`, its center;
"tight" becomes "every point is close to its representative." Score a candidate set of `k` centers
`C = {c_1,…,c_k}` by the total squared distance from each point to its nearest center,
`phi(C) = Σ_x min_c ||x − c||²`. Choosing the centers already chooses the clustering — the `min` assigns
each point to its closest center — so the whole problem collapses to: pick `k` centers minimizing `phi`.
Why squared distance? I will get a real reason in a moment; provisionally it is the quantity I can do
algebra with. And how hard is exact minimization? Even `k = 2` in the plane is NP-hard — moving the
boundary moves both centroids, which moves the boundary again. So an exact minimizer is off the table;
I need a good local search, and the entire game becomes how good a local solution I can reach.

`phi` tangles two kinds of variable: the center locations and the assignment of points to centers. The
move with tangled variables is to optimize one with the other fixed and alternate. Fix the centers:
each point contributes `min_c ||x − c||²`, minimized by assigning it to its nearest center — exactly
optimal, term by term, and it carves space into Voronoi cells. Fix the assignment: the best center for a
group `S` minimizes `Σ_{x∈S} ||x − z||²`. Compute it rather than assert it. Let `c = (1/|S|)Σ_{x∈S} x`
and write `x − z = (x − c) + (c − z)`; expanding, `Σ ||x − z||² = Σ ||x − c||² + 2(c − z)·Σ(x − c) +
|S| ||c − z||²`. The cross term carries `Σ(x − c) = |S|c − |S|c = 0` by the very definition of the
mean, so it vanishes, leaving `Σ ||x − z||² = Σ ||x − c||² + |S| ||c − z||²`. This is the parallel-axis
identity: the total squared distance to any point `z` is the squared spread about the centroid plus
`|S|` times `z`'s squared offset from it. The first term is independent of `z`; the second is
nonnegative and zero only at `z = c`. So the unique minimizer is the arithmetic mean — and now squared
distance pays off: under squared error the optimal representative is the *mean*, a closed-form one-pass
quantity; under absolute error it would be the median, which has no such linear form. The square is what
makes the centroid step trivial. Two forced steps, alternating.

Does the loop converge? The assignment step moves each point to its cost-minimizing center, so `phi`
cannot increase; the centroid step moves each center to its group's squared-error minimizer, so again
`phi` cannot increase. `phi` is monotonically non-increasing, bounded below by zero, and the loop marches
through *partitions*, of which there are finitely many; a monotone sequence over a finite set must
terminate at a fixed point — every point with its nearest center, every center the mean of its group. No
step size, no learning rate; the geometry of the two minimizations carries it. This is the contrast I
wanted with DBSCAN's `eps`: there is no global radius to mis-set, no knob whose mis-calibration
collapses the output. On digits, where DBSCAN's single `eps` produced nothing, this loop *cannot*
collapse to one cluster — it is constructed to return exactly `k = 10` non-empty groups, which alone
should lift digits off the −1.0 floor.

But here is the rub, and it is the whole story. The fixed point is only a *local* minimum of `phi`.
`phi`, as a function of the center locations, is a pointwise minimum of convex squared-distance pieces —
a bumpy, non-convex landscape with many valleys of different depths — and "never increases" only
guarantees I roll into whatever basin I started in. Which valley is decided entirely by the
*initialization*. With the naive start — `k` centers drawn uniformly from the data — picture five
well-separated blobs and `k = 5`: uniform sampling very likely lands two or three centers in one dense
blob and none in another, the empty blob gets swallowed and the over-seeded one gets split, and the loop
converges happily to a clustering that merges two true groups and splits a third. There is no upper
bound on how bad this is — uniform seeding gives an unbounded `phi/phi_OPT` ratio with non-negligible
probability. So the local search is fine; the *seeding* is the disease, and I cannot afford a bad basin
on this task, where blobs has five clusters of *varying* density — the loose blob is exactly the one a
clumped seeding will starve, echoing the very spread I watched cost DBSCAN.

What do I want from seeding? The centers spread one-per-group, robustness against a single weird point
hijacking a center, and ideally a provable bound — the thing the local search lacks. Force the spread
deterministically — first center somehow, then repeatedly add the point farthest from all chosen
centers — and it does spread, but "farthest point" *is* the most extreme outlier, so it preferentially
plants centers on outliers. The spread is bought by sacrificing robustness. So keep the *tendency* of
farthest-point without the brittleness: randomize it. Let `D(x)` be the distance from `x` to its
nearest chosen center and sample the next center with probability proportional to an increasing function
of `D(x)` — far-from-everything regions are favored (spread), but any single point, even an outlier, is
one point carrying a small slice of the probability mass (robust); a dense under-covered cluster has
many moderately-far points whose mass sums. Which function of `D(x)`? Let the objective decide:
`phi = Σ_x D(x)²`, so each point's contribution to the very quantity I am shrinking is `D(x)²`. Sample
proportional to `D(x)²` — `D²` weighting — and I place each new center where the current cost is
concentrated. The exponent is not a knob; it matches the squared-error objective. This is k-means++
seeding, and it earns a real guarantee the bare loop cannot: the parallel-axis identity gives that a
uniformly-seeded optimal cluster costs `2·phi_OPT(A)` in expectation, a `D²`-seed landing in `A` costs
`≤ 8·phi_OPT(A)` (via triangle plus the power-mean inequality), and an induction over (centers left to
place, clusters still uncovered) accumulates a harmonic factor to `E[phi] ≤ 8(ln k + 2)·phi_OPT` for
*any* data, with no separation assumption — and the subsequent local search only lowers `phi`. An
`O(log k)` guarantee for the whole procedure, exactly where uniform seeding had none.

Now the edit, and here the scaffold matters. I have `scikit-learn` in scope, and `sklearn.cluster.KMeans`
*is* this method: Lloyd's two forced steps, with `init="k-means++"` as its default seeding (the `D²`
construction above) and `n_init` independent restarts keeping the lowest-inertia run — the standard hedge
against the non-convex landscape, since a single seeding can still be unlucky. So I do not hand-roll the
`D²` sampler or the empty-cluster relocation here; I instantiate `KMeans(n_clusters=k,
random_state=seed, n_init=10, max_iter=300)`, where `k = n_clusters` from the harness (8 only as a
fallback that will not trigger, since the harness always passes the true count). `n_init = 10` is the
restart count that makes the `O(log k)` seeding robust in practice; `max_iter = 300` caps the local
loop, which converges far sooner on these sizes. `predict` delegates to the fitted model's nearest-center
rule. The class is in the answer.

Reading DBSCAN's numbers, here is what I expect this rung to do and where it must lose. On **digits** —
the setting DBSCAN threw away at silhouette −1.0 and ARI 0.0003 — this should be a large, decisive gain:
ten honest centroids in 64-D, no collapse, ARI and NMI well off the floor and silhouette comfortably
positive; the geometric-mean aggregate alone should jump from rescuing this one dead setting. On
**blobs**, K-Means should *beat* DBSCAN clearly — convex isotropic Gaussians are its exact model, and
`D²` seeding plus restarts handle the varying `cluster_std` far better than a single global `eps` did,
so I expect ARI up around the 0.85 range, above DBSCAN's 0.70 and with less seed-to-seed spread. The
honest cost is **moons**: this is where I knowingly give back DBSCAN's win. Two interleaving half-circles
are non-convex, and "nearest centroid" is a Voronoi tessellation that *must* cut each moon with a
straight bisector — so K-Means will slice the moons the wrong way and score far below DBSCAN's 0.972, I
expect ARI well under 0.5. That is the trade I am making with eyes open: I sacrifice the one setting
DBSCAN owned to stop *failing* the setting DBSCAN destroyed, and because the task aggregates by
geometric mean, trading a moons collapse for a digits collapse is the right direction — a method that is
mediocre-everywhere beats one that is excellent-on-one and zero-on-another. The falsifiable claims
against the prior numbers: digits silhouette goes from −1.0 to positive and digits ARI from ~0 to a real
value; blobs ARI rises above 0.70; and moons ARI *falls* below DBSCAN's 0.97 — and if it falls all the
way to a poor convex split, that moons failure is itself the next rung's target, because the ideal method
would keep DBSCAN's non-convex moons *and* K-Means's non-degenerate digits at once.
