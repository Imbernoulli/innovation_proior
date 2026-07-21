DBSCAN's numbers say exactly what I feared and exactly what to do next. On moons it was excellent —
ARI 0.972, NMI 0.944 on the one seed that ran the non-convex settings — vindicating the whole
density-backbone story: two interleaving arcs are precisely the follow-the-dense-path case, and DBSCAN
peeled the noise halo and kept the arcs apart. On blobs it was respectable but plainly capped: ARI
ran 0.59–0.76 across seeds (mean 0.70), and the spread itself is the tell — the seed-456 draw fell to
0.593 while seed-42 held 0.763, a 0.17 swing, and a method whose quality wobbles that much with the
random draw is a method fighting its own parameter. A single global `eps = 0.22` fits the tight blobs
and starves the loose ones, so the varying `cluster_std` cost me, seed-to-seed, exactly the points a
method that *modeled* each blob would keep. But digits is the catastrophe, and it is unambiguous: ARI
0.000302, NMI 0.0107, silhouette **−1.0**. That −1.0 is not "a bit worse" — it is the harness's
signature for a *degenerate* labeling, fewer than two real clusters. In 64 dimensions the `k`-distance
curve flattened (distances concentrate, every pair creeps toward the same value), the knee I read off
it was meaningless, the derived `eps` came out too small, and DBSCAN swept essentially the whole set
into noise plus one residual blob. Global density thresholding did not survive the curse of
dimensionality.

Before I decide what to build, I should read what that collapse costs on the *actual* scoring rule,
because the aggregate is a geometric mean and I want to feel exactly how much it punishes one dead
setting. Each setting scores the mean of its three metrics, silhouette floored at −1. So blobs comes to
`(0.697 + 0.814 + 0.610)/3 ≈ 0.707`, moons to `(0.972 + 0.944 + 0.224)/3 ≈ 0.714`, and digits to
`(0.0003 + 0.011 + (−1.0))/3 ≈ −0.330`. Two healthy settings near 0.71 and one at −0.33. A geometric
mean multiplies these, and a *negative* factor makes the product ill-defined — degenerate — which is
the sharpest possible statement that the task refuses to let one collapsed geometry be averaged away.
Even if I imagine the digits setting merely *bad* rather than degenerate, say lifted to a positive 0.1,
the aggregate would be `(0.707 · 0.714 · 0.1)^{1/3} = (0.0505)^{1/3} ≈ 0.37` — a single weak setting
drags the geometric mean far below the two strong ones, and a genuinely dead one annihilates it. This
is the whole lesson: on this scoring rule, being excellent on two geometries and dead on the third is
strictly worse than being mediocre on all three. The moons silhouette of 0.224 is a smaller version of
the same warning read from the other side — the ARI there is 0.972, a near-perfect partition, yet the
intrinsic silhouette is low, because two interleaving arcs are genuinely close in Euclidean space and no
*correct* labeling of them looks compact-and-separated to silhouette. So silhouette and ARI can pull
apart, and I should not chase silhouette on non-convex geometry; the metric to protect on digits is the
"don't be degenerate" floor, nothing subtler.

So the diagnosis is sharp: DBSCAN owns the non-convex setting but *throws away* a whole geometry, and on
the geometric-mean aggregate a single collapsed setting is ruinous. I need a method that is at least
*non-degenerate everywhere* — that returns `k` honest clusters on the high-dimensional digits where
DBSCAN returned nothing — even if I have to give up some of DBSCAN's shape flexibility to get it. The
harness hands me the true cluster count as `n_clusters`; DBSCAN ignored it, and on digits that refusal
is exactly what killed it. The obvious move is to *use* it — but "use `k`" names a whole family of
methods, and I should walk that family and let the choice be forced rather than reached for.

Four candidates on the shelf take `k` and could plausibly beat DBSCAN on digits, and three fall to
arithmetic. A Gaussian mixture tempts most — full per-component covariance could *model* the varying
`cluster_std` blobs DBSCAN capped, a tight ellipse on the tight blob and a loose one on the loose — but
on digits ten components each with a full `64×64` covariance is `10 · (64 · 65 / 2) = 20800` covariance
parameters from 1797 points, more than ten unknowns per point, so the covariances go singular, the
likelihood spikes to `+∞`, and EM collapses; the diagonal restriction (`10 · 64 = 640` parameters) is
estimable but is just K-Means with per-axis variance scaling, buying nothing on moons while adding a
fragile EM. Spectral clustering is sharper, since its Laplacian eigen-embedding follows non-convex
connectivity and could get moons *right* — but its Gaussian-kernel bandwidth `σ` is a single global scale
knob that will concentrate and misfire in 64-D for exactly the reason `eps` did, relocating the digits
failure into a new parameter, and its dense `O(n³)` eigendecomposition is about `5.8·10⁹` operations at
`n = 1797` with no explicit noise. K-medoids/PAM keeps a nearest-representative rule, so its cells stay
convex — nothing for moons — at a steeper `O(n²)` per iteration. The other direction, making DBSCAN's
radius adaptive per region, is real but a much larger build; what I need first is the simplest thing that
structurally *cannot* collapse on digits, so I set that aside as a later method's job and take the
minimal robust partitioner.

That leaves plain K-Means, and it is minimal for a reason worth spelling out: it is the one candidate
that is *constructed* to return exactly `k` non-empty groups, so it cannot produce the degenerate
single-cluster labeling that scored −1.0, and it carries no global scale knob to mis-set in 64-D. So let
me build it from what I want and write down a single number that scores a grouping, because "tight and
separated" is an intuition, not a procedure. Summarize each group by one representative point `c`, its
center; "tight" becomes "every point is close to its representative." Score a candidate set of `k`
centers `C = {c_1,…,c_k}` by the total squared distance from each point to its nearest center,
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
group `S` minimizes `Σ_{x∈S} ||x − z||²`. Work out the minimizer. Let `c = (1/|S|)Σ_{x∈S} x`
and write `x − z = (x − c) + (c − z)`; expanding, `Σ ||x − z||² = Σ ||x − c||² + 2(c − z)·Σ(x − c) +
|S| ||c − z||²`. The cross term carries `Σ(x − c) = |S|c − |S|c = 0` by the very definition of the
mean, so it vanishes, leaving `Σ ||x − z||² = Σ ||x − c||² + |S| ||c − z||²`. This is the parallel-axis
identity: the total squared distance to any point `z` is the squared spread about the centroid plus
`|S|` times `z`'s squared offset from it. Any `z ≠ c` adds the strictly positive `|S| ||c − z||²`, so the
minimizer is uniquely `z = c`, the mean. Now squared distance pays off: under squared error the optimal
representative is the *mean*, a closed-form one-pass quantity; under absolute error it would be the
median, which has no such linear form. The square is what makes the centroid step trivial. Two forced
steps, alternating.

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

It is worth being precise about *why* the same 64-D distance concentration that annihilated DBSCAN does
not annihilate this loop, because the distinction is the whole reason I switched. DBSCAN's core test is a
comparison of a distance against an *absolute* threshold `eps`: when every pairwise distance is squeezed
into a thin shell of relative width `~1/√d`, the threshold either admits almost everything or almost
nothing, and there is no setting in between — the estimate is a knife-edge, and it fell off the edge into
"everything is noise." K-Means never compares a distance to an absolute number. Its assignment step is an
`argmin` over the ten centers — a purely *relative* comparison — and even when all ten distances are
within a percent of each other, one of them is still smallest, so every point still gets assigned and
every center still gets a non-empty group. The centroid step is a per-coordinate average, which has no
scale threshold at all. So concentration cannot produce a degenerate labeling here; what it *can* do is
shrink the margins, so the ten digit groups will be only weakly separated and the silhouette on digits
will be modest and positive rather than large — a real cost, but a survivable one, and exactly the
opposite failure mode from DBSCAN's cliff.

But here is the rub, and it is the whole story. The fixed point is only a *local* minimum of `phi`.
`phi`, as a function of the center locations, is a pointwise minimum of convex squared-distance pieces —
a bumpy, non-convex landscape with many valleys of different depths — and "never increases" only
guarantees I roll into whatever basin I started in. Which valley is decided entirely by the
*initialization*. With the naive start — `k` centers drawn uniformly from the data — picture five
well-separated blobs and `k = 5`: uniform sampling very likely lands two or three centers in one dense
blob and none in another, the empty blob gets swallowed and the over-seeded one gets split, and the loop
converges happily to a clustering that merges two true groups and splits a third. And this is not a rare
tail event: the chance that five uniform draws land one-per-blob (ignoring blob-size differences) is
`5!/5⁵ = 120/3125 ≈ 0.038`, so about 96% of naive seedings start with some blob double-seeded and some
blob empty — the bad basin is the *typical* case, not the exception. Once a blob is empty at
initialization, the loop has no mechanism to re-populate it (no point ever prefers a far center it was
not seeded near), so that true cluster is merged into a neighbor for good while the double-seeded blob is
split down the middle, and there is no upper bound on how bad this is — uniform seeding gives an
unbounded `phi/phi_OPT` ratio with non-negligible probability. So the local search is fine; the *seeding*
is the disease, and I cannot afford a bad basin
on this task, where blobs has five clusters of *varying* density — the loose blob is exactly the one a
clumped seeding will starve, echoing the very spread I watched cost DBSCAN its 0.17 seed swing.

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
*any* data, with no separation assumption — and the subsequent local search only lowers `phi`. Put the
task's numbers into that bound to see its size: for digits `k = 10` it is `8(ln 10 + 2) = 8(2.30 + 2) =
8 · 4.30 ≈ 34`, and for blobs `k = 5` it is `8(ln 5 + 2) = 8(1.61 + 2) ≈ 29` — worst-case constants, not
what I expect in practice, but the point is they are *finite and independent of the data*, whereas
uniform seeding has no finite bound at all. An `O(log k)` guarantee for the whole procedure, exactly
where uniform seeding had none.

Now the edit, and here the scaffold matters. I have `scikit-learn` in scope, and `sklearn.cluster.KMeans`
*is* this method: Lloyd's two forced steps, with `init="k-means++"` as its default seeding (the `D²`
construction above) and `n_init` independent restarts keeping the lowest-inertia run — the standard hedge
against the non-convex landscape, since a single seeding can still be unlucky, and `n_init` restarts
shrink the tail probability of a bad basin geometrically. So I do not hand-roll the `D²` sampler or the
empty-cluster relocation here; I instantiate `KMeans(n_clusters=k, random_state=seed, n_init=10,
max_iter=300)`, where `k = n_clusters` from the harness (8 only as a fallback that will not trigger,
since the harness always passes the true count). `n_init = 10` is the restart count that makes the
`O(log k)` seeding robust in practice; `max_iter = 300` caps the local loop, which converges far sooner
on these sizes. `predict` delegates to the fitted model's nearest-center rule. The class is in the answer.

Reading DBSCAN's numbers, here is what I expect and where I must lose. On **digits** —
the setting DBSCAN threw away at silhouette −1.0 and ARI 0.0003 — this should be a large, decisive gain:
ten honest centroids in 64-D, no collapse, ARI and NMI well off the floor and silhouette comfortably
positive; the geometric-mean aggregate alone should jump from rescuing this one dead setting, since I
showed a moment ago that lifting the digits setting from a negative factor to any positive one is worth
more to the product than a couple of points anywhere else. On **blobs**, K-Means should *beat* DBSCAN
clearly — convex isotropic Gaussians are its exact model, and `D²` seeding plus restarts handle the
varying `cluster_std` better than a single global `eps`, so I expect ARI up toward the mid-0.8s, above
DBSCAN's 0.70 and with less seed spread. The honest cost is **moons**, where I knowingly give back
DBSCAN's win. Nearest-centroid is a Voronoi tessellation: with `k = 2` the two centroids are separated by
a single straight bisector, and no orientation of one straight line puts a whole crescent on each side,
because the interleaving arcs wrap past each other — each side holds a chunk of *both* true arcs, so each
predicted cluster is a half-and-half mixture, the low-ARI regime. Worse, the inertia objective is
actively hostile: the lowest-inertia 2-split is the compact left/right cut, precisely the wrong one, so
restarts converge to the same wrong-but-compact optimum rather than rescuing me. Moons ARI should fall
well below DBSCAN's 0.972, likely toward a coin-flip. That is the trade with eyes open — a geometric mean
rewards mediocre-everywhere over excellent-on-one-and-zero-on-another, which the degenerate
`(0.707 · 0.714 · −0.330)` product already proved. And if moons does fall to a poor convex split, that
failure is the next method's target, since the ideal method keeps DBSCAN's non-convex moons *and*
K-Means's non-degenerate digits at once.