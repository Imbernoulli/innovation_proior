Let me start from what I actually have and what I actually want. I have `n` points sitting in `R^d`, no labels, and a number `k`. I want to carve them into `k` groups so that within a group the points are tight and across groups they're far apart. Before I can search for anything I need a single number that says how good a grouping is, because "tight" and "far apart" are intuitions, not a procedure. The cleanest way to summarize a group is by a single representative point `c` — its center — and then "tight" becomes "every point is close to its representative." So let me score a candidate set of `k` centers `C = {c_1, ..., c_k}` by the total squared distance from each point to its nearest center,

  phi(C) = sum_{x in X} min_{c in C} || x - c ||^2.

Why squared distance rather than plain distance? I'll come back to that with a real reason in a moment, but provisionally it's the quantity I can do algebra with. And notice that choosing the centers already chooses the clustering: once I fix `C`, each point's natural group is the center it's closest to — the `min` in `phi` literally picks that center. So the whole problem collapses to: pick `k` points `C` that make `phi` small. That's the object. Now, how hard is it to minimize `phi` exactly? If I think about even `k = 2` in the plane, I'd have to consider every way of splitting `n` points into two groups and check the squared-error of each — and the optimal split isn't given by any simple sweep because moving the boundary changes both centroids, which moves the boundary again. This is in fact NP-hard, even for `k = 2`. So an exact minimizer is off the table for any real `n`. I'm going to have to settle for a heuristic that finds a good local solution, and the entire game becomes: how good a local solution, and can I say anything provable about it.

Let me get the local search itself nailed down first, because I need to understand exactly where its weakness lives before I can fix it. The objective `phi` has two kinds of variable tangled together: the locations of the centers, and the assignment of points to centers. When two things are tangled like that, the move is to optimize one with the other held fixed, alternate, and hope. So: fix the centers, what's the best assignment? Each point `x` contributes `min_{c} || x - c ||^2` — a minimum over centers — and that minimum is achieved by assigning `x` to its nearest center. There's no freedom and no cleverness here: for fixed centers, nearest-center assignment is exactly optimal, term by term. So step one is forced — assign each point to its nearest center, which carves space into the regions "closer to `c_i` than to any other center," and those regions are Voronoi cells whose boundaries bisect the segments between adjacent centers.

Now the other direction: fix the assignment, what's the best center for a group `S`? I want the point `z` that minimizes `sum_{x in S} || x - z ||^2`. Let me actually compute it instead of asserting "the mean." Let `c = c(S) = (1/|S|) sum_{x in S} x` be the center of mass, and write each `x - z = (x - c) + (c - z)`. Then

  sum_{x in S} || x - z ||^2 = sum_{x in S} || (x - c) + (c - z) ||^2
    = sum_{x in S} || x - c ||^2 + 2 (c - z) . sum_{x in S} (x - c) + |S| || c - z ||^2.

The cross term carries `sum_{x in S} (x - c)`, and that's `(sum x) - |S| c = |S| c - |S| c = 0` by the very definition of the mean. The cross term vanishes. So

  sum_{x in S} || x - z ||^2 = sum_{x in S} || x - c ||^2 + |S| || c - z ||^2.

This is a clean and load-bearing identity, so let me state what it says: the total squared distance from a group to *any* point `z` equals the total squared distance to its own center of mass, plus `|S|` times the squared offset of `z` from that center. The first term doesn't depend on `z` at all; the second is a nonnegative `|S| || c - z ||^2` that is zero exactly when `z = c`. So the minimizer is `z = c`, the arithmetic mean — uniquely. This is the parallel-axis theorem: the moment of inertia of a body about a point is minimized about its center of mass. And now I can answer why squared distance and not absolute distance: under squared error the optimal representative is the *mean*, a closed-form linear thing I can recompute in one pass; under absolute error it would be the median, which has no such linear form and is annoying to recompute. The square is what makes the centroid step trivial. So step two is also forced — move each center to the mean of its currently assigned points.

Two forced steps, alternating. Does the loop converge? Track `phi` across a full sweep. The assignment step moves each point to the center that minimizes its own term, so `phi` cannot increase — it can only drop or stay. The centroid step, by the identity I just proved, moves each center to the unique squared-error minimizer of its group, so again `phi` cannot increase. So `phi` is monotonically non-increasing across the whole loop. It's bounded below by zero. And the loop is really a march through *assignments* — there are at most `k^n` distinct partitions of the points — and once an assignment repeats, `phi` would have to have stayed flat, meaning I've hit a fixed point. A monotone sequence over a finite set must terminate. So the loop converges, in finitely many sweeps, to a fixed point: an assignment where every point is already with its nearest center and every center is already the mean of its group. No step size, no learning rate, nothing to tune — the geometry of the two minimizations carries it.

But here is the rub, and I have to be precise about it because it's the whole story. The fixed point is only a *local* minimum of `phi` — or worse, possibly a saddle. The loop never increases `phi`, but "never increases" only guarantees I roll downhill into whatever basin I started in. And `phi`, as a function of the center locations, is not convex: it's a pointwise minimum over centers of convex squared-distance pieces, so it's a bumpy landscape with many valleys of very different depths. Which valley I end up in is decided entirely by where I place the centers at the start. So the quality of this beautiful, parameter-free local search is hostage to its initialization, and I have no control over the outcome.

How bad can that be? Let me think about the naive start everyone uses: pick the `k` initial centers uniformly at random from the data. Picture five well-separated blobs and `k = 5`. Uniform sampling draws five points from the union; with five blobs the chance that I get exactly one center per blob is low — much more likely two or three of my centers land in the same dense blob and at least one blob gets no center at all. Now run the loop: the blob with no nearby center gets swallowed into its neighbor, while the over-seeded blob gets split in two. The loop happily converges — every center is the mean of its region — but it's converged to a clustering that merges two true groups and splits a third. And there's no upper bound on how bad this is: I can build instances where the `phi` the loop returns is an arbitrarily large multiple of `phi_OPT`, with `n` and `k` fixed, and this isn't an adversarial trick — even ordinary random seeding produces unboundedly bad ratios with non-negligible probability. So the local search is fine; the *seeding* is the disease. The whole question reduces to: how do I place the `k` initial centers so that the local search starts in a good basin, ideally with a guarantee?

So what do I want from the seeding? Three things. The centers should be spread out — one per true group, not clumped. It should be robust — a single weird point shouldn't be able to hijack the placement. And, the prize, it should come with a provable bound on the resulting `phi`, because that's exactly what the local search lacks.

First attempt: force the spread deterministically. Pick the first center somehow, then add the point farthest from all centers chosen so far, repeat. This certainly spreads the centers — each new one is as far as possible from the existing ones. But stare at what "farthest point" *is*: it's the most extreme outlier in the data. So this procedure preferentially plants centers on outliers — one anomalous point pulls a whole center onto itself, wasting it and distorting the rest. The spread is real but it's bought by sacrificing robustness completely. Wall. Deterministic-farthest is too aggressive — it always grabs the single most extreme point.

So I want the *tendency* of farthest-point — favor points far from the current centers — without the brittleness of always grabbing the single farthest. The fix for "always grab the max" is "sample with probability that grows with the distance." Randomize it. Let `D(x)` be the distance from `x` to the nearest center already chosen. If I sample the next center from the data with probability proportional to some increasing function of `D(x)`, then far-from-everything regions are strongly favored — I get the spread — but any single point, even an outlier, is just one point carrying a small slice of the total probability mass, so it can't dominate. Outliers are far, yes, but there are few of them; a dense under-covered cluster has many moderately-far points whose mass adds up. Randomization turns "the one farthest point" into "the under-covered regions," which is exactly what I want.

Now, what function of `D(x)`? Proportional to `D(x)` itself? Or `D(x)^2`? Let me let the objective decide rather than guess. The objective is `phi = sum_x min_c || x - c ||^2 = sum_x D(x)^2` once `D(x)` is the distance to the nearest chosen center. So each point's *contribution to the very quantity I'm trying to shrink* is `D(x)^2`. If I sample proportional to `D(x)^2`, I'm sampling each point in proportion to how much it currently hurts the objective — I put a new center, with high probability, exactly where the current cost is concentrated. That's the principled choice: sample with probability

  D(x_0)^2 / sum_{x in X} D(x)^2,

which I'll call `D^2` weighting. The square isn't a tuning knob; it's the matching exponent to a squared-error objective. (I'll later check that `D^1` corresponds to a different objective, but for squared error it's the square.) The seeding, then: first center uniformly at random from the data; each subsequent center sampled with `D^2` weighting against the centers chosen so far; stop at `k`; then run the alternating loop from those seeds.

Now I want the prize — a provable bound on the resulting `phi`. This is the part that justifies the whole construction, so let me actually prove it rather than wave at it. I'll compare against the optimal clustering `C_OPT` with potential `phi_OPT`, and I'll write `phi(A)` for the contribution of a point set `A` to the potential, and `phi_OPT(A)` for `A`'s contribution under the optimal centers. The key structural fact I already have — the parallel-axis identity — I'll restate in the form I'll lean on: for any point set `S` with center of mass `c(S)` and any point `z`,

  sum_{x in S} || x - z ||^2 - sum_{x in S} || x - c(S) ||^2 = |S| || c(S) - z ||^2.

Step one of the bound: what happens to a single optimal cluster `A` when I seed a center into it *uniformly at random*? This is the situation for the very first center. Let `A` be a cluster of `C_OPT`. Since `C_OPT` is optimal, the center it uses for `A` is `A`'s own center of mass `c(A)` (by the identity — any other choice has strictly larger contribution). If I pick the seed `a_0` uniformly from `A` and use it as the only center for `A`, the expected contribution is

  E[phi(A)] = (1/|A|) sum_{a_0 in A} sum_{a in A} || a - a_0 ||^2.

Apply the identity to the inner sum with `z = a_0` and `S = A`: `sum_{a in A} || a - a_0 ||^2 = sum_{a in A} || a - c(A) ||^2 + |A| || a_0 - c(A) ||^2`. Substitute:

  E[phi(A)] = (1/|A|) sum_{a_0 in A} ( sum_{a in A} || a - c(A) ||^2 + |A| || a_0 - c(A) ||^2 )
            = (1/|A|) ( |A| * sum_{a in A} || a - c(A) ||^2 + |A| * sum_{a_0 in A} || a_0 - c(A) ||^2 ).

Both inner sums are the same thing — `A`'s squared spread about its own centroid, which is `phi_OPT(A)`. So

  E[phi(A)] = phi_OPT(A) + phi_OPT(A) = 2 phi_OPT(A).

A factor of 2. That's the price of using a random data point instead of the true centroid: in expectation it costs exactly twice the optimal contribution. Intuitively, a random member of `A` sits at the cluster's RMS radius from the centroid, and by the identity that doubles the moment of inertia. Clean.

Step two: the trickier case — the rest of the centers are seeded by `D^2` weighting, not uniformly. I want an analogue of the factor-2 result, but now the new center for cluster `A` is drawn proportional to current squared distance, and there's already an arbitrary set of centers in place. Claim: if I add one center to the current clustering `C` by `D^2` weighting, and that center happens to land in some optimal cluster `A`, then `E[phi(A)] <= 8 phi_OPT(A)`. The factor is 8, not 2 — I pay more because the sampling isn't uniform within `A`, but it's still a constant. Let me prove it. The probability I pick a specific `a_0 in A` as the new center, conditioned on picking from `A`, is `D(a_0)^2 / sum_{a in A} D(a)^2`. After I add `a_0`, a point `a in A` contributes `min(D(a), || a - a_0 ||)^2` to the potential — either its old nearest center is still closer, or the new center `a_0` is. So

  E[phi(A)] = sum_{a_0 in A} ( D(a_0)^2 / sum_{a in A} D(a)^2 ) * sum_{a in A} min(D(a), || a - a_0 ||)^2.

Now I need to control `D(a_0)^2`, because it's weighting the whole thing and it could be large. By the triangle inequality, for any `a, a_0 in A`, `D(a_0) <= D(a) + || a - a_0 ||` — the nearest existing center to `a` is at distance `D(a)`, and `a_0` is within `|| a - a_0 ||` of `a`, so the nearest center to `a_0` is no farther than `D(a) + || a - a_0 ||`. Square it using the power-mean inequality `(p + q)^2 <= 2 p^2 + 2 q^2` (which is just `2p^2 + 2q^2 - (p+q)^2 = (p - q)^2 >= 0`): `D(a_0)^2 <= 2 D(a)^2 + 2 || a - a_0 ||^2`. This holds for *every* `a`, so average it over `a in A`:

  D(a_0)^2 <= (2/|A|) sum_{a in A} D(a)^2 + (2/|A|) sum_{a in A} || a - a_0 ||^2.

Substitute this bound for `D(a_0)^2` in the expectation. The expression splits into two pieces. In the first piece (the one carrying `(2/|A|) sum_a D(a)^2`), the factor `sum_{a in A} D(a)^2` in the numerator cancels the same factor in the denominator from the sampling probability, leaving

  (2/|A|) * sum_{a_0 in A} sum_{a in A} min(D(a), || a - a_0 ||)^2,

and in this piece I use `min(D(a), || a - a_0 ||)^2 <= || a - a_0 ||^2`. In the second piece (the one carrying `(2/|A|) sum_a || a - a_0 ||^2`), I use `min(D(a), || a - a_0 ||)^2 <= D(a)^2`, and the `sum_a D(a)^2` again cancels against the denominator. Both pieces collapse to the same shape:

  E[phi(A)] <= (4/|A|) sum_{a_0 in A} sum_{a in A} || a - a_0 ||^2.

But `(1/|A|) sum_{a_0} sum_{a} || a - a_0 ||^2` is exactly the quantity from step one — it equals `2 phi_OPT(A)` by the same uniform-seeding computation. So `E[phi(A)] <= 4 * 2 phi_OPT(A) = 8 phi_OPT(A)`. There's the 8. The factor of 2 from triangle/power-mean and the factor of 2 from the uniform-within-`A` average multiply.

So I have: a uniformly-seeded cluster costs `2 phi_OPT(A)` in expectation, and a `D^2`-seeded cluster, if the seed lands in it, costs at most `8 phi_OPT(A)`. The remaining worry is that `D^2` seeding might *not* place a center in every optimal cluster — it could seed two centers into one cluster and miss another, the same failure uniform seeding had. I need to show that across the `k` draws, the seeding covers the clusters well enough that the total is only an `O(log k)` factor worse than optimal. This is an induction, and I have to set it up carefully.

Let me define the bookkeeping. At some point in the process I have a clustering `C` (some centers placed). Call an optimal cluster "covered" if it already contains one of my centers, "uncovered" otherwise. Let `X_u` be the union of the points in `u` chosen uncovered clusters, `X_c = X - X_u` the rest. Suppose I'm about to add `t <= u` more centers by `D^2` weighting. The claim I need is:

  E[phi'] <= ( phi(X_c) + 8 phi_OPT(X_u) ) * (1 + H_t) + ((u - t)/u) * phi(X_u),

where `phi'` is the potential after adding the `t` centers and `H_t = 1 + 1/2 + ... + 1/t` is the harmonic sum. Read it before proving it: the covered part `X_c` plus eight times the optimal cost of the uncovered part `X_u`, inflated by `(1 + H_t)`, plus a leftover term for the `(u - t)` clusters I may fail to cover with only `t` centers. The `H_t` is where the `log k` will come from.

If `t = 0` and `u > 0`, I add no centers, so `phi' = phi(X_c) + phi(X_u)`. The right side becomes `phi(X_c) + 8 phi_OPT(X_u) + phi(X_u)`, which is at least `phi(X_c) + phi(X_u)`, so that edge case is safe. If `t = u = 1`, I add one center. It lands in the one uncovered cluster with probability `phi(X_u)/phi`; then the 8-bound gives `E[phi'] <= phi(X_c) + 8 phi_OPT(X_u)`. It lands in the covered part with probability `phi(X_c)/phi`; then adding a center can only decrease the old potential, so `phi' <= phi`. Combining the two cases,

  E[phi'] <= (phi(X_u)/phi) * (phi(X_c) + 8 phi_OPT(X_u)) + (phi(X_c)/phi) * phi
           <= 2 phi(X_c) + 8 phi_OPT(X_u),

and the target bound for `t = u = 1` is `2(phi(X_c) + 8 phi_OPT(X_u))`, so this is stronger than I need.

Now assume the claim for `(t - 1, u)` and `(t - 1, u - 1)`. Write `B = phi(X_c) + 8 phi_OPT(X_u)` and `phi_u = phi(X_u)`. The first of the `t` new centers lands either in the already covered part or in one of the uncovered optimal clusters. If it lands in `X_c`, with probability `phi(X_c)/phi`, the potential only decreases and the number of uncovered clusters stays `u`, so the `(t - 1, u)` hypothesis gives a contribution at most

  (phi(X_c)/phi) * ( B * (1 + H_{t-1}) + ((u - t + 1)/u) * phi_u ).

If it lands in an uncovered optimal cluster `A`, with probability `phi(A)/phi`, I need to be a little more careful. Condition on the chosen point inside `A`; let `p_a` be the `D^2`-weighted conditional probability of choosing `a in A`, and let `phi_a` be `A`'s resulting contribution after I add `a` as a center. Now `A` becomes covered, the uncovered cost becomes `phi_u - phi(A)`, and the covered contribution becomes `phi(X_c) + phi_a`. The `(t - 1, u - 1)` hypothesis gives, for that branch,

  ( phi(X_c) + phi_a + 8 phi_OPT(X_u) - 8 phi_OPT(A) ) * (1 + H_{t-1})
    + ((u - t)/(u - 1)) * (phi_u - phi(A)).

Averaging over `a` inside `A`, the single-cluster `D^2` bound gives `sum_{a in A} p_a phi_a <= 8 phi_OPT(A)`, so the `+ phi_a` and `- 8 phi_OPT(A)` terms cancel in expectation up to an inequality. The contribution from this particular `A` is therefore at most

  (phi(A)/phi) * ( B * (1 + H_{t-1}) + ((u - t)/(u - 1)) * (phi_u - phi(A)) ).

Now sum this over all uncovered clusters `A`. The first part sums to `(phi_u/phi) * B * (1 + H_{t-1})`. The leftover part contains `sum_A phi(A) * (phi_u - phi(A)) = phi_u^2 - sum_A phi(A)^2`. Power-mean gives `sum_A phi(A)^2 >= phi_u^2/u`, so

  phi_u^2 - sum_A phi(A)^2 <= ((u - 1)/u) * phi_u^2.

That turns the uncovered-branch leftover into at most `(phi_u/phi) * ((u - t)/u) * phi_u`. Combining covered and uncovered branches, the common `B * (1 + H_{t-1})` term has total weight one, and the leftover terms become

  ((u - t)/u) * phi_u + (phi(X_c)/phi) * (phi_u/u).

The last extra term is at most `B/u`, because `phi_u/phi <= 1` and `B` contains `phi(X_c)`. So

  E[phi'] <= B * (1 + H_{t-1} + 1/u) + ((u - t)/u) * phi_u.

Since `t <= u`, `1/u <= 1/t`, and therefore `1 + H_{t-1} + 1/u <= 1 + H_t`. That is exactly the claimed bound. Each new draw either spends itself inside already-covered mass or converts one uncovered optimal cluster into a covered cluster with expected cost at most `8 phi_OPT` for that cluster; the only accumulated loss is harmonic.

Now specialize to get the theorem. I run the full seeding: the first center is uniform, the remaining `k - 1` are `D^2`-weighted. Apply the induction with `t = u = k - 1`, taking the single cluster `A` that received the *first* (uniform) center as the only initially-covered cluster. Then `X_c = A`, `X_u = X - A` covers the other `k - 1` optimal clusters, and

at the moment I apply it the uncovered part contributes `8 phi_OPT(X - A)` and `A` contributes `phi(A)`, and with `t = u = k - 1` the leftover `((u - t)/u) phi(X_u)` term is zero. Writing the uncovered optimal cost as `phi_OPT - phi_OPT(A)`,

  E[phi] <= ( phi(A) + 8 phi_OPT - 8 phi_OPT(A) ) * (1 + H_{k-1}).

And `phi(A)`, the cost of the first cluster seeded uniformly, has expectation `2 phi_OPT(A)` by step one. Plug that in:

  E[phi] <= ( 2 phi_OPT(A) + 8 phi_OPT - 8 phi_OPT(A) ) * (1 + H_{k-1}) <= 8 phi_OPT * (1 + H_{k-1}),

since `2 phi_OPT(A) - 8 phi_OPT(A) <= 0` only helps and `8 phi_OPT >= 8 phi_OPT - 6 phi_OPT(A)`. Finally `H_{k-1} = 1 + 1/2 + ... + 1/(k-1) <= 1 + ln k`, so `1 + H_{k-1} <= 2 + ln k`, giving

  E[phi] <= 8 (ln k + 2) phi_OPT.

There's the prize. The seeding alone — before the local search even runs — produces a clustering whose expected cost is within an `O(log k)` factor of the global optimum, for *any* data set, no separation assumption. And the local search can only decrease `phi` from there, so the combined procedure inherits the `O(log k)` guarantee. This is exactly the thing the bare local search could never promise: uniform seeding has unbounded ratio, `D^2` seeding has `8(ln k + 2)`. The randomization I added to tame outliers turns out to be precisely what makes the bound provable — the expectation is over the seeding's own randomness.

Is the `log k` real, or an artifact of a loose proof? Let me see whether I can drive the bound down to a constant, because if `D^2` seeding were secretly `O(1)`-competitive I'd want to know. Consider `k` groups arranged so that any two group-centers are at mutual squared distance `Delta^2`, and within each group put points in a tiny regular simplex of side `delta << Delta`, in orthogonal dimensions so every inter-group pair is at distance `Delta` and every intra-group pair at distance `delta`. The optimal clustering is obviously one center per group, with `phi_OPT` proportional to `k delta^2`. Now run `D^2` seeding. After I've covered some groups, the uncovered groups are at distance `~Delta` from all my centers while the covered groups are at distance `~delta`; so the `D^2` mass on a single point in an uncovered group is `~Delta^2` and on a covered point `~delta^2`, with `Delta >> delta`. That strongly favors uncovered groups — good — but here's the catch: it favors them only in proportion to how many uncovered points remain versus covered points, and as more groups get covered the covered points pile up and start stealing probability. Running the same kind of induction *downward* to lower-bound the expected potential, the chance of leaving the last few groups uncovered doesn't vanish fast enough, and the expected potential keeps a harmonic term. After the constants are carried through and `n` and `Delta/delta` grow, the ratio approaches `2 ln k`. So `D^2` seeding is no better than logarithmic — the `log k` is genuinely there, my upper bound is tight up to the constant, and chasing an `O(1)` guarantee for this exact scheme is hopeless. Good to know: the analysis isn't loose, the method really is logarithmic.

One more generalization, because I want to understand *why* the exponent is 2 and not something I should reconsider. Suppose the objective were `phi^[l] = sum_x min_c || x - c ||^l` for some `l >= 1` — squared error is `l = 2`, and `l = 1` is the `k`-median objective (sum of plain distances). The matching seeding would be `D^l` weighting: sample proportional to `D(x)^l`, the contribution of `x` to *that* objective. Does the proof survive? The only place the inner-product structure of squared distance was essential is the uniform-seed computation `E[phi(A)] = 2 phi_OPT(A)`, which used the parallel-axis identity and is special to `l = 2`. For general `l` there's no centroid identity, but a weaker triangle-inequality bound suffices: `|| a - a_0 ||^l <= 2^{l-1}(|| a - c ||^l + || a_0 - c ||^l)` by the power-mean inequality, giving `E[phi^[l](A)] <= 2^l phi_OPT^[l](A)` for a uniform seed and, carrying the extra `2^{l-1}` through the single-cluster weighted-sampling analogue, an overall `E[phi^[l]] <= 2^{2l}(ln k + 2) phi_OPT^[l]`. So the same `D^l`-weighting idea works for `k`-median and beyond; for the squared-error objective I actually care about, `l = 2`, the centroid identity gives the sharp constant 8, and `D^2` is the right weighting because it matches the squared cost. The exponent isn't free — it's locked to the objective's exponent.

Before I write code, a few practical wrinkles have to line up with the mathematics instead of fighting it. The centroid step can produce an *empty cluster*: if some center wins no points in the assignment step, its mean is `0/0`, undefined. The natural repair is to relocate an emptied center to a point that is currently far from its assigned center, so the dead center is spent where the present objective is large. `D^2` sampling is also random, and a single draw can be unlucky; at each seeding step I can draw a small number of candidates by `D^2` weighting, compute for each the resulting potential `sum_x min(old D(x)^2, || x - candidate ||^2)`, and greedily keep the candidate that reduces the potential the most. A logarithmic number of candidates, exactly `2 + int(log k)`, is the lightweight version of that hedge. And since the underlying objective is non-convex with many local minima, I should run the whole procedure ten times from independent seedings, keep the run with the lowest final `phi`, and cap each Lloyd run at 300 iterations. The local loop should stop when labels are unchanged, or when the squared center shift falls below a tolerance; if it stops by center shift rather than exact label stability, I need one final assignment pass so the returned labels match the returned centers.

So let me assemble the whole thing into code, filling the one empty slot — how the centers are seeded — and writing the refine loop and the restart hedge around it.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


def _sq_dist_to_nearest(X, centers):
    """For each point: squared Euclidean distance to its nearest center, and which one."""
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)   # (n, k)
    return d2.min(axis=1), d2.argmin(axis=1)


def _kmeanspp_init(X, k, rng):
    """Seed k centers by D^2 weighting (greedy: a few candidates per center)."""
    n = X.shape[0]
    n_local_trials = 2 + int(np.log(k))           # logarithmic greedy variance reduction
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    indices = np.full(k, -1, dtype=int)

    # first center: uniform at random  -> the uniform-seed cluster costs 2*phi_OPT(A)
    center_id = rng.choice(n)
    centers[0] = X[center_id]
    indices[0] = center_id
    closest_d2 = ((X - centers[0]) ** 2).sum(axis=1)   # D(x)^2 to the one chosen center
    current_pot = closest_d2.sum()                       # phi = sum_x D(x)^2

    for c in range(1, k):
        # sample n_local_trials candidate points with probability proportional to D(x)^2
        rand_vals = rng.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(np.cumsum(closest_d2), rand_vals)
        np.clip(candidate_ids, None, n - 1, out=candidate_ids)

        # for each candidate, the potential if it were added: sum_x min(D(x)^2, ||x - cand||^2)
        distance_to_candidates = ((X[candidate_ids, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        np.minimum(closest_d2, distance_to_candidates, out=distance_to_candidates)
        candidate_pot = distance_to_candidates.sum(axis=1)

        best = np.argmin(candidate_pot)            # greedily keep the candidate that drops phi most
        current_pot = candidate_pot[best]
        closest_d2 = distance_to_candidates[best]  # D^2 to nearest of the centers chosen so far
        best_candidate = candidate_ids[best]
        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    return centers, indices


def _centers_from_labels(X, labels, centers, d2):
    """Move nonempty clusters to their means; relocate empty clusters to far points."""
    k = centers.shape[0]
    new_centers = centers.copy()
    counts = np.bincount(labels, minlength=k)
    for j in range(k):
        if counts[j] > 0:
            new_centers[j] = X[labels == j].mean(axis=0)     # centroid step
    empty = np.where(counts == 0)[0]
    if len(empty) > 0:
        farthest = np.argsort(d2)[::-1]
        for j, point_id in zip(empty, farthest):
            new_centers[j] = X[point_id]                     # empty cluster: relocate farthest
    return new_centers


def _lloyd(X, centers, max_iter=300, tol=1e-4):
    """Batch Lloyd iteration with strict-label and center-shift stopping."""
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    strict_convergence = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        d2, labels = _sq_dist_to_nearest(X, centers)         # assignment step
        new_centers = _centers_from_labels(X, labels, centers, d2)
        center_shift_tot = ((new_centers - centers) ** 2).sum()
        centers = new_centers

        if np.array_equal(labels, labels_old):
            strict_convergence = True
            break
        if center_shift_tot <= tol:
            break
        labels_old[:] = labels

    if not strict_convergence:
        d2, labels = _sq_dist_to_nearest(X, centers)         # final E-step after tol/max_iter
    else:
        d2 = ((X - centers[labels]) ** 2).sum(axis=1)
    inertia = d2.sum()                                      # final phi
    return centers, labels, inertia, n_iter


class CustomClustering(BaseEstimator, ClusterMixin):
    """Squared-error clustering: k-means++ seeding + Lloyd refinement, best of n_init restarts."""

    def __init__(self, n_clusters=None, random_state=42, n_init=10, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        k = self.n_clusters if self.n_clusters is not None else 8
        if k > X.shape[0]:
            raise ValueError("n_clusters must be no larger than n_samples")
        rng = np.random.RandomState(self.random_state)
        best_inertia, best_centers, best_labels, best_n_iter = np.inf, None, None, 0
        for _ in range(self.n_init):                          # non-convex -> hedge with restarts
            centers, _ = _kmeanspp_init(X, k, rng)            # the seeding slot, filled by D^2
            centers, labels, inertia, n_iter = _lloyd(X, centers, self.max_iter, self.tol)
            if inertia < best_inertia:                        # keep the lowest-phi run
                best_inertia = inertia
                best_centers, best_labels, best_n_iter = centers, labels, n_iter
        self.cluster_centers_, self.labels_ = best_centers, best_labels
        self.inertia_, self.n_iter_ = best_inertia, best_n_iter
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            self.fit(X)
        return _sq_dist_to_nearest(np.asarray(X, dtype=float), self.cluster_centers_)[1]
```

Let me trace the causal chain back. I wanted to partition unlabeled points by squared-distance compactness, so I scored a center set by `phi = sum_x min_c || x - c ||^2`; minimizing it exactly is NP-hard, forcing a local search. Untangling the two coupled variables gave two forced moves — assign each point to its nearest center (the pointwise `min`), and move each center to its group's mean (the parallel-axis identity, which is also why squared error and not absolute: squared error's optimal representative is the cheap closed-form mean). Alternating these makes `phi` monotonically non-increasing over a finite set of partitions, so the loop converges to a fixed point — but only a local minimum, and `phi` is non-convex, so the outcome is hostage to the seeding, with uniform seeding giving an unbounded cost ratio. I wanted seeding that spreads the centers, stays robust, and comes with a guarantee; deterministic farthest-point spreads but seeds outliers, so I randomized it, sampling each next center proportional to `D(x)^2` — the exponent matching the squared-error objective, the randomization defanging outliers. Proving its quality: the parallel-axis identity gives `E[phi(A)] = 2 phi_OPT(A)` for a uniform seed and, via triangle plus power-mean, `<= 8 phi_OPT(A)` for a `D^2` seed that lands in `A`; an induction over (centers-to-place, clusters-uncovered) accumulates a harmonic factor, and specializing to `t = u = k-1` yields `E[phi] <= 8(ln k + 2) phi_OPT` — an `O(log k)` guarantee for any data, which the subsequent local search can only improve. A simplex-of-simplices construction shows the `log k` is tight, and the `D^l` generalization shows the exponent is locked to the objective's. Then practicality: reseed empty clusters at the farthest point, take a logarithmic number of greedy `D^2` candidates per center to cut seeding variance, and restart the whole procedure several times keeping the lowest-`phi` run because the landscape is non-convex. The result is a near-linear, parameter-light clustering procedure with a provable approximation guarantee, which is exactly what the bare local search lacked.
