The strategy is the whole point, but it has to bolt onto a budget and a search space, and with the
plainest possible strategy that combination is the floor — so the thing to start from is just spending
30 queries on this space at all, without trying to be clever about which 30. I have 15,625 architectures
and exactly 30 lookups; the objective `f(a)` — validation accuracy on the tabular benchmark — is a black
box with no gradient over a discrete cell graph, and a single query costs one of my thirty units. The
question is narrow: out of a budget of 30 evaluations, which architectures do I evaluate, and which one
do I return? Before I reach for anything adaptive I want the honest yardstick that every adaptive method
has to beat, and that yardstick is the strategy that uses *none* of the history — draw each architecture
independently and uniformly, keep the best one by validation accuracy.

Let me write down why this is the right floor rather than an arbitrary one. The temptation in a budgeted
black-box search is to compound information: model the `(a, f(a))` pairs seen so far and let them steer
the next draw. That compounding is exactly what the rungs above this one will do — evolution exploits
the best-seen, a predictor models all of them — but each of those mechanisms can also *hurt*, because
each makes an assumption (the landscape is locally smooth, the surrogate generalizes) that a 30-query
budget may not justify. Uniform random sampling makes no such assumption. Every draw is an independent,
identically distributed gamble over the whole space, so the method has no way to be misled by a wrong
prior about where good architectures live — and that very property, "deliberately ignores the history,"
is what makes it the baseline against which adaptivity must prove its worth, not the last word.

What does 30 uniform draws actually buy me, quantitatively? Idealize "find a good architecture" as
hitting a target region that occupies a fraction `p` of the 15,625 architectures. Each draw lands in the
target with probability `p` and misses with `1 - p`; the draws are independent, so the chance all 30
miss is `(1 - p)^30`, and the chance at least one hits is `1 - (1 - p)^30`. Notice what is *not* in that
formula: the ambient size 15,625 never appears, only the *relative* volume `p` of the good region. So
the question of whether 30 draws suffice has nothing to do with how big the space is and everything to
do with how big a fraction of it is good — and on NAS-Bench-201 that fraction is mild. The benchmark is
small and many edge configurations are near-equivalent (swapping a `nor_conv_1x1` for a `nor_conv_3x3`
on a non-critical edge barely moves the accuracy), so a large fraction of the 15,625 cells sit within a
couple of accuracy points of the best. If, say, the top ~5% of architectures count as "good," then
`1 - 0.95^30 ≈ 0.79` — about four runs in five land at least one good architecture in 30 draws, purely
by volume. If "good" is the top ~10%, it is `1 - 0.90^30 ≈ 0.96`. That is the reason random search is
not a strawman on this benchmark and is genuinely hard to beat: the relative volume of the good region
is large enough that thirty independent shots usually catch it, so the ceiling a smarter method can claw
back above this floor is *narrow*. The whole sample-efficiency story lives in that narrow band — and it
is exactly the regime the research question targets (K ≤ 50), where the literature reports a measurable
but small gap.

It is worth being precise about *why* I draw uniformly rather than, say, on some structured sweep of the
six edges. A grid over the cell would fix a handful of operation values per edge and take their Cartesian
product, but with six edges that product is exponential, and worse, an aligned grid probes each
individual edge at only the sixth root of the budget — thirty grid points laid on the 6-dimensional cell
would resolve each edge at barely two settings, and project down onto any one edge as two stacks of
coincident points. Uniform independent draws collapse under no such projection: each draw chooses all six
edges independently, so the thirty draws probe every edge at (almost) thirty distinct settings at once,
giving whichever edges actually matter for accuracy the full budget's worth of resolution without my
having to know in advance which edges those are. That is the same per-axis-resolution argument that makes
random search beat grid in hyperparameter optimization, and it is why the scaffold's `random_architecture`
draws each of the six op-indices independently rather than enumerating a lattice.

The one thing I must get right is the bookkeeping, because the budget is hard. The loop calls
`search_step(epoch)` up to 30 times and counts every `query_val_accuracy` call; a 31st call aborts the
whole run. So each step must query exactly once, and I must never re-query or wrap a query to stretch the
budget — the task explicitly forbids it, and the harness would catch it. Each step I draw one valid
architecture from the fixed helper `random_architecture()` (which already rejects degenerate all-`none`
cells), query its validation accuracy, and if it beats the running best I store it. At the end
`get_best_architecture()` hands back the best-by-validation architecture, and the harness scores it on
the unbudgeted test split. There is no state worth keeping beyond the running best and its score — the
method is non-adaptive by construction, so the history is never consulted to decide the next draw.

There is a subtlety I should name even though I am not going to fix it here, because it is what the next
rung will react to. The architecture I *return* is chosen by its **validation** accuracy, but the metric
is **test** accuracy, and the two are not perfectly monotone: among several architectures with nearly
tied validation scores, which one I crown as best is itself a little noisy, and a different draw could
anoint a different one with a slightly different test score. With only 30 samples the running best is
often decided by one or two near-ties, so the test accuracy I report inherits that selection noise as
seed-to-seed variance. Uniform sampling does nothing to suppress it — it neither concentrates its queries
where the good region is nor re-examines near-ties — so I should expect the variance across the five
seeds to be the dominant feature of this baseline, not its mean. This is also why an honest report needs
both numbers: a method that lifts the mean by a few tenths of a point but does so *consistently* across
seeds is a real improvement over random search even when a single lucky random-search seed happens to
tie it, and the only way to see that is to read the per-seed column, not just the mean. The five seeds
here are not a nuisance to be averaged away; at K = 30 they *are* the signal, because the variance is on
the same order as the gap the ladder is trying to open.

So at step 1 the edit is the trivial one: the scaffold default already *is* uniform random search, and
the literal fill is exactly the placeholder `NASOptimizer` — sample, query once, track the best (the
distilled class is in the answer). Nothing is added.

Now reason about what this floor must do across the three settings, because that is the entire point of
running it. On all three datasets the mechanism is identical — thirty independent shots at a space whose
good region is a sizeable fraction — so the *mean* should land at a respectable but unremarkable accuracy
on each, a little below whatever an adaptive method can reach, and the *spread* should be visible. I
expect CIFAR-10 to be the tightest of the three, because its accuracy distribution over the benchmark is
compressed near the top (many cells reach ~93–94%), so even unlucky draws land close to good; CIFAR-100
and especially ImageNet16-120 have a wider spread of architecture quality, so a run that fails to draw
into the top region pays a larger penalty and the seed-to-seed variance should be larger there. The
diagnosis I am already pointing at the next rung is this: random search has no memory, so when a seed
draws a strong architecture early it cannot *build* on it, and when it draws poorly it cannot *recover*.
The obvious first lever is to stop throwing every draw away — keep a small set of the best-seen and spend
later queries near them instead of uniformly. Whatever the precise per-dataset numbers, the failure mode
is structural: thirty independent gambles with no exploitation of what they revealed, so the result is
governed by luck, and the way to beat it is to let the good draws steer the later ones.
