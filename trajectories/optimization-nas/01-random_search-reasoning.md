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
the next draw. That compounding is exactly what any adaptive successor to this floor will try — but any
such mechanism can also *hurt*, because it leans on an assumption (that the landscape is locally smooth,
or that a fitted model generalizes from a handful of points) that a 30-query budget may not justify.
Uniform random sampling makes no such
assumption. Every draw is an independent, identically distributed gamble over the whole space, so the
method has no way to be misled by a wrong prior about where good architectures live — and that very
property, "deliberately ignores the history," is what makes it the baseline against which adaptivity must
prove its worth, not the last word. The point of running it is not to win; it is to fix the number that
tells me how much room there even is above it, because if that room is narrow, a clever method that beats
the floor by a hair is not obviously worth its extra assumptions.

What does 30 uniform draws actually buy me, quantitatively? Idealize "find a good architecture" as
hitting a target region that occupies a fraction `p` of the 15,625 architectures. Each draw lands in the
target with probability `p` and misses with `1 - p`; the draws are independent, so the chance all 30
miss is `(1 - p)^30`, and the chance at least one hits is `1 - (1 - p)^30`. Notice what is *not* in that
formula: the ambient size 15,625 never appears, only the *relative* volume `p` of the good region. So
the question of whether 30 draws suffice has nothing to do with how big the space is and everything to
do with how big a fraction of it is good. Let me tabulate it so the shape is concrete. If the target is
the top 1% I get `1 - 0.99^30 ≈ 0.26`; the top 2%, `1 - 0.98^30 ≈ 0.45`; the top 3%, `≈ 0.60`; the top
5%, `1 - 0.95^30 ≈ 0.79`; the top 10%, `1 - 0.90^30 ≈ 0.96`; the top 20%, `≈ 0.999`. The curve is steep
in exactly the band that matters: somewhere between "top 1% is a coin-toss-and-worse" and "top 10% is
nearly certain." The half-way point of that curve is worth pinning down because it recurs below: the
target volume at which thirty draws touch the region with probability exactly one-half solves
`(1 - p)^30 = 0.5`, giving `p = 1 - 0.5^{1/30} ≈ 0.023` — the top ~2.3% is the fraction random search
reaches on a coin flip in thirty draws. And on NAS-Bench-201 that fraction is mild. The benchmark is small and many edge
configurations are near-equivalent (swapping a `nor_conv_1x1` for a `nor_conv_3x3` on a non-critical edge
barely moves the accuracy), so a large fraction of the 15,625 cells sit within a couple of accuracy
points of the best. If "good" means the top ~5%, four runs in five land at least one good architecture
in 30 draws, purely by volume. That is the reason random search is not a strawman on this benchmark and
is genuinely hard to beat: the relative volume of the good region is large enough that thirty independent
shots usually catch it, so the ceiling a smarter method can claw back above this floor is *narrow*. The
whole sample-efficiency story lives in that narrow band — and it is exactly the regime the research
question targets (K ≤ 50), where a measurable but small gap is what the ladder is fighting over.

The hit-or-miss framing is coarse, though — it tells me the good region is *reached* but not how good the
single best draw is, and the metric rewards the best one I return, not the mere event of touching the
region. So let me sharpen it with the order statistics, because that is where the seed variance is going
to come from and I want it on paper before I run anything. Rank every architecture from the top and call
its position the quantile `q` — `q = 0` is the single best cell, `q = 1` the worst. Thirty uniform draws
give thirty uniform quantiles, and the best draw is the minimum of thirty uniforms. The expectation of
the minimum of `n` uniforms is `1/(n+1)`, so with `n = 30` the best architecture I keep sits, in
expectation, at quantile `1/31 ≈ 0.032` — the top 3.2% of the benchmark. Two sanity checks that the
formula is the right one: with `n = 1` it gives `1/2`, and a single uniform draw is indeed expected at
the median, the top 50%; and as `n → ∞` it goes to `0`, the search saturating onto the true best, which
is what unbounded sampling should do. Both hold, so I trust `1/(n+1)`.

The expectation alone hides the story, because the minimum of uniforms is heavily skewed — it follows a
Beta(1, 30) distribution — and it is the *spread* of that distribution that becomes my seed-to-seed
variance. Reading off its percentiles: the 5th percentile of the best-of-30 quantile is `1 - 0.95^{1/30}
≈ 0.0017`, the median is `1 - 0.5^{1/30} ≈ 0.023`, and the 95th percentile is `1 - 0.05^{1/30} ≈ 0.095`.
So across seeds, the architecture random search returns lands — ninety per cent of the time — somewhere
between the top 0.17% and the top 9.5% of the benchmark, with the median at the top 2.3%. That is a
factor of roughly fifty in quantile between a lucky seed and an unlucky one, and it is not a defect of the
method; it is the intrinsic dispersion of the best-of-thirty order statistic. The five seeds are not
noise around a point estimate to be averaged away — they *sample* this Beta tail, and the tail is wide.

The two framings I have now — "probability of hitting a target of volume `p`" and "quantile of the best
draw" — are not independent, and checking that they agree is a free confirmation that I have not fooled
myself. The median of the best-of-30 quantile came out to `1 - 0.5^{1/30} ≈ 0.023`, and the coin-flip
target volume from the hit-probability curve came out to the *same* `1 - 0.5^{1/30} ≈ 0.023`. That is not
a coincidence: the statement "half the seeds return a cell in the top `p`" is literally the statement "the
median best quantile is `p`," so the median of the minimum order statistic must equal the `p` at which
`P(\text{hit}) = 0.5`. They are one identity written two ways, and they land on the same number, so the
two calculations are consistent. Median top-2.3%, ninety-per-cent band from top-0.17% to top-9.5% — that
is the profile of the returned architecture, before a single query is spent.

Whether that wide spread in *quantile* shows up as a wide spread in *accuracy* depends entirely on how
steep each dataset's accuracy-versus-rank curve is near the top. This is the bridge from the abstract
order statistic to the three columns I will report. Where the top of the distribution is compressed —
many cells piled within a point or two of the ceiling — even a top-9.5% draw is close in accuracy to a
top-0.2% draw, so the fifty-fold quantile spread collapses into a small accuracy spread. Where quality
falls off faster as you leave the very top, the same quantile band stretches into a wider accuracy band.
I do not have the per-dataset curves in front of me, but I can predict their ordering from what I know of
the benchmark: CIFAR-10 saturates near the top (a large mass of cells reach ~93–94%), so its
accuracy-versus-rank curve is flat there and I expect the *tightest* seed spread; CIFAR-100 and especially
ImageNet16-120 spread their architecture quality wider and have steeper curves off the top, so the same
Beta(1, 30) dispersion should map to a *larger* accuracy spread on those two. That is a falsifiable
mechanistic claim, and the per-seed columns will confirm or refute it — I am not asserting it, I am
predicting it from the order statistic and the shape of the top of each distribution.

There is one more number worth computing before I accept 30 as the budget, and it is the marginal value
of an extra uniform draw, because it explains why the *later* draws are the ones adaptivity will want to
repurpose. The expected best quantile after `k` draws is `1/(k+1)`: after 10 draws it is `1/11 ≈ 0.091`,
after 20 it is `1/21 ≈ 0.048`, after 29 it is `1/30 ≈ 0.033`, and the 30th draw moves it to `1/31 ≈
0.032`. The marginal improvement from the 30th shot is about a thousandth of a quantile, versus the first
ten draws that carry the quantile from 0.5 all the way to 0.09. The return on additional uniform sampling
is sharply diminishing — the knee is near ten draws — so the last twenty independent gambles each buy
almost nothing on their own. That is precisely the slack an adaptive method will try to reclaim: not by
sampling *more*, which the diminishing curve says is nearly worthless, but by spending those
low-marginal-value later draws *near* what the early draws revealed, where the return per query is no
longer governed by the flat tail of the order statistic. So the floor already tells me where the lever
is, before I have run a single adaptive step.

It is worth being precise about *why* I draw uniformly rather than on some structured sweep of the six
edges, because the alternative is tempting and wrong for a computable reason. A grid over the cell would
fix a handful of operation values per edge and take their Cartesian product, but with six edges that
product is exponential, and worse, an aligned grid probes each individual edge at only the sixth root of
the budget — thirty grid points laid on the 6-dimensional cell would resolve each edge at `30^{1/6} ≈
1.76` settings, barely two of the five operations, and project down onto any one edge as two stacks of
coincident points with three of the five operations on that edge never tried at all. Uniform independent
draws collapse under no such projection: each draw chooses all six edges independently, so the thirty
draws probe every edge at (almost) thirty distinct settings at once, giving whichever edges actually
matter for accuracy the full budget's worth of resolution without my having to know in advance which
edges those are. That is the same per-axis-resolution argument that makes random search beat grid in
hyperparameter optimization, and it is why the scaffold's `random_architecture` draws each of the six
op-indices independently rather than enumerating a lattice.

A small worry about the independent draws is duplicates — sampling with replacement could waste a query
re-evaluating a cell I already know — but the arithmetic says not to bother guarding against it. The
expected number of collisions in 30 draws from 15,625 is `C(30,2)/15,625 = 435/15,625 ≈ 0.028`, so the
probability of even one repeat across all thirty draws is under three per cent. Sampling with replacement
and sampling without are, at this budget, statistically identical, and adding a dedup guard would buy a
fraction of a query in expectation while complicating the loop. I leave the draws independent and spend
the budget on genuinely new evaluations by accident of the numbers, not by machinery.

There is a non-adaptive design one rung more sophisticated than i.i.d. that I should rule out on its
merits rather than by reflex: a stratified or low-discrepancy layout — a Latin-hypercube-style design that
forces each of the five operations to appear on each edge a balanced number of times across the thirty
draws, so the per-edge coverage is exactly even instead of merely even in expectation. On a continuous box
this reliably beats i.i.d., which is why it is tempting. But two things kill its advantage here. First, the
axis is categorical with five levels and I have thirty draws, so plain independent sampling *already*
lands roughly six draws on each of the five operations of each edge — the marginal per-edge coverage is
close to balanced by volume, and a stratified design only tidies up the last bit of imbalance the
`1/(k+1)` curve says is nearly worthless anyway. Second, and decisively, accuracy on this cell is driven
by *interactions* between edges — which operations sit on which paths together — not by any edge's
marginal operation frequency, and a Latin-hypercube design balances the marginals while doing nothing for
the joint coverage that actually matters. So it would add construction machinery and a fixed design that
cannot adapt, in exchange for tidying a marginal that is already fine and ignoring the interactions that
are not. Independent uniform draws are the honest floor; the stratified refinement is effort spent on the
wrong axis.

The one thing I must get right is the bookkeeping, because the budget is hard and the harness is
unforgiving. The loop calls `search_step(epoch)` up to 30 times and counts every `query_val_accuracy`
call; a 31st call aborts the whole run with `BudgetExceededError`. So each step must query exactly once,
and I must never re-query or wrap a query to stretch the budget — the task explicitly forbids it, and the
harness would catch it. Each step I draw one valid architecture from the fixed helper
`random_architecture()` (which already rejects degenerate all-`none` cells, so I do not spend a query on
a cell that cannot compute), query its validation accuracy, and if it beats the running best I store both
the architecture and its score. At the end `get_best_architecture()` hands back the best-by-validation
architecture, and the harness scores it on the unbudgeted test split. There is no state worth keeping
beyond the running best and its score — the method is non-adaptive by construction, so the history is
never consulted to decide the next draw, and `best_val_acc` is monotone non-decreasing across the thirty
steps by construction, which is the whole invariant the class has to maintain. It is worth a quick trace
to be sure the trivial code actually holds that invariant: initialize `best_val_acc = -1`; a first draw
scoring, say, 90 beats `-1` and is stored; a second draw at 88 fails the `> best_val_acc` test and is
discarded, best stays 90; a third at 92 wins and best becomes 92. The stored architecture only ever moves
to a strictly better validation score, so the returned cell is exactly the argmax-by-validation over the
thirty draws — the intended behavior, and nothing about the order of draws can break it.

The five seeds are what turn this single-run profile into the report, and it matters that they are
genuinely independent samples of that Beta(1, 30) tail. Each seed reseeds the RNG behind
`random_architecture`, so the five runs draw five entirely different sequences of thirty architectures —
five independent minima-of-thirty. That is precisely why the five per-seed numbers *are* a sample of the
order-statistic distribution rather than five noisy looks at one point: the lucky seed is one that drew a
low-quantile minimum, the unlucky seed one whose best-of-thirty landed out in the top-9% part of the tail,
and with only five draws from a distribution whose 90% span is a factor of fifty in quantile, seeing a wide
spread is the *expected* outcome, not an anomaly to explain away. I should read the five numbers as five
darts thrown at the same Beta tail, and judge any adaptive successor by whether it moves the *whole* cloud
up and tightens it, not by whether it beats the single luckiest random-search seed.

There is a subtlety I should name even though I am not going to fix it here, because it is what the next
rung will react to. The architecture I *return* is chosen by its **validation** accuracy, but the metric
is **test** accuracy, and the two are not perfectly monotone: among several architectures with nearly
tied validation scores, which one I crown as best is itself a little noisy, and a different draw could
anoint a different one with a slightly different test score. This is a second, independent source of seed
variance sitting on top of the order-statistic one. The Beta(1, 30) dispersion already says the best-by-
validation *quantile* wanders across seeds; the validation-to-test slippage then adds that even a
fixed-quantile validation pick maps to a spread of test scores. With only 30 samples the running best is
often decided by one or two near-ties in validation, so the test accuracy I report inherits that selection
noise. Uniform sampling does nothing to suppress either source — it neither concentrates its queries where
the good region is nor re-examines near-ties to break them on a second signal — so I should expect the
variance across the five seeds to be the dominant feature of this baseline, not its mean. This is also why
an honest report needs both numbers: a method that lifts the mean by a few tenths of a point but does so
*consistently* across seeds is a real improvement over random search even when a single lucky random-search
seed happens to tie it, and the only way to see that is to read the per-seed column, not just the mean. At
K = 30 the five seeds *are* the signal, because the variance is on the same order as the gap the ladder is
trying to open.

So at step 1 the edit is the trivial one: the scaffold default already *is* uniform random search, and
the literal fill is exactly the placeholder `NASOptimizer` — sample, query once, track the best (the
distilled class is in the answer). Nothing is added, and that is deliberate: adding anything here would be
adding an assumption I have no evidence to justify yet, and the entire purpose of this rung is to be the
assumption-free reference.

Now reason about what this floor must do across the three settings, because that is the entire point of
running it. On all three datasets the mechanism is identical — thirty independent shots at a space whose
good region is a sizeable fraction, with the best-kept sitting at the top ~2–3% in expectation — so the
*mean* should land at a respectable but unremarkable accuracy on each, a little below whatever an adaptive
method can reach, and the *spread* should be visible and should follow the ordering the order statistic
predicts through each dataset's top-of-distribution steepness. CIFAR-10 should be the tightest, its
compressed top flattening the Beta dispersion into a small accuracy band; CIFAR-100 wider; ImageNet16-120
the widest, on a lower base, because its quality falls off fastest from the top so an unlucky high-quantile
draw pays the largest accuracy penalty. If I see a clean staircase of per-seed values on ImageNet16-120 —
several seeds falling in near-monotone order with no plateau — that is the visual signature of pure
luck-of-the-draw with no structure, the Beta tail realized in five samples, and it would be the clearest
confirmation that the variance here is intrinsic to memoryless sampling rather than an artifact.

The diagnosis I am already pointing at the next rung follows directly from the marginal-value curve:
random search has no memory, so when a seed draws a strong architecture early it cannot *build* on it, and
when it draws poorly it cannot *recover* — and the later draws, which the `1/(k+1)` curve says are nearly
worthless as independent gambles, are exactly the queries an adaptive method should redirect. The obvious
first lever is to stop throwing every draw away: keep a small set of the best-seen and spend those
low-marginal-value later queries near them instead of uniformly, so a good early draw can be exploited and
a poor start can be climbed out of. Whatever the precise per-dataset numbers, the failure mode is
structural — thirty independent gambles with no exploitation of what they revealed, a result governed by
the Beta tail rather than by any reasoning — and the way to beat it is to let the good draws steer the
later ones.
