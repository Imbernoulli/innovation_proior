The histogram got me to 165.575 seconds per iteration on Higgs, down from the exact splitter's 3794.34,
and the cost is now cleanly localized: at every node, for every feature, I build a histogram by summing
the gradient and Hessian of *every* example into its bin. Let me size that so I know what I am fighting.
Building the histograms at one tree level touches every present entry once — on dense Higgs that is
n·d = 10.5M·28 ≈ 2.9×10⁸ bin increments per level, and a depth-8 tree pays that on the order of eight
times, ≈ 2.3×10⁹ increments per tree, ≈ 1.2×10¹² over 500 rounds. That product, (number of data points)
× (number of features) × (levels), is the whole bill now — the 255-bin scan the histogram introduced is
negligible beside it. The two leading factors are *independent* levers: I can try to use fewer data
points per histogram, and fewer features. The question is whether I can shrink either without giving up
the accuracy the histogram bought — because the reason the histogram was a free lunch (AUC actually rose
to 0.845314 while the time fell) is that its (g,h) sums were *unbiased* estimates of the exact sums, and
whatever I do to the data or feature factor has to preserve that.

Take the data factor first. The histogram sums *all* n examples, but they are not all equally
informative for finding splits. Recall what each example contributes to the split objective: through the
XGBoost gain G²/(H+λ), it enters only via its gradient gᵢ and Hessian hᵢ. An example whose gradient is
near zero is one the current model already fits well — it sits near the loss minimum, it barely tugs the
objective, and the split that is best for the large-gradient, under-fit examples is essentially unchanged
whether I include this well-fit example in the histogram or not. So the well-fit examples carry little
information about where the next split should go; the under-fit, large-gradient examples carry most of
it. The naive move is to just drop the small-gradient examples and build the histogram from the
large-gradient ones only. But that *changes the data distribution*: the histogram is supposed to estimate
sums over the full data, and if I throw away all the small-gradient examples the gradient sums Gⱼ are
biased downward on the leaves where those examples concentrate, the gain estimate is wrong, and — this is
the thing I must not break — the unbiasedness that made the histogram accurate is gone. Accuracy would
suffer in exactly the way the histogram was careful to avoid.

The fix is to keep *some* of the small-gradient examples and *correct for the ones I dropped*, so the
sums stay unbiased. Sort the examples by the magnitude of their gradient. Keep the top fraction a — the
large-gradient, informative examples — in full. From the *remaining* (1−a)n small-gradient examples,
randomly sample a fraction, keeping bn of them and throwing the rest away. Now I have under-sampled the
small-gradient tail: each kept small-gradient example stands in for several dropped ones, so when I add
its gradient into a histogram I must scale it up to compensate. The amplification is the inverse sampling
rate. A small-gradient example survives with probability (bn)/((1−a)n) = b/(1−a), so to make the kept
subset sum to the full population sum in expectation I multiply each kept one's g and h by the reciprocal
(1−a)/b. Check the algebra: the expected contribution of the sampled tail to a gradient sum is
(probability kept) × (amplification) × (full tail sum) = [b/(1−a)] · [(1−a)/b] · Σ_{small} g_j =
Σ_{small} g_j — the two factors cancel exactly, so the estimate is unbiased. Put numbers on it with the
defaults top_rate a = 0.2 and other_rate b = 0.1: the amplification is (1−0.2)/0.1 = 8, a kept
small-gradient example is counted with weight 8; the survival probability is 0.1/0.8 = 0.125; and
8 × 0.125 = 1, so on average each of the (1−a)n = 0.8n small examples contributes its full weight even
though only bn = 0.1n of them are physically present. The histogram is now built from only (a+b)n = 0.3n
examples — a 70% cut in the data factor — while its gradient and Hessian sums remain unbiased, so the
split gains, and therefore the accuracy, should be nearly unchanged. This is **Gradient-based One-Side
Sampling**, GOSS, and the per-iteration data factor drops from n to (a+b)n.

I should check GOSS against the obvious alternative, plain uniform sub-sampling — keep a random (a+b)n
fraction of *all* the examples and reweight by the inverse rate — because that is also unbiased and
simpler, so if it were as good I would prefer it. The difference is variance, not bias. Both estimators
have the right expectation, but uniform sampling spends its whole budget in proportion to the
population: the well-fit majority, which contributes almost nothing to the gain, gets most of the kept
rows, and the informative large-gradient tail is thinned by the same factor as everyone else. GOSS
instead spends its budget where the signal is — it keeps the entire top-a of large-gradient examples
exactly, with no sampling noise at all on the rows that most determine the split, and samples only the
redundant tail. For the same (a+b)n kept rows, the gain estimate under GOSS has lower variance, because
the high-leverage examples are measured exactly rather than sub-sampled. That is the whole reason to
prefer the more complicated scheme: uniform sampling and GOSS are tied on bias and GOSS wins on the
variance of the split gain, which is what actually decides whether the sampled histogram picks the same
split the full histogram would.

One detail in the ranking key is worth pausing on: I said "sort by gradient magnitude," but the objective
weights each example by its Hessian too, and the implementation ranks by |g·h| rather than |g| alone. The
reason is the completed-square form of the second-order objective — an example's pull on the
Hessian-weighted split loss scales with both how under-fit it is (|g|) and how much curvature it carries
(h) — so |g·h| is the more faithful measure of "how much this example can still move the next split,"
and it is that product the code thresholds on to pick the top-a set. Concretely, per bagging block: form
the per-example key |g·h|, find the threshold that picks out the top-a fraction, keep everyone above it,
and for everyone below it keep them with probability b/(1−a), multiplying the kept ones' g and h by the
amplification (1−a)/b. (The GOSS sampling-and-amplification loop is in the answer.)

There is a subtlety in *when* to start sampling. Early in boosting the model is crude and almost every
example has a large gradient, so the GOSS split into "informative top-a" and "redundant tail" is
meaningless — there is no well-fit tail yet. The right move is to build the first several trees on all
the data and only switch on the sampling once the model has begun to fit, which the implementation does
by skipping sub-sampling for the first ⌈1/learning_rate⌉ iterations; at learning rate 0.1 that is the
first ten rounds built on the full n, after which the (a+b)n histogram takes over. That warm-up is cheap
against 500 rounds and it keeps the early, structure-setting trees unbiased in the strongest sense —
built on everything.

Now the feature factor. The histogram is also built feature-by-feature, and real tabular data has *many*
features — and crucially, many of them are *sparse*: one-hot columns, indicators, mostly-zero counts.
Two sparse features that are almost never nonzero at the same time carry, between them, about as much
information as one feature — at any given example, at most one of them is "on." If I could merge such
mutually-exclusive features into a single histogram, I would build one histogram instead of two and lose
nothing, because they never collide. The construction: treat features as vertices of a graph, draw an
edge between two features whenever they take nonzero values on the same example (they "conflict"), and
partition the features into bundles such that within a bundle features rarely conflict. That is graph
coloring — NP-hard in general — but I do not need optimal bundles; I need good-enough ones fast, and I can
allow a small budget of conflicts per bundle. A few examples where two bundled features collide is
tolerable: it just adds a little noise to that bundle's histogram, the same kind of controlled
approximation the quantile sketch already makes. So I greedily order the features and, for each, try to
drop it into an existing bundle whose total conflict count would stay under a small budget; if none fits,
start a new bundle. The budget is set to single_val_max_conflict_cnt = total_sample_cnt / 10000 — at most
0.01% of the sampled rows may have two bundled features fire together, so on a 200k-row sample that is at
most 20 colliding rows per bundle, essentially exclusive.

The order in which I present features to this greedy pass matters, and the right order is by descending
number of nonzeros. A feature with many nonzeros is a high-degree vertex in the conflict graph — it
collides with more other features, so it is the hardest to place — and I want to place the hard ones
first, while the bundles are still empty and almost any assignment fits under the budget; the sparse,
low-degree features then slot into whatever bundles already exist. That is exactly the largest-degree-
first heuristic graph coloring uses, and for the same reason: seeding with the constrained vertices keeps
the number of colors (here, bundles) close to the true chromatic number without ever solving the NP-hard
problem exactly. And the cost EFB pays is symmetric to the one GOSS pays: GOSS trades a little gain
*variance* for a smaller data factor, while EFB trades a little histogram *noise* — the budgeted
collisions — for a smaller feature factor, both controlled approximations chosen so the split the learner
picks is the one the exact histogram would have picked.

Within a bundle I need the merged feature's values to keep the original features separable, so I offset
each feature's bin range. Suppose feature A occupies nonzero bins {1,2,3} (0 meaning "A absent") and
feature B occupies {1,2,3,4}. If I bundle them naively their bins overlap and a value of 2 is ambiguous.
So I shift B up by A's range: A keeps {1,2,3}, B becomes {4,5,6,7}, and 0 still means "both absent." Now
a bundled value of 5 unambiguously means "B fired in its bin 2," a value of 2 means "A fired in its bin
2," and as long as A and B are never both nonzero on the same row (the exclusivity the budget enforces)
the single bundled integer recovers exactly which original feature was on and which of its bins it landed
in. The histogram over the bundle is then built in one pass, and the number of *effective* features drops
from the raw count to the number of bundles. This is **Exclusive Feature Bundling**, EFB, and it shrinks
the feature factor — sharply on datasets full of sparse one-hot columns, and hardly at all on a dense
dataset where every feature fires on every row and nothing is exclusive. Put a number on the sparse case,
where EFB actually bites. Imagine a thousand one-hot columns, each nonzero on 0.1% of rows. Two such
columns both fire on the same row only about (0.001)² = 10⁻⁶ of the time in expectation, while the
conflict budget tolerates collisions on up to total_sample_cnt/10000 = 10⁻⁴ of the rows — so I can pile
on the order of a hundred columns into one bundle before the accumulated collisions reach the budget
(≈ 100 · 10⁻⁶ = 10⁻⁴). A thousand such columns then collapse into roughly ten bundles: a hundredfold cut
in the feature factor, essentially free because the columns almost never collide. The dense case is the
same arithmetic run backward — with 28 always-on Higgs features every pair collides on nearly every row,
no two fit under any sane budget, and the feature factor does not move — so the collision math itself
tells me which benchmark EFB will help and which it will leave untouched. (The greedy bundle-finder is in
the answer.)

There is a third lever, in how the tree itself grows. The histogram learners I have been building grow
trees *level-wise*: split every node at the current depth before going deeper. That keeps the tree
balanced but spends splits on nodes that barely reduce the loss, just to keep the level full. With the
gains now computed cheaply from histograms, I can instead grow *leaf-wise*: keep a priority queue of all
current leaves by their best achievable split gain, and always split the single leaf with the highest
gain, wherever it is in the tree. For a fixed budget of leaves this descends the loss faster, and I can
see why on a small case. Suppose after splitting the root into leaves L and R, L's best sub-split would
buy gain 8 and R's would buy gain 2, and I have budget for one more split. Level-wise growth insists on
filling the level, so it splits *both* L and R to complete depth 2 — but if I could only afford one, the
level-order rule has no way to prefer the gain-8 split; it would spend the split on R as readily as on L.
Leaf-wise pops the queue and splits L (gain 8), descending the loss by 8 rather than by 2 for the same
one-split budget, and only later comes back for R. Over a fixed leaf budget the greedy highest-gain
sequence dominates the level-order sequence in total gain descended. The risk is deeper, lopsided trees
that overfit, which I cap by limiting the number of leaves (and optionally the depth). Controlling
complexity by *number of leaves* rather than by depth is the natural pairing with leaf-wise growth — a
depth cap would waste the freedom that makes leaf-wise worthwhile.

Leaf-wise growth also unlocks a construction shortcut that pays back some of the cost of the priority
queue. When I split a node into two children, I do not have to build both children's histograms from
scratch: the parent's histogram is the bin-wise sum of its two children's, so if I build the histogram
of the *smaller* child (fewer rows to scan) I can recover the *larger* child's histogram by subtracting
the small one from the parent's, bin by bin — an O(#bins) subtraction instead of an O(rows) scan. Since
I always have the parent histogram in hand when I pop a leaf off the queue, and the smaller child is at
most half the rows, this histogram-subtraction trick roughly halves the accumulation work at every
split, on top of the GOSS and EFB reductions. It is the kind of saving the exact pre-sorted splitter
could never have: it exists only because the histogram is an additive, bin-indexed object.

Put the three together: GOSS cuts the data factor from n to (a+b)n by sampling away well-fit examples
while staying unbiased; EFB cuts the feature factor from d to the number of bundles by merging
mutually-exclusive sparse features; leaf-wise growth spends each split where it helps most. The
per-iteration cost was (#data)×(#features)×(levels) histogram updates; GOSS shrinks the first factor and
EFB the second, so the product drops on both axes at once. Because GOSS keeps the histogram unbiased and
EFB bundles only near-non-colliding features, the split gains — and therefore the accuracy — should be
preserved while the time per iteration falls below the histogram baseline. The two benchmarks should
behave *differently*, and that is the sharp, falsifiable part. Higgs is dense: 28 continuous features,
every one nonzero on every row, so EFB finds almost nothing to bundle and the feature factor barely moves;
the Higgs speedup below 165.575 s/iter therefore comes almost entirely from GOSS's 0.3n data factor and
the leaf-wise/engineering gains, so I expect a *modest* further drop, not another 22×, with the AUC
holding at or a touch above 0.845314 since the sums stay unbiased. MS LTR, by contrast, has 137 features,
many of them sparse ranking signals, so EFB should bundle aggressively and leaf-wise growth should find
sharper splits within the same leaf budget — there I expect a clear gain in ranking quality (NDCG@10 up),
not just a speed win. Were Higgs to speed up dramatically or MS-LTR's ranking metric to stay flat, the
mechanism would be wrong; the two tables will tell me.

This is **LightGBM**: histogram split finding plus Gradient-based One-Side Sampling, Exclusive Feature
Bundling, and leaf-wise tree growth — a further drop in per-iteration time at matched-or-better AUC on
Higgs, and a clear NDCG gain on ranking. The thing it has *not* touched is the one part of the pipeline
that has been the same since AdaBoost: how a *categorical* feature becomes a number a tree can split on.
Higgs and the ranking set are numeric, so nothing here has been tested against categories. But on a
dataset whose features are categorical identifiers with thousands of values, the standard move is to
replace each category with a statistic of the target — and that, done naively, quietly uses each example's
own label to encode its own features. Every learner so far, mine included, would do it that way, and on
numeric benchmarks the flaw is invisible. That is the next thing to question.
