The differentiable DNF landed at 0.7605 on the random family, 0.9088 on monotone, 0.8986 on sparse — a
geometric mean of 0.8532, and the shape of that split is exactly the failure I was worried about, only
sharper. The model with the *exact target structure baked in* is the weakest thing I have run, and it is
weak precisely where I predicted: the mixed-polarity random family, fully 14 points below the monotone
number and 9 below the sparse number. Now that I also have the base rates the driver reports — 0.5034 on
random, 0.6277 on monotone, 0.5112 on sparse — I can read those raw accuracies as *earned* accuracy
above chance, which is the honest scale. On random the model clears its base rate by `0.7605 − 0.5034 =
0.257`; on monotone by `0.9088 − 0.6277 = 0.281`; on sparse by `0.8986 − 0.5112 = 0.387`. So the ordering
by earned accuracy is sparse > monotone > random, and the random family is not just lowest in raw score,
it is lowest in genuine learning — the model recovered barely half of the gap to a perfect classifier
there. That base-rate table also settles the question I left open on the first rung: my independent-terms
estimate put monotone near 0.72, and the true base rate is 0.6277, confirming that the 20 terms overlap
heavily enough on 40 variables to pull the positive rate well below the independent figure — a real
correlation among term firings, exactly as the 80-literal-slots-in-40-variables packing predicted.

So two things are wrong at once, and the earned-accuracy view sharpens both. The noisy-OR over 32-plus
soft conjunctions is accumulating precision errors on the non-monotone family — every slightly-imprecise
mined term adds false-positive mass, and OR-ing them takes the union, so the random number bleeds down
toward its 0.50 base rate. My accumulation arithmetic from the first rung is quantitatively consistent
with what I see: to sit at 0.7605 with a base rate near 0.50, the union is flipping a substantial
fraction of true negatives to positive, which under `1 − (1−p)^{40}` needs a per-term residual leakage of
only a couple percent — precisely the regime I flagged as invisible clause-by-clause but ruinous in
aggregate, and precisely the family whose candidate pool was thinned to 73% before training. And even on
the easy families, the soft-DNF refinement is leaving accuracy on the table: a continuous relaxation of a
discrete formula never quite snaps to the formula, and the residual softness costs a few percent
everywhere — monotone and sparse sit near 0.90, respectable but nowhere near the near-perfection 20000
examples of a width-4 concept ought to permit. There is a cost signal too: the sparse fit took 37.97
seconds against 12.6 for the other two, a 3x tax, and I know exactly why — the candidate mining matches
each of tens of thousands of width-4 patterns against 20000 examples over `n = 60`, so the largest input
dimension pays the most. The lesson is blunt: hand-shaping the hypothesis class into relaxed DNF, trained
from uniform random examples, does *not* beat a model that simply carves the input space by variable. So
I should stop relaxing the logic and use a model that does exact, axis-aligned conjunctive splits
natively — a decision tree — and fix the variance problem that one tree has by averaging many.

Here is the structural fact that makes a tree the natural learner for this task, and it is the same fact
the noisy-OR was fumbling. A decision tree splits on one variable at a time; a root-to-leaf path of
length `w` is a conjunction of `w` literals — *exactly* a DNF term. So the set of leaves labelled 1 is a
DNF, and a sufficiently deep tree can represent the target DNF *exactly*, with no relaxation and no
threshold to tune. Mixed polarity, the thing that sank deep_dnf, is *free* for a tree: a split on `x_i`
sends `x_i = 1` one way and `x_i = 0` the other with no asymmetry, so a negative literal costs exactly
what a positive one does, unlike the three-way softmax that had to resolve a genuine polarity ambiguity
on every variable. A single unrestricted tree, grown until pure, will in fact memorize the 20000 training
points perfectly. The catch is the one every tree has: a single deep tree is a high-variance estimator.
Small changes in the training sample move the early splits, and because every later split is conditioned
on the earlier ones, an early perturbation cascades into a wholly different tree. With 20000 uniform
examples covering a 30-to-60-dimensional Boolean cube extremely sparsely — `2^{30} ≈ 10^9` up to `2^{60}
≈ 10^{18}` possible inputs, of which I see 20000 — the leaves near the bottom are supported by a handful
of points each, and their labels are essentially guesses. Concretely, isolating one width-4 term needs a
depth-4 path, and a pure tree keeps splitting well past that; a leaf eight or ten levels down partitions
the 20000 points into thousands of cells, so many bottom leaves hold one or two examples and predict from
noise. So one tree overfits: perfect on train, noisy on a fresh uniform test point that falls in a
sparsely-supported region. The diagnosis from deep_dnf carries over — I have a *variance* problem, not a
representation problem — and the fix is the same shape as averaging many soft conjunctions, except I
average many *exact* trees instead.

The averaging argument is precise. Take a predictor that is an average of
trees, each grown on a resampled version of the data, and consider its variance. If the individual trees
were independent and each had variance `σ²`, the average of `B` of them would have variance `σ²/B` —
variance vanishes as I add trees. Real trees grown on bootstrap resamples of the same data are *not*
independent; they share most of the data, so their predictions are positively correlated with some
pairwise correlation `ρ`, and the variance of the average is `ρσ² + (1−ρ)σ²/B`. Put numbers on it with
`B = 200` trees: the second term is `(1−ρ)σ²/200`, essentially gone, so the ensemble variance is pinned
almost entirely at the floor `ρσ²`. If bootstrap-only trees are correlated at, say, `ρ = 0.6`, I have
knocked variance down to `0.6σ²` and adding a thousand more trees would barely move it — the floor, not
the tree count, is what binds. So the whole game in turning a high-variance tree into a low-variance
ensemble is twofold: keep growing trees (drives the `(1−ρ)σ²/B` term down for free, and 200 already
drives it to under half a percent of `σ²`) and, more importantly, *drive down the correlation `ρ`*
between trees. Bagging — bootstrap aggregation — gets `ρ` down a little by giving each tree a different
resample (each bootstrap omits ≈ 37% of the data, since `(1 − 1/N)^N → e^{-1} = 0.368`, so trees
disagree on the held-out-of-bag regions). But bootstrapping alone leaves the trees highly correlated,
because a few strong variables dominate the top splits of *every* tree: if one variable is the single
most informative split, nearly every bootstrap tree puts it at the root, and the trees look alike near
the top where it matters most.

The decisive move is to decorrelate the trees by *also* randomizing the splits. At each node, instead of
searching all `n` variables for the best split, restrict the search to a fresh random subset of `m ≪ n`
variables and pick the best split among those. The default `m = sqrt(n)` gives `sqrt(30) ≈ 5` of 30,
`sqrt(40) ≈ 6` of 40, `sqrt(60) ≈ 7` of 60. Now no single strong variable can dominate every tree's
root, because in many trees it simply is not among the `m` candidates at the top node — on random, a
given variable is one of 5 drawn from 30, so it appears at the root of only about `5/30 = 1/6` of trees;
different trees are forced to build out of different variables, their errors decorrelate, `ρ` drops, and
the ensemble variance floor `ρσ²` drops with it. This is the one idea that separates a random forest from
plain bagged trees, and it is exactly what I need on this task: the target DNF has `s` terms over `w`
variables each, so there are several genuinely informative variables, and feature subsampling spreads the
trees across them so the ensemble sees the whole term structure rather than fixating on the few variables
with the largest marginal signal. Crucially, the variance reduction does *not* come at the cost of bias
the way pruning would: I grow each tree deep (unrestricted depth), so each individual tree is low-bias /
high-variance — it can represent any of the DNF terms — and I rely entirely on the average to kill the
variance. Deep, unpruned, decorrelated, averaged: that is the recipe.

Before I accept the `sqrt(n)` subset as free, I have to check what it does on the sparse family, because
there decorrelation and coverage pull against each other. Sparse has only 12 relevant variables out of
60, and each node draws `m = 7`. The probability that a random 7-subset of 60 contains *at least one* of
the 12 relevant variables is `1 − C(48,7)/C(60,7)`. Compute the miss probability as a falling product:
`(48/60)(47/59)(46/58)(45/57)(44/56)(43/55)(42/54) = 0.800 · 0.797 · 0.793 · 0.789 · 0.786 · 0.782 ·
0.778 = 0.191`. So about `1 − 0.191 = 0.81` of nodes have a relevant variable available to split on, and
roughly 19% of split attempts are forced to choose among pure noise variables. That is not fatal — four
in five nodes can still make progress toward the junta — but it is a real drag: a fifth of the tree's
splitting budget is spent carving the input space on irrelevant coordinates, which manufactures exactly
the sparsely-supported noise leaves I worried about. So sparse is the one family where the forest's
decorrelation trick could partly backfire, and I would watch for the sparse number coming in *below* the
random number even though the junta "should" be the easier target once you know which 12 variables
matter.

I should weigh the two knobs I am freest to move — the feature-subset size `m` and the leaf floor — against
the alternatives, because both cut two ways. On `m`, the tempting move is to push decorrelation further:
set `m = 1`, an extreme-randomized forest where each node splits on a *single* randomly chosen variable,
which drives `ρ` toward its minimum. I walk it and reject it for this task. With `m = 1` on the sparse
family, a node splits on a relevant variable only `12/60 = 0.20` of the time, so four nodes in five split
on pure noise, and the coverage calculation I just did collapses from 81% to 20% per node — the trees
would be so busy carving noise that they rarely assemble a clean width-4 relevant path, trading variance
for a bias I cannot afford on the junta. The `sqrt(n)` default is the balance point: small enough to keep
`ρ` low (a specific strong variable heads only ~1/6 of trees on random) yet large enough that 81% of
sparse nodes still find a relevant split. On the leaf floor, `min_samples_leaf` trades variance against
bias directly. Setting it to 1 lets a leaf be a single training point — pure noise on this sparse cube,
where a bottom leaf holding one of 20000 examples in a `2^{30}`-plus input space predicts from that one
label. Setting it high, say 50, would smooth away real width-4 terms: a term fires on `1/16` of inputs,
about 1250 of 20000, but a *specific* term reached only after several correct splits is supported by far
fewer points at its leaf, and a floor of 50 could refuse to isolate it. `min_samples_leaf=2` is the light
touch — it forbids only the single-point leaves that are provably noise while leaving every genuinely
supported term reachable. None of this required looking at the hidden terms.

So the forest's achievable accuracy is set by the `ρσ²` floor, and that floor is set by how much
decorrelation the sparse family will tolerate: pushing below it needs more feature randomization, which
the sparse-coverage argument just said I cannot spend. If the forest plateaus a few points short of
perfection on some family, the cure will not be "more trees" but a different aggregation entirely.

The fill is almost trivial, and that is the point — no bespoke network, just the right off-the-shelf
estimator: `RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, n_jobs=-1)` on
the default uniform sample, `.fit` then `.predict`. And I expect it *far* cheaper than deep_dnf — no
candidate mining, no gradient loop — so the 12-to-38-second fits should collapse toward a fraction of a
second. The full module is in the answer.

Expectations against deep_dnf, family by family. On **random** (0.7605) the forest should *clearly beat*
it: exact axis-aligned splits do not suffer the union-of-errors, mixed polarity is free (a split tests
`x_i = 0` or `x_i = 1` symmetrically), and the target is narrow, so random should jump well above 0.76 —
the comparison that vindicates the switch. On **sparse** (0.8986), by the 81%-coverage figure the junta
is learnable but a fifth of the splits waste on noise, so a gain but not a runaway, possibly sparse below
random. On **monotone** (0.9088) I am least sure the forest wins: 20 width-4 terms over 40 variables is a
*wide* target, and a specific term is covered only if some tree assembles its four-variable path — per-node
availability `6/40 = 0.15`, so the per-tree per-term probability `q` is small and the ensemble rescues it
as `1 − (1 − q)^{200}`, which is 0.98 for `q = 0.02` but collapses to 0.63 for `q = 0.005`. It takes only
a few under-covered terms to cap monotone a real distance below its representational ceiling — a coverage
shortfall, not a representation failure. So monotone flat-to-down.

The commitment: the geometric mean rises above 0.8532, driven almost entirely by the random gain. If
random fails to clear 0.76, my noisy-OR diagnosis was wrong. And if the forest still leaves any family
short of near-perfect, the next lever is to stop averaging *independent* trees and fit each new tree to
the *residual error* — boosting.
