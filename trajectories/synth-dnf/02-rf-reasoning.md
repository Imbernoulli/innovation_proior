The differentiable DNF landed at 0.7605 on the random family, 0.9088 on monotone, 0.8986 on sparse — a
geometric mean of 0.8532, and the shape of that split is exactly the failure I was worried about, only
sharper. The model with the *exact target structure baked in* is the weakest thing I have run, and it is
weak precisely where I predicted: the mixed-polarity random family, fully 14 points below the monotone
number and 9 below the sparse number. The monotone and sparse families, where the warm-start mining has
a clean variable-mean signal and a simpler polarity decision, sit near 0.90 — respectable but not
solved, and notably below their own ceiling given that 20000 examples of a width-4 concept should be
learnable to near-perfection. So two things are wrong at once. The noisy-OR over 32+ soft conjunctions
is accumulating precision errors on the non-monotone family — every slightly-imprecise mined term adds
false-positive mass, and OR-ing them takes the union, so the random number bleeds down toward the base
rate. And even on the easy families, the soft-DNF refinement is leaving accuracy on the table: a
continuous relaxation of a discrete formula never quite snaps to the formula, and the residual softness
costs a few percent everywhere. The lesson is blunt: hand-shaping the hypothesis class into relaxed DNF,
trained from uniform random examples, does *not* beat a model that simply carves the input space by
variable. So I should stop relaxing the logic and use a model that does exact, axis-aligned conjunctive
splits natively — a decision tree — and fix the variance problem that one tree has by averaging many.

Here is the structural fact that makes a tree the natural learner for this task, and it is the same fact
the noisy-OR was fumbling. A decision tree splits on one variable at a time; a root-to-leaf path of
length `w` is a conjunction of `w` literals — *exactly* a DNF term. So the set of leaves labelled 1 is a
DNF, and a sufficiently deep tree can represent the target DNF *exactly*, with no relaxation and no
threshold to tune. A single unrestricted tree, grown until pure, will in fact memorize the 20000
training points perfectly. The catch is the one every tree has: a single deep tree is a high-variance
estimator. Small changes in the training sample move the early splits, and because every later split is
conditioned on the earlier ones, an early perturbation cascades into a wholly different tree. With
20000 uniform examples covering a 30-to-60-dimensional Boolean cube extremely sparsely (2^30 to 2^60
possible inputs, of which I see 20000), the leaves near the bottom are supported by a handful of points
each, and their labels are essentially guesses. So one tree overfits: perfect on train, noisy on a fresh
uniform test point that falls in a sparsely-supported region. The diagnosis from deep_dnf carries over —
I have a *variance* problem, not a representation problem — and the fix is the same shape as averaging
many soft conjunctions, except I average many *exact* trees instead.

The averaging argument I want is precise, so let me make it. Take a predictor that is an average of
trees, each grown on a resampled version of the data, and consider its variance. If the individual trees
were independent and each had variance `σ²`, the average of `B` of them would have variance `σ²/B` —
variance vanishes as I add trees. Real trees grown on bootstrap resamples of the same data are *not*
independent; they share most of the data, so their predictions are positively correlated with some
pairwise correlation `ρ`, and the variance of the average is `ρσ² + (1−ρ)σ²/B`. The second term goes to
zero with more trees, but the first does not — the floor on the ensemble's variance is `ρσ²`, set by how
correlated the trees are. So the whole game in turning a high-variance tree into a low-variance ensemble
is twofold: keep growing trees (drives the `(1−ρ)σ²/B` term down for free) and, more importantly, *drive
down the correlation `ρ`* between trees. Bagging — bootstrap aggregation — gets `ρ` down a little by
giving each tree a different resample (each bootstrap omits ≈ 37% of the data, so trees disagree on the
held-out-of-bag regions). But bootstrapping alone leaves the trees highly correlated, because a few
strong variables dominate the top splits of *every* tree: if one variable is the single most informative
split, nearly every bootstrap tree puts it at the root, and the trees look alike near the top where it
matters most.

The decisive move is to decorrelate the trees by *also* randomizing the splits. At each node, instead of
searching all `n` variables for the best split, restrict the search to a fresh random subset of `m ≪ n`
variables and pick the best split among those. Now no single strong variable can dominate every tree's
root, because in many trees it simply is not among the `m` candidates at the top node; different trees
are forced to build out of different variables, their errors decorrelate, `ρ` drops, and the ensemble
variance floor `ρσ²` drops with it. This is the one idea that separates a random forest from plain
bagged trees, and it is exactly what I need on this task: the target DNF has `s` terms over `w`
variables each, so there are several genuinely informative variables, and feature subsampling spreads
the trees across them so the ensemble sees the whole term structure rather than fixating on the few
variables with the largest marginal signal. Crucially, the variance reduction does *not* come at the
cost of bias the way pruning would: I grow each tree deep (unrestricted depth), so each individual tree
is low-bias / high-variance — it can represent any of the DNF terms — and I rely entirely on the average
to kill the variance. Deep, unpruned, decorrelated, averaged: that is the recipe.

In the task's scaffold this is an almost trivial fill, and that is the point — I do not need a bespoke
network, I need the right off-the-shelf estimator wired into `fit_and_predict`. `build_model` returns an
`sklearn` `RandomForestClassifier` with `n_estimators=200` (enough trees to push the `(1−ρ)σ²/B` term
well below the `ρσ²` floor, so adding more buys little), `max_depth=None` (grow each tree until its
leaves are pure — full depth, because I want each tree low-bias and I am counting on the average for
variance), and `min_samples_leaf=2` (a light touch against single-point leaves, which on this sparse
cube are pure noise; requiring two points per leaf trims the most overfit twigs without imposing real
bias). `n_jobs=-1` parallelizes the 200 trees. `make_dataset` is the default uniform sample — 20000
labelled `(x, f(x))` pairs — and `fit_and_predict` just calls `.fit` on the float-cast Booleans and
`.predict` on the test set, returning the 0/1 vector. The default feature-subset size sklearn uses for
classification is `sqrt(n)` per split — about 5 of 30, 6 of 40, 8 of 60 — which is exactly the
decorrelation regime I argued for: small enough to force different trees apart, large enough that each
node still finds a useful split. The full module is in the answer.

Now the falsifiable expectations against the deep_dnf numbers, family by family. On **monotone**
(deep_dnf 0.9088): the forest should be *competitive but not obviously dominant*. Monotone DNF with 20
terms over 40 variables is a wide target; a width-4 term is a length-4 path, and with feature
subsampling and 200 trees the union of paths should cover most terms, but the high term count (s=20)
means many leaves and a real chance some terms are under-covered by the random splits. I would not be
shocked if the forest lands near or slightly below deep_dnf's monotone number — the warm-start mining
genuinely helps the differentiable model there. On **random** (deep_dnf 0.7605): I expect the forest to
*clearly beat* deep_dnf, because the tree's exact axis-aligned splits do not suffer the noisy-OR's
union-of-errors blowup; mixed polarity is free for a tree (a split tests `x_i = 0` or `x_i = 1`
symmetrically), so the random family should jump well above 0.76 — this is the single comparison that
should vindicate the switch. On **sparse** (deep_dnf 0.8986): the 12-variable junta is the forest's
friend in principle — irrelevant variables are simply never chosen as good splits — but feature
subsampling cuts both ways here: with only 12 of 60 variables relevant, a random subset of ~8 variables
at each node frequently contains *zero* relevant variables, forcing a wasted split on noise, which slows
the trees from concentrating on the junta. So sparse is the one family where the forest's decorrelation
trick could backfire, and I would watch for the sparse number coming in *below* the random number even
though sparse "should" be easier.

So the prediction I am committing to: the geometric mean should rise above deep_dnf's 0.8532, driven
almost entirely by a large gain on the random family, with monotone roughly flat and sparse uncertain.
If instead the forest's *random* number fails to clear deep_dnf's 0.76, my whole diagnosis — that the
noisy-OR's union-of-errors, not the representation, sank deep_dnf — is wrong, and I would have to look
elsewhere. And if the forest, despite exact splits and 200 trees, still leaves any family well short of
the near-perfect accuracy a width-4 concept ought to allow from 20000 examples, that points at the next
lever: a forest *bags independent trees* and never lets a later tree correct an earlier one's mistakes.
The way to push past flat averaging is to make each new tree fit the *residual error* of the ones so
far — to boost rather than bag — and to ask whether a plain gradient-trained MLP, which the differentiable
model abandoned, was actually a stronger generic learner than either tree ensemble all along.
