The fast histogram learners are excellent on numeric data — LightGBM reached 130.094 s/iter on Higgs at
0.845724 AUC and lifted MS-LTR NDCG@10 to 0.524252 — but every one of them has ignored the feature type
that dominates real tabular problems: categorical identifiers, an employee ID, a resource code, a city,
with thousands of distinct values and no natural order. A tree can't split on "employee #4471 <
threshold"; the value isn't a number, and there is no meaningful order to threshold. The standard move is
to turn each category into a number by a *target statistic* (TS): replace the category with an estimate
of E[y | category]. For a category c the obvious estimate is the average label over the training examples
sharing that category, smoothed toward a prior p so rare categories don't get wild estimates from one or
two rows:

  x̂ₖ = ( Σⱼ [xⱼ = xₖ]·yⱼ + a·p ) / ( Σⱼ [xⱼ = xₖ] + a ),

with smoothing a > 0. This is what every learner so far would do, and on the surface it's reasonable —
high-target categories get high encodings, low-target categories get low ones, and the tree can split on
the resulting number as if it were any other feature.

But stare at the numerator: the sum Σⱼ [xⱼ = xₖ]·yⱼ runs over *all* training examples, **including
example k itself**. So x̂ₖ is computed using yₖ — the very label I'm trying to predict from xₖ. The
feature value I hand the model for example k already contains a leak of example k's own answer. Let me
see exactly how badly this bites in the limit, with numbers, because a limit case I can compute is worth
more than the worry. Suppose a categorical feature where every category is unique — every example has its
own ID, which is the honest description of an employee-ID column. Then for example k the only training
example sharing its category is k itself, so the sum collapses to yₖ and x̂ₖ = (yₖ + a·p)/(1 + a). Take
binary labels y ∈ {0,1}, prior p = 0.5, smoothing a = 1: a positive example (yₖ=1) gets (1+0.5)/2 = 0.75,
a negative one (yₖ=0) gets (0+0.5)/2 = 0.25, and a single split at the midpoint t = (0.5 + a·p)/(1 + a) =
(0.5+0.5)/2 = 0.5 sends every positive one way and every negative the other. The model is *perfect* on
the training data — train AUC 1.0, train log-loss driving to zero. At test time the category was never
seen, so the TS falls back to the prior: every test example gets x̂ = p = 0.5, the split sends them all to
the same side, and the model predicts one constant for everyone — test AUC 0.5, pure chance. The encoding
*manufactured* a feature that is predictive only on the training set and carries literally no signal at
test.

That is the precise failure, and it is worth naming because the cure has to target it exactly: the
distribution of x̂ conditioned on the label, x̂ | y, is *different* for a training example than for a test
example, because the training encoding saw the training label and the test encoding did not. Call it a
**conditional shift** in the encoded feature. The property I actually want is that the encoding's
distribution given y is the *same* on train and test — E[x̂ | y = v] should match between a training
example and a test example that share the same y. The all-data TS violates this by construction, and the
violation is not a small-sample artifact I can smooth away: it is structural, present whenever the
numerator can see yₖ.

The first instinct is leave-one-out: compute x̂ₖ from all examples *except k*. That removes yₖ from k's
own numerator — but it doesn't fix the shift, and I can prove that to myself on the opposite extreme from
the all-unique case. Consider a *constant* categorical feature, every example in the same single
category. Leave-one-out TS gives x̂ₖ = (Σⱼ≠ₖ yⱼ + a·p)/(n − 1 + a). Now use the total-sum identity
Σⱼ≠ₖ yⱼ = S − yₖ, where S = Σⱼ yⱼ is the fixed total count of positives:

  x̂ₖ = ( S − yₖ + a·p ) / ( n − 1 + a ),

which is an *exactly* affine, strictly *decreasing* function of yₖ — slope −1/(n−1+a). Put numbers on it:
n = 101 examples, S = 50 positives, a = 1, p = 0.5. A positive example (yₖ=1) gets (50−1+0.5)/101 =
49.5/101 = 0.4901; a negative one (yₖ=0) gets (50+0.5)/101 = 50.5/101 = 0.5000. The gap is only 0.0099,
but it is a *clean* gap: every positive sits below 0.495 and every negative above it, so a single
threshold at 0.495 separates all 101 training rows perfectly — on a feature that is *constant*, the same
category for everyone. At test the constant category's encoding is (S + a·p)/(n + a) = 50.5/102 = 0.495
for every test row, right on the threshold, so it separates nothing and collapses to chance again. The
gap shrinks like 1/(n−1) as the data grows, but it never reaches zero and its sign is always aligned with
the label, so leave-one-out has not removed the leak — it has only made it small and quiet. The leak is
more structural than "don't use your own row."

Before I reach for anything elaborate, let me make sure a simpler fix doesn't already exist, because I'd
rather not build machinery if a fold or a hash would do. Three tempting alternatives, each rejected for a
concrete reason. I could hold out a separate fold to compute the encoding — encode each training row's
categoricals from a disjoint estimation fold — but a single held-out fold either wastes a chunk of data
on estimation only, or, if I rotate folds, still leaks within each fold and needs many of them; either
way each category is now averaged over fewer rows, so the encoding gets noisier exactly where it was
already the weakest. I could add noise to the encoding, jittering x̂ₖ so a split can't exploit the leaked
label precisely — but that attacks the symptom, not the disease: the conditional shift E[x̂ | y] is still
there in expectation, the noise only raises its variance, and I'd be trading a bias I can remove for a
variance I'm adding. Or I could abandon target statistics and one-hot the categories — but a column with
thousands of distinct IDs one-hots into thousands of near-empty binary features, which explodes the very
(#features) factor the feature-bundling work just shrank and gives the histogram almost nothing to bin;
hashing the IDs into a smaller space collides unrelated categories and destroys the high-cardinality
signal that made the ID worth keeping. None of the shortcuts both keep the data and remove the shift.
That is what pushes me toward ordering: I want a genuine holdout for every single example while still
using every row to help encode every other row.

What's the structural cure? I need each example's encoding to be computed from data that, relative to
that example, is genuinely *held out* — data that could not have peeked at yₖ in any way that correlates
with xₖ. The clean way to manufacture "held-out relative to k" out of a single dataset is to impose an
*artificial time order*: draw a random permutation σ of the examples and pretend they arrived in that
order. Then encode example k using only the examples that came **before** it in σ — its "history." For a
training example k, the encoding subset is Dₖ = {j : σ(j) < σ(k)}, and the ordered target statistic is

  x̂ₖ = ( Σ_{j ∈ Dₖ} [xⱼ = xₖ]·yⱼ + a·p ) / ( Σ_{j ∈ Dₖ} [xⱼ = xₖ] + a ),   Dₖ = {j : σ(j) < σ(k)}.

For a test example the history is the entire training set. Now yₖ is *never* in k's own encoding, and —
crucially — neither is any example that "comes after" k, so there's no path by which the label of k
leaks back into its own feature. The conditional shift is gone: x̂ | y has the same distribution on train
and test, because every encoding is built from a strict prefix that never includes the encoded example.
There's a cost, and I should be honest about its size: an example early in σ has a tiny history — a
category appearing at permutation position i has on the order of (i−1)·(category frequency) prior
occurrences to average over, which for small i is a handful or zero, so its TS is high-variance and
noisy. The fix is to average over several independent permutations: draw s of them, compute the ordered
TS under each, and average. Independent draws make the variance of the mean fall like 1/s, so a small
handful of permutations tames the worst-case unlucky early example without ever letting any single
permutation's prefix see the future.

Let me confirm the cure actually kills the case that broke the baseline, on the very same all-unique
feature, because a fix that doesn't act on the pathological limit isn't a fix. Under the ordered TS,
example k's history Dₖ contains no example sharing its category — the IDs are unique, so no prior row
matches xₖ, and k's own row is excluded by construction. The numerator is then just a·p and the
denominator a, so x̂ₖ = (a·p)/a = p for *every* training example, independent of yₖ. That is exactly the
value every test example gets, since an unseen category also falls back to p. Train and test now agree,
the manufactured split is gone, and the feature correctly carries no signal — which is the truth, because
a purely random ID genuinely predicts nothing. Where the all-data TS turned a useless feature into a
perfect-on-train, chance-on-test trap (0.75 versus 0.25 on the training rows, a flat 0.5 at test), the
ordered TS returns the honest constant p on both. The diagnosis — a conditional shift because the
encoding saw yₖ — and the cure — a strict prefix that cannot — line up on the numbers, which is the check
I wanted before trusting the construction on the real data.

The all-unique case shows the ordered TS *refusing* to invent signal; the opposite case — a category that
genuinely repeats — shows it still *recovers* real structure without reopening the leak. Take five
training rows sharing one category c, ordered by σ with labels (1, 0, 1, 1, 0), prior p = 0.5, smoothing
a = 1. The first has an empty history and gets the bare prior 0.5; the successive rows see histories
(1), (1,0), (1,0,1), (1,0,1,1) and get 0.75, 0.5, 0.625, 0.7. No row's number used its own label — the
fourth is labelled 1, but its 0.625 was built from the three rows before it — so a split cannot read a
row's answer off its own feature, yet the later rows are already settling toward the category's true
positive rate 3/5 = 0.6. The early-row noise (the first row marooned at the prior, the second at a
one-example 0.75) is exactly what averaging over independent permutations smooths: reorder, and the row
that was first now sits mid-order with a genuine history. The all-unique case gave the honest constant p,
the repeated case gives the honest rate; neither used yₖ, so I trust the ordered TS on the real
categorical mix that lies between them.

That fixes the *features*. But the exact same leak lives in the *boosting* itself, and noticing it is
what makes this more than an encoding trick — it is the same disease one level up. In ordinary gradient
boosting, the gradient I fit at step t for example k is the loss gradient evaluated at the current
model's prediction Fᵗ⁻¹(xₖ). But Fᵗ⁻¹ was trained on a dataset that *includes example k*. So the
gradient estimate for k is computed from a model that has already seen k's label — the same structural
leak as the TS, transposed from the feature encoding onto the residual. The prediction Fᵗ⁻¹(xₖ) | xₖ on a
training point is shifted from Fᵗ⁻¹(x) | x on a test point, because the training point participated in
fitting Fᵗ⁻¹, and this propagates: the shifted gradient produces a base learner biased away from the true
expected-gradient solution, which becomes a **prediction shift** in the final model, which is
overfitting. Let me estimate the size of the bias, because "there is a leak" is not enough — I want to
know whether it is 1% or a rounding error. Take the simplest base learner, a single constant fit to the
residuals r₁,…,rₙ by least squares: its value is the mean r̄ = (1/n)Σⱼ rⱼ, which includes example k's own
residual rₖ with weight 1/n. Ordinary boosting then measures k's next residual against this same model,
so k's residual is pulled toward zero by that 1/n self-contribution — the model already "knows" a 1/n
slice of k's own target. Against a model fit on the *other* n−1 examples (which is what a held-out
estimate uses), the difference is a per-example bias of order 1/(n−1) in the residual, and — this is why
it matters — its sign is the *same* every round (a model always over-explains its own training points),
so it does not average out across boosting iterations, it accumulates into a genuine train/test gap. A
deeper tree concentrates more of k's label into k's own leaf, so the true constant is larger than 1/n for
real trees, but the scaling ~1/(n−1) and the systematic sign are the robust features, and they are
exactly what the histogram learners never corrected, because every one of them computes gradients on, and
fits trees to, the *same* rows.

The cure is the same permutation idea applied to the residuals. Impose a random order σ. For example k,
compute its gradient/residual using a model that was trained **only on the examples preceding k** in σ —
a model that has never seen k. In the idealized version I maintain, for each position i, a separate
supporting model Mᵢ trained on the first i examples; the residual for k uses M_{σ(k)−1}, which excludes
k. Then the residual is an unbiased estimate of the true gradient at xₖ, the base learner is unbiased,
and the prediction shift vanishes. (The idealized ordered-boosting loop is written out in the answer.)

The idealized form keeps n separate models and is n× too expensive — on a dataset the size of the
categorical benchmark that is thousands of full models per boosting run, plainly unusable. But I don't
need a distinct model for every prefix length; I need, for each example, *some* model trained on a prefix
that excludes it, and a prefix that is a bit shorter than σ(k)−1 still excludes k. So maintain supporting
models only at prefix lengths that double — 1, 2, 4, 8, …, up to n — and let example k read the largest
such prefix that still ends before it. That is ⌈log₂ n⌉ models instead of n. On a categorical benchmark
of roughly 30,000 training rows, ⌈log₂ 30000⌉ ≈ 15 supporting models rather than 30,000 — a reduction of
three orders of magnitude in the overhead, turning an n× blowup into a log-n one, and the residual each
example reads is still computed from data that never saw it. To make the two permutation tricks
consistent, I use the *same* permutation σ for the ordered TS and for ordered boosting, so an example's
label is never used either to encode its own categorical features or to compute its own gradient — one
order, both leaks closed. And the sharing has to be exact, not merely tidy. If I boosted under one order
σ′ and encoded categoricals under a different σ, a supporting model Mᵢ (trained on σ′-prefix examples)
would be fed input features x̂ⱼ that were computed from σ-prefixes, and those σ-prefixes need not sit
inside Mᵢ's own σ′-prefix — so a label from outside what Mᵢ is allowed to see could re-enter through an
encoded feature and quietly reopen the leak I just sealed. Under a single σ the prefixes *nest*: every
example j that Mᵢ trains on has σ(j) ≤ i, and its encoding used only Dⱼ = {σ < σ(j)} ⊆ that same prefix,
so every label that ever touches Mᵢ's inputs already lies inside the prefix Mᵢ is permitted to see. One
order makes the containment transitive; two orders break it, which is why the same σ drives both the
statistic and the boosting.

One more design choice pairs naturally with all this. The base learners are **oblivious (symmetric)
decision trees**: the same split (feature + threshold) is used across an entire level of the tree, so a
depth-d tree asks only d distinct questions and its 2^d leaves are indexed by the d-bit vector of
pass/fail answers. Prediction is then d comparisons packed into a d-bit integer that indexes a table of
2^d leaf values — a branch-free lookup, which matters a great deal here because ordered boosting is
already paying for a logarithmic set of supporting models and I have to evaluate them cheaply. Trace one:
a depth-3 oblivious tree whose three questions are (A ≤ 2?), (B ≤ 5?), (C = "x"?) turns a row that passes
the first, fails the second, and passes the third into the bit pattern 101₂ = 5, and its prediction is
the leaf value w₅ read straight out of an eight-entry array — no node-by-node descent, no pointer
chasing, and the *same* three comparisons applied to every row in a batch, which vectorizes cleanly. A
free depth-3 tree, by contrast, could ask a different question at each of its seven internal nodes and
would be walked one branch at a time. The oblivious form gives up that flexibility on purpose and buys
back both the regularization and the lookup speed. And the
capacity cost of obliviousness is a *feature*, not a bug, for this learner: a depth-6 oblivious tree has
only 6 split questions where a free depth-6 tree could have up to 63 different (feature, threshold)
pairs, so the oblivious tree is markedly weaker per tree — which is exactly the regularization a
leakage-sensitive model wants, keeping any residual overfitting in check while the ordering removes the
systematic bias.

The fast histogram learners — the regularized second-order one and the sampling/bundling one — both
encode categoricals with the all-data target statistic and both compute gradients on the same data they
fit, so on a heavily *categorical* dataset they should suffer both pathologies at once: the conditional
shift in the encoding and the prediction shift in the residual. The natural test is the Amazon Employee
Access data, whose features are almost entirely high-cardinality categorical IDs — the all-unique regime
where the leak is most violent. Running both prior learners on Amazon with their default all-data target
statistic, their test log-losses land essentially on top of each other, separated by a hair — far tighter
than the gaps their split engines opened on the numeric benchmarks. That near-tie is the tell: on Higgs
and MS LTR the second-order histogram, the GOSS/EFB sampling, and the leaf-wise growth all moved the AUC
and NDCG, but on Amazon they collapse onto each other, which says the split engine is *not* the binding
constraint here — the shared all-data categorical encoding is, and both are stuck at the same place
because they share the same leak. So the bet is that removing *both* leaks — ordered target statistics for
the encoding and ordered boosting for the residuals, sharing one permutation — eliminates the conditional
shift, closes the train/test gap the all-data statistics open, and drives the Amazon log-loss distinctly
below both baselines rather than tying them. That is the falsifiable claim. This is **CatBoost** — ordered
target statistics, ordered boosting, oblivious trees — the learner that finally handles categorical
tabular data without leaking the target.
