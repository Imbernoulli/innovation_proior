The fast histogram learners are excellent on numeric data, but I've been ignoring the one feature type
that dominates real tabular problems: categorical identifiers — an employee ID, a resource code, a
city — with thousands of distinct values and no natural order. A tree can't split on "employee #4471 <
threshold"; the value isn't a number. The standard move is to turn each category into a number by a
*target statistic* (TS): replace the category with an estimate of E[y | category]. For a category c the
obvious estimate is the average label over the training examples sharing that category, smoothed toward
a prior p so rare categories don't get wild estimates:

  x̂ₖ = ( Σⱼ [xⱼ = xₖ]·yⱼ + a·p ) / ( Σⱼ [xⱼ = xₖ] + a ),

with smoothing a > 0. This is what every learner so far would do, and on the surface it's reasonable —
high-target categories get high encodings, the tree can split on them.

But stare at the numerator: the sum Σⱼ [xⱼ = xₖ]·yⱼ runs over *all* training examples, **including
example k itself**. So x̂ₖ is computed using yₖ — the very label I'm trying to predict from xₖ. The
feature value I hand the model for example k already contains a leak of example k's own answer. See how
badly this bites in the limit: suppose a categorical feature where every category is unique (every
example has its own ID). Then for example k the only training example sharing its category is k itself,
so x̂ₖ = (yₖ + a·p)/(1 + a) — a strictly increasing function of yₖ. A single split at the threshold
t = (0.5 + a·p)/(1 + a) perfectly separates the training labels. The model looks perfect on the training
data. At test time the category was never seen, the TS falls back to the prior p for every test example,
the split does nothing, and accuracy collapses to chance. The encoding *manufactured* a feature that is
predictive only on the training set.

That is the precise failure: the distribution of x̂ conditioned on the label, x̂ | y, is *different* for
a training example than for a test example, because the training encoding saw the training label and the
test encoding did not. Call it a **conditional shift** in the encoded feature. The property I actually
want is that the encoding's distribution given y is the *same* on train and test —
E[x̂ | y = v] should match between a training example and a test example with the same y. The all-data TS
violates it by construction.

The instinct is leave-one-out: compute x̂ₖ from all examples *except k*. That removes yₖ from k's own
numerator — but it doesn't fix the shift. Consider a constant categorical feature (every example in the
same single category). Leave-one-out TS gives x̂ₖ = (Σⱼ≠ₖ yⱼ + a·p)/(n − 1 + a), and the total-sum
identity makes this an exact decreasing-in-yₖ function again — a threshold split on it still separates
the classes on the training set and still fails at test. Leave-one-out is not enough; the leak is more
structural than "don't use your own row."

What's the structural cure? I need each example's encoding to be computed from data that, relative to
that example, is genuinely *held out* — data that could not have peeked at yₖ in any way that correlates
with xₖ. The clean way to manufacture "held-out relative to k" out of one dataset is to impose an
*artificial time order*: draw a random permutation σ of the examples and pretend they arrived in that
order. Then encode example k using only the examples that came **before** it in σ — its "history." For a
training example k, the encoding subset is Dₖ = {j : σ(j) < σ(k)}, and the ordered target statistic is

  x̂ₖ = ( Σ_{j ∈ Dₖ} [xⱼ = xₖ]·yⱼ + a·p ) / ( Σ_{j ∈ Dₖ} [xⱼ = xₖ] + a ),   Dₖ = {j : σ(j) < σ(k)}.

For a test example the history is the entire training set. Now yₖ is *never* in k's own encoding, and —
crucially — neither is any example that "comes after" k, so there's no path by which the label of k
leaks back into its feature. The conditional shift is gone: x̂ | y has the same distribution on train and
test, because every encoding is built from a strict prefix that never includes the encoded example.
There's a cost — early examples in σ have tiny histories, so their TS is high-variance — but I can
average over several independent permutations to damp that variance, so no single example is unlucky.

That fixes the *features*. But the exact same leak lives in the *boosting* itself, and noticing it is
what makes this more than an encoding trick. In ordinary gradient boosting, the gradient I fit at step t
for example k is gᵗ(xₖ, yₖ) = the loss gradient evaluated at the current model's prediction Fᵗ⁻¹(xₖ).
But Fᵗ⁻¹ was trained on a dataset that *includes example k*. So the gradient estimate for k is computed
from a model that has already seen k's label — the same structural leak, one level up. The model's
prediction Fᵗ⁻¹(xₖ) | xₖ on a training point is shifted from Fᵗ⁻¹(x) | x on a test point, because the
training point participated in fitting Fᵗ⁻¹. This propagates: the shifted gradient → a base learner
biased away from the true expected-gradient solution → a **prediction shift** in the final model →
overfitting. And the bias is real and quantifiable: on a small example it scales like 1/(n−1), inversely
with dataset size, exactly the signature of a leakage bias that the histogram learners never corrected
because they reuse the same data to compute gradients and to fit on them.

The cure is the same permutation idea applied to the residuals. Impose a random order σ. For example k,
compute its gradient/residual using a model that was trained **only on the examples preceding k** in σ —
a model that has never seen k. In the idealized version I maintain, for each position i, a separate
supporting model Mᵢ trained on the first i examples; the residual for k uses M_{σ(k)−1}, which excludes
k. Then the residual is an unbiased estimate of the true gradient at xₖ, the base learner is unbiased,
and the prediction shift vanishes:

```text
Algorithm — Ordered boosting (idealized):
  input: {(x_k, y_k)}_{k=1..n}, number of iterations I
  sigma  <- random permutation of [1..n]
  M_i    <- 0   for i = 1..n
  for t = 1 to I:
      for i = 1 to n:
          r_i  <- y_i - M_{sigma(i)-1}(x_i)     # residual from a model that has NOT seen example i
      for i = 1 to n:
          dM   <- LearnModel( (x_j, r_j) : sigma(j) <= i )   # fit using only the prefix up to i
          M_i  <- M_i + dM
  return M_n
```

The idealized form keeps n separate models and is n× too expensive, so the practical version maintains a
logarithmic set of supporting models (a model for each prefix length that doubles), all sharing one tree
structure per step, which recovers the unbiasedness at tractable cost. And to make the two permutation
tricks consistent, I use the *same* permutation σ for the ordered TS and for ordered boosting, so an
example's label is never used either to encode its own categorical features or to compute its own
gradient. One more design choice pairs naturally: the base learners are **oblivious (symmetric)
decision trees** — the same split (feature + threshold) is used across an entire level of the tree.
Oblivious trees are weaker per-tree, which is a *good* regularizer for a leakage-sensitive learner, and
they make prediction a fast indexed lookup, which matters because ordered boosting already pays for the
supporting models.

This is the closing move, so let me set the bar and why I believe it clears it. The fast histogram
learners — the regularized one and the sampling/bundling one — both encode categoricals with the
all-data target statistic and both compute gradients on the same data they fit, so on a heavily
*categorical* dataset they should overfit to the leaked encoding and the prediction shift. The natural
test is the Amazon Employee Access data, whose features are almost entirely high-cardinality categorical
IDs. The standing numbers to beat are those two learners' test log-loss on Amazon — XGBoost at 0.1633
and LightGBM at 0.1636 (lower is better). The bet is that removing *both* leaks — ordered target
statistics for the categorical encoding and ordered boosting for the residuals, sharing one permutation —
eliminates the conditional shift that inflates their test loss, and so drives the Amazon log-loss
distinctly below 0.1636. If ordering the encoding and ordering the boosting closes the train/test gap
that the all-data statistics open, this is the learner that finally handles categorical tabular data
without leaking the target, and that is where the ladder ends.
