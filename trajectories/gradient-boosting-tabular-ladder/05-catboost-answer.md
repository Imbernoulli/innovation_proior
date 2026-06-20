**Problem (from step 4).** Every prior rung encodes a categorical feature with an all-data **target
statistic** — replace category c with a smoothed average of y over training rows in c — whose numerator
includes the encoded example's own label yₖ. This leaks the target: the encoded feature's distribution
x̂ | y differs between train and test (a **conditional shift**), so a split on it can be perfect on the
training set and useless at test. The same leak exists one level up in boosting: the step-t gradient for
example k is computed from a model Fᵗ⁻¹ trained *on* example k, biasing the residual — a **prediction
shift** that scales like 1/(n−1) and inflates test loss on categorical data.

**Key idea.** **CatBoost**: remove both leaks with a shared random permutation σ. (1) **Ordered target
statistics** — encode example k using only the examples that *precede* it in σ:
x̂ₖ = (Σ_{j∈Dₖ}[xⱼ=xₖ]yⱼ + a·p)/(Σ_{j∈Dₖ}[xⱼ=xₖ] + a), Dₖ = {j : σ(j) < σ(k)}, with prior p and
smoothing a (test example uses the full training set). yₖ is never in its own encoding, so x̂ | y matches on train
and test. (2) **Ordered boosting** — compute example k's residual from a model trained *only* on the
examples preceding k in σ (it has never seen k), giving an unbiased gradient and killing the prediction
shift. Use the *same* σ for both. Base learners are **oblivious (symmetric) trees** — one split per
level — which regularizes a leakage-sensitive learner and makes prediction a fast indexed lookup.

**Why it works.** Both pathologies are the same structural leak: a quantity for example k is computed
from data that has already seen yₖ. Restricting each example's encoding *and* its residual to a strict
permutation-prefix that excludes it makes both unbiased — the conditional shift in the encoding and the
prediction shift in the gradient both vanish — so the train/test gap the all-data statistics open is
closed. Leave-one-out does *not* fix it (a constant or all-unique categorical still leaks via the
total-sum identity); a *prefix* does. Averaging over several permutations damps the high variance of
early-in-order examples. This is the learner that finally handles heavily categorical tabular data
without leaking the target.

**Change / code.** Ordered boosting (the idealized algorithm that defines the unbiased residual) and the
ordered target statistic. The practical CatBoost keeps O(log n) supporting models instead of n.

```text
Ordered boosting
  input: {(x_k, y_k)}_{k=1..n}, number of iterations I
  sigma <- random permutation of [1..n]
  M_i   <- 0   for i = 1..n
  for t = 1 to I:
      for i = 1 to n:
          r_i <- y_i - M_{sigma(i)-1}(x_i)              # residual from a model that has NOT seen i
      for i = 1 to n:
          dM  <- LearnModel( (x_j, r_j) : sigma(j) <= i )   # fit using only the prefix up to i
          M_i <- M_i + dM
  return M_n
```

```text
Ordered target statistic for a categorical value (uses only the permutation history of each example):
  for a TRAINING example k:   D_k = { j : sigma(j) < sigma(k) }
  for a TEST example:         D_k = all training indices

           sum_{ j in D_k } [ x_j = x_k ] * y_j  +  a * p
  x_hat_k = -------------------------------------------------
              sum_{ j in D_k } [ x_j = x_k ]      +  a

  p = prior (e.g. average target);  a > 0 = smoothing;  same sigma as ordered boosting;
  average over several independent permutations to reduce the variance of early-in-order examples.
```

Greedy (all-data) TS x̂ₖ = (Σⱼ[xⱼ=xₖ]yⱼ + a·p)/(Σⱼ[xⱼ=xₖ] + a) is the leaking baseline this replaces;
the ordered subset Dₖ = {j : σ(j) < σ(k)} is the fix. Base learners are oblivious (symmetric) decision
trees.
