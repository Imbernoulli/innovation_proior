The fast histogram learners are excellent on numeric data — LightGBM reaches $130.094$ s/iter on Higgs at $0.845724$ AUC and lifts MS-LTR NDCG@10 to $0.524252$ — but they have all ignored the feature type that dominates real tabular problems: categorical identifiers like an employee ID, a resource code, a city, with thousands of distinct values and no natural order. A tree cannot split on "employee #4471 $<$ threshold"; the value is not a number. The standard move is to turn each category into a number by a *target statistic* (TS): replace category $c$ with an estimate of $E[y\mid c]$, smoothed toward a prior $p$ so rare categories do not get wild estimates,

$$\hat x_k = \frac{\sum_j [x_j=x_k]\,y_j + a\,p}{\sum_j [x_j=x_k] + a},$$

with smoothing $a>0$. On the surface this is reasonable — high-target categories get high encodings, the tree can split on them — and it is what every learner so far would do. But stare at the numerator: the sum runs over *all* training examples, **including example $k$ itself**, so $\hat x_k$ is computed using $y_k$, the very label I am trying to predict from $x_k$. The feature value I hand the model for example $k$ already contains a leak of its own answer. In the limit where every category is unique — every example has its own ID — the only training example sharing $k$'s category is $k$, so $\hat x_k=(y_k+a p)/(1+a)$, a strictly increasing function of $y_k$, and a single split at $t=(0.5+a p)/(1+a)$ separates the training labels perfectly. At test time the category was never seen, the TS falls back to $p$ for every test example, the split does nothing, and accuracy collapses to chance. The encoding *manufactured* a feature predictive only on the training set. Precisely: the distribution $\hat x\mid y$ differs between train and test, because the training encoding saw the training label and the test encoding did not — a **conditional shift**. The property I actually want is $E[\hat x\mid y=v]$ matching on train and test, which the all-data TS violates by construction.

I propose **CatBoost**, which removes this leak — and a second one hiding in the boosting itself — with a shared random permutation. The first instinct, leave-one-out (compute $\hat x_k$ from all examples except $k$), removes $y_k$ from $k$'s own numerator but does *not* fix the shift: for a constant categorical feature, the total-sum identity makes the leave-one-out TS an exact decreasing-in-$y_k$ function again, and a threshold split on it still separates the classes on the training set and fails at test. The leak is more structural than "do not use your own row." The structural cure is to give each example an encoding computed from data that is genuinely *held out relative to that example* — data that could not have peeked at $y_k$. I manufacture "held-out relative to $k$" from one dataset by imposing an *artificial time order*: draw a random permutation $\sigma$, pretend the examples arrived in that order, and encode example $k$ using only the examples that came **before** it. This is the **ordered target statistic**: for a training example $D_k=\{j:\sigma(j)<\sigma(k)\}$, for a test example $D_k$ is the entire training set, and

$$\hat x_k = \frac{\sum_{j\in D_k}[x_j=x_k]\,y_j + a\,p}{\sum_{j\in D_k}[x_j=x_k] + a},\qquad D_k=\{j:\sigma(j)<\sigma(k)\}.$$

Now $y_k$ is never in $k$'s own encoding, and neither is any example that comes after $k$, so there is no path by which $k$'s label leaks back into its feature; the conditional shift is gone because every encoding is built from a strict prefix that never includes the encoded example. Early examples in $\sigma$ have tiny histories and high-variance encodings, which I damp by averaging over several independent permutations so no single example is unlucky.

That fixes the features, but the same leak lives in the *boosting*, and noticing it is what makes this more than an encoding trick. In ordinary gradient boosting the step-$t$ gradient for example $k$ is the loss gradient evaluated at $F^{t-1}(x_k)$ — but $F^{t-1}$ was trained on a dataset that *includes* $k$, so the gradient for $k$ comes from a model that has already seen $k$'s label. The prediction $F^{t-1}(x_k)\mid x_k$ on a training point is shifted from $F^{t-1}(x)\mid x$ on a test point, and this propagates: shifted gradient $\to$ base learner biased from the true expected-gradient solution $\to$ a **prediction shift** in the final model $\to$ overfitting. The bias is quantifiable, scaling like $1/(n-1)$ — the signature of a leakage bias the histogram learners never corrected because they reuse the same data to compute gradients and to fit on them. The cure is the same permutation idea applied to the residuals: impose the order $\sigma$ and, for example $k$, compute its residual from a model trained **only on the examples preceding $k$**, a model that has never seen $k$. In the idealized **ordered boosting** I maintain, for each position $i$, a supporting model $M_i$ trained on the first $i$ examples, and the residual for $k$ uses $M_{\sigma(k)-1}$, which excludes $k$ — so the residual is an unbiased estimate of the true gradient at $x_k$, the base learner is unbiased, and the prediction shift vanishes. The idealized form keeps $n$ separate models and is $n\times$ too expensive; the practical version maintains an $O(\log n)$ set of supporting models (one per prefix length that doubles), all sharing one tree structure per step, recovering unbiasedness at tractable cost. The two permutation tricks use the *same* $\sigma$, so an example's label is never used either to encode its own categorical features or to compute its own gradient.

One design choice pairs naturally: the base learners are **oblivious (symmetric) decision trees**, where the same split (feature and threshold) is used across an entire level of the tree. Oblivious trees are weaker per-tree, which is a *good* regularizer for a leakage-sensitive learner, and they make prediction a fast indexed lookup, which matters because ordered boosting already pays for the supporting models. Removing *both* leaks — ordered target statistics for the encoding and ordered boosting for the residuals, sharing one permutation — eliminates the conditional shift that inflated the histogram learners' test loss, and so handles heavily categorical tabular data without leaking the target, which is where the ladder ends.

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
