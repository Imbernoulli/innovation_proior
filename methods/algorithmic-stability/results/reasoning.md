I start from the failure mode of empirical risk. The predictor is chosen by looking at the sample, so the same sample cannot be treated as an independent test set. If I try to solve this by taking a supremum over the whole hypothesis class, I get a valid route, but I also throw away information about how the algorithm actually selects one predictor. A regularized algorithm, for example, may search an enormous space while still being hard to move.

So I focus on the learning map itself. I imagine changing one coordinate of `S` and asking whether the losses of the two resulting predictors are almost the same on every possible test point. If the answer is yes, then no individual observation has much leverage over the final predictor. This feels like the quantity that should matter for overfitting, because the empirical loss becomes too optimistic precisely when the learner can exploit idiosyncrasies of its own sample.

Before I commit to that intuition I want to see the optimism actually appear, on something concrete, so I am not chasing a phantom. Take the simplest stable rule I can write down: a one-dimensional shrunken mean `A(S) = (sum_i y_i)/(m + lambda)` with squared loss clipped at `M`, on fair coin labels `y in {0,1}`, with `m = 5` and `lambda = 3`. I can simulate this. Averaging over many samples I get `E R(A,S) ≈ 0.305` for the true risk but `E R_emp ≈ 0.242` for the empirical risk, a gap of about `+0.063`. So empirical risk really is biased low, and by a definite amount; the problem I am trying to certify is not imaginary. Good — now I know there is a real `E R - E R_emp > 0` to bound.

The first tempting definition is to compare the two output hypotheses directly, but that is too tied to the representation of the hypothesis space. I only need the output through the loss. I therefore measure the worst-case loss change after deleting one point: `sup_z |ell(A(S), z) - ell(A(S \ i), z)|`. If this is at most `beta`, I can compare the full-sample predictor with the leave-one-out predictor on any example. Replacement is not a separate conceptual obstacle, because replacing one point can be decomposed through the common sample with that point deleted: `A(S)` to `A(S \ i)` costs `beta`, and `A(S \ i)` to `A(S^i)` costs another `beta`, so a replacement costs at most `2 beta` by the triangle inequality.

Now I need to see why this sensitivity statement controls the bias I just measured, and I want to do the bookkeeping rather than wave at it. Pick a training point `z_i`. The dangerous term is `ell(A(S), z_i)`, since the model was trained on `z_i`. But `A(S \ i)` was not trained on `z_i`, so by exchangeability the loss of `A(S \ i)` on `z_i` behaves like a fresh test loss for that leave-one-out model. Let me check that this exchangeability claim is the literal truth and not a slogan. In the same simulation I can compute, for each sample, `R_loo(A,S) = (1/m) sum_i ell(A(S \ i), z_i)` and separately the true risk of the deletion predictors `(1/m) sum_i R(A, S \ i)`. The first is `≈ 0.3161`, the second `≈ 0.3166` — equal to within Monte-Carlo noise. So `E R_loo` really does equal `E R(A, S \ i)`: the leave-one-out loss is an unbiased estimate of the `(m-1)`-sample risk. That is the swap I need. Stability then lets me replace `A(S \ i)` by `A(S)` on every coordinate while paying `beta` per coordinate, which is how the `E R - E R_emp` gap gets pinned to order `beta`.

Expectation is not enough. I need the gap to concentrate for a random sample. I view the gap `Phi(S) = R(A,S) - R_emp(A,S)` as a function of the independent coordinates of `S`, and I ask how much `Phi` can move when I change a single coordinate `z_k` to `z_k'`. This is the McDiarmid bounded-differences quantity, so I should compute the per-coordinate constant honestly term by term rather than declare it.

The true-risk part `R(A,S) = E_z ell(A(S),z)` only sees `z_k` through the predictor. Changing `z_k` is a replacement, so `|R(A,S) - R(A,S^k)| <= sup_z |ell(A(S),z) - ell(A(S^k),z)| <= 2 beta`.

The empirical part `R_emp(A,S) = (1/m) sum_i ell(A(S),z_i)` splits into two kinds of summand. The `m-1` summands with `i != k` only move because the predictor moves, `A(S) -> A(S^k)`, each by at most `2 beta`; with weight `1/m` they contribute at most `(m-1)/m · 2 beta`. The single summand `i = k` is special: both the predictor moves (`<= 2 beta`) and the evaluation point itself changes, `ell(A(S),z_k)` versus `ell(A(S^k),z_k')`, where the point change alone can swing the bounded loss by up to `M`; with weight `1/m` it contributes at most `(2 beta + M)/m`.

Adding the three pieces:

```
c_k <= 2 beta            (true risk)
     + (m-1)/m · 2 beta  (m-1 unchanged empirical summands)
     + (2 beta + M)/m    (the changed empirical summand)
```

Collecting the `beta` coefficient: `2 + 2(m-1)/m + 2/m = 2 + (2m - 2 + 2)/m = 2 + 2 = 4`. The `M` coefficient is `1/m`. So `c_k <= 4 beta + M/m`, the same for every coordinate. (I confirmed this collapse symbolically; the cross terms in `(m-1)/m` and `1/m` really do cancel to a clean `4 beta`.)

Now McDiarmid: with `m` coordinates each of width `c_k = 4 beta + M/m`,

```
P(Phi - E Phi >= t) <= exp(-2 t^2 / (m c_k^2)).
```

Setting the right side to `delta` and solving, `t = c_k sqrt(m log(1/delta) / 2) = (4 beta + M/m) sqrt(m log(1/delta)/2)`. Pulling the `1/m` out of the bracket, `(4 beta + M/m) = (4 m beta + M)/m`, and `sqrt(m)/m = 1/sqrt(m)`, so

```
t = (4 m beta + M) sqrt(log(1/delta) / (2m)).
```

I want to be sure I did not drop a factor in that rearrangement, so I evaluate both forms numerically: for `m = 100`, `beta = 0.01`, `M = 1`, `delta = 0.05`, the direct expression `sqrt(m c_k^2 log(1/delta)/2)` gives `0.61194`, and the closed form `(4 m beta + M) sqrt(log(1/delta)/(2m))` also gives `0.61194`. They agree across the `m` and `beta` values I tried, so the algebra is right.

This assembles into the method: prove a uniform loss-sensitivity bound `beta` for the algorithm, plug it into the concentration result, and read off

```
R(A,S) <= R_emp(A,S) + 2 beta + (4 m beta + M) sqrt(log(1/delta)/(2m))
```

with probability `1 - delta`, and the leave-one-out variant with `R_loo` and a `beta` (not `2 beta`) bias term, since the leave-one-out estimate already does the deletion swap on the page. The shape now makes sense from the derivation rather than by stipulation: a direct stability term, because the expectation is biased by sample reuse, and a sampling-fluctuation term carrying exactly the bounded-difference constants I computed. If `beta` scales like `1/m`, then `m beta` is constant, the fluctuation term decays like `1/sqrt(m)`, the `2 beta` term decays like `1/m`, and the whole bound vanishes with growing sample size.

I then ask which algorithms can actually satisfy such a strong condition. Unregularized empirical-risk minimization can jump between nearly tied minimizers, so a single point may flip the selected hypothesis, and then `beta` need not be small at all. A regularized objective is different. Removing one loss term perturbs the average loss by one part in `m`, and the penalty supplies curvature or a norm barrier that limits how far the minimizer can move. In an RKHS, bounded kernels and an admissible loss let me turn a movement bound in function norm into a uniform loss-change bound, giving `beta` on the order of `1/(lambda m)` — concretely `beta <= sigma^2 kappa^2 / (2 lambda m)` for `sigma`-admissible loss and `k(x,x) <= kappa^2`. I should sanity-check that this is consistent with the optimism I measured: larger `lambda` should mean smaller `beta` and a smaller true-vs-empirical gap. The bound is monotone the right way — `beta` falls as either `lambda` or `m` grows — which lines up with the simulation, where the shrinkage toward a fixed predictor was exactly what kept the rule from chasing its own sample.

That also prevents me from overstating the result. A predictor can be stable and still bad if the regularizer forces it toward a poor function: in my toy example, pushing `lambda` very large shrinks the predictor toward `0` and makes `beta` tiny, but the true risk stops improving and can get worse. So the certificate controls overfitting, not approximation error. The final picture is a tradeoff that I can now read off both the bound and the experiment: larger `lambda` makes the algorithm less sensitive to one example and shrinks the stability terms, while smaller `lambda` can fit the sample more closely but weakens the sensitivity certificate.

What I am left with is a local question standing in for a global one. Instead of bounding the worst gap over every function in a class, I bound how much one observation can move this particular learning rule; generalization follows because evaluating on the training sample is then almost like evaluating on points that were not responsible for the fitted predictor — and the numbers above show that "almost" is quantitatively the `beta` I have to prove.
