# Conformal Prediction

Conformal prediction converts a heuristic measure of how strange a labeled example looks into a finite-sample prediction set.

Full form. Let `z_i = (x_i,y_i)` and choose a bag-valued, permutation-symmetric nonconformity measure `A(B,z)`, where larger values mean that `z` fits the bag `B` worse. For a new object `x_n` and candidate label `y`, provisionally set `z_n = (x_n,y)`, let `B_n` be the completed bag containing one occurrence of each `z_i`, and compute

`alpha_i = A(B_n \ {one occurrence of z_i}, z_i),  i = 1,...,n`.

Define the candidate-label p-value

`p_y = #{i in {1,...,n}: alpha_i >= alpha_n} / n`.

At significance level `eps`, output

`Gamma^eps(z_1,...,z_{n-1},x_n) = {y: p_y > eps}`.

If the completed examples are exchangeable, then

`P{Y_n notin Gamma^eps(z_1,...,z_{n-1},X_n)} <= eps`.

Split form. Train or choose a score function `s(x,y)` before using a held-out calibration set. For calibration examples, compute `S_i = s(X_i,Y_i)`, `i=1,...,n`. For `0 <= alpha < 1`, set

`k = ceil((n+1)(1-alpha))`

and define `q_hat` as the `k`th smallest element of the bag `{S_1,...,S_n,+infty}`. Equivalently, if `k <= n`, `q_hat = S_(k)` with scores sorted increasingly; if `k = n+1`, `q_hat = +infty`.

For a new input `x`, output

`C(x) = {y: s(x,y) <= q_hat}`.

Under exchangeability of the calibration examples and the future example, after the score is fixed,

`P{Y_new in C(X_new)} >= 1 - alpha`.

The proof is a permutation-rank proof. For the true future label, its score is exchangeable with the old scores. It can occupy an extreme rank only as often as symmetry permits, and the p-value or adjusted order statistic rejects only those extreme ranks. The model or heuristic controls efficiency, not validity: a bad score gives large or unhelpful sets, while a good score gives smaller sets under the same coverage guarantee.

Canonical code correspondence. The retrieved `aangelopoulos/conformal_classification` implementation is a RAPS/APS split-conformal classifier wrapper. Its calibration path computes class probabilities from temperature-scaled logits, uses `giq` to record the cumulative probability mass plus penalties at which the true class enters the set, sets `Qhat = np.quantile(E, 1-alpha, interpolation='higher')`, and deploys `gcq` to return sorted top classes up to the threshold-crossing size computed from cumulative mass plus penalties, with optional randomization and a zero-set convention.
