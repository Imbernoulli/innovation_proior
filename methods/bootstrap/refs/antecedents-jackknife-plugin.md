# Antecedents (captured this run)

## The Quenouille–Tukey jackknife
- Quenouille (1949, "Approximate tests of correlation in time-series"; 1956 "Notes on bias in
  estimation", Biometrika 43:353–360): leave-one-out recomputation to REDUCE BIAS of an
  estimator from O(1/n) to O(1/n^2).
- Tukey (1958, abstract, Ann. Math. Statist. 29:614): named it "jackknife"; proposed using the
  spread of the **pseudo-values** to ESTIMATE VARIANCE and build approximate-t confidence
  intervals; conjectured pseudo-values behave ~ iid.
- Miller, R.G. (1974a) "The jackknife — a review", Biometrika 61:1–15: the standard review
  Efron cites [14]. Documents where the jackknife works and where it FAILS (notably the
  sample median).

### Formulas (leave-one-out)
- theta_hat = t(x_1,...,x_n); theta_hat_(-i) = t with x_i deleted.
- pseudo-value: theta_tilde_i = n*theta_hat - (n-1)*theta_hat_(-i).
- jackknife estimate: theta_jack = mean_i theta_tilde_i.
- implicit bias estimate: (n-1)*(theta_bar_(-) - theta_hat), theta_bar_(-) = mean_i theta_hat_(-i).
- jackknife variance: v_jack = (n-1)/n * sum_i (theta_hat_(-i) - theta_bar_(-))^2
  = (1/n)*(1/(n-1))*sum_i (theta_tilde_i - theta_jack)^2.

## Jaeckel's infinitesimal jackknife
- Jaeckel, L. (1972) "The infinitesimal jackknife", Bell Labs Memo MM 72-1215-11. Treats the
  statistic as a function of the cell-weight vector P = (P_1,...,P_n) (P_i = mass on x_i),
  differentiates at P = (1/n,...,1/n). The directional derivatives U_i = dR/dP_i are the
  "influence" components; the variance approximation is sum_i U_i^2 / n^2. This is the
  delta-method / linear-expansion form Efron's Section 5 lands on (Method 3).

## The plug-in principle and the empirical distribution
- Empirical distribution F_hat: puts mass 1/n at each observed x_i. It is the nonparametric MLE
  of F.
- Plug-in: to estimate a functional theta = T(F), use T(F_hat). The bootstrap is the plug-in
  estimate of the *whole sampling distribution* of R(X,F): replace F by F_hat everywhere, so
  the law of R(X*, F_hat) under sampling X* from F_hat estimates the law of R(X,F) under F.
- Fisher consistency: making the estimate exactly right when F = F_hat (Efron 1979 §2).

## The difficulty being solved
- For a complicated statistic (sample median, ratio estimate, correlation, error rate in
  discriminant analysis, nonlinear-regression coefficients) there is no closed-form standard
  error. Classic options: (a) hand-derive an asymptotic variance by the delta method
  (laborious, and fails for nondifferentiable statistics like the median), or (b) jackknife
  (automatic but can fail — median). Need: an automatic, dependable standard error / sampling
  distribution for ANY statistic.

## Canonical implementation grounding
- scipy.stats.bootstrap (`scipy/stats/_resampling.py`, downloaded to code/scipy_resampling.py):
  `_bootstrap_resample` draws integer indices i ~ Uniform{0..n-1} of shape (B,n) -> resamples
  with replacement; `statistic(*resampled, axis=-1)` builds the bootstrap distribution;
  `standard_error = std(theta_hat_b, correction=1)`; percentile + BCa (`_bca_interval`) — BCa
  uses jackknife to estimate the acceleration a_hat, tying the two methods together, "closely
  follows [1] 14.3 and 15.4" (Efron & Tibshirani, An Introduction to the Bootstrap).
