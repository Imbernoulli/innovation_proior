# Synthesis — Bootstrap (Efron 1979), grounded for Phase 2

## Pain point
Assign accuracy (standard error / sampling distribution) to a statistic theta_hat = t(x) for an
ARBITRARY, complicated statistic where no analytic SE formula exists or is painful: sample
median, ratio E_F Y / E_F Z, Pearson correlation, error rate in linear discriminant analysis,
nonlinear-regression coefficients. Efron's own framing: "how do you assign accuracy to a
statistical estimate? The older theories require mathematical analysis, making models."

## Tools on the table and where each falls short
1. Analytic / delta method: derive Var(t) by hand via a Taylor expansion. Laborious, per-statistic,
   and breaks for nondifferentiable t (the median's derivative is too irregular — Efron §8 Rmk J).
2. Quenouille–Tukey jackknife: automatic; leave-one-out recompute, pseudo-values give bias and
   variance. But "hot and mysterious" (Efron) and Miller's "A trustworthy jackknife" shows it
   FAILS sometimes — notably the median (Efron §3: jackknife variance for the median is not even
   consistent; n*Var ~ (1/4f^2)(chi_1^2/... )^2 with mean 2, variance 20).
3. Hartigan's subsample-value / replaced-sample methods: resample by taking subsets; asymptotically
   valid confidence statements (Efron §8 Rmk I). Related but not the same; smaller artificial
   samples, weaker small-sample matching.

## The question that unlocks it (Efron's backbone)
"Put the jackknife on familiar statistical grounds" = find the MORE PRIMITIVE thing the jackknife
is an approximation OF. The jackknife perturbs the data by deleting one point; what is the
"true" object it is linearizing? Answer: the sampling distribution of R(X,F) itself.

## The leap
- Empirical distribution F_hat: mass 1/n at each x_i (the nonparametric MLE of F).
- Plug-in principle: we cannot sample new datasets from the unknown F, but we CAN sample from
  F_hat — which is our best estimate of F. So substitute F_hat for F everywhere.
- Bootstrap algorithm: draw X* = (X1*,...,Xn*) iid from F_hat (i.e., sample n points WITH
  REPLACEMENT from {x_1,...,x_n}); recompute R* = R(X*, F_hat); the distribution of R* (over the
  resampling randomness, F_hat fixed) approximates the distribution of R(X,F).
- Why it works: it is EXACTLY right when F = F_hat (Fisher consistency); F_hat is the central
  point among likely F's given the data; any reasonable nonparametric estimator must agree there
  (Efron §2). Finite-sample-space rationale (Rmk G): both i|f ~ Mult_L(n,f) and i*|i ~ Mult_L(n,i);
  since i -> f, the conditional law of Q(i*,i) matches the sampling law of Q(i,f). Asymptotic
  proof (Rmk G eqs 8.10–8.11): both n^{1/2}(i-f) and n^{1/2}(i*-i) -> N(0, Sigma_f), so the
  bootstrap law of n^{1/2}Q(i*,i) -> sampling law of n^{1/2}Q(i,f), both N(0, u' Sigma_f u).

## Three ways to compute the bootstrap distribution (Efron §2)
- Method 1 — direct theory (closed form when possible: median via binomial, §3 eq 3.5).
- Method 2 — Monte Carlo: generate X*1,...,X*N, histogram of R(X*b,F_hat); cost ~ N x original.
  THE practical one ("computationally intensive... almost automatic"). Rmk A: N=100, m=n=20
  discriminant trial ~0.15s, 40 cents on Stanford 370/168; N=1000 ~ $4.
- Method 3 — Taylor/delta expansion of R(P*) about P*=e/n => recovers the jackknife
  (= Jaeckel infinitesimal jackknife). THIS is the unification: jackknife = linear approx to
  bootstrap.

## Section 5 derivation (jackknife = delta method on the bootstrap) — must be lived in reasoning
- P* = N*/n, N* = resample counts, multinomial: E_* P* = e/n, Cov_* P* = (1/n)(I/n - e'e/n^2)
  i.e. (5.2) Cov = I/n^2 - e'e/n^3.
- R(P*) = R(e/n) + (P*-e/n)U + 1/2 (P*-e/n)V(P*-e/n)', U_i = dR/dP_i, V = Hessian.
- Homogeneous extension R(aP*)=R(P*) gives eU=0, eV=-nU', eVe'=0 (5.7).
- E_* R* ≈ R(e/n) + (1/2n) tr-ish term -> bias ≈ (1/2n) sum V_ii style; Var_* R* ≈ sum_i U_i^2 / n^2
  (5.10). For R = theta(F_hat*) - theta(F_hat), R(e/n)=0, so Var_F theta_hat ≈ sum U_i^2 / n^2 —
  Jaeckel's infinitesimal jackknife (5.11), and ordinary jackknife = same up to 1+O(1/n).
- Median: jackknife fails because U is dominated by behavior of Q very near i (deletions are
  O(1/n) away; bootstrap vectors are O_p(n^{-1/2}) away) — the derivative is too irregular for
  the quadratic extrapolation (Rmk J). Bootstrap, sampling at the right O(n^{-1/2}) scale, works.

## Examples that motivate (context / motivating, not proposed-method "wins")
- median (jackknife fails, bootstrap consistent), ratio estimate, correlation (tanh transform,
  Rmk B/E pivotality), discriminant error rate vs cross-validation, regression (resample
  residuals, §7).

## Design choices -> why
- Resample WITH replacement, size n (not without; not subsets): so X* ~ F_hat exactly, matching
  the original sample size keeps the sampling-distribution scale right; "not a permutation
  distribution." Hartigan's without-replacement subsamples are O(n^{-1/2}) but weaker small-sample
  match.
- F_hat = empirical (nonparametric MLE): the central/most-likely F given data; makes the plug-in
  exactly right at F=F_hat.
- SE = sd of bootstrap replicates (with Bessel correction): direct Monte-Carlo estimate of
  sqrt(Var_* R*).
- Percentile interval: quantiles of R* directly (monotone-transform respecting, Rmk B).
- BCa (later refinement, grounded in scipy): bias-correction z0 + acceleration a_hat (a_hat from
  the jackknife influence values) — keep light in reasoning (1979 paper only has percentile);
  put full BCa in answer.md as the canonical implementation's CI, faithful to scipy.
- Smoothed bootstrap (§3 eq 3.11): add small noise to resamples if F assumed smooth; symmetrized
  bootstrap if F assumed symmetric — optional variants; Table 1 shows plain bootstrap already does
  well, so default is plain.
- N (Monte Carlo size): N->inf removes MC error; SE of estimate ~ (sd)/sqrt(N); 1000–10000 typical.
- Parametric bootstrap (Rmk K): sample from parametric MLE F_theta_hat instead of F_hat; gives
  1/Fisher-information for the variance of the MLE.

## Canonical code (scipy _resampling.py) — final landing
- `_bootstrap_resample`: i = rng.integers(0,n,(B,n)); resample = sample[..., i]  (with replacement).
- loop in `bootstrap`: for each batch, resample each sample, theta_hat_b = statistic(*resampled, axis=-1).
- standard_error = std(theta_hat_b, ddof=1).
- percentile interval: quantiles of theta_hat_b at alpha/2, 1-alpha/2.
- BCa `_bca_interval`: z0_hat = ndtri(P(theta_hat_b < theta_hat)); a_hat from jackknife
  U_i=(n-1)(theta_dot - theta_(-i)), a_hat = (1/6) sum U_i^3 / (sum U_i^2)^{3/2};
  alpha_1 = ndtr(z0 + (z0+z_a)/(1 - a(z0+z_a))), etc.

## In-frame discipline
- Never cite "the paper"/Efron/Annals/1979. May name "the bootstrap" and "jackknife"/Quenouille/
  Tukey/Jaeckel/Hartigan as prior art. Self-account shapes voice (Miller's trustworthy jackknife,
  "put it on familiar grounds", computationally-intensive/automatic), not as a citation.
