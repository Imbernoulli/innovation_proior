# Synthesis — EGO (GP/kriging surrogate + Expected Improvement)

## Pain point
- Objective y(x) is an EXPENSIVE deterministic black box: an engineering sim / experiment.
  Automotive crash sim ~20 hours per evaluation (paper, Intro). No gradients available
  (black box). May be multimodal. Budget is tens of evaluations, not thousands.
- Existing global optimizers (grid, random, multistart local, genetic/branch-and-bound on the
  true function) need far MORE evaluations than affordable. Local methods get stuck in a local
  min on multimodal funcs. So the central object is: find global min in as few evals as possible.

## Tools on the table (ancestors, with their gaps)
- **Linear regression response surfaces** (Box-Hunter-Hunter 1978, classical RSM): fit
  Σ β_h f_h(x)+ε with independent noise. TWO failures for a deterministic code: (1) you don't
  know the functional form (if you did you wouldn't need the expensive code); flexible forms
  have many params → many evals. (2) independent-error assumption is *false* for a deterministic
  code — there's no measurement noise, the "error" is left-out terms in x, which is continuous,
  so nearby points have nearly-equal errors → errors are CORRELATED, not independent. RSM also
  doesn't interpolate and gives no honest local uncertainty.
- **Kriging / geostatistics** (Matheron 1963 "Principles of geostatistics"; Krige; Cressie):
  model the residual as a spatially correlated random field; correlation high for near points,
  low for far. Gives the Best Linear Unbiased Predictor that *interpolates* the data and a
  mean-squared-error that is zero at data points and grows away from them. This is exactly the
  μ(x) AND σ(x) we want.
- **DACE** (Sacks, Welch, Mitchell, Wynn 1989, "Design and Analysis of Computer Experiments",
  Stat. Science 4:409–435): kriging applied to deterministic computer experiments. The model:
  y(x)=μ+ε(x), ε Gaussian mean 0 var σ², Corr(ε(x_i),ε(x_j))=exp(−d), with the special weighted
  distance d(x_i,x_j)=Σ_h θ_h |x_{ih}−x_{jh}|^{p_h}, θ_h≥0, p_h∈[1,2]. θ_h = activity/relevance
  of variable h; p_h = smoothness (p=2 smooth, →1 less smooth). 2k+2 params: μ,σ²,θ_1..θ_k,
  p_1..p_k, fit by MAXIMUM LIKELIHOOD (concentrated likelihood after closing μ̂,σ̂² in closed form).
- **Kushner 1964** ("A new method of locating the maximum point of an arbitrary multipeak curve
  in the presence of noise", J. Basic Eng. 86:97–106): Wiener-process model in 1-D; sample to
  maximize the PROBABILITY OF IMPROVEMENT P(Y<f_min), with a tradeoff knob (more-global vs
  more-local). Gap: PI ignores the *magnitude* of improvement → biased to exploitation, hugs the
  incumbent, picks tiny near-certain gains; the knob is ad hoc and the model is 1-D.
- **Mockus, Tiesis, Zilinskas 1978** ("The application of Bayesian methods for seeking the
  extremum", Towards Global Optimisation v2): multidimensional Bayesian method; introduces
  EXPECTED IMPROVEMENT — score by expected value of the gain, which weights probability by
  magnitude. The conceptual seed; what EGO does is make it computable with a DACE surrogate and
  maximize it to guaranteed optimality.

## The DACE math (Section 2 — all in context/reasoning)
- distance Eq (1): d(x_i,x_j)=Σ θ_h |x_{ih}−x_{jh}|^{p_h}
- corr Eq (2): Corr=exp(−d)
- model Eq (3): y(x_i)=μ+ε(x_i)
- likelihood Eq (4): (2π)^{-n/2}(σ²)^{-n/2}|R|^{-1/2} exp[−(y−1μ)'R⁻¹(y−1μ)/(2σ²)]
- closed-form MLE: μ̂=(1'R⁻¹y)/(1'R⁻¹1) Eq(5); σ̂²=(y−1μ̂)'R⁻¹(y−1μ̂)/n Eq(6)
- BLUP predictor Eq (7): ŷ(x*)=μ̂ + r'R⁻¹(y−1μ̂), r_i=Corr(ε(x*),ε(x_i))
  - interpolation: at x*=x_i, r=R_i (i-th col), r'R⁻¹=e_i' → ŷ=y_i (Eq 8)
- MSE Eq (9): s²(x*)=σ²[1 − r'R⁻¹r + (1−1'R⁻¹r)²/(1'R⁻¹1)]
  - at x*=x_i: r'R⁻¹r=1, 1'R⁻¹r=1 → s²=0 (Eqs 10,11). Far away r≈0 → s²≈σ². RMSE s=√s².
- Appendix 1 — alternative "augmented-likelihood" derivation of the predictor: add pseudo-obs
  (x*,y*), the only y*-dependent part of the augmented quadratic form is
  (1/(1−r'R⁻¹r))(y*−μ̂)² − (2 r'R⁻¹(y−1μ̂)/(1−r'R⁻¹r))(y*−μ̂) + const; set derivative=0:
  (2/(1−r'R⁻¹r))(y*−μ̂) − 2 r'R⁻¹(y−1μ̂)/(1−r'R⁻¹r)=0 ⇒ y*=μ̂+r'R⁻¹(y−1μ̂) = Eq(7). (The y* that
  best "fits" with the data is the prediction.)

## Why surrogate-min alone fails (Section 4.1)
- Fit surface, jump to its min, resample, iterate → converges to a LOCAL min (Fig 8: DACE min at
  x=2.8, a local min). Pure exploitation ignores uncertainty. Pure exploration (sample max σ,
  x≈8.3) wastes evals. Need a figure of merit balancing both.

## EI derivation (Section 4.1 — the heart)
- Treat unknown y(x) as a random variable Y~Normal(ŷ, s²) (DACE predictor & std error).
  Improvement I=max(f_min − Y, 0), f_min=min observed.
- E[I(x)] = E[max(f_min−Y,0)] Eq(14).
- Closed form Eq(15): E[I] = (f_min−ŷ)Φ((f_min−ŷ)/s) + s φ((f_min−ŷ)/s).
  With z=(f_min−ŷ)/s: EI = (f_min−ŷ)Φ(z) + s φ(z). Note it's s, not s².
- **Full derivation (I verified):** E[I]=∫_{−∞}^{f_min}(f_min−y)(1/s)φ((y−ŷ)/s)dy. Sub u=(y−ŷ)/s,
  upper limit z=(f_min−ŷ)/s: =∫_{−∞}^{z}(f_min−ŷ−s u)φ(u)du
  = (f_min−ŷ)Φ(z) − s∫_{−∞}^{z}uφ(u)du. Since ∫uφ(u)du=−φ(u), ∫_{−∞}^{z}uφ du=−φ(z).
  ⇒ E[I]=(f_min−ŷ)Φ(z)+s φ(z). ✓
- Two terms: (f_min−ŷ)Φ(z) = exploitation (large when predicted mean ŷ is below incumbent);
  s φ(z) = exploration (large when s big). Automatic balance, no hand-set knob.
- EI=0 at sampled points (s=0 → both terms 0, and z=±∞ harmless since s φ→0), positive between.
  Highly MULTIMODAL (Fig 11: two peaks at 2.8 and 8.3; first sample 2.8 then driven to 8.8 →
  global search emerges).
- **Monotonicity (Section 4.1):** derivatives simplify (terms cancel):
  ∂E(I)/∂ŷ = −Φ((f_min−ŷ)/s) < 0  (lower ŷ → more EI)
  ∂E(I)/∂s = φ((f_min−ŷ)/s) > 0  (higher s → more EI)
  ⇒ EI monotone decreasing in ŷ, increasing in s. Used for branch-and-bound bounds:
  upper-bound EI over a box by lower-bounding ŷ (y_L) and upper-bounding s (s_U) and plugging in.
  - verify ∂/∂ŷ: d/dŷ[(f_min−ŷ)Φ(z)+sφ(z)], z=(f_min−ŷ)/s, dz/dŷ=−1/s.
    = −Φ(z) + (f_min−ŷ)φ(z)(−1/s) + s φ'(z)(−1/s). φ'(z)=−zφ(z).
    = −Φ(z) − ((f_min−ŷ)/s)φ(z) + (−1)(−zφ(z)) = −Φ(z) − zφ(z) + zφ(z) = −Φ(z). ✓ <0.
  - verify ∂/∂s: dz/ds=−(f_min−ŷ)/s²=−z/s.
    = (f_min−ŷ)φ(z)(−z/s) + φ(z) + s φ'(z)(−z/s)
    = −z·((f_min−ŷ)/s)φ(z) + φ(z) + (−z)(−zφ(z))
    = −z²φ(z) + φ(z) + z²φ(z) = φ(z). ✓ >0.

## EGO algorithm (Section 4.2)
1. Space-filling initial design (Latin hypercube, ~10k points; good 1-D & 2-D projections).
2. Evaluate objective on design; fit DACE by MLE.
3. Diagnostics (cross-validated standardized residuals < 3); if bad, try log or −1/y transform.
4. Iterate: maximize EI (branch-and-bound to global optimality using the monotonicity bounds);
   if max EI < 1% of current best |f| → STOP; else evaluate at argmax EI, re-fit DACE, repeat.
   (For log-transformed funcs, stop when EI on log scale < 0.01 absolute ≈ 1% relative.)

## Ill-conditioning (Discussion)
- R nearly singular when (a) function very smooth → columns ≈ all-ones, collinear; (b) late in run
  points cluster → near-duplicate columns. Handled via SVD of R, zeroing tiny singular values
  (Numerical Recipes). [In modern code: a small "nugget"/jitter / WhiteKernel on the diagonal.]

## EXCLUDED — proposed method's own results (DO NOT use)
- Table 1 numbers (Branin 28 evals 0.2%, Goldstein-Price 32 / 0.1%, Hartman3 34 / 1.7%,
  Hartman6 84 / 1.9%, etc.), timing (139 s Branin first iterate), Fig 12 "finds all three global
  minima in 3 iterates". These are EGO's OWN evaluation outcomes — banned from all files.

## Canonical code grounding (skopt)
- skopt/acquisition.py gaussian_ei: improve = y_opt − xi − mu; scaled = improve/std;
  values = improve*norm.cdf(scaled) + std*norm.pdf(scaled). (xi = small exploration margin,
  default 0.01.) Maximized to pick next point. Exactly Eq(15) plus the xi margin.
- skopt/optimizer/gp.py gp_minimize: default base estimator a GP with Matern kernel + per-dim
  length scales + noise (WhiteKernel), n_initial_points=10, acq_func, acq_optimizer lbfgs/sampling,
  normalize_y. (Matern ν is the modern analog of DACE's p smoothness; ARD length scales are the
  analog of θ_h.) ask/tell loop = fit GP → maximize acquisition → eval → tell → refit.

## Design-decision → why table
- Correlated (not independent) errors: deterministic code has no noise; "error" is continuous
  left-out terms → nearby errors near-equal. Independence is provably wrong for a sim.
- exp(−Σθ|Δ|^p) kernel: gives Corr→1 near, →0 far; θ=activity (anisotropy), p=smoothness.
  Power p∈[1,2] interpolates rough↔smooth; p=2 = Gaussian-smooth (and lets the B&B bound work).
- constant mean μ (drop regressors): the correlation structure is powerful enough that regressors
  aren't needed; fewer params to fit on a tiny sample.
- MLE of θ,p (concentrated likelihood): μ̂,σ̂² close in closed form, leaving a k(or 2k)-dim
  likelihood to maximize — far fewer than fitting a flexible regression form.
- Interpolation (s=0 at data): deterministic function — once sampled, value is known exactly,
  uncertainty must be 0 there. Kriging delivers this automatically.
- EI over PI: PI (Kushner) ignores magnitude → hugs incumbent (exploitation bias). EI weights
  by how much you'd gain → naturally explores. No separate tradeoff knob needed.
- EI over pure-explore (max s) or pure-exploit (min ŷ): each alone is pathological (local min /
  wasted evals). EI's two terms are exactly exploit + explore, balanced by the math.
- Maximize EI globally (branch-and-bound): EI is multimodal with flat near-zero plateaus →
  multistart-local is unreliable; closed form + monotonicity in (ŷ,s) gives valid box bounds.
- Space-filling LHS init (~10k): need a spread to fit θ,p before any EI step; LHS with good
  low-dim projections covers the box without clustering.
- Stop at EI<1% of best: EI is the model's own estimate of remaining gain → a credible,
  self-contained stopping rule (a key selling point vs fixed eval budgets).
- (modern) Matern ν + ARD length scales = p smoothness + θ activity; nugget/WhiteKernel = the SVD
  small-singular-value fix for R ill-conditioning.

## URLs
- Paper (full text, 38 pp): http://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/f84f7ac703bf5862c12576d8002f5259/$FILE/Jones98.pdf
  (J. Global Optimization 13:455–492, 1998; DOI 10.1023/A:1008306431147)
- skopt acquisition: https://raw.githubusercontent.com/scikit-optimize/scikit-optimize/master/skopt/acquisition.py
- skopt gp_minimize: https://raw.githubusercontent.com/scikit-optimize/scikit-optimize/master/skopt/optimizer/gp.py
- Brochu, Cora, de Freitas tutorial (history of PI/EI ancestors): https://arxiv.org/pdf/1012.2599
