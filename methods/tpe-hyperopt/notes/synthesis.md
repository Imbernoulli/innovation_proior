# TPE synthesis

## Pain point / setup
- Tuning DBNs / deep nets: 10–50 hyper-parameters, mixed discrete/ordinal/continuous, and TREE-STRUCTURED:
  leaf vars (2nd-layer units) only defined when a node var (n. layers) takes a value. Must optimize over which
  variables to optimize. Each trial = train+evaluate a model = expensive. Budget = tens to low hundreds of trials.
- Random search (Bergstra & Bengio 2012, ref [19]/[21]) is the ancestor baseline: draw configs from the generative
  prior, evaluate. Beats grid; matches manual for nets; but UNRELIABLE for hard DBN datasets (convex, MRBI) —
  converges slowly / plateaus below manual. So we need something that USES the history.

## SMBO frame (Section 2)
- True fitness f: X→R costly. Approximate with surrogate M; inner loop = optimize a cheap criterion S(x,M) to get x*.
  Template: H←∅; for t=1..T: x*←argmin_x S(x,M_{t-1}); evaluate f(x*) (expensive); H←H∪(x*,f(x*)); fit M_t to H.
- Criterion = Expected Improvement (Mockus, Tiesis, Zilinskas 1978; Jones 2001 EGO uses it). With threshold y*:
  EI_{y*}(x) = ∫_{-∞}^{y*} (y* − y) p_M(y|x) dy.   (eq 1) — expectation of how far below threshold f(x) lands.

## The GP route (Section 3) — the sibling story, NOT to duplicate
- Model p(y|x) directly with a GP; y* = best observed; EI closed form via posterior mean/variance; optimize EI
  (multimodal, non-negative, 0 at data) by EDA on discrete part + CMA-ES on continuous part, restarts. Tree-structure
  handled by grouping params and placing independent GPs per group. Cost cubic in |H|.

## The TPE move (Section 4) — the intellectual contribution
- Instead of p(y|x), model p(x|y) and p(y). Define, by splitting the history at a threshold y*:
    p(x|y) = l(x) if y < y* ;  g(x) if y ≥ y*.   (eq 2)
  l(x) = density over configs whose loss was BELOW y* (the "good" ones); g(x) = density over the rest ("bad").
- y* chosen as a QUANTILE γ of observed y: γ = p(y < y*). So p(y<y*)=γ by construction. No model of p(y) needed
  beyond this. Aggressive y*=best (GP) is replaced by a quantile so that l(x) is fit on >1 point.

## The EI = l/g derivation (Section 4.1) — MATH MUST BE EXACT
Parametrize p(x,y)=p(y)p(x|y) (that's why p(x|y) is modeled). Then
  EI_{y*}(x) = ∫_{-∞}^{y*}(y*−y) p(y|x) dy = ∫_{-∞}^{y*}(y*−y) p(x|y)p(y)/p(x) dy.   (eq 3)
Denominator:  p(x) = ∫_R p(x|y)p(y) dy = γ l(x) + (1−γ) g(x).
  [because for y<y*, p(x|y)=l(x) and ∫_{y<y*}p(y)=γ; for y≥y*, p(x|y)=g(x) and ∫=1−γ.]
Numerator:  ∫_{-∞}^{y*}(y*−y) p(x|y)p(y) dy = l(x) ∫_{-∞}^{y*}(y*−y)p(y) dy
          = l(x)[ y* ∫_{-∞}^{y*}p(y)dy − ∫_{-∞}^{y*} y p(y)dy ]
          = γ y* l(x) − l(x) ∫_{-∞}^{y*} p(y) dy.
  [since ∫_{-∞}^{y*}p(y)dy = γ, the first term = γ y* l(x). The second term ∫ y p(y) dy is left as a constant in x.]
So  EI_{y*}(x) = [ γ y* l(x) − l(x) ∫_{-∞}^{y*} p(y)dy ] / [ γ l(x) + (1−γ) g(x) ].
Factor l(x) out of numerator; divide num and denom by l(x):
  numerator/l(x) = γ y* − ∫_{-∞}^{y*} p(y) dy   — a CONSTANT in x (call it A>0; positive since y*≥ the y-mass it integrates... actually A = γy* − E[y·1{y<y*}], positive because y<y* on the support so the integral < γy*).
  denom/l(x)     = γ + (1−γ) g(x)/l(x).
So  EI_{y*}(x) = A · ( γ + (1−γ) g(x)/l(x) )^{-1}  ∝  ( γ + g(x)/l(x) (1−γ) )^{-1}.
=> EI is a DECREASING function of the ratio g(x)/l(x). Maximize EI  ⇔  minimize g(x)/l(x)  ⇔  MAXIMIZE l(x)/g(x).
Intuition: want x with high density under l (good) and low under g (bad).

## Why this is great
- Tree structure trivially handled: l, g are just the generative prior with its leaf distributions re-estimated from
  the good/bad subsets — same conditional graph, runtime LINEAR in |H| and in #dims (vs GP cubic).
- Sampling: because we only need argmax l/g and l is a density we can SAMPLE from, draw n_EI_candidates from l(x),
  evaluate the ratio g/l (or log l − log g) at each, take the best. No global optimizer over a black-box acquisition.

## Adaptive Parzen estimator (Section 4.2) — the densities
- Continuous var with uniform prior on (a,b): l(x) = equally-weighted mixture of the box prior (a Gaussian / the
  uniform) PLUS one Gaussian centered at each good observation x^(i)∈B. (Hyperopt: prior is a broad Gaussian at the
  box midpoint with prior_sigma=(b−a); each data Gaussian gets weight 1, prior gets prior_weight.)
- Bandwidth (sigma) of each data Gaussian = the GREATER of the distance to its left and right neighbor (in sorted
  order), with a and b counted as neighbors; clipped to [minsigma, maxsigma]. minsigma = prior_sigma/min(100,1+N).
  This is the "adaptive" part: dense regions get narrow kernels, sparse regions wide ones.
- log-uniform → same in log domain (exponentiated truncated Gaussian mixture). categorical with prior probs p_i →
  posterior ∝ N p_i + C_i where C_i = count of choice i in the good set B; "re-weighted categorical."
- Hyperopt also linear-forgetting-weights older observations (DEFAULT_LF=25) so recent points count more — a
  practical robustness knob, not in the core derivation.

## Constants from canonical code (hyperopt/tpe.py)
- _default_gamma = 0.25 ; in hyperopt n_below = ceil(γ·√N) (the paper states γ as a plain quantile).
- _default_n_EI_candidates = 24 ; _default_n_startup_jobs = 20 (random warm-up before TPE kicks in).
- scoring: below_llik − above_llik = log l(x) − log g(x); broadcast_best picks the max.

## Three sources
- PRIMARY: Bergstra, Bardenet, Bengio, Kégl 2011 NeurIPS, full text read (refs/bergstra2011-tpe.pdf), Sections 1,2,4
  incl. the eq-3 EI=l/g derivation and 4.2 adaptive Parzen.
- BACKGROUND: SMBO + EI (Mockus 1978; Jones et al. 1998 EGO), Parzen-window KDE (Parzen 1962), random search
  (Bergstra & Bengio 2012). EI/SMBO substance shared with sibling bayesopt-ego anchor.
- THIRD-PARTY: Optuna TPESampler docs / source (the l(x)/g(x) acquisition, n_ei_candidates) + Hyperopt tpe.py
  canonical implementation (adaptive_parzen_normal bandwidth rule, gamma, splitting).
